import argparse
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


UQ_DIR = Path(__file__).resolve().parents[1]
if str(UQ_DIR) not in sys.path:
    sys.path.insert(0, str(UQ_DIR))

from train_restoration_control import (
    install_control,
    load_trainer,
    run_training,
    validate_args,
)


class RestorationControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trainer = load_trainer()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.block_dir = self.root / "blocks"
        self.deg_dir = self.root / "degraded"
        (self.block_dir / "val").mkdir(parents=True)
        (self.deg_dir / "val").mkdir(parents=True)

        stem = "tile_bx00_by00_deg_v3_mild"
        points = np.array(
            [[0.1, 0.1, 1.0], [1.1, 0.1, 2.0],
             [0.1, 1.1, 3.0], [1.1, 1.1, 4.0]], dtype=np.float32
        )
        np.savez(
            self.block_dir / "val" / f"{stem}.npz",
            points=points,
            points_local=points,
            features=np.arange(4, dtype=np.float32)[:, None],
            labels=np.arange(4, dtype=np.int64),
            point_hard_mask=np.zeros(4, dtype=np.bool_),
        )
        maps = np.arange(36, dtype=np.float32).reshape(9, 2, 2)
        np.savez(
            self.deg_dir / "val" / f"{stem}_25d.npz",
            maps_norm=maps, x0=0.0, y0=0.0, resolution=1.0,
            valid_mask=np.ones((2, 2), dtype=np.float32),
            normalization_stats_sha256=(
                "7c85f5d61a316d9d8a576bfff57e9e284634fd3e760fbe15e834c80ed937df75"
            ),
            normalization_protocol="shared_25d_normalization_v3",
            normalization_fit_split="train",
        )
        self.expected = maps[:, [0, 0, 1, 1], [0, 1, 0, 1]].T

    def tearDown(self):
        self.tmp.cleanup()

    def _dataset(self, dataset_cls):
        return dataset_cls(
            block_dir=self.block_dir,
            clean_map_dir=self.root / "missing_clean",
            deg_map_dir=self.deg_dir,
            restored_map_dir=self.root / "missing_restored",
            split="val", mode="restored_rgdi", num_points=4,
            num_layers=1, num_neighbors=1, sub_sampling_ratio=[1],
            degradation_tag="deg_v3_mild", seed=7,
        )

    def test_observation_dataset_reads_only_degraded_and_preserves_alignment(self):
        original_loader = self.trainer.load_npz

        def forbid_restored(path):
            if "restored" in str(path) or "clean" in str(path):
                raise AssertionError(f"forbidden prior read: {path}")
            return original_loader(path)

        self.trainer.load_npz = forbid_restored
        try:
            dataset_cls = install_control(self.trainer, "observation")
            sample = self._dataset(dataset_cls)[0]
        finally:
            self.trainer.load_npz = original_loader

        aligned = self.expected[sample["labels"]]
        np.testing.assert_array_equal(sample["degraded"], aligned)
        np.testing.assert_array_equal(sample["delta"], aligned)
        np.testing.assert_array_equal(sample["restored"], np.zeros((4, 9)))
        np.testing.assert_array_equal(
            sample["base"][:, 3], sample["labels"].astype(np.float32)
        )

    def test_observation_second_degraded_branch_is_trainable(self):
        fusion = self.trainer.RGDIFusion(
            embed_ch=8, use_explicit_quality_modulation=False,
            use_reliability_descriptors=False,
        )
        fusion.eval()
        base = torch.randn(2, 4, 4)
        degraded = torch.randn(2, 4, 9)
        delta = degraded.detach().clone().requires_grad_(True)
        output = fusion(base, degraded, delta, torch.zeros(2, 4, 6))
        output.square().mean().backward()
        self.assertGreater(float(delta.grad.abs().sum()), 0.0)
        self.assertGreater(
            float(fusion.delta_embed[0].weight.grad.abs().sum()), 0.0
        )

    def test_observation_rejects_wrong_normalization_protocol(self):
        map_path = next((self.deg_dir / "val").glob("*.npz"))
        with np.load(map_path) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        for key, wrong in (
            ("normalization_stats_sha256", "wrong"),
            ("normalization_protocol", "legacy"),
            ("normalization_fit_split", "all"),
        ):
            with self.subTest(key=key):
                changed = dict(arrays)
                changed[key] = wrong
                np.savez(map_path, **changed)
                dataset_cls = install_control(self.trainer, "observation")
                with self.assertRaisesRegex(RuntimeError, "normalization"):
                    self._dataset(dataset_cls)[0]

    def test_restoration_control_keeps_original_dataset(self):
        original = self.trainer._RESTORATION_CONTROL_ORIGINAL_DATASET
        selected = install_control(self.trainer, "restoration")
        self.assertIs(selected, original)
        self.assertIs(self.trainer.DALESRandLADataset, original)

    def test_validation_rejects_test_access_and_protected_outputs(self):
        common = dict(
            mode="restored_rgdi", inject_stages=1, gate_mode="learned",
            disable_explicit_quality_modulation=True,
            disable_reliability_descriptors=True, skip_test=True,
            resume="", eval_only_checkpoint="", workers=4,
        )
        valid_out = self.root / "outputs_uq" / "run"
        args = argparse.Namespace(out_dir=str(valid_out), **common)
        validate_args(args, self.root, "observation")
        self.assertEqual(args.workers, 4)

        args.skip_test = False
        with self.assertRaisesRegex(ValueError, "skip_test"):
            validate_args(args, self.root, "observation")

        args.skip_test = True
        args.out_dir = str(self.root / "outputs" / "protected")
        with self.assertRaisesRegex(ValueError, "outputs_uq"):
            validate_args(args, self.root, "observation")

        args.out_dir = str(valid_out)
        valid_out.mkdir(parents=True)
        (valid_out / "old.txt").write_text("old", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "non-empty"):
            validate_args(args, self.root, "observation")

    def test_cpu_test_run_records_control_semantics_and_completion(self):
        out = self.root / "outputs_uq" / "metadata_run"
        args = argparse.Namespace(
            out_dir=str(out), mode="restored_rgdi", inject_stages=1,
            gate_mode="learned", disable_explicit_quality_modulation=True,
            disable_reliability_descriptors=True, skip_test=True, resume="",
            eval_only_checkpoint="", workers=9,
        )

        class FakeTrainer:
            RANDLANET_BACKEND = "pure_torch_fallback (test)"
            _RESTORATION_CONTROL_ORIGINAL_DATASET = object
            DALESRandLADataset = object

            @staticmethod
            def _RESTORATION_CONTROL_ORIGINAL_TRAIN(received):
                (Path(received.out_dir) / "config.json").write_text(
                    __import__("json").dumps(vars(received)), encoding="utf-8"
                )

        run_training(
            FakeTrainer, args, "restoration", ["--skip_test"], root=self.root
        )
        metadata = __import__("json").loads(
            (out / "control_run_metadata.json").read_text(encoding="utf-8")
        )
        config = __import__("json").loads(
            (out / "config.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["status"], "complete")
        self.assertEqual(metadata["peak_cuda_allocated_bytes"], 0)
        self.assertEqual(config["restoration_control"], "restoration")
        self.assertEqual(config["control"], "restoration")
        self.assertIn("restored-minus-degraded", config["correction_semantics"])
        self.assertEqual(
            config["expected_normalization_sha256"],
            "7c85f5d61a316d9d8a576bfff57e9e284634fd3e760fbe15e834c80ed937df75",
        )
        self.assertEqual(config["workers"], 4)


if __name__ == "__main__":
    unittest.main()
