import argparse
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


UQ_DIR = Path(__file__).resolve().parents[1]
if str(UQ_DIR) not in sys.path:
    sys.path.insert(0, str(UQ_DIR))

import train_feature_control as feature_control
from frozen_unet_features import extract_decoder_features, load_unet_module
from train_feature_control import (
    FeatureFusionRandLASegmentationModel,
    install_feature_dataset,
    load_trainer,
    validate_feature_cache,
    validate_feature_args,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FeatureControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.unet = load_unet_module()
        cls.trainer = load_trainer()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.block_dir = self.root / "blocks"
        self.deg_dir = self.root / "degraded"
        self.cache_dir = self.root / "cache"
        self.checkpoint_path = self.cache_dir / "checkpoint.pt"
        for split in ("train", "val"):
            (self.block_dir / split).mkdir(parents=True)
            (self.deg_dir / split).mkdir(parents=True)
            (self.cache_dir / split).mkdir(parents=True)
            stem = f"tile_{split}_deg_v3_mild"
            points = np.array(
                [[.1, .1, 1], [1.1, .1, 2], [.1, 1.1, 3], [1.1, 1.1, 4],
                 [.1, .1, 5], [1.1, .1, 6], [.1, 1.1, 7], [1.1, 1.1, 8]],
                dtype=np.float32,
            )
            np.savez(
                self.block_dir / split / f"{stem}.npz", points=points,
                points_local=points, features=np.arange(8, dtype=np.float32)[:, None],
                labels=np.arange(8, dtype=np.int64),
                point_hard_mask=np.zeros(8, dtype=np.bool_),
            )
            maps = np.arange(36, dtype=np.float32).reshape(9, 2, 2)
            deg = self.deg_dir / split / f"{stem}_25d.npz"
            np.savez(
                deg, maps_norm=maps, x0=0., y0=0., resolution=1.,
                valid_mask=np.ones((2, 2), dtype=np.float32),
                structural_hard_mask=np.zeros((2, 2), dtype=np.float32),
                low_density_mask=np.zeros((2, 2), dtype=np.float32),
                high_gradient_mask=np.zeros((2, 2), dtype=np.float32),
                high_roughness_mask=np.zeros((2, 2), dtype=np.float32),
                normalization_stats_sha256="7c85f5d61a316d9d8a576bfff57e9e284634fd3e760fbe15e834c80ed937df75",
                normalization_protocol="shared_25d_normalization_v3",
                normalization_fit_split="train",
            )
            feature = np.zeros((48, 2, 2), dtype=np.float32)
            for ch in range(48):
                feature[ch] = ch * 10 + np.array([[0, 1], [2, 3]])
            feature_path = self.cache_dir / split / f"{stem}_25d_features.npy"
            np.save(feature_path, feature, allow_pickle=False)
        self.checkpoint_path.write_bytes(b"checkpoint fixture")
        self.saved_checkpoint_sha = feature_control.CHECKPOINT_SHA256_BY_TAG["deg_v3_mild"]
        feature_control.CHECKPOINT_SHA256_BY_TAG["deg_v3_mild"] = sha(self.checkpoint_path)
        self.manifest_path = self._write_manifest()

    def tearDown(self):
        feature_control.CHECKPOINT_SHA256_BY_TAG["deg_v3_mild"] = self.saved_checkpoint_sha
        self.tmp.cleanup()

    def _write_manifest(self, **changes):
        entries = []
        for split in ("train", "val"):
            stem = f"tile_{split}_deg_v3_mild"
            deg = self.deg_dir / split / f"{stem}_25d.npz"
            feature = self.cache_dir / split / f"{stem}_25d_features.npy"
            entries.append({
                "split": split, "degraded_filename": deg.name,
                "feature_path": f"{split}/{feature.name}",
                "degraded_sha256_before": sha(deg),
                "degraded_sha256_after": sha(deg), "feature_sha256": sha(feature),
                "shape": [48, 2, 2], "dtype": "float32",
                "x0": 0., "y0": 0., "resolution": 1.,
            })
        payload = {
            "schema_version": 1, "status": "complete", "degradation_tag": "deg_v3_mild",
            "feature_channels": 48, "spatial_shape": [2, 2],
            "normalization": {"stats_sha256": "7c85f5d61a316d9d8a576bfff57e9e284634fd3e760fbe15e834c80ed937df75", "protocol": "shared_25d_normalization_v3", "fit_split": "train"},
            "counts": {"train": 1, "val": 1}, "entries": entries,
            "checkpoint": {"path": str(self.checkpoint_path), "sha256": sha(self.checkpoint_path)},
            "sources": {"unet_trainer_sha256": "source"},
            "config": {"degradation_tag": "deg_v3_mild"},
        }
        payload["config_sha256"] = hashlib.sha256(json.dumps(payload["config"], sort_keys=True).encode()).hexdigest()
        payload.update(changes)
        path = self.cache_dir / "feature_cache_manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_decoder_features_reproduce_original_forward_before_output_head(self):
        torch.manual_seed(4)
        model = self.unet.UNet2D(in_ch=14, out_ch=9, base_ch=2).eval()
        x = torch.randn(2, 14, 16, 16)
        with torch.no_grad():
            features = extract_decoder_features(model, x)
            reconstructed = torch.sigmoid(model.out(features))
            original = model(x)
        self.assertEqual(tuple(features.shape), (2, 2, 16, 16))
        torch.testing.assert_close(reconstructed, original, rtol=0, atol=0)

    def test_dataset_samples_48d_features_with_original_point_alignment(self):
        dataset_cls = install_feature_dataset(self.trainer, self.manifest_path)
        dataset = dataset_cls(
            block_dir=self.block_dir, clean_map_dir=self.root / "forbidden_clean",
            deg_map_dir=self.deg_dir, restored_map_dir=self.root / "forbidden_restored",
            split="val", mode="restored_rgdi", num_points=8, num_layers=1,
            num_neighbors=1, sub_sampling_ratio=[1], degradation_tag="deg_v3_mild", seed=7,
        )
        sample = dataset[0]
        self.assertEqual(sample["restored"].shape, (8, 48))
        expected_pixel = sample["labels"] % 4
        np.testing.assert_array_equal(sample["restored"][:, 0], expected_pixel)
        np.testing.assert_array_equal(sample["degraded"], np.arange(36).reshape(9, 4)[:, expected_pixel].T)
        np.testing.assert_array_equal(sample["delta"], np.zeros((8, 9)))
        np.testing.assert_array_equal(sample["rel"], np.zeros((8, 6)))

    def test_projection_receives_gradients_and_is_used_at_both_rgdi_branches(self):
        torch.manual_seed(9)
        model = FeatureFusionRandLASegmentationModel(
            mode="restored_rgdi", num_points=8, num_neighbors=1, num_layers=1,
            sub_sampling_ratio=[1], dim_output=[4], dim_features=2, embed_ch=4,
            gate_bias=.3, inject_stages=1, internal_hidden_ch=4,
            internal_alpha_init=.1, use_explicit_quality_modulation=False,
            use_reliability_descriptors=False, gate_mode="learned", dropout=0.,
            device=torch.device("cpu"), feature_channels=48,
        )
        seen = {}
        def capture_input(_module, args):
            seen["input_delta"] = args[2].detach().clone()

        def capture_internal(_module, args):
            seen["internal_delta"] = args[2].detach().clone()

        model.rgdi.register_forward_pre_hook(capture_input)
        model.internal_injectors[0].register_forward_pre_hook(capture_internal)
        batch = {
            "base": torch.randn(1, 8, 4), "restored": torch.randn(1, 8, 48),
            "degraded": torch.randn(1, 8, 9), "delta": torch.zeros(1, 8, 9),
            "rel": torch.zeros(1, 8, 6), "coords": [torch.randn(1, 8, 3)],
            "neighbor_indices": [torch.zeros(1, 8, 1, dtype=torch.long)],
            "sub_idx": [torch.arange(8).reshape(1, 8, 1)],
            "interp_idx": [torch.arange(8).reshape(1, 8, 1)],
        }
        projected = model.feature_projection(batch["restored"])
        model(batch).sum().backward()
        torch.testing.assert_close(seen["input_delta"], projected.detach())
        torch.testing.assert_close(seen["internal_delta"], projected.detach())
        self.assertGreater(float(model.feature_projection.weight.grad.abs().sum()), 0.)
        self.assertEqual(sum(p.numel() for p in model.feature_projection.parameters()), 441)

    def test_projection_initialization_does_not_perturb_original_model_rng_or_weights(self):
        kwargs = dict(mode="restored_rgdi", num_points=8, num_neighbors=1, num_layers=1,
            sub_sampling_ratio=[1], dim_output=[4], dim_features=2, embed_ch=4,
            gate_bias=.3, inject_stages=1, internal_hidden_ch=4, internal_alpha_init=.1,
            use_explicit_quality_modulation=False, use_reliability_descriptors=False,
            gate_mode="learned", dropout=0., device=torch.device("cpu"))
        torch.manual_seed(123)
        original = self.trainer._FEATURE_CONTROL_ORIGINAL_MODEL(**kwargs)
        original_rng = torch.get_rng_state().clone()
        torch.manual_seed(123)
        feature = FeatureFusionRandLASegmentationModel(**kwargs)
        torch.testing.assert_close(torch.get_rng_state(), original_rng)
        feature_state = feature.state_dict()
        for name, value in original.state_dict().items():
            torch.testing.assert_close(feature_state[name], value)

    def test_manifest_validation_fails_closed_for_status_hash_path_and_coverage(self):
        validate_feature_cache(self.manifest_path, self.block_dir, self.deg_dir, "deg_v3_mild", 1, 1)
        base = json.loads(self.manifest_path.read_text())
        cases = []
        bad = dict(base); bad["status"] = "running"; cases.append((bad, "complete"))
        bad = json.loads(json.dumps(base)); bad["entries"][0]["feature_sha256"] = "bad"; cases.append((bad, "hash"))
        bad = json.loads(json.dumps(base)); bad["entries"][0]["feature_path"] = "../escape.npy"; cases.append((bad, "path"))
        bad = json.loads(json.dumps(base)); bad["entries"] = bad["entries"][1:]; bad["counts"]["train"] = 0; cases.append((bad, "coverage"))
        for payload, message in cases:
            with self.subTest(message=message):
                self.manifest_path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex((ValueError, RuntimeError, FileNotFoundError), message):
                    validate_feature_cache(self.manifest_path, self.block_dir, self.deg_dir, "deg_v3_mild", 1, 1)

    def test_protocol_flags_reject_test_resume_eval_and_wrong_smoke_counts(self):
        common = dict(mode="restored_rgdi", inject_stages=1, gate_mode="learned",
            disable_explicit_quality_modulation=True, disable_reliability_descriptors=True,
            skip_test=True, resume="", eval_only_checkpoint="", workers=4,
            out_dir=str(self.root / "outputs_uq" / "run"), feature_cache=str(self.manifest_path),
            block_dir=str(self.block_dir), deg_map_dir=str(self.deg_dir), degradation_tag="deg_v3_mild",
            max_train_files=1, max_val_files=1, max_test_files=None, amp=True)
        validate_feature_args(argparse.Namespace(**common), self.root)
        for key, value in (("skip_test", False), ("resume", "x"), ("eval_only_checkpoint", "x"),
                           ("max_test_files", 1), ("amp", False), ("max_train_files", 2)):
            bad = dict(common); bad[key] = value
            with self.subTest(key=key), self.assertRaises((ValueError, RuntimeError)):
                validate_feature_args(argparse.Namespace(**bad), self.root)


if __name__ == "__main__":
    unittest.main()
