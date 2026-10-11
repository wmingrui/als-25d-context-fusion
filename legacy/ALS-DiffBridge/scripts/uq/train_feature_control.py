"""Frozen U-Net feature control for the unchanged RandLA-Net train loop."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[2]
TRAINER_PATH = ROOT / "scripts/train/39_train_randlanet_reliability_ablation.py"
EXPECTED_NORMALIZATION_PROTOCOL = "shared_25d_normalization_v3"
EXPECTED_NORMALIZATION_SHA256 = "7c85f5d61a316d9d8a576bfff57e9e284634fd3e760fbe15e834c80ed937df75"
CHECKPOINT_SHA256_BY_TAG = {
    "deg_v3_mild": "fefef416d23580df46e7fcb5cd83d819fb9441ac255e94eceb079bea9e0036fe",
    "deg_v3_medium": "7328146ed0c810951e75418e7d9434047cecea8b9e9dad921164031f198c49ff",
    "deg_v3_severe": "979d683a6de65e10c101e85bdca850a5fdcd31eeece6aea1ab77b995bef75b0e",
}
_TRAINER = None


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def configure_runtime():
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "4"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_trainer():
    global _TRAINER
    if _TRAINER is not None:
        return _TRAINER
    train_dir = str(TRAINER_PATH.parent)
    if train_dir not in sys.path:
        sys.path.insert(0, train_dir)
    saved = sys.modules.get("open3d", ...)
    sys.modules["open3d"] = None
    try:
        spec = importlib.util.spec_from_file_location("feature_control_frozen_trainer", TRAINER_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if saved is ...:
            sys.modules.pop("open3d", None)
        else:
            sys.modules["open3d"] = saved
    if not str(module.RANDLANET_BACKEND).startswith("pure_torch_fallback"):
        raise RuntimeError("Feature control requires pure_torch_fallback")
    module._FEATURE_CONTROL_ORIGINAL_DATASET = module.DALESRandLADataset
    module._FEATURE_CONTROL_ORIGINAL_MODEL = module.RandLASegmentationModel
    _TRAINER = module
    return module


def _safe_feature_path(root, relative):
    candidate = (root / relative).resolve()
    if root != candidate and root not in candidate.parents:
        raise ValueError(f"Invalid feature path outside cache: {relative}")
    return candidate


def validate_feature_cache(manifest_path, block_dir, deg_dir, tag, max_train, max_val):
    manifest_path = Path(manifest_path).resolve()
    cache_root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError("Feature manifest must have status complete")
    if manifest.get("degradation_tag") != tag:
        raise ValueError("Feature manifest degradation tag mismatch")
    if int(manifest.get("feature_channels", -1)) != 48:
        raise ValueError("Feature manifest must contain 48 channels")
    expected_config_hash = hashlib.sha256(
        json.dumps(manifest.get("config", {}), sort_keys=True).encode()
    ).hexdigest()
    if manifest.get("config_sha256") != expected_config_hash:
        raise RuntimeError("Feature manifest config hash mismatch")
    checkpoint = manifest.get("checkpoint", {})
    checkpoint_path = Path(checkpoint.get("path", ""))
    if not checkpoint_path.is_file() or sha256(checkpoint_path) != checkpoint.get("sha256"):
        raise RuntimeError("Feature manifest checkpoint hash mismatch")
    normalization = manifest.get("normalization", {})
    if (normalization.get("stats_sha256") != EXPECTED_NORMALIZATION_SHA256 or
            normalization.get("protocol") != EXPECTED_NORMALIZATION_PROTOCOL or
            normalization.get("fit_split") != "train"):
        raise RuntimeError("Feature manifest normalization protocol mismatch")
    by_key = {}
    for entry in manifest.get("entries", []):
        split = entry.get("split")
        if split not in ("train", "val"):
            raise ValueError(f"Invalid manifest split: {split!r}")
        key = (split, entry.get("degraded_filename"))
        if key in by_key:
            raise ValueError(f"Duplicate manifest entry: {key}")
        relative = entry.get("feature_path", "")
        feature_path = _safe_feature_path(cache_root, relative)
        if not feature_path.is_file():
            raise FileNotFoundError(f"Missing feature cache file: {feature_path}")
        if sha256(feature_path) != entry.get("feature_sha256"):
            raise RuntimeError(f"Feature hash mismatch: {feature_path}")
        array = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        if array.dtype != np.float32 or array.ndim != 3 or array.shape[0] != 48 or list(array.shape) != entry.get("shape"):
            raise RuntimeError(f"Invalid cached feature dimensions/dtype: {feature_path}")
        if not np.isfinite(array).all():
            raise RuntimeError(f"Non-finite cached feature values: {feature_path}")
        degraded = Path(deg_dir) / split / entry["degraded_filename"]
        if not degraded.is_file() or sha256(degraded) != entry.get("degraded_sha256_after"):
            raise RuntimeError(f"Degraded input hash mismatch: {degraded}")
        with np.load(degraded, allow_pickle=False) as data:
            if tuple(array.shape[1:]) != tuple(data["maps_norm"].shape[1:]):
                raise RuntimeError(f"Feature/map dimensions mismatch: {feature_path}")
            actual = {"stats_sha256": str(np.asarray(data["normalization_stats_sha256"]).item()),
                      "protocol": str(np.asarray(data["normalization_protocol"]).item()),
                      "fit_split": str(np.asarray(data["normalization_fit_split"]).item())}
            if actual != normalization:
                raise RuntimeError(f"Input normalization hash mismatch: {degraded}")
            for metadata_name in ("x0", "y0", "resolution"):
                actual_value = float(np.asarray(data[metadata_name]).item())
                if actual_value != float(entry[metadata_name]):
                    raise RuntimeError(
                        f"Feature alignment metadata mismatch for {metadata_name}: {degraded}"
                    )
        by_key[key] = entry
    for split, maximum in (("train", max_train), ("val", max_val)):
        manifest_count = int(manifest.get("counts", {}).get(split, -1))
        if maximum is not None and int(maximum) != manifest_count:
            raise RuntimeError(
                f"Feature cache coverage mismatch for {split}: "
                f"--max_{split}_files={maximum}, manifest count={manifest_count}"
            )
        blocks = sorted((Path(block_dir) / split).glob(f"*_{tag}.npz"))
        if maximum is not None:
            blocks = blocks[:maximum]
        expected = {(split, f"{path.stem}_25d.npz") for path in blocks}
        actual = {key for key in by_key if key[0] == split}
        if actual != expected:
            raise RuntimeError(f"Feature cache coverage mismatch for {split}: expected {len(expected)}, got {len(actual)}")
        if manifest_count != len(actual):
            raise RuntimeError(f"Feature manifest count mismatch for {split}")
    manifest["_manifest_path"] = str(manifest_path)
    manifest["_entries_by_key"] = by_key
    return manifest


def install_feature_dataset(trainer, manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = {(e["split"], e["degraded_filename"]): e for e in manifest["entries"]}
    cache_root = manifest_path.parent
    original = trainer._FEATURE_CONTROL_ORIGINAL_DATASET

    class FeatureDataset(original):
        """The `restored` field carries point-aligned 48-D latent features."""
        def __getitem__(self, idx):
            fp = self.files[idx]
            with np.load(fp, allow_pickle=True) as block:
                points = block["points"].astype(np.float32)
                points_local = block["points_local"].astype(np.float32)
                intensity = block["features"].astype(np.float32)
                labels = block["labels"].astype(np.int64)
                hard_all = block["point_hard_mask"].astype(np.bool_) if "point_hard_mask" in block else np.zeros(len(points), dtype=np.bool_)
            n = len(points)
            if self.split == "train":
                choice = np.random.choice(n, self.num_points, replace=n < self.num_points)
            else:
                choice = np.random.default_rng(trainer.stable_seed(self.seed, fp.name)).choice(n, self.num_points, replace=n < self.num_points)
            pts, coords, inten, y = points[choice], points_local[choice].astype(np.float32), intensity[choice], labels[choice]
            hard = hard_all[choice].astype(np.float32)
            xyz_feature = coords.copy()
            xyz_feature[:, 0] /= 25.0; xyz_feature[:, 1] /= 25.0
            xyz_feature[:, 2] = np.clip(xyz_feature[:, 2] / 20.0, -2.0, 2.0)
            base = np.concatenate([xyz_feature, inten], axis=1).astype(np.float32)
            degraded_path = self.deg_map_dir / self.split / f"{fp.stem}_25d.npz"
            degraded_npz = trainer.load_npz(degraded_path)
            prior_degraded = trainer.sample_maps(pts, degraded_npz)
            entry = entries[(self.split, degraded_path.name)]
            latent = np.load(_safe_feature_path(cache_root, entry["feature_path"]), allow_pickle=False)
            latent_npz = {"maps_norm": latent, "x0": entry["x0"], "y0": entry["y0"], "resolution": entry["resolution"]}
            prior_features = trainer.sample_maps(pts, latent_npz)
            hierarchy = trainer.build_randlanet_hierarchy(coords=coords, num_layers=self.num_layers,
                num_neighbors=self.num_neighbors, sub_sampling_ratio=self.sub_sampling_ratio)
            return {"base": base, "restored": prior_features, "degraded": prior_degraded,
                    "delta": np.zeros((self.num_points, 9), np.float32),
                    "rel": np.zeros((self.num_points, 6), np.float32), "labels": y, "hard": hard,
                    "coords": hierarchy["coords"], "neighbor_indices": hierarchy["neighbor_indices"],
                    "sub_idx": hierarchy["sub_idx"], "interp_idx": hierarchy["interp_idx"], "name": fp.name}

    FeatureDataset.__name__ = "FrozenUNetFeatureDALESRandLADataset"
    trainer.DALESRandLADataset = FeatureDataset
    return FeatureDataset


class FeatureFusionRandLASegmentationModel(load_trainer()._FEATURE_CONTROL_ORIGINAL_MODEL):
    def __init__(self, *args, feature_channels=48, **kwargs):
        super().__init__(*args, **kwargs)
        with torch.random.fork_rng(devices=[]):
            self.feature_projection = nn.Linear(feature_channels, 9, bias=True)

    def forward(self, batch):
        local = dict(batch)
        local["delta"] = self.feature_projection(batch["restored"])
        local["restored"] = torch.zeros_like(local["degraded"])
        return super().forward(local)


def validate_feature_args(args, root=ROOT):
    required = {"mode": "restored_rgdi", "inject_stages": 1, "gate_mode": "learned",
                "disable_explicit_quality_modulation": True, "disable_reliability_descriptors": True,
                "skip_test": True, "amp": True}
    for name, expected in required.items():
        if getattr(args, name, None) != expected:
            raise ValueError(f"--{name} must be {expected!r} for pretrained feature control")
    for name in ("resume", "eval_only_checkpoint"):
        if getattr(args, name, ""):
            raise ValueError(f"--{name} is forbidden for pretrained feature control")
    if getattr(args, "max_test_files", None) is not None:
        raise ValueError("--max_test_files is forbidden")
    out = Path(args.out_dir); root = Path(root).resolve()
    out = (root / out).resolve() if not out.is_absolute() else out.resolve()
    uq = (root / "outputs_uq").resolve()
    if out == uq or uq not in out.parents:
        raise ValueError(f"--out_dir must be below {uq}")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing non-empty output directory: {out}")
    args.out_dir = str(out); args.workers = min(4, max(0, int(args.workers)))
    manifest = validate_feature_cache(args.feature_cache, args.block_dir, args.deg_map_dir,
                                      args.degradation_tag, args.max_train_files, args.max_val_files)
    expected_checkpoint = CHECKPOINT_SHA256_BY_TAG.get(args.degradation_tag)
    if expected_checkpoint is None:
        raise ValueError(f"No frozen checkpoint registered for severity {args.degradation_tag!r}")
    if manifest["checkpoint"]["sha256"] != expected_checkpoint:
        raise RuntimeError(
            f"Checkpoint SHA256 does not match frozen {args.degradation_tag} protocol"
        )
    return manifest


def run_training(trainer, args, forwarded, root=ROOT):
    manifest = validate_feature_args(args, root)
    out = Path(args.out_dir); out.mkdir(parents=True)
    install_feature_dataset(trainer, args.feature_cache)
    trainer.RandLASegmentationModel = FeatureFusionRandLASegmentationModel
    args.control = "pretrained_feature"
    args.feature_manifest_sha256 = sha256(args.feature_cache)
    args.feature_checkpoint_sha256 = manifest["checkpoint"]["sha256"]
    args.feature_source_hashes = {
        **manifest.get("sources", {}),
        "feature_control_adapter_sha256": sha256(Path(__file__).resolve()),
        "randlanet_trainer_sha256": sha256(TRAINER_PATH),
    }
    args.feature_projection_trainable_parameters = 441
    args.frozen_unet_parameter_count = int(manifest["frozen_unet_parameter_count"])
    args.runtime_amp = True; args.runtime_tf32 = False
    metadata = {"status": "running", "control": "pretrained_feature", "manifest_sha256": args.feature_manifest_sha256,
                "checkpoint_sha256": args.feature_checkpoint_sha256, "source_hashes": args.feature_source_hashes,
                "additional_trainable_parameters": 441,
                "frozen_unet_parameter_count": args.frozen_unet_parameter_count,
                "runtime": {"python": platform.python_version(),
                "torch": str(torch.__version__), "amp": True, "tf32_matmul": False, "tf32_cudnn": False}}
    path = out / "control_run_metadata.json"; write_json(path, metadata)
    started = time.monotonic()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    try:
        trainer._FEATURE_CONTROL_ORIGINAL_TRAIN(args)
    except BaseException as exc:
        metadata.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc)}, walltime_seconds=time.monotonic()-started,
                        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0)
        write_json(path, metadata); raise
    metadata.update(status="complete", walltime_seconds=time.monotonic()-started,
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0)
    write_json(path, metadata)


def main(argv=None, allow_cpu_for_tests=False):
    configure_runtime(); argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False); parser.add_argument("--feature-cache", required=True)
    control, forwarded = parser.parse_known_args(argv)
    if not allow_cpu_for_tests and not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for feature-control training")
    trainer = load_trainer(); original_train = trainer.train
    trainer._FEATURE_CONTROL_ORIGINAL_TRAIN = original_train
    trainer.train = lambda args: (setattr(args, "feature_cache", control.feature_cache), run_training(trainer, args, forwarded))[1]
    saved = sys.argv
    try:
        sys.argv = [str(TRAINER_PATH), *forwarded]; trainer.main()
    finally:
        sys.argv = saved; trainer.train = original_train
        trainer.DALESRandLADataset = trainer._FEATURE_CONTROL_ORIGINAL_DATASET
        trainer.RandLASegmentationModel = trainer._FEATURE_CONTROL_ORIGINAL_MODEL


if __name__ == "__main__":
    main()
