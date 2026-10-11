"""Bounded R1 observation-only / R5 restoration training adapter.

The original trainer remains the single implementation of sampling, RandLA-Net,
optimization, checkpointing, and validation.  This module changes only the R1
prior source and adds launch-time safety/provenance controls.
"""

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


ROOT = Path(__file__).resolve().parents[2]
TRAINER_PATH = ROOT / "scripts/train/39_train_randlanet_reliability_ablation.py"
CONTROLS = ("observation", "restoration")
EXPECTED_NORMALIZATION_SHA256 = (
    "7c85f5d61a316d9d8a576bfff57e9e284634fd3e760fbe15e834c80ed937df75"
)
EXPECTED_NORMALIZATION_PROTOCOL = "shared_25d_normalization_v3"
EXPECTED_NORMALIZATION_FIT_SPLIT = "train"
_TRAINER = None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def configure_runtime() -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "4"
    torch.set_num_threads(4)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(4)
        except RuntimeError:
            pass
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_trainer():
    """Load the frozen trainer while forcing its pure-PyTorch backend."""
    global _TRAINER
    if _TRAINER is not None:
        return _TRAINER
    train_dir = str(TRAINER_PATH.parent)
    if train_dir not in sys.path:
        sys.path.insert(0, train_dir)
    saved_open3d = sys.modules.get("open3d", ...)
    sys.modules["open3d"] = None
    try:
        spec = importlib.util.spec_from_file_location(
            "restoration_control_frozen_trainer", TRAINER_PATH
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load trainer: {TRAINER_PATH}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if saved_open3d is ...:
            sys.modules.pop("open3d", None)
        else:
            sys.modules["open3d"] = saved_open3d
    if not str(module.RANDLANET_BACKEND).startswith("pure_torch_fallback"):
        raise RuntimeError("Restoration controls require pure_torch_fallback")
    module._RESTORATION_CONTROL_ORIGINAL_DATASET = module.DALESRandLADataset
    _TRAINER = module
    return module


def install_control(trainer, control: str):
    """Select the dataset implementation and return the selected class."""
    if control not in CONTROLS:
        raise ValueError(f"Unknown control: {control!r}")
    original = trainer._RESTORATION_CONTROL_ORIGINAL_DATASET
    if control == "restoration":
        trainer.DALESRandLADataset = original
        return original

    class ObservationOnlyDataset(original):
        """Use degraded maps twice without opening clean/restored products."""

        def _load_prior_files(self, block_stem):
            deg_fp = (
                self.deg_map_dir / self.split / f"{block_stem}_25d.npz"
            )
            degraded = trainer.load_npz(deg_fp)
            expected_normalization = {
                "normalization_stats_sha256": EXPECTED_NORMALIZATION_SHA256,
                "normalization_protocol": EXPECTED_NORMALIZATION_PROTOCOL,
                "normalization_fit_split": EXPECTED_NORMALIZATION_FIT_SPLIT,
            }
            actual_normalization = {
                key: str(np.asarray(degraded[key]).item())
                if key in degraded else None
                for key in expected_normalization
            }
            if actual_normalization != expected_normalization:
                raise RuntimeError(
                    "Observation control normalization metadata mismatch: "
                    f"expected {expected_normalization}, got {actual_normalization} "
                    f"in {deg_fp}"
                )
            return degraded, degraded

        def __getitem__(self, idx):
            sample = super().__getitem__(idx)
            degraded = sample["degraded"].copy()
            sample["delta"] = degraded.copy()
            sample["restored"] = np.zeros_like(degraded)
            sample["rel"] = np.zeros_like(sample["rel"])
            return sample

    ObservationOnlyDataset.__name__ = "ObservationOnlyDALESRandLADataset"
    trainer.DALESRandLADataset = ObservationOnlyDataset
    return ObservationOnlyDataset


def validate_args(args, root: Path = ROOT, control: str = "observation") -> Path:
    if control not in CONTROLS:
        raise ValueError(f"--control must be one of {CONTROLS}")
    required = {
        "mode": "restored_rgdi",
        "inject_stages": 1,
        "gate_mode": "learned",
        "disable_explicit_quality_modulation": True,
        "disable_reliability_descriptors": True,
        "skip_test": True,
    }
    for name, expected in required.items():
        if getattr(args, name, None) != expected:
            raise ValueError(f"--{name} must be {expected!r} for this protocol")
    if getattr(args, "resume", ""):
        raise ValueError("--resume is forbidden for restoration controls")
    if getattr(args, "eval_only_checkpoint", ""):
        raise ValueError("--eval_only_checkpoint is forbidden for restoration controls")

    root = Path(root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir = out_dir.resolve()
    uq_root = (root / "outputs_uq").resolve()
    if out_dir == uq_root or uq_root not in out_dir.parents:
        raise ValueError(f"--out_dir must be below outputs_uq: {uq_root}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing non-empty output directory: {out_dir}")
    args.out_dir = str(out_dir)
    args.workers = min(4, max(0, int(args.workers)))
    return out_dir


def run_training(
    trainer, args, control: str, forwarded_argv: list[str], root: Path = ROOT
) -> None:
    out_dir = validate_args(args, root, control)
    install_control(trainer, control)
    out_dir.mkdir(parents=True, exist_ok=True)

    semantics = (
        "R1 observation-only: degraded d feeds both independent learned "
        "degraded/correction embeddings; restored and reliability tensors are zero"
        if control == "observation"
        else "R5 complete restoration: original restored-minus-degraded correction"
    )
    source_hashes = {
        "adapter_sha256": _sha256(Path(__file__).resolve()),
        "trainer_sha256": _sha256(TRAINER_PATH),
    }
    args.restoration_control = control
    args.control = control
    args.correction_semantics = semantics
    args.expected_normalization_sha256 = EXPECTED_NORMALIZATION_SHA256
    args.expected_normalization_protocol = EXPECTED_NORMALIZATION_PROTOCOL
    args.expected_normalization_fit_split = EXPECTED_NORMALIZATION_FIT_SPLIT
    args.forwarded_cli_args = list(forwarded_argv)
    args.source_hashes = source_hashes

    metadata = {
        "status": "running",
        "control": control,
        "correction_semantics": semantics,
        "expected_normalization_sha256": EXPECTED_NORMALIZATION_SHA256,
        "expected_normalization_protocol": EXPECTED_NORMALIZATION_PROTOCOL,
        "expected_normalization_fit_split": EXPECTED_NORMALIZATION_FIT_SPLIT,
        "source_hashes": source_hashes,
        "forwarded_cli_args": list(forwarded_argv),
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "randlanet_backend": trainer.RANDLANET_BACKEND,
            "thread_cap": 4,
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        },
    }
    metadata_path = out_dir / "control_run_metadata.json"
    _write_json(metadata_path, metadata)
    started = time.monotonic()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        trainer._RESTORATION_CONTROL_ORIGINAL_TRAIN(args)
    except BaseException as exc:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        metadata.update(
            status="failed",
            failure={"type": type(exc).__name__, "message": str(exc)},
            walltime_seconds=time.monotonic() - started,
            peak_cuda_allocated_bytes=(
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            ),
        )
        _write_json(metadata_path, metadata)
        raise
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    metadata.update(
        status="complete",
        walltime_seconds=time.monotonic() - started,
        peak_cuda_allocated_bytes=(
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        ),
    )
    _write_json(metadata_path, metadata)


def main(argv=None, *, allow_cpu_for_tests: bool = False) -> None:
    configure_runtime()
    argv = list(sys.argv[1:] if argv is None else argv)
    control_parser = argparse.ArgumentParser(add_help=False)
    control_parser.add_argument("--control", required=True, choices=CONTROLS)
    control_ns, forwarded = control_parser.parse_known_args(argv)
    if not allow_cpu_for_tests and not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for restoration-control training")

    trainer = load_trainer()
    original_train = trainer.train
    trainer._RESTORATION_CONTROL_ORIGINAL_TRAIN = original_train
    trainer.train = lambda args: run_training(
        trainer, args, control_ns.control, forwarded
    )
    saved_argv = sys.argv
    try:
        sys.argv = [str(TRAINER_PATH), *forwarded]
        trainer.main()
    finally:
        sys.argv = saved_argv
        trainer.train = original_train


if __name__ == "__main__":
    main()
