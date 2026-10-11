"""Export frozen U-Net decoder features from degraded train/val maps only."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
UNET_PATH = ROOT / "scripts/train/12_train_unet_25d_restoration.py"
MANIFEST_NAME = "feature_cache_manifest.json"
EXPECTED_SHAPE = (48, 100, 100)
EXPECTED_NORMALIZATION = {
    "stats_sha256": "7c85f5d61a316d9d8a576bfff57e9e284634fd3e760fbe15e834c80ed937df75",
    "protocol": "shared_25d_normalization_v3",
    "fit_split": "train",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def load_unet_module():
    spec = importlib.util.spec_from_file_location("frozen_feature_unet", UNET_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load U-Net source: {UNET_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_decoder_features(model, x):
    """Run the original U-Net layers once and return the tensor entering out."""
    e1 = model.enc1(x)
    e2 = model.enc2(model.pool(e1))
    e3 = model.enc3(model.pool(e2))
    e4 = model.enc4(model.pool(e3))
    b = model.bottleneck(model.pool(e4))
    d4 = model.dec4(torch.cat([model.up4(F.interpolate(b, size=e4.shape[-2:], mode="bilinear", align_corners=False)), e4], 1))
    d3 = model.dec3(torch.cat([model.up3(F.interpolate(d4, size=e3.shape[-2:], mode="bilinear", align_corners=False)), e3], 1))
    d2 = model.dec2(torch.cat([model.up2(F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)), e2], 1))
    return model.dec1(torch.cat([model.up1(F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)), e1], 1))


class DegradedMapDataset(Dataset):
    def __init__(self, files):
        self.files = list(files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        with np.load(path, allow_pickle=False) as data:
            maps = data["maps_norm"].astype(np.float32)
            masks = [data[k].astype(np.float32)[None] for k in (
                "valid_mask", "structural_hard_mask", "low_density_mask",
                "high_gradient_mask", "high_roughness_mask")]
        return torch.from_numpy(np.concatenate([maps, *masks])), str(path)


def _scalar(data, key):
    return str(np.asarray(data[key]).item()) if key in data else None


def _discover(degraded_dir, tag, maximum):
    result = {}
    for split in ("train", "val"):
        files = sorted((degraded_dir / split).glob(f"*_{tag}_25d.npz"))
        if maximum is not None:
            files = files[:maximum]
        if not files:
            raise RuntimeError(f"No {split} degraded maps found for {tag!r}")
        result[split] = files
    return result


def export_cache(args):
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for feature export")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    degraded_dir = Path(args.degraded_map_dir).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    uq_root = (ROOT / "outputs_uq").resolve()
    if cache_dir == uq_root or uq_root not in cache_dir.parents:
        raise ValueError(f"--cache-dir must be below {uq_root}")
    if cache_dir.exists() and any(cache_dir.iterdir()):
        raise FileExistsError(f"Refusing non-empty cache directory: {cache_dir}")
    files = _discover(degraded_dir, args.degradation_tag, args.max_files_per_split)
    input_hashes = {str(path): sha256(path) for paths in files.values() for path in paths}
    expected_bytes = sum(len(v) for v in files.values()) * int(np.prod(EXPECTED_SHAPE)) * 4
    free = shutil.disk_usage(cache_dir.parent if cache_dir.parent.exists() else uq_root).free
    required = int(expected_bytes * 1.10) + 1024 ** 3
    if free < required:
        raise RuntimeError(f"Insufficient disk space: need {required}, available {free}")
    cache_dir.mkdir(parents=True)
    manifest_path = cache_dir / MANIFEST_NAME
    source_before = sha256(UNET_PATH)
    checkpoint_before = sha256(checkpoint_path)
    manifest = {
        "schema_version": 1, "status": "running", "degradation_tag": args.degradation_tag,
        "feature_channels": 48, "spatial_shape": [100, 100], "dtype": "float32",
        "counts": {k: len(v) for k, v in files.items()}, "entries": [],
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_before},
        "sources": {"unet_trainer_path": str(UNET_PATH), "unet_trainer_sha256": source_before,
                    "exporter_sha256": sha256(Path(__file__).resolve())},
        "config": vars(args),
        "config_sha256": hashlib.sha256(json.dumps(vars(args), sort_keys=True).encode()).hexdigest(),
        "expected_bytes": expected_bytes,
        "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                    "device": torch.cuda.get_device_name(0), "amp": False,
                    "tf32_matmul": False, "tf32_cudnn": False},
    }
    write_json(manifest_path, manifest)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        unet = load_unet_module()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        ck_args = checkpoint.get("args", {})
        if int(ck_args.get("base_ch", 48)) != 48:
            raise RuntimeError("Checkpoint base_ch must be 48")
        if int(ck_args.get("in_ch", 14)) != 14 or int(ck_args.get("out_ch", 9)) != 9:
            raise RuntimeError("Checkpoint architecture must use in_ch=14 and out_ch=9")
        model = unet.UNet2D(in_ch=14, out_ch=9, base_ch=48, dropout=float(ck_args.get("dropout", 0.0)))
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval().requires_grad_(False).cuda()
        manifest["checkpoint"]["args"] = ck_args
        manifest["checkpoint"]["args_sha256"] = hashlib.sha256(
            json.dumps(ck_args, sort_keys=True, default=str).encode()
        ).hexdigest()
        manifest["frozen_unet_parameter_count"] = sum(p.numel() for p in model.parameters())
        normalization = None
        for split, split_files in files.items():
            (cache_dir / split).mkdir()
            loader = DataLoader(DegradedMapDataset(split_files), batch_size=args.batch_size,
                                shuffle=False, num_workers=min(4, max(0, args.workers)), pin_memory=True)
            for inputs, paths in loader:
                with torch.no_grad():
                    features = extract_decoder_features(model, inputs.cuda(non_blocking=True).float())
                    if args.verify_forward:
                        torch.testing.assert_close(torch.sigmoid(model.out(features)), model(inputs.cuda().float()))
                arrays = features.cpu().numpy().astype(np.float32, copy=False)
                for array, raw_path in zip(arrays, paths):
                    raw = Path(raw_path)
                    before = input_hashes[str(raw)]
                    if tuple(array.shape) != EXPECTED_SHAPE:
                        raise RuntimeError(f"Unexpected feature shape {array.shape} for {raw}")
                    if not np.isfinite(array).all():
                        raise RuntimeError(f"Non-finite feature values for {raw}")
                    target = cache_dir / split / f"{raw.stem}_features.npy"
                    np.save(target, array, allow_pickle=False)
                    with np.load(raw, allow_pickle=False) as data:
                        current_norm = {"stats_sha256": _scalar(data, "normalization_stats_sha256"),
                                        "protocol": _scalar(data, "normalization_protocol"),
                                        "fit_split": _scalar(data, "normalization_fit_split")}
                        if normalization is None:
                            normalization = current_norm
                        elif current_norm != normalization:
                            raise RuntimeError("Inconsistent normalization metadata")
                        if current_norm != EXPECTED_NORMALIZATION:
                            raise RuntimeError(
                                f"Unexpected train-fitted normalization metadata in {raw}: {current_norm}"
                            )
                        metadata = {k: float(np.asarray(data[k]).item()) for k in ("x0", "y0", "resolution")}
                    after = sha256(raw)
                    if before != after:
                        raise RuntimeError(f"Degraded source changed during export: {raw}")
                    manifest["entries"].append({"split": split, "degraded_filename": raw.name,
                        "feature_path": target.relative_to(cache_dir).as_posix(),
                        "degraded_sha256_before": before, "degraded_sha256_after": after,
                        "feature_sha256": sha256(target), "shape": list(array.shape),
                        "dtype": str(array.dtype), **metadata})
                write_json(manifest_path, manifest)
                elapsed = max(time.monotonic() - started, 1e-9)
                completed = sum(e["split"] == split for e in manifest["entries"])
                print(f"[EXPORT] split={split} files={completed}/{len(split_files)} elapsed={elapsed:.1f}s files_per_sec={len(manifest['entries']) / elapsed:.2f}", flush=True)
        if sha256(UNET_PATH) != source_before or sha256(checkpoint_path) != checkpoint_before:
            raise RuntimeError("U-Net source or checkpoint changed during export")
        manifest["sources"]["unet_trainer_sha256_after"] = sha256(UNET_PATH)
        manifest["checkpoint"]["sha256_after"] = sha256(checkpoint_path)
        manifest.update(status="complete", normalization=normalization,
                        walltime_seconds=time.monotonic() - started,
                        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        write_json(manifest_path, manifest)
    except BaseException as exc:
        manifest.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc)},
                        walltime_seconds=time.monotonic() - started,
                        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        write_json(manifest_path, manifest)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--degraded-map-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--degradation-tag", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-files-per-split", type=int)
    parser.add_argument("--verify-forward", action="store_true")
    export_cache(parser.parse_args(argv))


if __name__ == "__main__":
    main()
