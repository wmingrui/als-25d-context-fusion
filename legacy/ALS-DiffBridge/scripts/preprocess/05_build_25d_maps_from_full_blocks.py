from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import ndimage

RAW_CHANNELS = [
    "height_max",
    "height_mean",
    "height_min",
    "height_std",
    "density",
    "log_density",
    "intensity_mean",
    "height_gradient",
    "boundary_response",
]

NORM_CHANNELS = [
    "height_max_norm",
    "height_mean_norm",
    "height_min_norm",
    "height_std_norm",
    "density_norm",
    "log_density_norm",
    "intensity_mean",
    "height_gradient_norm",
    "boundary_response",
]

CLASS_NAMES = {
    0: "ground",
    1: "vegetation",
    2: "cars",
    3: "trucks",
    4: "power_lines",
    5: "fences",
    6: "poles",
    7: "buildings",
}

def fill_nearest(arr, valid, fill_value=0.0):
    arr = arr.astype(np.float32)
    if valid.all():
        return arr.copy()
    if not valid.any():
        return np.full_like(arr, fill_value, dtype=np.float32)


    _, indices = ndimage.distance_transform_edt(~valid, return_indices=True)
    filled = arr[tuple(indices)]
    return filled.astype(np.float32)

def safe_percentile(values, q, default=1.0):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return default
    return float(np.percentile(values, q))

def build_one_map(npz_path: Path, out_path: Path, resolution: float, num_classes: int = 8):
    d = np.load(npz_path, allow_pickle=True)

    points = d["points"].astype(np.float32)
    intensity = d["intensity"].astype(np.float32)
    labels = d["labels"].astype(np.int64)

    block_size = float(d["block_size"])
    x0 = float(d["x0"])
    y0 = float(d["y0"])

    H = int(np.ceil(block_size / resolution))
    W = int(np.ceil(block_size / resolution))
    M = H * W

    x_rel = points[:, 0] - x0
    y_rel = points[:, 1] - y0
    z = points[:, 2]

    ix = np.floor(x_rel / resolution).astype(np.int32)
    iy = np.floor(y_rel / resolution).astype(np.int32)

    ix = np.clip(ix, 0, W - 1)
    iy = np.clip(iy, 0, H - 1)

    flat = iy * W + ix

    count = np.bincount(flat, minlength=M).astype(np.float32)
    valid_flat = count > 0

    z_sum = np.bincount(flat, weights=z, minlength=M).astype(np.float32)
    z2_sum = np.bincount(flat, weights=z * z, minlength=M).astype(np.float32)
    inten_sum = np.bincount(flat, weights=intensity, minlength=M).astype(np.float32)

    height_mean = np.zeros(M, dtype=np.float32)
    height_std = np.zeros(M, dtype=np.float32)
    intensity_mean = np.zeros(M, dtype=np.float32)

    height_mean[valid_flat] = z_sum[valid_flat] / count[valid_flat]
    z2_mean = np.zeros(M, dtype=np.float32)
    z2_mean[valid_flat] = z2_sum[valid_flat] / count[valid_flat]
    height_std[valid_flat] = np.sqrt(np.maximum(z2_mean[valid_flat] - height_mean[valid_flat] ** 2, 0.0))

    intensity_mean[valid_flat] = inten_sum[valid_flat] / count[valid_flat]
    intensity_mean = intensity_mean / 65535.0

    height_max = np.full(M, -np.inf, dtype=np.float32)
    height_min = np.full(M, np.inf, dtype=np.float32)
    np.maximum.at(height_max, flat, z)
    np.minimum.at(height_min, flat, z)

    height_max[~valid_flat] = 0.0
    height_min[~valid_flat] = 0.0

    density = count
    log_density = np.log1p(density).astype(np.float32)

    valid = valid_flat.reshape(H, W)

    height_max_2d = height_max.reshape(H, W)
    height_mean_2d = height_mean.reshape(H, W)
    height_min_2d = height_min.reshape(H, W)
    height_std_2d = height_std.reshape(H, W)
    density_2d = density.reshape(H, W)
    log_density_2d = log_density.reshape(H, W)
    intensity_mean_2d = intensity_mean.reshape(H, W)


    height_for_grad = fill_nearest(height_mean_2d, valid, fill_value=0.0)

    gy, gx = np.gradient(height_for_grad, resolution, resolution)
    height_gradient = np.sqrt(gx * gx + gy * gy).astype(np.float32)

    grad_valid_values = height_gradient[valid]
    grad_p95 = safe_percentile(grad_valid_values, 95, default=1.0)
    if grad_p95 < 1e-6:
        grad_p95 = 1.0
    boundary_response = np.clip(height_gradient / grad_p95, 0.0, 1.0).astype(np.float32)

    # Label map: majority semantic class per cell. -1 means invalid cell.
    label_counts = np.zeros((num_classes, M), dtype=np.int32)
    for c in range(num_classes):
        m = labels == c
        if np.any(m):
            label_counts[c] = np.bincount(flat[m], minlength=M).astype(np.int32)

    label_map_flat = np.argmax(label_counts, axis=0).astype(np.int16)
    label_map_flat[~valid_flat] = -1
    label_map = label_map_flat.reshape(H, W)


    density_valid = density_2d[valid]
    rough_valid = height_std_2d[valid]
    grad_valid = height_gradient[valid]

    density_q25 = safe_percentile(density_valid, 25, default=1.0)
    rough_q85 = safe_percentile(rough_valid, 85, default=0.0)
    grad_q85 = safe_percentile(grad_valid, 85, default=0.0)

    low_density_mask = valid & (density_2d <= density_q25)
    high_roughness_mask = valid & (height_std_2d >= rough_q85)
    high_gradient_mask = valid & (height_gradient >= grad_q85)

    structural_hard_mask = valid & (
        low_density_mask | high_roughness_mask | high_gradient_mask
    )

    # Raw physical maps.
    maps_raw = np.stack(
        [
            height_max_2d,
            height_mean_2d,
            height_min_2d,
            height_std_2d,
            density_2d,
            log_density_2d,
            intensity_mean_2d,
            height_gradient,
            boundary_response,
        ],
        axis=0,
    ).astype(np.float32)

    # Per-block normalized maps for direct U-Net / diffusion input.
    z_valid = points[:, 2]
    z_min = float(np.percentile(z_valid, 1))
    z_max = float(np.percentile(z_valid, 99))
    z_range = max(z_max - z_min, 1e-3)

    density_p95 = safe_percentile(density_valid, 95, default=1.0)
    log_density_p95 = np.log1p(max(density_p95, 1.0))

    rough_p95 = safe_percentile(rough_valid, 95, default=1.0)
    grad_p95 = safe_percentile(grad_valid, 95, default=1.0)

    if rough_p95 < 1e-6:
        rough_p95 = 1.0
    if grad_p95 < 1e-6:
        grad_p95 = 1.0

    height_max_norm = np.clip((height_max_2d - z_min) / z_range, 0.0, 1.0)
    height_mean_norm = np.clip((height_mean_2d - z_min) / z_range, 0.0, 1.0)
    height_min_norm = np.clip((height_min_2d - z_min) / z_range, 0.0, 1.0)
    height_std_norm = np.clip(height_std_2d / rough_p95, 0.0, 1.0)
    density_norm = np.clip(density_2d / max(density_p95, 1.0), 0.0, 1.0)
    log_density_norm = np.clip(log_density_2d / max(log_density_p95, 1e-6), 0.0, 1.0)
    height_gradient_norm = np.clip(height_gradient / grad_p95, 0.0, 1.0)

    maps_norm = np.stack(
        [
            height_max_norm,
            height_mean_norm,
            height_min_norm,
            height_std_norm,
            density_norm,
            log_density_norm,
            intensity_mean_2d,
            height_gradient_norm,
            boundary_response,
        ],
        axis=0,
    ).astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        maps_raw=maps_raw,
        maps_norm=maps_norm,
        raw_channel_names=np.array(RAW_CHANNELS),
        norm_channel_names=np.array(NORM_CHANNELS),
        valid_mask=valid.astype(np.bool_),
        label_map=label_map.astype(np.int16),
        low_density_mask=low_density_mask.astype(np.bool_),
        high_gradient_mask=high_gradient_mask.astype(np.bool_),
        high_roughness_mask=high_roughness_mask.astype(np.bool_),
        structural_hard_mask=structural_hard_mask.astype(np.bool_),
        resolution=np.float32(resolution),
        block_size=np.float32(block_size),
        x0=np.float32(x0),
        y0=np.float32(y0),
        tile=d["tile"],
        split=d["split"],
        source_block=np.array(npz_path.name),
        z_min=np.float32(z_min),
        z_max=np.float32(z_max),
        density_q25=np.float32(density_q25),
        density_p95=np.float32(density_p95),
        rough_q85=np.float32(rough_q85),
        rough_p95=np.float32(rough_p95),
        grad_q85=np.float32(grad_q85),
        grad_p95=np.float32(grad_p95),
    )

    row = {
        "split": str(d["split"]),
        "tile": str(d["tile"]),
        "source_block": npz_path.name,
        "map_file": out_path.name,
        "num_points": int(points.shape[0]),
        "H": H,
        "W": W,
        "resolution": resolution,
        "valid_cells": int(valid.sum()),
        "valid_ratio": float(valid.mean()),
        "hard_cells": int(structural_hard_mask.sum()),
        "hard_ratio_valid": float(structural_hard_mask.sum() / max(valid.sum(), 1)),
        "z_min_p1": z_min,
        "z_max_p99": z_max,
        "density_q25": float(density_q25),
        "density_p95": float(density_p95),
        "rough_q85": float(rough_q85),
        "grad_q85": float(grad_q85),
    }


    for cid in range(num_classes):
        row[f"cells_class_{cid}_{CLASS_NAMES[cid]}"] = int(np.sum(label_map == cid))

    return row

def collect_files(input_dir: Path, split: str):
    return sorted((input_dir / split).glob("*.npz"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/root/autodl-tmp/ALS-DiffBridge/data/processed/dales_objects/full_blocks_b50",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/root/autodl-tmp/ALS-DiffBridge/data/processed/dales_objects/maps_25d_b50_r050",
    )
    parser.add_argument("--resolution", type=float, default=0.5)
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--max_files_per_split", type=int, default=None)

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    config = {
        "input_dir": str(input_dir),
        "out_dir": str(out_dir),
        "resolution": args.resolution,
        "raw_channels": RAW_CHANNELS,
        "norm_channels": NORM_CHANNELS,
        "class_names": CLASS_NAMES,
    }
    with open(out_dir / "map_config.json", "w") as f:
        json.dump(config, f, indent=2)

    all_rows = []

    for split in splits:
        files = collect_files(input_dir, split)
        if args.max_files_per_split is not None:
            files = files[:args.max_files_per_split]

        print(f"\n[PROCESS SPLIT] {split}, files={len(files)}")
        split_out = out_dir / split
        split_out.mkdir(parents=True, exist_ok=True)

        for fp in tqdm(files, desc=f"maps {split}"):
            out_name = fp.stem + "_25d.npz"
            out_fp = split_out / out_name
            row = build_one_map(fp, out_fp, resolution=args.resolution)
            all_rows.append(row)

        if len(all_rows) > 0:
            pd.DataFrame(all_rows).to_csv(out_dir / "maps_metadata_partial.csv", index=False)

    meta = pd.DataFrame(all_rows)
    meta_csv = out_dir / "maps_metadata.csv"
    meta.to_csv(meta_csv, index=False)

    print("\n[SAVED]", meta_csv)
    print("[NUM MAPS BY SPLIT]")
    print(meta.groupby("split")["map_file"].count())

    print("\n[VALID / HARD SUMMARY]")
    print(meta.groupby("split")[["valid_ratio", "hard_ratio_valid"]].mean())

    print("\n[DONE]")

if __name__ == "__main__":
    main()
