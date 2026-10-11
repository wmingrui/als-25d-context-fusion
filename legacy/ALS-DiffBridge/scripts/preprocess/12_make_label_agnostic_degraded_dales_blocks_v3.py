from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from tqdm import tqdm


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




SEVERITY_PRESETS = {
    "mild": {
        "base_keep_ratio": 0.88,
        "low_density_keep_ratio": 0.65,
        "roughness_keep_ratio": 0.72,
        "boundary_keep_ratio": 0.62,
        "hard_keep_ratio": 0.60,
        "num_holes": 1,
        "hole_min_size": 2.0,
        "hole_max_size": 5.0,
        "hole_keep_ratio": 0.35,
        "num_stripes": 1,
        "stripe_width_min": 0.5,
        "stripe_width_max": 1.2,
        "stripe_keep_ratio": 0.40,
        "add_z_noise": True,
        "normal_z_noise_sigma": 0.005,
        "hard_z_noise_sigma": 0.025,
        "min_keep_points": 2048,
    },
    "medium": {
        "base_keep_ratio": 0.75,
        "low_density_keep_ratio": 0.45,
        "roughness_keep_ratio": 0.55,
        "boundary_keep_ratio": 0.40,
        "hard_keep_ratio": 0.45,
        "num_holes": 2,
        "hole_min_size": 3.0,
        "hole_max_size": 8.0,
        "hole_keep_ratio": 0.15,
        "num_stripes": 1,
        "stripe_width_min": 0.8,
        "stripe_width_max": 2.0,
        "stripe_keep_ratio": 0.20,
        "add_z_noise": True,
        "normal_z_noise_sigma": 0.010,
        "hard_z_noise_sigma": 0.060,
        "min_keep_points": 2048,
    },
    "severe": {
        "base_keep_ratio": 0.62,
        "low_density_keep_ratio": 0.28,
        "roughness_keep_ratio": 0.36,
        "boundary_keep_ratio": 0.25,
        "hard_keep_ratio": 0.28,
        "num_holes": 3,
        "hole_min_size": 4.0,
        "hole_max_size": 10.0,
        "hole_keep_ratio": 0.08,
        "num_stripes": 2,
        "stripe_width_min": 1.2,
        "stripe_width_max": 3.0,
        "stripe_keep_ratio": 0.10,
        "add_z_noise": True,
        "normal_z_noise_sigma": 0.020,
        "hard_z_noise_sigma": 0.100,
        "min_keep_points": 2048,
    },
}

TUNABLE_KEYS = tuple(next(iter(SEVERITY_PRESETS.values())).keys())


def stable_seed(base_seed, name):
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    return (base_seed + int(digest[:8], 16)) % (2**32 - 1)


def validate_ratio(name, value):
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}.")


def resolve_settings(args):
    settings = dict(SEVERITY_PRESETS[args.severity])
    for key in TUNABLE_KEYS:
        value = getattr(args, key)
        if value is not None:
            settings[key] = value

    for key in [
        "base_keep_ratio",
        "low_density_keep_ratio",
        "roughness_keep_ratio",
        "boundary_keep_ratio",
        "hard_keep_ratio",
        "hole_keep_ratio",
        "stripe_keep_ratio",
    ]:
        validate_ratio(key, float(settings[key]))

    if settings["hole_min_size"] > settings["hole_max_size"]:
        raise ValueError("hole_min_size cannot exceed hole_max_size.")
    if settings["stripe_width_min"] > settings["stripe_width_max"]:
        raise ValueError("stripe_width_min cannot exceed stripe_width_max.")
    if settings["num_holes"] < 0 or settings["num_stripes"] < 0:
        raise ValueError("num_holes and num_stripes must be non-negative.")
    if settings["min_keep_points"] < 1:
        raise ValueError("min_keep_points must be positive.")
    return settings


def load_geometry_masks(map_fp):
    with np.load(map_fp, allow_pickle=True) as data:
        required = [
            "valid_mask",
            "low_density_mask",
            "high_gradient_mask",
            "high_roughness_mask",
            "structural_hard_mask",
            "resolution",
        ]
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"{map_fp} lacks geometry fields: {missing}")

        return {
            "valid": data["valid_mask"].astype(bool),
            "low_density": data["low_density_mask"].astype(bool),
            "high_gradient": data["high_gradient_mask"].astype(bool),
            "high_roughness": data["high_roughness_mask"].astype(bool),
            "hard": data["structural_hard_mask"].astype(bool),
            "resolution": float(data["resolution"]),
        }


def points_to_cells(points, x0, y0, resolution, height, width):
    ix = np.floor((points[:, 0] - x0) / resolution).astype(np.int32)
    iy = np.floor((points[:, 1] - y0) / resolution).astype(np.int32)
    ix = np.clip(ix, 0, width - 1)
    iy = np.clip(iy, 0, height - 1)
    return iy, ix


def rectangular_dropout_mask(
    points,
    x0,
    y0,
    block_size,
    rng,
    num_holes,
    min_size,
    max_size,
):
    mask = np.zeros(len(points), dtype=bool)
    for _ in range(num_holes):
        width = rng.uniform(min_size, max_size)
        height = rng.uniform(min_size, max_size)
        cx = rng.uniform(x0, x0 + block_size)
        cy = rng.uniform(y0, y0 + block_size)
        mask |= (np.abs(points[:, 0] - cx) <= width / 2.0) & (
            np.abs(points[:, 1] - cy) <= height / 2.0
        )
    return mask


def stripe_dropout_mask(
    points,
    x0,
    y0,
    block_size,
    rng,
    num_stripes,
    min_width,
    max_width,
):
    mask = np.zeros(len(points), dtype=bool)
    for _ in range(num_stripes):
        width = rng.uniform(min_width, max_width)
        if rng.random() < 0.5:
            center = rng.uniform(x0, x0 + block_size)
            mask |= np.abs(points[:, 0] - center) <= width / 2.0
        else:
            center = rng.uniform(y0, y0 + block_size)
            mask |= np.abs(points[:, 1] - center) <= width / 2.0
    return mask


def recompute_points_local(points, x0, y0, block_size):
    local = points.copy().astype(np.float32)
    local[:, 0] -= x0 + block_size / 2.0
    local[:, 1] -= y0 + block_size / 2.0
    local[:, 2] -= np.mean(local[:, 2])
    return local


def ensure_minimum_points(keep_mask, min_keep_points, rng):
    target = min(int(min_keep_points), len(keep_mask))
    missing = target - int(keep_mask.sum())
    if missing <= 0:
        return
    candidates = np.flatnonzero(~keep_mask)
    chosen = rng.choice(
        candidates,
        size=missing,
        replace=False,
    )
    keep_mask[chosen] = True


def degrade_one_block(
    block_fp,
    map_fp,
    out_fp,
    settings,
    seed,
    version,
    severity,
):
    metadata_keys = [
        "tile",
        "split",
        "block_x",
        "block_y",
        "block_size",
        "x0",
        "x1",
        "y0",
        "y1",
        "tile_origin_x",
        "tile_origin_y",
    ]
    with np.load(block_fp, allow_pickle=True) as clean:
        points = clean["points"].astype(np.float32)
        intensity = clean["intensity"].astype(np.float32)
        metadata = {key: clean[key] for key in metadata_keys}
    geometry = load_geometry_masks(map_fp)

    block_size = float(metadata["block_size"])
    x0 = float(metadata["x0"])
    y0 = float(metadata["y0"])

    height, width = geometry["valid"].shape
    iy, ix = points_to_cells(
        points,
        x0,
        y0,
        geometry["resolution"],
        height,
        width,
    )
    low = geometry["low_density"][iy, ix]
    gradient = geometry["high_gradient"][iy, ix]
    rough = geometry["high_roughness"][iy, ix]
    hard = geometry["hard"][iy, ix]

    rng = np.random.default_rng(stable_seed(seed, block_fp.name))
    keep_probability = np.full(
        len(points),
        settings["base_keep_ratio"],
        dtype=np.float32,
    )
    for mask, ratio in [
        (low, settings["low_density_keep_ratio"]),
        (rough, settings["roughness_keep_ratio"]),
        (gradient, settings["boundary_keep_ratio"]),
        (hard, settings["hard_keep_ratio"]),
    ]:
        keep_probability[mask] = np.minimum(
            keep_probability[mask],
            ratio,
        )

    rectangle = rectangular_dropout_mask(
        points,
        x0,
        y0,
        block_size,
        rng,
        settings["num_holes"],
        settings["hole_min_size"],
        settings["hole_max_size"],
    )
    stripe = stripe_dropout_mask(
        points,
        x0,
        y0,
        block_size,
        rng,
        settings["num_stripes"],
        settings["stripe_width_min"],
        settings["stripe_width_max"],
    )
    keep_probability[rectangle] = np.minimum(
        keep_probability[rectangle],
        settings["hole_keep_ratio"],
    )
    keep_probability[stripe] = np.minimum(
        keep_probability[stripe],
        settings["stripe_keep_ratio"],
    )


    keep_mask = rng.random(len(points)) < keep_probability
    ensure_minimum_points(
        keep_mask,
        settings["min_keep_points"],
        rng,
    )
    kept_indices = np.flatnonzero(keep_mask)

    points_degraded = points[kept_indices].copy()
    hard_kept = hard[kept_indices]
    if settings["add_z_noise"]:
        sigma = np.full(
            len(kept_indices),
            settings["normal_z_noise_sigma"],
            dtype=np.float32,
        )
        sigma[hard_kept] = settings["hard_z_noise_sigma"]
        z_noise = rng.normal(0.0, sigma).astype(np.float32)
        points_degraded[:, 2] += z_noise
    else:
        z_noise = np.zeros(
            len(kept_indices),
            dtype=np.float32,
        )

    intensity_degraded = intensity[kept_indices]



    with np.load(block_fp, allow_pickle=True) as clean:
        labels_degraded = clean["labels"][kept_indices].astype(np.int64)
        sem_class_degraded = clean["sem_class"][kept_indices].astype(np.int64)
        ins_class_degraded = clean["ins_class"][kept_indices].astype(np.int64)
    point_spatial_dropout = (rectangle | stripe)[kept_indices]
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_fp,
        points=points_degraded.astype(np.float32),
        points_local=recompute_points_local(
            points_degraded,
            x0,
            y0,
            block_size,
        ),
        features=(intensity_degraded / 65535.0).astype(np.float32)[:, None],
        intensity=intensity_degraded.astype(np.float32),
        labels=labels_degraded,
        sem_class=sem_class_degraded,
        ins_class=ins_class_degraded,
        kept_indices=kept_indices.astype(np.int64),
        keep_mask=keep_mask.astype(np.bool_),
        point_hard_mask=hard_kept.astype(np.bool_),
        point_low_density_mask=low[kept_indices].astype(np.bool_),
        point_boundary_mask=gradient[kept_indices].astype(np.bool_),
        point_roughness_mask=rough[kept_indices].astype(np.bool_),
        point_spatial_dropout_mask=point_spatial_dropout.astype(np.bool_),


        point_building_tree_mixed_mask=np.zeros(
            len(kept_indices),
            dtype=np.bool_,
        ),
        z_noise=z_noise,
        tile=metadata["tile"],
        split=metadata["split"],
        block_x=metadata["block_x"],
        block_y=metadata["block_y"],
        block_size=metadata["block_size"],
        x0=metadata["x0"],
        x1=metadata["x1"],
        y0=metadata["y0"],
        y1=metadata["y1"],
        tile_origin_x=metadata["tile_origin_x"],
        tile_origin_y=metadata["tile_origin_y"],
        source_clean_block=np.array(block_fp.name),
        source_clean_map=np.array(map_fp.name),
        degradation_version=np.array(version),
        degradation_severity=np.array(severity),
        semantic_labels_used_for_degradation=np.array(False),
    )

    class_counts = np.bincount(
        labels_degraded,
        minlength=len(CLASS_NAMES),
    )
    row = {
        "split": str(metadata["split"]),
        "tile": str(metadata["tile"]),
        "clean_block": block_fp.name,
        "clean_map": map_fp.name,
        "degraded_block": out_fp.name,
        "severity": severity,
        "semantic_labels_used_for_degradation": False,
        "num_clean_points": len(points),
        "num_degraded_points": len(points_degraded),
        "keep_ratio": (len(points_degraded) / max(len(points), 1)),
        "hard_point_ratio_clean": float(hard.mean()),
        "rect_point_ratio_clean": float(rectangle.mean()),
        "stripe_point_ratio_clean": float(stripe.mean()),
        "spatial_dropout_ratio_clean": float((rectangle | stripe).mean()),
        "num_hard_points_degraded": int(hard_kept.sum()),
        "z_noise_abs_mean": float(np.mean(np.abs(z_noise))),
    }

    for class_id, class_name in CLASS_NAMES.items():
        row[f"class_{class_id}_{class_name}"] = int(class_counts[class_id])
    return row


def add_tunable_arguments(parser):
    for key in [
        "base_keep_ratio",
        "low_density_keep_ratio",
        "roughness_keep_ratio",
        "boundary_keep_ratio",
        "hard_keep_ratio",
        "hole_min_size",
        "hole_max_size",
        "hole_keep_ratio",
        "stripe_width_min",
        "stripe_width_max",
        "stripe_keep_ratio",
        "normal_z_noise_sigma",
        "hard_z_noise_sigma",
    ]:
        parser.add_argument(f"--{key}", type=float, default=None)
    for key in [
        "num_holes",
        "num_stripes",
        "min_keep_points",
    ]:
        parser.add_argument(f"--{key}", type=int, default=None)

    noise_group = parser.add_mutually_exclusive_group()
    noise_group.add_argument(
        "--add_z_noise",
        dest="add_z_noise",
        action="store_true",
    )
    noise_group.add_argument(
        "--no_z_noise",
        dest="add_z_noise",
        action="store_false",
    )
    parser.set_defaults(add_z_noise=None)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create label-agnostic DALES degradations at a declared "
            "severity. Semantic labels are copied only after the "
            "geometry-only degradation decision."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data/processed/dales_objects/full_blocks_b50"),
    )
    parser.add_argument(
        "--map_dir",
        type=Path,
        default=Path("data/processed/dales_objects/maps_25d_b50_r050"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--severity",
        choices=sorted(SEVERITY_PRESETS),
        required=True,
    )
    parser.add_argument(
        "--version_prefix",
        default="v3",
        help=(
            "Output tag becomes deg_<prefix>_<severity>; the default "
            "medium files therefore end in _deg_v3_medium.npz."
        ),
    )
    parser.add_argument(
        "--splits",
        default="train,val,test",
    )
    parser.add_argument(
        "--max_files_per_split",
        type=int,
        default=None,
    )
    parser.add_argument("--seed", type=int, default=123)
    add_tunable_arguments(parser)
    args = parser.parse_args()

    settings = resolve_settings(args)
    version = f"{args.version_prefix}_{args.severity}"
    degradation_tag = f"deg_{version}"
    if args.out_dir is None:
        out_dir = Path("data/processed/dales_objects/" f"degraded_blocks_b50_{version}")
    else:
        out_dir = args.out_dir

    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {out_dir}. "
            "Use a new directory to preserve provenance."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    invalid_splits = set(splits) - {"train", "val", "test"}
    if not splits or invalid_splits:
        raise ValueError(f"Invalid --splits value: {args.splits!r}")

    config = {
        "protocol": "label_agnostic_degradation_v3",
        "input_dir": str(args.input_dir),
        "map_dir": str(args.map_dir),
        "out_dir": str(out_dir),
        "severity": args.severity,
        "version": version,
        "degradation_tag": degradation_tag,
        "splits": splits,
        "seed": args.seed,
        "max_files_per_split": args.max_files_per_split,
        "semantic_labels_used_for_degradation": False,
        "geometry_inputs": [
            "low_density_mask",
            "high_gradient_mask",
            "high_roughness_mask",
            "structural_hard_mask",
            "seeded_rectangular_dropout",
            "seeded_stripe_dropout",
        ],
        "settings": settings,
    }
    with open(
        out_dir / "degradation_config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(config, f, indent=2)

    rows = []
    for split in splits:
        block_files = sorted((args.input_dir / split).glob("*.npz"))
        if args.max_files_per_split is not None:
            block_files = block_files[: args.max_files_per_split]
        if not block_files:
            raise RuntimeError(f"No clean blocks found for split {split}.")

        print(
            f"[PROCESS] severity={args.severity} "
            f"split={split} files={len(block_files)}"
        )
        for block_fp in tqdm(
            block_files,
            desc=f"{args.severity}:{split}",
        ):
            map_fp = args.map_dir / split / f"{block_fp.stem}_25d.npz"
            if not map_fp.exists():
                raise FileNotFoundError(map_fp)
            out_fp = out_dir / split / f"{block_fp.stem}_{degradation_tag}.npz"
            rows.append(
                degrade_one_block(
                    block_fp,
                    map_fp,
                    out_fp,
                    settings,
                    args.seed,
                    version,
                    args.severity,
                )
            )
        pd.DataFrame(rows).to_csv(
            out_dir / "degradation_metadata_partial.csv",
            index=False,
        )

    metadata = pd.DataFrame(rows)
    metadata.to_csv(
        out_dir / "degradation_metadata.csv",
        index=False,
    )
    summary = {
        "num_blocks_by_split": (
            metadata.groupby("split")["degraded_block"].count().to_dict()
        ),
        "mean_keep_ratio_by_split": (
            metadata.groupby("split")["keep_ratio"].mean().to_dict()
        ),
        "semantic_labels_used_for_degradation": False,
    }
    with open(
        out_dir / "degradation_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    print("[SAVED]", out_dir)
    print("[TAG]", degradation_tag)
    print("[KEEP RATIOS]", summary["mean_keep_ratio_by_split"])


if __name__ == "__main__":
    main()
