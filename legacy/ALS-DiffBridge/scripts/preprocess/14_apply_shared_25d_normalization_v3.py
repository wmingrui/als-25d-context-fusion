from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from tqdm import tqdm


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


def canonical_hash(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_rng(seed, name):
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    derived = (seed + int(digest[:8], 16)) % (2**32 - 1)
    return np.random.default_rng(derived)


def sample_values(values, max_values, rng):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if len(values) <= max_values:
        return values
    indices = rng.choice(
        len(values),
        size=max_values,
        replace=False,
    )
    return values[indices]


def fit_shared_stats(
    fit_dir,
    split,
    max_values_per_map,
    seed,
    z_low_q,
    z_high_q,
    scale_q,
):
    files = sorted((fit_dir / split).glob("*_25d.npz"))
    if not files:
        raise RuntimeError(f"No fit maps found in {fit_dir / split}.")

    samples = {
        "height": [],
        "height_std": [],
        "density": [],
        "height_gradient": [],
    }
    for path in tqdm(files, desc="fit shared normalization"):
        with np.load(path, allow_pickle=True) as data:
            maps_raw = data["maps_raw"].astype(np.float32)
            valid = data["valid_mask"].astype(bool)
        rng = stable_rng(seed, path.name)
        height_values = np.concatenate([maps_raw[index][valid] for index in [0, 1, 2]])
        samples["height"].append(
            sample_values(
                height_values,
                max_values_per_map * 3,
                rng,
            )
        )
        for name, index in [
            ("height_std", 3),
            ("density", 4),
            ("height_gradient", 7),
        ]:
            samples[name].append(
                sample_values(
                    maps_raw[index][valid],
                    max_values_per_map,
                    rng,
                )
            )

    merged = {key: np.concatenate(value) for key, value in samples.items()}
    stats = {
        "protocol": "shared_25d_normalization_v3",
        "fit_dir": str(fit_dir.resolve()),
        "fit_split": split,
        "num_fit_maps": len(files),
        "semantic_labels_used": False,
        "sampling_seed": seed,
        "max_values_per_map": max_values_per_map,
        "quantiles": {
            "height_low": z_low_q,
            "height_high": z_high_q,
            "positive_scale": scale_q,
        },
        "z_min": float(np.percentile(merged["height"], z_low_q)),
        "z_max": float(np.percentile(merged["height"], z_high_q)),
        "rough_scale": float(np.percentile(merged["height_std"], scale_q)),
        "density_scale": float(np.percentile(merged["density"], scale_q)),
        "gradient_scale": float(
            np.percentile(
                merged["height_gradient"],
                scale_q,
            )
        ),
    }
    stats["z_max"] = max(
        stats["z_max"],
        stats["z_min"] + 1e-3,
    )
    stats["rough_scale"] = max(
        stats["rough_scale"],
        1e-6,
    )
    stats["density_scale"] = max(
        stats["density_scale"],
        1.0,
    )
    stats["gradient_scale"] = max(
        stats["gradient_scale"],
        1e-6,
    )
    stats["stats_sha256"] = canonical_hash(stats)
    return stats


def validate_stats(stats):
    recorded = stats.get("stats_sha256")
    unsigned = dict(stats)
    unsigned.pop("stats_sha256", None)
    if recorded != canonical_hash(unsigned):
        raise RuntimeError("Shared normalization statistics hash mismatch.")
    if stats.get("fit_split") != "train":
        raise RuntimeError("Shared normalization must be fitted on train only.")
    if stats.get("semantic_labels_used") is not False:
        raise RuntimeError("Shared normalization stats are not label-agnostic.")


def normalize_raw_maps(maps_raw, stats):
    z_range = stats["z_max"] - stats["z_min"]
    height = [
        np.clip(
            (maps_raw[index] - stats["z_min"]) / z_range,
            0.0,
            1.0,
        )
        for index in [0, 1, 2]
    ]
    height_std = np.clip(
        maps_raw[3] / stats["rough_scale"],
        0.0,
        1.0,
    )
    density = np.clip(
        maps_raw[4] / stats["density_scale"],
        0.0,
        1.0,
    )
    log_scale = np.log1p(stats["density_scale"])
    log_density = np.clip(
        maps_raw[5] / max(log_scale, 1e-6),
        0.0,
        1.0,
    )
    intensity = np.clip(maps_raw[6], 0.0, 1.0)
    height_gradient = np.clip(
        maps_raw[7] / stats["gradient_scale"],
        0.0,
        1.0,
    )
    boundary = height_gradient.copy()
    return np.stack(
        [
            *height,
            height_std,
            density,
            log_density,
            intensity,
            height_gradient,
            boundary,
        ],
        axis=0,
    ).astype(np.float32)


def apply_one(input_path, output_path, stats):
    with np.load(input_path, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}
    maps_raw = payload["maps_raw"].astype(np.float32)
    maps_norm = normalize_raw_maps(maps_raw, stats)



    maps_raw = maps_raw.copy()
    maps_raw[8] = maps_norm[8]
    payload.update(
        {
            "maps_raw": maps_raw,
            "maps_norm": maps_norm,
            "raw_channel_names": np.array(RAW_CHANNELS),
            "norm_channel_names": np.array(NORM_CHANNELS),
            "z_min": np.float32(stats["z_min"]),
            "z_max": np.float32(stats["z_max"]),
            "rough_p95": np.float32(stats["rough_scale"]),
            "density_p95": np.float32(stats["density_scale"]),
            "grad_p95": np.float32(stats["gradient_scale"]),
            "normalization_protocol": np.array(stats["protocol"]),
            "normalization_stats_sha256": np.array(stats["stats_sha256"]),
            "normalization_fit_split": np.array(stats["fit_split"]),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return {
        "split": str(payload.get("split", "")),
        "input_map": input_path.name,
        "output_map": output_path.name,
        "normalization_stats_sha256": stats["stats_sha256"],
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fit train-only shared 2.5D normalization statistics and/or "
            "apply them to clean or degraded map directories."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--stats_json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--fit",
        action="store_true",
        help=(
            "Fit stats from --fit_dir/--fit_split before applying. "
            "Without this flag an existing --stats_json is required."
        ),
    )
    parser.add_argument(
        "--fit_dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--fit_split",
        default="train",
        choices=["train"],
    )
    parser.add_argument(
        "--splits",
        default="train,val,test",
    )
    parser.add_argument(
        "--max_values_per_map",
        type=int,
        default=2048,
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--z_low_q", type=float, default=1.0)
    parser.add_argument("--z_high_q", type=float, default=99.0)
    parser.add_argument("--scale_q", type=float, default=99.0)
    parser.add_argument(
        "--max_files_per_split",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    out_dir = args.out_dir.resolve()
    stats_json = args.stats_json.resolve()
    if input_dir == out_dir:
        raise ValueError("--out_dir must differ from --input_dir.")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out_dir}")

    if args.fit:
        fit_dir = args.fit_dir.resolve() if args.fit_dir is not None else input_dir
        stats = fit_shared_stats(
            fit_dir,
            args.fit_split,
            args.max_values_per_map,
            args.seed,
            args.z_low_q,
            args.z_high_q,
            args.scale_q,
        )
        if stats_json.exists():
            existing = json.loads(stats_json.read_text(encoding="utf-8"))
            if existing != stats:
                raise FileExistsError(
                    "Existing stats file differs from the newly fitted "
                    f"statistics: {stats_json}"
                )
        else:
            stats_json.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            with open(
                stats_json,
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(stats, f, indent=2)
    else:
        if not stats_json.exists():
            raise FileNotFoundError(stats_json)
        with open(stats_json, encoding="utf-8") as f:
            stats = json.load(f)
    validate_stats(stats)

    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    invalid = set(splits) - {"train", "val", "test"}
    if not splits or invalid:
        raise ValueError(f"Invalid --splits: {args.splits!r}")

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in splits:
        files = sorted((input_dir / split).glob("*_25d.npz"))
        if args.max_files_per_split is not None:
            files = files[: args.max_files_per_split]
        if not files:
            raise RuntimeError(f"No maps found in {input_dir / split}.")
        for input_path in tqdm(
            files,
            desc=f"shared norm:{split}",
        ):
            rows.append(
                apply_one(
                    input_path,
                    out_dir / split / input_path.name,
                    stats,
                )
            )

    pd.DataFrame(rows).to_csv(
        out_dir / "normalization_metadata.csv",
        index=False,
    )
    complete = {
        "status": "complete",
        "protocol": stats["protocol"],
        "stats_json": str(stats_json),
        "stats_sha256": stats["stats_sha256"],
        "input_dir": str(input_dir),
        "out_dir": str(out_dir),
        "splits": splits,
        "num_maps": len(rows),
        "semantic_labels_used": False,
    }
    with open(
        out_dir / "normalization_complete.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(complete, f, indent=2)

    print("[DONE]", out_dir)
    print("[STATS]", stats_json)
    print("[SHA256]", stats["stats_sha256"])


if __name__ == "__main__":
    main()
