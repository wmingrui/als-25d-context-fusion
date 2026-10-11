# -*- coding: utf-8 -*-
"""Summarize retained-point observation support in DALES test blocks."""













import argparse, json, sys
from pathlib import Path
import numpy as np

BLOCK_DIR = Path("/root/autodl-tmp/als-transfer/data/degraded_blocks_b50_v3_medium/test")
OUT = Path(__file__).resolve().parent / "f44_observation_scale.json"


def probe():
    files = sorted(BLOCK_DIR.glob("*.npz"))
    if not files:
        print("no npz in", BLOCK_DIR)
        return None
    f = files[0]
    with np.load(f, allow_pickle=True) as z:
        print("SAMPLE:", f.name)
        for k in z.files:
            a = z[k]
            try:
                print("  %-20s shape=%s dtype=%s" % (k, np.shape(a), a.dtype))
            except Exception:
                print("  %-20s <scalar>" % k)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200, help="max blocks to scan")
    a = ap.parse_args()

    files = probe()
    if files is None:
        sys.exit(1)
    print("\nTOTAL BLOCKS:", len(files))

    rows = []
    for f in files[: a.limit]:
        with np.load(f, allow_pickle=True) as z:
            keys = set(z.files)
            if "labels" not in keys:
                print("  skip (no labels):", f.name)
                continue
            lab = z["labels"]
            n_retained = int(len(lab))
            rec = dict(block=f.stem, retained=n_retained)



            if "keep_mask" in keys:
                km = np.asarray(z["keep_mask"])
                rec["original"] = int(km.size)
                rec["kept_true"] = int(km.sum())
            if "point_hard_mask" in keys:
                ph = np.asarray(z["point_hard_mask"])
                rec["hard_points"] = int(ph.sum())
            rows.append(rec)

    if not rows:
        print("no usable rows")
        sys.exit(1)

    retained = np.array([r["retained"] for r in rows], dtype=np.float64)
    summary = dict(
        blocks_scanned=len(rows),
        retained_points=dict(
            total=int(retained.sum()),
            mean=float(retained.mean()),
            sd=float(retained.std(ddof=1)) if len(retained) > 1 else 0.0,
            min=int(retained.min()),
            max=int(retained.max()),
        ),
        fields_present=sorted({k for r in rows for k in r if k != "block"}),
    )


    has_orig = [r for r in rows if "original" in r and r["original"] > 0]
    if has_orig:
        orig = np.array([r["original"] for r in has_orig], dtype=np.float64)
        kept = np.array([r["retained"] for r in has_orig], dtype=np.float64)
        ratio = kept / orig
        summary["retention"] = dict(
            blocks=len(has_orig),
            original_total=int(orig.sum()),
            retained_total=int(kept.sum()),
            pooled_ratio=float(kept.sum() / orig.sum()),
            block_ratio_mean=float(ratio.mean()),
            block_ratio_sd=float(ratio.std(ddof=1)) if len(ratio) > 1 else 0.0,
            block_ratio_min=float(ratio.min()),
            block_ratio_max=float(ratio.max()),
        )

    has_hard = [r for r in rows if "hard_points" in r]
    if has_hard:
        h = np.array([r["hard_points"] for r in has_hard], dtype=np.float64)
        k = np.array([r["retained"] for r in has_hard], dtype=np.float64)
        summary["hard_fraction"] = dict(
            blocks=len(has_hard),
            pooled=float(h.sum() / k.sum()),
            mean=float((h / np.maximum(k, 1)).mean()),
        )

    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSUMMARY:")
    print(json.dumps(summary, indent=2))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
