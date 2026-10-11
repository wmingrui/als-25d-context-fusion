# -*- coding: utf-8 -*-
"""Rescore fixed predictions under full and sampled scoring populations."""














































import argparse, hashlib, json, sys, time, io
from pathlib import Path
import numpy as np


Q        = Path(__file__).resolve().parent
CORE_CTRL = Q.parent / "core_controls"
FULL_DIR = Q / "full_retained"
OUT_DIR  = Q / "f21_out"
DALES_TEST_BLOCK_DIR = Path("/root/autodl-tmp/als-transfer/data/degraded_blocks_b50_v3_medium/test")
NC = {"DALES": 8, "WHU": 8, "H3D": 11}

for p in (str(Q), str(CORE_CTRL)):
    if p not in sys.path:
        sys.path.insert(0, p)

_EXT = {"loaded": False, "load_external": None, "mods": {}}

def stable_seed(base_seed, name):
    """Exact copy of the trainer's stable_seed (39_train_randlanet_...py:63)."""
    h = hashlib.md5(name.encode("utf-8")).hexdigest()
    return (base_seed + int(h[:8], 16)) % (2**32 - 1)


def write_json(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(x, indent=2), encoding="utf-8")
    tmp.replace(p)

def metrics_from_confusion(cm):
    """Supplement S2 definitions; valid class = present in ground truth (r>0)."""
    cm = np.asarray(cm, dtype=np.int64)
    t = np.diag(cm).astype(np.float64)
    r = cm.sum(1).astype(np.float64)
    c = cm.sum(0).astype(np.float64)
    valid = r > 0
    iou = np.where(valid, t / np.maximum(r + c - t, 1e-12), 0.0)
    rec = np.where(valid, t / np.maximum(r, 1e-12), 0.0)
    f1  = np.where(valid, 2 * t / np.maximum(r + c, 1e-12), 0.0)
    return {"mIoU": 100.0 * float(iou[valid].mean()),
            "OA":   100.0 * float(t.sum() / max(cm.sum(), 1)),
            "mAcc": 100.0 * float(rec[valid].mean()),
            "mF1":  100.0 * float(f1[valid].mean())}

def confusion(labels, pred, nc):
    assert labels.min() >= 0 and labels.max() < nc, "label out of range"
    assert pred.min() >= 0 and pred.max() < nc, "prediction out of range"
    return np.bincount((labels.astype(np.int64) * nc + pred.astype(np.int64)).reshape(-1),
                       minlength=nc * nc).reshape(nc, nc)


def _lazy_load_external(dataset):
    if not _EXT["loaded"]:
        from external_controls import load_external
        _EXT["load_external"] = load_external
        _EXT["loaded"] = True
    if dataset not in _EXT["mods"]:
        _EXT["mods"][dataset] = _EXT["load_external"](dataset)
    return _EXT["mods"][dataset]

class ExternalLabels:
    """Lazy per-block labels + exact file names for WHU/H3D test blocks.

    Mirrors full_evaluation.setup: ds = original_cls('test', 42, method, 'medium');
    block file paths are ds.files; labels come from module.points(path)['labels'].
    Only one ds instance per dataset is kept (files are method-independent).
    """
    def __init__(self, dataset, method="observation"):
        m, original, _stats = _lazy_load_external(dataset)
        self.module = m
        self.ds = original("test", 42, method, "medium")
        self.stem_to_name = {Path(p).stem: Path(p).name for p in self.ds.files}
        self.stem_to_path = {Path(p).stem: p for p in self.ds.files}

    def name_of(self, stem):
        return self.stem_to_name[stem]

    def labels_of(self, stem):
        x = self.module.points(self.stem_to_path[stem])
        return np.asarray(dict(x)["labels"], dtype=np.int64)


def list_jobs():
    """38 full-retained jobs. Prefer full_evaluation.all_jobs(); fall back to
    scanning full_retained/ for completed results.json."""
    try:
        from full_evaluation import all_jobs
        jobs = [dict(id=j["id"], dataset=j["dataset"], variant=j["variant"],
                     method=j.get("method")) for j in all_jobs()]
        if jobs:
            return jobs
    except Exception as e:
        print("[warn] full_evaluation.all_jobs unavailable (%r); scanning %s" % (e, FULL_DIR))
    out = []
    for d in sorted(FULL_DIR.iterdir()):
        rj = d / "results.json"
        if rj.exists():
            r = json.loads(rj.read_text(encoding="utf-8"))
            j = r.get("job", {})
            out.append(dict(id=d.name, dataset=j.get("dataset", "?"),
                            variant=j.get("variant", "?"), method=j.get("method")))
    return out


def run_job(job, boot, seed0, tol_pp, ext_cache):
    jid, dataset = job["id"], job["dataset"]
    if dataset not in NC:
        return None, "unknown dataset %r" % dataset
    nc = NC[dataset]
    jdir = FULL_DIR / jid
    rj = jdir / "results.json"
    published = json.loads(rj.read_text(encoding="utf-8")) if rj.exists() else None
    blocks = sorted(p for p in jdir.iterdir() if p.suffix == ".npz")
    if not blocks:
        return None, "no block npz files under %s" % jdir

    ext = None
    if dataset != "DALES":
        if ext_cache is not None:
            if dataset not in ext_cache:
                ext_cache[dataset] = ExternalLabels(dataset, method=job.get("method") or "observation")
            ext = ext_cache[dataset]
        else:
            ext = ExternalLabels(dataset, method=job.get("method") or "observation")

    conf_all  = np.zeros((nc, nc), np.int64)
    conf_g0   = np.zeros((nc, nc), np.int64)
    conf_boot = [np.zeros((nc, nc), np.int64) for _ in range(boot)]
    t0, nblocks = time.time(), len(blocks)
    for bi, bp in enumerate(blocks):
        with np.load(bp, allow_pickle=True) as z:
            pred = z["prediction"].astype(np.int64)
            stored_cm = np.asarray(z["confusion"], dtype=np.int64)
        stem = bp.stem
        if dataset == "DALES":
            name = stem + ".npz"
            with np.load(DALES_TEST_BLOCK_DIR / name, allow_pickle=True) as z:
                labels = z["labels"].astype(np.int64)
        else:
            name = ext.name_of(stem)
            labels = ext.labels_of(stem)
        n = len(labels)
        assert len(pred) == n, "%s: prediction len %d != labels len %d" % (stem, len(pred), n)

        cm_chk = confusion(labels, pred, nc)
        if not np.array_equal(cm_chk, stored_cm):
            return None, ("%s: recomputed confusion != stored confusion "
                          "(label order assumption broken)" % stem)

        order = np.random.default_rng(stable_seed(42, name)).permutation(n)
        g0 = order[: min(8192, n)]

        conf_all += cm_chk
        conf_g0  += confusion(labels[g0], pred[g0], nc)
        for k in range(boot):
            rng = np.random.default_rng(stable_seed(seed0, "%s__f21boot%d" % (name, k)))
            sub = rng.choice(n, size=min(8192, n), replace=False)
            conf_boot[k] += confusion(labels[sub], pred[sub], nc)

        if (bi + 1) % 100 == 0:
            print("  %s: %d/%d blocks (%.0fs)" % (jid, bi + 1, nblocks, time.time() - t0), flush=True)

    m_all   = metrics_from_confusion(conf_all)
    m_g0    = metrics_from_confusion(conf_g0)
    m_boot  = [metrics_from_confusion(cb) for cb in conf_boot]
    boot_stats = {k: dict(mean=float(np.mean([m[k] for m in m_boot])),
                          sd=float(np.std([m[k] for m in m_boot], ddof=1)),
                          min=float(np.min([m[k] for m in m_boot])),
                          max=float(np.max([m[k] for m in m_boot])))
                  for k in ("mIoU", "OA", "mAcc", "mF1")}
    pub_check = None
    if published and "metrics" in published:
        pm = published["metrics"]



        keymap = {"OA": "oa", "mIoU": "mIoU", "mAcc": "mAcc"}
        diffs = {}
        for ours, theirs in keymap.items():
            if theirs in pm:
                diffs[ours] = abs(m_all[ours] - 100.0 * float(pm[theirs]))
        pub_check = dict(matches=bool(max(diffs.values()) <= tol_pp) if diffs else False,
                         max_abs_diff_pp=round(max(diffs.values()), 4) if diffs else None,
                         compared=sorted(diffs),
                         skipped=[k for k in m_all if k not in keymap])

    return dict(
        job_id=jid, dataset=dataset, variant=job["variant"], nc=nc,
        blocks=nblocks, points=int(conf_all.sum()), boot_subsets=boot, seed0=seed0,
        A_full_all_points={k: round(v, 4) for k, v in m_all.items()},
        B_group0_first_permutation={k: round(v, 4) for k, v in m_g0.items()},
        C_bootstrap_mean_sd={k: {kk: round(vv, 4) for kk, vv in d.items()} for k, d in boot_stats.items()},
        coverage_effect_pp={k: round(m_all[k] - boot_stats[k]["mean"], 4) for k in m_all},
        published_full_metric=(published or {}).get("metrics"),
        published_metric_check=pub_check,
        seconds=round(time.time() - t0, 1),
    ), None


def main():
    ap = argparse.ArgumentParser(description="F21 coverage isolation (no training, no GPU)")
    ap.add_argument("--jobs", nargs="*", help="job id substrings to keep")
    ap.add_argument("--boot", type=int, default=16)
    ap.add_argument("--seed0", type=int, default=900)
    ap.add_argument("--tol-pp", type=float, default=0.02)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = list_jobs()
    if args.jobs:
        jobs = [j for j in jobs if any(f in j["id"] for f in args.jobs)]
    print("%d jobs selected" % len(jobs))
    if args.dry_run:
        for j in jobs:
            print("  %-42s %-6s %-12s method=%s" % (j["id"], j["dataset"], j["variant"], j.get("method")))
        return

    OUT_DIR.mkdir(exist_ok=True)
    results, errors, ext_cache = [], [], {}
    for job in jobs:
        print("== %s" % job["id"], flush=True)
        try:
            res, err = run_job(job, args.boot, args.seed0, args.tol_pp, ext_cache)
        except Exception as e:
            res, err = None, repr(e)
        if res is None:
            errors.append(dict(job=job["id"], error=err))
            print("   ERROR: %s" % err, flush=True)
        else:
            results.append(res)
            pc = res["published_metric_check"]
            c = res["C_bootstrap_mean_sd"]["mIoU"]
            print("   A mIoU=%.3f | B mIoU=%.3f±%.3f | coverage Δ=%+.3f pp | pub_check=%s"
                  % (res["A_full_all_points"]["mIoU"], c["mean"], c["sd"],
                     res["coverage_effect_pp"]["mIoU"],
                     ("OK" if pc["matches"] else "MISMATCH %+.4f" % pc["max_abs_diff_pp"]) if pc else "n/a"),
                  flush=True)

    write_json(OUT_DIR / "f21_coverage_results.json",
               dict(created=time.time(), boot=args.boot, seed0=args.seed0,
                    results=results, errors=errors))

    buf = io.StringIO()
    buf.write("# F21 coverage isolation — fixed predictions, varying scoring population\n\n")
    buf.write("A = full-retained predictions on all retained points (published protocol, "
              "cross-checked against results.json). B = the same predictions scored on "
              "%d random 8,192-record subsets per block (mean ± SD across subsets). "
              "A − B = pure scoring-coverage effect. The grouping/prediction effect is "
              "(published fixed-sampled metric) − B, to be compared against manuscript "
              "Table 3 / supplement Tables S5–S10.\n\n" % args.boot)
    buf.write("| job | ds | variant | mIoU A | mIoU B | cov Δ pp | OA A | OA B | pub |\n")
    buf.write("|---|---|---|---|---|---|---|---|---|\n")
    for r in results:
        pc = r["published_metric_check"]
        c = r["C_bootstrap_mean_sd"]["mIoU"]
        buf.write("| %s | %s | %s | %.2f | %.2f±%.2f | %+.2f | %.2f | %.2f | %s |\n" % (
            r["job_id"], r["dataset"], r["variant"], r["A_full_all_points"]["mIoU"],
            c["mean"], c["sd"], r["coverage_effect_pp"]["mIoU"],
            r["A_full_all_points"]["OA"], c and r["C_bootstrap_mean_sd"]["OA"]["mean"],
            ("OK" if pc["matches"] else "MISMATCH") if pc else "n/a"))
    if errors:
        buf.write("\n## Errors\n\n")
        for e in errors:
            buf.write("- %s: %s\n" % (e["job"], e["error"]))
    (OUT_DIR / "f21_coverage_report.md").write_text(buf.getvalue(), encoding="utf-8")
    print("\nwrote f21_out/f21_coverage_results.json + f21_coverage_report.md "
          "(%d jobs, %d errors)" % (len(results), len(errors)))

if __name__ == "__main__":
    main()
