# -*- coding: utf-8 -*-
"""Evaluate the additional DALES seeds using the locked full-retained protocol."""















import json, os, sys, time, hashlib
from pathlib import Path

EVAL = Path("/root/autodl-tmp/cg_evaluation_20260914_01")
sys.path.insert(0, str(EVAL))
os.environ.setdefault("F16_EXPECTED", "4")

import full_evaluation as F
from evaluation import sha, write

LOCK = EVAL / "FULL_PROTOCOL_LOCK.json"
NEW_IDS = ["train__DALES_sampled_45", "train__DALES_point_45",
           "train__DALES_sampled_46", "train__DALES_point_46"]

print("=== inventory ===", flush=True)
jobs = {j["id"]: j for j in F.all_jobs()}
print("total jobs:", len(jobs), flush=True)
missing = [i for i in NEW_IDS if i not in jobs]
assert not missing, "missing F16 jobs: %s" % missing
for i in NEW_IDS:
    j = jobs[i]
    print("  %-28s ds=%s variant=%s ckpt_exists=%s" % (i, j["dataset"], j["variant"], Path(j["checkpoint"]).exists()), flush=True)


lockdir = EVAL / ("f16_lock_backup_%s" % time.strftime("%Y%m%d_%H%M%S"))
lockdir.mkdir(exist_ok=True)
(lockdir / "FULL_PROTOCOL_LOCK.json.orig").write_bytes(LOCK.read_bytes())
print("\nlock backed up -> %s" % lockdir, flush=True)
print("  original lock sha256 =", sha(LOCK)[:16], flush=True)

lock = json.loads(LOCK.read_text())
have_ids = {j["id"] for j in lock["jobs"]}
added = 0
for i in NEW_IDS:
    if i in have_ids:
        print("  = already in lock:", i, flush=True); continue
    j = jobs[i]
    entry = dict(id=j["id"], dataset=j["dataset"], variant=j["variant"],
                 new=j["new"], checkpoint=j["checkpoint"], sha256=j["sha256"])
    lock["jobs"].append(entry)
    lock["checkpoints"][j["id"]] = j["sha256"]
    added += 1
    print("  + lock entry:", i, flush=True)
if added:
    lock["validated_adapters"] = lock.get("validated_adapters", 38) + added
    LOCK.write_text(json.dumps(lock, indent=2))
    print("  lock now has %d jobs (was %d)" % (len(lock["jobs"]), len(lock["jobs"]) - added), flush=True)


print("\n=== evaluating ===", flush=True)
results = {}
for i in NEW_IDS:
    outdir = EVAL / "full_retained" / i
    if (outdir / "results.json").exists():
        print("  = already evaluated:", i, flush=True)
        results[i] = json.loads((outdir / "results.json").read_text())["metrics"]
        continue
    t0 = time.time()
    print("  -> %s  start %s" % (i, time.strftime("%H:%M:%S")), flush=True)
    F.evaluate_full(jobs[i])
    m = json.loads((outdir / "results.json").read_text())["metrics"]
    results[i] = m
    print("  <- %s  done in %.1f min  mIoU=%.4f" % (i, (time.time() - t0) / 60, m.get("mIoU", -1)), flush=True)

print("\n=== SUMMARY (mIoU, 0-100) ===", flush=True)
for i in NEW_IDS:
    m = results[i]
    print("  %-28s mIoU=%.4f  OA=%.4f  mAcc=%.4f" % (i, m.get("mIoU", -1), m.get("OA", -1), m.get("mAcc", -1)), flush=True)

(EVAL / "f16_new_results.json").write_text(json.dumps(results, indent=2))
print("\nwrote", EVAL / "f16_new_results.json", flush=True)
