"""D3: O's input-fusion block transplanted onto the stable PointNeXt-S backbone.

Locked protocol: D1_D3_PROTOCOL_LOCK_CN.md section 2. Summary --
  * backbone / recipe : identical to the completed C-group formal PointNeXt-S runs
                        (radius 1.0 m, lr 1e-3, AdamW wd 1e-4, cosine -> 0.05*lr,
                         grad-norm 1, FP32 with TF32 off, DALES 50ep/batch2,
                         H3D 100ep/batch16, validation seed 42).
  * the ONLY change    : O's input-fusion block (RGDIFusion, embed_ch=32,
                         gate_bias=0.3, learned gate, no quality/reliability
                         descriptors) is placed at the PointNeXt stem, so
                         in_channels 4 -> 36. The observation prior is identical
                         to O's (degraded raster; restored/reliability zero).
  * no internal injection: PointNeXt SA stages differ from RandLA, so injecting
                         would be a new method rather than the same one on a new
                         backbone. Recorded as an "input-fusion adaptation".

The independent-backbone environment is reused read-only from the completed
C-group run directory, and its hashes are asserted at launch.
"""
import argparse, csv, hashlib, json, os, sys, time
from pathlib import Path

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from controls import T, ROOT, module
from d1_controls import install_dataset
from external_controls import ROOTS

Q = Path(__file__).resolve().parent
RUN02 = Path("/root/autodl-tmp/cg_upgrade_20260914_run02")
VENDOR = RUN02 / "vendor/PointNeXt-master"
sys.path.insert(0, str(VENDOR))
from openpoints.utils import EasyConfig                      # noqa: E402
from openpoints.models import build_model_from_cfg           # noqa: E402

LOCK = json.loads((Q / "D3_PROTOCOL_LOCK.json").read_text())
DALES_REFERENCE = ROOT / "PROJECT_NAV_20260909/05_new_work/results/observation_seeds43_44_20260910_01/train/s43_observation/config.json"
SMOKE_STEPS = 12
SMOKE_VAL = 4


def verify_environment():
    """Fail closed if the reused independent-backbone environment drifted."""
    for rel, expected in LOCK["reused_environment"]["hashes"].items():
        p = RUN02 / rel
        assert p.exists(), f"missing reused environment file: {p}"
        assert hashlib.file_digest(p.open("rb"), "sha256").hexdigest() == expected, f"environment drift: {rel}"


class PointNextObs(torch.nn.Module):
    """PointNeXt-S whose stem consumes O's fused observation features (4+32 ch)."""

    def __init__(self, classes, radius, embed_ch, gate_bias):
        super().__init__()
        cfg = EasyConfig(); cfg.load(str(VENDOR / "cfgs/s3dis/pointnext-s.yaml"))
        cfg.model.encoder_args.radius = radius
        cfg.model.encoder_args.in_channels = 4 + embed_ch
        cfg.model.cls_args.num_classes = classes
        cfg.model.cls_args.in_channels = None
        self.net = build_model_from_cfg(cfg.model)
        self.rgdi = T.RGDIFusion(base_ch=4, prior_ch=9, rel_ch=6, embed_ch=embed_ch,
                                 gate_bias=gate_bias, use_explicit_quality_modulation=False,
                                 use_reliability_descriptors=False, gate_mode="learned")

    def forward(self, coords, base, degraded, delta, rel):
        fused = self.rgdi(base, degraded, delta, rel)
        return self.net({"pos": coords.contiguous(),
                         "x": fused.transpose(1, 2).contiguous()}).transpose(1, 2).contiguous()


def build_datasets(name, seed):
    """Return (train, val, collate, class_weights, num_classes, batch, workers, ignore_index)."""
    spec = LOCK["datasets"][name]
    if name == "DALES":
        rc = json.loads(DALES_REFERENCE.read_text())
        cls = install_dataset()
        kw = dict(block_dir=rc["block_dir"], clean_map_dir=rc["clean_map_dir"],
                  deg_map_dir=rc["deg_map_dir"], restored_map_dir=rc["restored_map_dir"],
                  mode="restored_rgdi", num_points=8192, num_layers=4, num_neighbors=16,
                  sub_sampling_ratio=[4] * 4, degradation_tag=rc["degradation_tag"], seed=seed)
        tr, va = cls(split="train", **kw), cls(split="val", **kw)
        w = T.compute_class_weights(kw["block_dir"])
        assert w is not None, "DALES class weights unavailable"
        return tr, va, T.randla_collate, torch.tensor(w, dtype=torch.float32), 8, spec["batch"], spec["workers"], -100
    m = module("d3_h3d_train_complete", ROOTS["H3D"] / "train_complete.py")
    tr = m.Points("train", seed, "observation")
    va = m.Points("val", seed, "observation")
    counts = np.array(json.loads((ROOTS["H3D"] / "train_class_counts.json").read_text())) + 1
    w = 1 / np.sqrt(counts / counts.sum()); w = np.clip(w / w.mean(), .5, 8)
    return tr, va, m.collate, torch.tensor(w, dtype=torch.float32), 11, spec["batch"], spec["workers"], -100


def write(p, d):
    tmp = p.with_suffix(".tmp"); tmp.write_text(json.dumps(d, indent=2)); tmp.replace(p)


def obs_batch(b):
    """Pull the four observation tensors + metric coords out of a collated batch."""
    cdev = lambda k: b[k].cuda(non_blocking=True)
    return (b["coords"][0].cuda(non_blocking=True), cdev("base"), cdev("degraded"),
            cdev("delta"), cdev("rel"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["DALES", "H3D"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--job", required=True)
    a = p.parse_args()
    assert "/" not in a.job and ".." not in a.job

    verify_environment()
    spec = LOCK["datasets"][a.dataset]
    out = Q / ("smoke" if a.smoke else "d3_runs") / a.job
    out.mkdir(parents=True, exist_ok=False)

    T.seed_everything(a.seed)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    ds, vs, collate, weights, nc, batch, workers, ignore = build_datasets(a.dataset, a.seed)
    epochs = 1 if a.smoke else spec["epochs"]


    max_train = SMOKE_STEPS if a.smoke else None
    max_val = SMOKE_VAL if a.smoke else None
    gen = torch.Generator().manual_seed(a.seed)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, generator=gen, num_workers=workers,
                    persistent_workers=workers > 0, pin_memory=True,
                    drop_last=(a.dataset == "DALES"), collate_fn=collate)
    vl = DataLoader(vs, batch_size=batch, num_workers=2, persistent_workers=True,
                    pin_memory=True, collate_fn=collate)

    m = PointNextObs(nc, spec["radius_metres"], spec["embed_ch"], spec["gate_bias"]).cuda()
    n_par = int(sum(q.numel() for q in m.parameters()))
    opt = torch.optim.AdamW(m.parameters(), lr=spec["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=spec["lr"] * .05)
    weights = weights.cuda()

    write(out / "config.json", dict(
        **vars(a), experiment="D3", epochs=epochs, batch=batch, workers=workers,
        num_classes=nc, parameters=n_par, embed_ch=spec["embed_ch"], gate_bias=spec["gate_bias"],
        radius_metres=spec["radius_metres"], lr=spec["lr"],
        adaptation="O input-fusion block (identical RGDIFusion) at the PointNeXt-S stem; in_channels 4->%d; no internal injection" % (4 + spec["embed_ch"]),
        precision="FP32, TF32 disabled", validation_sampling_seed=42, test_access=False,
        point_inputs="existing normalized XYZ/scalar base features; metric local XYZ for ball queries; no RGB/pretrained/extra augmentation",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        vendor=str(VENDOR), python=sys.executable,
        pointnext_commit=LOCK["reused_environment"]["pointnext_commit"],
        openpoints_commit=LOCK["reused_environment"]["openpoints_commit"],
        resume="not automatic; worker RNG not claimed exactly restorable"))

    started = time.time(); best = -1.0
    try:
        for ep in range(1, epochs + 1):
            if hasattr(ds, "epoch"):
                ds.epoch.value = ep
            m.train(); loss_sum = 0.0; seen = 0; t0 = time.time()
            for it, b in enumerate(dl):
                coords, x, deg, dlt, rel = obs_batch(b)
                y = b["labels"].cuda(non_blocking=True)
                opt.zero_grad(set_to_none=True)
                z = m(coords, x, deg, dlt, rel)
                loss = F.cross_entropy(z.reshape(-1, nc), y.reshape(-1), weight=weights, ignore_index=ignore)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss")
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1, error_if_nonfinite=True)
                if not torch.isfinite(gn):
                    raise FloatingPointError("Nonfinite gradients")
                opt.step()
                loss_sum += float(loss.detach()) * len(y); seen += len(y)
                if it % 20 == 0 or a.smoke:
                    write(out / "status.json", dict(status="running", epoch=ep, epochs=epochs,
                                                    batch=it + 1, batches=len(dl), elapsed=time.time() - started))
                if max_train and it + 1 >= max_train:
                    break
            m.eval(); cm = np.zeros((nc, nc), np.int64)
            with torch.no_grad():
                for bi, b in enumerate(vl):
                    if max_val and bi >= max_val:
                        break
                    coords, x, deg, dlt, rel = obs_batch(b)
                    z = m(coords, x, deg, dlt, rel)
                    if not torch.isfinite(z).all():
                        raise FloatingPointError("Nonfinite validation logits")
                    pred = z.argmax(-1).cpu().numpy().reshape(-1)
                    y = b["labels"].numpy().reshape(-1)
                    valid = (y >= 0) & (y < nc)
                    cm += np.bincount((y[valid] * nc + pred[valid]), minlength=nc * nc).reshape(nc, nc)
            metrics = T.metrics_from_confusion(cm); sched.step()
            row = dict(epoch=ep, train_loss=loss_sum / max(seen, 1), val_mIoU=metrics["mIoU"],
                       val_oa=metrics["oa"], seconds=time.time() - t0, evaluated_points=int(cm.sum()))
            with (out / "history.csv").open("a", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(row))
                if f.tell() == 0:
                    wr.writeheader()
                wr.writerow(row)
            ck = dict(model=m.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                      epoch=ep, metrics=metrics, torch_rng=torch.get_rng_state(),
                      cuda_rng=torch.cuda.get_rng_state_all(), numpy_rng=np.random.get_state(),
                      python_rng=__import__("random").getstate(), loader_rng=gen.get_state())
            torch.save(ck, out / "last.tmp"); (out / "last.tmp").replace(out / "last.pt")
            if metrics["mIoU"] > best:
                best = metrics["mIoU"]
                torch.save(ck, out / "best.tmp"); (out / "best.tmp").replace(out / "best.pt")
                np.save(out / "best_val_confusion.npy", cm)
                write(out / "best_val_metrics.json", dict(epoch=ep, **metrics))
            print(json.dumps(row), flush=True)
        write(out / "status.json", dict(status="complete", epochs=epochs, best_val_mIoU=best,
                                        seconds=time.time() - started, parameters=n_par))
    except BaseException as e:
        write(out / "status.json", dict(status="failed", error=repr(e))); raise


if __name__ == "__main__":
    main()
