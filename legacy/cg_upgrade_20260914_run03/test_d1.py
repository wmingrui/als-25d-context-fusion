"""D1 preflight: structural + numerical verification BEFORE any formal training.

Answers, on the frozen code paths:
  1. Do the three factorisations have the intended submodules and input widths?
  2. Are the fusion submodules structurally identical to O's (same keys/shapes)?
  3. Does a real validation batch forward produce finite, correctly shaped logits?
  4. Does the plain forward path equal the multiscale path when no hook is active
     (so input_only vs O-full differ ONLY by the injector, not by code path)?
  5. Do gradients reach the expected parameters and stay finite?

Writes D1_PREFLIGHT.json and exits non-zero if any check fails. Reads no test data.
"""
import json, sys, traceback
from pathlib import Path
import numpy as np
import torch

from controls import T, ROOT, adapter
from d1_controls import install_dataset, model_class, model_args

Q = Path(__file__).resolve().parent
REFERENCE = ROOT / "PROJECT_NAV_20260909/05_new_work/results/observation_seeds43_44_20260910_01/train/s43_observation/config.json"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_dev(x):
    if torch.is_tensor(x):
        return x.to(DEV)
    if isinstance(x, dict):
        return {k: to_dev(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_dev(v) for v in x]
    return x


def build_val_batch():
    cfg = json.loads(REFERENCE.read_text())
    cls = install_dataset()
    sub = [int(x) for x in str(cfg["sub_sampling_ratio"]).split(",")]
    ds = cls(split="val", max_files=2,
             block_dir=cfg["block_dir"], clean_map_dir=cfg["clean_map_dir"],
             deg_map_dir=cfg["deg_map_dir"], restored_map_dir=cfg["restored_map_dir"],
             mode=cfg["mode"], num_points=cfg["num_points"], num_layers=cfg["num_layers"],
             num_neighbors=cfg["num_neighbors"], sub_sampling_ratio=sub,
             degradation_tag=cfg["degradation_tag"], seed=cfg["seed"])
    items = [ds[i] for i in range(2)]
    return T.randla_collate(items)


def sd_signature(module):
    if module is None:
        return None
    return {k: list(v.shape) for k, v in module.state_dict().items()}


def in_channels_of(model):
    w = model.backbone.fc0.weight
    return int(w.shape[1]) if w.ndim >= 2 else None


def main():
    torch.manual_seed(0)
    cfg = json.loads(REFERENCE.read_text())
    report = {"reference_config": str(REFERENCE), "device": str(DEV), "checks": {}}
    checks = report["checks"]
    ok = True

    def check(name, value, expected):
        nonlocal ok
        good = value == expected
        checks[name] = {"value": value, "expected": expected, "pass": bool(good)}
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] {name}: {value} (expected {expected})")

    models = {}
    for variant in ("O_full", "input_only", "stage1_only"):
        vcfg = dict(cfg)
        if variant == "O_full":
            vcfg["cg_variant"] = "input_only"
            cls = T.RandLASegmentationModel
            args = model_args(vcfg)
            args["inject_stages"] = 1
        else:
            vcfg["cg_variant"] = variant
            cls = model_class(variant)
            args = model_args(vcfg)
        models[variant] = cls(**args, device=str(DEV)).to(DEV)


    for variant, want_inj, want_bypass in (("O_full", 1, False),
                                           ("input_only", 0, False),
                                           ("stage1_only", 1, True)):
        m = models[variant]
        check(f"{variant}.internal_injectors", len(m.internal_injectors), want_inj)
        check(f"{variant}.has_rgdi", m.rgdi is not None, not want_bypass)
        check(f"{variant}.backbone_in_channels", in_channels_of(m), 4 if want_bypass else 36)
        check(f"{variant}.inject_stages", int(m.inject_stages), want_inj)


    check("rgdi_signature_O_eq_input_only",
          sd_signature(models["input_only"].rgdi), sd_signature(models["O_full"].rgdi))
    check("injector0_signature_O_eq_stage1_only",
          sd_signature(models["stage1_only"].internal_injectors[0]),
          sd_signature(models["O_full"].internal_injectors[0]))
    check("rgdi_param_shapes_O_eq_input_only",
          [list(p.shape) for p in models["input_only"].rgdi.parameters()],
          [list(p.shape) for p in models["O_full"].rgdi.parameters()])


    batch = to_dev(build_val_batch())
    n_params = {}
    for variant, m in models.items():
        n_params[variant] = int(sum(p.numel() for p in m.parameters()))
        m.eval()
        with torch.no_grad():
            out = m(batch)
        check(f"{variant}.logits_shape", list(out.shape), [2, cfg["num_points"], 8])
        check(f"{variant}.logits_finite", bool(torch.isfinite(out).all()), True)

        if variant == "input_only":
            feat = m._build_input_features(batch)
            with torch.no_grad():
                via_plain = m.backbone({"features": feat, "coords": batch["coords"],
                                        "neighbor_indices": batch["neighbor_indices"],
                                        "sub_idx": batch["sub_idx"], "interp_idx": batch["interp_idx"]})
                via_ms = m._forward_multiscale_rgdi(feat, batch)
            diff = float((via_plain - via_ms).abs().max())
            checks["forward_path_max_abs_diff"] = {"value": diff, "expected": "<=1e-5", "pass": bool(diff <= 1e-5)}
            ok &= diff <= 1e-5
            print(f"[{'PASS' if diff <= 1e-5 else 'FAIL'}] forward_path_max_abs_diff: {diff}")

        m.train()
        out = m(batch)
        loss = out.reshape(-1, 8).mean()
        loss.backward()
        grads = {n: float(p.grad.detach().abs().sum()) for n, p in m.named_parameters() if p.grad is not None}
        bad = [n for n, g in grads.items() if not np.isfinite(g)]
        check(f"{variant}.finite_grads", len(bad), 0)
        checks[f"{variant}.grad_coverage"] = {
            "params_with_grad": len(grads),
            "params_total": len(list(m.parameters())),
            "pass": True,
        }
        if variant == "stage1_only":
            injector_grad = sum(g for n, g in grads.items() if n.startswith("internal_injectors"))
            check("stage1_only.injector_grad_nonzero", bool(injector_grad > 0), True)
        if variant == "input_only":
            check("input_only.rgdi_grad_nonzero",
                  bool(sum(g for n, g in grads.items() if n.startswith("rgdi")) > 0), True)

        feat = m._build_input_features(batch)
        checks[f"{variant}.input_feature_shape"] = {"value": list(feat.shape), "pass": True}
        m.zero_grad(set_to_none=True)

    report["parameter_counts"] = n_params
    report["status"] = "passed" if ok else "failed"
    (Q / "D1_PREFLIGHT.json").write_text(json.dumps(report, indent=2))
    print("\nPARAMETER COUNTS:", json.dumps(n_params))
    print("STATUS:", report["status"])
    return 0 if ok else 1


if __name__ == "__main__":


    try:
        code = main()
    except Exception:
        traceback.print_exc()
        (Q / "D1_PREFLIGHT.json").write_text(
            json.dumps({"status": "error", "traceback": traceback.format_exc()}, indent=2))
        sys.exit(2)
    sys.exit(code)
