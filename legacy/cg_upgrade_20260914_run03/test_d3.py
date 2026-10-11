"""D3 preflight: verify the O-on-PointNeXt adaptation BEFORE any formal training.

Checks, on the frozen code paths and real validation data:
  1. the reused independent-backbone environment still matches its locked hashes;
  2. PointNextObs.rgdi is structurally identical to O's RGDIFusion (keys/shapes);
  3. the PointNeXt stem actually consumes 4+32=36 input channels;
  4. a real validation batch forward yields finite logits of shape (B, 8192, nc);
  5. a backward pass yields finite gradients that reach both the fusion block and
     the backbone.

Writes D3_PREFLIGHT.json and exits non-zero on failure. Reads no test data.
"""
import json, sys, traceback
from pathlib import Path
import torch
import torch.nn.functional as F

from controls import T
from d3_runner import PointNextObs, build_datasets, verify_environment, obs_batch, LOCK

Q = Path(__file__).resolve().parent
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    report = {"device": str(DEV), "checks": {}, "parameter_counts": {}}
    checks = report["checks"]
    ok = True

    def check(name, value, expected):
        nonlocal ok
        good = value == expected
        checks[name] = {"value": value, "expected": expected, "pass": bool(good)}
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] {name}: {value} (expected {expected})")

    try:
        verify_environment()
        check("environment_hashes_locked", True, True)
    except AssertionError as e:
        check("environment_hashes_locked", str(e), True)


    o_rgdi = T.RGDIFusion(base_ch=4, prior_ch=9, rel_ch=6, embed_ch=32, gate_bias=0.3,
                          use_explicit_quality_modulation=False, use_reliability_descriptors=False,
                          gate_mode="learned")
    o_sig = {k: list(v.shape) for k, v in o_rgdi.state_dict().items()}
    o_rgdi_params = int(sum(p.numel() for p in o_rgdi.parameters()))

    for name, ncls in (("DALES", 8), ("H3D", 11)):
        spec = LOCK["datasets"][name]
        m = PointNextObs(ncls, spec["radius_metres"], spec["embed_ch"], spec["gate_bias"]).to(DEV)
        n_par = int(sum(p.numel() for p in m.parameters()))
        report["parameter_counts"][name] = dict(total=n_par, rgdi=int(sum(p.numel() for p in m.rgdi.parameters())))

        check(f"{name}.rgdi_signature_matches_O",
              {k: list(v.shape) for k, v in m.rgdi.state_dict().items()}, o_sig)
        check(f"{name}.rgdi_param_count_matches_O",
              int(sum(p.numel() for p in m.rgdi.parameters())), o_rgdi_params)

        ds, vs, collate, weights, nc, bsz, workers, ignore = build_datasets(name, 43)
        check(f"{name}.num_classes", nc, ncls)
        items = [vs[i] for i in range(2)]
        b = collate(items)
        check(f"{name}.class_weight_len", int(weights.numel()), ncls)

        coords, x, deg, dlt, rel = obs_batch(b)
        check(f"{name}.obs_shapes", [list(x.shape), list(deg.shape), list(dlt.shape), list(rel.shape)],
              [[2, 8192, 4], [2, 8192, 9], [2, 8192, 9], [2, 8192, 6]])



        n = coords.shape[1]
        pos = coords.float()

        def _try(width):
            try:
                with torch.no_grad():
                    m.net({"pos": pos, "x": torch.randn(2, width, n, device=DEV)})
                return True
            except Exception:
                return False

        check(f"{name}.backbone_accepts_36ch", _try(36), True)
        check(f"{name}.backbone_rejects_4ch", _try(4), False)

        m.eval()
        with torch.no_grad():
            z = m(coords, x, deg, dlt, rel)
        check(f"{name}.logits_shape", list(z.shape), [2, 8192, ncls])
        check(f"{name}.logits_finite", bool(torch.isfinite(z).all()), True)

        m.train()
        z = m(coords, x, deg, dlt, rel)
        y = b["labels"].to(DEV)
        loss = F.cross_entropy(z.reshape(-1, ncls), y.reshape(-1), weight=weights.to(DEV), ignore_index=ignore)
        check(f"{name}.loss_finite", bool(torch.isfinite(loss)), True)
        loss.backward()
        rgrad = sum(float(p.grad.abs().sum()) for k, p in m.named_parameters()
                    if k.startswith("rgdi") and p.grad is not None)
        bgrad = sum(float(p.grad.abs().sum()) for k, p in m.named_parameters()
                    if k.startswith("net.") and p.grad is not None)
        check(f"{name}.rgdi_grad_nonzero", bool(rgrad > 0), True)
        check(f"{name}.backbone_grad_nonzero", bool(bgrad > 0), True)
        bad = [k for k, p in m.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        check(f"{name}.finite_grads", len(bad), 0)
        m.zero_grad(set_to_none=True)
        del m, ds, vs, b
        torch.cuda.empty_cache()

    report["status"] = "passed" if ok else "failed"
    (Q / "D3_PREFLIGHT.json").write_text(json.dumps(report, indent=2))
    print("\nPARAMETER COUNTS:", json.dumps(report["parameter_counts"]))
    print("STATUS:", report["status"])
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        traceback.print_exc()
        (Q / "D3_PREFLIGHT.json").write_text(
            json.dumps({"status": "error", "traceback": traceback.format_exc()}, indent=2))
        sys.exit(2)
    sys.exit(code)
