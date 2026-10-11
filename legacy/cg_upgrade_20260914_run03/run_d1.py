"""Train a single locked D1 dual-position ablation run. Never touches test, never overwrites.

Reuses O's frozen trainer loop verbatim (optimizer, schedule, precision, loader
topology) and only substitutes the model wiring and the observation dataset.
"""
import argparse, hashlib, inspect, json, os, random, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from controls import T, ROOT
from d1_controls import install_dataset, model_class, WIRING

Q = Path(__file__).resolve().parent
REFERENCE = ROOT / "PROJECT_NAV_20260909/05_new_work/results/observation_seeds43_44_20260910_01/train/s43_observation/config.json"


def write(p, d):
    tmp = p.with_suffix(".tmp"); tmp.write_text(json.dumps(d, indent=2)); tmp.replace(p)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["input_only", "stage1_only"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()

    out = Q / ("smoke" if a.smoke else "train") / f"D1_{a.variant}_{a.seed}"
    out.mkdir(parents=True, exist_ok=False)

    cfg = json.loads(REFERENCE.read_text())
    for path, expected in cfg["source_hashes"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path

    cfg.update(
        out_dir=str(out), seed=a.seed, skip_test=True,
        inject_stages=WIRING[a.variant]["inject_stages"],
        cg_variant=a.variant, cg_experiment="D1",
        cg_bypass_input_fusion=WIRING[a.variant]["bypass_input_fusion"],
        cg_source_hashes={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in Q.glob("*.py")},
        cg_reference_config=str(REFERENCE),
        cg_training_resume="not implemented; do not silently resume worker RNG",
        cg_test_access=False,
    )
    if a.smoke:
        cfg.update(epochs=1, max_train_files=16, max_val_files=4)

    T.DALESRandLADataset = install_dataset()
    T.RandLASegmentationModel = model_class(a.variant)


    source = inspect.getsource(T.train)
    source = source.replace(
        "            scaler.scale(loss).backward()",
        '            if not torch.isfinite(loss): raise FloatingPointError("Nonfinite training loss")\n            scaler.scale(loss).backward()',
    )
    source = source.replace("                torch.nn.utils.clip_grad_norm_(", "                checked_clip(")
    source = source.replace(
        "        checkpoint = {",
        '        checkpoint = {\n            "torch_rng": torch.get_rng_state(),\n            "cuda_rng": torch.cuda.get_rng_state_all(),\n            "numpy_rng": np.random.get_state(),\n            "python_rng": __import__("random").getstate(),',
    )
    overflow = dict(count=0, consecutive=0, steps=0)

    def checked_clip(parameters, max_norm):


        norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
        overflow["steps"] += 1
        if not torch.isfinite(norm):
            overflow["count"] += 1; overflow["consecutive"] += 1
            with (out / "amp_overflow.jsonl").open("a") as f:
                f.write(json.dumps(dict(**overflow, time=time.time(),
                                        action="original GradScaler skips optimizer step and reduces scale")) + "\n")
            if not cfg["amp"] or overflow["consecutive"] > 3:
                raise FloatingPointError("Persistent nonfinite gradient; stopped")
        else:
            overflow["consecutive"] = 0
        return norm

    scope = dict(T.__dict__); scope["checked_clip"] = checked_clip
    exec(compile(source, "<cg_d1_checked_original_train>", "exec"), scope)

    status = dict(status="running", experiment="D1", variant=a.variant, seed=a.seed,
                  pid=os.getpid(), started=time.time(), output=str(out))
    write(out / "cg_status.json", status)
    try:
        scope["train"](SimpleNamespace(**cfg))
        status.update(status="complete", completed=time.time())
    except BaseException as e:
        status.update(status="failed", error=repr(e), completed=time.time()); raise
    finally:
        write(out / "cg_status.json", status)


if __name__ == "__main__":
    main()
