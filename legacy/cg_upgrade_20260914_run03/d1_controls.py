"""D1: DALES dual-injection-position ablation (isolated; frozen trainer untouched).

Question (EXPERIMENT_PLAN_CN.md section 6, D1):
    Does the *dual-position* design itself matter, or is the complete system
    the only thing that is being compared?

Both variants reuse O's observation prior unchanged (degraded raster; restored
and reliability tensors are zero), O's dataset, O's optimizer/schedule/precision
and O's 50-epoch DALES budget. Only the wiring of the two fusion positions
changes:

    O-full       mode=restored_rgdi, inject_stages=1                  input fusion + first-stage injection
    input_only   mode=restored_rgdi, inject_stages=0                  input fusion only
    stage1_only  mode=restored_rgdi, inject_stages=1, input fusion bypassed
                                                                      first-stage injection only

The 4th factorial corner ("neither position") is the existing Base model, which
additionally does not load the observation prior at all -- this asymmetry is
recorded and must be stated when interpreting the ablation.

Nothing here edits the frozen trainer file; the class substitution happens at
runtime exactly as `run_dales.py` already does for dataset variants.
"""

from controls import T, ROOT, adapter  # noqa: F401  (adapter re-exported for runners)

VARIANTS = ("input_only", "stage1_only")


WIRING = {
    "input_only": dict(inject_stages=0, bypass_input_fusion=False),
    "stage1_only": dict(inject_stages=1, bypass_input_fusion=True),
}


def install_dataset():
    """O's R1 observation-only dataset (degraded prior; no clean/restored leak)."""
    return adapter.install_control(T, "observation")


def model_class(variant):
    """Return a RandLASegmentationModel subclass implementing the D1 wiring.

    `stage1_only` forces the backbone input width to the 4-dim base features and
    drops the (now unused) input-fusion module, so the observation prior reaches
    the network *only* through the first encoder-stage injector.
    """
    if variant not in VARIANTS:
        raise ValueError(variant)

    inject_stages, bypass = WIRING[variant]["inject_stages"], WIRING[variant]["bypass_input_fusion"]
    Original = T.RandLASegmentationModel
    RealRandLANet = T.RandLANet

    class D1Model(Original):
        cg_variant = variant
        cg_bypass_input_fusion = bypass

        def __init__(self, **kwargs):
            if bypass:

                def forced_randlanet(*args, **kw):
                    kw["in_channels"] = 4
                    return RealRandLANet(*args, **kw)

                T.RandLANet = forced_randlanet
            try:
                super().__init__(**kwargs)
            finally:
                T.RandLANet = RealRandLANet
            if bypass:

                self.rgdi = None

        def _build_input_features(self, batch):
            if bypass:
                return batch["base"]
            return super()._build_input_features(batch)

    D1Model.__name__ = "D1_" + variant
    D1Model.__qualname__ = D1Model.__name__
    return D1Model


def model_args(cfg):
    """Explicit model kwargs, so tests and runners cannot drift apart."""
    return dict(
        mode=cfg["mode"],
        num_points=cfg["num_points"],
        num_neighbors=cfg["num_neighbors"],
        num_layers=cfg["num_layers"],
        sub_sampling_ratio=[int(x) for x in str(cfg["sub_sampling_ratio"]).split(",")],
        dim_output=[int(x) for x in str(cfg["dim_output"]).split(",")],
        dim_features=cfg["dim_features"],
        embed_ch=cfg["embed_ch"],
        gate_bias=cfg["gate_bias"],
        inject_stages=WIRING[cfg["cg_variant"]]["inject_stages"],
        internal_hidden_ch=cfg["internal_hidden_ch"],
        internal_alpha_init=cfg["internal_alpha_init"],
        use_explicit_quality_modulation=not cfg["disable_explicit_quality_modulation"],
        use_reliability_descriptors=not cfg["disable_reliability_descriptors"],
        gate_mode=cfg["gate_mode"],
        dropout=cfg["dropout"],
    )
