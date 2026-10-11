# ALS 2.5D context fusion

Code for **Evaluating observation support and restoration pretraining in 2.5D context fusion for aerial point-cloud segmentation**.

This private preparation repository collects the implementation and archived configurations used for the manuscript. It includes the October four-seed DALES extension and the scoring-coverage and observation-support diagnostics. It is not a public data or trained-model release. Author metadata, publication identifiers and the release license for original code remain to be confirmed.

## Contents

| Location | Contents |
|---|---|
| `legacy/ALS-DiffBridge/scripts/` | Raster construction, normalization, RandLA implementation, restoration and frozen-feature controls |
| `legacy/cg_upgrade_20260914_run02/` | DALES support/capacity controls, external support control and PointNeXt adapter |
| `legacy/cg_upgrade_20260914_run03/` | Fusion-position controls and the PointNeXt input-fusion adaptation |
| `legacy/cg_evaluation_20260914_01/` | Fixed-sampled and full-retained evaluation |
| `legacy/cg_analysis_20260914_01/` | Terrain evaluation and cost measurements |
| `legacy/f16_run/`, `legacy/f21_run/`, `legacy/f44/` | Additional DALES seeds, fixed-prediction rescoring and support-size diagnostics |
| `legacy/whu25d/`, `legacy/h3d25d/` | Dataset adapters and required metadata |
| `configs/archive/` | Original experiment configurations; not fresh-run output directories |
| `provenance/` | Original/cleaned source hashes, protocol locks and observed environments |
| `tools/materialize_workspace.py` | Create a separate working copy with relocated paths and traceable runtime source hashes |
| `tests/` | Workspace-preparation tests and bounded integration checks |

Directory names in `legacy/` preserve imports and source-inspection dependencies. Do not run the archived source directly against the original experiment root. Create a separate workspace first.

## Start here

The observed server environments use Linux, Python 3.12.3, PyTorch 2.8.0+cu128 and NumPy 1.26.4. See `provenance/environments.json`. PointNeXt needs additional upstream packages and compiled operators; `requirements-core.txt` alone does not install them.

```bash
python -m pip install -r requirements-core.txt
python -m unittest discover -v
python tools/materialize_workspace.py --workspace /path/to/new-als-workspace
```

The target must not exist and must be outside this repository. Dataset directories, feature caches, checkpoint files and PointNeXt dependencies are deliberately not copied by this command. The script does not train models or modify the source repository.

After providing the processed DALES data at the locations described in [the reproduction guide](docs/REPRODUCIBILITY.md), run the original control checks:

```bash
cd /path/to/new-als-workspace/cg_upgrade_20260914_run02
python -m unittest test_controls -v
```

An optional one-epoch training smoke run, requiring the appropriate training environment and data, is:

```bash
python run_dales.py --variant sampled --seed 43 --smoke
```

This smoke run is not a reproduction of the manuscript results. Full training, checkpoint selection and evaluation must follow the archived dataset-specific configurations.

## Evidence and scope

Observation support, point capacity, frozen restoration features and evaluation grouping are distinct controls. The datasets were trained separately. The four-seed extension applies only to DALES O-sampled and O-point; it does not turn the entire study into a four-seed experiment. Higher segmentation mIoU is not a claim of improved survey-grade elevation accuracy.

[The method-to-code map](docs/METHOD_MAP.md) identifies the relevant modules. [Verification notes](docs/VERIFICATION.md) state exactly what was tested in this repository preparation.

## Comment cleanup and provenance

Process comments were removed from project-owned source without changing its executable AST. Three internal module descriptions were shortened; outdated dataset descriptions in the WHU and H3D adapters were corrected. Functional raster-source markers, encoding declarations and third-party attribution were preserved. `provenance/source_cleanup.json` records every source hash and retained comment.

Historical protocol locks and configurations remain archival evidence. A generated workspace has different source-file hashes because comments and machine paths changed; its `WORKSPACE_PROVENANCE.json` records this explicitly. It must not be presented as the byte-identical source snapshot used for the original training runs.

## Data, licenses and citation

Raw third-party point clouds, model weights, prediction caches, credentials and internal manuscript drafts are excluded. A private repository URL does not provide public code access. See [third-party notices](third_party/README.md) before selecting a public release license. Citation author names and publication details should be added only after author confirmation.

