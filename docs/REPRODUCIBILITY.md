# Reproduction guide

## Preparation versus a complete reproduction

This repository organizes code and configurations. It does not supply raw datasets, trained weights or complete prediction/feature caches. Recreating every manuscript row requires those inputs or new training runs. The preparation checks do not establish that a fresh end-to-end run reproduces every reported number.

Use Python 3.12 on Linux for the recorded environment. Some evaluation code uses `hashlib.file_digest`, requiring Python 3.11 or newer. Install the CUDA-enabled PyTorch build appropriate to the recorded environment before compiling PointNeXt operators. The observed package versions are in `provenance/environments.json`; the repository does not recreate the full binary environment.

## Workspace data layout

First run `tools/materialize_workspace.py --workspace /absolute/path/to/new-workspace`. The command requires a new target and records both repository and runtime hashes. Supply the following data separately, retaining the provider's access conditions:

| Workspace-relative location | Required content |
|---|---|
| `ALS-DiffBridge/data/processed/dales_objects/` | DALES processed train/validation blocks and shared-normalized maps |
| `als-transfer/data/` | DALES processed test blocks and maps |
| `whu25d/data/` | WHU raw/processed maps referenced by its protocol |
| `whu25d/completion_v2/data/point_cache/` | WHU point caches by severity and split |
| `h3d25d/run_v2/data/` | H3D point caches and processed products referenced by its protocol |
| `ALS-DiffBridge/outputs_uq/` | Frozen-feature caches required by the selected F-pre/F-rnd configuration |
| `cg_upgrade_20260914_run02/vendor/PointNeXt-master/` | PointNeXt with OpenPoints in its expected location |

No data download or redistribution is performed automatically. Existing data can be mounted or linked into a new workspace for evaluation. Do not run preprocessing repair scripts against a link to the only copy of a dataset; create a separate writable preprocessing destination.

The included DALES raster and degradation scripts are original implementations. Their CLI/default paths should be inspected before preparing new data. This release does not claim to provide a complete raw-download-to-paper pipeline for all three providers.

## Source hashes

`configs/archive/` and `provenance/protocols/` retain historical metadata. They can contain original machine paths and source/checkpoint hashes, which identify the historical runs. They are not secrets and are not silently rewritten.

The materialization tool rewrites the historical root in a separate copy of the runnable source and required reference configurations. It refreshes file-based source hashes in those runtime reference configurations and records old and new values. This only prepares a new run; it does not authenticate an old checkpoint against modified source. Original normalization JSON files are copied byte for byte because their content and checksums are part of the data contract.

Historical evaluation scripts expect selected checkpoints, inventories and protocol locks at their original relative locations. Restore the matching artifacts or create new run-specific inventories before using these scripts. Merely copying an old lock into a new workspace is not a complete evaluation setup. F16's evaluator updates its workspace-local full-retained lock and writes a backup before doing so.

## Training controls

With processed data and the required environment in place, the DALES support/capacity entry point is:

```bash
cd /path/to/new-workspace/cg_upgrade_20260914_run02
python run_dales.py --variant sampled --seed 43
python run_dales.py --variant point --seed 43
```

Use seeds 43 and 44 for the original matched controls; DALES O-sampled and O-point also have seeds 45 and 46. The scripts create new output directories and intentionally do not silently resume a run. Existing archived configurations are stored separately to avoid occupying those fresh output directories.

The external support-control entry point accepts `--dataset WHU` or `--dataset H3D`, `--seed`, and optional `--smoke`. The external `train_complete.py` entry points accept `--method base|observation|feature|random_feature|unet`. Their precision and data settings differ from the PointNeXt configurations: do not apply one “all experiments” recipe across both backbones.

PointNeXt's point-only entry point requires `--dataset`, `--seed`, `--radius`, `--lr`, `--epochs` and `--job`. Use the dataset-specific archived configuration, not guessed defaults. The recorded selected radius and learning rate are 1.0 m and 0.001. The D3 adaptation also validates a locked environment; supplying a fresh, different environment requires a separately recorded compatibility check, not bypassing its assertions.

Materialization includes `cg_upgrade_20260914_run02/pointnext_commit.json` and `openpoints_ref.json` byte for byte. D3 checks these dependency records against its lock, so retain them alongside the matching vendor sources. These records do not install or verify the compiled operators.

## Evaluation and diagnostics

- Fixed-sampled and full-retained scoring use the same checkpoint but differ in coverage and inference grouping. They are not interchangeable metric names.
- The full-retained evaluator expects the original 38-job inventory plus four additional DALES jobs when `F16_EXPECTED=4`. This environment variable does not create the missing checkpoints.
- F21 expects cached full-retained predictions and performs confusion-matrix/order checks. Its `--dry-run` is an inventory operation, not proof of score reproducibility.
- F44's default `--limit` is 200; use the intended extent when interpreting support-size summaries.
- Terrain diagnostics report coverage as well as errors. Retain the common-cell and per-tile evidence, including unfavorable results.

See `docs/VERIFICATION.md` for the checks actually executed during repository preparation.

