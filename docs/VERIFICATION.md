# Verification record — 2026-10-11

These checks concern repository preparation, not a new training campaign.

| Check | Result | Scope |
|---|---|---|
| Original-to-cleaned Python executable AST | 57 / 57 match | Ignores docstrings; ordinary comments have no AST nodes |
| Python syntax | Passed | All project Python files parsed/compiled |
| Source provenance | Passed | SHA256 of each original and cleaned file matches the cleanup manifest |
| Workspace preparation unit tests | 7 passed | New target only, path relocation, unchanged numeric settings, normalization byte preservation and runtime hash records |
| DALES control tests | 4 passed | Real training/validation blocks; full-raster regression, point mapping, unique sampled support and dataset sampling identity |
| Terrain unit tests | 4 passed | Empty cells, median grids, coverage, errors and slope |
| Fusion-position preflight | Passed | Real validation batch; topology, output shape, finite values, equivalent forward path and gradient reachability |
| WHU / H3D bounded integration | Passed | Two real validation blocks per dataset; raster agreement and unchanged point sampling/hierarchy |
| Credential-pattern scan | No matches | Private-key, GitHub token, access-key and literal-password patterns; this is not a guarantee against every possible secret format |

The server checks ran in a newly created workspace with source paths relocated. Existing processed data and the dependency directory were linked for reading. No formal training, checkpoint updates or original experiment-directory edits were performed.

## Issues encountered and resolved

The first server import test exposed two removed comment prefixes used as source-slicing markers. They were restored, the cleanup manifest was updated, and the DALES and fusion-position checks then passed. This is why AST equality alone is not used as proof of runtime equivalence. The raster's three functional markers remain in the release.

The original external-dataset test preloads whole training/validation splits and exited with signal 9 in the current environment. The container reports a 2 GiB memory limit; the exact kill source was not established. The original test is retained unchanged. A separate bounded integration check reads two validation blocks per dataset and passed for both WHU and H3D. This does not claim that the original whole-split test passed.

## Not rerun

- Full training for every seed, dataset and backbone.
- Complete fixed-sampled/full-retained evaluation against every checkpoint.
- Full PointNeXt/CUDA-extension installation or D3 training in a fresh environment.
- F21 rescoring over every prediction cache or the complete terrain/cost campaign.
- End-to-end raw-data preprocessing for all providers.

The repository contains the corresponding source and selected configurations, but those tasks also require separately held data, caches, checkpoints and dependency builds. Historical numerical results are not regenerated or altered by this preparation.

