# Third-party components and release status

| Component | Source / recorded revision | Treatment |
|---|---|---|
| Open3D-ML RandLA-Net | https://github.com/isl-org/Open3D-ML | The local pure-PyTorch fallback states that it adapts this implementation. Its attribution remains in the source; `Open3D-ML/LICENSE` is included. The original adaptation did not record an exact upstream commit, so none is invented here. |
| PointNeXt | https://github.com/guochengqian/PointNeXt — `b7ec10fd8b9acdb1c3b18e557f0d3465eb919164` | Dependency source and compiled operators are not vendored. `PointNeXt/LICENSE` was copied from the experiment archive. |
| OpenPoints | https://github.com/guochengqian/openpoints — `db31e0d94ede35bc672ef06d1b6e4073a9c269d5` | Required by PointNeXt. `OpenPoints/LICENSE` was copied from the experiment archive. |

Use the pinned upstream sources and their installation instructions to obtain the PointNeXt/OpenPoints dependency tree. The experiment's vendored environment may include local build adjustments; a new installation is not assumed to be binary-identical. The recorded revisions, configuration locks and environment metadata remain available under `provenance/`.

The three included license files apply to their respective third-party components. No new open-source license has been selected for the project's original code in this private preparation. Confirm ownership and choose a release license before making a public release. Dataset access and redistribution are governed separately by their providers.

