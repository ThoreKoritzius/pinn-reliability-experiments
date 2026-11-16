# Reference artifact provenance

The initial snapshot was created on 16 November 2026. It contains two soft-IC
PINNs, two hard-IC PINNs and one supervised diagnostic. Each has an Adam checkpoint
and a final L-BFGS checkpoint. These are generated project outputs, not downloaded
models. All runs concern the same dimensionless nu=0.1, T=2 Taylor–Green problem.

The execution environment recorded during the original experiment was Python
3.13.4, PyTorch 2.9.0, NumPy 2.4.6 and Matplotlib 3.10.7, CPU float64, one CPU thread.
Direct dependencies are pinned to those versions. The lockfile was resolved during
repository preparation, after training; it is not a historical capture of every
transitive package in the original process. In particular, no claim of bitwise
identical cross-platform retraining is made.

## What is preserved

- `results/<run>/metrics.json`: configuration, parameter count, PyTorch version,
  elapsed time, sampled training history and temporal diagnostics.
- `results/<run>/history.csv`: scalar history, including line-search trial points.
- `results/<run>/after_*.pt`: weights-only-loadable dictionaries containing the
  network state and configuration. Construct the model with the config's
  `hard_ic` flag (default false for older soft-IC artifacts).
- Per-run plots, spatial-grid comparison and the exploratory uncertainty audit.
- `artifact-manifest.json`: SHA-256 digests of checkpoints, metrics and histories.
  Generated figures are excluded so rendering changes do not alter numerical
  provenance. Hashes identify the snapshot; they do not certify scientific validity.

CSV line endings were normalized to LF during repository preparation; numerical
values, histories and checkpoint tensors were not changed. The manifest records
the normalized files as committed.

The original artifacts recorded PyTorch version but not every software version or
a source commit. The first repository commit establishes the versioned source
snapshot. Saved checkpoint metrics are re-evaluated by `tests/test_science.py`.
Later research should record source revisions and complete environments at run time.

## Evaluation details

Reported relative L2 errors use the square root of the ratio of summed squared
errors to summed squared reference values over both spatial points and times;
velocity sums both components. The endpoint-excluding spatial grid is 32² and
the temporal grid has 17 equally spaced points including 0 and 2. No trapezoidal
endpoint weighting is used. Pressure means are removed separately at each time.
These are sampled norm ratios, not rigorous continuous-space error bounds.

The 64² check refines only the spatial evaluation grid, not training, temporal
quadrature or an independent CFD discretization. The explorer uses 96² cell-center
samples at 33 times, so its sampled summaries need not exactly equal the report's.

Audit coordinates are a separate scrambled Sobol design. That is useful for an
independent residual probe but does not supply IID/exchangeable samples for an
ordinary conformal coverage claim. Both seeds change the network initialization
and training scramble. No results are independent replications across flow regimes.

Timing includes diagnostics and is not a clean optimizer benchmark. The two
hard-IC runs were launched concurrently, whereas some earlier runs were not.
Only the architecture, coordinates, hyperparameters and iteration limits were
matched; do not infer a speedup from these timing fields.

## Rendering and generated bundles

`make_readme_figure.py` makes the README figure directly from checkpoints at 128²
sample locations. Vorticity uses automatic differentiation; streamlines use the
sampled instantaneous velocity. Each bottom-row error map has its own labelled
color scale. The figure is not a screenshot of the browser application.

`demo.html`, the historical `explorer.fragment.html`, and intermediate browser
payloads are ignored by Git and can be regenerated. The standalone builder uses
only the templates in this repository and installed Python packages. The optional
inline export still refers to a public chart runtime for use inside Codex, but it
is not needed to train, test, plot, or build the standalone demo.
