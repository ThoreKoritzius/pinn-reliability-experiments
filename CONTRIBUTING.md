# Contributing

This is a small experimental repository. Reproductions, numerical corrections
and narrowly scoped ablations are especially useful.

For an experiment, describe the question, the one factor changed, the reference
solution, the training and evaluation coordinates, seeds, optimizer budget and
metric definitions. Include individual runs, not just a best seed. Distinguish
iteration-matched from time-matched comparisons, and label the supervised control
as receiving extra information.

Use a fresh output directory rather than replacing `results/`. Commit small
configs and scalar results with provenance; discuss large generated datasets or
HTML bundles before adding them. Never interpret many points or checkpoints from
one flow as independent problem-level replication.

Before a pull request:

```bash
uv sync --frozen
uv run python -m unittest discover -s tests -v
uv run python export_explorer.py --standalone --resolution 12 --frames 3 --out results/test-demo.html
node validate_explorer.cjs results/explorer-data.html
node validate_standalone.cjs
```

When changing figures, regenerate them from the saved numerical data and state
any clipping, smoothing, normalization and independent color scales. A visually
plausible flow is not a substitute for error measurement. Proposed uncertainty
methods need both coverage and sharpness evaluation and an explicit sampling
assumption; ensemble spread alone is not a calibrated interval.

Contributions are made under the repository's MIT license. Please cite external
ideas and do not add copied papers, proprietary runtimes or unlicensed assets.
