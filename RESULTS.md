# First Taylor–Green results

These are exploratory runs, not a benchmark or a publication claim. Relative L2
errors aggregate a uniform 32×32 spatial grid at 17 equally spaced times. Pressure
is centered separately at each time. RMS residuals and relative solution errors
are different quantities and their numerical magnitudes are not interchangeable.

| Run | Stage | Velocity relative L2 | Pressure relative L2 | Train momentum RMS | Audit momentum RMS |
|---|---|---:|---:|---:|---:|
| hard_ic_seed0 | after_adam | 1.190% | 7.042% | 0.01245 | 0.01255 |
| hard_ic_seed0 | after_lbfgs | 0.334% | 2.143% | 0.00344 | 0.00352 |
| hard_ic_seed1 | after_adam | 1.089% | 6.347% | 0.01243 | 0.01264 |
| hard_ic_seed1 | after_lbfgs | 0.268% | 1.923% | 0.00308 | 0.00319 |
| pinn_seed0 | after_adam | 5.366% | 14.928% | 0.03289 | 0.03319 |
| pinn_seed0 | after_lbfgs | 2.098% | 6.406% | 0.01487 | 0.01512 |
| pinn_seed1 | after_adam | 5.914% | 15.661% | 0.04045 | 0.04155 |
| pinn_seed1 | after_lbfgs | 1.729% | 5.753% | 0.01431 | 0.01457 |
| supervised_seed0 | after_adam | 2.750% | 5.705% | 0.03996 | 0.03999 |
| supervised_seed0 | after_lbfgs | 0.942% | 2.403% | 0.01476 | 0.01472 |

## What these runs tell us

- Refinement from Adam to L-BFGS reduces error without changing architecture or
  training points. Earlier error was reducible. Extra computation and optimizer
  choice are confounded; this is not a matched-budget optimizer comparison.
- The supervised diagnostic uses the same base MLP and initialization as soft-IC
  seed 0, with exact interior labels. It is more accurate than soft-IC training,
  while hard-IC training is more accurate still in these runs. It is not an
  equal-information baseline or an irreducible-error lower bound.
- Training and independently sampled audit residuals are close in these runs.
  We do not observe a large collocation gap at these sample sizes. That does not
  certify the solution or eliminate all sampling effects.
- Two PINN seeds change both initialization and the training Sobol scramble, so
  their difference cannot be attributed to initialization alone.
- The plots separate projected decay-amplitude error from the orthogonal spatial
  shape error. Inspect these before assuming the residual is only a decay error.
- Changing the residual audit or grid is not an experiment on new physical flows.
  We have not evaluated generalization across flow regimes or calibrated a detector.

`results/grid_refinement.json` checks reported final metrics on 64×64 spatial
grids at the same 17 times. This checks spatial quadrature sensitivity, not PDE
solver convergence or temporal quadrature convergence. Saved training seeds,
configuration, software version and checkpoints are in each run directory.

See `results/comparison.png` and each run's `diagnostics.png` and `fields.png`.

Next controlled experiment: compare fixed and resampled collocation with exact
initial conditions, keeping other choices fixed. Reuse development
diagnostics freely for learning; reserve new flows for any later research claims.
