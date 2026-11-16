# From a correct PDE to a useful uncertainty estimate

## Zero residual is not the same as solving the requested initial-value problem

Zero velocity and spatially constant pressure satisfy the differential equations
and periodic boundary conditions. They violate our nonzero initial velocity.
The relative velocity error is exactly ||0-u||/||u||=1.
Our implemented loss already includes the initial condition, so zero flow does
not minimize the full loss. The pressure-gauge constraint does not remove zero
flow; it only fixes the arbitrary additive pressure offset.

More generally, scale exact Taylor–Green velocity by A and its pressure by A².
That solves the same PDE with a different initial amplitude. For the requested
amplitude one, its relative velocity error is |A-1| despite zero PDE residual.
`uncertainty_probe.py` checks A=0, 0.5, 1 and 1.5 by automatic differentiation.

The complete specification needs momentum equations, incompressibility, initial
velocity, spatial periodicity and a pressure gauge. These are all present in our
experiment; periodicity is built into the network rather than penalized.

## Improve the solution before adding uncertainty machinery

A useful controlled intervention is exact initial-condition enforcement:

    velocity(x,y,t) = initial_velocity(x,y) + (t/T) * network_velocity(x,y,t).

With periodic network inputs this enforces both the supplied initial velocity
and periodicity. It uses only known initial data, not the exact future solution.
It rules out the zero-flow candidate but does not eliminate optimization,
finite-sampling or approximation error. Pressure remains an independent output
with its gauge. This is now implemented via `--hard-ic`: see
`BASELINE_IMPROVEMENTS.md` for the matched two-seed comparison, which reduced
velocity error from about 1.7–2.1% to 0.27–0.33%.

For soft constraints, inspect individual loss terms and their gradient scales,
and vary the IC weight with initialization and points fixed. Adding more
penalties is not automatically beneficial: energy balance can be a useful
independent audit before it becomes another training objective.

## Initial measured uncertainty diagnostic

The target here is numerical/learning error for a deterministic problem with
known inputs, not observational noise or a distribution over physical viscosities.

We evaluated the two existing PINN checkpoints on 4,096 fresh Sobol points with
scramble seed 90210, excluding the supervised fit. At every coordinate, form the
ensemble mean and sample standard deviation of u and v separately. These two
networks differ in initialization and training-point scramble.

After L-BFGS:

- Ensemble mean velocity relative L2 error: about 1.82%.
- Spearman rank correlation of vector ensemble spread with vector error: 0.20.
- The uncalibrated mean ± 1.96 sample-standard-deviation bands cover about 51%
  of exact u values and 58% of exact v values on those sampled points.
- Mean member local residual magnitude has rank correlation about -0.30 with
  ensemble-mean velocity error. This particular local score does not rank the
  errors usefully on this run and point distribution.

The 1.96 multiplier is a deliberately naive baseline, not a justified 95%
interval. Two members do not reliably estimate an uncertainty distribution, and
different networks can share systematic bias. These are descriptive spatial
statistics on one PDE, not independent trials or evidence of a general failure
of ensembles/residual methods. Exact values are in `results/uncertainty_probe.json`.

## A defensible next research experiment

1. Train 5–10 independent PINNs per fixed problem, with an explicit ensemble
   construction. Separate initialization changes from point-sampling changes.
2. Compare ensemble spread, fresh residual (including IC/BC checks), and their
   combinations as error-ranking scores. Residuals have differential units, so
   normalize consistently before combining unlike quantities.
3. Use exact solutions to evaluate the scores; do not expose exact test errors
   when constructing the deployed score.
4. For scalar component intervals, use split conformal calibration with scores
   |truth-mean|/max(spread, epsilon). Fit scale choices before calibration, choose
   the finite-sample calibration quantile, and evaluate on untouched samples.
   IID randomly drawn coordinate samples can support a marginal coordinate-level
   experiment conditional on a fixed trained model. A deterministic grid or Sobol
   design does not automatically meet the ordinary exchangeability assumptions.
5. Do not equate marginal pointwise coverage with coverage of a whole flow field.
   For a run-level reliability claim, calibrate/test on independent whole problems
   or independent appropriately defined runs, using a run-level failure metric.
6. Extend the exact family with amplitude, viscosity, phase and higher spatial
   wavenumber. Hold out full configurations. Higher wavenumbers test more spatial
   structure; Taylor–Green still lacks nonlinear mode interactions. Treat shifted
   test regimes as stress tests, not as automatically covered by conformal theory.
7. Report coverage versus interval width and accepted-run accuracy versus
   acceptance rate. A method that rejects everything or returns huge intervals
   is not practically informative.

Residual-space coverage does not directly give velocity-error coverage. Consult
[Gopakumar et al.](https://arxiv.org/html/2502.04406v2), especially their limitations,
[Yu et al.](https://arxiv.org/abs/2509.13717) for labelled conformal calibration,
and [De Ryck et al.](https://arxiv.org/abs/2203.09346) for Navier–Stokes error bounds.

The immediate question is: **does disagreement track actual error, and can
calibration correct its overconfidence on a specified test distribution?**
Transfer of that correction to new problem configurations is the next question.
