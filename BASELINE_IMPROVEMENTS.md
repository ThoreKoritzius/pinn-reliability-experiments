# Baseline improvement 1: exact initial velocity

The network now supports `--hard-ic`. Its velocity output is

    u_hat(x,y,t) = u_initial(x,y) + (t/T) * network_velocity(x,y,t).

The initial velocity is supplied problem data. No exact future velocity or decay
rate is inserted. Pressure is learned independently. Sine/cosine inputs enforce
spatial periodicity for both formulations. The divergence penalty remains necessary.

## Matched comparison

We matched each new run to an existing soft-IC seed: the same raw network
initialization, 2,403 trainable parameters, collocation and initial-condition points,
float64 precision, viscosity, horizon, Adam learning rate and step count, and
L-BFGS maximum iteration count. Both use 2,000 Adam steps and at most 300 L-BFGS
iterations; line-search objective counts differ slightly. This is an iteration
budget comparison, not a matched wall-clock comparison. The ansatz changes the
initial predicted field and optimization geometry; that is the intervention.

| Seed | Soft IC velocity error | Exact IC velocity error | Soft IC pressure error | Exact IC pressure error |
|---|---:|---:|---:|---:|
| 0 | 2.098% | 0.334% | 6.406% | 2.143% |
| 1 | 1.729% | 0.268% | 5.753% | 1.923% |

Errors are relative L2 over the same 32² × 17 evaluation grid. Pressure is
centered per snapshot. A 64² spatial-grid check agrees to displayed precision.
Both seeds improved by roughly a factor of six in velocity. Two seeds do not
establish broad statistical robustness or performance on other flow regimes.

The mathematical checks assert exact initial velocity at arbitrary sampled
coordinates and finite parameter gradients through the momentum residual.
The initial-condition loss is identically zero with the new ansatz; leaving that
zero term in the loss preserves reporting and does not affect the gradients.

## Visualization quality

The standalone explorer reevaluates checkpoints on a 96² spatial grid at 33
time values. These are actual network evaluations, not interpolation of the old
24² data. Vorticity, dv/dx − du/dy, is computed with autograd. Scalar display
pixels are smoothly scaled for rendering; streamline tracing uses bilinear
interpolation of velocity samples and does not represent a particle trajectory.
Flow and error scales stay fixed over time for each run/stage/quantity. The
optional common error/flow scale permits direct magnitude comparisons.

The denser standalone dataset makes this page larger than the inline explorer.
Static field images are now evaluated at 128² points.

## Reproduce

```bash
python3 taylor_green.py --check
python3 taylor_green.py --hard-ic --seed 0 --out results/hard_ic_seed0
python3 taylor_green.py --hard-ic --seed 1 --out results/hard_ic_seed1
python3 summarize.py
python3 export_explorer.py --standalone --out explorer.fragment.html
```

The next controlled ablation should hold this exact-IC formulation fixed and
compare fixed collocation with fresh/resampled points at matched training cost.
Do not simultaneously increase width, training duration and point count: that
would obscure which change helped. More difficult multi-mode initial states
remain necessary before making a general Navier–Stokes claim.
