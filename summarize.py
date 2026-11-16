"""Rebuild plots and a result note from saved runs; check metric grid refinement."""
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from taylor_green import PeriodicMLP, diagnostics, grid, plot_run, sample


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    files = sorted(Path("results").glob("*/metrics.json"))
    if not files:
        raise SystemExit("No runs found. Run taylor_green.py first.")
    rows, refinements = [], {}
    fig, axs = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for path in files:
        data = json.loads(path.read_text())
        cfg = SimpleNamespace(**data["config"])
        folder = path.parent
        model = PeriodicMLP(cfg.width, cfg.depth, cfg.horizon, hard_ic=getattr(cfg, "hard_ic", False))
        model.load_state_dict(torch.load(folder/f"{data['stages'][-1]['name']}.pt", weights_only=True)["state_dict"])
        points, ts = grid(cfg.horizon)
        audit = sample(2048, cfg.horizon, 12345)
        plot_run(model, cfg, data["history"], data["stages"], audit, points, ts, folder)
        refined, _ = grid(cfg.horizon, n=64)
        metrics64 = diagnostics(model, cfg, audit, refined, n=64)
        final = data["stages"][-1]["metrics"]
        refinements[folder.name] = {
            key: {"32_grid": final[key], "64_grid": metrics64[key]}
            for key in ("velocity_rel_l2", "pressure_rel_l2_centered")}
        for stage in data["stages"]:
            m = stage["metrics"]
            rows.append(f"| {folder.name} | {stage['name']} | {100*m['velocity_rel_l2']:.3f}% | "
                        f"{100*m['pressure_rel_l2_centered']:.3f}% | {m['train_momentum_rms']:.5f} | "
                        f"{m['audit_momentum_rms']:.5f} |")
        history = data["history"]
        axs[0].semilogy([r["evaluation"] for r in history],
                        [100*r["velocity_rel_l2"] for r in history], label=folder.name)
        axs[1].plot(ts, 100*np.array(final["velocity_rel_l2_by_time"]), label=folder.name)
        axs[2].plot(ts, final["energy_by_time"], label=folder.name)
    axs[0].set(title="Solution accuracy during optimization", xlabel="Objective evaluations",
               ylabel="Velocity relative L2 (%)")
    axs[1].set(title="Final model: accuracy across time", xlabel="t", ylabel="Velocity relative L2 (%)")
    # All default runs have the same physical setup. Use exact curves for each
    # distinct setup if the user later adds runs with different nu/horizon.
    setups = {(json.loads(p.read_text())["config"]["nu"],
               json.loads(p.read_text())["config"]["horizon"]) for p in files}
    for nu, horizon in sorted(setups):
        t = np.linspace(0, horizon, 100)
        axs[2].plot(t, .25*np.exp(-4*nu*t), "k--", label=f"exact ν={nu:g}, T={horizon:g}")
    axs[2].set(title="Energy decay", xlabel="t", ylabel="Mean kinetic energy")
    for ax in axs:
        ax.legend(fontsize=7)
    fig.suptitle("Initial Taylor–Green experiments — supervised fitting uses extra labels")
    fig.savefig("results/comparison.png", dpi=170)
    plt.close(fig)
    Path("results/grid_refinement.json").write_text(json.dumps(refinements, indent=2)+"\n")
    note = """# First Taylor–Green results

These are exploratory runs, not a benchmark or a publication claim. Relative L2
errors aggregate a uniform 32×32 spatial grid at 17 equally spaced times. Pressure
is centered separately at each time. RMS residuals and relative solution errors
are different quantities and their numerical magnitudes are not interchangeable.

| Run | Stage | Velocity relative L2 | Pressure relative L2 | Train momentum RMS | Audit momentum RMS |
|---|---|---:|---:|---:|---:|
""" + "\n".join(rows) + """

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
"""
    Path("RESULTS.md").write_text(note)
    print("Updated RESULTS.md, comparison plot, per-run plots and spatial grid check.")
    for name, values in refinements.items():
        print(name, values)


if __name__ == "__main__":
    main()
