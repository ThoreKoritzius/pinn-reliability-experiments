"""Physics-only Taylor–Green PINN, with exact-solution diagnostics.

Run `python3 taylor_green.py --help`. CPU/float64 is intentional: tiny
second-derivative workloads do not necessarily benefit from GPU execution.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


def exact(z: torch.Tensor, nu: float) -> torch.Tensor:
    """u=sin(x)cos(y)a, v=-cos(x)sin(y)a; gauge p(0,0,t)=0."""
    x, y, t = z.split(1, dim=1)
    a = torch.exp(-2 * nu * t)
    return torch.cat((x.sin() * y.cos() * a,
                      -x.cos() * y.sin() * a,
                      0.25 * ((2*x).cos() + (2*y).cos() - 2) * a.square()), 1)


def derivative(f: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(f.sum(), z, create_graph=True)[0]


def residual(field, points: torch.Tensor, nu: float) -> torch.Tensor:
    """Return x momentum, y momentum, and divergence, in physical coordinates."""
    z = points.detach().clone().requires_grad_(True)
    u, v, p = field(z).split(1, dim=1)
    du, dv, dp = (derivative(f, z) for f in (u, v, p))
    lap_u = derivative(du[:, 0:1], z)[:, 0:1] + derivative(du[:, 1:2], z)[:, 1:2]
    lap_v = derivative(dv[:, 0:1], z)[:, 0:1] + derivative(dv[:, 1:2], z)[:, 1:2]
    ru = du[:, 2:3] + u*du[:, 0:1] + v*du[:, 1:2] + dp[:, 0:1] - nu*lap_u
    rv = dv[:, 2:3] + u*dv[:, 0:1] + v*dv[:, 1:2] + dp[:, 1:2] - nu*lap_v
    return torch.cat((ru, rv, du[:, 0:1] + dv[:, 1:2]), 1)


class PeriodicMLP(nn.Module):
    """Only the domain periodicity is supplied, not the solution's product modes."""
    def __init__(self, width: int, depth: int, horizon: float, hard_ic: bool = False):
        super().__init__()
        self.horizon = horizon
        self.hard_ic = hard_ic
        layers = []
        sizes = [5] + [width] * depth + [3]
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            linear = nn.Linear(a, b)
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            if i < len(sizes)-2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        x, y, t = z.split(1, dim=1)
        output = self.net(torch.cat((x.sin(), x.cos(), y.sin(), y.cos(),
                                    2*t/self.horizon - 1), 1))
        if self.hard_ic:
            initial = torch.cat((x.sin()*y.cos(), -x.cos()*y.sin()), 1)
            return torch.cat((initial + (t/self.horizon)*output[:, :2], output[:, 2:]), 1)
        return output


def sample(n: int, horizon: float, seed: int) -> torch.Tensor:
    # Scrambled Sobol points: independent scrambles for training and auditing.
    points = torch.quasirandom.SobolEngine(3, scramble=True, seed=seed).draw(n).double()
    return points * torch.tensor([2*math.pi, 2*math.pi, horizon])


def grid(horizon: float, n=32, nt=17):
    axis = torch.arange(n) * (2*math.pi/n)  # exclude duplicate periodic endpoint
    ts = torch.linspace(0, horizon, nt)
    t, y, x = torch.meshgrid(ts, axis, axis, indexing="ij")
    return torch.stack((x, y, t), -1).reshape(-1, 3), ts


def diagnostics(model, cfg, audit, eval_points, nt=17, n=32):
    with torch.no_grad():
        pred = model(eval_points).reshape(nt, n*n, 3)
        truth = exact(eval_points, cfg.nu).reshape(nt, n*n, 3)
        error = pred[:, :, :2] - truth[:, :, :2]
        rel_time = (error.square().sum((1, 2)) / truth[:, :, :2].square().sum((1, 2))).sqrt()
        p = pred[:, :, 2] - pred[:, :, 2].mean(1, keepdim=True)
        p_true = truth[:, :, 2] - truth[:, :, 2].mean(1, keepdim=True)
        # Projection onto the exact velocity pattern separates amplitude and shape.
        amplitude_ratio = ((pred[:, :, :2]*truth[:, :, :2]).sum((1, 2)) /
                           truth[:, :, :2].square().sum((1, 2)))
        shape = pred[:, :, :2] - amplitude_ratio[:, None, None]*truth[:, :, :2]
        result = {
            "velocity_rel_l2": float((error.square().sum()/truth[:, :, :2].square().sum()).sqrt()),
            "pressure_rel_l2_centered": float(((p-p_true).square().sum()/p_true.square().sum()).sqrt()),
            "final_velocity_rel_l2": float(rel_time[-1]),
            "velocity_rel_l2_by_time": rel_time.tolist(),
            "energy_by_time": (0.5*pred[:, :, :2].square().sum(2).mean(1)).tolist(),
            "amplitude_ratio_by_time": amplitude_ratio.tolist(),
            "shape_error_by_time": (shape.square().sum((1, 2))/truth[:, :, :2].square().sum((1, 2))).sqrt().tolist(),
        }
    r = residual(model, audit, cfg.nu).detach()
    result.update({"audit_momentum_rms": float(r[:, :2].square().mean().sqrt()),
                   "audit_divergence_rms": float(r[:, 2].square().mean().sqrt())})
    return result


def loss_terms(model, cfg, interior, initial, gauge):
    if cfg.mode == "supervised":
        return {"data": (model(interior)-exact(interior, cfg.nu)).square().mean()}
    r = residual(model, interior, cfg.nu)
    ic = model(initial)[:, :2] - exact(initial, cfg.nu)[:, :2]
    return {"momentum": r[:, :2].square().mean(),
            "divergence": r[:, 2].square().mean(),
            "initial": ic.square().mean(),
            "gauge": model(gauge)[:, 2].square().mean()}


def objective(terms, cfg):
    return sum(value * (cfg.ic_weight if key == "initial" else 1.0)
               for key, value in terms.items())


def plot_run(model, cfg, history, stages, audit, eval_points, ts, out):
    final = stages[-1]["metrics"]
    fig, axs = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    steps = [r["evaluation"] for r in history]
    for key in ("momentum", "divergence", "initial", "gauge", "data"):
        if key in history[0]:
            axs[0, 0].semilogy(steps, [max(r[key], 1e-18) for r in history], label=key)
    axs[0, 0].set(title="Unweighted training losses", xlabel="Objective evaluations")
    axs[0, 0].legend()
    for key, label in (("velocity_rel_l2", "velocity relative L2"),
                       ("pressure_rel_l2_centered", "pressure relative L2"),
                       ("audit_momentum_rms", "fresh momentum RMS")):
        axs[0, 1].semilogy(steps, [r[key] for r in history], label=label)
    axs[0, 1].set(title="Residual and solution error differ", xlabel="Objective evaluations")
    axs[0, 1].legend(fontsize=8)
    for stage in stages:
        axs[0, 2].plot(ts, stage["metrics"]["velocity_rel_l2_by_time"], label=stage["name"])
    axs[0, 2].set(title="Velocity error across time", xlabel="t", ylabel="Relative L2")
    axs[0, 2].legend()
    axs[1, 0].plot(ts, 0.25*np.exp(-4*cfg.nu*ts.numpy()), "k--", label="exact")
    axs[1, 0].plot(ts, final["energy_by_time"], label="prediction")
    axs[1, 0].set(title="Mean kinetic energy", xlabel="t")
    axs[1, 0].legend()
    axs[1, 1].plot(ts, 100*(np.array(final["amplitude_ratio_by_time"])-1), label="projected amplitude error")
    axs[1, 1].plot(ts, 100*np.array(final["shape_error_by_time"]), label="relative shape error")
    axs[1, 1].axhline(0, color="gray", ls=":")
    axs[1, 1].set(title="Decay amplitude versus spatial shape", xlabel="t", ylabel="Error (%)")
    axs[1, 1].legend(fontsize=8)
    r = residual(model, audit, cfg.nu).detach().norm(dim=1).numpy()
    with torch.no_grad():
        err = (model(audit)[:, :2]-exact(audit, cfg.nu)[:, :2]).norm(dim=1).numpy()
    axs[1, 2].scatter(np.maximum(r, 1e-15), np.maximum(err, 1e-15), s=5, alpha=.3)
    axs[1, 2].set(xscale="log", yscale="log", xlabel="Local residual norm",
                  ylabel="Local absolute velocity error", title="Can local residual rank local error?")
    fig.suptitle(f"Taylor–Green | {cfg.mode} | seed {cfg.seed} | ν={cfg.nu} | float64")
    fig.savefig(out/"diagnostics.png", dpi=160)
    plt.close(fig)

    with torch.no_grad():
        field_points, _ = grid(cfg.horizon, n=128, nt=1)
        field_points[:, 2] = cfg.horizon
        pred = torch.cat([model(batch) for batch in field_points.split(4096)]).reshape(128, 128, 3).numpy()
        truth = exact(field_points, cfg.nu).reshape(128, 128, 3).numpy()
    pred[:, :, 2] -= pred[:, :, 2].mean()
    truth[:, :, 2] -= truth[:, :, 2].mean()
    fig, axs = plt.subplots(3, 3, figsize=(11, 9), constrained_layout=True)
    for row, field in enumerate(("u", "v", "p (spatial mean removed)")):
        bound = max(np.abs(truth[:, :, row]).max(), np.abs(pred[:, :, row]).max())
        for col, data in enumerate((truth[:, :, row], pred[:, :, row], pred[:, :, row]-truth[:, :, row])):
            limit = bound if col < 2 else max(np.abs(data).max(), 1e-12)
            im = axs[row, col].imshow(data, origin="lower", extent=(0, 2*math.pi, 0, 2*math.pi),
                                      cmap="RdBu_r", vmin=-limit, vmax=limit)
            axs[row, col].set(title=f"{field}: {('exact', 'prediction', 'signed error')[col]}", xlabel="x", ylabel="y")
            fig.colorbar(im, ax=axs[row, col], shrink=.8)
    fig.suptitle(f"Final time t={cfg.horizon:g}; error panels use their own color scale")
    fig.savefig(out/"fields.png", dpi=160)
    plt.close(fig)


def verify():
    """Check signs, second derivatives, energy, periodicity and pressure gauge."""
    torch.set_default_dtype(torch.float64)
    z = sample(128, 2.0, 77)
    r = residual(lambda q: exact(q, .1), z, .1)
    assert r.abs().max() < 1e-12, r.abs().max()
    points, ts = grid(2.0)
    fields = exact(points, .1).reshape(17, 32*32, 3)
    energy = .5*fields[:, :, :2].square().sum(2).mean(1)
    assert torch.allclose(energy, .25*torch.exp(-.4*ts), atol=1e-13)
    # A time-dependent pressure offset must leave the residual unchanged.
    def shifted(q):
        offset = torch.cat((q[:, :2]*0, q[:, 2:3].sin()), 1)
        return exact(q, .1) + offset
    assert residual(shifted, z, .1).abs().max() < 1e-12
    net = PeriodicMLP(8, 2, 2.0)
    hard = PeriodicMLP(8, 2, 2.0, hard_ic=True)
    initial = z.clone(); initial[:, 2] = 0
    assert torch.equal(hard(initial)[:, :2], exact(initial, .1)[:, :2])
    residual(hard, z, .1).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in hard.parameters())
    for axis in (0, 1):
        a = z.clone(); b = z.clone()
        a[:, axis] = 0; b[:, axis] = 2*math.pi
        assert torch.allclose(net(a), net(b), atol=1e-13)
        assert torch.allclose(residual(net, a, .1), residual(net, b, .1), atol=1e-12)
    # Zero flow is an exact PDE solution but violates the requested initial state.
    def zero(q):
        return torch.cat((q[:, :1]**3*0, q[:, 1:2]**3*0, q[:, 2:3]**3*0), 1)
    assert residual(zero, z, .1).abs().max() == 0
    assert torch.allclose((zero(points)[:, :2]-exact(points, .1)[:, :2]).norm() /
                          exact(points, .1)[:, :2].norm(), torch.tensor(1.0))
    print("PASS: exact PDE residual, energy decay, pressure gauge, periodicity, zero-flow counterexample")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Run mathematical implementation checks only")
    parser.add_argument("--mode", choices=("pinn", "supervised"), default="pinn")
    parser.add_argument("--hard-ic", action="store_true", help="Enforce initial velocity exactly using u0 + (t/T)*NN")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nu", type=float, default=.1)
    parser.add_argument("--horizon", type=float, default=2.0)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument("--initial-points", type=int, default=256)
    parser.add_argument("--adam-steps", type=int, default=2000)
    parser.add_argument("--lbfgs-steps", type=int, default=300, help="Maximum L-BFGS iterations; may stop early")
    parser.add_argument("--ic-weight", type=float, default=10.)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("runs/pinn_seed0"))
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing run's artifacts")
    cfg = parser.parse_args()
    if cfg.check:
        verify(); return
    if (cfg.out/"metrics.json").exists() and not cfg.overwrite:
        parser.error("Run already exists; choose a fresh --out or explicitly pass --overwrite")
    if min(cfg.nu, cfg.horizon, cfg.width, cfg.depth, cfg.points, cfg.initial_points,
           cfg.ic_weight, cfg.lr, cfg.threads) <= 0 or min(cfg.adam_steps, cfg.lbfgs_steps) < 0:
        parser.error("Physical/training sizes and weights must be positive; step counts must be nonnegative")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    cfg.out.mkdir(parents=True, exist_ok=True)
    interior = sample(cfg.points, cfg.horizon, cfg.seed+100)
    initial = sample(cfg.initial_points, cfg.horizon, cfg.seed+200)
    initial[:, 2] = 0
    gauge = torch.zeros(64, 3)
    gauge[:, 2] = torch.linspace(0, cfg.horizon, 64)
    # Audit seed is fixed across experiments; auditing never enters the objective.
    audit = sample(2048, cfg.horizon, 12345)
    eval_points, ts = grid(cfg.horizon)
    model = PeriodicMLP(cfg.width, cfg.depth, cfg.horizon, hard_ic=cfg.hard_ic)
    history, stages = [], []
    started = time.perf_counter()
    evaluations = 0

    def log(stage):
        terms = loss_terms(model, cfg, interior, initial, gauge)
        record = {"stage": stage, "evaluation": evaluations, "seconds": time.perf_counter()-started,
                  "total": float(objective(terms, cfg).detach()),
                  **{k: float(v.detach()) for k, v in terms.items()},
                  **diagnostics(model, cfg, audit, eval_points)}
        # Fixed-sample training residual versus independent audit residual.
        train_r = residual(model, interior, cfg.nu).detach()
        record["train_momentum_rms"] = float(train_r[:, :2].square().mean().sqrt())
        history.append(record)
        print(f"{stage:12s} eval={evaluations:5d} loss={record['total']:.3e} "
              f"vel={record['velocity_rel_l2']:.3%} p={record['pressure_rel_l2_centered']:.3%} "
              f"audit_mom={record['audit_momentum_rms']:.3e}", flush=True)

    def save_stage(name):
        stages.append({"name": name, "metrics": history[-1]})
        torch.save({"state_dict": model.state_dict(), "config": {**vars(cfg), "out": str(cfg.out)}},
                   cfg.out/f"{name}.pt")

    log("initial")
    adam = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    for step in range(1, cfg.adam_steps+1):
        adam.zero_grad(set_to_none=True)
        objective(loss_terms(model, cfg, interior, initial, gauge), cfg).backward()
        adam.step(); evaluations += 1
        if step % 200 == 0 or step == cfg.adam_steps:
            log("adam")
    save_stage("after_adam")
    if cfg.lbfgs_steps:
        optimizer = torch.optim.LBFGS(model.parameters(), max_iter=cfg.lbfgs_steps,
                                     history_size=50, line_search_fn="strong_wolfe",
                                     tolerance_grad=1e-10, tolerance_change=1e-13)
        def closure():
            nonlocal evaluations
            optimizer.zero_grad(set_to_none=True)
            value = objective(loss_terms(model, cfg, interior, initial, gauge), cfg)
            value.backward(); evaluations += 1
            if evaluations % 100 == 0:
                log("lbfgs_trial")  # includes line-search trial points, not accepted iterates
            return value
        optimizer.step(closure)
        log("lbfgs_final")
        save_stage("after_lbfgs")

    # Flatten scalar history for simple inspection; retain temporal arrays in JSON.
    scalars = [{k: v for k, v in row.items() if not isinstance(v, list)} for row in history]
    with (cfg.out/"history.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(scalars[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(scalars)
    result = {"config": {**vars(cfg), "out": str(cfg.out)}, "torch_version": torch.__version__,
              "environment": {"python": sys.version.split()[0], "torch": str(torch.__version__),
                              "numpy": np.__version__, "matplotlib": matplotlib.__version__},
              "dtype": "float64", "device": "cpu", "parameters": sum(p.numel() for p in model.parameters()),
              "seconds": time.perf_counter()-started, "stages": stages, "history": history,
              "times": ts.tolist()}
    (cfg.out/"metrics.json").write_text(json.dumps(result, indent=2)+"\n")
    plot_run(model, cfg, history, stages, audit, eval_points, ts, cfg.out)
    print(f"Saved checkpoints, metrics, history and plots to {cfg.out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
