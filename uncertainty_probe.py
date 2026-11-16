"""Exploratory two-member uncertainty audit; no calibrated coverage guarantee."""
import json
from pathlib import Path

import numpy as np
import torch

from taylor_green import PeriodicMLP, exact, residual, sample


def rank_correlation(x, y):
    # Values here are continuous; no materially tied observations are expected.
    return float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    z = sample(4096, 2.0, 90210)
    truth = exact(z, .1)[:, :2]
    result = {"members": 2, "points": len(z), "note": "Uncalibrated diagnostic on one fixed PDE, not independent calibration/test cases", "stages": {}}
    for stage in ("after_adam", "after_lbfgs"):
        nets = []
        for seed in (0, 1):
            ckpt = torch.load(f"results/pinn_seed{seed}/{stage}.pt", weights_only=True)
            c = ckpt["config"]
            assert c["nu"] == .1 and c["horizon"] == 2.0
            net = PeriodicMLP(c["width"], c["depth"], c["horizon"])
            net.load_state_dict(ckpt["state_dict"])
            nets.append(net)
        with torch.no_grad():
            predictions = torch.stack([net(z)[:, :2] for net in nets])
            mean = predictions.mean(0)
            std = predictions.std(0, unbiased=True)
            error = (mean-truth).norm(dim=1)
            spread = std.norm(dim=1)
            covered = ((mean-truth).abs() <= 1.96*std).double().mean(0)
        # Audit the mean of network residual magnitudes, not the nonlinear residual
        # of the ensemble mean (those are different for Navier–Stokes).
        scores = torch.stack([residual(net, z, .1).detach().norm(dim=1) for net in nets]).mean(0)
        m = {"mean_velocity_relative_l2": float((mean-truth).norm()/truth.norm()),
             "ensemble_spread_vs_error_spearman": rank_correlation(spread.numpy(), error.numpy()),
             "mean_member_residual_vs_mean_error_spearman": rank_correlation(scores.numpy(), error.numpy()),
             "uncalibrated_1.96std_component_coverage": {"u": float(covered[0]), "v": float(covered[1])}}
        result["stages"][stage] = m
    # A whole family of exact PDE solutions has the wrong initial amplitude.
    witness = []
    for amplitude in (0., .5, 1., 1.5):
        def candidate(q):
            f = exact(q, .1)
            return torch.cat((amplitude*f[:, :2], amplitude**2*f[:, 2:]), 1)
        r = residual(candidate, z, .1).detach()
        with torch.no_grad():
            rel = (candidate(z)[:, :2]-truth).norm()/truth.norm()
        assert r.abs().max() < 1e-12
        assert abs(float(rel)-abs(1-amplitude)) < 1e-12
        witness.append({"amplitude": amplitude, "max_abs_residual": float(r.abs().max()), "relative_velocity_error": float(rel)})
    result["exact_wrong_initial_state_witnesses"] = witness
    Path("results/uncertainty_probe.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
