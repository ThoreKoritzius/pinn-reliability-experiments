"""Pack saved checkpoint evaluations into a compact, offline visualization fragment."""
import argparse
import base64
import json
import math
from pathlib import Path

import numpy as np
import torch

from taylor_green import PeriodicMLP, exact, derivative


def pack(values):
    values = np.asarray(values)
    scale = max(float(np.abs(values).max()), 1e-12)/32767
    encoded = np.rint(values/scale).astype("<i2")
    return {"scale": scale, "data": base64.b64encode(encoded.tobytes()).decode()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--standalone", action="store_true", help="Build an offline page that opens directly from file://")
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--frames", type=int, default=33)
    args = parser.parse_args()
    if args.resolution < 4 or args.frames < 2:
        parser.error("resolution must be at least 4 and frames at least 2")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    n = args.resolution if args.standalone else 24
    axis = (torch.arange(n)+.5)*(2*math.pi/n)
    runs = []
    for folder in sorted(Path("results").glob("*/metrics.json")):
        saved = json.loads(folder.read_text())
        cfg = saved["config"]
        if cfg["nu"] != .1 or cfg["horizon"] != 2.:
            raise ValueError("This explorer currently compares only nu=0.1, T=2 runs")
        ts = torch.linspace(0, cfg["horizon"], args.frames) if args.standalone else torch.tensor(saved["times"])
        t, y, x = torch.meshgrid(ts, axis, axis, indexing="ij")
        coords = torch.stack((x, y, t), -1).reshape(-1, 3)
        with torch.no_grad():
            target = exact(coords, cfg["nu"]).reshape(len(ts), n*n, 3)
            target[:, :, 2] -= target[:, :, 2].mean(1, keepdim=True)
        run = {"id": folder.parent.name, "label": folder.parent.name.replace("pinn_seed", "Soft IC · seed ").replace("supervised_seed", "Supervised · seed ").replace("hard_ic_seed", "Exact IC · seed "),
               "nu": cfg["nu"], "horizon": cfg["horizon"], "times": ts.tolist(), "stages": {},
               "history": [{k: row[k] for k in ("evaluation", "velocity_rel_l2", "audit_momentum_rms", "train_momentum_rms")}
                           for row in saved["history"]]}
        for stage in saved["stages"]:
            net = PeriodicMLP(cfg["width"], cfg["depth"], cfg["horizon"], hard_ic=cfg.get("hard_ic", False))
            checkpoint = torch.load(folder.parent/f"{stage['name']}.pt", weights_only=True)
            net.load_state_dict(checkpoint["state_dict"])
            with torch.no_grad():
                pred = torch.cat([net(batch) for batch in coords.split(8192)]).reshape(len(ts), n*n, 3)
                pred[:, :, 2] -= pred[:, :, 2].mean(1, keepdim=True)
                e = pred-target
                vel_norm = target[:, :, :2].square().sum((1, 2)).sqrt()
                err_norm = e[:, :, :2].square().sum((1, 2)).sqrt()
                p_norm = target[:, :, 2].square().sum(1).sqrt()
                p_err = e[:, :, 2].square().sum(1).sqrt()
                stats = {"velocity": (err_norm/vel_norm).tolist(),
                         "pressure": (p_err/p_norm).tolist(),
                         "energy": (.5*pred[:, :, :2].square().sum(2).mean(1)).tolist(),
                         "velocity_error_norm": err_norm.tolist(), "velocity_reference_norm": vel_norm.tolist(),
                         "pressure_error_norm": p_err.tolist(), "pressure_reference_norm": p_norm.tolist()}
            omega = []
            for batch in coords.split(4096):
                q = batch.detach().clone().requires_grad_(True)
                f = net(q)
                omega.append((derivative(f[:, 1:2], q)[:, 0]-derivative(f[:, 0:1], q)[:, 1]).detach())
            pred = torch.cat((pred, torch.cat(omega).reshape(len(ts), n*n, 1)), 2)
            # Vorticity is differentiated by autograd, not by a display-grid stencil.
            fields = [pack(pred[:, :, j].numpy()) for j in range(4)]
            # Check quantization error stays bounded by half a quantization step.
            for j, packed in enumerate(fields):
                decoded = np.frombuffer(base64.b64decode(packed["data"]), dtype="<i2")*packed["scale"]
                assert np.max(np.abs(decoded-pred[:, :, j].numpy().ravel())) <= packed["scale"]*.501
            run["stages"][stage["name"]] = {"fields": fields, "stats": stats,
                                            "global_velocity": stage["metrics"]["velocity_rel_l2"]}
        runs.append(run)
    if not runs:
        raise SystemExit("No checkpoint results found")
    payload = {"version": 1, "n": n, "runs": runs}
    source = Path("templates/explorer.fragment.html").read_text()
    assert source.count("__TAYLOR_GREEN_DATA__") == 1
    source = source.replace("__GRID_LABEL__", f"{n} × {n} evaluated grid · {len(ts)} evaluated times")
    output = source.replace("__TAYLOR_GREEN_DATA__", json.dumps(payload, separators=(",", ":")))
    if not args.standalone:
        assert len(output.encode()) < 1_000_000
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.standalone:
        start = output.index("try {\n  const {mountChart}")
        output = output[:start] + Path("templates/standalone-chart.js").read_text() + "\n</script>\n"
        fragment = Path("results/explorer-data.html").resolve()
        fragment.write_text(output)
        shell = Path("templates/standalone.html").read_text()
        args.out.write_text(shell.replace("<!-- EXPLORER_CONTENT -->", output))
    else:
        args.out.write_text(output)
    print(f"Exported {len(runs)} runs × 2 checkpoints × {len(ts)} times at {n}²; {len(output.encode()):,} bytes to {args.out}")


if __name__ == "__main__":
    main()
