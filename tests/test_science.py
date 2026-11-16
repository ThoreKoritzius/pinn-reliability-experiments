"""Mathematical identities and reproducibility of the committed results."""
import json
import hashlib
from pathlib import Path
import unittest

import torch

from taylor_green import PeriodicMLP, diagnostics, grid, residual, sample, verify


class ScientificChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_default_dtype(torch.float64)
        torch.set_num_threads(1)

    def test_equations_and_constraints(self):
        verify()

    def test_artifact_hashes(self):
        manifest = json.loads(Path("artifact-manifest.json").read_text())
        for name, digest in manifest["sha256"].items():
            with self.subTest(artifact=name):
                self.assertEqual(hashlib.sha256(Path(name).read_bytes()).hexdigest(), digest)

    def test_committed_checkpoint_metrics(self):
        points, _ = grid(2.)
        audit = sample(2048, 2., 12345)
        for path in sorted(Path("results").glob("*/metrics.json")):
            data = json.loads(path.read_text())
            cfg = data["config"]
            from types import SimpleNamespace
            for stage in data["stages"]:
                with self.subTest(run=path.parent.name,stage=stage["name"]):
                    checkpoint = torch.load(path.parent/f"{stage['name']}.pt",weights_only=True)
                    self.assertEqual(checkpoint["config"], cfg)
                    net = PeriodicMLP(cfg["width"],cfg["depth"],cfg["horizon"],cfg.get("hard_ic",False))
                    net.load_state_dict(checkpoint["state_dict"])
                    self.assertEqual(sum(p.numel() for p in net.parameters()), data["parameters"])
                    measured = diagnostics(net,SimpleNamespace(**cfg),audit,points)
                    for key in ("velocity_rel_l2","pressure_rel_l2_centered","audit_momentum_rms"):
                        self.assertAlmostEqual(measured[key],stage["metrics"][key],delta=1e-10)

    def test_matching_initialization_and_points(self):
        for seed in (0,1):
            soft=json.loads(Path(f"results/pinn_seed{seed}/metrics.json").read_text())["config"]
            hard=json.loads(Path(f"results/hard_ic_seed{seed}/metrics.json").read_text())["config"]
            for key in soft.keys()-{"out","hard_ic"}:
                self.assertEqual(soft[key],hard[key])
            torch.manual_seed(seed); a=PeriodicMLP(32,3,2.)
            torch.manual_seed(seed); b=PeriodicMLP(32,3,2.,True)
            for x,y in zip(a.parameters(),b.parameters()):
                self.assertTrue(torch.equal(x,y))


if __name__ == "__main__":
    unittest.main()
