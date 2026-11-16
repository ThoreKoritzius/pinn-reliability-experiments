"""Run the five documented experiments. Refuse to overwrite existing run folders."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("reproduced"))
    args = parser.parse_args()
    runs = [("pinn_seed0", []), ("pinn_seed1", ["--seed", "1"]),
            ("hard_ic_seed0", ["--hard-ic"]), ("hard_ic_seed1", ["--hard-ic", "--seed", "1"]),
            ("supervised_seed0", ["--mode", "supervised"])]
    if any((args.root/name).exists() for name, _ in runs):
        parser.error("A target run directory already exists; choose a fresh --root")
    for name, options in runs:
        subprocess.run([sys.executable, "taylor_green.py", *options, "--out", str(args.root/name)], check=True)


if __name__ == "__main__":
    main()
