"""Batch runner for the two web-matched automated scenarios."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np


COMPONENT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    my_seed_range = np.arange(101, 141, 1)
    failures = 0

    for model_ch in (8,):
        for node_max in (6,):
            for my_seed in my_seed_range:
                cmd = [
                    sys.executable,
                    str(COMPONENT_ROOT / "main.py"),
                    "--my_seed",
                    str(my_seed),
                    "--node_max",
                    str(node_max),
                    "--model_CH",
                    str(model_ch),
                ]
                print("Running", cmd)
                completed = subprocess.run(cmd, cwd=COMPONENT_ROOT, check=False)
                if completed.returncode != 0:
                    failures += 1
                    print(f"FAILED ({completed.returncode}):", cmd)

    for model_ch in (5,):
        for node_max in (6,):
            for my_seed in my_seed_range:
                cmd = [
                    sys.executable,
                    str(COMPONENT_ROOT / "main_ch1.py"),
                    "--my_seed",
                    str(my_seed),
                    "--node_max",
                    str(node_max),
                    "--model_CH",
                    str(model_ch),
                ]
                print("Running", cmd)
                completed = subprocess.run(cmd, cwd=COMPONENT_ROOT, check=False)
                if completed.returncode != 0:
                    failures += 1
                    print(f"FAILED ({completed.returncode}):", cmd)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
