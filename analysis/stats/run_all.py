"""Run the paper statistics and the Appendix B protection plot.

  02 Human-study protection outcomes (Table 1)
  03 Human-study mixed-effects model (Table 1)
  04 Human-study reselection tests (Table 1, Figure 3)
  05 Offline protection vs. graph size (Figure 6)

Scripts 02–04 need the restricted human data under restricted_data/hitl/curated/.

Usage (repo root):
  python analysis/stats/run_all.py
  python analysis/stats/run_all.py --skip_hitl   # only script 05
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

HITL_SCRIPTS = [
    "02_hitl_primary_outcomes.py",
    "03_hitl_mixed_effects.py",
    "04_hitl_prospect_tests.py",
]
OFFLINE_SCRIPTS = ["05_plot_offline_protection.py"]


def run(script: str) -> None:
    cmd = [sys.executable, str(HERE / script)]
    print("\n===", " ".join(cmd), "===")
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip_hitl", action="store_true", help="Skip scripts that need restricted human data")
    args = p.parse_args()

    scripts = ([] if args.skip_hitl else HITL_SCRIPTS) + OFFLINE_SCRIPTS
    for script in scripts:
        run(script)
    print(f"\nOutputs: {ROOT / 'artifacts'}")


if __name__ == "__main__":
    main()
