"""Run six-node Cases 1–4 with reward-curve dumps, then plot Figure 7 (Appendix B).

Writes:
  components/offline_simulation/results/cases1_4/.../6 node/reward_curves_{1..10}.npz
  artifacts/figures/offline/Reward_Defender.png
  artifacts/figures/offline/Reward_Attacker.png

Usage:
  python components/offline_simulation/batch/run_reward_curves.py --jobs 4
  python components/offline_simulation/batch/run_reward_curves.py --plot_only
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COMPONENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COMPONENT_ROOT.parents[1]
FIG_DIR = REPO_ROOT / "artifacts" / "figures" / "offline"
NODE_MAX = 6

# Appendix B Cases 1–4
CASES = [
    # (model_CH, driver, scenario_subdir, label, color, linestyle)
    (
        4,
        "main_ch1_salient.py",
        "DQN-CH-level1 def DQN p_expl_att random/new/",
        "<CASE 1> D: CHT-DQN | A: Random",
        "magenta",
        "-",
    ),
    (
        5,
        "main_ch1_salient.py",
        "DQN-CH-level1 def DQN p_expl_att DQN/new/",
        "<CASE 2> D: CHT-DQN | A: DQN",
        "indigo",
        "-",
    ),
    (
        7,
        "main.py",
        "def DQN_att random/new/",
        "<CASE 3> D: DQN | A: Random",
        "lime",
        ":",
    ),
    (
        8,
        "main.py",
        "def DQN_att DQN/new/",
        "<CASE 4> D: DQN | A: DQN",
        "darkgreen",
        ":",
    ),
]


def reward_npz_path(scenario: str, seed: int) -> Path:
    return (
        COMPONENT_ROOT
        / "results/cases1_4"
        / scenario
        / f"{NODE_MAX} node"
        / f"reward_curves_{seed - 100}.npz"
    )


def run_one(driver: str, model_ch: int, seed: int, time_max: int) -> str:
    command = [
        sys.executable,
        "-u",
        str(COMPONENT_ROOT / driver),
        "--my_seed",
        str(seed),
        "--node_max",
        str(NODE_MAX),
        "--model_CH",
        str(model_ch),
        "--time_max",
        str(time_max),
    ]
    label = f"model={model_ch} nodes={NODE_MAX} seed={seed}"
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "MPLBACKEND": "Agg",
        }
    )
    proc = subprocess.run(
        command,
        cwd=COMPONENT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed:\n{proc.stderr[-4000:]}")
    return f"done {label}"


def mean_running_avg(scenario: str, key: str, seed_min: int, seed_max: int) -> np.ndarray:
    curves = []
    for seed in range(seed_min, seed_max):
        path = reward_npz_path(scenario, seed)
        if not path.exists():
            raise FileNotFoundError(path)
        data = np.load(path)
        curves.append(np.asarray(data[key], dtype=np.float64))
    stacked = np.stack(curves, axis=0)
    return np.mean(stacked, axis=0)


def plot_reward_curves(seed_min: int, seed_max: int) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    for key, out_name, ylabel, title in [
        (
            "running_avg_def",
            "Reward_Defender.png",
            "Running average reward (defender)",
            "Convergence Analysis of SOC Analyst (Defender)",
        ),
        (
            "running_avg_att",
            "Reward_Attacker.png",
            "Running average reward (attacker)",
            "Convergence Analysis of Attacker",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(8, 6))
        for _, _, scenario, label, color, ls in CASES:
            y = mean_running_avg(scenario, key, seed_min, seed_max)
            x = np.arange(1, len(y) + 1)
            ax.plot(x, y, color=color, linestyle=ls, linewidth=1.8, label=label)
        ax.axvline(1000, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_xlabel("Time step", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title, fontsize=13)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.35)
        ax.tick_params(axis="both", labelsize=11)
        fig.tight_layout()
        out = FIG_DIR / out_name
        fig.savefig(out, dpi=180)
        plt.close(fig)
        print("Wrote", out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_max", type=int, default=2000)
    parser.add_argument("--seed_min", type=int, default=101)
    parser.add_argument("--seed_max", type=int, default=111)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Re-run even if reward_curves exist")
    parser.add_argument("--plot_only", action="store_true")
    parser.add_argument("--skip_sims", action="store_true")
    args = parser.parse_args()

    if args.plot_only:
        plot_reward_curves(args.seed_min, args.seed_max)
        return

    if not args.skip_sims:
        jobs = []
        for model_ch, driver, scenario, *_ in CASES:
            for seed in range(args.seed_min, args.seed_max):
                path = reward_npz_path(scenario, seed)
                if path.exists() and not args.force:
                    print("Skip existing", path, flush=True)
                    continue
                jobs.append((driver, model_ch, seed, args.time_max))

        print(f"Queued {len(jobs)} runs with {args.jobs} workers", flush=True)
        if not jobs:
            print("Nothing to run.")
        elif args.jobs <= 1:
            for job in jobs:
                print(run_one(*job), flush=True)
        else:
            failed = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
                futures = {pool.submit(run_one, *job): job for job in jobs}
                for fut in concurrent.futures.as_completed(futures):
                    job = futures[fut]
                    try:
                        print(fut.result(), flush=True)
                    except Exception as exc:  # noqa: BLE001
                        failed += 1
                        print(f"FAILED {job}: {exc}", flush=True)
            if failed:
                raise SystemExit(f"{failed} runs failed; see log above")

    plot_reward_curves(args.seed_min, args.seed_max)


if __name__ == "__main__":
    main()
