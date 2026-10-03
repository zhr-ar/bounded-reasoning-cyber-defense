"""Run Appendix B Cases 5–6: Double DQN defenders vs. a Double DQN attacker.

Cases (both sides use Double DQN bootstrap):
  - Double DQN | Double DQN     --model_CH 8 --double_dqn --double_dqn_att
  - CHT–Double DQN | Double DQN --model_CH 5 --double_dqn --double_dqn_att

Writes under results/cases5_8/ with folder tags `_double_att_double`.

Example:
  python batch/run_cases5_6.py --jobs 4
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path

COMPONENT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = COMPONENT_ROOT / "results/cases5_8"

CASES = [
    # name, model_CH, driver
    ("double_vs_double", 8, "main.py"),
    ("cht_double_vs_double", 5, "main_ch1_salient.py"),
]


def scenario_subdir(model_ch: int) -> str:
    if model_ch == 5:
        return "DQN-CH-level1 def DQN p_expl_att DQN_double_att_double/new/"
    if model_ch == 8:
        return "def DQN_att DQN_double_att_double/new/"
    raise ValueError(model_ch)


def out_txt(model_ch: int, node_max: int, seed: int) -> Path:
    return (
        COMPONENT_ROOT
        / "results/cases5_8"
        / scenario_subdir(model_ch)
        / f"{node_max} node"
        / f"{seed - 100}.txt"
    )


def run_one(driver: str, model_ch: int, node_max: int, seed: int, time_max: int) -> str:
    command = [
        sys.executable,
        "-u",
        str(COMPONENT_ROOT / driver),
        "--my_seed",
        str(seed),
        "--node_max",
        str(node_max),
        "--model_CH",
        str(model_ch),
        "--time_max",
        str(time_max),
        "--results_root",
        "results/cases5_8",
        "--double_dqn",
        "--double_dqn_att",
    ]
    label = f"model={model_ch} double|double nodes={node_max} seed={seed}"
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed_min", type=int, default=101)
    p.add_argument("--seed_max", type=int, default=111)
    p.add_argument("--node_min", type=int, default=2)
    p.add_argument("--node_max", type=int, default=11)
    p.add_argument("--time_max", type=int, default=2000)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    jobs = []
    for name, model_ch, driver in CASES:
        for node in range(args.node_min, args.node_max):
            for seed in range(args.seed_min, args.seed_max):
                path = out_txt(model_ch, node, seed)
                if path.exists() and not args.force:
                    continue
                jobs.append((name, driver, model_ch, node, seed))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} jobs queued", flush=True)
    if not jobs:
        print("Nothing to run.")
        return

    def _work(job):
        name, driver, model_ch, node, seed = job
        return run_one(driver, model_ch, node, seed, args.time_max)

    ok, fail = 0, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = {ex.submit(_work, j): j for j in jobs}
        for fut in concurrent.futures.as_completed(futs):
            try:
                print(fut.result(), flush=True)
                ok += 1
            except Exception as exc:  # noqa: BLE001
                print(f"FAILED {futs[fut]}: {exc}", file=sys.stderr, flush=True)
                fail += 1
    print(f"Finished ok={ok} fail={fail}", flush=True)
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
