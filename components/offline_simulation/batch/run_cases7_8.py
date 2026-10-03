"""Run Appendix B Cases 7–8: Double DQN defenders vs. a standard DQN attacker.

  Case 7: CHT–Double DQN | DQN   --model_CH 5 --double_dqn
  Case 8: Double DQN | DQN       --model_CH 8 --double_dqn

Writes under components/offline_simulation/results/cases5_8/.

Example (smoke, 1 seed, 1 graph size):
  python batch/run_cases7_8.py --seed_min 101 --seed_max 102 \\
      --node_min 6 --node_max 7 --time_max 200 --jobs 2

Full sweep:
  python batch/run_cases7_8.py --jobs 4
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


# (name, model_CH, driver, double_dqn, use_dueling)
CASES = [
    ("cht_double_dqn", 5, "main_ch1_salient.py", True, False),
    ("double_dqn", 8, "main.py", True, False),
]


def scenario_subdir(model_ch: int, double_dqn: bool, use_dueling: bool) -> str:
    if model_ch == 5:
        base = "DQN-CH-level1 def DQN p_expl_att DQN"
    elif model_ch == 8:
        base = "def DQN_att DQN"
    else:
        raise ValueError(model_ch)
    tags = []
    if double_dqn:
        tags.append("double")
    if use_dueling:
        tags.append("dueling")
    if tags:
        base = f"{base}_{'_'.join(tags)}"
    return f"{base}/new/"


def out_txt(model_ch: int, node_max: int, seed: int, double_dqn: bool, use_dueling: bool) -> Path:
    return (
        COMPONENT_ROOT
        / "results/cases5_8"
        / scenario_subdir(model_ch, double_dqn, use_dueling)
        / f"{node_max} node"
        / f"{seed - 100}.txt"
    )


def run_one(
    driver: str,
    model_ch: int,
    node_max: int,
    seed: int,
    time_max: int,
    double_dqn: bool,
    use_dueling: bool,
) -> str:
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
    ]
    if double_dqn:
        command.append("--double_dqn")
    if use_dueling:
        command.append("--use_dueling")

    label = (
        f"model={model_ch} double={double_dqn} dueling={use_dueling} "
        f"nodes={node_max} seed={seed}"
    )
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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed_min", type=int, default=101)
    p.add_argument("--seed_max", type=int, default=111, help="Exclusive upper bound (paper uses 101–110).")
    p.add_argument("--node_min", type=int, default=2)
    p.add_argument("--node_max", type=int, default=11, help="Exclusive upper bound (paper uses 2–10).")
    p.add_argument("--time_max", type=int, default=2000)
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--skip_existing", action="store_true", default=True)
    p.add_argument("--force", action="store_true", help="Re-run even if output exists.")
    p.add_argument(
        "--baselines",
        nargs="*",
        default=None,
        help="Subset of case names (default: both).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    baselines = list(CASES)
    if args.baselines:
        wanted = set(args.baselines)
        baselines = [b for b in baselines if b[0] in wanted]
        missing = wanted - {b[0] for b in baselines}
        if missing:
            raise SystemExit(f"Unknown baselines: {sorted(missing)}")

    jobs = []
    for name, model_ch, driver, double_dqn, use_dueling in baselines:
        for node_max in range(args.node_min, args.node_max):
            for seed in range(args.seed_min, args.seed_max):
                path = out_txt(model_ch, node_max, seed, double_dqn, use_dueling)
                if path.exists() and args.skip_existing and not args.force:
                    continue
                jobs.append((name, driver, model_ch, node_max, seed, double_dqn, use_dueling))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = OUT_ROOT / "cases7_8_jobs.txt"
    with open(manifest, "w") as f:
        for j in jobs:
            f.write("\t".join(map(str, j)) + "\n")
    print(f"{len(jobs)} jobs queued (manifest: {manifest})")

    if not jobs:
        print("Nothing to run.")
        return

    def _work(job):
        name, driver, model_ch, node_max, seed, double_dqn, use_dueling = job
        return run_one(driver, model_ch, node_max, seed, args.time_max, double_dqn, use_dueling)

    ok, fail = 0, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = {ex.submit(_work, j): j for j in jobs}
        for fut in concurrent.futures.as_completed(futs):
            try:
                print(fut.result())
                ok += 1
            except Exception as exc:  # noqa: BLE001
                print(f"FAILED {futs[fut]}: {exc}", file=sys.stderr)
                fail += 1
    print(f"Finished ok={ok} fail={fail}")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
