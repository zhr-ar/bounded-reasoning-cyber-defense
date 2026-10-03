"""Run Appendix B Cases 1–4 (models 4, 5, 7, 8) into results/cases1_4/ and export
per-seed protection tables read by analysis/stats/05_plot_offline_protection.py."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


COMPONENT_ROOT = Path(__file__).resolve().parents[1]

CASES = [
    # (model_CH, driver, scenario_subdir, xlsx_stem)
    (4, "main_ch1_salient.py", "DQN-CH-level1 def DQN p_expl_att random/new/", "data_prot_littman_def_DQN_p_expl_att_random"),
    (5, "main_ch1_salient.py", "DQN-CH-level1 def DQN p_expl_att DQN/new/", "data_prot_littman_def_DQN_p_expl_att_DQN"),
    (7, "main.py", "def DQN_att random/new/", "data_prot_DQN_att_random"),
    (8, "main.py", "def DQN_att DQN/new/", "data_prot_def_DQN_att_DQN"),
]


def out_txt_path(model_ch: int, node_max: int, seed: int) -> Path:
    scenario = [c[2] for c in CASES if c[0] == model_ch][0]
    return COMPONENT_ROOT / "results/cases1_4" / scenario / f"{node_max} node" / f"{seed - 100}.txt"


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
    ]
    label = f"model={model_ch} nodes={node_max} seed={seed}"
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


def export_protection_xlsx(scenario: str, out_name: str, seed_min: int, seed_max: int) -> Path:
    data = {}
    for node_max in range(2, 11):
        vals = []
        for seed in range(seed_min, seed_max):
            path = COMPONENT_ROOT / "results/cases1_4" / scenario / f"{node_max} node" / f"{seed - 100}.txt"
            with open(path, "r") as f:
                last = f.readlines()[-3:]
            # Use full-horizon average (first of the three summary lines), as plotted in Figure 6.
            vals.append(float(last[0].strip().split()[-1]))
        data[node_max] = vals
    df = pd.DataFrame(data)
    df.index = range(seed_min, seed_max)
    out = COMPONENT_ROOT / "results/cases1_4" / f"{out_name}.xlsx"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_max", type=int, default=2000)
    parser.add_argument("--seed_min", type=int, default=101)
    parser.add_argument("--seed_max", type=int, default=111)
    parser.add_argument("--node_min", type=int, default=2)
    parser.add_argument("--node_max", type=int, default=10)
    parser.add_argument("--jobs", type=int, default=4, help="Parallel simulation workers")
    parser.add_argument("--skip_sims", action="store_true", help="Only re-export the protection tables")
    args = parser.parse_args()

    if not args.skip_sims:
        jobs = []
        for model_ch, driver, _, _ in CASES:
            for node in range(args.node_min, args.node_max + 1):
                for seed in range(args.seed_min, args.seed_max):
                    if out_txt_path(model_ch, node, seed).exists():
                        print("Skip existing", out_txt_path(model_ch, node, seed), flush=True)
                        continue
                    jobs.append((driver, model_ch, node, seed, args.time_max))

        print(f"Queued {len(jobs)} runs with {args.jobs} workers", flush=True)
        if args.jobs <= 1:
            for job in jobs:
                print(run_one(*job), flush=True)
        else:
            # Thread pool is safer here: each job already launches an isolated
            # subprocess. ProcessPool + PyTorch has caused silent early exits.
            failed = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
                futures = {pool.submit(run_one, *job): job for job in jobs}
                for fut in concurrent.futures.as_completed(futures):
                    job = futures[fut]
                    try:
                        print(fut.result(), flush=True)
                    except Exception as exc:
                        failed += 1
                        print(f"FAILED {job}: {exc}", flush=True)
            if failed:
                raise SystemExit(f"{failed} runs failed; see log above")

    for _, _, scenario, stem in CASES:
        print("Wrote", export_protection_xlsx(scenario, stem, args.seed_min, args.seed_max))


if __name__ == "__main__":
    main()
