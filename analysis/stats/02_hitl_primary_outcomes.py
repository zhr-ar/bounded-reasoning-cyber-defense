"""02 — HITL primary outcomes: participant-level protection + inferential tests.

Primary outcome analysis:
  - participant as independent unit
  - overall + four 10-round stages
  - Welch t-test + Mann–Whitney + permutation robustness
  - bootstrap 95% CI on mean difference
  - Hedges' g / Cliff's delta
  - Benjamini–Hochberg FDR on stage-wise tests

Usage (from repo root):
  python analysis/stats/02_hitl_primary_outcomes.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "analysis" / "hitl"))
from scenarios import scenarios  # noqa: E402
from stats_common import (  # noqa: E402
    benjamini_hochberg,
    cliffs_delta,
    hedges_g,
    mannwhitney,
    mean_ci,
    permutation_mean_diff,
    welch_ttest,
)

OUT = ROOT / "artifacts" / "hitl_stats"
STAGES = [("overall", None), ("R1-10", (0, 10)), ("R11-20", (10, 20)), ("R21-30", (20, 30)), ("R31-40", (30, 40))]
HUMAN = ("cht_human_vs_dqn_attacker", "dqn_human_vs_dqn_attacker")
LABELS = {
    "cht_human_vs_dqn_attacker": "transition_aware",
    "dqn_human_vs_dqn_attacker": "reward_aware",
}


def load_participants(folder: str):
    rows = []
    for path in sorted(Path(folder).glob("*.json")):
        data = json.loads(path.read_text())
        hist = np.asarray(data.get("data_prot_history", []), dtype=float)
        if len(hist) == 0:
            continue
        # Prefer true weighted history if stored as node vectors.
        if hist.ndim == 2:
            data_hist = np.asarray(data.get("data_history", [[1.0] * hist.shape[1]] * len(hist)), dtype=float)
            weighted = np.sum(hist * data_hist, axis=1) / np.maximum(np.sum(data_hist, axis=1), 1e-12)
        else:
            weighted = hist
        rows.append(
            {
                "participant_id": path.stem,
                "path": str(path),
                "n_rounds": int(len(weighted)),
                "avg_protection": float(np.mean(weighted)),
                "score": data.get("score"),
                "elapsed_time": data.get("elapsed_time"),
                "protection_by_round": weighted.tolist(),
                "complete_40": len(weighted) >= 40,
            }
        )
    return rows


def stage_mean(prot: list[float], sl):
    if sl is None:
        return float(np.mean(prot))
    a, b = sl
    return float(np.mean(prot[a:b])) if len(prot) >= b else float("nan")


def compare_conditions(a_vals, b_vals, label_a, label_b, outcome):
    a = np.asarray(a_vals, dtype=float)
    b = np.asarray(b_vals, dtype=float)
    diff = float(np.mean(a) - np.mean(b)) if len(a) and len(b) else float("nan")
    rng = np.random.default_rng(0)
    n_boot = 5000
    diffs = []
    for _ in range(n_boot):
        diffs.append(
            np.mean(rng.choice(a, size=len(a), replace=True))
            - np.mean(rng.choice(b, size=len(b), replace=True))
        )
    diffs = np.asarray(diffs)
    ci_low, ci_high = np.quantile(diffs, [0.025, 0.975])
    welch = welch_ttest(a, b)
    mw = mannwhitney(a, b)
    perm = permutation_mean_diff(a, b)
    return {
        "outcome": outcome,
        "condition_a": label_a,
        "condition_b": label_b,
        "n_a": len(a),
        "n_b": len(b),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "diff_a_minus_b": diff,
        "boot_ci_low": float(ci_low),
        "boot_ci_high": float(ci_high),
        "welch_stat": welch["stat"],
        "welch_p": welch["pvalue"],
        "mannwhitney_p": mw["pvalue"],
        "permutation_p": perm["pvalue"],
        "hedges_g": hedges_g(a, b),
        "cliffs_delta": cliffs_delta(a, b),
        "mean_ci_a": mean_ci(a),
        "mean_ci_b": mean_ci(b),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include_incomplete", action="store_true", help="Include sessions with <40 rounds.")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    by_cond = {}
    for key in HUMAN:
        parts = load_participants(scenarios[key])
        if not args.include_incomplete:
            n_before = len(parts)
            parts = [p for p in parts if p["complete_40"]]
            print(f"{LABELS[key]}: kept {len(parts)}/{n_before} complete-40 participants")
        by_cond[key] = parts

    # Participant table
    part_rows = []
    for key, parts in by_cond.items():
        for p in parts:
            row = {
                "condition": LABELS[key],
                "participant_id": p["participant_id"],
                "n_rounds": p["n_rounds"],
                "score": p["score"],
                "elapsed_time": p["elapsed_time"],
                "avg_protection": p["avg_protection"],
            }
            for name, sl in STAGES:
                row[f"protection_{name}"] = stage_mean(p["protection_by_round"], sl)
            part_rows.append(row)

    part_csv = OUT / "02_hitl_participant_outcomes.csv"
    with open(part_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in part_rows for k in r}))
        w.writeheader()
        w.writerows(part_rows)
    print(f"Wrote {part_csv}")

    # Inferential comparisons transition vs reward
    a_key, b_key = HUMAN
    comparisons = []
    stage_pvals = []
    for name, sl in STAGES:
        a_vals = [stage_mean(p["protection_by_round"], sl) for p in by_cond[a_key]]
        b_vals = [stage_mean(p["protection_by_round"], sl) for p in by_cond[b_key]]
        a_vals = [v for v in a_vals if not np.isnan(v)]
        b_vals = [v for v in b_vals if not np.isnan(v)]
        cmp_ = compare_conditions(a_vals, b_vals, LABELS[a_key], LABELS[b_key], f"protection_{name}")
        comparisons.append(cmp_)
        stage_pvals.append(cmp_["welch_p"])

    bh = benjamini_hochberg(stage_pvals)
    for cmp_, bh_row in zip(comparisons, bh):
        cmp_["welch_q_bh"] = bh_row["qvalue"]
        cmp_["welch_reject_bh"] = bh_row["reject"]

    cmp_csv = OUT / "02_hitl_condition_contrasts.csv"
    flat = []
    for c in comparisons:
        flat.append({k: v for k, v in c.items() if not isinstance(v, dict)})
    with open(cmp_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in flat for k in r}))
        w.writeheader()
        w.writerows(flat)
    print(f"Wrote {cmp_csv}")

    # Elapsed time secondary (humans only; medians + IQR)
    time_rows = []
    for key in HUMAN:
        times = np.asarray([p["elapsed_time"] for p in by_cond[key] if p["elapsed_time"] is not None], dtype=float)
        time_rows.append(
            {
                "condition": LABELS[key],
                "n": len(times),
                "mean": float(np.mean(times)) if len(times) else float("nan"),
                "median": float(np.median(times)) if len(times) else float("nan"),
                "iqr_low": float(np.percentile(times, 25)) if len(times) else float("nan"),
                "iqr_high": float(np.percentile(times, 75)) if len(times) else float("nan"),
            }
        )
    ta = [p["elapsed_time"] for p in by_cond[a_key] if p["elapsed_time"] is not None]
    tb = [p["elapsed_time"] for p in by_cond[b_key] if p["elapsed_time"] is not None]
    mw_t = mannwhitney(ta, tb)
    time_contrast = {
        "outcome": "elapsed_time",
        "mannwhitney_p": mw_t["pvalue"],
        "hedges_g": hedges_g(ta, tb),
        "cliffs_delta": cliffs_delta(ta, tb),
        "note": "Human-vs-human only; do not compare to automated elapsed times inferentially.",
    }
    time_csv = OUT / "02_hitl_elapsed_time.csv"
    with open(time_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(time_rows[0].keys()))
        w.writeheader()
        w.writerows(time_rows)
        w.writerow({k: time_contrast.get(k, "") for k in time_rows[0].keys()})
    print(f"Wrote {time_csv}")
    print("Contrast summary:")
    for c in comparisons:
        print(
            f"  {c['outcome']}: diff={c['diff_a_minus_b']:.4f} "
            f"[{c['boot_ci_low']:.4f},{c['boot_ci_high']:.4f}] "
            f"Welch p={c['welch_p']:.4g} q_BH={c['welch_q_bh']:.4g} g={c['hedges_g']:.3f}"
        )


if __name__ == "__main__":
    main()
