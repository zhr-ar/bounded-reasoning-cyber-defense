"""04 — Prospect / reselection inferential tests (HITL).

For each participant:
  P(reselect | previous success), P(reselect | previous failure)
Paired Wilcoxon / permutation on the within-participant contrast.
Optional round-level logistic mixed model:
  reselect ~ previous_outcome * condition + round + (1 | participant)

Usage:
  python analysis/stats/04_hitl_prospect_tests.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "analysis" / "hitl"))
from scenarios import scenarios  # noqa: E402
from stats_common import mean_ci, paired_wilcoxon, permutation_mean_diff  # noqa: E402

OUT = ROOT / "artifacts" / "hitl_stats"
HUMAN = {
    "cht_human_vs_dqn_attacker": "transition_aware",
    "dqn_human_vs_dqn_attacker": "reward_aware",
}


def participant_reselection(data: dict) -> dict | None:
    def_a = np.asarray(data["action_history"], dtype=int)
    att_a = np.asarray(data["action_att_history"], dtype=int)
    n = min(len(def_a), len(att_a))
    if n < 2:
        return None
    def_a, att_a = def_a[:n], att_a[:n]
    success = def_a[:-1] == att_a[:-1]
    failure = ~success
    reselect = def_a[1:] == def_a[:-1]

    def rate(mask):
        if mask.sum() == 0:
            return float("nan")
        return float(np.mean(reselect[mask]))

    return {
        "p_reselect_success": rate(success),
        "p_reselect_failure": rate(failure),
        "n_success": int(success.sum()),
        "n_failure": int(failure.sum()),
        "contrast_succ_minus_fail": rate(success) - rate(failure)
        if success.sum() and failure.sum()
        else float("nan"),
    }


def build_round_frame():
    rows = []
    for key, label in HUMAN.items():
        for path in sorted(Path(scenarios[key]).glob("*.json")):
            data = json.loads(path.read_text())
            def_a = np.asarray(data["action_history"], dtype=int)
            att_a = np.asarray(data["action_att_history"], dtype=int)
            n = min(len(def_a), len(att_a))
            if n < 2:
                continue
            for t in range(n - 1):
                prev_success = int(def_a[t] == att_a[t])
                rows.append(
                    {
                        "participant": path.stem,
                        "condition": label,
                        "round": t + 2,
                        "previous_success": prev_success,
                        "reselect": int(def_a[t + 1] == def_a[t]),
                    }
                )
    return pd.DataFrame(rows)


def fit_logistic_mixed(df: pd.DataFrame) -> dict:
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf

        try:
            from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

            model = BinomialBayesMixedGLM.from_formula(
                "reselect ~ previous_success * C(condition) + round",
                {"participant": "0 + C(participant)"},
                data=df,
            )
            result = model.fit_vb()
            params = {}
            if hasattr(result, "fe_mean"):
                # fe_mean may be ndarray aligned with model.fenames
                names = getattr(result, "model", model).fenames if hasattr(model, "fenames") else []
                means = list(result.fe_mean)
                params = {str(n): float(v) for n, v in zip(names, means)}
            return {
                "engine": "BinomialBayesMixedGLM",
                "note": "Variational Bayes fit; report median/mean params cautiously.",
                "params": params,
            }
        except Exception:
            model = smf.glm(
                "reselect ~ previous_success * C(condition) + round",
                data=df,
                family=sm.families.Binomial(),
            )
            result = model.fit(cov_type="cluster", cov_kwds={"groups": df["participant"]})
            conf = result.conf_int()
            terms = []
            for name, coef in result.params.items():
                terms.append(
                    {
                        "term": name,
                        "coef": float(coef),
                        "odds_ratio": float(np.exp(coef)),
                        "pvalue": float(result.pvalues[name]),
                        "or_ci_low": float(np.exp(conf.loc[name, 0])),
                        "or_ci_high": float(np.exp(conf.loc[name, 1])),
                    }
                )
            return {
                "engine": "GLM_Binomial_cluster(participant)",
                "formula": "reselect ~ previous_success * C(condition) + round",
                "terms": terms,
                "summary": str(result.summary()),
            }
    except Exception as exc:  # noqa: BLE001
        return {"engine": "unavailable", "error": str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    part_rows = []
    contrasts_by_cond = {}
    for key, label in HUMAN.items():
        contrasts = []
        for path in sorted(Path(scenarios[key]).glob("*.json")):
            data = json.loads(path.read_text())
            stats = participant_reselection(data)
            if stats is None:
                continue
            row = {"condition": label, "participant_id": path.stem, **stats}
            part_rows.append(row)
            if not np.isnan(row["contrast_succ_minus_fail"]):
                contrasts.append(row["contrast_succ_minus_fail"])
        contrasts_by_cond[label] = contrasts

    part_csv = OUT / "04_hitl_participant_reselection.csv"
    with open(part_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in part_rows for k in r}))
        w.writeheader()
        w.writerows(part_rows)
    print(f"Wrote {part_csv}")

    # Within-participant: is success-reselect > failure-reselect? (pooled humans + by condition)
    tests = []
    for label in list(HUMAN.values()) + ["all_humans"]:
        if label == "all_humans":
            rows = part_rows
        else:
            rows = [r for r in part_rows if r["condition"] == label]
        succ = [r["p_reselect_success"] for r in rows if not np.isnan(r["p_reselect_success"]) and not np.isnan(r["p_reselect_failure"])]
        fail = [r["p_reselect_failure"] for r in rows if not np.isnan(r["p_reselect_success"]) and not np.isnan(r["p_reselect_failure"])]
        wx = paired_wilcoxon(succ, fail)
        perm = permutation_mean_diff(succ, fail)
        tests.append(
            {
                "group": label,
                "n": len(succ),
                "mean_p_succ": float(np.mean(succ)) if succ else float("nan"),
                "mean_p_fail": float(np.mean(fail)) if fail else float("nan"),
                "mean_contrast": float(np.mean(np.asarray(succ) - np.asarray(fail))) if succ else float("nan"),
                "wilcoxon_p": wx["pvalue"],
                "permutation_p": perm["pvalue"],
                "mean_ci_contrast": mean_ci(np.asarray(succ) - np.asarray(fail)) if succ else {},
            }
        )

    tests_csv = OUT / "04_hitl_reselection_paired_tests.csv"
    flat = []
    for t in tests:
        flat.append({k: v for k, v in t.items() if not isinstance(v, dict)})
    with open(tests_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in flat for k in r}))
        w.writeheader()
        w.writerows(flat)
    print(f"Wrote {tests_csv}")

    long_df = build_round_frame()
    long_df.to_csv(OUT / "04_hitl_reselection_round_panel.csv", index=False)
    logistic = fit_logistic_mixed(long_df)
    summary = logistic.pop("summary", "")
    (OUT / "04_hitl_reselection_logistic.json").write_text(json.dumps(logistic, indent=2))
    if summary:
        (OUT / "04_hitl_reselection_logistic_summary.txt").write_text(summary)
    print(f"Logistic engine: {logistic.get('engine')}")
    for t in tests:
        print(
            f"  {t['group']}: contrast={t['mean_contrast']:.3f} "
            f"Wilcoxon p={t['wilcoxon_p']:.4g}"
        )


if __name__ == "__main__":
    main()
