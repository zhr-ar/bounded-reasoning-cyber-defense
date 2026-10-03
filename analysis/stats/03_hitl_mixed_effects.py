"""03 — HITL longitudinal mixed-effects model.

Model:
  protection ~ condition * stage + (1 | participant)

Uses statsmodels MixedLM when available; otherwise falls back to OLS with
cluster-robust SEs by participant and writes a clear note in the output JSON.

Usage:
  python analysis/stats/03_hitl_mixed_effects.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "analysis" / "hitl"))
from scenarios import scenarios  # noqa: E402

OUT = ROOT / "artifacts" / "hitl_stats"
HUMAN = {
    "cht_human_vs_dqn_attacker": "transition_aware",
    "dqn_human_vs_dqn_attacker": "reward_aware",
}
STAGE_BOUNDS = [(0, 10), (10, 20), (20, 30), (30, 40)]
STAGE_NAMES = ["R1-10", "R11-20", "R21-30", "R31-40"]


def build_long_frame(include_incomplete: bool = False) -> pd.DataFrame:
    rows = []
    for key, label in HUMAN.items():
        folder = Path(scenarios[key])
        for path in sorted(folder.glob("*.json")):
            data = json.loads(path.read_text())
            hist = np.asarray(data.get("data_prot_history", []), dtype=float)
            if hist.ndim == 2:
                data_hist = np.asarray(
                    data.get("data_history", [[1.0] * hist.shape[1]] * len(hist)), dtype=float
                )
                prot = np.sum(hist * data_hist, axis=1) / np.maximum(np.sum(data_hist, axis=1), 1e-12)
            else:
                prot = hist
            if (not include_incomplete) and len(prot) < 40:
                continue
            pid = path.stem
            for t, val in enumerate(prot[:40]):
                stage_idx = min(t // 10, 3)
                rows.append(
                    {
                        "participant": pid,
                        "condition": label,
                        "round": t + 1,
                        "stage": STAGE_NAMES[stage_idx],
                        "stage_idx": stage_idx,
                        "protection": float(val),
                        "is_transition": int(label == "transition_aware"),
                    }
                )
    return pd.DataFrame(rows)


def fit_mixedlm(df: pd.DataFrame) -> dict:
    import statsmodels.formula.api as smf

    # Random intercept for participant; fixed condition × stage.
    model = smf.mixedlm(
        "protection ~ C(condition) * C(stage)",
        data=df,
        groups=df["participant"],
    )
    result = model.fit(method="lbfgs", reml=True)
    params = result.params.to_dict()
    conf = result.conf_int().rename(columns={0: "ci_low", 1: "ci_high"})
    table = []
    for name, coef in params.items():
        table.append(
            {
                "term": name,
                "coef": float(coef),
                "pvalue": float(result.pvalues.get(name, np.nan)),
                "ci_low": float(conf.loc[name, "ci_low"]) if name in conf.index else float("nan"),
                "ci_high": float(conf.loc[name, "ci_high"]) if name in conf.index else float("nan"),
            }
        )
    return {
        "engine": "statsmodels.MixedLM",
        "formula": "protection ~ C(condition) * C(stage) + (1|participant)",
        "n_obs": int(len(df)),
        "n_participants": int(df["participant"].nunique()),
        "converged": bool(getattr(result, "converged", True)),
        "terms": table,
        "summary": str(result.summary()),
    }


def fit_ols_cluster(df: pd.DataFrame) -> dict:
    import statsmodels.formula.api as smf

    model = smf.ols("protection ~ C(condition) * C(stage)", data=df)
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df["participant"]})
    conf = result.conf_int()
    table = []
    for name, coef in result.params.items():
        table.append(
            {
                "term": name,
                "coef": float(coef),
                "pvalue": float(result.pvalues[name]),
                "ci_low": float(conf.loc[name, 0]),
                "ci_high": float(conf.loc[name, 1]),
            }
        )
    return {
        "engine": "statsmodels.OLS+cluster(participant)",
        "formula": "protection ~ C(condition) * C(stage)  [cluster-robust by participant]",
        "n_obs": int(len(df)),
        "n_participants": int(df["participant"].nunique()),
        "note": "Fallback when MixedLM is unavailable or fails to converge.",
        "terms": table,
        "summary": str(result.summary()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include_incomplete", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    df = build_long_frame(include_incomplete=args.include_incomplete)
    long_csv = OUT / "03_hitl_longitudinal_panel.csv"
    df.to_csv(long_csv, index=False)
    print(f"Wrote {long_csv} ({len(df)} rows, {df['participant'].nunique()} participants)")

    result = None
    errors = []
    try:
        result = fit_mixedlm(df)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"MixedLM failed: {exc}")
        try:
            result = fit_ols_cluster(df)
        except Exception as exc2:  # noqa: BLE001
            errors.append(f"OLS-cluster failed: {exc2}")
            raise SystemExit("; ".join(errors))

    out_json = OUT / "03_hitl_mixed_effects.json"
    payload = {"errors": errors, **result}
    # Keep JSON lean: drop huge text summary into sidecar.
    summary = payload.pop("summary", "")
    out_json.write_text(json.dumps(payload, indent=2))
    (OUT / "03_hitl_mixed_effects_summary.txt").write_text(summary)
    print(f"Wrote {out_json}")
    print(f"Engine: {payload['engine']}")
    for term in payload["terms"][:12]:
        print(f"  {term['term']}: coef={term['coef']:.4f} p={term['pvalue']:.4g}")


if __name__ == "__main__":
    main()
