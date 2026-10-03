"""Plot offline protection vs. attack-graph size (Appendix B, Figure 6).

Outputs (identical figsize / dpi / fonts):
  artifacts/figures/offline/Average_Weighted_Data_Protection_Ratio_Cases1_6.png
  artifacts/figures/offline/Average_Weighted_Data_Protection_Ratio_Cases5_8_DoubleDQN.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OFFLINE = ROOT / "components" / "offline_simulation"
FIG_DIR = ROOT / "artifacts" / "figures" / "offline"

FIGSIZE = (10.0, 8.0)
DPI = 180
FONT_LABEL = 14
FONT_TICK = 13
FONT_LEGEND = 10
FONT_TITLE = 14

NODES = np.arange(2, 11)
SEED_MIN, SEED_MAX = 101, 111

CASES_1_4 = [
    (
        "results/cases1_4/data_prot_littman_def_DQN_p_expl_att_random.xlsx",
        "DQN-CH-level1 def DQN p_expl_att random/new/",
        "-s",
        "magenta",
        "<CASE 1> D: CHT-DQN | A: Random",
    ),
    (
        "results/cases1_4/data_prot_littman_def_DQN_p_expl_att_DQN.xlsx",
        "DQN-CH-level1 def DQN p_expl_att DQN/new/",
        "-s",
        "indigo",
        "<CASE 2> D: CHT-DQN | A: DQN",
    ),
    (
        "results/cases1_4/data_prot_DQN_att_random.xlsx",
        "def DQN_att random/new/",
        ":v",
        "lime",
        "<CASE 3> D: DQN | A: Random",
    ),
    (
        "results/cases1_4/data_prot_def_DQN_att_DQN.xlsx",
        "def DQN_att DQN/new/",
        ":v",
        "darkgreen",
        "<CASE 4> D: DQN | A: DQN",
    ),
]

CASES_5_6 = [
    (
        "DQN-CH-level1 def DQN p_expl_att DQN_double_att_double/new",
        "-P",
        "crimson",
        "<CASE 5> D: CHT-Double DQN | A: Double DQN",
    ),
    (
        "def DQN_att DQN_double_att_double/new",
        "--D",
        "crimson",
        "<CASE 6> D: Double DQN | A: Double DQN",
    ),
]

CASES_5_8 = CASES_5_6 + [
    (
        "DQN-CH-level1 def DQN p_expl_att DQN_double/new",
        "-P",
        "darkorange",
        "<CASE 7> D: CHT-Double DQN | A: DQN",
    ),
    (
        "def DQN_att DQN_double/new",
        "--D",
        "darkorange",
        "<CASE 8> D: Double DQN | A: DQN",
    ),
]


def style_ax(ax: plt.Axes, title: str, ylabel: str | None = None) -> None:
    ax.set_xlabel("Number of Nodes in Attack Graph", fontsize=FONT_LABEL)
    ax.set_xticks(NODES)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax.set_title(title, fontsize=FONT_TITLE)
    ax.grid(True, alpha=0.4)
    ax.tick_params(axis="both", labelsize=FONT_TICK)


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("Wrote", path)


def load_xlsx_means_stds(xlsx_path: Path):
    arr = pd.read_excel(xlsx_path, engine="openpyxl").to_numpy()
    data = arr[:, -9:]
    return np.mean(data, axis=0), np.std(data, axis=0)


def read_seed_protection(path: Path):
    if not path.exists():
        return None
    for line in path.read_text().strip().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            return float(parts[-1])
        except ValueError:
            continue
    return None


def load_case_means_stds(scenario_subdir: str):
    means, stds = [], []
    root = OFFLINE / "results/cases5_8" / scenario_subdir
    for node in NODES:
        vals = []
        for seed in range(SEED_MIN, SEED_MAX):
            v = read_seed_protection(root / f"{node} node" / f"{seed - 100}.txt")
            if v is not None:
                vals.append(v)
        if len(vals) < 1:
            raise FileNotFoundError(f"No seeds under {root / f'{node} node'}")
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals, ddof=0)))
    return np.asarray(means), np.asarray(stds)


def plot_cases1_6() -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for xlsx, _, fmt, color, label in CASES_1_4:
        mean, std = load_xlsx_means_stds(OFFLINE / xlsx)
        ax.errorbar(
            NODES, mean, yerr=std, fmt=fmt, color=color, label=label,
            capsize=4, linewidth=2, markersize=7,
        )
    for scenario, fmt, color, label in CASES_5_6:
        mean, std = load_case_means_stds(scenario)
        ax.errorbar(
            NODES, mean, yerr=std, fmt=fmt, color=color, label=label,
            capsize=4, linewidth=2, markersize=7,
        )
    style_ax(ax, "Avg Weighted Data Protection Ratio", "Avg Weighted Data Protection Ratio")
    ax.legend(loc="lower right", fontsize=FONT_LEGEND)
    save_fig(fig, FIG_DIR / "Average_Weighted_Data_Protection_Ratio_Cases1_6.png")


def plot_cases5_8() -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for scenario, fmt, color, label in CASES_5_8:
        mean, std = load_case_means_stds(scenario)
        ax.errorbar(
            NODES, mean, yerr=std, fmt=fmt, color=color, label=label,
            capsize=4, linewidth=2, markersize=7,
        )
    style_ax(
        ax,
        "Avg Weighted Data Protection Ratio (Cases 5–8)",
        "Avg Weighted Data Protection Ratio",
    )
    ax.legend(loc="lower right", fontsize=FONT_LEGEND)
    save_fig(fig, FIG_DIR / "Average_Weighted_Data_Protection_Ratio_Cases5_8_DoubleDQN.png")


def main() -> None:
    plot_cases1_6()
    plot_cases5_8()


if __name__ == "__main__":
    main()
