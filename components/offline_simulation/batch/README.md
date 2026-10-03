# Offline Batch Runs

- `run_cases1_4.py` — Cases 1–4 across graph sizes into `results/cases1_4/`, plus the per-seed protection tables (`data_prot_*.xlsx`).
- `run_cases5_6.py` — Cases 5–6 (Double DQN attacker) into `results/cases5_8/`.
- `run_cases7_8.py` — Cases 7–8 (DQN attacker) into `results/cases5_8/`.
- `run_reward_curves.py` — six-node Cases 1–4 with reward-curve dumps, and Figure 7.

Figure 6 is drawn by `analysis/stats/05_plot_offline_protection.py`.

These are long-running experiments. Review `docs/MODEL_CH_CODEBOOK.md` before rerunning them.
