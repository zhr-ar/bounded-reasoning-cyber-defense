# Offline Simulation

This component runs the offline experiments in Appendix B: eight defender/attacker pairings on attack graphs with 2–10 nodes, 2000 steps, and 10 seeds.

Entry points:

- `main.py` — DQN / Double DQN defender driver
- `main_ch1_salient.py` — CHT-DQN / CHT–Double DQN defender driver
- `batch/` — Appendix B batch runners

Outputs:

- `results/cases1_4/` — Cases 1–4 (CHT-DQN and DQN vs. random and DQN attackers), including six-node reward curves
- `results/cases5_8/` — Cases 5–8 (Double DQN variants)

Each scenario folder is named after the original experiment tags, for example `DQN-CH-level1 def DQN p_expl_att DQN/` (CHT-DQN vs. DQN) and `def DQN_att DQN/` (DQN vs. DQN). See `docs/MODEL_CH_CODEBOOK.md`.
