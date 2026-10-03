# Architecture

## Repository boundaries

This repository contains three independently runnable Python components plus a
small shared library under `src/cht_dqn/`. Their similarly named `agents`,
`buffer`, and `envs` packages are intentionally not merged: each component
captures a different experimental configuration (offline, matched, web).

See also [`MODEL_CH_CODEBOOK.md`](MODEL_CH_CODEBOOK.md).

## Offline simulation

`components/offline_simulation/`

- Explores 2–10 attack-graph nodes and multiple defender/attacker policy combinations (Appendix B).
- Uses long simulation runs, multi-seed sweeps, and 100-step target-network updates.
- `batch/` runs the Appendix B cases into `results/cases1_4/` and `results/cases5_8/`.

## Human web experiment

`components/web_experiment/`

- `app_reward_aware.py`: the H_reward game (human-playable DQN) vs. DQN attacker.
- `app_transition_aware.py`: the H_forecast game (human-playable CHT-DQN) vs. DQN attacker.
- `templates/` and `static/`: Flask/D3 user interfaces.
- Writes participant records to the repository-level restricted-data area.
- Writes temporary session state, pickles, and logs to `.runtime/`.

## Matched automated defenders

`components/web_matched_simulations/`

- Reproduces the web experiment's six-node, 40-round format.
- Auto-CHT uses model 5: CHT-DQN defender vs. DQN attacker.
- Auto-DQN uses model 8: DQN defender vs. DQN attacker.
- `results/` holds the 40-seed outputs used in the paper; reruns write to `results_rerun/`.

## Human-study analysis

`analysis/hitl/`

- `protection_and_time.py`: weighted data protection by stage (Figure 2) and elapsed time.
- `prospect_behavior.py`: reselection after failure and success (Figure 3).
- `action_frequencies.py`: action frequencies (Figure 4).
- `scenarios.py`: one map from condition labels to data locations.

`analysis/stats/`

- `02_*`, `03_*`, `04_*`: condition contrasts, mixed-effects model, and reselection tests (Table 1).
- `05_plot_offline_protection.py`: Appendix B protection panels (Figure 6).
- `run_all.py`: runs 02–05 in order.

Human cohorts are stored locally under `restricted_data/hitl/curated/` and are not distributed.
