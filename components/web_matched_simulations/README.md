# Matched Automated Defenders

This component generates the automated defenders compared with the human games:

- Auto-CHT: CHT-DQN defender vs. DQN attacker (`model_CH=5`)
- Auto-DQN: DQN defender vs. DQN attacker (`model_CH=8`)
- six attack-graph nodes
- 40 rounds
- 40 defender seeds

Entry points:

- `main.py`
- `main_ch1.py`
- `batch/run_matched_controls.py` — both 40-seed automated runs
- `analysis/plot_convergence.py` — matched reward/convergence figures

The 40-seed outputs used in the paper are in `results/` and are read by `../../analysis/hitl/scenarios.py`. Rerunning `main.py` / `main_ch1.py` writes to `results_rerun/`, so the published outputs are never overwritten.
Runtime logs are written to `.runtime/logs/web_matched_simulations/`.
