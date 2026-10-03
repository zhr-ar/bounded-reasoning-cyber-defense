# Bounded Reasoning: Cognitive Hierarchy in Human-versus-AI Cyber Defense

Code, simulation results, and analysis scripts for:

> Zahra Aref, Sheng Wei, and Narayan B. Mandayam. **Bounded Reasoning: Cognitive Hierarchy in Human-versus-AI Cyber Defense.** NeurIPS 2026 Workshop on Human-AI Coevolution: Measuring Human-Agent Teams in the Agentic Era (HAIC), 2026.

The study compares two human-playable games that operationalize DQN and CHT-DQN defenders with automated DQN and CHT-DQN defenders, all against the same DQN attacker on a six-node cloud attack graph.

Repository: <https://github.com/zhr-ar/bounded-reasoning-cyber-defense>

## Repository map

```text
.
├── src/cht_dqn/                  Shared contracts (epsilon schedules, model_CH codebook)
├── components/
│   ├── offline_simulation/       Offline 2–10-node experiments (Appendix B)
│   ├── web_experiment/           Flask/D3 human-study games (Figure 1)
│   └── web_matched_simulations/  Six-node, 40-round automated defenders (Auto-DQN, Auto-CHT)
├── analysis/
│   ├── hitl/                     Figures 2–4 (human vs. automated defenders)
│   └── stats/                    Table 1 statistics and Appendix B Figure 6
├── artifacts/
│   ├── figures/hitl/             Figures 2–4
│   ├── figures/offline/          Figures 6–7 (Appendix B)
│   └── hitl_stats/               De-identified human-study statistics (Table 1)
├── tests/                        Characterization tests
├── docs/                         Architecture, data access, reproducibility notes
├── requirements/                 Python dependency groups
└── restricted_data/              Local-only human-subject data (Git-ignored, not distributed)
```

## Paper artifacts

| Paper item | File(s) | Produced by |
|---|---|---|
| Figure 2 | `artifacts/figures/hitl/Data_Protection_Exploration-Exploitation_Stages.png` | `analysis/hitl/protection_and_time.py` |
| Figure 3 | `artifacts/figures/hitl/Reselection_likelihood_after_*.png` | `analysis/hitl/prospect_behavior.py` |
| Figure 4 | `artifacts/figures/hitl/Avg_Frequency_of_Actions.png` | `analysis/hitl/action_frequencies.py` |
| Table 1 | `artifacts/hitl_stats/` | `analysis/stats/02_*`, `03_*`, `04_*` |
| Figure 6 | `artifacts/figures/offline/Average_Weighted_Data_Protection_Ratio_*.png` | `analysis/stats/05_plot_offline_protection.py` |
| Figure 7 | `artifacts/figures/offline/Reward_*.png` | `components/offline_simulation/batch/run_reward_curves.py` |

The human-study scripts (Figures 2–4, Table 1) need the restricted participant data, which is not distributed. Their outputs are included in `artifacts/`.

## Components

### Offline simulation (Appendix B)

```bash
cd components/offline_simulation
python main.py               # DQN defender
python main_ch1_salient.py   # CHT-DQN defender
```

Batch runners for the Appendix B cases are in `components/offline_simulation/batch/` (`run_cases1_4.py`, `run_cases5_6.py`, `run_cases7_8.py`, `run_reward_curves.py`). Their outputs are in `results/cases1_4/` and `results/cases5_8/`.

### Matched automated defenders

Runs Auto-DQN and Auto-CHT against the DQN attacker: six nodes, 40 rounds, and 40 seeds. The outputs used in the paper are in `components/web_matched_simulations/results/`. Rerunning writes to `results_rerun/` so the published outputs are never overwritten.

```bash
cd components/web_matched_simulations
python main.py       # Auto-DQN
python main_ch1.py   # Auto-CHT
```

### Human web experiment

Runs the two Flask/D3 games. Set one stable session secret for all workers.

```bash
export FLASK_SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
cd components/web_experiment
gunicorn app_reward_aware:app -b 0.0.0.0:5001       # H_reward game
gunicorn app_transition_aware:app -b 0.0.0.0:5002   # H_forecast game
```

Local participant output goes to `restricted_data/hitl/raw/`, and runtime sessions, logs, and pickles go to `.runtime/`. Both locations are excluded from Git.

### Human-study analysis

These scripts read automated data from `components/web_matched_simulations/results/` and the human cohorts from `restricted_data/hitl/curated/`.

```bash
python analysis/hitl/protection_and_time.py
python analysis/hitl/action_frequencies.py
python analysis/hitl/prospect_behavior.py
python analysis/stats/run_all.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/base.txt
pip install -r requirements/web.txt   # additionally, for the web games
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
make test
```

Exact historical package versions were not preserved; see [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md). The `model_CH` switch meanings are recorded in [`docs/MODEL_CH_CODEBOOK.md`](docs/MODEL_CH_CODEBOOK.md).

## Data policy

Simulation results are kept with the components that produce them. De-identified, participant-level summary statistics used in the paper are in `artifacts/hitl_stats/`. Raw human session JSON, curated human cohorts, and Mechanical Turk records are not distributed. The human study was approved by the Rutgers University IRB (Protocol No. Pro2024000556).

## Citation

```bibtex
@inproceedings{aref2026bounded,
  title     = {Bounded Reasoning: Cognitive Hierarchy in Human-versus-AI Cyber Defense},
  author    = {Aref, Zahra and Wei, Sheng and Mandayam, Narayan B.},
  booktitle = {NeurIPS 2026 Workshop on Human-AI Coevolution: Measuring Human-Agent Teams in the Agentic Era},
  year      = {2026}
}
```

## License

Released under the MIT License; see [`LICENSE`](LICENSE).
