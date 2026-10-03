# Data Access and Classification

## Released research data

Simulation outputs are kept with the code that generated them:

- `components/offline_simulation/results/cases1_4/` and `results/cases5_8/` (Appendix B)
- `components/web_matched_simulations/results/` (Auto-DQN and Auto-CHT)

These include result summaries, action frequencies, reward histories, and automated-game JSON records.

De-identified, participant-level summary statistics used in the paper are in `artifacts/hitl_stats/`.

## Restricted local data

`restricted_data/` is excluded from Git and is not distributed. It contains:

- `hitl/raw/transition_aware/`: H_forecast human sessions
- `hitl/raw/reward_aware/`: H_reward human sessions
- `hitl/curated/`: symbolic-link cohorts selecting the records used in the paper
- `hitl/mturk/`: worker and assignment records

Human-subject data are handled under Rutgers University IRB Protocol No. Pro2024000556. Filenames, timestamps, completion codes, worker IDs, and assignment IDs may be identifying or linkable.

## Excluded material

The repository excludes private-key material, Flask session caches, serialized runtime pickles, logs, Python bytecode, and editor metadata.
