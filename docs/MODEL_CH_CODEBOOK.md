# Model CH Codebook

Canonical meanings of the numeric `model_CH` switch used across the three components.
Source of truth for code: `src/cht_dqn/contracts/profiles.py`.

| ID | Label | Defender Bellman | Attacker | Defender reward uses `p_expl` | Attacker reward uses `p_expl` | CH ratio feedback |
|---|---|---|---|---|---|---|
| 1 | DQN + p_expl vs random | max Q | forced random | yes | no | no |
| 2 | DQN + p_expl vs DQN | max Q | DQN | yes | no | no |
| 3 | DQN + p_expl vs DQN + p_expl | max Q | DQN | yes | yes | no |
| 4 | CHT-DQN vs random | CHT-scaled max Q | forced random | no | no | yes in CH mains |
| 5 | CHT-DQN vs DQN | CHT-scaled max Q | DQN | no | no | yes in CH mains |
| 6 | CHT-DQN vs DQN + p_expl | CHT-scaled max Q | DQN | no | yes | yes in CH mains |
| 7 | DQN vs random | max Q | forced random | no | no | no |
| 8 | DQN vs DQN | max Q | DQN | no | no | no |
| 9 | DQN vs DQN + p_expl | max Q | DQN | no | yes | no |

Notes:

- Attacker CHT Bellman branches exist in source but are unreachable for models 1–9.
- CHT-scaled max Q currently multiplies `max_next_Q` by `sum(CH_ratio * p_expl)`.
- Canonical HITL controls are model 5 (transition-aware / Scenario 3) and model 8 (reward-aware / Scenario 4).
