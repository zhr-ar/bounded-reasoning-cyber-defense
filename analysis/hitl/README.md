# Human-Study Analysis

Analysis for the four conditions compared in the paper, all against the same DQN attacker:

- H_forecast: human playing the transition-aware (CHT-DQN) game
- H_reward: human playing the reward-aware (DQN) game
- Auto-CHT: automated CHT-DQN defender
- Auto-DQN: automated DQN defender

`scenarios.py` centralizes all data paths. Human inputs are local and restricted. Automated inputs are read from `components/web_matched_simulations/results/`.

Run from the repository root:

```bash
python analysis/hitl/protection_and_time.py   # Figure 2
python analysis/hitl/prospect_behavior.py     # Figure 3
python analysis/hitl/action_frequencies.py    # Figure 4
```

The published figures are in `artifacts/figures/hitl/`.
