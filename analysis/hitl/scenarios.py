"""Canonical data locations for the four conditions compared in the paper."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MATCHED_RESULTS = PROJECT_ROOT / "components" / "web_matched_simulations" / "results"
RESTRICTED_HITL = PROJECT_ROOT / "restricted_data" / "hitl" / "curated"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

scenarios = {
    "cht_human_vs_dqn_attacker": str(RESTRICTED_HITL / "transition_aware"),
    "dqn_human_vs_dqn_attacker": str(RESTRICTED_HITL / "reward_aware"),
    "cht_sim_vs_dqn_attacker": str(
        MATCHED_RESULTS
        / "DQN-CH-level1 def DQN p_expl_att DQN"
        / "new"
        / "6 node"
    ),
    "dqn_sim_vs_dqn_attacker": str(
        MATCHED_RESULTS / "def DQN_att DQN" / "new" / "6 node"
    ),
}
