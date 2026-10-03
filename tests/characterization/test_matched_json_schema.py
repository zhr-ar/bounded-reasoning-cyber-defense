"""JSON schema characterization for matched simulation outputs."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MATCHED_RESULTS = (
    REPO_ROOT
    / "components"
    / "web_matched_simulations"
    / "results"
)

REQUIRED_KEYS = {
    "action_history",
    "action_att_history",
    "reward_history",
    "reward_att_history",
    "data_prot_history",
    "eps_history",
    "avg_data_prot_w_ratio",
    "elapsed_time",
    "score",
    "global_variables",
}


class MatchedJsonSchemaTests(unittest.TestCase):
    def test_canonical_scenario_json_keys(self):
        scenarios = [
            "DQN-CH-level1 def DQN p_expl_att DQN",
            "def DQN_att DQN",
        ]
        for scenario in scenarios:
            folder = MATCHED_RESULTS / scenario / "new" / "6 node"
            files = sorted(folder.glob("my_seed_def_*_model_CH_*.json"))
            self.assertTrue(files, msg=f"No JSON found for {scenario}")
            with files[0].open() as handle:
                payload = json.load(handle)
            missing = REQUIRED_KEYS - set(payload)
            self.assertFalse(missing, msg=f"{scenario} missing {missing}")


if __name__ == "__main__":
    unittest.main()
