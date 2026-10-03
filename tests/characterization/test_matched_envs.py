"""Characterization tests that lock current environment reward semantics."""

from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
MATCHED = REPO_ROOT / "components" / "web_matched_simulations"


def _load_matched_envs():
    sys.path.insert(0, str(MATCHED))
    try:
        env_defender = importlib.import_module("envs.env_defender")
        env_attacker = importlib.import_module("envs.env_attacker")
        return env_defender, env_attacker
    finally:
        if str(MATCHED) in sys.path:
            sys.path.remove(str(MATCHED))


class MatchedEnvironmentCharacterization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env_defender, env_attacker = _load_matched_envs()
        cls.EnvDefender = env_defender.EnvDefender
        cls.EnvAttacker = env_attacker.EnvAttacker

    def setUp(self):
        self.data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.ct = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        self.p_expl = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        self.zeros = np.zeros(6)
        self.ones = np.ones(6)

    def test_data_prot_rule(self):
        env = self.EnvDefender(state_size=6, action_size=6, model_CH=8)
        next_state, reward, done, data_prot = env.step(
            action=1,
            action_att=2,
            data=self.data,
            data_att=self.data,
            p_ag=self.zeros,
            p_d=self.zeros,
            p_expl=self.p_expl,
            ct=self.ct,
            lamda=10,
            mu=1,
            expected_Q_CH_ratio_att=self.ones,
            g_pt=1,
        )
        self.assertEqual(data_prot[2], 0)
        self.assertTrue(all(data_prot[i] == 1 for i in range(6) if i != 2))
        self.assertEqual(next_state, data_prot)
        self.assertFalse(done)
        # contested unprotected node 2: loss = -lamda*data - mu*ct
        # all other nodes protected: gain = +lamda*data - mu*ct
        expected = 0.0
        for i in range(6):
            if i == 2:
                expected += (-10 * self.data[i] - 1 * self.ct[i])
            else:
                expected += (+10 * self.data[i] - 1 * self.ct[i])
        self.assertAlmostEqual(reward, expected)

    def test_model_1_uses_p_expl_in_defender_reward(self):
        env = self.EnvDefender(state_size=6, action_size=6, model_CH=1)
        _, reward, _, data_prot = env.step(
            action=0,
            action_att=0,
            data=self.data,
            data_att=self.data,
            p_ag=self.zeros,
            p_d=self.zeros,
            p_expl=self.p_expl,
            ct=self.ct,
            lamda=10,
            mu=1,
            expected_Q_CH_ratio_att=self.ones,
            g_pt=1,
        )
        # matched defend/attack on node 0 => all protected
        expected = 0.0
        for i in range(6):
            expected += (+10 * self.data[i] - 1 * self.ct[i]) * (1 - self.p_expl[i])
        self.assertTrue(all(v == 1 for v in data_prot))
        self.assertAlmostEqual(reward, expected)

    def test_attacker_model_8_without_p_expl(self):
        env = self.EnvAttacker(state_size_att=6, action_size_att=6, model_CH=8)
        next_state, reward, done = env.step(
            action_att=2,
            action=1,
            data_att=self.data,
            data=self.data,
            p_ag=self.zeros,
            p_d=self.zeros,
            p_expl=self.p_expl,
            ct_att=self.ct,
            lamda_att=10,
            mu_att=1,
            expected_Q_CH_ratio=self.ones,
            g_pt=1,
        )
        expected = 0.0
        for i in range(6):
            if i == 2:
                expected += (+10 * self.data[i] - 1 * self.ct[i])
            else:
                expected += (-10 * self.data[i] - 1 * self.ct[i])
        self.assertAlmostEqual(reward, expected)
        self.assertEqual(next_state[2], 0)
        self.assertFalse(done)


class ProfileAndEpsilonCharacterization(unittest.TestCase):
    def test_canonical_profiles(self):
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from cht_dqn.contracts import MODEL_CH_CODEBOOK, PROFILES
        from cht_dqn.contracts.epsilon import exponential_epsilon, linear_offline_epsilon

        self.assertEqual(PROFILES["matched_dqn"].model_ch, 8)
        self.assertEqual(PROFILES["matched_cht"].model_ch, 5)
        self.assertEqual(PROFILES["hitl_reward_aware"].buffer_size, 10_000)
        self.assertEqual(MODEL_CH_CODEBOOK[5].defender_bellman, "cht_scaled_max_q")
        self.assertTrue(MODEL_CH_CODEBOOK[1].attacker_forced_random)

        self.assertAlmostEqual(exponential_epsilon(0, 0.9, 0.93, 0.05), 0.9)
        self.assertAlmostEqual(exponential_epsilon(40, 0.9, 0.93, 0.05), 0.05)
        self.assertAlmostEqual(linear_offline_epsilon(0, 2000, 2.0, 0.05), 1.0)
        self.assertAlmostEqual(linear_offline_epsilon(1000, 2000, 2.0, 0.05), 0.05)


if __name__ == "__main__":
    unittest.main()
