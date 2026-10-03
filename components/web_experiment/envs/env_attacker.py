"""Attacker environment for the human web-experiment kernel."""

from __future__ import annotations

import math

import numpy as np


class ProspectTheory:
    def __init__(self, a_pt, b_pt, l_pt, g_pt):
        self.a_pt = a_pt
        self.b_pt = b_pt
        self.l_pt = l_pt
        self.g_pt = g_pt

    def weighting_func(self, x):
        weighted_prob = np.zeros(len(x))
        for i in range(len(x)):
            if x[i] == 0:
                weighted_prob[i] = 0
            else:
                weighted_prob[i] = math.exp(-1 * np.power(-(math.log(x[i])), self.g_pt))
        return weighted_prob

    def gain_value_func(self, y):
        return np.power(y, self.a_pt)

    def loss_value_func(self, z):
        return (-self.l_pt) * np.power(-z, self.b_pt)


class EnvAttacker:
    def __init__(self, state_size_att, action_size_att, model_CH):
        self.rr = 10
        self.observation_space_att = state_size_att
        self.action_space_att = action_size_att
        self.model_CH = model_CH

    def step(
        self,
        action_att,
        action,
        data_att,
        data,
        p_ag,
        p_d,
        p_expl,
        ct_att,
        lamda_att,
        mu_att,
        expected_Q_CH_ratio,
        g_pt,
    ):
        action_att_array = np.zeros(self.action_space_att)
        action_att_array[action_att] = 1
        action_array = np.zeros(self.action_space_att)
        action_array[action] = 1

        data_prot = [
            1 if not (att == 1 and defn == 0) else 0
            for att, defn in zip(action_att_array, action_array)
        ]
        reward_att = 0.0

        for node in range(self.action_space_att):
            if self.model_CH in [3, 6, 9]:
                reward_att += (
                    (+lamda_att * data_att[node] - mu_att * ct_att[node])
                    * (1 - data_prot[node])
                    * (p_expl[node])
                    + (-lamda_att * data[node] - mu_att * ct_att[node])
                    * (data_prot[node])
                    * (1 - p_expl[node])
                )
            else:
                reward_att += (
                    (+lamda_att * data_att[node] - mu_att * ct_att[node]) * (1 - data_prot[node])
                    + (-lamda_att * data[node] - mu_att * ct_att[node]) * (data_prot[node])
                )

        next_state_att = data_prot
        done_att = False
        return next_state_att, reward_att, done_att

    def reset(self):
        return np.ones((self.observation_space_att,), dtype=int)
