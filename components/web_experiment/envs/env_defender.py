"""Defender environment for the human web-experiment kernel."""

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


class EnvDefender:
    def __init__(self, state_size, action_size, model_CH):
        self.rr = 10
        self.observation_space = state_size
        self.action_space = action_size
        self.model_CH = model_CH

    def calculate_potential_rewards_matrix(self, data, data_att, p_ag, p_d, p_expl, ct, lamda, mu):
        rewards_matrix = np.zeros((self.action_space, self.action_space))

        for def_action in range(self.action_space):
            for att_action in range(self.action_space):
                action_array = np.zeros(self.action_space)
                action_array[def_action] = 1
                action_att_array = np.zeros(self.action_space)
                action_att_array[att_action] = 1

                data_prot = [
                    1 if not (att == 1 and defn == 0) else 0
                    for att, defn in zip(action_att_array, action_array)
                ]
                reward = 0.0

                for node in range(self.action_space):
                    if self.model_CH in [1, 2, 3]:
                        reward += (
                            (+lamda * data[node] - mu * ct[node])
                            * (data_prot[node])
                            * (1 - p_expl[node])
                            + (-lamda * data[node] - mu * ct[node])
                            * (1 - data_prot[node])
                            * (p_expl[node])
                        )
                    else:
                        reward += (
                            (+lamda * data[node] - mu * ct[node]) * (data_prot[node])
                            + (-lamda * data[node] - mu * ct[node]) * (1 - data_prot[node])
                        )

                rewards_matrix[def_action][att_action] = round(reward, 2)

        return rewards_matrix

    def step(
        self,
        action,
        action_att,
        data,
        data_att,
        p_ag,
        p_d,
        p_expl,
        ct,
        lamda,
        mu,
        expected_Q_CH_ratio_att,
        g_pt,
    ):
        action_array = np.zeros(self.action_space)
        action_array[action] = 1
        action_att_array = np.zeros(self.action_space)
        action_att_array[action_att] = 1

        data_prot = [
            1 if not (att == 1 and defn == 0) else 0
            for att, defn in zip(action_att_array, action_array)
        ]
        reward = 0.0

        for node in range(self.action_space):
            if self.model_CH in [1, 2, 3]:
                reward += (
                    (+lamda * data[node] - mu * ct[node]) * (data_prot[node]) * (1 - p_expl[node])
                    + (-lamda * data[node] - mu * ct[node]) * (1 - data_prot[node]) * (p_expl[node])
                )
            else:
                reward += (
                    (+lamda * data[node] - mu * ct[node]) * (data_prot[node])
                    + (-lamda * data[node] - mu * ct[node]) * (1 - data_prot[node])
                )

        next_state = data_prot
        done = False
        return next_state, reward, done, data_prot

    def reset(self):
        return np.ones((self.observation_space,), dtype=int)
