"""Corrected defender agent (Eqs. 10, 13–16, 35) + Double DQN baselines.

Changes vs. the original implementation:
- Softmax CH ratios use temperature betta (beta).
- CHT Bellman (models 4–6) uses counterfactual expectation over attacker
  actions: sum_a pi_A(a) * [R(s,a_D,a) + gamma * max Q(s'(a_D,a))],
  matching appendix Eq. (35) under deterministic action-conditioned transitions.
- Keeps epsilon-greedy action selection used in the published experiments.

Baseline extensions:
- double_dqn: Double DQN / CHT–Double DQN bootstrap
  Q_target(s', argmax_a Q_online(s', a)).
- use_dueling: Dueling architecture (optional stronger value baseline).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import random
import torch
import torch.nn as nn

from buffer import buffer

sys.path.append(os.getcwd())


class ConvDQNDefender(nn.Module):
    def __init__(self, input_dim, output_dim, seed):
        super(ConvDQNDefender, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conv1 = nn.Conv2d(self.input_dim, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, self.output_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)


class DQNDefender(nn.Module):
    def __init__(self, input_dim, output_dim, seed):
        super(DQNDefender, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim),
        )

    def forward(self, current_state):
        return self.fc(current_state)


class DuelingDQNDefender(nn.Module):
    """Dueling DQN head: Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)."""

    def __init__(self, input_dim, output_dim, seed):
        super(DuelingDQNDefender, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.feature = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim),
        )

    def forward(self, current_state):
        feats = self.feature(current_state)
        value = self.value(feats)
        advantage = self.advantage(feats)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class AgentDefender:
    def __init__(
        self,
        env,
        state_size,
        action_size,
        action_max,
        buffer_size,
        seed,
        lr,
        batch_size,
        gamma,
        betta,
        model_CH,
        use_conv=False,
        double_dqn=False,
        use_dueling=False,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.t_step = 0
        self.env = env
        self.action_max = action_max
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        self.gamma = gamma
        self.betta = betta
        self.model_CH = model_CH
        self.replay_buffer = buffer.BasicBuffer(seed, max_size=buffer_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_conv = use_conv
        self.double_dqn = bool(double_dqn)
        self.use_dueling = bool(use_dueling)
        self.model = self.create_model(seed)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        self.MSE_loss = nn.MSELoss()
        self.UPDATE_TARGET_EVERY = 100
        self.target_model = self.create_model(seed)
        self.hard_update_target_model()

    def create_model(self, seed):
        if self.use_conv:
            return ConvDQNDefender(self.state_size, self.action_size, seed).to(self.device)
        if self.use_dueling:
            return DuelingDQNDefender(self.state_size, self.action_size, seed).to(self.device)
        return DQNDefender(self.state_size, self.action_size, seed).to(self.device)

    def _bootstrap_next_values(self, states):
        """Next-state values under DQN or Double DQN."""
        if self.double_dqn:
            next_actions = self.model(states).argmax(dim=1, keepdim=True)
            return self.target_model(states).gather(1, next_actions).squeeze(1)
        return torch.max(self.target_model(states), dim=1)[0]

    def hard_update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def _softmax_policy(self, q_values):
        """pi(a|s) ∝ exp(beta * Q(s,a)), numerically stabilized."""
        q = q_values - torch.max(q_values)
        return torch.softmax(self.betta * q, dim=-1)

    def _counterfactual_reward_and_next_state(self, action_def, action_att, data, ct, lamda, mu):
        """Deterministic env outcome for joint actions (matches EnvDefender for models 4–9)."""
        data = np.asarray(data, dtype=np.float64)
        ct = np.asarray(ct, dtype=np.float64)
        data_prot = np.ones(self.action_size, dtype=np.float64)
        if int(action_att) != int(action_def):
            data_prot[int(action_att)] = 0.0

        reward = float(
            np.sum(
                (lamda * data - mu * ct) * data_prot
                + (-lamda * data - mu * ct) * (1.0 - data_prot)
            )
        )
        return reward, data_prot

    def _pack_aux(self, expected_Q_CH_ratio_att, data, ct, lamda, mu):
        return {
            "ch_ratio": np.asarray(expected_Q_CH_ratio_att, dtype=np.float64),
            "data": None if data is None else np.asarray(data, dtype=np.float64),
            "ct": None if ct is None else np.asarray(ct, dtype=np.float64),
            "lamda": lamda,
            "mu": mu,
        }

    def get_action(self, current_state, action_max, p_ag, p_d, p_expl, expected_Q_CH_ratio_att, eps):
        current_state = torch.FloatTensor(current_state).float().unsqueeze(0).to(self.device)
        qvals = self.model.forward(current_state).detach().cpu().numpy()

        # Published experiments use epsilon-greedy, not the softmax policy of Eq. (13).
        if random.random() > eps and len(self.replay_buffer) > self.batch_size:
            action = np.argmax(qvals[0, 0 : int(self.action_max)]).item()
        else:
            action = random.choice(np.arange(action_max))
        return action

    def remember(
        self,
        current_state,
        action,
        reward,
        next_state,
        expected_Q_CH_ratio_att,
        data=None,
        ct=None,
        lamda=None,
        mu=None,
    ):
        aux = self._pack_aux(expected_Q_CH_ratio_att, data, ct, lamda, mu)
        self.replay_buffer.add(current_state, action, reward, next_state, aux)

    def learn(
        self,
        current_state,
        action,
        reward,
        next_state,
        expected_Q_CH_ratio_att,
        p_ag,
        p_d,
        p_expl,
        data=None,
        ct=None,
        lamda=None,
        mu=None,
    ):
        aux = self._pack_aux(expected_Q_CH_ratio_att, data, ct, lamda, mu)
        self.replay_buffer.add(current_state, action, reward, next_state, aux)

        self.batch = self.replay_buffer.sample(self.batch_size)
        current_states, actions, rewards, next_states, infos = self.batch

        current_states = torch.FloatTensor(np.asarray(current_states, dtype=np.float32)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states, dtype=np.float32)).to(self.device)

        current_Q = self.model(current_states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.model_CH in [1, 2, 3, 7, 8, 9]:
                # DQN / Double DQN target (Eq. 15 + optional Double bootstrap).
                max_target_Q = self._bootstrap_next_values(next_states)
                target_Q = rewards + self.gamma * max_target_Q
            else:
                # CHT-DQN / CHT–Double DQN target (Eqs. 10, 14, 35).
                batch_n = len(next_states)
                n_att = self.action_size
                cf_states = np.zeros((batch_n, n_att, n_att), dtype=np.float32)
                cf_rewards = np.zeros((batch_n, n_att), dtype=np.float32)
                pi_batch = np.zeros((batch_n, n_att), dtype=np.float32)
                use_fallback = np.zeros(batch_n, dtype=bool)

                for idx in range(batch_n):
                    info = infos[idx]
                    if isinstance(info, dict):
                        ch_ratio = np.asarray(info["ch_ratio"], dtype=np.float64)
                        data_i = info.get("data")
                        ct_i = info.get("ct")
                        lamda_i = info.get("lamda")
                        mu_i = info.get("mu")
                    else:
                        ch_ratio = np.asarray(info, dtype=np.float64)
                        data_i, ct_i, lamda_i, mu_i = data, ct, lamda, mu

                    if data_i is None or ct_i is None or lamda_i is None or mu_i is None:
                        use_fallback[idx] = True
                        continue

                    pi = np.asarray(ch_ratio, dtype=np.float64).reshape(-1)
                    if pi.shape[0] != n_att:
                        pi = np.ones(n_att, dtype=np.float64) / n_att
                    pi = np.maximum(pi, 0.0)
                    pi_sum = pi.sum()
                    pi = (np.ones(n_att) / n_att) if pi_sum <= 0 else (pi / pi_sum)
                    pi_batch[idx] = pi

                    a_def = int(actions[idx].item())
                    for a_att in range(n_att):
                        r_cf, s_cf = self._counterfactual_reward_and_next_state(
                            a_def, a_att, data_i, ct_i, lamda_i, mu_i
                        )
                        cf_rewards[idx, a_att] = r_cf
                        cf_states[idx, a_att] = s_cf

                flat_states = torch.FloatTensor(cf_states.reshape(batch_n * n_att, n_att)).to(self.device)
                max_next_Q = self._bootstrap_next_values(flat_states).view(batch_n, n_att)
                expected = torch.FloatTensor(pi_batch).to(self.device) * (
                    torch.FloatTensor(cf_rewards).to(self.device) + self.gamma * max_next_Q
                )
                target_Q = expected.sum(dim=1)

                if use_fallback.any():
                    max_std = self._bootstrap_next_values(next_states)
                    for idx in np.where(use_fallback)[0]:
                        target_Q[idx] = rewards[idx] + self.gamma * max_std[idx]

        # Defender-level softmax policy ratios (Eq. 13), returned for logging / CH bookkeeping.
        current_state_t = torch.FloatTensor(np.asarray(current_state, dtype=np.float32)).to(self.device)
        current_Q_CH = self.model.forward(current_state_t)
        expected_Q_CH_ratio = self._softmax_policy(current_Q_CH).cpu().detach().numpy()

        loss = self.MSE_loss(current_Q, target_Q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss = loss.item()

        if self.t_step % self.UPDATE_TARGET_EVERY == 0:
            self.hard_update_target_model()

        return loss, expected_Q_CH_ratio
