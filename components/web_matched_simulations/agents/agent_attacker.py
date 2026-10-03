"""Corrected attacker agent (Eqs. 11–12, 15).

Changes vs. the original implementation:
- Softmax level-0 policy ratios use temperature betta: pi ∝ exp(beta Q).
- Attacker Bellman remains standard DQN for models 1–9 (Eq. 11), as in the paper.
- Keeps epsilon-greedy / forced-random action selection used in published runs.
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


class ConvDQNAttacker(nn.Module):
    def __init__(self, input_dim, output_dim, seed):
        super(ConvDQNAttacker, self).__init__()
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


class DQNAttacker(nn.Module):
    def __init__(self, input_dim, output_dim, seed):
        super(DQNAttacker, self).__init__()
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


class AgentAttacker:
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
        self.replay_buffer_att = buffer.BasicBuffer(seed, max_size=buffer_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_conv = use_conv
        self.model = self.create_model(seed)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        self.MSE_loss = nn.MSELoss()
        self.UPDATE_TARGET_EVERY = 100
        self.target_model = self.create_model(seed)
        self.hard_update_target_model()

    def create_model(self, seed):
        if self.use_conv:
            return ConvDQNAttacker(self.state_size, self.action_size, seed).to(self.device)
        return DQNAttacker(self.state_size, self.action_size, seed).to(self.device)

    def hard_update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def _softmax_policy(self, q_values):
        """Level-0 policy pi(a|s) ∝ exp(beta * Q(s,a)) from Eq. (12)."""
        q = q_values - torch.max(q_values)
        return torch.softmax(self.betta * q, dim=-1)

    def get_action(self, current_state, action_max, p_ag, p_d, p_expl, expected_Q_CH_ratio, eps):
        current_state = torch.FloatTensor(current_state).float().unsqueeze(0).to(self.device)
        qvals = self.model.forward(current_state).detach().cpu().numpy()

        # Models 1, 4, 7 are random attackers; otherwise epsilon-greedy (published protocol).
        if (
            random.random() > eps
            and len(self.replay_buffer_att) > self.batch_size
            and self.model_CH not in [1, 4, 7]
        ):
            action = np.argmax(qvals[0, 0 : int(self.action_max)]).item()
        else:
            action = random.choice(np.arange(self.action_max))
        return action

    def remember(self, current_state, action, reward, next_state, expected_Q_CH_ratio):
        self.replay_buffer_att.add(current_state, action, reward, next_state, expected_Q_CH_ratio)

    def learn(self, current_state, action, reward, next_state, expected_Q_CH_ratio, p_ag, p_d, p_expl):
        self.replay_buffer_att.add(current_state, action, reward, next_state, expected_Q_CH_ratio)

        self.batch = self.replay_buffer_att.sample(self.batch_size)
        current_states, actions, rewards, next_states, _expected_Q_CH_ratios = self.batch

        current_states = torch.FloatTensor(np.asarray(current_states, dtype=np.float32)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states, dtype=np.float32)).to(self.device)

        current_Q = self.model(current_states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Level-0 attacker: standard DQN Bellman (Eq. 11) for all paper models 1–9.
            target_Q_values = self.target_model.forward(next_states)
            max_target_Q = torch.max(target_Q_values, 1)[0]
            target_Q = rewards + self.gamma * max_target_Q

        current_state_t = torch.FloatTensor(np.asarray(current_state, dtype=np.float32)).to(self.device)
        current_Q_CH = self.model.forward(current_state_t)
        expected_Q_CH_ratio_att = self._softmax_policy(current_Q_CH).cpu().detach().numpy()

        loss = self.MSE_loss(current_Q, target_Q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss = loss.item()

        if self.t_step % self.UPDATE_TARGET_EVERY == 0:
            self.hard_update_target_model()

        return loss, expected_Q_CH_ratio_att
