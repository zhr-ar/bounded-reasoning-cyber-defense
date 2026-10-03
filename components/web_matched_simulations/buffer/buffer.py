"""Replay buffer for the web-matched simulation kernel."""

from __future__ import annotations

import copy
import random
from collections import deque


class BasicBuffer:
    def __init__(self, seed, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.experience = (0, 0, 0, 0)
        self.seed = random.seed(seed)

    def add(self, current_state, action, reward, next_state, done):
        self.experience = copy.deepcopy((current_state, action, reward, next_state, done))
        self.buffer.append(self.experience)

    def add_with_ratio(self, current_state, action, reward, next_state, done):
        self.experience = copy.deepcopy((current_state, action, reward, next_state, done))
        self.buffer.append(self.experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            current_state, action, reward, next_state, done = experience
            state_batch.append(current_state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)
