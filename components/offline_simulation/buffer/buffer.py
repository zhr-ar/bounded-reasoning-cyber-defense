
import os
import sys
import glob
import argparse 
# from tqdm import tqdm 
sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
import math
import random

# import gym
from collections import deque

import torch
import torch.nn as nn
import torch.autograd as autograd
import copy

import tempfile
import zipfile
import datetime
import sys

import time
import pickle
import csv

# sys.path.append('E:/PycharmProjects/FederateL_ProspectT/action-branching-agents/agents')
# sys.path.append('/home/za161/Scratch/FederateL_ProspectT/action-branching-agents/agents/bdq/deepq')
# path_logs = '/home/za161/Scrtach/FederateL_ProspectT/action-branching-agents/agents/bdq'

# sys.path.append('G:/My Drive/PycharmProjects/Paper2-CognitiveH_2')


class BasicBuffer:
    def __init__(self, seed, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.experience = (0,0,0,0)
        self.seed = random.seed(seed)

    def add(self, current_state, action, reward, next_state, done):
        ## """"Add a new experience to buffer.""""
        # self.experience = copy.deepcopy((current_state, action, np.array([reward]), next_state, done))
        self.experience = copy.deepcopy((current_state, action, reward, next_state, done))
        self.buffer.append(self.experience)

    def add_with_ratio(self, current_state, action, reward, next_state, done):
        ## """"Add a new experience to buffer.""""
        # self.experience = copy.deepcopy((current_state, action, np.array([reward]), next_state, done))
        self.experience = copy.deepcopy((current_state, action, reward, next_state, done))
        self.buffer.append(self.experience)

    def sample(self, batch_size): 
        ## "Randomly sample a batch of experiences from buffer"
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

