 
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
# import zipfile
import datetime
import sys

import time
import pickle
import csv

# sys.path.append('E:/PycharmProjects/FederateL_ProspectT/action-branching-agents/agents')
# sys.path.append('/home/za161/Scratch/FederateL_ProspectT/action-branching-agents/agents/bdq/deepq')
# path_logs = '/home/za161/Scrtach/FederateL_ProspectT/action-branching-agents/agents/bdq'

# sys.path.append('G:/My Drive/PycharmProjects/Paper2-CognitiveH_2')


class ProspectTheory():
    def __init__(self, a_pt, b_pt, l_pt, g_pt):

        self.a_pt = a_pt
        self.b_pt = b_pt
        self.l_pt = l_pt
        self.g_pt = g_pt

    def weighting_func(self,x):
        weighted_prob = np.zeros(len(x))
        for i in range(len(x)):
            if x[i] == 0:
                weighted_prob[i] = 0
            else:
                weighted_prob[i] = math.exp(-1 * np.power(-(math.log(x[i])), self.g_pt))
            # print("math.log(x)", math.log(x))
        return weighted_prob

    def gain_value_func(self, y):
        gain_val = np.power(y, self.a_pt)
        # print("y, gain_val",y, gain_val)
        return gain_val

    def loss_value_func(self, z):
        loss_val = (-self.l_pt)* np.power(-z, self.b_pt)
        # print("z, loss_val",z, loss_val)
        return loss_val

    # def defender_ut_PT(self, current_state, action, next_state, secure, action_att, data, e):
    #     defender_ut_PT = 0
    #     pi_def = self.weighting_func(transition_prob[e][int(current_state[0])][action][att])
    #     defender_ut_PT = defender_ut_PT + \
    #                         (self.gain_value_func(def_ut)) * pi_def
    #     return defender_ut_PT, next_state[0]


class EnvAttacker():
    def __init__(self, state_size_att, action_size_att, model_CH):
        # Define action and observation space
        # self.observation_space = spaces.Box(np.array([1]), np.array([1]),dtype=np.float32)
        self.rr = 10 
        # self.action_space_att = spaces.Box(np.zeros(1,),np.ones(1,),dtype=np.int)
        # # Example for using image as input:
        # self.observation_space_att = spaces.Discrete(state_size_att)
        self.observation_space_att = state_size_att
        self.action_space_att = action_size_att
        self.model_CH = model_CH


    def step(self, action_att, action, data_att,data, p_ag,p_d,p_expl, ct_att, lamda_att, mu_att,expected_Q_CH_ratio,g_pt):
    # def step(self, action, action_att, secure, data, defender_cpu_edge_max):

        action_att_array = np.zeros(self.action_space_att)
        action_att_array [action_att] = 1
        action_array = np.zeros(self.action_space_att)
        action_array [action] = 1
        next_state_att = np.zeros((self.observation_space_att,), dtype=int)
        prospect = ProspectTheory(a_pt=1, b_pt=1, l_pt=1, g_pt=g_pt)
        
        # return PT weighted probabs (if g_pt != 1)
        p_ag_pt = prospect.weighting_func(p_ag)
        # print("Att p_ag:",p_ag)
        # pr_trans_att = p_ag * expected_Q_CH_ratio

        data_prot = [1 if not (att == 1 and defn == 0) else 0 for att, defn in zip(action_att_array, action_array)]
        reward_att = 0
        
        for action_att in range(self.action_space_att):
            if (self.model_CH in [3, 6, 9]): 
                reward_att += ( (+lamda_att*data_att[action_att] - mu_att*ct_att[action_att])*(1-data_prot[action_att])*(p_expl[action_att])\
                            + (-lamda_att*data[action_att] - mu_att*ct_att[action_att])*(data_prot[action_att]) \
                            *(1-p_expl[action_att]) )
            else:           
                reward_att += ( (+lamda_att*data_att[action_att] - mu_att*ct_att[action_att])*(1-data_prot[action_att])\
                            + (-lamda_att*data[action_att] - mu_att*ct_att[action_att])*(data_prot[action_att]))  

        next_state_att = data_prot
        # next_state_att = action_att_array

        done_att = False
        return next_state_att, reward_att, done_att
    
    
    def reset(self):
        # Reset the state of the environment to an initial state
        # state_att = np.array([0,0,0,0])
        state_att = np.ones((self.observation_space_att,), dtype=int)
        return state_att

    
  
