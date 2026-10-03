 
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



class EnvDefender():
    def __init__(self, state_size, action_size, model_CH):
        # Define action and observation space
        # self.observation_space = spaces.Box(np.array([1]), np.array([1]),dtype=np.float32)
        self.rr = 10
        # self.action_space = spaces.Box(np.zeros(1,),np.ones(1,),dtype=np.int)
        # # Example for using image as input:
        # self.observation_space = spaces.Discrete(state_size)
        self.observation_space = state_size
        self.action_space = action_size
        self.model_CH = model_CH

    def step(self, action, action_att, data,data_att, p_ag,p_d,p_expl, ct, lamda, mu, expected_Q_CH_ratio_att,g_pt):
        # estimate attacker's CPUs based on success or failure
        # if data_protection == 0:
        #     attacker_cpu_est = min(action + 1, att_cpu_max)
        reward_node = np.zeros(self.action_space)
        action_array = np.zeros(self.action_space)
        action_array [action] = 1
        action_att_array = np.zeros(self.action_space)
        action_att_array [action_att] = 1
        next_state = np.zeros((self.observation_space,), dtype=int)
        prospect = ProspectTheory(a_pt=1, b_pt=1, l_pt=1, g_pt=g_pt)
        
        # if expected_Q_CH_ratio_att == 1:
        #     p_ag = np.ones(self.action_space)
        # return PT weighted probabs (if g_pt != 1)
        p_ag_pt = prospect.weighting_func(p_ag)
        # print("Def Weighted p_ag_pt",p_ag_pt)
        # pr_trans = p_ag * expected_Q_CH_ratio_att
        
        data_prot = [1 if not (att == 1 and defn == 0) else 0 for att, defn in zip(action_att_array, action_array)]
        reward = 0
        
        for action in range(self.action_space): ## with or without (p_ag[action])* ????
            if (self.model_CH in [1,2,3]):
                reward += (  (+lamda*data[action] - mu*ct[action])*(data_prot[action])*(1-p_expl[action]) \
                            + (-lamda*data[action] - mu*ct[action])*(1-data_prot[action])*(p_expl[action])  )
            else:
                reward += (  (+lamda*data[action] - mu*ct[action])*(data_prot[action]) \
                            + (-lamda*data[action] - mu*ct[action])*(1-data_prot[action])  )
                                               
        next_state = data_prot
        # next_state = action_array

        done = False
        return next_state, reward, done, data_prot
    
    
    def reset(self):
        # Reset the state of the environment to an initial state
        # state = np.array([0,0,0,0])
        state = np.ones((self.observation_space,), dtype=int)
        return state

    
  
