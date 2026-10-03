 
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
from collections import Counter

import torch
import torch.nn as nn
import torch.autograd as autograd
import copy

import tempfile
import zipfile
import datetime
import sys
import ast 

import time
import pickle
import csv

# sys.path.append('E:/PycharmProjects/FederateL_ProspectT/action-branching-agents/agents')
# sys.path.append('/home/za161/Scratch/FederateL_ProspectT/action-branching-agents/agents/bdq/deepq')
# path_logs = '/home/za161/Scrtach/FederateL_ProspectT/action-branching-agents/agents/bdq'

# sys.path.append('G:/My Drive/PycharmProjects/Paper2-CognitiveH_2')

from envs import env_attacker, env_defender
from agents import agent_attacker, agent_defender
from buffer import buffer


def parse_arguments():
    parser = argparse.ArgumentParser(description='CH')

    # parser.add_argument("--level", type=str, required=True)
    # parser.add_argument("--alpha", type=float, default=0.01, help="Alpha for BRTDP")

    parser.add_argument('--model_CH', default=2, type=int, help='Models 1-6 where DEF DQN AG;'
                        '<1> ATT Random; <2> ATT DQN; <3> ATT DQN and AG;'
                        '<4> Littman ATT Random; <5> Littman ATT DQN; <6> Littman ATT DQN AG;'
                        'Models 7-9 where DEF DQN <7> ATT Random; <8> ATT DQN; <9> ATT DQN and AG;')
    parser.add_argument('--node_max', default=4, type=int, help='number of edge servers')
    
    parser.add_argument('--time_max', default=2000, type=int, help='number of time steps, at least=batch_size')
    parser.add_argument('--update_interval', default=100, type=int, help='')
    parser.add_argument('--data', default=1, type=float, help='cloud data size by defender')
    parser.add_argument('--data_att', default=1, type=float, help='cloud data size by attacker')
    parser.add_argument('--data_min', default=1, type=int, help='')
    parser.add_argument('--data_max', default=4, type=int, help='')
    parser.add_argument('--cost_def', default=5, type=int, help='')
    parser.add_argument('--cost_att', default=1, type=int, help='')

    parser.add_argument('--lamda', default=10, type=int, help='data coeficient of defender')
    parser.add_argument('--lamda_att', default=10, type=int, help='data coeficient of attacker')
    parser.add_argument('--mu', default=1, type=int, help='cost coeficient of defender')
    parser.add_argument('--mu_att', default=1, type=int, help='cost coeficient of attacker')

    parser.add_argument('--batch_size', default=64, type=int, help='DQN sampling:batch_size')
    parser.add_argument('--buffer_size', default=int(1e6), type=int, help='')
    parser.add_argument('--gamma', default=0.98, type=float, help='DQN: gamma')
    parser.add_argument('--betta', default=1, type=float, help='CH: betta'
    'When betta=infinity the observer believes that agents are perfect optimizers, as betta=0 the observer believes the other agents are acting randomly.')
    parser.add_argument('--learning_rate', default=5e-2, type=float, help='DQN: learning_rate')

    parser.add_argument('--ct', default='[1,2]' , type=str, help='Defender cost at each node in Attack Graph')
    parser.add_argument('--ct_att', default='[9,8]' , type=str, help='Attacker cost at each node in Attack Graph')
    parser.add_argument('--p_ag', default='[0.1,0.2]' , type=str, help='Probabilty vector of the edges in Attack Graph')
    parser.add_argument('--p_d', default='[0.1,0.2]' , type=str, help='Probabilty vector of defending nodes by attacker')
    parser.add_argument('--p_expl', default='[0.1,0.2]' , type=str, help='Probabilty vector of successfull exploitation')
    
    parser.add_argument('--eps_start', default=1, type=float, help='DQN: max epsilon of e-greedy (greater value does more exploration)')
    parser.add_argument('--eps_end', default=0.05, type=float, help='DQN: min epsilon of e-greedy')
    parser.add_argument('--eps_decay', default=2, type=float, help='DQN: multiplicative factor for decreasing epsilon')
    parser.add_argument('--my_seed', default=101, type=int, help='')

    parser.add_argument(
        '--double_dqn',
        action='store_true',
        help='Use Double DQN bootstrap (CHT–Double DQN for models 4–6).',
    )
    parser.add_argument(
        '--double_dqn_att',
        action='store_true',
        help='Use Double DQN bootstrap for the attacker.',
    )
    parser.add_argument(
        '--use_dueling',
        action='store_true',
        help='Use a dueling Q-network architecture for the defender.',
    )
    parser.add_argument(
        '--results_root',
        default='',
        type=str,
        help='Override results root (default: results/cases1_4, or results/cases5_8 when double/dueling flags set).',
    )

    return parser.parse_args()




if __name__ == '__main__':
    arg = parse_arguments()

    # Hyper-parameters
    model_CH = arg.model_CH
    node_max = arg.node_max
    # max_episodes = arg.max_episodes
    time_max = arg.time_max
    update_interval = arg.update_interval
    data = arg.data
    data_att = arg.data_att

    data_min = arg.data_min
    data_max = arg.data_max
    cost_def = arg.cost_def
    cost_att = arg.cost_att

    lamda = arg.lamda
    lamda_att = arg.lamda_att
    mu = arg.mu
    mu_att = arg.mu_att

    my_seed = arg.my_seed
    learning_rate = arg.learning_rate
    eps_start = arg.eps_start
    eps_end = arg.eps_end
    eps_decay = arg.eps_decay
    batch_size = arg.batch_size
    buffer_size = arg.buffer_size
    gamma = arg.gamma
    betta = arg.betta
    double_dqn = bool(arg.double_dqn)
    double_dqn_att = bool(arg.double_dqn_att)
    use_dueling = bool(arg.use_dueling)
    results_root = arg.results_root.strip()
    if not results_root:
        results_root = (
            "results/cases5_8"
            if (double_dqn or double_dqn_att or use_dueling)
            else "results/cases1_4"
        )

    p_ag = np.array(ast.literal_eval(arg.p_ag))
    p_d = np.array(ast.literal_eval(arg.p_d))
    p_expl = np.array(ast.literal_eval(arg.p_expl))
    ct = np.array(ast.literal_eval(arg.ct))
    ct_att = np.array(ast.literal_eval(arg.ct_att))

    # device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    np.random.seed(seed=my_seed)
    # tf.set_random_seed(seed=my_seed)
    torch.manual_seed(seed=my_seed)  # Sets seed for PyTorch RNG
    torch.cuda.manual_seed_all(seed=my_seed)  # Sets seeds of GPU RNG

    losses = []
    # prospect_th_aggr = PTtheory_Aggregation()

    state_size = node_max
    # action_size = 1
    action_size = node_max
    action_max = node_max
    expected_Q_CH_ratio = np.ones(action_size)

    ## attack graph's probabilities
    p_ag = np.zeros((node_max))
    p_ag_numer = np.zeros((node_max))
    p_d = np.zeros((node_max))
    p_d_numer = np.zeros((node_max))
    p_expl = np.zeros((node_max))
    p_expl_numer = np.zeros((node_max))
    # # def/att cost at each node
    # ct = p_ag * 10
    # ct_att = (1-p_ag) * 10

    # p_d = np.random.uniform(size=(1,node_max))
    
    env = env_defender.EnvDefender(state_size=state_size, action_size=action_size, model_CH=model_CH)
    # DQNtrain(env, state_size, agent)
    model = agent_defender.DQNDefender(input_dim=state_size, output_dim=action_size, seed=my_seed).to("cpu")
    agent = agent_defender.AgentDefender(
        env=env,
        state_size=state_size,
        action_size=action_size,
        action_max=action_max,
        buffer_size=buffer_size,
        seed=my_seed,
        lr=learning_rate,
        batch_size=batch_size,
        gamma=gamma,
        betta=betta,
        model_CH=model_CH,
        use_conv=False,
        double_dqn=double_dqn,
        use_dueling=use_dueling,
    )

    state_size_att = node_max
    losses_att = []
    env_att_name = 'APT'
    # action_size_att = 1
    action_size_att = node_max
    action_max_att = node_max
    expected_Q_CH_ratio_att = np.ones(action_size_att)

    env_att = env_attacker.EnvAttacker(state_size_att=state_size_att, action_size_att=action_size_att, model_CH=model_CH)
    ##"q_func: of ATTACKER for DQN algorithm. ###hiddens_value=[128] when dueling
    model_att = agent_attacker.DQNAttacker(input_dim=state_size_att, output_dim=action_size_att, seed=my_seed).to("cpu")
    agent_att = agent_attacker.AgentAttacker(
        env=env_att,
        state_size=state_size_att,
        action_size=action_size_att,
        action_max=action_max_att,
        buffer_size=buffer_size,
        seed=my_seed,
        lr=learning_rate,
        batch_size=batch_size,
        gamma=gamma,
        betta=betta,
        model_CH=model_CH,
        use_conv=False,
        double_dqn=double_dqn_att,
    )



    # learning_starts = 100
    prev_time = time.time()
    n_trainings = 0
    current_state = env.reset()
    next_state = np.zeros((state_size))
    # data_s = np.zeros((2, edge_max))
    total_loss = []
    eps = eps_start  # initialize epsilon
    current_state_att = env_att.reset()
    next_state_att = np.zeros((state_size))

    avg_reward = 0
    reward_history = []
    avg_reward_att = 0
    reward_history_att = []
    data_prot_history = []
    action_history = []
    action_att_history = []

    for time_step in range(time_max):
        agent.t_step += 1
        agent_att.t_step += 1

        # data = np.random.randint(data_min,node_max+1,size=node_max)+np.random.choice([1,-1])*np.random.rand(node_max)
        # data_att = np.random.randint(data_min,node_max+1,size=node_max)+np.random.choice([1,-1])*np.random.rand(node_max)
        data = np.arange(1, node_max+1) + np.random.choice([1,-1])*np.random.rand(node_max)
        data_att = data
        # data_att = random.uniform(0, 1) * data
        ct = data + np.random.choice([1,-1])*np.random.rand(node_max)
        ct[ct <= 0] = 0.001
        ct_att = data_att + np.random.choice([1,-1])*np.random.rand(node_max)
        ct_att[ct_att <= 0] = 0.001
        # ct = data + np.random.rand(node_max)
        # ct_att = data_att + np.random.rand(node_max)
        
        ## Salient actions are in the range of 0 to action_max # action_max=2 #####
        # action_att = agent_att.get_action(current_state=current_state_att, action_max=action_max,eps=eps)###
        action_att = agent_att.get_action(current_state=current_state_att, action_max=action_max, p_ag=p_ag,p_d=p_d,\
                                          p_expl=p_expl,expected_Q_CH_ratio=expected_Q_CH_ratio, eps=eps)###
        ## DEFENDER "get action from agent by e-greedy" 
        # action = agent.get_action(current_state=current_state, action_max=action_max,eps=eps)
        action = agent.get_action(current_state=current_state, action_max=action_max, p_ag=p_ag,p_d=p_d,\
                                   p_expl=p_expl,expected_Q_CH_ratio_att=expected_Q_CH_ratio_att, eps=eps)
        # print("action,current_state",action,current_state)

        # print("eps",eps)
        # print("action_att,current_state_att",action_att,current_state_att)
    
        
        ## ATTACKER
        next_state_att, reward_att, _ = env_att.step(action_att=action_att,action=action,data_att=data_att,data=data,\
            p_ag=p_ag, p_d=p_d, p_expl=p_expl, ct_att=ct_att, lamda_att=lamda_att,mu_att=mu_att,expected_Q_CH_ratio=np.ones(action_size),g_pt=1)
        
        if len(agent_att.replay_buffer_att) <= batch_size:
            agent_att.remember(current_state=current_state_att, action=action_att,\
                reward=reward_att, next_state=next_state_att, expected_Q_CH_ratio=np.ones(action_size))
            expected_Q_CH_ratio_att = np.ones(action_size_att)
        else:
            loss_att, expected_Q_CH_ratio_att = agent_att.learn(current_state=current_state_att, action=action_att,\
                reward=reward_att, next_state=next_state_att, expected_Q_CH_ratio=np.ones(action_size), p_ag=p_ag,\
                p_d=p_d, p_expl=p_expl)
        # print("next_state_att, reward_att",next_state_att, reward_att)



        ## DEFENDER
        next_state, reward, _, data_prot = env.step(action=action,action_att=action_att,data=data,data_att=data_att,\
            p_ag=p_ag, p_d=p_d, p_expl=p_expl, ct=ct, lamda=lamda, mu=mu, expected_Q_CH_ratio_att=expected_Q_CH_ratio_att,g_pt=1)
        # "save experience in replay buffer, build experience reply
        # # If enough samples are available in memory, get random subset and learn q
        
        if len(agent.replay_buffer) <= batch_size:
            agent.remember(current_state=current_state, action=action,\
            reward=reward, next_state=next_state, expected_Q_CH_ratio_att=np.ones(action_size_att),\
            data=data, ct=ct, lamda=lamda, mu=mu)
            expected_Q_CH_ratio = np.ones(action_size)
        else:
            loss, expected_Q_CH_ratio = agent.learn(current_state=current_state, action=action,\
                reward=reward, next_state=next_state, expected_Q_CH_ratio_att=expected_Q_CH_ratio_att, p_ag=p_ag,\
                p_d=p_d, p_expl=p_expl, data=data, ct=ct, lamda=lamda, mu=mu)
        # print("next_state, reward",next_state, reward)   
         
        ## update p_ag of attack graph, def/att cost at each node, weighted ratio of data_prot
        data_prot_w_ratio = np.sum(data_prot*data)/np.sum(data)
        p_ag_numer[action_att] = p_ag_numer[action_att] + 1
        p_ag_denom = time_step + 1
        p_d_numer[action] = p_d_numer[action] + 1
        p_d_denom = time_step + 1
        p_expl_numer = p_expl_numer + (np.ones(node_max)-data_prot)
        p_expl_denom = time_step + 1

        if len(agent.replay_buffer) <= batch_size or len(agent_att.replay_buffer_att) <= batch_size:
            p_ag = p_ag_numer/p_ag_denom
            p_d = p_d_numer/p_d_denom
            p_expl = p_expl_numer/p_expl_denom

        if time_step % update_interval == 0:
            p_ag = p_ag_numer/p_ag_denom
            p_d = p_d_numer/p_d_denom
            p_expl = p_expl_numer/p_expl_denom
            # ct = p_ag * 10
            # ct_att = (1-p_ag) * 10    

        # next_state = next_state_att
        current_state = next_state_att
        # "Attacker agent
        current_state_att = next_state_att

        eps = max(((time_max - eps_decay*time_step)/time_max), eps_end)
        # eps = 0.01


        reward_history.append(reward)
        avg_reward = np.mean(reward_history[-100:])
        reward_history_att.append(reward_att)
        avg_reward_att = np.mean(reward_history_att[-100:])
        # data_prot_history.append(data_prot)
        data_prot_history.append(data_prot_w_ratio)
        action_history.append(action)
        action_att_history.append(action_att)
        
    print("Avg data protection",np.mean(data_prot_history))
    train_start = time_max//eps_decay
    data_prot_history_eval = data_prot_history[train_start:]
    print("Avg data protection in evaluation phase",np.mean(data_prot_history_eval))
    print("Avg data protection in 1000 timesteps of evaluation phase",np.mean(data_prot_history[1000:]))

    ########## folder path #######
    # folder_path = "results/DQN-CH-level1 def DQN p_ag_att DQN p_ag/hyper1_c=d_ca=da/"+str(node_max)+" node/"
    # folder_path = "results/DQN-CH-level1 def DQN p_expl_att random/data_arange/att_policy_same/hyper1_c=d_ca=da/"+str(node_max)+" node/"
   
    if model_CH == 4:
        main_folder = f"{results_root}/DQN-CH-level1 def DQN p_expl_att random/"
    elif model_CH == 5:
        main_folder = f"{results_root}/DQN-CH-level1 def DQN p_expl_att DQN/"
    elif model_CH == 6:
        main_folder = f"{results_root}/DQN-CH-level1 def DQN p_expl_att DQN p_expl/"
    else:
        raise ValueError(f"main_ch1_salient.py expects model_CH in {{4,5,6}}, got {model_CH}")

    tag_parts = []
    if double_dqn:
        tag_parts.append("double")
    if double_dqn_att:
        tag_parts.append("att_double")
    if use_dueling:
        tag_parts.append("dueling")
    if tag_parts:
        main_folder = main_folder.rstrip("/") + "_" + "_".join(tag_parts) + "/"

    folder_path = main_folder + "new/"+str(node_max)+" node/"
    file_name = folder_path+str(my_seed-100)+".txt"

    lines = ["Avg data protection", "Avg data protection in evaluation phase", "Avg data protection in 1000 timesteps of evaluation phase"]
    values = [str(np.mean(data_prot_history)), str(np.mean(data_prot_history_eval)), str(np.mean(data_prot_history[1000:]))]
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))
        
    with open(file_name, 'w') as f:
        f.write('    parser.add_argument(--eps_end, default='+str(eps_end)+', type=float, help=DQN: min epsilon of e-greedy)'+'\n')
        f.write('    parser.add_argument(--eps_decay, default='+str(eps_decay)+', type=float, help=DQN: multiplicative factor for decreasing epsilon)'+'\n')
        f.write('    parser.add_argument(--my_seed, default='+str(my_seed)+', type=int, help=)'+'\n'+'\n'+'\n'+'\n')
        for line, value in zip(lines, values):
            f.write(line+' '+value+'\n')
    f.close

    action_frequency = Counter(action_history[train_start:])
    action_att_frequency = Counter(action_att_history[train_start:])
    
    file_name_1 = folder_path+"/action_frequencies_"+str(my_seed-100)+".txt"
    if not os.path.exists(os.path.dirname(file_name_1)):
        os.makedirs(os.path.dirname(file_name_1))
    with open(file_name_1, 'w') as file:
        file.write('Defender Action Frequencies in evaluation:\n')
        for action, frequency in action_frequency.items():
            file.write(f'Action {action}: {frequency}\n')
        file.write('\n')
        file.write('Attacker Action Frequencies in evaluation:\n')
        for action, frequency in action_att_frequency.items():
            file.write(f'Action {action}: {frequency}\n')

    running_avg = np.zeros(len(reward_history))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(reward_history[max(0, i-100):(i+1)])

    running_avg_att = np.zeros(len(reward_history_att))
    for i in range(len(running_avg_att)):
        running_avg_att[i] = np.mean(reward_history_att[max(0, i-100):(i+1)])

    # Persist reward curves for Fig 4 (running averages + raw histories).
    np.savez_compressed(
        folder_path + "reward_curves_" + str(my_seed - 100) + ".npz",
        reward_def=np.asarray(reward_history, dtype=np.float64),
        reward_att=np.asarray(reward_history_att, dtype=np.float64),
        running_avg_def=running_avg,
        running_avg_att=running_avg_att,
    )

