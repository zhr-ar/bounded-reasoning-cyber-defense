import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque, Counter
import torch
import torch.nn as nn
import ast
from datetime import datetime
import pytz
import time 
import json
from pathlib import Path

# Import custom modules
from envs import env_attacker, env_defender
from agents import agent_attacker, agent_defender
from buffer import buffer
import logging

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = REPOSITORY_ROOT / ".runtime" / "logs" / "web_matched_simulations"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=LOG_DIR / "dqn.log", level=logging.DEBUG)

# Set timezone to US Eastern Timezone
us_eastern = pytz.timezone('US/Eastern')
now = datetime.now(tz=us_eastern)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Cognitive Hierarchy Theory-infused Deep Reinforcement Learning for Attack Graph-based Cloud Data Protection')
    
    # Model selection arguments
    parser.add_argument('--model_CH', default=8, type=int, help='Model selection for the defender: 1-6 for DEF DQN AG; 7-9 for DEF DQN')
    parser.add_argument('--model_PT', default=0, type=int, help='Prospect Theory Model: 0 for Not PT, 1 for PT')

    # Environment setup arguments
    parser.add_argument('--node_max', default=6, type=int, help='Number of edge servers (nodes)')
    parser.add_argument('--time_max', default=40, type=int, help='Maximum number of time steps')
    parser.add_argument('--update_interval', default=5, type=int, help='Update interval')
    parser.add_argument('--data', default=1, type=float, help='Cloud data size by defender')
    parser.add_argument('--data_att', default=1, type=float, help='Cloud data size by attacker')
    parser.add_argument('--data_min', default=1, type=int, help='Minimum cloud data size')
    parser.add_argument('--data_max', default=4, type=int, help='Maximum cloud data size')
    parser.add_argument('--cost_def', default=5, type=int, help='Cost for defender')
    parser.add_argument('--cost_att', default=1, type=int, help='Cost for attacker')
    parser.add_argument('--lamda', default=10, type=int, help='Data coefficient for defender')
    parser.add_argument('--lamda_att', default=10, type=int, help='Data coefficient for attacker')
    parser.add_argument('--mu', default=1, type=int, help='Cost coefficient for defender')
    parser.add_argument('--mu_att', default=1, type=int, help='Cost coefficient for attacker')

    # Training parameters
    parser.add_argument('--batch_size', default=5, type=int, help='Batch size for DQN sampling')
    parser.add_argument('--buffer_size', default=int(1e3), type=int, help='Replay buffer size')
    parser.add_argument('--gamma', default=0.98, type=float, help='Discount factor for DQN')
    parser.add_argument('--betta', default=1, type=float, help='Cognitive Hierarchy parameter')
    parser.add_argument('--learning_rate', default=5e-2, type=float, help='Learning rate for DQN')

    # Epsilon-greedy parameters
    parser.add_argument('--eps_start', default=0.9, type=float, help='Starting epsilon for e-greedy policy')
    parser.add_argument('--eps_end', default=0.05, type=float, help='Minimum epsilon for e-greedy policy')
    parser.add_argument('--eps_decay', default=0.93, type=float, help='Decay factor for epsilon')

    # Seed for reproducibility
    parser.add_argument('--my_seed', default=121, type=int, help='Random seed of defender')
    parser.add_argument('--my_seed_att', default=101, type=int, help='Random seed of attacker')

    # Prospect Theory parameters
    parser.add_argument('--a_pt', default=0.5, type=float, help='Prospect Theory: power in Framing gain function')
    parser.add_argument('--b_pt', default=0.6, type=float, help='Prospect Theory: power in Framing loss function')
    parser.add_argument('--l_pt', default=1.5, type=float, help='Prospect Theory: coefficient in Framing loss function')
    parser.add_argument('--g_pt', default=0.6, type=float, help='Prospect Theory: power in Weighting function')

    # Attack Graph parameters
    parser.add_argument('--ct', default='[1,2]', type=str, help='Defender cost at each node in Attack Graph')
    parser.add_argument('--ct_att', default='[9,8]', type=str, help='Attacker cost at each node in Attack Graph')
    parser.add_argument('--p_ag', default='[0.1,0.2]', type=str, help='Probability vector of the edges in Attack Graph')
    parser.add_argument('--p_d', default='[0.1,0.2]', type=str, help='Probability vector of defending nodes by attacker')
    parser.add_argument('--p_expl', default='[0.1,0.2]', type=str, help='Probability vector of successful exploitation')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    # Extract arguments
    model_CH = args.model_CH
    model_PT = args.model_PT
    node_max = args.node_max
    time_max = args.time_max
    update_interval = args.update_interval
    data = args.data
    data_att = args.data_att
    data_min = args.data_min
    data_max = args.data_max
    cost_def = args.cost_def
    cost_att = args.cost_att
    lamda = args.lamda
    lamda_att = args.lamda_att
    mu = args.mu
    mu_att = args.mu_att
    my_seed = args.my_seed
    my_seed_att = args.my_seed_att
    learning_rate = args.learning_rate
    eps_start = args.eps_start
    eps_end = args.eps_end
    eps_decay = args.eps_decay
    batch_size = args.batch_size
    buffer_size = args.buffer_size
    gamma = args.gamma
    betta = args.betta
    a_pt = args.a_pt
    b_pt = args.b_pt
    l_pt = args.l_pt
    g_pt = args.g_pt
    p_ag = np.array(ast.literal_eval(args.p_ag))
    p_d = np.array(ast.literal_eval(args.p_d))
    p_expl = np.array(ast.literal_eval(args.p_expl))
    ct = np.array(ast.literal_eval(args.ct))
    ct_att = np.array(ast.literal_eval(args.ct_att))

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # # Set random seed for reproducibility
    # np.random.seed(my_seed)
    # torch.manual_seed(my_seed)
    # torch.cuda.manual_seed_all(my_seed)

    state_size = node_max
    action_size = node_max
    action_max = node_max
    expected_Q_CH_ratio = np.ones(action_size)

    # Initialize attack graph probabilities and cost at each node
    p_ag = np.zeros(node_max)
    p_ag_numer = np.zeros(node_max)
    p_d = np.zeros(node_max)
    p_d_numer = np.zeros(node_max)
    p_expl = np.zeros(node_max)
    p_expl_numer = np.zeros(node_max)

    # Environment initialization
    env = env_defender.EnvDefender(state_size=state_size, action_size=action_size, model_CH=model_CH)
    model = agent_defender.DQNDefender(input_dim=state_size, output_dim=action_size, seed=my_seed).to("cpu")
    agent = agent_defender.AgentDefender(env=env, state_size=state_size, action_size=action_size, action_max=action_max,
                                         buffer_size=buffer_size, seed=my_seed, lr=learning_rate, batch_size=batch_size, 
                                         gamma=gamma, betta=betta, model_CH=model_CH, use_conv=False)

    state_size_att = node_max
    action_size_att = node_max
    action_max_att = node_max
    expected_Q_CH_ratio_att = np.ones(action_size_att)
    env_att = env_attacker.EnvAttacker(state_size_att=state_size_att, action_size_att=action_size_att, model_CH=model_CH)
    model_att = agent_attacker.DQNAttacker(input_dim=state_size_att, output_dim=action_size_att, seed=my_seed_att).to("cpu")
    agent_att = agent_attacker.AgentAttacker(env=env_att, state_size=state_size_att, action_size=action_size_att, action_max=action_max_att,
                                             buffer_size=buffer_size, seed=my_seed_att, lr=learning_rate, batch_size=batch_size, 
                                             gamma=gamma, betta=betta, model_CH=model_CH, use_conv=False)
    
    start_time = datetime.now(tz=us_eastern)

    max_timesteps = int(1e8)
    print_freq = 10
    prev_time = time.time()
    n_trainings = 0
    current_state = env.reset()
    next_state = np.ones(state_size)
    eps = eps_start  # Initialize epsilon for e-greedy policy
    current_state_att = env_att.reset()
    next_state_att = np.ones(state_size)
    running_avg_reward = np.zeros(time_max)
    running_avg_reward_att = np.zeros(time_max)

    avg_reward = 0
    # reward_history = []
    avg_reward_att = 0
    # reward_att_history = []
    # data_prot_history = []
    loss_history = []
    loss_att_history = []
    # action_history = []
    # action_att_history = []

    current_state_history = []
    current_state_att_history = []
    eps_history = []
    action_history = []
    reward_history = []
    action_att_history = []
    reward_att_history = []
    expected_Q_CH_ratio_history = []
    expected_Q_CH_ratio_att_history = []
    data_prot_history = []
    p_expl_history = []

    data_history = []
    data_att_history = []
    ct_history = []
    ct_att_history = []

    # Store args
    global_variables = {
        'node_max': node_max,
        'time_max': time_max,
        'update_interval': update_interval,
        'lamda': lamda,
        'lamda_att': lamda_att,
        'mu': mu,
        'mu_att': mu_att,
        'my_seed': my_seed,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'buffer_size': buffer_size,
        'gamma': gamma,
        'betta': betta,
        'eps_start': eps_start,
        'eps_end': eps_end,
        'eps_decay': eps_decay,
        'model_CH': model_CH
    }

    for time_step in range(time_max):
        agent.t_step += 1
        agent_att.t_step += 1

        data = np.arange(1, node_max + 1) + np.random.choice([1, -1]) * np.random.rand(node_max)
        data_att = data
        ct = data + np.random.choice([1, -1]) * np.random.rand(node_max)
        ct[ct <= 0] = 0.001
        ct_att = data_att + np.random.choice([1, -1]) * np.random.rand(node_max)
        ct_att[ct_att <= 0] = 0.001

        data_history.append(data)
        data_att_history.append(data_att)
        ct_history.append(ct)
        ct_att_history.append(ct_att)

        # ATTACKER selects action
        action_att = agent_att.get_action(current_state=current_state_att, action_max=action_max, p_ag=p_ag, p_d=p_d, p_expl=p_expl, expected_Q_CH_ratio=np.ones(action_size), eps=eps)
        # DEFENDER selects action
        action = agent.get_action(current_state=current_state, action_max=action_max, p_ag=p_ag, p_d=p_d, p_expl=p_expl, expected_Q_CH_ratio_att=np.ones(action_size_att), eps=eps)

        # ATTACKER
        next_state_att, reward_att, _ = env_att.step(action_att=action_att, action=action, data_att=data_att, data=data, p_ag=p_ag, p_d=p_d, p_expl=p_expl, ct_att=ct_att, lamda_att=lamda_att, mu_att=mu_att, expected_Q_CH_ratio=np.ones(action_size), g_pt=g_pt)
        if len(agent_att.replay_buffer_att) <= batch_size:
            agent_att.remember(current_state=current_state_att, action=action_att, reward=reward_att, next_state=next_state_att, expected_Q_CH_ratio=np.ones(action_size))
            expected_Q_CH_ratio_att = np.ones(action_size_att)
        else:
            loss_att, expected_Q_CH_ratio_att = agent_att.learn(current_state=current_state_att, action=action_att, reward=reward_att, next_state=next_state_att, expected_Q_CH_ratio=np.ones(action_size), p_ag=p_ag, p_d=p_d, p_expl=p_expl)
            loss_att_history.append(loss_att)

        # DEFENDER
        next_state, reward, _, data_prot = env.step(action=action, action_att=action_att, data=data, data_att=data_att, p_ag=p_ag, p_d=p_d, p_expl=p_expl, ct=ct, lamda=lamda, mu=mu, expected_Q_CH_ratio_att=np.ones(action_size_att), g_pt=g_pt)
        if len(agent.replay_buffer) <= batch_size:
            agent.remember(current_state=current_state, action=action, reward=reward, next_state=next_state, expected_Q_CH_ratio_att=np.ones(action_size_att), data=data, ct=ct, lamda=lamda, mu=mu)
            expected_Q_CH_ratio = np.ones(action_size)
        else:
            loss, expected_Q_CH_ratio = agent.learn(current_state=current_state, action=action, reward=reward, next_state=next_state, expected_Q_CH_ratio_att=np.ones(action_size_att), p_ag=p_ag, p_d=p_d, p_expl=p_expl, data=data, ct=ct, lamda=lamda, mu=mu)
            loss_history.append(loss)

        # update p_ag of attack graph and weighted ratio of data_prot
        data_prot_w_ratio = np.sum(data_prot * data) / np.sum(data)
        p_ag_numer[action_att] = p_ag_numer[action_att] + 1
        p_ag_denom = time_step + 1
        p_d_numer[action] = p_d_numer[action] + 1
        p_d_denom = time_step + 1
        p_expl_numer = p_expl_numer + (np.ones(node_max) - data_prot)
        p_expl_denom = time_step + 1

        if len(agent.replay_buffer) <= batch_size or len(agent_att.replay_buffer_att) <= batch_size:
            p_ag = p_ag_numer / p_ag_denom
            p_d = p_d_numer / p_d_denom
            p_expl = p_expl_numer / p_expl_denom

        if time_step % update_interval == 0:
            p_ag = p_ag_numer / p_ag_denom
            p_d = p_d_numer / p_d_denom
            p_expl = p_expl_numer / p_expl_denom
        
        eps = max(eps_start * (eps_decay ** time_step), eps_end)
        logging.info(f"time_step {time_step}: eps {eps}, batch_size {batch_size}")    
        logging.info(f"len att {len(agent_att.replay_buffer_att)}, p_expl {p_expl} expected_Q_CH_ratio_att {expected_Q_CH_ratio_att}")    
        logging.info(f"action {action}, action_att {action_att}, reward {reward}, reward_att {reward_att}") 

        current_state = next_state_att
        current_state_att = next_state_att
        
        reward_history.append(reward)
        # avg_reward = np.mean(reward_history[-100:])
        reward_att_history.append(reward_att)
        # avg_reward_att = np.mean(reward_att_history[-100:])
        # running_avg_reward[time_step] = np.mean(reward_history[max(0, time_step - 100):(time_step + 1)])
        # running_avg_reward_att[time_step] = np.mean(reward_att_history[max(0, time_step - 100):(time_step + 1)])
        data_prot_history.append(data_prot_w_ratio)
        action_history.append(action)
        action_att_history.append(action_att)

        current_state_history.append(current_state)
        current_state_att_history.append(current_state_att)
        eps_history.append(float(eps))
        # expected_Q_CH_ratio_history.append(expected_Q_CH_ratio.tolist())
        expected_Q_CH_ratio_att_history.append(expected_Q_CH_ratio_att.tolist())
        p_expl_history.append(p_expl.tolist())

    # Calculate the elapsed time
    end_time = datetime.now(tz=us_eastern)
    elapsed_time = (end_time - start_time).total_seconds()
    avg_data_prot_w_ratio = np.mean(data_prot_history)
    score = int(100*float(f"{avg_data_prot_w_ratio:.2f}"))
    logging.info(f"-------------- score {score}, model_CH {model_CH} --------")  
    logging.info(f"-----------------------------------------------------------------------------------------")  


    print("Avg data protection", np.mean(data_prot_history))
    # train_start = time_max // eps_decay
    # data_prot_history_eval = data_prot_history[train_start:]
    # print("Avg data protection in evaluation phase", np.mean(data_prot_history_eval))
    # print("Avg data protection in 1000 timesteps of evaluation phase", np.mean(data_prot_history[1000:]))

    # folder path
    if model_CH == 1:
        main_folder = "results_rerun/def DQN p_expl_att random/"
    elif model_CH == 2:
        main_folder = "results_rerun/def DQN p_expl_att DQN/"
    elif model_CH == 3:
        main_folder = "results_rerun/def DQN p_expl_att DQN p_expl/"
    elif model_CH == 7:
        main_folder = "results_rerun/def DQN_att random/"
    elif model_CH == 8:
        main_folder = "results_rerun/def DQN_att DQN/"
    elif model_CH == 9:
        main_folder = "results_rerun/def DQN_att DQN p_expl/"
    else:
        raise ValueError(f"main.py expects model_CH in {{1,2,3,7,8,9}}, got {model_CH}")

    folder_path = main_folder + "new/" + str(node_max) + " node/"
    file_name = folder_path + str(my_seed - 100) + ".txt"

    lines = ["Avg data protection"]
    values = [str(np.mean(data_prot_history))]
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))

    with open(file_name, 'w') as f:
        f.write('parser.add_argument(--eps_end, default=' + str(eps_end) + ', type=float, help=DQN: min epsilon of e-greedy)\n')
        f.write('parser.add_argument(--eps_decay, default=' + str(eps_decay) + ', type=float, help=DQN: multiplicative factor for decreasing epsilon)\n')
        f.write('parser.add_argument(--my_seed, default=' + str(my_seed) + ', type=int, help=)\n\n\n\n')
        for line, value in zip(lines, values):
            f.write(line + ' ' + value + '\n')

    action_frequency = Counter(action_history)
    action_att_frequency = Counter(action_att_history)

    # Save action frequencies to file
    with open(folder_path + "/action_frequencies_" + str(my_seed - 100) + ".txt", 'w') as file:
        file.write('Defender Action Frequencies:\n')
        for action, frequency in action_frequency.items():
            file.write(f'Action {action}: {frequency}\n')
        file.write('\n')
        file.write('Attacker Action Frequencies:\n')
        for action, frequency in action_att_frequency.items():
            file.write(f'Action {action}: {frequency}\n')

    # Save average reward to file
    with open(folder_path + 'average_reward_' + str(my_seed - 100) + '.txt', 'w') as f:
        for i, reward in enumerate(reward_history):
            f.write(f"Time step {i + 1}: {reward}\n")
        f.write(f"\nAverage Reward: {avg_reward}\n")

    with open(folder_path + 'average_reward_attacker_' + str(my_seed - 100) + '.txt', 'w') as f:
        for i, reward_att in enumerate(reward_att_history):
            f.write(f"Time step {i + 1}: {reward_att}\n")
        f.write(f"\nAverage Reward: {avg_reward_att}\n")

    # # Save convergence analysis to file
    # with open(folder_path + 'convergence_analysis_' + str(my_seed - 100) + '.txt', 'w') as f:
    #     for i, avg in enumerate(running_avg_reward):
    #         f.write(f"Time step {i + 1}: {avg}\n")

    # with open(folder_path + 'convergence_analysis_attacker_' + str(my_seed - 100) + '.txt', 'w') as f:
    #     for i, avg_att in enumerate(running_avg_reward_att):
    #         f.write(f"Time step {i + 1}: {avg_att}\n")

    save_path = os.path.join(folder_path, f"my_seed_def_{my_seed}_score_{score}_model_CH_{model_CH}.json")         
    with open(save_path, 'w') as f:
        json.dump({
            'current_state_history': current_state_history,
            'current_state_att_history': current_state_att_history,
            'eps_history': eps_history,
            'action_history': [int(a) for a in action_history],  # cast to int
            'reward_history': reward_history,
            'action_att_history': [int(a_att) for a_att in action_att_history],  # cast to int
            'reward_att_history': reward_att_history,                 
            # 'expected_Q_CH_ratio_history': [expected_Q_CH_ratio.tolist() for expected_Q_CH_ratio in expected_Q_CH_ratio_history],
            # 'expected_Q_CH_ratio_att_history': [expected_Q_CH_ratio_att.tolist() for expected_Q_CH_ratio_att in expected_Q_CH_ratio_att_history],
            'data_history': [data.tolist() for data in data_history],  # cast to list if it's a numpy array
            'data_att_history': [data_att.tolist() for data_att in data_att_history],  # cast to list
            'ct_history': [ct.tolist() for ct in ct_history],  # cast to list
            'ct_att_history': [ct_att.tolist() for ct_att in ct_att_history],  # cast to list
            'data_prot_history': data_prot_history,
            'p_expl_history': p_expl_history,  # cast to list
            'avg_data_prot_w_ratio': float(avg_data_prot_w_ratio),  # cast to float
            'score': int(score),  # cast to int
            'elapsed_time': elapsed_time,
            'global_variables': global_variables               
        }, f)

    # x = [i + 1 for i in range(time_max)]
    # running_avg = np.zeros(len(reward_history))
    # for i in range(len(running_avg)):
    #     running_avg[i] = np.mean(reward_history[max(0, i - 100):(i + 1)])
    # plt.plot(x, running_avg)
    # plt.ylabel('Average of previous 100 rewards of Defender')
    # plt.xlabel('Time step')
    # plt.show()

    # running_avg_att = np.zeros(len(reward_att_history))
    # for i in range(len(running_avg_att)):
    #     running_avg_att[i] = np.mean(reward_att_history[max(0, i - 100):(i + 1)])
    # plt.plot(x, running_avg_att)
    # plt.ylabel('Average of previous 100 rewards of Attacker')
    # plt.xlabel('Time step')
    # plt.show()
