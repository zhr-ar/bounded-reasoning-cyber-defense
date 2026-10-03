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
from datetime import datetime, timedelta
from threading import Timer
import pytz
import time
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_session import Session
import jsonpickle
import pickle
import json
import threading
import ipaddress

# Import custom modules
# Agents: corrected CHT Bellman + beta-softmax.
from envs import env_attacker, env_defender
from agents import agent_attacker, agent_defender
from buffer import buffer
from runtime_config import condition_paths, flask_secret_key

import logging

DATA_PATH, PICKLE_DIR, SESSION_DIR, LOG_FILE = condition_paths("reward_aware")
logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG)

# Set timezone to US Eastern Timezone
us_eastern = pytz.timezone('US/Eastern')
now = datetime.now(tz=us_eastern)

app = Flask(__name__, static_folder='static')
app.secret_key = flask_secret_key()
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = str(SESSION_DIR)
Session(app)

# Global variables/parameters 
node_max = 6
time_max = 40
update_interval = 5
lamda = 10
lamda_att = 10
mu = 1
mu_att = 1
my_seed = 101
learning_rate = 0.05
batch_size = 5
buffer_size = int(1e4)
gamma = 0.98
betta = 1
eps_start = 0.9  # High epsilon to encourage exploration initially
eps_end = 0.05  # Low epsilon for more deterministic actions later
eps_decay = 0.93 # Decay rate per round (or timestep)
model_CH = 8 # def DQN_att DQN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(seed=my_seed)
torch.manual_seed(seed=my_seed)
torch.cuda.manual_seed_all(seed=my_seed)

state_size = node_max
action_size = node_max
action_max = node_max

state_size_att = node_max
action_size_att = node_max
action_max_att = node_max

p_ag = np.zeros((node_max))
p_d = np.zeros((node_max))

def save_session_object(session_id, key, obj):
    file_path = os.path.join(PICKLE_DIR, f'{session_id}_{key}.pth')
    # if isinstance(obj, torch.nn.Module):
    #     torch.save(obj.state_dict(), file_path)
    # else:
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def load_session_object(session_id, key, obj=None):
    file_path = os.path.join(PICKLE_DIR, f'{session_id}_{key}.pth')
    if os.path.exists(file_path):
        # if isinstance(obj, torch.nn.Module):
        #     obj.load_state_dict(torch.load(file_path))
        #     return obj
        # else:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return None

def initialize_session():
    """Initialize the session variables only if the session is new."""
    session_id = session.sid  # Unique session ID

    env = env_defender.EnvDefender(state_size=state_size, action_size=action_size, model_CH=model_CH)
    model = agent_defender.DQNDefender(input_dim=state_size, output_dim=action_size, seed=my_seed).to("cpu")
    replay_buffer = buffer.BasicBuffer(seed=my_seed, max_size=buffer_size)
    agent = agent_defender.AgentDefender(env=env, state_size=state_size, action_size=action_size, action_max=action_max,
                                            replay_buffer=replay_buffer, buffer_size=buffer_size, seed=my_seed, lr=learning_rate, batch_size=batch_size,
                                            gamma=gamma, betta=betta, model_CH=model_CH, use_conv=False)
    
    env_att = env_attacker.EnvAttacker(state_size_att=state_size_att, action_size_att=action_size_att, model_CH=model_CH)
    model_att = agent_attacker.DQNAttacker(input_dim=state_size_att, output_dim=action_size_att, seed=my_seed).to("cpu")
    replay_buffer_att = buffer.BasicBuffer(seed=my_seed, max_size=buffer_size)
    agent_att = agent_attacker.AgentAttacker(env=env_att, state_size=state_size_att, action_size=action_size_att, action_max=action_max_att,
                                                replay_buffer_att=replay_buffer_att, buffer_size=buffer_size, seed=my_seed, lr=learning_rate, batch_size=batch_size,
                                                gamma=gamma, betta=betta, model_CH=model_CH, use_conv=False)
    logging.info(f"In initialize_session: session_id {session.sid}")
    
    session['visited'] = True   
    # Save to pickle files
    save_session_object(session_id, 'env', env)
    save_session_object(session_id, 'model', model)
    save_session_object(session_id, 'replay_buffer', replay_buffer)
    save_session_object(session_id, 'agent', agent)
    save_session_object(session_id, 'env_att', env_att)
    save_session_object(session_id, 'model_att', model_att)
    save_session_object(session_id, 'replay_buffer_att', replay_buffer_att)
    save_session_object(session_id, 'agent_att', agent_att)

    session['start_time'] = datetime.now(tz=us_eastern).isoformat()
    session['current_state'] = env.reset().tolist()
    session['next_state'] = env.reset().tolist()
    session['current_state_att'] = env_att.reset().tolist()
    session['next_state_att'] = env_att.reset().tolist()
    session['time_step'] = 0
    session['eps_start'] = eps_start
    session['eps_end'] = eps_end
    session['eps_decay'] = eps_decay
    session['eps'] = session['eps_start']
    session['rewards_matrix'] = np.zeros((node_max, node_max)).tolist()
    session['p_expl_numer'] = np.zeros((node_max)).tolist()
    session['p_expl'] = np.zeros((node_max)).tolist()
    session['expected_Q_CH_ratio'] = np.ones(node_max).tolist()
    session['expected_Q_CH_ratio_att'] = np.ones(node_max).tolist()
    session['avg_data_prot_w_ratio'] = 0
    session['score'] = 0

    session['current_state_history'] = []
    session['current_state_att_history'] = []
    session['eps_history'] = []
    session['action_history'] = []
    session['reward_history'] = []
    session['action_att_history'] = []
    session['reward_att_history'] = []
    # session['expected_Q_CH_ratio_history'] = []
    session['expected_Q_CH_ratio_att_history'] = []
    session['data_prot_history'] = []
    session['p_expl_history'] = []

    session['data_history'] = []
    session['data_att_history'] = []
    session['ct_history'] = []
    session['ct_att_history'] = []

    # Store global variables in session
    session['global_variables'] = {
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


@app.route('/')
def ch1():
    logging.info(f"In ch1(): Session ID: {session.sid}")

    if 'visited' not in session:
        # New session or session reset
        initialize_session()

    elif request.headers.get('Cache-Control') == 'max-age=0':
        # Page was refreshed, clear session and redirect to reinitialize
        session.clear()
        logging.info("Page refreshed, session cleared.")
        return redirect(url_for('ch1'))   
    return render_template('dqn.html')

# Function to shutdown the Flask application
def shutdown():
    os._exit(0)  # Force the process to exit immediately

@app.route('/dqn_node_selected', methods=['POST'])
def dqn_node_selected():
    session_id = session.sid
    if 'current_state' not in session:
        return jsonify({'message': 'Session expired. Please refresh the page to start a new game.'}), 400
    time_step = session['time_step']

    if time_step < time_max:
        # Load objects from pickle files
        env = load_session_object(session_id, 'env')
        model = load_session_object(session_id, 'model')
        replay_buffer = load_session_object(session_id, 'replay_buffer')
        agent = load_session_object(session_id, 'agent')
        env_att = load_session_object(session_id, 'env_att')
        model_att = load_session_object(session_id, 'model_att')
        replay_buffer_att = load_session_object(session_id, 'replay_buffer_att')
        agent_att = load_session_object(session_id, 'agent_att')

        current_state = np.array(session['current_state'])
        current_state_att = np.array(session['current_state_att'])
        time_step = session['time_step']
        eps_start = session['eps_start']
        eps_end = session['eps_end']
        eps_decay = session['eps_decay']
        eps = session['eps']
        rewards_matrix = np.array(session['rewards_matrix'])
        p_expl = np.array(session['p_expl'])
        p_expl_numer = np.array(session['p_expl_numer'])
        data = np.array(session['data'])
        data_att = np.array(session['data_att'])
        ct = np.array(session['ct'])
        ct_att = np.array(session['ct_att'])

        # Defender's action
        node_web = request.json
        action = node_web['node_id'] - 1  # Convert node_id to zero-based index

        # Attacker's action
        action_att = agent_att.get_action(current_state=current_state_att, action_max=node_max, p_ag=p_ag, p_d=p_d,
                                          p_expl=p_expl, expected_Q_CH_ratio=np.ones(node_max), eps=eps)
        # print(f"Time step {time_step}, Defender action {action} Attacker action {action_att} ")

        # ATTACKER
        next_state_att, reward_att, _ = env_att.step(action_att=action_att, action=action, data_att=data_att,
                                                     data=data, p_ag=p_ag, p_d=p_d, p_expl=p_expl,
                                                     ct_att=ct_att, lamda_att=lamda_att, mu_att=mu_att,
                                                     expected_Q_CH_ratio=np.ones(node_max), g_pt=1)
        if len(agent_att.replay_buffer_att) <= batch_size:
            agent_att.remember(current_state=current_state_att, action=action_att, reward=reward_att,
                               next_state=next_state_att, expected_Q_CH_ratio=np.ones(node_max))
            expected_Q_CH_ratio_att = np.ones(node_max)
        else:
            loss_att, expected_Q_CH_ratio_att = agent_att.learn(current_state=current_state_att, action=action_att, 
                                                                reward=reward_att, next_state=next_state_att,
                                                                expected_Q_CH_ratio=np.ones(node_max), p_ag=p_ag, p_d=p_d, p_expl=p_expl)
            logging.info(f"When len>: len att {len(agent_att.replay_buffer_att)}, expected_Q_CH_ratio_att {expected_Q_CH_ratio_att}")  
            agent_att.loss_att_history.append(loss_att)

        # DEFENDER
        next_state, reward, _, data_prot = env.step(action=action, action_att=action_att, data=data,
                                                    data_att=data_att, p_ag=p_ag, p_d=p_d, p_expl=p_expl,
                                                    ct=ct, lamda=lamda, mu=mu,
                                                    expected_Q_CH_ratio_att=np.ones(node_max), g_pt=1)
        if len(agent.replay_buffer) <= batch_size:
            agent.remember(
                current_state=current_state,
                action=action,
                reward=reward,
                next_state=next_state,
                expected_Q_CH_ratio_att=np.ones(node_max),
                data=data,
                ct=ct,
                lamda=lamda,
                mu=mu,
            )
            expected_Q_CH_ratio = np.ones(node_max)
        else:
            loss, expected_Q_CH_ratio = agent.learn(
                current_state=current_state,
                action=action,
                reward=reward,
                next_state=next_state,
                expected_Q_CH_ratio_att=np.ones(node_max),
                p_ag=p_ag,
                p_d=p_d,
                p_expl=p_expl,
                data=data,
                ct=ct,
                lamda=lamda,
                mu=mu,
            )
            agent.loss_history.append(loss)

        # update p_ag of attack graph, def/att cost at each node, weighted ratio of data_prot
        data_prot_w_ratio = np.sum(data_prot * data) / np.sum(data)
        p_expl_numer = p_expl_numer + (np.ones(node_max) - data_prot)
        p_expl_denom = time_step + 1
        if len(agent.replay_buffer) <= batch_size or len(agent_att.replay_buffer_att) <= batch_size:
            p_expl = p_expl_numer / p_expl_denom
        if time_step % update_interval == 0:
            p_expl = p_expl_numer / p_expl_denom

        # time_step += 1
        eps = max(eps_start * (eps_decay ** time_step), eps_end)
        logging.info(f"time_step {time_step}: eps {eps}, batch_size {batch_size}")    
        logging.info(f"len att {len(agent_att.replay_buffer_att)}, p_expl {p_expl} expected_Q_CH_ratio_att {expected_Q_CH_ratio_att}")    
        logging.info(f"action {action}, action_att {action_att}, reward {reward}, reward_att {reward_att}")       

        current_state = next_state
        current_state_att = next_state_att

        # If objects were modified, save them again
        save_session_object(session_id, 'env', env)
        save_session_object(session_id, 'model', model)
        save_session_object(session_id, 'replay_buffer', replay_buffer)
        save_session_object(session_id, 'agent', agent)
        save_session_object(session_id, 'env_att', env_att)
        save_session_object(session_id, 'model_att', model_att)
        save_session_object(session_id, 'replay_buffer_att', replay_buffer_att)
        save_session_object(session_id, 'agent_att', agent_att)

        session['current_state'] = current_state
        session['next_state'] = next_state
        session['current_state_att'] = current_state_att
        session['next_state_att'] = next_state_att
        session['time_step'] = time_step + 1
        session['eps_start'] = eps_start
        session['eps_end'] = eps_end
        session['eps_decay'] = eps_decay
        session['eps'] = eps
        session['rewards_matrix'] = rewards_matrix.tolist()
        session['p_expl'] = p_expl.tolist()
        session['p_expl_numer'] = p_expl_numer.tolist()
        session['expected_Q_CH_ratio'] = expected_Q_CH_ratio.tolist()
        session['expected_Q_CH_ratio_att'] = expected_Q_CH_ratio_att.tolist()

        session['current_state_history'].append(current_state)
        session['current_state_att_history'].append(current_state_att)
        session['eps_history'].append(float(eps))
        session['action_history'].append(int(action))
        session['reward_history'].append(float(reward))
        session['action_att_history'].append(int(action_att))
        session['reward_att_history'].append(float(reward_att))
        # session['expected_Q_CH_ratio_history'].append(expected_Q_CH_ratio.tolist())
        session['expected_Q_CH_ratio_att_history'].append(expected_Q_CH_ratio_att.tolist())
        session['data_prot_history'].append(float(data_prot_w_ratio))
        session['p_expl_history'].append(p_expl.tolist())

        if session['time_step'] >= time_max:
            # Calculate the elapsed time
            end_time = datetime.now(tz=us_eastern)
            start_time = datetime.fromisoformat(session['start_time'])
            elapsed_time = (end_time - start_time).total_seconds()
            participant_id = datetime.now(tz=us_eastern).strftime("%Y%m%d%H%M%S")
            avg_data_prot_w_ratio = sum(session['data_prot_history']) / len(session['data_prot_history'])
            session['avg_data_prot_w_ratio'] = avg_data_prot_w_ratio
            session['score'] = int(100*float(f"{avg_data_prot_w_ratio:.2f}"))
            logging.info(f"-------------- participant_id {participant_id}, score {session['score']}, model_CH {model_CH} --------")  
            logging.info(f"-----------------------------------------------------------------------------------------")  

            # Save results without IP address
            save_path = os.path.join(DATA_PATH, f"participant_{participant_id}_score_{session['score']}_model_CH_{model_CH}.json")         
            with open(save_path, 'w') as f:
                json.dump({
                    'current_state_history': session['current_state_history'],
                    'current_state_att_history': session['current_state_att_history'],
                    'eps_history': session['eps_history'],
                    'action_history': session['action_history'],
                    'reward_history': session['reward_history'],
                    'action_att_history': session['action_att_history'],
                    'reward_att_history': session['reward_att_history'],                   
                    # 'expected_Q_CH_ratio_history': session['expected_Q_CH_ratio_history'],
                    # 'expected_Q_CH_ratio_att_history': session['expected_Q_CH_ratio_att_history'],
                    'data_history': session['data_history'],
                    'data_att_history': session['data_att_history'],
                    'ct_history': session['ct_history'],
                    'ct_att_history': session['ct_att_history'],
                    'data_prot_history': session['data_prot_history'],
                    'p_expl_history': session['p_expl_history'],
                    'avg_data_prot_w_ratio': session['avg_data_prot_w_ratio'],
                    'score': session['score'],
                    'elapsed_time': elapsed_time,
                    'global_variables': session.get('global_variables')                  
                }, f)
            # Return a game over message and remove time step display
            response = jsonify({'message': 'Game Over! Thank you for participating.',
                                'avg_data_prot_w_ratio': float(f"{avg_data_prot_w_ratio:.2f}"),
                                'participant_id': int(f"{participant_id}")
                                })
            # Schedule the shutdown
            threading.Timer(3, shutdown).start()
            return response

        avg_data_prot_w_ratio = sum(session['data_prot_history']) / len(session['data_prot_history'])  
        response = jsonify({
            'attacker_choice': int(action_att + 1),  # Convert back to one-based index for the frontend
            'reward': float(f"{reward:.2f}"),  # Ensure the reward is converted to a JSON serializable type
            'rewards_matrix': rewards_matrix.tolist(),
            "avg_data_prot_w_ratio": float(f"{avg_data_prot_w_ratio:.2f}"),
            'time_step': session['time_step'],  
            'time_max': int(time_max)
        })

        return response
    else:
        return jsonify({'message': 'Game over. Maximum timesteps reached.'})


# Route to send potential rewards matrix 
@app.route('/get_potential_rewards', methods=['GET'])
def get_potential_rewards():
    session_id = session.sid
    if 'time_step' not in session:
        session['time_step'] = 0  # or another default value
    elif session['time_step'] >= time_max:
        return jsonify({'message': 'Game over. Maximum timesteps reached.'}), 400
    if 'p_expl_numer' not in session:
        session['p_expl_numer'] = np.zeros((node_max)).tolist()
    if 'p_expl' not in session:
        session['p_expl'] = np.zeros((node_max)).tolist()
    env = load_session_object(session_id, 'env')

    # Update data, data_att, ct, and ct_att for this timestep
    data = np.arange(1, node_max + 1) + np.random.choice([1, -1]) * np.random.rand(node_max)
    data_att = data
    ct = data + np.random.choice([1, -1]) * np.random.rand(node_max)
    ct[ct <= 0] = 0.001
    ct_att = data_att + np.random.choice([1, -1]) * np.random.rand(node_max)
    ct_att[ct_att <= 0] = 0.001

    logging.info(f"data {data}, data_att {data_att}")  
    logging.info(f"ct {ct}, ct_att {ct_att}")  

    session['data'] = data.tolist()
    session['data_att'] = data_att.tolist()
    session['ct'] = ct.tolist()
    session['ct_att'] = ct_att.tolist()
    p_expl_numer = np.array(session['p_expl_numer'])
    p_expl = np.array(session['p_expl'])

    session['data_history'].append(session['data'])
    session['data_att_history'].append(session['data_att'])
    session['ct_history'].append(session['ct'])
    session['ct_att_history'].append(session['ct_att'])

    # Calculate potential rewards matrix
    rewards_matrix = env.calculate_potential_rewards_matrix(data=data, data_att=data_att, p_ag=p_ag, p_d=p_d, p_expl=p_expl,
                                                            ct=ct, lamda=lamda, mu=mu)
    session['rewards_matrix'] = rewards_matrix.tolist()

    response = jsonify({
        'rewards_matrix': rewards_matrix.tolist(),
        'time_step': int(session['time_step']+1),
        'time_max': int(time_max)
    })
    return response


if __name__ == '__main__':
    # args, unknown = parse_arguments()
    # initialize_game()
    app.run(host='0.0.0.0', port=5001, debug=True)


    # # in paper, we ran the experiments for model_CH=4,5,7,8
    # if model_CH == 1:
    #     main_folder = "results/def DQN p_expl_att random/"
    # elif model_CH == 2:
    #     main_folder = "results/def DQN p_expl_att DQN/"
    # elif model_CH == 3:
    #     main_folder = "results/def DQN p_expl_att DQN p_expl/"
    # elif model_CH == 4:
    #     main_folder = "results/DQN-CH-level1 def DQN p_expl_att random/"
    # elif model_CH == 5:
    #     main_folder = "results/DQN-CH-level1 def DQN p_expl_att DQN/"
    # elif model_CH == 6:
    #     main_folder = "results/DQN-CH-level1 def DQN p_expl_att DQN p_expl/"
    # elif model_CH == 7:
    #     main_folder = "results/def DQN_att random/"
    # elif model_CH == 8:
    #     main_folder = "results/def DQN_att DQN/"
    # elif model_CH == 9:
    #     main_folder = "results/def DQN_att DQN p_expl/"
