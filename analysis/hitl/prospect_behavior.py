import json
import os
import numpy as np
import matplotlib.pyplot as plt

from scenarios import scenarios

# Define colors and line styles for each scenario
scenario_styles = {
    'cht_human_vs_dqn_attacker': {'color': 'orchid', 'linestyle': '-', 'marker': '', 'label': 'D: Reward + Transition-Aware Human Subject | A: DQN'},
    'dqn_human_vs_dqn_attacker': {'color': 'darkturquoise', 'linestyle': '-', 'marker': '', 'label': 'D: Reward-Aware Human Subject | A: DQN'},
    'cht_sim_vs_dqn_attacker': {'color': 'purple', 'linestyle': ':', 'marker': '+', 'label': 'D: CHT-DQN | A: DQN'},
    'dqn_sim_vs_dqn_attacker': {'color': 'green', 'linestyle': ':', 'marker': '+', 'label': 'D: DQN | A: DQN'}
    
}


# Define time ranges
ranges = [(0, 9), (10, 19), (20, 29), (30, 39)]
node_max = 6

# Load data from all files in the scenario folder
def load_scenario_data(folder):
    files = [f for f in os.listdir(folder) if f.endswith('.json')]
    scenario_data = []
    for file in files:
        with open(os.path.join(folder, file), 'r') as f:
            data = json.load(f)
            scenario_data.append(data)
    return scenario_data

# Extract action histories from data
def extract_action_histories(data):
    def_actions = []
    att_actions = []
    
    for entry in data:
        def_actions.append(entry['action_history'])
        att_actions.append(entry['action_att_history'])
    
    return np.array(def_actions), np.array(att_actions)




# Define the number of rounds
rounds = 40

# Function to check if defender failed in a round
def action_failed(def_action, att_action):
    return def_action != att_action

# Figure 1: Frequency of reselection of a failed action in subsequent rounds
def reselection_after_failure(def_actions, att_actions, rounds):
    # Initialize a list to track reselection frequencies
    reselection_frequencies = np.zeros((rounds - 1,))
    
    # Iterate over all rounds except the last one (as there is no "next" round for the last one)
    for t in range(rounds - 1):
        failed_actions = def_actions[:, t] != att_actions[:, t]  # Track where failure occurred
        reselected = (def_actions[:, t + 1] == def_actions[:, t])  # Check if the action is reselected
        reselection_frequencies[t] = np.mean(failed_actions & reselected)  # Avg reselection after failure

    return reselection_frequencies

# Figure 2: Compare action selection after success vs. failure
def compare_success_failure(def_actions, att_actions, rounds):
    success_frequencies = np.zeros((rounds - 1,))
    failure_frequencies = np.zeros((rounds - 1,))
    
    for t in range(rounds - 1):
        successful_actions = def_actions[:, t] == att_actions[:, t]  # Success condition
        failed_actions = def_actions[:, t] != att_actions[:, t]      # Failure condition
        
        success_reselected = (def_actions[:, t + 1] == def_actions[:, t])  # Reselection after success
        failure_reselected = (def_actions[:, t + 1] == def_actions[:, t])  # Reselection after failure
        
        # Calculate average frequency of reselection
        if np.sum(successful_actions) > 0:
            success_frequencies[t] = np.mean(success_reselected[successful_actions])
        if np.sum(failed_actions) > 0:
            failure_frequencies[t] = np.mean(failure_reselected[failed_actions])
    
    return success_frequencies, failure_frequencies


# Figure 3: Calculate reselection within a window after failure
def reselection_after_failure_window(def_actions, att_actions, rounds, window=5):
    reselection_frequencies = np.zeros((rounds - 1,))

    # Iterate over all rounds except the last one
    for t in range(rounds - 1):
        failed_actions = (def_actions[:, t] != att_actions[:, t]).astype(bool)  # Track where failure occurred

        # Check if the action is reselected in any round within the window
        reselected = np.zeros(failed_actions.shape, dtype=bool)  # Make sure `reselected` is a boolean array
        for w in range(1, min(window, rounds - t)):  # Limit window if it exceeds available rounds
            '''if the same action is reselected in any of the rounds in the window, 
            the corresponding entry in reselected will be set to True'''
            reselected = reselected | (def_actions[:, t + w] == def_actions[:, t])

        # Avg reselection after failure within the window
        reselection_frequencies[t] = np.mean(failed_actions & reselected)

    return reselection_frequencies




# Load and analyze the data for both human and simulation scenarios
scenario_resel_after_failure = {}
scenario_success_vs_failure = {}
scenario_resel_after_failure_window = {}

for scenario, path in scenarios.items():
    print(f"Processing scenario: {scenario}")
    data = load_scenario_data(path)
    def_actions, att_actions = extract_action_histories(data)
    
    # Figure 1: Calculate the frequency of reselection after failure
    scenario_resel_after_failure[scenario] = reselection_after_failure(def_actions, att_actions, rounds)
    
    # Figure 2: Calculate success vs failure reselection frequencies
    scenario_success_vs_failure[scenario] = compare_success_failure(def_actions, att_actions, rounds)

    # Figure 3: Calculate reselection within a window after failure
    scenario_resel_after_failure_window[scenario] = reselection_after_failure_window(def_actions, att_actions, rounds, window=5)


# Plot Figure 1: Frequency of reselection after failure
def plot_reselection_after_failure(scenario_resel_after_failure):
    plt.figure(figsize=(10, 6))
    for scenario, frequencies in scenario_resel_after_failure.items():
        style = scenario_styles[scenario]
        plt.plot(np.arange(1, rounds), frequencies, color=style['color'], linestyle=style['linestyle'], marker=style['marker'], label=style['label'])
    
    # plt.title('Frequency of Reselection After Failure')
    plt.xlabel('Rounds', fontsize=14)
    plt.ylabel('Node Reselection Likelihood After Failure', fontsize=14)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2, fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot Figure 1
plot_reselection_after_failure(scenario_resel_after_failure)


# Plot Figure 2: Comparison of reselection frequencies after success vs failure
def plot_success_vs_failure(scenario_success_vs_failure):
    plt.figure(figsize=(10, 6))
    for scenario, (success_freqs, failure_freqs) in scenario_success_vs_failure.items():
        style = scenario_styles[scenario]
        plt.plot(np.arange(1, rounds), success_freqs, color=style['color'], linestyle=style['linestyle'], marker=style['marker'], label=style['label'])
        # plt.plot(np.arange(1, rounds), failure_freqs, color=style['color'], linestyle='--', marker='x', label=f"{style['label']} (Failure)")
    
    # plt.title('Reselection Frequency After Success')
    plt.xlabel('Rounds', fontsize=14)
    plt.ylabel('Node Reselection Likelihood After Success', fontsize=14)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2, fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot Figure 2
plot_success_vs_failure(scenario_success_vs_failure)


# Plot Figure 3: Frequency of reselection within a window after failure
def plot_reselection_after_failure_window(scenario_resel_after_failure_window):
    plt.figure(figsize=(10, 6))
    for scenario, frequencies in scenario_resel_after_failure_window.items():
        style = scenario_styles[scenario]
        plt.plot(np.arange(1, rounds), frequencies, color=style['color'], linestyle=style['linestyle'], marker=style['marker'], label=style['label'])
    
    # plt.title('Frequency of Reselection After Failure')
    plt.xlabel('Rounds', fontsize=14)
    plt.ylabel('Node Reselection Likelihood After Failure within Window=5', fontsize=12)
    # plt.legend(loc='upper left', prop={'size': 8})
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2, fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot Figure 3
plot_reselection_after_failure_window(scenario_resel_after_failure_window)
