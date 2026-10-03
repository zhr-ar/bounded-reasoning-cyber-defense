import json
import os
import numpy as np
import matplotlib.pyplot as plt

from scenarios import scenarios

# Define colors and line styles for each scenario
scenario_styles = {
    'cht_human_vs_dqn_attacker': {'color': 'orchid', 'linestyle': '-', 'marker': '', 'label': 'D: Reward + Transition-Aware Human Subject | A: DQN'},
    'cht_sim_vs_dqn_attacker': {'color': 'purple', 'linestyle': ':', 'marker': '+', 'label': 'D: CHT-DQN | A: DQN'},
    'dqn_human_vs_dqn_attacker': {'color': 'darkturquoise', 'linestyle': '-', 'marker': '', 'label': 'D: Reward-Aware Human Subject | A: DQN'},
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

# Calculate cumulative discrepancy between defender and attacker actions for each round
def cumulative_discrepancy(def_actions, att_actions, rounds):
    cumulative_diffs = np.zeros(rounds)

    for t in range(rounds):
        # Sum the absolute differences between defender and attacker actions for each round
        cumulative_diffs[t] = np.abs(def_actions[:, t] - att_actions[:, t]).mean()

    return np.cumsum(cumulative_diffs)  # Return cumulative discrepancy

# Load and calculate discrepancies for all scenarios
rounds = 40
scenario_discrepancies = {}

for scenario, path in scenarios.items():
    print(f"Processing scenario: {scenario}")
    data = load_scenario_data(path)
    def_actions, att_actions = extract_action_histories(data)
    scenario_discrepancies[scenario] = cumulative_discrepancy(def_actions, att_actions, rounds)

# Plotting cumulative discrepancies for each scenario with different styles
def plot_cumulative_discrepancies(discrepancies, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    for scenario, discrepancy in discrepancies.items():
        style = scenario_styles[scenario]
        plt.plot(np.arange(1, rounds+1), discrepancy, color=style['color'], linestyle=style['linestyle'], marker=style['marker'], label=style['label'])
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='upper left')
    plt.grid(True)
    # Use tight layout to minimize white space
    plt.tight_layout()
    plt.show()

# Plot cumulative discrepancies over 40 rounds
plot_cumulative_discrepancies(scenario_discrepancies, 
                              'Cumulative Discrepancy in Action Frequencies Between Defender and Attacker',
                              'Rounds',
                              'Cumulative Discrepancy')

# Split rounds into 4 ranges: 0-9, 10-19, 20-29, 30-39
def discrepancy_in_ranges(discrepancies, ranges):
    range_diffs = {}

    for scenario, discrepancy in discrepancies.items():
        range_diffs[scenario] = []
        for start, end in ranges:
            range_diffs[scenario].append(np.sum(discrepancy[start:end+1]))

    return range_diffs


# Calculate discrepancies for each range
range_discrepancies = discrepancy_in_ranges(scenario_discrepancies, ranges)

# Plot discrepancies in each range with different styles
def plot_discrepancies_by_ranges(range_discrepancies, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    x = np.arange(1, len(ranges) + 1)  # Number of ranges (4 in this case)
    
    for scenario, diffs in range_discrepancies.items():
        style = scenario_styles[scenario]
        plt.plot(x, diffs, '-s', color=style['color'], linestyle=style['linestyle'], marker=style['marker'], label=style['label'])
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(x, ['1-10', '11-20', '21-30', '31-40'])
    plt.legend(loc='upper left')
    plt.grid(True)
    # Use tight layout to minimize white space
    plt.tight_layout()
    plt.show()

# Plot discrepancies across ranges
plot_discrepancies_by_ranges(range_discrepancies, 
                             'Cumulative Discrepancy in Action Frequencies Across Time Ranges',
                             'Rounds',
                             'Cumulative Discrepancy')






# Function to calculate average node action frequencies over time ranges
def avg_node_frequencies(actions, node_max, ranges):
    freq_per_node_per_range = np.zeros((node_max, len(ranges)))  # Adjust dimensions

    for idx, (start, end) in enumerate(ranges):
        actions_in_range = actions[:, start:end + 1]  # Get actions for the given range (defender or attacker)
        
        # Calculate frequency of each node being acted upon in that range
        for node in range(node_max):
            freq_per_node_per_range[node, idx] = np.mean(actions_in_range == node)

    return freq_per_node_per_range
    

# Create a grid of subplots: 2 rows (Defended/Attacked) x 4 columns (Scenarios)
fig, axs = plt.subplots(2, len(scenarios), figsize=(14, 7), sharey=False)

# Define the labels for the x-axis (time ranges)
x_labels = ['1-10', '11-20', '21-30', '31-40']
x = np.arange(len(x_labels))  # Number of time ranges
cluster_width = 0.7  # Total width for each cluster of bars
y_ticks = [round(i, 2) for i in np.arange(0.0, 0.75, 0.05)]

# Plot average frequencies for each scenario
for idx, (scenario, path) in enumerate(scenarios.items()):
    print(f"Processing scenario: {scenario}")
    data = load_scenario_data(path)
    def_actions, att_actions = extract_action_histories(data)
    
    # Calculate average frequency of defended and attacked nodes
    def_freq = avg_node_frequencies(def_actions, node_max, ranges)
    att_freq = avg_node_frequencies(att_actions, node_max, ranges)
    
    bar_width = cluster_width / def_freq.shape[0]  # Adjust the width of the bars based on the number of nodes
    
    # Plot for Defended Nodes (Row 1)
    for i in range(def_freq.shape[0]):
        x_positions = x + (bar_width * i) - cluster_width / 2
        axs[0, idx].grid(axis='y', linestyle=':', linewidth=0.9, alpha=0.8) 
        axs[0, idx].bar(x_positions, def_freq[i], width=bar_width, label=f"Node {i + 1}", color=plt.cm.viridis(i/def_freq.shape[0]))
        

    axs[0, idx].set_xticks(x)
    axs[0, idx].set_xticklabels(x_labels, fontsize=8)
    axs[0, idx].set_yticks(y_ticks)
    axs[0, idx].set_yticklabels(y_ticks, fontsize=8)
    if idx == 0:
        # axs[0, idx].set_ylabel('')  # Only set the ylabel for the first subplot of the row
        axs[0, idx].set_ylabel('Avg Frequency of Defended Nodes', fontsize=10)

    axs[0, idx].set_title(f"{scenario_styles[scenario]['label']}", fontsize=8, weight='bold')

    
    # Plot for Attacked Nodes (Row 2)
    for i in range(att_freq.shape[0]):
        x_positions = x + (bar_width * i) - cluster_width / 2
        axs[1, idx].grid(axis='y', linestyle=':', linewidth=0.9, alpha=0.8) 
        axs[1, idx].bar(x_positions, att_freq[i], width=bar_width, label=f"Node {i + 1}", color=plt.cm.viridis(i/att_freq.shape[0]))
        
    
    axs[1, idx].set_xticks(x)
    axs[1, idx].set_xticklabels(x_labels, fontsize=8)
    axs[1, idx].set_yticks(y_ticks)
    axs[1, idx].set_yticklabels(y_ticks, fontsize=8)
    if idx == 0:
        # axs[1, idx].set_ylabel('')  # Only set the ylabel for the first subplot of the row
        axs[1, idx].set_ylabel('Avg Frequency of Attacked Nodes', fontsize=10)

# # Remove repeated x-axis labels in the first row
# for ax in axs[0, :]:
#     ax.set_xlabel('')

# # Set the common x-axis label for the bottom row
# for ax in axs[1, :]:
#     ax.set_xlabel('Stages')


# fig.subplots_adjust(hspace=-3)
fig.subplots_adjust(bottom=0.15)

# Add a single legend outside the plot for all subplots
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.005), fancybox=True, shadow=True,ncol=len(handles), title="Nodes", fontsize=8, title_fontsize=10)
fig.supxlabel('Exploration-Exploitation Stages', x=0.5, y=0.09, fontsize=12)

# fig.tight_layout()
plt.show()