import json
import os
import numpy as np
import matplotlib.pyplot as plt

from scenarios import ARTIFACTS_DIR, scenarios

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
    elapsed_times = [] 
    for file in files:
        with open(os.path.join(folder, file), 'r') as f:
            data = json.load(f)
            scenario_data.append(data)
            if 'elapsed_time' in data:  # Check if "elapsed_time" exists in the file
                elapsed_times.append(data['elapsed_time'])
    return scenario_data, elapsed_times

# Extract relevant metrics from data
def extract_metrics(data):
    metrics = {
        'reward_history': [],
        'attacker_reward_history': [],
        'data_prot_history': [],
        'eps_history': [],
        'action_history': [],
        'action_att_history': []
    }
    
    for entry in data:
        metrics['reward_history'].append(entry['reward_history'])
        metrics['attacker_reward_history'].append(entry['reward_att_history'])
        metrics['data_prot_history'].append(entry['data_prot_history'])
        metrics['eps_history'].append(entry['eps_history'])
        metrics['action_history'].append(entry['action_history'])
        metrics['action_att_history'].append(entry['action_att_history'])
    
    # Convert to numpy arrays for easier averaging
    for key in metrics:
        metrics[key] = np.array(metrics[key])
    
    return metrics

# Simple Moving Average for Smoothing
def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Load and extract metrics for all scenarios
scenario_metrics = {}
for scenario, path in scenarios.items():
    print(f"Processing scenario: {scenario}")
    data, _ = load_scenario_data(path)
    scenario_metrics[scenario] = extract_metrics(data)

# Average the metrics across all 10 files per scenario
def average_metrics(metrics):
    avg_metrics = {}
    for key, values in metrics.items():
        avg_metrics[key] = np.mean(values, axis=0)
    return avg_metrics

# Averaging metrics for all scenarios
avg_metrics = {}
for scenario, metrics in scenario_metrics.items():
    avg_metrics[scenario] = average_metrics(metrics)

# Plotting function for comparison between scenarios with colors and line styles
def plot_comparison(metric_name, metric_label, avg_metrics, scenario_styles, title):
    plt.figure(figsize=(10, 6))
    for scenario, metrics in avg_metrics.items():
        smoothed_data = moving_average(metrics[metric_name])  # Applying moving average for smoothing
        style = scenario_styles[scenario]
        plt.plot(smoothed_data, color=style['color'], linestyle=style['linestyle'], marker=style['marker'], label=style['label'])
    
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel(metric_label)
    plt.legend(loc='upper right')
    plt.grid(True)
    # Use tight layout to minimize white space
    plt.tight_layout()
    plt.show()

# Function to compute cumulative rewards
def plot_cumulative_rewards(avg_metrics, scenario_styles):
    plt.figure(figsize=(10, 6))

    for scenario, metrics in avg_metrics.items():
        cumulative_reward = np.cumsum(metrics['reward_history'])
        style = scenario_styles[scenario]
        plt.plot(cumulative_reward, color=style['color'], linestyle=style['linestyle'], marker=style['marker'], label=style['label'])
    
    plt.title('Cumulative Defender Reward Across Scenarios')
    plt.xlabel('Time Steps')
    plt.ylabel('Cumulative Reward')
    plt.legend(loc='upper left')
    plt.grid(True)
    # Use tight layout to minimize white space
    plt.tight_layout()
    plt.show()

# Plotting cumulative rewards with styles
plot_cumulative_rewards(avg_metrics, scenario_styles)

# Plot reward history comparison (smoothed) with styles
plot_comparison('reward_history', 'Average Reward (Defender)', avg_metrics, scenario_styles, 
                'Comparison of Defender Reward Across Scenarios')

# Plot attacker reward history comparison (smoothed) with styles
plot_comparison('attacker_reward_history', 'Average Reward (Attacker)', avg_metrics, scenario_styles, 
                'Comparison of Attacker Reward Across Scenarios')

# Plot data protection efficiency comparison (smoothed) with styles
plot_comparison('data_prot_history', 'Average Data Protection Ratio', avg_metrics, scenario_styles, 
                'Comparison of Data Protection Efficiency Across Scenarios')



# Extract epsilon histories from data
def extract_eps_histories(data):
    eps_histories = []
    for entry in data:
        eps_histories.append(entry['eps_history'])
    return np.array(eps_histories)

# Extract relevant metrics from data
def extract_data_protection_histories(data):
    data_prot_histories = []
    for entry in data:
        data_prot_histories.append(entry['data_prot_history'])
    return np.array(data_prot_histories)

# Function to calculate mean and standard deviation of data protection over time ranges
def range_data_protection(data_prot_histories, ranges):
    range_mean = np.zeros(len(ranges))
    range_std = np.zeros(len(ranges))
    for i, (start, end) in enumerate(ranges):
        # Calculate the mean and standard deviation of data protection for each range
        protection_values = np.mean(data_prot_histories[:, start:end+1], axis=1)
        range_mean[i] = np.mean(protection_values)
        range_std[i] = np.std(protection_values)   
    return range_mean, range_std

# Function to calculate average epsilon values over time ranges
def range_eps_history(eps_histories, ranges):
    range_eps_avg = np.zeros(len(ranges))
    for i, (start, end) in enumerate(ranges):
        # Calculate the mean epsilon value for each range
        eps_values = np.mean(eps_histories[:, start:end+1], axis=1)
        range_eps_avg[i] = np.mean(eps_values)
    return range_eps_avg

# Calculate mean and standard deviation for data protection in all scenarios
scenario_protection = {}
scenario_eps = {}
for scenario, path in scenarios.items():
    print(f"Processing scenario: {scenario}")
    data, _ = load_scenario_data(path)
    data_prot_histories = extract_data_protection_histories(data)
    scenario_protection[scenario] = range_data_protection(data_prot_histories, ranges)
    eps_histories = extract_eps_histories(data)
    scenario_eps[scenario] = range_eps_history(eps_histories, ranges)
    epsilon_means = scenario_eps[scenario]










import matplotlib.pyplot as plt
import numpy as np

# Plotting average data protection as a bar plot with error bars for each scenario
def plot_avg_data_protection_bar_with_epsilon(scenario_protection, ranges, epsilon_means):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_labels = ['Rounds 1-10', 'Rounds 11-20', 'Rounds 21-30', 'Rounds 31-40']
    x = np.arange(len(x_labels))  # Number of time ranges
    cluster_width = 0.7
    bar_width = cluster_width / len(scenario_protection)  # Adjust the width of the bars based on the number of scenarios

    bar_handles = []
    # Plot the bar chart for average data protection
    for i, (scenario, (protection_mean, protection_std)) in enumerate(scenario_protection.items()):
        style = scenario_styles[scenario]
        x_positions = x + (bar_width * i) - cluster_width / 2 + bar_width / 2  # Shift bars for each scenario

        # Plot bars
        bar = ax.bar(x_positions, protection_mean, width=bar_width, color=style['color'], edgecolor='black',
                      label=style['label'], alpha=0.8)
        bar_handles.append(bar)
        
        # Add error bars
        ax.errorbar(x_positions, protection_mean, yerr=protection_std, fmt='o', color='black', capsize=5)

    # Plot horizontal lines representing epsilon values for each stage
    epsilon_handles = []
    for i, epsilon in enumerate(epsilon_means):
        epsilon_line = ax.hlines(y=epsilon, xmin=i - cluster_width / 2, xmax=i + cluster_width / 2, color='red',
                                   linestyle='--', linewidth=2, label=r'$\epsilon$ Average' if i == 0 else "")
        if i == 0:
            epsilon_handles.append(epsilon_line)


    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=12)
    ax.set_xlabel('Exploration-Exploitation Stages', fontsize=14)
    ax.set_ylabel('', fontsize=14)
    ax.set_title('Average Weighted Data Protection Ratio with $\epsilon$ Average per Stage', fontsize=16)
    
    # Adding more horizontal lines for better visual separation between bars
    ax.set_yticks(np.arange(0.0, 1.05, 0.05))  # Set y-axis ticks for more granular separation
    ax.grid(axis='y', linestyle=':', linewidth=0.7, alpha=0.7)  # More dotted lines for better differentiation

    # Adding separate legends for bars and epsilon lines
    bar_legend = ax.legend(handles=bar_handles, loc='upper center', bbox_to_anchor=(0.4, -0.15), 
                            fancybox=True, shadow=True, ncol=2, fontsize=12)
    ax.add_artist(bar_legend)
    epsilon_legend = ax.legend(handles=epsilon_handles, loc='upper center', bbox_to_anchor=(1, -0.15),
                               fancybox=True, shadow=True, ncol=2, fontsize=10)
    # plt.gca().add_artist(bar_legend)

    # Adding more space at the bottom for the legend
    plt.subplots_adjust(bottom=0.25)  # Increase the bottom margin to fit the legend
    
    # Adding a tight layout and reducing whitespace
    # plt.tight_layout()
    plt.show()

# Plot the average data protection as a bar plot with error bars and epsilon lines
plot_avg_data_protection_bar_with_epsilon(scenario_protection, ranges, epsilon_means)





# Load and calculate elapsed times for each scenario
scenario_elapsed_times = {}
for scenario, path in scenarios.items():
    _, elapsed_times = load_scenario_data(path)
    scenario_elapsed_times[scenario] = np.mean(elapsed_times) if elapsed_times else 0  # Average elapsed time if data exists

# Save mean elapsed times to a text file
with (ARTIFACTS_DIR / "elapsed_times_summary.txt").open("w") as file:
    file.write("Scenario,Average Elapsed Time (seconds)\n")
    for scenario, elapsed_time in scenario_elapsed_times.items():
        file.write(f"{scenario},{elapsed_time:.4f}\n")


# Load data from all files in the scenario folder
def load_scenario_score(folder):
    files = [f for f in os.listdir(folder) if f.endswith('.json')]
    scenario_data = []
    avg_data_prot_w_ratios = [] 
    for file in files:
        with open(os.path.join(folder, file), 'r') as f:
            data = json.load(f)
            scenario_data.append(data)
            avg_data_prot_w_ratios.append(data["avg_data_prot_w_ratio"])
    return avg_data_prot_w_ratios

scenario_avg_data_prot_w_ratios = {}
for scenario, path in scenarios.items():
    avg_data_prot_w_ratios = load_scenario_score(path)
    scenario_avg_data_prot_w_ratios[scenario] = np.mean(avg_data_prot_w_ratios) 

# Save mean 
with (ARTIFACTS_DIR / "avg_data_prot_w_ratios_summary.txt").open("w") as file:
    file.write("Scenario,Average Weighted Data Protection Ratio\n")
    for scenario, avg_data_prot_w_ratio in scenario_avg_data_prot_w_ratios.items():
        file.write(f"{scenario},{avg_data_prot_w_ratio:.4f}\n")








# Plotting average data protection as a bar plot with error bars for each scenario
def plot_avg_data_protection_bar(scenario_protection, ranges):
    plt.figure(figsize=(10, 6))
    
    x_labels = ['Rounds 1-10', 'Rounds 11-20', 'Rounds 21-30', 'Rounds 31-40']
    x = np.arange(len(x_labels))  # Number of time ranges
    cluster_width = 0.7
    bar_width = cluster_width / len(scenario_protection)  # Adjust the width of the bars based on the number of scenarios

    for i, (scenario, (protection_mean, protection_std)) in enumerate(scenario_protection.items()):
        style = scenario_styles[scenario]
        x_positions = x + (bar_width * i) - cluster_width / 2  # Shift bars for each scenario
        
        # Plot bars
        plt.bar(x_positions, protection_mean, width=bar_width, color=style['color'], edgecolor='black', 
                label=style['label'], alpha=0.8)
        
        # Add error bars
        plt.errorbar(x_positions, protection_mean, yerr=protection_std, fmt='o', color='black', capsize=5)
    
    plt.xticks(x, x_labels, fontsize=12)
    plt.xlabel('Exploration-Exploitation Stages', fontsize=14)
    plt.ylabel('', fontsize=14)
    plt.title('Average Weighted Data Protection Ratio per Stage with Standard Deviation', fontsize=16)
    
    # Adding more horizontal lines for better visual separation between bars
    plt.yticks(np.arange(0.0, 1.05, 0.05))  # Set y-axis ticks for more granular separation
    plt.grid(axis='y', linestyle=':', linewidth=0.7, alpha=0.7)  # More dotted lines for better differentiation

    # Adjust legend placement outside the plot area for better visibility
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=2, fontsize=14)

    # Adding a tight layout and reducing whitespace
    plt.tight_layout()
    plt.show()

# Plot the average data protection as a bar plot with error bars
plot_avg_data_protection_bar(scenario_protection, ranges)