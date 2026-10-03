import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path


## the same attacker policy for non-Littman and Littman (not softmax for littman cases)
### New utility functions with np.arange(1, node_max+1) including all nodes
## Plot the defender/attacker action frequencies in ***4 scenarios***.
## p_expl=1-data_prot is learned but updated every 100 timesteps.
## Varying node_max from 2 to 10. hyper1_c=d_ca=da ### 02.22.2024

width = 13
height = 7
COMPONENT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = COMPONENT_ROOT / "results"
FIGURES_DIR = REPOSITORY_ROOT / "artifacts" / "figures" / "matched_simulations"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def read_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    data = [float(line.split(': ')[1]) for line in lines if line.startswith("Time step")]
    return data

def plot_average_reward(scenarios, base_folder, node_max):
    plt.figure(figsize=(width,height))

    for scenario, props in scenarios.items():
        linestyles = props['style']
        color = props['color']
        label = props['label']
        data = []
        for seed in range(1, 41):
            file_path = os.path.join(base_folder, scenario, "new", f"{node_max} node", f"average_reward_{seed}.txt")
            if os.path.exists(file_path):
                data.append(read_data(file_path))
        if data:
            data = np.array(data)
            avg_data = np.mean(data, axis=0)
            plt.plot(avg_data, linestyles, color=color, label=label)

    plt.xlabel('Time Step', fontsize=15)
    plt.ylabel('Average Reward', fontsize=15)
    plt.legend(loc='best', fontsize=15)
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    plt.title('Average Reward Over Time for Different Scenarios', fontsize=15)
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(FIGURES_DIR / "average_defender_reward.png", bbox_inches='tight')

def plot_convergence_analysis(scenarios, base_folder, node_max):
    plt.figure(figsize=(width,height))

    for scenario, props in scenarios.items():
        linestyles = props['style']
        color = props['color']
        label = props['label']
        data = []
        for seed in range(1, 41):
            file_path = os.path.join(base_folder, scenario, "new", f"{node_max} node", f"convergence_analysis_{seed}.txt")
            if os.path.exists(file_path):
                data.append(read_data(file_path))
        if data:
            data = np.array(data)
            avg_data = np.mean(data, axis=0)
            plt.plot(avg_data, linestyles, color=color, label=label)

    plt.xlabel('Time Step', fontsize=15)
    plt.ylabel('Running Average Reward of Defender', fontsize=15)
    plt.legend(loc='best', fontsize=15)
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    plt.title('Convergence Analysis of Defender Over Time', fontsize=15)
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(FIGURES_DIR / "running_average_defender_reward.png", bbox_inches='tight')

def plot_average_reward_attacker(scenarios, base_folder, node_max):
    plt.figure(figsize=(width,height))

    for scenario, props in scenarios.items():
        linestyles = props['style']
        color = props['color']
        label = props['label']
        data = []
        for seed in range(1, 41):
            file_path = os.path.join(base_folder, scenario, "new", f"{node_max} node", f"average_reward_attacker_{seed}.txt")
            if os.path.exists(file_path):
                data.append(read_data(file_path))
        if data:
            data = np.array(data)
            avg_data = np.mean(data, axis=0)
            plt.plot(avg_data, linestyles, color=color, label=label)

    plt.xlabel('Time Step', fontsize=15)
    plt.ylabel('Average Reward (Attacker)', fontsize=15)
    plt.legend(loc='best', fontsize=15)
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    plt.title('Average Reward Over Time for Different Scenarios (Attacker)', fontsize=15)
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(FIGURES_DIR / "average_attacker_reward.png", bbox_inches='tight')


def plot_convergence_analysis_attacker(scenarios, base_folder, node_max):
    plt.figure(figsize=(width,height))

    for scenario, props in scenarios.items():
        linestyles = props['style']
        color = props['color']
        label = props['label']
        data = []
        for seed in range(1, 41):
            file_path = os.path.join(base_folder, scenario, "new", f"{node_max} node", f"convergence_analysis_attacker_{seed}.txt")
            if os.path.exists(file_path):
                data.append(read_data(file_path))
        if data:
            data = np.array(data)
            avg_data = np.mean(data, axis=0)
            plt.plot(avg_data, linestyles, color=color, label=label)

    plt.xlabel('Time Step', fontsize=15)
    plt.ylabel('Running Average Reward of Attacker', fontsize=15)
    plt.legend(loc='best', fontsize=15)
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    plt.title('Convergence Analysis of Attacker Over Time', fontsize=15)
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(FIGURES_DIR / "running_average_attacker_reward.png", bbox_inches='tight')

def main():
    scenarios = {
        "DQN-CH-level1 def DQN p_expl_att random": {'style': '-', 'color': 'magenta', 'label': "<CASE 1> D: CHT-DQN G | A: Random"},
        "DQN-CH-level1 def DQN p_expl_att DQN": {'style': '-', 'color': 'indigo', 'label': "<CASE 2> D: CHT-DQN G | A: DQN"},
        "def DQN_att random": {'style': ':', 'color': 'lime', 'label': "<CASE 3> D: DQN | A: Random"},
        "def DQN_att DQN": {'style': ':', 'color': 'darkgreen', 'label': "<CASE 4> D: DQN | A: DQN"}
    }
    node_max = 6

    plot_average_reward(scenarios, RESULTS_DIR, node_max)
    plot_convergence_analysis(scenarios, RESULTS_DIR, node_max)
    plot_average_reward_attacker(scenarios, RESULTS_DIR, node_max)
    plot_convergence_analysis_attacker(scenarios, RESULTS_DIR, node_max)


if __name__ == "__main__":
    main()

