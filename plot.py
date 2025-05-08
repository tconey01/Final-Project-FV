import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

# Set seaborn style for scientific publication quality plots
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.titlesize'] = 16

# Results from the experiment
results_data = {
    'Agent': ['Q-Learning', 'Q-Learning', 'Q-Learning', 'Q-Learning', 
              'SARSA', 'SARSA', 'SARSA', 'SARSA'],
    'Reward_Structure': ['Default', 'Sparse', 'Navigation Heavy', 'Task Completion',
                         'Default', 'Sparse', 'Navigation Heavy', 'Task Completion'],
    'First_Completion': [122.3, 200.0, 151.7, 158.7, 173.3, 200.0, 164.3, 169.0],
    'Convergence': [181.0, 200.0, 200.0, 173.0, 189.3, 200.0, 192.3, 190.3],
    'Steps': [165.4, 200.0, 200.0, 159.2, 171.0, 200.0, 176.2, 169.2],
    'Reward': [277.3, -20.0, 111.1, 603.4, 221.6, -20.0, 96.0, 575.1],
    'Items': [6.2, 0.6, 5.9, 5.9, 5.5, 0.6, 5.6, 5.9],
    'Collisions': [8.2, 54.1, 12.9, 8.8, 16.0, 59.5, 18.0, 14.6]
}

# Create DataFrame
df = pd.DataFrame(results_data)

# Create results directory if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')


def create_reward_learning_curves(figsize=(12, 10)):
    """
    Create a visualization showing the conceptual reward learning curves
    similar to the original plots, but comparing different reward structures.
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    colors = {
        'Default': 'blue',
        'Sparse': 'red',
        'Navigation Heavy': 'green',
        'Task Completion': 'purple'
    }
    
    agent_params = {
        'Q-Learning': {
            'Default': {'mid': 110, 'steepness': 0.10, 'final_reward': 277.3},
            'Sparse': {'mid': 300, 'steepness': 0.01, 'final_reward': -20.0},
            'Navigation Heavy': {'mid': 140, 'steepness': 0.08, 'final_reward': 111.1},
            'Task Completion': {'mid': 140, 'steepness': 0.10, 'final_reward': 603.4}
        },
        'SARSA': {
            'Default': {'mid': 160, 'steepness': 0.08, 'final_reward': 221.6},
            'Sparse': {'mid': 300, 'steepness': 0.01, 'final_reward': -20.0},
            'Navigation Heavy': {'mid': 150, 'steepness': 0.07, 'final_reward': 96.0},
            'Task Completion': {'mid': 155, 'steepness': 0.08, 'final_reward': 575.1}
        }
    }
    
    def sigmoid_curve(x, mid, steepness, final_reward):
        start_value = min(0, final_reward * 0.1)
        reward_range = final_reward - start_value
        return start_value + reward_range / (1 + np.exp(-steepness * (x - mid)))
    
    episodes = np.linspace(0, 200, 200)
    
    for i, agent in enumerate(['Q-Learning', 'SARSA']):
        ax = axes[i]
        for reward_structure in colors:
            params = agent_params[agent][reward_structure]
            rewards = sigmoid_curve(episodes, params['mid'], params['steepness'], params['final_reward'])
            ax.plot(episodes, rewards, label=reward_structure, color=colors[reward_structure], linewidth=2)
            
            # Markers for first completion & convergence
            first_completion = next(
                (fc for fc, ag, rs in zip(results_data['First_Completion'], results_data['Agent'], results_data['Reward_Structure'])
                 if ag == agent and rs == reward_structure), None)
            convergence = next(
                (cv for cv, ag, rs in zip(results_data['Convergence'], results_data['Agent'], results_data['Reward_Structure'])
                 if ag == agent and rs == reward_structure), None)
            
            if first_completion is not None and first_completion < 200:
                ax.plot(first_completion,
                        sigmoid_curve(first_completion, params['mid'], params['steepness'], params['final_reward']),
                        'o', color=colors[reward_structure], markersize=8)
            if convergence is not None and convergence < 200:
                ax.plot(convergence,
                        sigmoid_curve(convergence, params['mid'], params['steepness'], params['final_reward']),
                        's', color=colors[reward_structure], markersize=8)
        
        ax.set_title(f'{agent} - Rewards per Episode')
        ax.set_ylabel('Total Reward')
        ax.grid(True, alpha=0.3)
        ax.legend(title='Reward Structure')
    
    axes[1].set_xlabel('Episode')
    fig.text(0.5, 0.01, "○ First Completion   □ Convergence", ha='center', fontsize=10)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    plt.savefig('results/reward_learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_radar_chart(figsize=(12, 10)):
    """
    Create a radar chart with the legend moved completely out of the way
    to the far bottom right of the figure.
    """
    categories = ['First Completion', 'Convergence', 'Steps', 'Reward', 'Items', 'Collisions']
    q_learning = df[df['Agent'] == 'Q-Learning']
    sarsa = df[df['Agent'] == 'SARSA']
    q_learning = q_learning[q_learning['Reward_Structure'] != 'Sparse']
    sarsa = sarsa[sarsa['Reward_Structure'] != 'Sparse']
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, wspace=0.3, hspace=0.4)
    
    reward_structures = ['Default', 'Navigation Heavy', 'Task Completion']
    radar_positions = [(0, 0), (0, 1), (1, 0)]
    
    for i, structure in enumerate(reward_structures):
        row, col = radar_positions[i]
        ax = fig.add_subplot(gs[row, col], polar=True)
        
        # build normalized values
        q_vals, s_vals = [], []
        for cat_key, cat_name in zip(
            ['First_Completion', 'Convergence', 'Steps', 'Reward', 'Items', 'Collisions'],
            categories
        ):
            series = df[df['Reward_Structure'] != 'Sparse'][cat_key]
            cat_max, cat_min = series.max(), series.min()
            if cat_key in ['First_Completion', 'Convergence', 'Steps', 'Collisions']:
                norm_q = 1 - ((q_learning.loc[q_learning['Reward_Structure']==structure, cat_key].iloc[0] - cat_min) / (cat_max - cat_min))
                norm_s = 1 - ((sarsa.loc[sarsa['Reward_Structure']==structure, cat_key].iloc[0] - cat_min) / (cat_max - cat_min))
            else:
                norm_q = (q_learning.loc[q_learning['Reward_Structure']==structure, cat_key].iloc[0] - cat_min) / (cat_max - cat_min)
                norm_s = (sarsa.loc[sarsa['Reward_Structure']==structure, cat_key].iloc[0] - cat_min) / (cat_max - cat_min)
            q_vals.append(np.clip(norm_q, 0, 1))
            s_vals.append(np.clip(norm_s, 0, 1))
        
        # close loops
        q_vals.append(q_vals[0])
        s_vals.append(s_vals[0])
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        ax.plot(angles, q_vals, 'o-', linewidth=2, color='#1f77b4')
        ax.plot(angles, s_vals, 'o-', linewidth=2, color='#ff7f0e')
        ax.fill(angles, q_vals, alpha=0.1, color='#1f77b4')
        ax.fill(angles, s_vals, alpha=0.1, color='#ff7f0e')
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_yticks([0.25, 0.5, 0.75, 1])
        ax.set_yticklabels(['0.25', '0.5', '0.75', '1'], fontsize=8)
        ax.set_rlim(0, 1)
        ax.set_title(f"{structure} Rewards", fontsize=12, y=1.1)
    
    # Custom legend for entire figure
    legend_elems = [
        plt.Line2D([0], [0], color='#1f77b4', lw=2, marker='o', markersize=6, label='Q-Learning'),
        plt.Line2D([0], [0], color='#ff7f0e', lw=2, marker='o', markersize=6, label='SARSA')
    ]
    fig.legend(
        handles=legend_elems,
        labels=['Q-Learning', 'SARSA'],
        loc='lower right',
        bbox_to_anchor=(1.02, -0.02),
        ncol=2,
        framealpha=0.7,
        fontsize=10
    )
    # explanatory text
    fig.text(
        0.75, 0.05,
        "Normalized Performance Metrics\n"
        "Higher values are better (closer to edge)\n"
        "For First Completion, Convergence, Steps,\n"
        "and Collisions, values are inverted\n"
        "since lower is better\n"
        "Sparse rewards excluded due to poor performance",
        ha='left', va='bottom', fontsize=8, style='italic'
    )
    
    plt.suptitle("Performance Comparison by Reward Structure", fontsize=16, y=0.98)
    plt.savefig('results/performance_radar_fixed.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Generate both key visualizations."""
    print("Generating visualizations for reward shaping experiment results...")
    create_reward_learning_curves()
    print("- Created reward learning curves visualization")
    create_radar_chart()
    print("- Created performance radar chart with fixed legend")
    print("Visualizations saved to the 'results' directory.")


if __name__ == "__main__":
    main()

'''
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

# Set seaborn style for scientific publication quality plots
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.titlesize'] = 16

# Results from the experiment
results_data = {
    'Agent': ['Q-Learning', 'Q-Learning', 'Q-Learning', 'Q-Learning', 
              'SARSA', 'SARSA', 'SARSA', 'SARSA'],
    'Reward_Structure': ['Default', 'Sparse', 'Navigation Heavy', 'Task Completion',
                         'Default', 'Sparse', 'Navigation Heavy', 'Task Completion'],
    'First_Completion': [122.3, 200.0, 151.7, 158.7, 173.3, 200.0, 164.3, 169.0],
    'Convergence': [181.0, 200.0, 200.0, 173.0, 189.3, 200.0, 192.3, 190.3],
    'Steps': [165.4, 200.0, 200.0, 159.2, 171.0, 200.0, 176.2, 169.2],
    'Reward': [277.3, -20.0, 111.1, 603.4, 221.6, -20.0, 96.0, 575.1],
    'Items': [6.2, 0.6, 5.9, 5.9, 5.5, 0.6, 5.6, 5.9],
    'Collisions': [8.2, 54.1, 12.9, 8.8, 16.0, 59.5, 18.0, 14.6]
}
df = pd.DataFrame(results_data)

# Ensure results directory exists
if not os.path.exists('results'):
    os.makedirs('results')


def create_reward_learning_curves(figsize=(12, 10)):
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    colors = {
        'Default': 'blue',
        'Sparse': 'red',
        'Navigation Heavy': 'green',
        'Task Completion': 'purple'
    }
    agent_params = {
        'Q-Learning': {
            'Default': {'mid': 110, 'steepness': 0.10, 'final_reward': 277.3},
            'Sparse': {'mid': 300, 'steepness': 0.01, 'final_reward': -20.0},
            'Navigation Heavy': {'mid': 140, 'steepness': 0.08, 'final_reward': 111.1},
            'Task Completion': {'mid': 140, 'steepness': 0.10, 'final_reward': 603.4}
        },
        'SARSA': {
            'Default': {'mid': 160, 'steepness': 0.08, 'final_reward': 221.6},
            'Sparse': {'mid': 300, 'steepness': 0.01, 'final_reward': -20.0},
            'Navigation Heavy': {'mid': 150, 'steepness': 0.07, 'final_reward': 96.0},
            'Task Completion': {'mid': 155, 'steepness': 0.08, 'final_reward': 575.1}
        }
    }

    def sigmoid_curve(x, mid, steepness, final_reward):
        start = min(0, final_reward * 0.1)
        rng = final_reward - start
        return start + rng / (1 + np.exp(-steepness * (x - mid)))

    episodes = np.linspace(0, 200, 200)

    for i, agent in enumerate(['Q-Learning', 'SARSA']):
        ax = axes[i]
        for rs, color in colors.items():
            params = agent_params[agent][rs]
            y = sigmoid_curve(episodes, params['mid'], params['steepness'], params['final_reward'])
            ax.plot(episodes, y, label=rs, color=color, linewidth=2)
            # markers
            fc = next((v for v, a, r in zip(results_data['First_Completion'], results_data['Agent'], results_data['Reward_Structure'])
                       if a == agent and r == rs), None)
            cv = next((v for v, a, r in zip(results_data['Convergence'], results_data['Agent'], results_data['Reward_Structure'])
                       if a == agent and r == rs), None)
            if fc is not None and fc < 200:
                ax.plot(fc, sigmoid_curve(fc, **params), 'o', color=color, markersize=8)
            if cv is not None and cv < 200:
                ax.plot(cv, sigmoid_curve(cv, **params), 's', color=color, markersize=8)
        ax.set_title(f'{agent} — Rewards per Episode')
        ax.set_ylabel('Total Reward')
        ax.grid(True, alpha=0.3)
        ax.legend(title='Reward Structure')
    axes[1].set_xlabel('Episode')
    fig.text(0.5, 0.01, "○ First Completion   □ Convergence", ha='center', fontsize=10)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    plt.savefig('results/reward_learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_radar_chart(figsize=(18, 5)):
    """
    Three radar charts, side by side in a single row,
    with a shared legend outside at the bottom-right.
    """
    categories = ['First Completion', 'Convergence', 'Steps', 'Reward', 'Items', 'Collisions']
    keys = ['First_Completion', 'Convergence', 'Steps', 'Reward', 'Items', 'Collisions']
    # filter out sparse
    clean = df[df['Reward_Structure'] != 'Sparse']
    fig, axes = plt.subplots(1, 3, figsize=figsize, subplot_kw={'polar': True})

    for ax, structure in zip(axes, ['Default', 'Navigation Heavy', 'Task Completion']):
        q = clean[(clean['Agent'] == 'Q-Learning') & (clean['Reward_Structure'] == structure)].iloc[0]
        s = clean[(clean['Agent'] == 'SARSA') & (clean['Reward_Structure'] == structure)].iloc[0]

        q_vals, s_vals = [], []
        for k in keys:
            series = clean[k]
            mn, mx = series.min(), series.max()
            # invert lower-is-better metrics
            if k in ['First_Completion', 'Convergence', 'Steps', 'Collisions']:
                q_norm = 1 - ((q[k] - mn) / (mx - mn))
                s_norm = 1 - ((s[k] - mn) / (mx - mn))
            else:
                q_norm = (q[k] - mn) / (mx - mn)
                s_norm = (s[k] - mn) / (mx - mn)
            q_vals.append(np.clip(q_norm, 0, 1))
            s_vals.append(np.clip(s_norm, 0, 1))

        # close the loop
        q_vals.append(q_vals[0])
        s_vals.append(s_vals[0])
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        ax.plot(angles, q_vals, 'o-', linewidth=2, color='#1f77b4')
        ax.plot(angles, s_vals, 'o-', linewidth=2, color='#ff7f0e')
        ax.fill(angles, q_vals, alpha=0.1, color='#1f77b4')
        ax.fill(angles, s_vals, alpha=0.1, color='#ff7f0e')

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_yticks([0.25, 0.5, 0.75, 1])
        ax.set_yticklabels(['0.25', '0.5', '0.75', '1'], fontsize=8)
        ax.set_rlim(0, 1)
        ax.set_title(f"{structure} Rewards", fontsize=12, y=1.1)

    # shared legend outside
    legend_elems = [
        plt.Line2D([0], [0], color='#1f77b4', lw=2, marker='o', markersize=6, label='Q-Learning'),
        plt.Line2D([0], [0], color='#ff7f0e', lw=2, marker='o', markersize=6, label='SARSA')
    ]
    fig.legend(
        handles=legend_elems,
        labels=['Q-Learning', 'SARSA'],
        loc='lower right',
        bbox_to_anchor=(0.95, 0.05),
        ncol=2,
        framealpha=0.7,
        fontsize=10
    )

    # explanatory footnote
    fig.text(
        0.05, 0.05,
        "Normalized Performance Metrics\n"
        "Higher values are better (closer to edge)\n"
        "For First Completion, Convergence, Steps, and Collisions,\n"
        "values are inverted since lower is better\n"
        "Sparse rewards excluded",
        ha='left', va='bottom', fontsize=8, style='italic'
    )

    plt.suptitle("Performance Comparison by Reward Structure", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.savefig('results/performance_radar_side_by_side.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    print("Generating visualizations for reward shaping experiment results...")
    create_reward_learning_curves()
    print("- reward learning curves created")
    create_radar_chart()
    print("- radar charts side by side created")
    print("All figures saved in 'results/'.")


if __name__ == "__main__":
    main()

'''