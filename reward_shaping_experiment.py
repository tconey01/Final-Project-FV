"""
Reward Shaping Experiment for Warehouse Delivery
------------------------------------------------
This module implements experiments to compare different reward structures
for Q-Learning and SARSA algorithms in a warehouse delivery task.
"""

import numpy as np
import sys
import time
import matplotlib
# Use non-interactive Agg backend to avoid Tkinter issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from typing import Dict, List, Set, Tuple, Optional, Any, Callable
import os

# Import the warehouse environment and base RL algorithms
import warehouse_wrld as env
from warehouse_qrl import QLearningAgent
from warehouse_srl import SarsaAgent

# Disable pygame display since we don't need visualization
os.environ['SDL_VIDEODRIVER'] = 'dummy'


class RewardShapingExperiment:
    """Class to run experiments comparing different reward structures."""
    
    def __init__(self):
        """Initialize the experiment with different reward structures."""
        # Define reward structures to test
        self.reward_structures = {
            "default": self._default_reward_function,
            "sparse": self._sparse_reward_function,
            "navigation_heavy": self._navigation_heavy_reward_function,
            "task_completion": self._task_completion_reward_function
        }
        
        # Tracking metrics
        self.results = {}
    
    def _default_reward_function(self, old_pos, new_pos, old_inventory, new_inventory, 
                                old_delivered, new_delivered, total_items, done, step, max_steps):
        """
        The default reward function from the original implementation.
        Used as a baseline for comparison.
        """
        # Small penalty for each step (encouraging efficiency)
        reward = -0.1
        
        # Small penalty for hitting walls or not moving
        if old_pos == new_pos:
            reward -= 0.5  # Bigger penalty for not moving
        
        # Track which items were newly picked up this turn
        new_pickups = [item for item, (old_val, new_val) in 
                       zip(old_inventory.keys(), zip(old_inventory.values(), new_inventory.values()))
                       if not old_val and new_val]
        
        # Reward for picking up items
        if new_pickups:
            reward += 5.0 * len(new_pickups)
        
        # Track which items were delivered this turn
        newly_delivered = new_delivered - old_delivered
        
        # Reward for successful delivery
        if newly_delivered:
            # Large reward for each item delivered
            reward += 20.0 * len(newly_delivered)
        
        # Check if close to completion - more items delivered is better
        completion_percentage = len(new_delivered) / total_items
        
        # Progressive rewards based on percentage complete
        if completion_percentage >= 0.25 and len(new_delivered) > len(old_delivered):
            reward += 5.0  # Additional reward at 25% completion
        if completion_percentage >= 0.5 and len(new_delivered) > len(old_delivered):
            reward += 10.0  # Additional reward at 50% completion
        if completion_percentage >= 0.75 and len(new_delivered) > len(old_delivered):
            reward += 15.0  # Additional reward at 75% completion
        
        # Big reward for completing the task
        if done:
            # Massive reward that accounts for efficiency (fewer steps = better reward)
            completion_reward = 100.0 + max(0, (max_steps - step) * 1.0)
            reward += completion_reward
            
        return reward
    
    def _sparse_reward_function(self, old_pos, new_pos, old_inventory, new_inventory, 
                               old_delivered, new_delivered, total_items, done, step, max_steps):
        """
        Sparse reward function - only terminal rewards for task completion.
        """
        # Small penalty for each step
        reward = -0.1
        
        # Big reward only for completing the task
        if done:
            reward += 300.0
            
        return reward
        
    def _navigation_heavy_reward_function(self, old_pos, new_pos, old_inventory, new_inventory, 
                                         old_delivered, new_delivered, total_items, done, step, max_steps):
        """
        Navigation-focused reward function - emphasizes efficient movement.
        """
        # Larger penalty for each step
        reward = -0.5
        
        # Higher penalty for hitting walls or not moving
        if old_pos == new_pos:
            reward -= 2.0
        
        # Track which items were newly picked up this turn
        new_pickups = [item for item, (old_val, new_val) in 
                       zip(old_inventory.keys(), zip(old_inventory.values(), new_inventory.values()))
                       if not old_val and new_val]
        
        # Moderate reward for picking up items
        if new_pickups:
            reward += 10.0 * len(new_pickups)
        
        # Track which items were delivered this turn
        newly_delivered = new_delivered - old_delivered
        
        # Moderate reward for successful delivery
        if newly_delivered:
            reward += 15.0 * len(newly_delivered)
        
        # Moderate reward for completing the task
        if done:
            reward += 100.0
            
        return reward
    
    def _task_completion_reward_function(self, old_pos, new_pos, old_inventory, new_inventory, 
                                        old_delivered, new_delivered, total_items, done, step, max_steps):
        """
        Task-completion focused reward function - emphasizes picking up and delivering items.
        """
        # Very small penalty for each step
        reward = -0.05
        
        # Small penalty for hitting walls or not moving
        if old_pos == new_pos:
            reward -= 0.2
        
        # Track which items were newly picked up this turn
        new_pickups = [item for item, (old_val, new_val) in 
                       zip(old_inventory.keys(), zip(old_inventory.values(), new_inventory.values()))
                       if not old_val and new_val]
        
        # Large reward for picking up items
        if new_pickups:
            reward += 20.0 * len(new_pickups)
        
        # Track which items were delivered this turn
        newly_delivered = new_delivered - old_delivered
        
        # Very large reward for successful delivery
        if newly_delivered:
            reward += 50.0 * len(newly_delivered)
        
        # Big reward for completing the task but less focus on efficiency
        if done:
            reward += 150.0
            
        return reward
    
    def train_agent(self, agent_type, reward_structure, n_episodes=200, max_steps=200, 
                   n_seeds=5, render_every=None):
        """
        Train an agent with the specified reward structure and track performance.
        
        Args:
            agent_type: 'qlearning' or 'sarsa'
            reward_structure: Name of the reward structure to use
            n_episodes: Number of episodes to train
            max_steps: Maximum steps per episode
            n_seeds: Number of random seeds to use
            render_every: If not None, render environment every n episodes
        """
        # Validate input parameters
        if agent_type not in ['qlearning', 'sarsa']:
            raise ValueError("agent_type must be 'qlearning' or 'sarsa'")
        
        if reward_structure not in self.reward_structures:
            raise ValueError(f"reward_structure must be one of {list(self.reward_structures.keys())}")
        
        reward_function = self.reward_structures[reward_structure]
        
        # Dictionary to store aggregated results
        aggregated_results = {
            'steps_per_episode': [],
            'rewards_per_episode': [],
            'items_delivered_per_episode': [],
            'first_completion_episode': [],
            'convergence_episode': [],  # When 5+ consecutive episodes complete all items
            'wall_collisions_per_episode': []
        }
        
        print(f"Training {agent_type} agent with {reward_structure} reward structure...")
        print("=" * 60)
        
        # Run n_seeds trials and aggregate results
        for seed in range(n_seeds):
            print(f"Running seed {seed+1}/{n_seeds}...")
            
            # Set random seed
            random.seed(seed)
            np.random.seed(seed)
            
            # Create agent based on type
            if agent_type == 'qlearning':
                agent = QLearningAgent()
            else:  # sarsa
                agent = SarsaAgent()
            
            # Reset metrics for this trial
            steps_per_episode = []
            rewards_per_episode = []
            items_delivered_per_episode = []
            wall_collisions_per_episode = []
            
            # Track when task is first completed
            first_completion = None
            # Track when task converges (5+ consecutive completions)
            consecutive_completions = 0
            convergence_episode = None
            
            # Training loop
            for episode in range(1, n_episodes + 1):
                # Create a fresh simulation for each episode
                simulation = env.GridWorldSimulation()
                
                # Get initial state components
                robot_pos = simulation.robot_pos
                inventory = dict(simulation.inventory.inventory)  # Make a copy
                delivered_items = set(simulation.inventory.delivered_items)  # Make a copy
                
                total_reward = 0
                steps = 0
                wall_collisions = 0
                
                # Total items to be delivered
                total_items = len(simulation.inventory.inventory)
                
                # Choose initial action for SARSA
                if agent_type == 'sarsa':
                    action_idx = agent.choose_action(robot_pos, inventory, delivered_items)
                
                # Run the episode
                for step in range(max_steps):
                    # Choose action
                    if agent_type == 'qlearning':
                        action_idx = agent.choose_action(robot_pos, inventory, delivered_items)
                    
                    action = agent.actions[action_idx]
                    
                    # Take action
                    old_pos = robot_pos
                    old_inventory = dict(inventory)  # Capture inventory before move
                    old_delivered = set(delivered_items)  # Capture delivered before move
                    
                    # Try to move in the selected direction
                    if simulation.layout.is_valid_move(simulation.robot_pos, action):
                        # Calculate new position based on the action
                        dx, dy = {
                            env.Direction.UP: (0, 1),
                            env.Direction.DOWN: (0, -1),
                            env.Direction.LEFT: (-1, 0),
                            env.Direction.RIGHT: (1, 0)
                        }[action]
                        
                        simulation.robot_pos = env.Position(
                            simulation.robot_pos.x + dx, 
                            simulation.robot_pos.y + dy
                        )
                        
                        # Check for interactions at new position
                        simulation._check_interaction()
                    else:
                        # Track wall collisions
                        wall_collisions += 1
                        
                    # Get the new state components
                    new_robot_pos = simulation.robot_pos
                    new_inventory = dict(simulation.inventory.inventory)
                    new_delivered_items = set(simulation.inventory.delivered_items)
                    
                    # Calculate reward using specified reward function
                    done = len(new_delivered_items) == total_items
                    reward = reward_function(
                        old_pos, new_robot_pos,
                        old_inventory, new_inventory,
                        old_delivered, new_delivered_items,
                        total_items, done, step, max_steps
                    )
                    
                    # Prepare state tuples for algorithm update
                    current_state = agent.state_to_key(robot_pos, inventory, delivered_items)
                    next_state = agent.state_to_key(new_robot_pos, new_inventory, new_delivered_items)
                    
                    # Update Q-table (different for Q-learning and SARSA)
                    if agent_type == 'qlearning':
                        agent.update_q_table(current_state, action_idx, reward, next_state)
                    else:  # sarsa
                        # Choose next action for SARSA
                        next_action_idx = agent.choose_action(new_robot_pos, new_inventory, new_delivered_items)
                        # SARSA update with the next action
                        agent.update_q_table(current_state, action_idx, reward, next_state, next_action_idx)
                        # Update current action for next iteration
                        action_idx = next_action_idx
                    
                    # Update current state
                    robot_pos = new_robot_pos
                    inventory = new_inventory
                    delivered_items = new_delivered_items
                    
                    total_reward += reward
                    steps += 1
                    
                    # End episode if done
                    if done:
                        # Track first completion
                        if first_completion is None:
                            first_completion = episode
                        
                        # Track convergence
                        consecutive_completions += 1
                        if consecutive_completions >= 5 and convergence_episode is None:
                            convergence_episode = episode - 4  # First of the 5 consecutive episodes
                        
                        break
                
                # If not completed, reset consecutive completions
                if not done:
                    consecutive_completions = 0
                
                # Decay exploration rate
                agent.decay_exploration()
                
                # Store metrics
                steps_per_episode.append(steps)
                rewards_per_episode.append(total_reward)
                items_delivered_per_episode.append(len(delivered_items))
                wall_collisions_per_episode.append(wall_collisions)
                
                # Print progress
                if episode % 10 == 0:
                    completion_percent = len(delivered_items) / total_items * 100 if total_items > 0 else 0
                    print(f"Episode {episode}/{n_episodes}, "
                          f"Steps: {steps}, Reward: {total_reward:.2f}, "
                          f"Items: {len(delivered_items)}/{total_items} ({completion_percent:.1f}%), "
                          f"Collisions: {wall_collisions}, "
                          f"Exploration: {agent.exploration_rate:.3f}")
            
            # Store results for this seed
            aggregated_results['steps_per_episode'].append(steps_per_episode)
            aggregated_results['rewards_per_episode'].append(rewards_per_episode)
            aggregated_results['items_delivered_per_episode'].append(items_delivered_per_episode)
            aggregated_results['wall_collisions_per_episode'].append(wall_collisions_per_episode)
            
            # Store convergence metrics
            if first_completion is not None:
                aggregated_results['first_completion_episode'].append(first_completion)
            else:
                aggregated_results['first_completion_episode'].append(n_episodes)  # Never completed
                
            if convergence_episode is not None:
                aggregated_results['convergence_episode'].append(convergence_episode)
            else:
                aggregated_results['convergence_episode'].append(n_episodes)  # Never converged
        
        # Average results across seeds
        avg_results = {
            'steps_per_episode': np.mean(aggregated_results['steps_per_episode'], axis=0),
            'rewards_per_episode': np.mean(aggregated_results['rewards_per_episode'], axis=0),
            'items_delivered_per_episode': np.mean(aggregated_results['items_delivered_per_episode'], axis=0),
            'wall_collisions_per_episode': np.mean(aggregated_results['wall_collisions_per_episode'], axis=0),
            'first_completion_episode': np.mean(aggregated_results['first_completion_episode']),
            'convergence_episode': np.mean(aggregated_results['convergence_episode']),
        }
        
        # Store results in main results dictionary
        key = f"{agent_type}_{reward_structure}"
        self.results[key] = avg_results
        
        # Print summary
        print("\n" + "="*60)
        print(f"RESULTS SUMMARY FOR {agent_type.upper()} WITH {reward_structure.upper()} REWARDS")
        print("="*60)
        print(f"First completion episode (avg): {avg_results['first_completion_episode']:.1f}")
        print(f"Convergence episode (avg): {avg_results['convergence_episode']:.1f}")
        print(f"Final 50 episodes - Average steps: {np.mean(avg_results['steps_per_episode'][-50:]):.1f}")
        print(f"Final 50 episodes - Average rewards: {np.mean(avg_results['rewards_per_episode'][-50:]):.1f}")
        print(f"Final 50 episodes - Average items delivered: {np.mean(avg_results['items_delivered_per_episode'][-50:]):.1f}")
        print(f"Final 50 episodes - Average wall collisions: {np.mean(avg_results['wall_collisions_per_episode'][-50:]):.1f}")
        
        return avg_results
    
    def run_all_experiments(self, n_episodes=200, max_steps=200, n_seeds=3, render_every=None):
        """Run experiments for all combinations of agent types and reward structures."""
        agent_types = ['qlearning', 'sarsa']
        
        for agent_type in agent_types:
            for reward_structure in self.reward_structures:
                print(f"\nStarting experiment: {agent_type} with {reward_structure} rewards")
                self.train_agent(agent_type, reward_structure, n_episodes, max_steps, n_seeds, render_every)
    
    def plot_comparison(self, metric='rewards_per_episode', save_path=None, show=True, max_steps=200):
        """
        Plot comparison of different reward structures for both agent types.
        
        Args:
            metric: Which metric to plot ('steps_per_episode', 'rewards_per_episode', 
                   'items_delivered_per_episode', 'wall_collisions_per_episode')
            save_path: If provided, save figure to this path
            show: Whether to display the plot
            max_steps: Maximum steps per episode (for reference line)
        """
        if not self.results:
            print("No results to plot. Run experiments first.")
            return
        
        # Create a new figure with two subplots (one for each agent type)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Set titles and labels
        fig.suptitle(f'Comparison of Reward Structures: {metric.replace("_", " ").title()}', fontsize=16)
        
        for i, agent_type in enumerate(['qlearning', 'sarsa']):
            ax = axes[i]
            ax.set_title(f"{agent_type.title()} Agent")
            ax.set_xlabel('Episode')
            
            # Set y-label based on metric
            if metric == 'steps_per_episode':
                ax.set_ylabel('Steps')
            elif metric == 'rewards_per_episode':
                ax.set_ylabel('Total Reward')
            elif metric == 'items_delivered_per_episode':
                ax.set_ylabel('Items Delivered')
            elif metric == 'wall_collisions_per_episode':
                ax.set_ylabel('Wall Collisions')
            
            # Plot data for each reward structure
            for reward_structure in self.reward_structures:
                key = f"{agent_type}_{reward_structure}"
                if key in self.results:
                    # Get data to plot
                    data = self.results[key][metric]
                    episodes = range(1, len(data) + 1)
                    
                    # Plot line
                    ax.plot(episodes, data, label=reward_structure.replace('_', ' ').title())
            
            # Add max_steps reference line if plotting steps
            if metric == 'steps_per_episode':
                ax.axhline(y=max_steps, color='k', linestyle='--', alpha=0.7, 
                          label=f'Max Steps ({max_steps})')
            
            # Add grid and legend
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            directory = os.path.dirname(save_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show plot if requested
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    def plot_all_metrics(self, save_dir='results', show=False, max_steps=200):
        """Generate comparison plots for all metrics and save them."""
        metrics = [
            'steps_per_episode', 
            'rewards_per_episode', 
            'items_delivered_per_episode', 
            'wall_collisions_per_episode'
        ]
        
        for metric in metrics:
            save_path = os.path.join(save_dir, f'{metric}_comparison.png')
            self.plot_comparison(metric=metric, save_path=save_path, show=show, max_steps=max_steps)
    
    def generate_summary_table(self):
        """
        Generate a summary table of key performance metrics for all agent and reward combinations.
        Returns a dictionary with the formatted table data.
        """
        if not self.results:
            print("No results to summarize. Run experiments first.")
            return None
        
        summary = {
            'agent_reward': [],
            'first_completion': [],
            'convergence': [],
            'final_steps': [],
            'final_rewards': [],
            'final_items': [],
            'final_collisions': []
        }
        
        agent_types = ['qlearning', 'sarsa']
        reward_structures = list(self.reward_structures.keys())
        
        for agent_type in agent_types:
            for reward_structure in reward_structures:
                key = f"{agent_type}_{reward_structure}"
                if key in self.results:
                    result = self.results[key]
                    
                    summary['agent_reward'].append(f"{agent_type.title()} - {reward_structure.replace('_', ' ').title()}")
                    summary['first_completion'].append(f"{result['first_completion_episode']:.1f}")
                    summary['convergence'].append(f"{result['convergence_episode']:.1f}")
                    summary['final_steps'].append(f"{np.mean(result['steps_per_episode'][-50:]):.1f}")
                    summary['final_rewards'].append(f"{np.mean(result['rewards_per_episode'][-50:]):.1f}")
                    summary['final_items'].append(f"{np.mean(result['items_delivered_per_episode'][-50:]):.1f}")
                    summary['final_collisions'].append(f"{np.mean(result['wall_collisions_per_episode'][-50:]):.1f}")
        
        return summary
    
    def print_summary_table(self):
        """Print a formatted summary table of results."""
        summary = self.generate_summary_table()
        
        if not summary:
            return
        
        # Print header
        print("\n" + "="*100)
        print("SUMMARY OF RESULTS ACROSS ALL EXPERIMENTS")
        print("="*100)
        
        # Column headers
        print(f"{'Agent - Reward Structure':<30} | {'First':<6} | {'Conv.':<6} | {'Steps':<6} | {'Reward':<8} | {'Items':<6} | {'Collis.':<6}")
        print("-"*100)
        
        # Data rows
        for i in range(len(summary['agent_reward'])):
            print(f"{summary['agent_reward'][i]:<30} | {summary['first_completion'][i]:<6} | "
                  f"{summary['convergence'][i]:<6} | {summary['final_steps'][i]:<6} | "
                  f"{summary['final_rewards'][i]:<8} | {summary['final_items'][i]:<6} | "
                  f"{summary['final_collisions'][i]:<6}")
        
        print("="*100)
        print("First: Episode of first task completion")
        print("Conv.: Episode where agent converged (5+ consecutive completions)")
        print("Steps, Reward, Items, Collis.: Averages over final 50 episodes")
    
    def save_results_to_file(self, filename='reward_shaping_results.txt'):
        """Save the summary table to a text file."""
        summary = self.generate_summary_table()
        
        if not summary:
            return
        
        with open(filename, 'w') as f:
            # Write header
            f.write("=" * 100 + "\n")
            f.write("SUMMARY OF RESULTS ACROSS ALL EXPERIMENTS\n")
            f.write("=" * 100 + "\n\n")
            
            # Column headers
            f.write(f"{'Agent - Reward Structure':<30} | {'First':<6} | {'Conv.':<6} | {'Steps':<6} | {'Reward':<8} | {'Items':<6} | {'Collis.':<6}\n")
            f.write("-" * 100 + "\n")
            
            # Data rows
            for i in range(len(summary['agent_reward'])):
                f.write(f"{summary['agent_reward'][i]:<30} | {summary['first_completion'][i]:<6} | "
                      f"{summary['convergence'][i]:<6} | {summary['final_steps'][i]:<6} | "
                      f"{summary['final_rewards'][i]:<8} | {summary['final_items'][i]:<6} | "
                      f"{summary['final_collisions'][i]:<6}\n")
            
            f.write("=" * 100 + "\n")
            f.write("First: Episode of first task completion\n")
            f.write("Conv.: Episode where agent converged (5+ consecutive completions)\n")
            f.write("Steps, Reward, Items, Collis.: Averages over final 50 episodes\n")
        
        print(f"Results saved to {filename}")


def main():
    """Main function to run the reward shaping experiments."""
    print("Starting reward shaping experiments for warehouse delivery task...")
    
    # Create experiment
    experiment = RewardShapingExperiment()
    
    # Set parameters (reduced for faster testing)
    n_episodes = 200    # Use 200 for final results, can reduce for testing
    max_steps = 200
    n_seeds = 3         # Use 3 or 5 for final results, can reduce for testing
    
    # Run experiments for all combinations
    # Note: This will take a long time to run
    print(f"Running all experiments with {n_episodes} episodes, {n_seeds} seeds each...")
    experiment.run_all_experiments(n_episodes=n_episodes, max_steps=max_steps, n_seeds=n_seeds)
    
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Generate plots
    print("Generating comparison plots...")
    experiment.plot_all_metrics(save_dir='results', max_steps=max_steps, show=False)
    
    # Print and save summary table
    experiment.print_summary_table()
    experiment.save_results_to_file('results/reward_shaping_results.txt')
    
    print("\nExperiments completed! Results are saved in the 'results' directory.")


if __name__ == "__main__":
    main()