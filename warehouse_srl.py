"""
SARSA Reinforcement Learning for Warehouse Delivery
--------------------------------------------------
This module implements SARSA algorithm for a simplified
warehouse delivery task with one truck accepting all items.
"""

import numpy as np
import pygame
import sys
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from typing import Dict, List, Set, Tuple, Optional

# Import the simplified environment
import warehouse_wrld as env


class SarsaAgent:
    """Agent implementing SARSA algorithm."""
    
    def __init__(self, learning_rate=0.2, discount_factor=0.95, exploration_rate=1.0, 
                 exploration_decay=0.99, min_exploration=0.05):
        """
        Initialize the SARSA agent.
        
        Args:
            learning_rate: Alpha - how quickly the agent learns from new experiences
            discount_factor: Gamma - how much future rewards are valued
            exploration_rate: Epsilon - probability of taking a random action
            exploration_decay: Rate at which exploration decreases over time
            min_exploration: Minimum exploration rate
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration
        
        # Initialize Q-table as a default dictionary
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        
        # Define action mapping (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
        self.actions = [
            env.Direction.UP,
            env.Direction.DOWN,
            env.Direction.LEFT, 
            env.Direction.RIGHT
        ]
        
        # Store training metrics
        self.steps_per_episode = []
        self.rewards_per_episode = []
        self.items_delivered_per_episode = []
        
    def state_to_key(self, robot_pos, inventory, delivered_items):
        """
        Convert a state to a string key for the Q-table.
        
        The state includes:
        - Robot position
        - Items in inventory
        - Items delivered
        """
        # Get position
        pos = (robot_pos.x, robot_pos.y)
        
        # Get inventory state as a tuple of booleans
        inventory_tuple = tuple(inventory.values())
        
        # Get delivered items as a frozenset (immutable)
        delivered = frozenset(delivered_items)
        
        # Combine into a hashable state representation
        return (pos, inventory_tuple, delivered)
    
    def choose_action(self, robot_pos, inventory, delivered_items):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            robot_pos: Current position of the robot
            inventory: Dictionary of items in inventory
            delivered_items: Set of delivered items
            
        Returns:
            Selected action index
        """
        # Convert state to key
        state_key = self.state_to_key(robot_pos, inventory, delivered_items)
        
        # Exploration: choose a random action
        if random.random() < self.exploration_rate:
            return random.choice(range(len(self.actions)))
        
        # Exploitation: choose the action with the highest Q-value
        return np.argmax(self.q_table[state_key])
    
    def update_q_table(self, state, action, reward, next_state, next_action):
        """
        Update Q-table using the SARSA update rule.
        
        Q(s,a) = Q(s,a) + α * [r + γ * Q(s', a') - Q(s,a)]
        
        This is an on-policy algorithm that uses the next action chosen by the current policy.
        """
        # Get current Q-value
        current_q = self.q_table[state][action]
        
        # Get Q-value for next state-action pair
        next_q = self.q_table[next_state][next_action]
        
        # Calculate new Q-value using SARSA formula
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_q - current_q
        )
        
        # Update Q-table
        self.q_table[state][action] = new_q
    
    def decay_exploration(self):
        """Decay the exploration rate over time."""
        self.exploration_rate = max(self.min_exploration, 
                                   self.exploration_rate * self.exploration_decay)


def train_agent(n_episodes=200, max_steps=200, render_every=None):
    """
    Train a SARSA agent in the simplified warehouse environment.
    
    Args:
        n_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        render_every: If not None, render environment every n episodes
    """
    # Create SARSA agent with optimized parameters
    agent = SarsaAgent()
    
    print("Training SARSA agent...")
    print("=" * 60)
    print("Progress will be shown every 10 episodes.")
    print("=" * 60)
    
    # Track completion success
    completions = []
    completion_episodes = []
    completion_steps = []
    items_delivered_per_episode = []
    
    # Set up live plotting
    plt.ion()  # Turn on interactive mode
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot titles and labels
    ax1.set_title('Steps per Episode')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Steps')
    ax1.grid(True)
    
    ax2.set_title('Rewards per Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Reward')
    ax2.grid(True)
    
    ax3.set_title('Items Delivered per Episode')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Items Delivered')
    ax3.grid(True)
    
    # Initialize empty plots
    steps_line, = ax1.plot([], [], 'b-', label='Steps')
    rewards_line, = ax2.plot([], [], 'g-', label='Reward')
    delivered_line, = ax3.plot([], [], 'r-', label='Items Delivered')
    
    # Add max_steps reference line
    ax1.axhline(y=max_steps, color='k', linestyle='--', alpha=0.7, 
               label=f'Max Steps ({max_steps})')
    
    # Setup legends
    ax1.legend()
    ax2.legend()
    ax3.legend()
    
    plt.tight_layout()
    plt.pause(0.1)  # Needed for initial plot display
    
    for episode in range(1, n_episodes + 1):
        # Create a fresh simulation for each episode
        simulation = env.GridWorldSimulation()
        
        # Get initial state components
        robot_pos = simulation.robot_pos
        inventory = dict(simulation.inventory.inventory)  # Make a copy
        delivered_items = set(simulation.inventory.delivered_items)  # Make a copy
        
        total_reward = 0
        steps = 0
        
        # Track items picked up for reward shaping
        items_picked_up = set()
        
        # Store the last positions to detect getting stuck
        last_positions = []
        
        # Total items to be delivered
        total_items = len(simulation.inventory.inventory)
        
        # *** SARSA DIFFERENCE: Choose the first action before the loop starts ***
        # Choose initial action
        if len(last_positions) > 10 and all(pos == last_positions[0] for pos in last_positions):
            action_idx = random.randint(0, 3)  # Force random action if stuck
        else:
            action_idx = agent.choose_action(robot_pos, inventory, delivered_items)
        
        # Run the episode
        for step in range(max_steps):
            # Render if needed
            if render_every and episode % render_every == 0:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        plt.close('all')
                        sys.exit()
                        
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        plt.close('all')
                        sys.exit()
                
                simulation.notification_system.update()
                simulation.renderer.render_frame(
                    simulation.robot_pos,
                    simulation.inventory,
                    simulation.notification_system
                )
                simulation.renderer.clock.tick(10)
            
            # *** SARSA DIFFERENCE: Use the pre-selected action ***
            action = agent.actions[action_idx]
            
            # Take action
            old_delivered = len(delivered_items)
            old_inventory = dict(inventory)  # Capture inventory before move
            
            # Try to move in the selected direction
            old_pos = robot_pos
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
            
            # Track positions to detect getting stuck
            last_positions.append(robot_pos)
            if len(last_positions) > 10:
                last_positions.pop(0)
            
            # Get the new state components
            new_robot_pos = simulation.robot_pos
            new_inventory = dict(simulation.inventory.inventory)
            new_delivered_items = set(simulation.inventory.delivered_items)
            
            # OPTIMIZED REWARD STRUCTURE
            # ---------------------------
            
            # Small penalty for each step (encouraging efficiency)
            reward = -0.1
            
            # Small penalty for hitting walls or not moving
            if old_pos == new_robot_pos:
                reward -= 0.5  # Bigger penalty for not moving
            
            # Reward for picking up items (but only once per unique item)
            item_at_pos = simulation.layout.get_object_at(simulation.robot_pos)
            if item_at_pos in simulation.inventory.emoji_to_item:
                item_name = simulation.inventory.emoji_to_item[item_at_pos]
                if item_name not in items_picked_up and new_inventory[item_name]:
                    items_picked_up.add(item_name)
                    reward += 5.0  # Bigger reward for finding items
            
            # Track which items were delivered this turn
            newly_delivered = new_delivered_items - delivered_items
            
            # Reward for successful delivery
            if newly_delivered:
                # Large reward for each item delivered
                reward += 20.0 * len(newly_delivered)
            
            # Check if close to completion - more items delivered is better
            completion_percentage = len(new_delivered_items) / total_items
            
            # Progressive rewards based on percentage complete
            if completion_percentage >= 0.25 and len(new_delivered_items) > len(delivered_items):
                reward += 5.0  # Additional reward at 25% completion
            if completion_percentage >= 0.5 and len(new_delivered_items) > len(delivered_items):
                reward += 10.0  # Additional reward at 50% completion
            if completion_percentage >= 0.75 and len(new_delivered_items) > len(delivered_items):
                reward += 15.0  # Additional reward at 75% completion
            
            # Check if done (all items delivered)
            done = len(new_delivered_items) == total_items
            
            # Big reward for completing the task
            if done:
                # Massive reward that accounts for efficiency (fewer steps = better reward)
                completion_reward = 100.0 + max(0, (max_steps - step) * 1.0)
                reward += completion_reward
                
                # Record completion details
                completions.append(True)
                completion_episodes.append(episode)
                completion_steps.append(step)
                
                print(f"TASK COMPLETED! Episode {episode}, Steps: {step}, Items: {len(new_delivered_items)}/{total_items}")
            
            # *** SARSA DIFFERENCE: Choose next action for SARSA update ***
            # Choose next action
            if len(last_positions) > 10 and all(pos == last_positions[0] for pos in last_positions):
                next_action_idx = random.randint(0, 3)  # Force random action if stuck
            else:
                next_action_idx = agent.choose_action(new_robot_pos, new_inventory, new_delivered_items)
            
            # Prepare state tuples for SARSA update
            current_state = agent.state_to_key(robot_pos, inventory, delivered_items)
            next_state = agent.state_to_key(new_robot_pos, new_inventory, new_delivered_items)
            
            # *** SARSA DIFFERENCE: Use SARSA update rule with next action ***
            # Update Q-table using SARSA rule (including next_action_idx)
            agent.update_q_table(current_state, action_idx, reward, next_state, next_action_idx)
            
            # Update current state and action
            robot_pos = new_robot_pos
            inventory = new_inventory
            delivered_items = new_delivered_items
            action_idx = next_action_idx  # Use the next action on next iteration
            
            total_reward += reward
            steps += 1
            
            # End episode if done
            if done:
                break
        
        # If not completed, record that
        if not done:
            completions.append(False)
        
        # Decay exploration rate
        agent.decay_exploration()
        
        # Store metrics
        agent.steps_per_episode.append(steps)
        agent.rewards_per_episode.append(total_reward)
        items_delivered_per_episode.append(len(delivered_items))
        
        # Print progress
        if episode % 10 == 0:
            completion_percent = len(delivered_items) / total_items * 100 if total_items > 0 else 0
            print(f"Episode {episode}/{n_episodes}, "
                  f"Steps: {steps}, Reward: {total_reward:.2f}, "
                  f"Items Delivered: {len(delivered_items)}/{total_items} ({completion_percent:.1f}%), "
                  f"Exploration: {agent.exploration_rate:.4f}")
            
            # Update live plot
            episodes = list(range(1, episode + 1))
            steps_line.set_data(episodes, agent.steps_per_episode)
            rewards_line.set_data(episodes, agent.rewards_per_episode)
            delivered_line.set_data(episodes, items_delivered_per_episode)
            
            # Adjust axes limits
            for ax in [ax1, ax2, ax3]:
                ax.relim()
                ax.autoscale_view()
            
            # Redraw
            fig.canvas.draw_idle()
            plt.pause(0.01)
    
    # Final performance summary
    print("\n" + "="*60)
    print("FINAL PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Total Episodes: {n_episodes}")
    print(f"Total Completions: {sum(completions)}")
    print(f"Completion Rate: {sum(completions)/n_episodes*100:.2f}%")
    
    if completion_episodes:
        print(f"First Completion: Episode {completion_episodes[0]}")
        print(f"Average Completion Steps: {sum(completion_steps)/len(completion_steps):.2f}")
    else:
        print("No successful completions recorded")
    
    # Plot final performance graphs
    plt.ioff()  # Turn off interactive mode for final plots
    plt.figure(figsize=(15, 12))
    
    # Steps per episode
    plt.subplot(3, 1, 1)
    plt.plot(agent.steps_per_episode, 'b-')
    plt.axhline(y=max_steps, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Steps per Episode (SARSA)')
    plt.grid(True)
    
    # Rewards per episode
    plt.subplot(3, 1, 2)
    plt.plot(agent.rewards_per_episode, 'g-')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Rewards per Episode (SARSA)')
    plt.grid(True)
    
    # Items delivered per episode
    plt.subplot(3, 1, 3)
    plt.plot(items_delivered_per_episode, 'r-')
    plt.xlabel('Episode')
    plt.ylabel('Items Delivered')
    plt.title('Items Delivered per Episode (SARSA)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('sarsa_performance.png', dpi=300)
    plt.show()
    
    return agent  # Return the trained agent


def demonstrate_agent(agent, max_steps=200):
    """Run a visual demonstration of a trained agent."""
    # Create fresh environment
    simulation = env.GridWorldSimulation()
    
    # Get initial state components
    robot_pos = simulation.robot_pos
    inventory = dict(simulation.inventory.inventory)
    delivered_items = set(simulation.inventory.delivered_items)
    
    steps = 0
    done = False
    
    # Run the episode
    for step in range(max_steps):
        # Render environment
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
                
        simulation.notification_system.update()
        simulation.renderer.render_frame(
            simulation.robot_pos,
            simulation.inventory,
            simulation.notification_system
        )
        simulation.renderer.clock.tick(5)  # Slow down for visibility
        
        # Choose action using trained policy (no exploration)
        old_exploration = agent.exploration_rate
        agent.exploration_rate = 0  # No exploration during demonstration
        action_idx = agent.choose_action(robot_pos, inventory, delivered_items)
        agent.exploration_rate = old_exploration
        
        action = agent.actions[action_idx]
        
        # Take action
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
        
        # Get the new state
        robot_pos = simulation.robot_pos
        inventory = dict(simulation.inventory.inventory)
        delivered_items = set(simulation.inventory.delivered_items)
        
        # Check if done (all items delivered)
        total_items = len(simulation.inventory.inventory)
        done = len(delivered_items) == total_items
        
        steps += 1
        
        # End episode if done
        if done:
            print(f"Demonstration complete in {steps} steps!")
            # Keep the final state visible for a moment
            time.sleep(2)
            break
    
    # If not done, report that
    if not done:
        print(f"Demonstration ended after maximum {max_steps} steps without completing all deliveries.")
    
    # Clean up
    pygame.quit()


def compare_with_qlearning():
    """Compare SARSA results with Q-learning results if available."""
    try:
        # Load Q-learning performance results if they exist
        q_results = np.load('q_learning_performance.npz', allow_pickle=True)
        q_steps = q_results['steps']
        q_rewards = q_results['rewards']
        q_items_delivered = q_results['items_delivered']
        
        # Load SARSA performance results if they exist
        sarsa_results = np.load('sarsa_performance.npz', allow_pickle=True)
        sarsa_steps = sarsa_results['steps']
        sarsa_rewards = sarsa_results['rewards']
        sarsa_items_delivered = sarsa_results['items_delivered']
        
        # Create comparison plots
        plt.figure(figsize=(15, 12))
        
        # Steps per episode comparison
        plt.subplot(3, 1, 1)
        plt.plot(q_steps, 'b-', label='Q-Learning')
        plt.plot(sarsa_steps, 'r-', label='SARSA')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.title('Steps per Episode Comparison')
        plt.legend()
        plt.grid(True)
        
        # Rewards per episode comparison
        plt.subplot(3, 1, 2)
        plt.plot(q_rewards, 'b-', label='Q-Learning')
        plt.plot(sarsa_rewards, 'r-', label='SARSA')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Rewards per Episode Comparison')
        plt.legend()
        plt.grid(True)
        
        # Items delivered per episode comparison
        plt.subplot(3, 1, 3)
        plt.plot(q_items_delivered, 'b-', label='Q-Learning')
        plt.plot(sarsa_items_delivered, 'r-', label='SARSA')
        plt.xlabel('Episode')
        plt.ylabel('Items Delivered')
        plt.title('Items Delivered per Episode Comparison')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('algorithm_comparison.png', dpi=300)
        plt.show()
        
        print("Key Performance Metrics Comparison:")
        print("=" * 60)
        print(f"Final 50 Episodes Average Steps - Q-Learning: {np.mean(q_steps[-50:]):.2f}, SARSA: {np.mean(sarsa_steps[-50:]):.2f}")
        print(f"Final 50 Episodes Average Rewards - Q-Learning: {np.mean(q_rewards[-50:]):.2f}, SARSA: {np.mean(sarsa_rewards[-50:]):.2f}")
        print(f"Final 50 Episodes Average Items Delivered - Q-Learning: {np.mean(q_items_delivered[-50:]):.2f}, SARSA: {np.mean(sarsa_items_delivered[-50:]):.2f}")
        
    except Exception as e:
        print(f"Could not load comparison data: {e}")
        print("Run both algorithms first to enable comparison.")


if __name__ == "__main__":
    # Train the SARSA agent
    print("Starting training for simplified warehouse delivery task using SARSA...")
    trained_agent = train_agent(n_episodes=200, max_steps=200, render_every=None)
    
    # Save SARSA performance data for later comparison
    np.savez('sarsa_performance.npz', 
             steps=np.array(trained_agent.steps_per_episode),
             rewards=np.array(trained_agent.rewards_per_episode),
             items_delivered=np.array(trained_agent.rewards_per_episode))
    
    # Demonstrate the trained agent
    print("\nDemonstrating trained SARSA agent behavior...")
    demonstrate_agent(trained_agent, max_steps=200)
    
    # Compare with Q-learning if available
    print("\nComparing SARSA with Q-learning (if data available)...")
    compare_with_qlearning()