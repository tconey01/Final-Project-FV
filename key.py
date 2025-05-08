"""
Clean Keyboard Controlled Warehouse Demo
---------------------------------
This script allows you to control the warehouse robot using keyboard inputs,
with no UI overlays for clean recording.

Controls:
- Arrow keys: Move the robot (up, down, left, right)
- Space: Toggle automatic navigation using a trained agent (Q-Learning)
- R: Reset the environment
- ESC: Quit
"""

import pygame
import sys
import time
import numpy as np

# Import your environment code - adjust these imports based on your file structure
import warehouse_wrld as env
from warehouse_qrl import QLearningAgent

def clean_keyboard_controlled_demo():
    """Run a keyboard-controlled demo with no UI overlays."""
    # Initialize Pygame
    pygame.init()
    
    # Initialize the simulation environment
    simulation = env.GridWorldSimulation()
    
    # Try to load a trained Q-Learning agent as a fallback for automatic navigation
    agent = QLearningAgent()
    try:
        # Attempt to load a saved Q-table - adjust the path/filename as needed
        agent.q_table = np.load('qlearning_default_q_table.npy', allow_pickle=True).item()
        print("Loaded saved Q-table for automated assistance.")
        agent_loaded = True
    except FileNotFoundError:
        print("No saved Q-table found. Automated assistance will use untrained agent.")
        agent_loaded = False
    
    # Flag to track if we're using keyboard control or automated agent
    using_keyboard = True
    
    # Set up the clock for stable framerate
    clock = pygame.time.Clock()
    
    # Game loop
    running = True
    
    # Track items collection
    total_items = len(simulation.inventory.inventory)
    
    print("\nClean Keyboard Control Demo Started:")
    print("Use arrow keys to move the robot")
    print("Press SPACE to toggle automatic navigation")
    print("Press R to reset the environment")
    print("Press ESC to quit")
    print("NO UI overlays for clean recording")
    
    # Keep track of time for automated agent to slow it down
    last_auto_move_time = time.time()
    auto_move_delay = 0.3  # seconds between automated moves
    
    while running:
        # Track current state for checking completion
        items_delivered = len(simulation.inventory.delivered_items)
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Handle key presses
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                
                # Toggle between keyboard and automated control
                elif event.key == pygame.K_SPACE:
                    using_keyboard = not using_keyboard
                    print(f"{'Keyboard' if using_keyboard else 'Automated'} control activated")
                
                # Reset the environment
                elif event.key == pygame.K_r:
                    simulation = env.GridWorldSimulation()
                    total_items = len(simulation.inventory.inventory)
                    print("Environment reset")
                
                # Handle arrow key movements when in keyboard mode
                elif using_keyboard:
                    if event.key == pygame.K_UP:
                        move_direction = env.Direction.UP
                    elif event.key == pygame.K_DOWN:
                        move_direction = env.Direction.DOWN
                    elif event.key == pygame.K_LEFT:
                        move_direction = env.Direction.LEFT
                    elif event.key == pygame.K_RIGHT:
                        move_direction = env.Direction.RIGHT
                    else:
                        # Not an arrow key
                        continue
                    
                    # Try to move in the selected direction
                    if simulation.layout.is_valid_move(simulation.robot_pos, move_direction):
                        # Calculate new position based on the action
                        dx, dy = {
                            env.Direction.UP: (0, 1),
                            env.Direction.DOWN: (0, -1),
                            env.Direction.LEFT: (-1, 0),
                            env.Direction.RIGHT: (1, 0)
                        }[move_direction]
                        
                        simulation.robot_pos = env.Position(
                            simulation.robot_pos.x + dx, 
                            simulation.robot_pos.y + dy
                        )
                        
                        # Check for interactions at new position
                        simulation._check_interaction()
        
        # If using automated control, let the agent choose the action with a time delay
        current_time = time.time()
        if not using_keyboard and current_time - last_auto_move_time >= auto_move_delay:
            # Get current state
            robot_pos = simulation.robot_pos
            inventory = dict(simulation.inventory.inventory)
            delivered_items = set(simulation.inventory.delivered_items)
            
            # Turn off exploration for demonstration
            old_exploration = agent.exploration_rate
            agent.exploration_rate = 0.0
            
            # Choose action
            action_idx = agent.choose_action(robot_pos, inventory, delivered_items)
            agent.exploration_rate = old_exploration
            
            # Get the action
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
            
            # Update last move time
            last_auto_move_time = current_time
        
        # Update notification system
        simulation.notification_system.update()
        
        # Render the main game state - without any overlays
        simulation.renderer.render_frame(
            simulation.robot_pos,
            simulation.inventory,
            simulation.notification_system
        )
        
        # Update the display
        pygame.display.flip()
        
        # Maintain consistent framerate
        clock.tick(30)
        
        # Check if all items have been delivered
        if items_delivered == total_items:
            # In clean mode, we'll just print to console instead of showing an overlay
            print("All items delivered! Press R to reset or ESC to quit.")
            
            # For keyboard mode, still need to wait for input
            if using_keyboard:
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                            waiting = False
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_r:
                                simulation = env.GridWorldSimulation()
                                total_items = len(simulation.inventory.inventory)
                                waiting = False
                            elif event.key == pygame.K_ESCAPE:
                                running = False
                                waiting = False
            else:
                # Brief pause in automated mode
                time.sleep(2)
                
    # Clean up
    pygame.quit()
    print("Demo ended.")


if __name__ == "__main__":
    clean_keyboard_controlled_demo()