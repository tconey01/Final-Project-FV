# Warehouse RL Simulation

A grid-based warehouse environment where agents learn to pick up and deliver items using Q-Learning and SARSA. Includes visualization, custom reward shaping experiments, and performance comparison plots.

## Dependencies

Install with:

```bash
pip install numpy matplotlib pandas seaborn pygame
```

## Files Overview

* **`warehouse_wrld.py`**: Core GridWorld environment with warehouse layout, items, and robot simulation.
* **`warehouse_qrl.py`**: Q-Learning agent implementation and training logic.
* **`warehouse_srl.py`**: SARSA agent implementation and training logic.
* **`reward_shaping_experiment.py`**: Runs comparative experiments using different reward structures for both agents.
* **`plot.py`**: Generates learning curve and radar visualizations from experiment data.
* **`key.py`**: Keyboard-controlled demo for interacting with the environment manually or with a trained agent.

## Output

Plots and experiment results are saved in the `results/` directory.