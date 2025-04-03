# Rubik's Cube Solver

Welcome to the Rubik's Cube Solver project! This project implements a Q-Learning agent to solve a 3x3x3 Rubik's Cube using reinforcement learning techniques.

## Project Overview

The Rubik's Cube Solver project is designed to explore reinforcement learning by training an agent to solve a Rubik's Cube. The project includes:

- A Rubik's Cube environment that simulates the cube and its possible moves.
- A feature extractor to represent the cube's state.
- A Q-Learning agent that learns to solve the cube through trial and error.
- A training loop to iteratively improve the agent's performance.

## Features

- **Rubik's Cube Environment**: Simulates a 3x3x3 Rubik's Cube with standard moves.
- **Q-Learning Agent**: Uses reinforcement learning to learn optimal moves.
- **Feature Extraction**: Converts cube states into feature vectors for learning.
- **Training and Evaluation**: Includes scripts to train the agent and evaluate its performance.

## Getting Started

### Prerequisites

- Python 3.x
- Required Python packages: `numpy`, `matplotlib`, `rubik-cube`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/RL-Rubiks-Solver/Rubiks-Cube-Solver.git
   cd rubiks-cube-solver
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Train the Agent**:
   Run the training script to train the Q-Learning agent:
   ```bash
   python cube_solver.py
   ```

2. **Evaluate the Agent**:
   Evaluate the trained agent's performance:
   ```bash
   python evaluate_model.py
   ```

### Project Structure

- `cube_solver.py`: Contains the Rubik's Cube environment, feature extractor, and Q-Learning agent.
- `evaluate_model.py`: Script to evaluate the trained agent and visualize results.
- `README.md`: Project documentation.

### Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

### License

This project is licensed under the MIT License.

### Acknowledgments

- Inspired by the classic Rubik's Cube puzzle.
- Utilizes the `rubik-cube` library for cube simulation.
