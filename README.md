# Rubik's Cube Reinforcement Learning Solver

Welcome to the **Rubik's Cube RL Solver Project**! This project implements and compares multiple model-free reinforcement learning algorithms — including **Q-Learning**, **REINFORCE**, and **Proximal Policy Optimization (PPO)** — to train agents capable of solving a **2x2x2 Rubik's Cube** environment.

---

## 🚀 Project Overview

This project is a hands-on experiment in applying classic and modern reinforcement learning techniques to a simplified cube-solving task. It includes:

- ✅ A custom-built 2x2x2 Rubik's Cube simulator
- ✅ Feature-based state representation
- ✅ Three RL agents: Q-Learning, REINFORCE (Policy Gradient), and PPO
- ✅ Evaluation tools for comparing agent performance
- ✅ Visualization of learning curves and success rates
- ✅ One-click `run_all.py` orchestration script

---

## 📦 Features

- 🧩 **2x2x2 Cube Simulation**: Lightweight, fast environment with scramble and step methods
- 🧠 **Q-Learning Agent**: Tabular, epsilon-greedy agent with feature-based state representation
- 🎯 **Policy Gradient (REINFORCE)**: Neural network policy optimized via Monte Carlo gradients
- 🔄 **PPO Agent**: Clipped policy optimization with actor-critic architecture for stability
- 📈 **Evaluation & Visualization**: Success rate, steps, and reward trends over episodes
- ✅ **Robustness**: Includes numerical stability fixes for reward normalization and logits

---

## 🛠️ Getting Started

### ✅ Prerequisites

- Python 3.8+
- Install dependencies:

```bash
pip install numpy matplotlib torch
```

---

## 📁 Project Structure

```bash
rubiks-cube-project/
├── cube_solver.py              # 2x2x2 environment + Q-learning agent
├── policy_gradient_rubik.py   # REINFORCE agent with PyTorch
├── ppo_rubik_solver.py        # PPO agent with clipped optimization
├── evaluate_model.py          # Evaluation for Q-learning
├── compare_rl_agents.py       # Side-by-side comparison of all agents
├── rl_analysis_report.py      # Summary metrics + plots
├── run_all.py                 # Orchestrated runner for full pipeline
├── README.md                  # This file
```

---

## 🧪 How to Use

### 🔁 Run Everything
```bash
python run_all.py
```

This will:
1. Train all three agents
2. Evaluate each one
3. Plot comparison graphs
4. Show full reward and step analysis

### 🧠 Individual Training
```bash
python cube_solver.py            # Q-Learning
python policy_gradient_rubik.py  # REINFORCE
python ppo_rubik_solver.py       # PPO
```

### 📊 Evaluation & Analysis
```bash
python evaluate_model.py         # Q-learning performance
python compare_rl_agents.py      # Compare Q vs PG vs PPO
python rl_analysis_report.py     # Show reward and steps per agent
```

