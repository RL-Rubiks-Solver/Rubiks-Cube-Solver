import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from cube_solver import RubiksCube2x2, MOVES

# Convert actions to index
action_to_idx = {move: i for i, move in enumerate(MOVES)}
idx_to_action = {i: move for move, i in action_to_idx.items()}

# Simple state featurizer for the policy network
class StateFeaturizer:
    def extract(self, state_str):
        state = eval(state_str)
        flat = []
        for pos, orient in state:
            flat.extend([pos, orient])
        return np.array(flat, dtype=np.float32)

# Policy Network
class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

# Policy Gradient Agent (REINFORCE)
class PolicyGradientAgent:
    def __init__(self, featurizer, lr=0.01, gamma=0.99):
        self.featurizer = featurizer
        self.gamma = gamma
        self.policy_net = PolicyNet(input_dim=16, output_dim=len(MOVES))
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def choose_action(self, state_str):
        features = torch.tensor(self.featurizer.extract(state_str)).unsqueeze(0)
        probs = self.policy_net(features).squeeze()

        # Adding the nan check
        if torch.isnan(probs).any():
            print("[Warning] NaN in action probabilities. Resetting to uniform.")
            probs = torch.ones_like(probs) / len(probs)
                
        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample()
        return idx_to_action[action_idx.item()], dist.log_prob(action_idx)

    def update_policy(self, log_probs, rewards):
        discounted = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            discounted.insert(0, R)
        discounted = torch.tensor(discounted)
        if len(discounted) > 1:
            discounted = (discounted - discounted.mean()) / (discounted.std(unbiased=False) + 1e-8)
        else:
            discounted = discounted * 0
        loss = -sum(lp * r for lp, r in zip(log_probs, discounted))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Training Loop

def train_policy_gradient(episodes=300):
    env = RubiksCube2x2()
    featurizer = StateFeaturizer()
    agent = PolicyGradientAgent(featurizer)

    for ep in range(episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        done = False
        steps = 0
        while not done and steps < 50:
            action, log_prob = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
            steps += 1
        agent.update_policy(log_probs, rewards)
        print(f"Episode {ep+1}: steps={steps}, solved={done}, total_reward={sum(rewards):.2f}")

    return agent

if __name__ == "__main__":
    trained_pg_agent = train_policy_gradient(episodes=300)
