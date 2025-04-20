import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from cube_solver import RubiksCube2x2, MOVES
from collections import namedtuple

# PPO Hyperparameters
GAMMA = 0.99
CLIP_EPS = 0.2
UPDATE_EPOCHS = 5
BATCH_SIZE = 32

Transition = namedtuple('Transition', ['state', 'action', 'log_prob', 'reward', 'next_state', 'done'])

# Action mapping
action_to_idx = {move: i for i, move in enumerate(MOVES)}
idx_to_action = {i: move for move, i in action_to_idx.items()}

class StateFeaturizer:
    def extract(self, state_str):
        state = eval(state_str)
        flat = []
        for pos, orient in state:
            flat.extend([pos, orient])
        return np.array(flat, dtype=np.float32)

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 64)
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return self.actor(x), self.critic(x)

class PPOAgent:
    def __init__(self, featurizer, lr=3e-4):
        self.featurizer = featurizer
        self.model = ActorCritic(16, len(MOVES))
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.buffer = []

    def choose_action(self, state_str):
        x = torch.tensor(self.featurizer.extract(state_str)).unsqueeze(0)
        logits, _ = self.model(x)
        probs = torch.softmax(logits, dim=-1)
        logits = torch.nan_to_num(logits, nan=0.0)

        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample()
        return idx_to_action[action_idx.item()], dist.log_prob(action_idx)

    def store(self, *args):
        self.buffer.append(Transition(*args))

    def update(self):
        transitions = self.buffer
        self.buffer = []

        states = torch.tensor([self.featurizer.extract(t.state) for t in transitions])
        actions = torch.tensor([action_to_idx[t.action] for t in transitions])
        rewards = [t.reward for t in transitions]
        dones = [t.done for t in transitions]
        old_log_probs = torch.stack([t.log_prob for t in transitions])

        # Compute returns
        returns = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + GAMMA * R * (1 - d)
            returns.insert(0, R)
        returns = torch.tensor(returns)

        returns = torch.tensor(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
        else:
            returns = returns * 0


        for _ in range(UPDATE_EPOCHS):
            logits, values = self.model(states)
            # Adding the nan check
            if torch.isnan(logits).any():
                print("[Warning] NaN in logits. Resetting to uniform.")
                return
            
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = (new_log_probs - old_log_probs.detach()).exp()
            advantage = returns - values.squeeze()

            surrogate1 = ratio * advantage
            surrogate2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantage
            policy_loss = -torch.min(surrogate1, surrogate2).mean()
            value_loss = nn.functional.mse_loss(values.view(-1), returns)


            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# Training PPO

def train_ppo(episodes=300):
    env = RubiksCube2x2()
    featurizer = StateFeaturizer()
    agent = PPOAgent(featurizer)

    for ep in range(episodes):
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < 50:
            action, log_prob = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.store(state, action, log_prob, reward, next_state, done)
            state = next_state
            steps += 1

        agent.update()
        print(f"Episode {ep+1}: steps={steps}, solved={done}, reward_sum={sum(t.reward for t in agent.buffer):.2f}")

    return agent

if __name__ == "__main__":
    trained_ppo_agent = train_ppo(episodes=300)
