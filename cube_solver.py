import numpy as np
import random
from collections import defaultdict

# All possible 2x2x2 moves
MOVES = ["U", "U'", "R", "R'", "F", "F'"]

# Corner position indices: each corner is [pos, orientation]
SOLVED_CUBE = [(i, 0) for i in range(8)]

class RubiksCube2x2:
    def __init__(self):
        self.state = list(SOLVED_CUBE)

    def reset(self, scramble_moves=10):
        self.state = list(SOLVED_CUBE)
        for _ in range(scramble_moves):
            self.apply_move(random.choice(MOVES))
        return self._state_str()

    def _state_str(self):
        return str(self.state)

    def apply_move(self, move):
        # Very simplified: this only simulates index shuffling
        # Ideally, use a full 2x2x2 corner-twisting model
        mapping = {
            "U": [3, 0, 1, 2, 4, 5, 6, 7],
            "U'": [1, 2, 3, 0, 4, 5, 6, 7],
            "R": [0, 2, 6, 3, 4, 1, 5, 7],
            "R'": [0, 5, 1, 3, 4, 6, 2, 7],
            "F": [1, 5, 2, 0, 4, 6, 3, 7],
            "F'": [3, 0, 2, 6, 4, 1, 5, 7],
        }
        self.state = [self.state[i] for i in mapping[move]]

    def step(self, move):
        self.apply_move(move)
        reward = 1.0 if self.is_solved() else -0.1
        done = self.is_solved()
        return self._state_str(), reward, done

    def is_solved(self):
        return self.state == SOLVED_CUBE

class FeatureExtractor:
    def extract(self, state_str):
        # Count how many corners are in correct pos (very naive feature)
        state = eval(state_str)
        correct = sum(1 for i, (pos, _) in enumerate(state) if pos == i)
        return np.array([correct])

class QLearningAgent:
    def __init__(self, actions, feature_extractor, alpha=0.1, gamma=0.9, epsilon=0.3):
        self.actions = actions
        self.feature_extractor = feature_extractor
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.weights = defaultdict(float)

    def _features(self, state, action):
        features = self.feature_extractor.extract(state)
        return tuple((f, action) for f in features)

    def get_q_value(self, state, action):
        return sum(self.weights[f] for f in self._features(state, action))

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q_vals = [self.get_q_value(state, a) for a in self.actions]
        max_q = max(q_vals)
        return self.actions[q_vals.index(max_q)]

    def update(self, state, action, reward, next_state):
        max_q_next = max(self.get_q_value(next_state, a) for a in self.actions)
        target = reward + self.gamma * max_q_next
        pred = self.get_q_value(state, action)
        diff = target - pred
        for f in self._features(state, action):
            self.weights[f] += self.alpha * diff

def train_agent(episodes=200):
    env = RubiksCube2x2()
    feat_extractor = FeatureExtractor()
    agent = QLearningAgent(MOVES, feat_extractor)

    for ep in range(episodes):
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < 50:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            steps += 1
        print(f"Episode {ep+1}: steps={steps}, solved={done}")

    return agent

if __name__ == "__main__":
    trained_agent = train_agent(episodes=100)
