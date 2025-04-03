import matplotlib.pyplot as plt
import numpy as np
from cube_solver import RubiksCube2x2, FeatureExtractor, QLearningAgent, MOVES

def evaluate_agent(agent, episodes=100):
    env = RubiksCube2x2()
    feature_extractor = FeatureExtractor()
    
    original_epsilon = agent.epsilon
    agent.epsilon = 0  # Pure greedy evaluation

    steps_per_episode = []
    solved_flags = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < 50:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            state = next_state
            steps += 1
        steps_per_episode.append(steps)
        solved_flags.append(int(done))
        print(f"Evaluation Episode {ep+1}: steps={steps}, solved={done}")

    agent.epsilon = original_epsilon  # Restore original epsilon

    return steps_per_episode, solved_flags

def plot_metrics(steps_per_episode, solved_flags):
    episodes = list(range(1, len(steps_per_episode)+1))
    moving_avg = np.convolve(solved_flags, np.ones(10)/10, mode='valid')

    plt.figure(figsize=(12, 6))

    # 1. Solved Binary
    plt.subplot(1, 2, 1)
    plt.plot(episodes, solved_flags, label="Solved (0/1)", marker='o', alpha=0.6)
    plt.title("Solved per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Solved")
    plt.grid(True)
    plt.legend()

    # 2. Moving Average Success Rate
    plt.subplot(1, 2, 2)
    plt.plot(range(10, len(moving_avg)+10), moving_avg, label="Moving Average (10)", color='green')
    plt.title("Success Rate (Moving Avg)")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def summary_stats(steps, solved):
    total = len(steps)
    success = sum(solved)
    avg_steps = sum(steps) / total
    print("\n=== Evaluation Summary ===")
    print(f"Total Episodes: {total}")
    print(f"Solved Episodes: {success}")
    print(f"Success Rate: {success/total*100:.2f}%")
    print(f"Average Steps: {avg_steps:.2f}")

if __name__ == "__main__":
    # Re-initialize same agent structure (must match trained version)
    feat_extractor = FeatureExtractor()
    agent = QLearningAgent(MOVES, feat_extractor)

    # ⚠️ Insert your own loading logic here if you saved weights!
    # Otherwise, evaluate right after training
    print("Evaluating untrained/fresh agent...")

    steps, solved = evaluate_agent(agent, episodes=100)
    summary_stats(steps, solved)
    plot_metrics(steps, solved)
