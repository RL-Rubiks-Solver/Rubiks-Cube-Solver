import matplotlib.pyplot as plt
import numpy as np
from cube_solver import RubiksCube2x2, MOVES
from cube_solver import QLearningAgent, FeatureExtractor as QFeat
from policy_gradient_rubik import PolicyGradientAgent, StateFeaturizer as PGFeat
from ppo_rubik_solver import PPOAgent, StateFeaturizer as PPOFeat

# Common Evaluation Function
def evaluate_agent(agent, method="q", episodes=100):
    env = RubiksCube2x2()
    steps_per_episode = []
    solved_flags = []

    # Disable exploration during eval
    if hasattr(agent, 'epsilon'):
        original_epsilon = agent.epsilon
        agent.epsilon = 0

    for ep in range(episodes):
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < 50:
            if method == "q":
                action = agent.choose_action(state)
            else:
                action, *_ = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            state = next_state
            steps += 1
        steps_per_episode.append(steps)
        solved_flags.append(int(done))
        print(f"[{method.upper()}] Eval Ep {ep+1}: Steps={steps}, Solved={done}")

    if hasattr(agent, 'epsilon'):
        agent.epsilon = original_epsilon

    return steps_per_episode, solved_flags

# Plotting Comparison
def plot_comparison(metrics):
    plt.figure(figsize=(14, 6))
    for label, solved_flags in metrics.items():
        moving_avg = np.convolve(solved_flags, np.ones(10)/10, mode='valid')
        plt.plot(range(10, len(moving_avg)+10), moving_avg, label=label)

    plt.title("Moving Average Success Rate (10-episode window)")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Instantiate Agents
    print("Evaluating Q-Learning...")
    q_agent = QLearningAgent(MOVES, QFeat())
    q_steps, q_solved = evaluate_agent(q_agent, method="q", episodes=100)

    print("Evaluating Policy Gradient...")
    pg_agent = PolicyGradientAgent(PGFeat())
    pg_steps, pg_solved = evaluate_agent(pg_agent, method="pg", episodes=100)

    print("Evaluating PPO...")
    ppo_agent = PPOAgent(PPOFeat())
    ppo_steps, ppo_solved = evaluate_agent(ppo_agent, method="ppo", episodes=100)

    # Plot Comparison
    plot_comparison({
        "Q-Learning": q_solved,
        "Policy Gradient": pg_solved,
        "PPO": ppo_solved
    })
