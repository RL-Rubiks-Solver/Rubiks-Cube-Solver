import numpy as np
import matplotlib.pyplot as plt
from cube_solver import RubiksCube2x2, MOVES
from cube_solver import QLearningAgent, FeatureExtractor as QFeat
from policy_gradient_rubik import PolicyGradientAgent, StateFeaturizer as PGFeat
from ppo_rubik_solver import PPOAgent, StateFeaturizer as PPOFeat

# Utility: Evaluate and collect detailed metrics
def evaluate_agent(agent, method="q", episodes=100):
    env = RubiksCube2x2()
    steps_per_episode = []
    solved_flags = []
    reward_per_episode = []

    if hasattr(agent, 'epsilon'):
        original_epsilon = agent.epsilon
        agent.epsilon = 0

    for ep in range(episodes):
        state = env.reset()
        done = False
        steps = 0
        total_reward = 0

        while not done and steps < 50:
            if method == "q":
                action = agent.choose_action(state)
            else:
                action, *_ = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            state = next_state
            steps += 1
            total_reward += reward

        steps_per_episode.append(steps)
        solved_flags.append(int(done))
        reward_per_episode.append(total_reward)

    if hasattr(agent, 'epsilon'):
        agent.epsilon = original_epsilon

    return steps_per_episode, solved_flags, reward_per_episode

# Utility: Print final summary
def print_summary(agent_name, steps, solved, rewards):
    print(f"\n=== {agent_name} Summary ===")
    print(f"Total Episodes: {len(steps)}")
    print(f"Solved: {sum(solved)}")
    print(f"Success Rate: {sum(solved) / len(solved) * 100:.2f}%")
    print(f"Average Steps: {np.mean(steps):.2f}")
    print(f"Average Reward: {np.mean(rewards):.2f}")

# Utility: Plot reward & step metrics
def plot_metrics_comparison(all_metrics):
    plt.figure(figsize=(14, 6))

    for label, (steps, solved, rewards) in all_metrics.items():
        # Plot reward trends
        plt.subplot(1, 2, 1)
        plt.plot(rewards, label=label)
        plt.title("Total Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True)

        # Plot step trends
        plt.subplot(1, 2, 2)
        plt.plot(steps, label=label)
        plt.title("Steps Taken per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.grid(True)

    plt.subplot(1, 2, 1)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    metrics = {}

    print("\nEvaluating Q-Learning Agent...")
    q_agent = QLearningAgent(MOVES, QFeat())
    q_steps, q_solved, q_rewards = evaluate_agent(q_agent, method="q")
    print_summary("Q-Learning", q_steps, q_solved, q_rewards)
    metrics["Q-Learning"] = (q_steps, q_solved, q_rewards)

    print("\nEvaluating Policy Gradient Agent...")
    pg_agent = PolicyGradientAgent(PGFeat())
    pg_steps, pg_solved, pg_rewards = evaluate_agent(pg_agent, method="pg")
    print_summary("Policy Gradient", pg_steps, pg_solved, pg_rewards)
    metrics["Policy Gradient"] = (pg_steps, pg_solved, pg_rewards)

    print("\nEvaluating PPO Agent...")
    ppo_agent = PPOAgent(PPOFeat())
    ppo_steps, ppo_solved, ppo_rewards = evaluate_agent(ppo_agent, method="ppo")
    print_summary("PPO", ppo_steps, ppo_solved, ppo_rewards)
    metrics["PPO"] = (ppo_steps, ppo_solved, ppo_rewards)

    plot_metrics_comparison(metrics)
