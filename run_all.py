import os

print("==============================")
print("🚀 STARTING RL RUBIK'S CUBE PIPELINE")
print("==============================")

# Step 1: Train Q-Learning Agent
print("\n🔵 Training Q-Learning Agent...")
os.system("python cube_solver.py")

# Step 2: Train Policy Gradient Agent (REINFORCE)
print("\n🟢 Training Policy Gradient Agent...")
os.system("python policy_gradient_rubik.py")

# Step 3: Train PPO Agent
print("\n🔴 Training PPO Agent...")
os.system("python ppo_rubik_solver.py")

# Step 4: Evaluate All Agents with Side-by-Side Plot
print("\n📊 Comparing All Agents (Success Rate)...")
os.system("python compare_rl_agents.py")

# Step 5: Run Full Analysis Report with Reward + Steps Plots
print("\n📈 Running Final Analysis Report...")
os.system("python rl_analysis_report.py")

print("\n✅ Pipeline Complete! Review plots and summaries.")
