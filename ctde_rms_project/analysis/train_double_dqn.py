
import os
import sys
sys.path.append(r"C:\Users\91944\OneDrive\Desktop\ctde_rms_project")

import os
import numpy as np
import pandas as pd
from env.environment import ImprovedRMS_Env
from agents.dqn_agent import DQNAgent
from agents.double_dqn_agent import DoubleDQNAgent

# Create results folder
os.makedirs("results", exist_ok=True)

# Initialize environment
env = ImprovedRMS_Env()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Initialize Double DQN agent
agent = DoubleDQNAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    lr=1e-3,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.05
)

episodes = 500
rewards, losses = [], []

print("\n============================================================")
print("Training Double DQN on ImprovedRMS_Env")
print("============================================================")

for ep in range(episodes):
    state = env.reset()
    done = False
    total_reward, total_loss = 0.0, 0.0

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        loss = agent.train_step()
        state = next_state
        total_reward += reward
        if loss:
            total_loss += loss
        agent.decay_epsilon()

    rewards.append(total_reward)
    losses.append(total_loss)

    if (ep + 1) % 50 == 0:
        print(f"Episode {ep+1:03d}/{episodes} | Reward: {total_reward:.3f} | Loss: {total_loss:.5f} | ε={agent.epsilon:.3f}")

# Save metrics to CSV
results_path = os.path.join("results", "Double_DQN_rewards.csv")
pd.DataFrame({"Reward": rewards, "Loss": losses}).to_csv(results_path, index=False)
print(f"\n✅ Double DQN Training complete. Results saved to {results_path}")
