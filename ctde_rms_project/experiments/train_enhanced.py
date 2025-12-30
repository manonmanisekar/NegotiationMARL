"""
train_enhanced_dqn.py
---------------------
Training script for the Dueling + Attention Enhanced DQN.
Integrates with ImprovedRMS_Env and saves logs for analysis.
"""
import os
import sys
sys.path.append(r"C:\Users\91944\OneDrive\Desktop\ctde_rms_project")

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from env.environment import ImprovedRMS_Env
from agents.dueling_attention_dqn import DuelingAttentionAgent


def train_enhanced_dqn(
    episodes=500,
    n_jobs=30,
    n_machines=4,
    gamma=0.99,
    lr=1e-4,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.998,
    reward_norm_factor=10.0,
    save_dir="results_enhanced",
    model_path="models/dueling_attention_dqn.pt",
):
    """Train Enhanced DQN with attention + dueling heads."""
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Compute dimensions
    state_dim = 5 + (7 * n_machines) + 3
    action_dim = n_machines

    # Initialize agent and optimizer
    agent = DuelingAttentionAgent(state_dim, action_dim, device="cpu")
    optimizer = optim.Adam(agent.q_network.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()  # Huber loss

    logs = {"episode_rewards": [], "episode_losses": []}
    epsilon = epsilon_start

    print(f"🚀 Starting Enhanced DQN training for {episodes} episodes...\n")

    for ep in range(episodes):
        env = ImprovedRMS_Env(n_jobs=n_jobs, n_machines=n_machines, seed=ep)
        total_reward, total_loss = 0.0, 0.0

        for job in env.jobs:
            state = np.random.rand(state_dim).astype(np.float32)

            # ε-greedy exploration
            if np.random.rand() < epsilon:
                action = np.random.randint(action_dim)
            else:
                action = agent.predict(state)

            # Environment step
            reward, _ = env.assign_job(job, training=True, return_components=True)
            reward = np.clip(reward / reward_norm_factor, -1.0, 1.0)

            next_state = np.random.rand(state_dim).astype(np.float32)
            done = job.completed

            # Compute Q-target
            state_t = torch.tensor(state).unsqueeze(0)
            next_t = torch.tensor(next_state).unsqueeze(0)
            q_val = agent.q_network(state_t)[0, action]
            with torch.no_grad():
                q_next = agent.q_network(next_t).max()
                target = torch.tensor(reward + (0 if done else gamma * q_next))

            loss = criterion(q_val, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.q_network.parameters(), 1.0)
            optimizer.step()

            total_reward += reward
            total_loss += loss.item()

        # Update exploration rate
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        logs["episode_rewards"].append(total_reward)
        logs["episode_losses"].append(total_loss)

        print(f"Ep {ep+1:03d}/{episodes} | Reward={total_reward:7.3f} | "
              f"Loss={total_loss:8.5f} | ε={epsilon:.3f}")

    # Save model + logs
    torch.save(agent.q_network.state_dict(), model_path)
    pd.DataFrame({
        "Episode": range(1, episodes+1),
        "Reward": logs["episode_rewards"],
        "Loss": logs["episode_losses"]
    }).to_csv(os.path.join(save_dir, "enhanced_dqn_training_logs.csv"), index=False)

    print("\n✅ Enhanced DQN training complete.")
    print("📁 Logs:", os.path.join(save_dir, "enhanced_dqn_training_logs.csv"))
    print("💾 Model:", model_path)


if __name__ == "__main__":
    train_enhanced_dqn()
