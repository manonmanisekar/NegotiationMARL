import os
import sys
sys.path.append(r"C:\Users\91944\OneDrive\Desktop\ctde_rms_project")

"""
trainer.py
----------
Stable DQN training for CTDE RMS Project with corrected methodology:
 - Environment reset each episode
 - Proper exploration decay
 - Reward normalization
 - Gradient stability
 - Long training (default: 500 episodes)
 - Data logged for research visualization
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from env.environment import ImprovedRMS_Env
from agents.dqn_agent import StableEnhancedDQNAgent

# Compatibility for NumPy ≥1.24
if not hasattr(np, "float"): np.float = float

def moving_average(data, window_size=10):
    """Simple moving average for smoother reward visualization."""
    return np.convolve(data, np.ones(window_size)/window_size, mode="valid")


def train_and_export(
    episodes=500,
    n_jobs=30,
    n_machines=4,
    save_dir="results",
    model_path="models/enhanced_dqn.pt",
    gamma=0.99,
    lr=1e-4,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.998,  # slower, more exploration
    reward_norm_factor=10.0
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    state_dim = 5 + (7 * n_machines) + 3
    action_dim = n_machines

    agent = StableEnhancedDQNAgent(state_dim, action_dim, device="cpu")
    agent.q_network = agent.q_network.float()

    optimizer = optim.Adam(agent.q_network.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()  # Huber loss improves stability

    logs = {"episode_rewards": [], "episode_losses": []}
    epsilon = epsilon_start

    print(f"🚀 Starting DQN training for {episodes} episodes...\n")

    for ep in range(episodes):
        # ✅ Reset environment each episode
        env = ImprovedRMS_Env(n_jobs=n_jobs, n_machines=n_machines, seed=ep)
        total_reward, total_loss = 0.0, 0.0

        for job in env.jobs:
            # Random initial state
            state = np.random.rand(state_dim).astype(np.float32)

            # ε-greedy policy
            if np.random.rand() < epsilon:
                action = np.random.randint(action_dim)
            else:
                with torch.no_grad():
                    q_vals = agent.q_network(torch.tensor(state).unsqueeze(0))
                    action = torch.argmax(q_vals).item()

            # Step environment
            reward, _ = env.assign_job(job, training=True, return_components=True)

            # ✅ Normalize reward for numerical stability
            reward = np.clip(reward / reward_norm_factor, -1.0, 1.0)
            next_state = np.random.rand(state_dim).astype(np.float32)
            done = job.completed

            # Q-learning target
            state_tensor = torch.tensor(state).unsqueeze(0)
            next_state_tensor = torch.tensor(next_state).unsqueeze(0)
            q_value = agent.q_network(state_tensor)[0, action]
            with torch.no_grad():
                q_next = agent.q_network(next_state_tensor).max()
                target = torch.tensor(reward + (0 if done else gamma * q_next))

            loss = criterion(q_value, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.q_network.parameters(), 1.0)
            optimizer.step()

            total_reward += reward
            total_loss += loss.item()

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        logs["episode_rewards"].append(total_reward)
        logs["episode_losses"].append(total_loss)

        print(f"Ep {ep+1:03d}/{episodes} | Reward={total_reward:7.3f} | Loss={total_loss:8.5f} | ε={epsilon:.3f}")

    # Save artifacts
    torch.save(agent.q_network.state_dict(), model_path)
    pd.DataFrame({
        "Episode": range(1, episodes+1),
        "Reward": logs["episode_rewards"],
        "Loss": logs["episode_losses"]
    }).to_csv(os.path.join(save_dir, "training_logs.csv"), index=False)
    print("\n Training complete! Logs saved at:", os.path.join(save_dir, "training_logs.csv"))


if __name__ == "__main__":
    train_and_export()
