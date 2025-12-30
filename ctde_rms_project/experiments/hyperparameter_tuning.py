"""
run_hyperparameter_tuning.py
-----------------------------
Runs multiple training sessions with varying hyperparameters (learning rate, etc.)
and saves logs in 'results_tuning' for LaTeX analysis and plotting.
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


def train_variant(lr, gamma=0.99, epsilon_decay=0.998, episodes=200):
    """Train a short run with specified hyperparameters."""
    save_dir = "results_tuning"
    os.makedirs(save_dir, exist_ok=True)

    n_jobs, n_machines = 30, 4
    state_dim = 5 + (7 * n_machines) + 3
    action_dim = n_machines

    agent = DuelingAttentionAgent(state_dim, action_dim, device="cpu")
    optimizer = optim.Adam(agent.q_network.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()

    epsilon, eps_end = 1.0, 0.05
    logs = {"Episode": [], "Reward": [], "Loss": []}

    for ep in range(episodes):
        env = ImprovedRMS_Env(n_jobs=n_jobs, n_machines=n_machines, seed=ep)
        total_reward, total_loss = 0.0, 0.0

        for job in env.jobs:
            state = np.random.rand(state_dim).astype(np.float32)
            action = np.random.randint(action_dim) if np.random.rand() < epsilon else agent.predict(state)
            reward, _ = env.assign_job(job, training=True, return_components=True)
            reward = np.clip(reward / 10.0, -1.0, 1.0)
            next_state = np.random.rand(state_dim).astype(np.float32)
            done = job.completed

            # Q-learning update
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

        epsilon = max(eps_end, epsilon * epsilon_decay)
        logs["Episode"].append(ep + 1)
        logs["Reward"].append(total_reward)
        logs["Loss"].append(total_loss)
        print(f"[lr={lr}] Ep {ep+1}/{episodes} | Reward={total_reward:.3f} | Loss={total_loss:.4f}")

    # Save log file
    out_path = os.path.join(save_dir, f"lr_{lr}.csv")
    pd.DataFrame(logs).to_csv(out_path, index=False)
    print(f"✅ Saved tuning log: {out_path}")


if __name__ == "__main__":
    # Try a few learning rates
    for lr in [1e-3, 5e-4, 1e-4, 5e-5]:
        train_variant(lr=lr, episodes=200)
