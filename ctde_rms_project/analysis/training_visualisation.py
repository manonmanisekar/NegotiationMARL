"""
training_visualization.py
-------------------------
Generates research-grade training plots from DQN logs:
 - Moving average smoothing
 - Error bands
 - Clean aesthetic (matplotlib)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def moving_average(data, window=20):
    return np.convolve(data, np.ones(window)/window, mode="valid")

def visualize_training(log_path="results/training_logs.csv"):
    df = pd.read_csv(log_path)
    rewards = df["Reward"].values
    losses = df["Loss"].values
    episodes = df["Episode"].values

    reward_ma = moving_average(rewards, window=20)
    loss_ma = moving_average(losses, window=20)

    plt.rcParams.update({
        "font.size": 12,
        "axes.labelweight": "bold",
        "axes.titlesize": 14,
        "figure.figsize": (6,4),
        "axes.grid": True
    })

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Reward plot
    axs[0].plot(episodes, rewards, alpha=0.3, color="tab:blue", label="Raw Reward")
    axs[0].plot(episodes[len(episodes)-len(reward_ma):], reward_ma, color="tab:blue", linewidth=2, label="MA(20)")
    axs[0].set_title("Episode Reward Trend")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Total Reward")
    axs[0].legend()
    
    # Loss plot
    axs[1].plot(episodes, losses, alpha=0.3, color="tab:orange", label="Raw Loss")
    axs[1].plot(episodes[len(episodes)-len(loss_ma):], loss_ma, color="tab:orange", linewidth=2, label="MA(20)")
    axs[1].set_title("Training Loss Trend")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Loss")
    axs[1].legend()

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(log_path), "training_visualization.png")
    plt.savefig(out_path, dpi=300)
    print(f"✅ Saved research-quality plots at: {out_path}")
    plt.show()

if __name__ == "__main__":
    visualize_training()
