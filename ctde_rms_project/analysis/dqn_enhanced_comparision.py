"""
compare_dqn_vs_enhanced.py
---------------------------
Generates research-grade comparative plots for:
 - Baseline Enhanced DQN (Stable)
 - Dueling + Attention Enhanced DQN
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def moving_average(data, window=20):
    return np.convolve(data, np.ones(window)/window, mode="valid")


def compare_dqn_vs_enhanced(
    baseline_log="results/training_logs.csv",
    enhanced_log="results_enhanced/enhanced_dqn_training_logs.csv",
    save_dir="results_comparison"
):
    os.makedirs(save_dir, exist_ok=True)

    # Load both results
    df_base = pd.read_csv(baseline_log)
    df_enh = pd.read_csv(enhanced_log)

    base_reward = df_base["Reward"].values
    base_loss = df_base["Loss"].values
    enh_reward = df_enh["Reward"].values
    enh_loss = df_enh["Loss"].values

    # Moving averages
    base_reward_ma = moving_average(base_reward)
    enh_reward_ma = moving_average(enh_reward)
    base_loss_ma = moving_average(base_loss)
    enh_loss_ma = moving_average(enh_loss)

    plt.rcParams.update({
        "font.size": 12,
        "axes.labelweight": "bold",
        "axes.titlesize": 14,
        "figure.figsize": (12, 4),
        "axes.grid": True
    })

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # === Reward Comparison ===
    axs[0].plot(base_reward_ma, color="tab:blue", linewidth=2, label="Baseline DQN (MA)")
    axs[0].plot(enh_reward_ma, color="tab:green", linewidth=2, label="Dueling + Attention DQN (MA)")
    axs[0].set_title("Episode Reward Comparison")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Total Reward")
    axs[0].legend()
    axs[0].grid(True, linestyle="--", alpha=0.6)

    # === Loss Comparison ===
    axs[1].plot(base_loss_ma, color="tab:orange", linewidth=2, label="Baseline DQN (MA)")
    axs[1].plot(enh_loss_ma, color="tab:red", linewidth=2, label="Dueling + Attention DQN (MA)")
    axs[1].set_title("Training Loss Comparison")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Loss")
    axs[1].legend()
    axs[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "dqn_comparison.png")
    plt.savefig(out_path, dpi=300)
    print(f" Saved comparative plots at: {out_path}")
    plt.show()


if __name__ == "__main__":
    compare_dqn_vs_enhanced()
