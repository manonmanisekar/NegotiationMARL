import os
import sys
sys.path.append(r"C:\Users\91944\OneDrive\Desktop\ctde_rms_project")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel

sns.set(style="whitegrid")

ALGO_COLORS = {
    "DQN": "#1f77b4",
    "Double DQN": "#ff7f0e",
    "SAC": "#2ca02c",
    "Dueling-Attn-DQN": "#d62728"
}

# ===============================================================
# Helper: moving average
# ===============================================================
def moving_average(x, window=20):
    if len(x) < window:
        return np.array(x)
    return np.convolve(x, np.ones(window) / window, mode="valid")

# ===============================================================
# Load Real Data
# ===============================================================
def load_real_rewards(result_dir="results"):
    """
    Loads reward data from CSV files named like:
    results/DQN_rewards.csv, results/Double_DQN_rewards.csv, etc.
    Each CSV must have a column named 'Reward'.
    """
    reward_logs = {}
    for algo in ALGO_COLORS.keys():
        file_path = os.path.join(result_dir, f"{algo.replace(' ', '_')}_rewards.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if "Reward" not in df.columns:
                raise ValueError(f"CSV {file_path} must contain a 'Reward' column.")
            reward_logs[algo] = df["Reward"].to_numpy()
            print(f"✅ Loaded {algo}: {len(df)} episodes from {file_path}")
        else:
            print(f"⚠️ Warning: Missing file for {algo}: {file_path}")
    return reward_logs

# ===============================================================
# Plot 1: Learning and Convergence
# ===============================================================
def plot_learning_and_convergence(reward_logs, save_dir="results"):
    plt.rcParams.update({
        "font.size": 10,
        "font.family": "serif",
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "savefig.dpi": 300
    })

    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Learning and Convergence Comparison of RL Algorithms", fontsize=12, fontweight="bold")

    # (a) Learning Curves (no smoothing)
    for algo, rewards in reward_logs.items():
        axes[0].plot(rewards, label=algo, color=ALGO_COLORS[algo], linewidth=1.3)
    axes[0].set_title("(a) Learning Curves", fontweight="bold", loc="left")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.3)

    # (b) Convergence (Cumulative Mean Reward)
    for algo, rewards in reward_logs.items():
        cum_mean = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
        axes[1].plot(cum_mean, label=algo, color=ALGO_COLORS[algo], linewidth=1.3)
    axes[1].set_title("(b) Convergence Behavior", fontweight="bold", loc="left")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Cumulative Mean Reward")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison_learning.png"), bbox_inches="tight")
    plt.savefig(os.path.join(save_dir, "comparison_learning.pdf"), bbox_inches="tight")
    plt.show()
    print("📊 Saved learning curves → results/comparison_learning.[png|pdf]")

# ===============================================================
# Plot 2: Statistical Results and LaTeX Table
# ===============================================================
def plot_statistics_and_summary(reward_logs, efficiency_data=None, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    means, stds, maxs, mins, meds = [], [], [], [], []
    algos = list(reward_logs.keys())

    for algo in algos:
        rewards = np.array(reward_logs[algo])
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))
        maxs.append(np.max(rewards))
        mins.append(np.min(rewards))
        meds.append(np.median(rewards))

    df_summary = pd.DataFrame({
        "Algorithm": algos,
        "Mean": means,
        "Std": stds,
        "Max": maxs,
        "Min": mins,
        "Median": meds
    })

    if efficiency_data:
        df_summary["Efficiency"] = [efficiency_data.get(a, np.nan) for a in algos]

    baseline = df_summary.loc[df_summary["Algorithm"] == "DQN", "Mean"].values[0]
    df_summary["Improvement_%"] = ((df_summary["Mean"] - baseline) / baseline) * 100

    csv_path = os.path.join(save_dir, "rl_comparison_summary.csv")
    tex_path = os.path.join(save_dir, "rl_comparison_table.tex")
    df_summary.to_csv(csv_path, index=False)

    with open(tex_path, "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Performance Comparison of RL Algorithms in RMS Scheduling}\n")
        f.write("\\label{tab:rl_comparison}\n")
        f.write("\\begin{tabular}{lcccccc}\n\\hline\n")
        f.write("Algorithm & Mean & Std & Max & Min & Median & Improvement(\\%) \\\\\n\\hline\n")
        for _, row in df_summary.iterrows():
            f.write(f"{row['Algorithm']} & {row['Mean']:.3f} & {row['Std']:.3f} & "
                    f"{row['Max']:.3f} & {row['Min']:.3f} & {row['Median']:.3f} & "
                    f"{row['Improvement_%']:.2f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    print(f"📄 Exported summary CSV → {csv_path}")
    print(f"📘 Exported LaTeX table → {tex_path}")

    plt.rcParams.update({
        "font.size": 10,
        "font.family": "serif",
        "savefig.dpi": 300
    })

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Comparative Statistical Analysis of RL Algorithms", fontsize=12, fontweight="bold")

    sns.barplot(x="Algorithm", y="Mean", data=df_summary, ax=axes[0],
                palette=ALGO_COLORS, yerr=df_summary["Std"])
    axes[0].set_title("(a) Mean Reward ± Std", fontweight="bold", loc="left")
    axes[0].set_ylabel("Mean Reward")
    axes[0].grid(alpha=0.3)

    if "Efficiency" in df_summary:
        sns.barplot(x="Algorithm", y="Efficiency", data=df_summary, ax=axes[1],
                    palette=ALGO_COLORS)
        axes[1].set_title("(b) Efficiency (Reward / Time)", fontweight="bold", loc="left")
        axes[1].set_ylabel("Efficiency")
        axes[1].grid(alpha=0.3)
    else:
        axes[1].set_visible(False)

    sns.barplot(x="Algorithm", y="Improvement_%", data=df_summary, ax=axes[2],
                palette=ALGO_COLORS)
    axes[2].set_title("(c) Relative Improvement over DQN", fontweight="bold", loc="left")
    axes[2].set_ylabel("Improvement (%)")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison_statistics.png"), bbox_inches="tight")
    plt.savefig(os.path.join(save_dir, "comparison_statistics.pdf"), bbox_inches="tight")
    plt.show()
    print("📊 Saved statistical plots → results/comparison_statistics.[png|pdf]")

# ===============================================================
# Statistical Significance Tests
# ===============================================================
def run_significance_tests(reward_logs):
    keys = list(reward_logs.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a1, a2 = keys[i], keys[j]
            _, p = ttest_rel(reward_logs[a1], reward_logs[a2])
            print(f"p-value ({a1} vs {a2}) = {p:.4e}")

# ===============================================================
# Main Execution
# ===============================================================
if __name__ == "__main__":
    # Load your real experimental data
    reward_logs = load_real_rewards("results")

    if not reward_logs:
        raise FileNotFoundError("❌ No reward CSV files found in 'results/'. Please add them first.")

    # Optional: Provide real efficiency data (normalized reward per training time)
    efficiency = {
        "DQN": 0.211,
        "Double DQN": 0.178,
        "SAC": 0.129,
        "Dueling-Attn-DQN": 0.178
    }

    plot_learning_and_convergence(reward_logs)
    plot_statistics_and_summary(reward_logs, efficiency_data=efficiency)
    run_significance_tests(reward_logs)
