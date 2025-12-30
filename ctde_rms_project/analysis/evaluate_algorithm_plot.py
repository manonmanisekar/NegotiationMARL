"""
visualize_agent_performance_advanced.py
---------------------------------------
Generates multiple advanced comparative visualizations for scheduling agents.
"""
import os
import sys
sys.path.append(r"C:\Users\91944\OneDrive\Desktop\ctde_rms_project")

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pandas.plotting import parallel_coordinates

# -----------------------------------------------------------------
# 1️⃣ Load and preprocess data
# -----------------------------------------------------------------
DATA_PATH = "results/compare_ext_results.csv"  # Update path as needed
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip() for c in df.columns]

metrics = ["Makespan", "TotalTardiness", "AvgUtilization", "AvgSetupTime"]
agents = df["Agent"].unique()
os.makedirs("results/plots", exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.2)
palette = sns.color_palette("husl", len(agents))

# ============================================================
# 1️⃣ Boxplots for each metric
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
titles = [
    "Makespan (↓ better)",
    "Total Tardiness (↓ better)",
    "Average Utilization (↑ better)",
    "Average Setup Time (↓ better)"
]

for i, metric in enumerate(metrics):
    sns.boxplot(data=df, x="Agent", y=metric, ax=axes[i],
                palette=palette, linewidth=1.2, showfliers=False)
    sns.swarmplot(data=df, x="Agent", y=metric, ax=axes[i],
                  color="black", alpha=0.5, size=3)
    axes[i].set_title(titles[i], fontsize=13, weight="bold")
    axes[i].set_xlabel("")
    axes[i].tick_params(axis="x", rotation=20)

fig.suptitle("Real Performance Comparison of Scheduling Agents", fontsize=16, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("results/plots/real_boxplots.png", dpi=300)
plt.close()

# ============================================================
# 2️⃣ Mean ± Std Bar Plot
# ============================================================
summary = df.groupby("Agent")[metrics].agg(["mean", "std"])
summary.columns = [f"{a}_{b}" for a, b in summary.columns]
summary = summary.reset_index()

bar_width = 0.18
x = np.arange(len(summary))
fig, ax = plt.subplots(figsize=(10, 6))

for i, metric in enumerate(metrics):
    ax.bar(x + i * bar_width, summary[f"{metric}_mean"],
           yerr=summary[f"{metric}_std"], width=bar_width,
           label=metric, capsize=3, alpha=0.9)

ax.set_xticks(x + 1.5 * bar_width)
ax.set_xticklabels(summary["Agent"])
ax.set_ylabel("Actual Value")
ax.set_title("Mean ± Std Across Real Metrics", fontsize=14, weight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("results/plots/real_bar_error.png", dpi=300)
plt.close()

# ============================================================
# 3️⃣ Scatter (Pareto Frontier: Makespan vs. Tardiness)
# ============================================================
plt.figure(figsize=(7, 5))
for agent in agents:
    subset = df[df["Agent"] == agent]
    plt.scatter(subset["Makespan"], subset["TotalTardiness"],
                s=70, alpha=0.8, label=agent)
plt.title("Pareto Frontier: Makespan vs. Total Tardiness", fontsize=14, weight="bold")
plt.xlabel("Makespan")
plt.ylabel("Total Tardiness")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/plots/real_pareto.png", dpi=300)
plt.close()

# ============================================================
# 4️⃣ Parallel Coordinates (raw values, scaled only for view)
# ============================================================
scaled = df.copy()
for m in metrics:
    scaled[m] = (scaled[m] - scaled[m].min()) / (scaled[m].max() - scaled[m].min())

plt.figure(figsize=(9, 6))
parallel_coordinates(scaled[["Agent"] + metrics], "Agent", color=palette, linewidth=2)
plt.title("Parallel Coordinates (Scaled for Visualization)", fontsize=14, weight="bold")
plt.ylabel("Scaled Value (0–1 for plotting only)")
plt.tight_layout()
plt.savefig("results/plots/real_parallel.png", dpi=300)
plt.close()

# ============================================================
# 5️⃣ Line Profile (Real mean trends per agent)
# ============================================================
means = df.groupby("Agent")[metrics].mean()
plt.figure(figsize=(7, 5))
for agent in agents:
    plt.plot(metrics, means.loc[agent], marker="o", linewidth=2, label=agent)
plt.title("Performance Profile (Real Values)", fontsize=14, weight="bold")
plt.ylabel("Metric Value")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/plots/real_profile.png", dpi=300)
plt.close()

print("✅ All real-data plots saved → results/plots/")
