import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import variation

# -------------------------------------------------
# 1. Load and normalize data
# -------------------------------------------------
files = {
    "With Negotiation + Reconfigurable": "C:/Users/91944/OneDrive/Desktop/ctde_rms_project/results/objective_metrics/with_neg_reconfig.csv",
    "Without Negotiation + Reconfigurable": "C:/Users/91944/OneDrive/Desktop/ctde_rms_project/results/objective_metrics/without_neg_reconfig.csv",
    "With Negotiation + Fixed": "C:/Users/91944/OneDrive/Desktop/ctde_rms_project/results/objective_metrics/with_neg_fixed.csv",
    "Without Negotiation + Fixed": "C:/Users/91944/OneDrive/Desktop/ctde_rms_project/results/objective_metrics/without_neg_fixed.csv",
}

dfs = []
for label, path in files.items():
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["Scenario"] = label
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

# Force numeric conversion (important!)
for col in ["makespan", "tardiness", "utilization"]:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

os.makedirs("results", exist_ok=True)
data.to_csv("results/negotiation_reconfig_metrics.csv", index=False)
print("✅ Combined dataset saved → results/negotiation_reconfig_metrics.csv")

# -------------------------------------------------
# 2. Compute summary statistics
# -------------------------------------------------
summary = data.groupby("Scenario").agg(
    makespan_mean=("makespan", "mean"),
    makespan_std=("makespan", "std"),
    tardiness_mean=("tardiness", "mean"),
    tardiness_std=("tardiness", "std"),
    util_mean=("utilization", "mean"),
    util_std=("utilization", "std"),
).reset_index()

summary["util_cv"] = data.groupby("Scenario")["utilization"].apply(variation).values

# -------------------------------------------------
# 3. Compute Negotiation Efficiency (handle divide-by-zero safely)
# -------------------------------------------------
def safe_eff_gain(with_val, without_val):
    if without_val == 0 or np.isnan(without_val):
        return 0.0
    return (without_val - with_val) / without_val * 100

eff_table = []
pairs = [
    ("With Negotiation + Reconfigurable", "Without Negotiation + Reconfigurable"),
    ("With Negotiation + Fixed", "Without Negotiation + Fixed"),
]
for w, wo in pairs:
    d_with = summary[summary["Scenario"] == w]
    d_wo = summary[summary["Scenario"] == wo]
    eff_table.append({
        "Condition": w.replace("With Negotiation + ", ""),
        "Makespan_Reduction(%)": safe_eff_gain(d_with["makespan_mean"].values[0], d_wo["makespan_mean"].values[0]),
        "Tardiness_Reduction(%)": safe_eff_gain(d_with["tardiness_mean"].values[0], d_wo["tardiness_mean"].values[0]),
        "Utilization_Balance_Improvement(%)": safe_eff_gain(d_with["util_cv"].values[0], d_wo["util_cv"].values[0]),
    })

eff_df = pd.DataFrame(eff_table)
print("\n=== Negotiation Efficiency Summary ===")
print(eff_df.round(2))

# -------------------------------------------------
# 4. Fairness (Gini Index)
# -------------------------------------------------
def gini_coefficient(x):
    x = np.sort(np.array(x))
    n = len(x)
    if n == 0 or np.sum(x) == 0:
        return np.nan
    cumx = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

gini_summary = data.groupby("Scenario")["utilization"].apply(gini_coefficient)
print("\n=== Gini Fairness Index (lower = fairer) ===")
print(gini_summary)

# -------------------------------------------------
# 5. Boxplot Visualization (force numeric only)
# -------------------------------------------------
sns.set(style="whitegrid", font_scale=1.2)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
palette = ["skyblue", "lightcoral", "lightgreen", "orange"]

metrics = ["makespan", "tardiness", "utilization"]
titles = ["Makespan", "Total Tardiness", "Avg Utilization"]

for ax, metric, title in zip(axes, metrics, titles):
    clean_df = data.dropna(subset=[metric])
    if clean_df[metric].dtype not in [np.float64, np.float32, np.int64]:
        print(f"⚠️ Skipping non-numeric column: {metric}")
        continue
    sns.boxplot(data=clean_df, x="Scenario", y=metric, palette=palette, ax=ax, showfliers=False)
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=25)

fig.suptitle("Impact of Negotiation and Reconfigurability on RMS Performance (Part 1)",
             fontsize=15, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])

os.makedirs("results/plots", exist_ok=True)
plot_path = os.path.join("results/plots", "negotiation_reconfig_part1_fixed.png")
plt.savefig(plot_path, dpi=300)
plt.show()
print(f"\n📊 Plot saved → {plot_path}")

# -------------------------------------------------
# 6. Save summary and efficiency data
# -------------------------------------------------
summary_path = "results/negotiation_summary_stats.csv"
eff_path = "results/negotiation_efficiency.csv"
summary.to_csv(summary_path, index=False)
eff_df.to_csv(eff_path, index=False)
print(f"📁 Summary saved → {summary_path}")
print(f"📁 Efficiency saved → {eff_path}")
