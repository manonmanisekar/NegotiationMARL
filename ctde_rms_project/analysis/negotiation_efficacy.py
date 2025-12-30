import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ======================================================
# Negotiation & Reconfigurability Analysis (Real Data)
# Includes: Scenario Comparison + Dynamics Plot
# ======================================================

def prepare_scenario_files(source_file="results/objective_analysis.csv", save_dir="results/objective_metrics"):
    """Split the main RMS experiment data into four negotiation/reconfigurability scenarios."""
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Source data not found: {source_file}")

    df = pd.read_csv(source_file)
    df.columns = [c.strip().lower() for c in df.columns]

    required = ["episode", "makespan", "tardiness", "utilization", "negotiation", "reconfigurable"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing required column: '{col}'")

    os.makedirs(save_dir, exist_ok=True)
    scenarios = {
        "with_neg_reconfig": (True, True),
        "without_neg_reconfig": (False, True),
        "with_neg_fixed": (True, False),
        "without_neg_fixed": (False, False),
    }

    saved_files = []
    for name, (neg, reconf) in scenarios.items():
        subset = df[(df["negotiation"] == neg) & (df["reconfigurable"] == reconf)]
        if not subset.empty:
            file_path = os.path.join(save_dir, f"{name}.csv")
            subset.to_csv(file_path, index=False)
            saved_files.append(file_path)
            print(f"[INFO] Saved {len(subset)} rows → {file_path}")
        else:
            print(f"[WARN] No records found for scenario: {name}")

    return saved_files


def plot_negotiation_dynamics(save_dir="results/objective_metrics", out_path="results/negotiation_dynamics.png"):
    """Generate boxplots for makespan, tardiness, and utilization."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    csv_files = [f for f in os.listdir(save_dir) if f.endswith(".csv")]

    if not csv_files:
        raise FileNotFoundError(f"No scenario CSVs found in {save_dir}")

    all_data = []
    for file in csv_files:
        df = pd.read_csv(os.path.join(save_dir, file))
        scenario_name = file.replace(".csv", "").replace("_", " ").title().replace("Reconfig", "Reconfigurable")
        df["Scenario"] = scenario_name
        all_data.append(df)

    full_df = pd.concat(all_data, ignore_index=True)
    metrics = ["makespan", "tardiness", "utilization"]
    titles = ["Makespan", "Total Tardiness", "Avg Utilization"]
    colors = ["skyblue", "lightcoral", "lightgreen", "orange"]
    scenarios = full_df["Scenario"].unique()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Impact of Negotiation and Reconfigurability on RMS Performance", fontsize=14, fontweight="bold")

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        bp = axes[i].boxplot(
            [full_df[full_df["Scenario"] == s][metric] for s in scenarios],
            patch_artist=True,
            labels=scenarios,
            medianprops=dict(color="black")
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        axes[i].set_title(title, fontweight="bold")
        axes[i].tick_params(axis="x", rotation=25)
        axes[i].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"[INFO] Negotiation efficacy boxplot saved at: {out_path}")


def plot_negotiation_time_series(source_file="results/objective_analysis.csv",
                                 out_path="results/negotiation_dynamics_time.png"):
    """Plot negotiation success rate and reconfiguration frequency over episodes."""
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Source file not found: {source_file}")

    df = pd.read_csv(source_file)
    df.columns = [c.strip().lower() for c in df.columns]

    # Require relevant fields
    needed = ["episode", "negotiation_success", "reconfig_freq"]
    for col in needed:
        if col not in df.columns:
            raise KeyError(f"Missing column: '{col}' in {source_file}")

    df_grouped = df.groupby("episode").mean()

    # Smoothing
    df_grouped["negotiation_success_smooth"] = df_grouped["negotiation_success"].rolling(window=5, min_periods=1).mean()
    df_grouped["reconfig_freq_smooth"] = df_grouped["reconfig_freq"].rolling(window=5, min_periods=1).mean()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(df_grouped.index, df_grouped["negotiation_success_smooth"], label="Negotiation Success Rate", linewidth=2)
    plt.plot(df_grouped.index, df_grouped["reconfig_freq_smooth"], label="Reconfiguration Frequency", linewidth=2, linestyle="--")
    plt.title("Negotiation Dynamics over Training Episodes", fontsize=13, fontweight="bold")
    plt.xlabel("Episode")
    plt.ylabel("Rate / Frequency")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()

    print(f"[INFO] Negotiation time-series plot saved at: {out_path}")


if __name__ == "__main__":
    print("========== Negotiation & Reconfigurability Analysis ==========")
    try:
        # Step 1: Split scenarios
        prepare_scenario_files(source_file="results/objective_analysis.csv",
                               save_dir="results/objective_metrics")

        # Step 2: Scenario comparison (boxplots)
        plot_negotiation_dynamics(save_dir="results/objective_metrics",
                                  out_path="results/negotiation_dynamics.png")

        # Step 3: Negotiation time-series
        plot_negotiation_time_series(source_file="results/objective_analysis.csv",
                                     out_path="results/negotiation_dynamics_time.png")

    except Exception as e:
        print(f"[ERROR] {e}")
