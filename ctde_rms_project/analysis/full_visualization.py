import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

plt.style.use("ggplot")

def full_visualization(results_dir="results"):
    save_dir = os.path.join(results_dir, "analysis")
    os.makedirs(save_dir, exist_ok=True)

    # Episode Rewards
    kpi_path = os.path.join(results_dir, "kpi_results.csv")
    if os.path.exists(kpi_path):
        df = pd.read_csv(kpi_path)
        plt.figure(figsize=(6, 4))
        plt.plot(df["Episode"], df["TotalReward"], marker="o")
        plt.title("Episode Rewards over Training")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.tight_layout()
        plt.show()
        # ---- DQN Loss Curve ----
    if os.path.exists(kpi_path):
        df = pd.read_csv(kpi_path)
        if "Loss" in df.columns:
            plt.figure(figsize=(6, 4))
            plt.plot(df["Episode"], df["Loss"], marker="s", color="orange")
            plt.title("Training Loss per Episode")
            plt.xlabel("Episode")
            plt.ylabel("Loss")
            plt.tight_layout()
            plt.show()

    # Agent comparison
    comp_path = os.path.join(results_dir, "compare_ext_results.csv")
    if os.path.exists(comp_path):
        dfc = pd.read_csv(comp_path)
        plt.figure(figsize=(7, 5))
        sns.boxplot(data=dfc, x="Agent", y="Makespan")
        plt.title("Makespan Comparison Across Agents")
        plt.show()

        plt.figure(figsize=(7, 5))
        sns.boxplot(data=dfc, x="Agent", y="TotalTardiness")
        plt.title("Total Tardiness Comparison Across Agents")
        plt.show()

    # Negotiation analysis (if available)
    negoti_path = os.path.join(results_dir, "negotiation_log.json")
    if os.path.exists(negoti_path):
        with open(negoti_path, "r") as f:
            logs = json.load(f)
        n_jobs = len(logs)
        if n_jobs:
            level3_changes = sum(1 for l in logs if l.get("level3_choice") != l.get("level1_choice"))
            avg_bids = sum(len(l.get("level1_bids", [])) for l in logs) / n_jobs
            print(f"Average negotiation bids per job: {avg_bids:.2f}")
            print(f"Global overrides (L3 ≠ L1): {level3_changes}/{n_jobs} = {100*level3_changes/n_jobs:.1f}%")

    print(f"Plots saved to: {save_dir}")

if __name__ == "__main__":
    full_visualization()
