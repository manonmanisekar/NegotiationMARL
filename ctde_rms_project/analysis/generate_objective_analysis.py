
import os,sys
sys.path.append(r"C:\Users\91944\OneDrive\Desktop\ctde_rms_project")

"""
generate_objective_analysis.py
---------------------------------------
Runs evaluation of different RMS configurations:
 - With negotiation & reconfiguration
 - Without negotiation / reconfiguration
Generates:
 - results/objective_analysis.csv
 - plots/objective_comparison.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import correct environment
from env.envm import ImprovedRMS_Env
from agents.dueling_attention_dqn import DuelingAttentionDQN  # or your agent class


# -------------------------------------------------------------
# 🧠 Utility: Ensure folders
# -------------------------------------------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# -------------------------------------------------------------
# ⚙️ Main Objective Analysis Function
# -------------------------------------------------------------
def generate_objective_analysis(
    env_class=ImprovedRMS_Env,
    agent_class=DuelingAttentionDQN,
    output_file="results/objective_analysis.csv",
    num_episodes=5,
):
    ensure_dir("results/objective_metrics")

    configs = {
        "with_neg_reconfig": {"neg": True, "reconf": True},
        "with_neg_fixed": {"neg": True, "reconf": False},
        "without_neg_reconfig": {"neg": False, "reconf": True},
        "without_neg_fixed": {"neg": False, "reconf": False},
    }

    all_results = []

    for name, cfg in configs.items():
        print(f"\n🔹 Evaluating configuration: {name}")

        env = env_class(negotiation=cfg["neg"], reconfigurable=cfg["reconf"], eval_mode=True)

        # ✅ FIXED: Extract actual state/action dimensions
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        agent = agent_class(state_dim, action_dim)

        config_results = []
        for ep in tqdm(range(num_episodes), desc=f"Running {name}"):
            env.reset()
            done = False
            total_reward = 0.0

            while not done:
                # Use agent policy if available, else random
                if hasattr(agent, "select_action"):
                    state = env._get_state()
                    action = agent.select_action(state)
                else:
                    action = env.action_space.sample()

                obs, reward, done, info = env.step(action)
                total_reward += reward

            metrics = env.compute_metrics()
            metrics["total_reward"] = total_reward
            metrics["config"] = name
            config_results.append(metrics)

        df = pd.DataFrame(config_results)
        df.to_csv(f"results/objective_metrics/{name}.csv", index=False)
        all_results.append(df)

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(output_file, index=False)
    print(f"\n✅ Saved objective analysis results to {output_file}")

    # ---------------------------------------------------------
    # 📊 Plot Visualization
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ["makespan", "tardiness", "utilization", "total_reward"]

    for i, metric in enumerate(metrics_to_plot, 1):
        plt.subplot(2, 2, i)
        final_df.boxplot(column=metric, by="config", grid=False)
        plt.title(metric)
        plt.xticks(rotation=20)
        plt.tight_layout()

    plt.suptitle("Objective Comparison across RMS Configurations", fontsize=14)
    plt.savefig("results/objective_metrics/objective_comparison.png", dpi=300, bbox_inches="tight")
    print("📊 Saved plot: results/objective_metrics/objective_comparison.png")

    return final_df


# -------------------------------------------------------------
# 🚀 Entry Point
# -------------------------------------------------------------
if __name__ == "__main__":
    results = generate_objective_analysis(
        env_class=ImprovedRMS_Env,
        agent_class=DuelingAttentionDQN,
        output_file="results/objective_analysis.csv",
        num_episodes=5,
    )
    print("\n✅ Objective analysis completed successfully.")
    print(results.groupby("config").mean()[["makespan", "tardiness", "utilization", "total_reward"]])
