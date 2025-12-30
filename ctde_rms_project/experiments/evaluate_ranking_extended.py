# evaluate_ranking_extended.py
# Extended multi-instance evaluation + TOPSIS + paired t-tests + publication outputs

import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from scipy.stats import rankdata, ttest_rel
from pathlib import Path

# Import your environment and agents/training utilities
# Assumes debug_rl_comparison_full.py is in PYTHONPATH or same folder
from debug_rl_comparison import (
    EnhancedRMSWrapper,
    DoubleDQNAgent,
    SACAgent,
    DuelingAttentionDQNAgent,
    train_agent_simple  # training helper (used only if no saved model)
)

# -----------------------------
# CONFIG
# -----------------------------
OUT_DIR = Path("results/extended")
MODELS_DIR = Path("results/models")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

AGENTS = {
    "Double DQN": lambda s_dim, a_dim: DoubleDQNAgent(s_dim, a_dim),
    "SAC": lambda s_dim, a_dim: SACAgent(s_dim, a_dim),
    "Dueling-Attention DQN": lambda s_dim, a_dim: DuelingAttentionDQNAgent(s_dim, a_dim),
}

# scenarios to sweep — adjust or extend as needed
N_JOBS_LIST = [10, 20, 40]              # vary number of jobs
N_MACHINES_LIST = [3, 6]               # vary number of machines
SEEDS = [0, 1, 2]                      # matched seeds across agents (paired tests need same seeds)
EVAL_EPISODES = 20                     # episodes per instance for evaluation
MAX_STEPS = 1000                       # safety cap

# model handling
TRAIN_IF_MISSING = True                # if True, train short if model not found
TRAIN_EPISODES = 200                   # quick training if no saved model (increase for publication)
SAVE_MODELS = True                     # whether to save trained models

# TOPSIS weights (sum to 1). Adjust if you want to weight certain KPIs higher.
TOPSIS_WEIGHTS = np.array([0.25, 0.2, 0.25, 0.15, 0.15])  # [Makespan, Util, Tardiness, Setup, Reward]

# -----------------------------
# UTILITIES
# -----------------------------
def model_path(agent_name, seed, nj, nm):
    safe = agent_name.replace(" ", "_").replace("-", "_")
    return MODELS_DIR / f"{safe}_seed{seed}_nj{nj}_nm{nm}.pt"

def save_model_if_possible(agent, path):
    try:
        # Most agents have q_net or policy object with state_dict
        sd = {}
        if hasattr(agent, "q_net"):
            sd["q_net"] = agent.q_net.state_dict()
        if hasattr(agent, "target_net"):
            sd["target_net"] = agent.target_net.state_dict()
        if hasattr(agent, "policy"):
            sd["policy"] = agent.policy.state_dict()
        torch_save = False
        if sd:
            import torch
            torch.save(sd, path)
            return True
    except Exception as e:
        print("Warning saving model:", e)
    return False

def load_model_if_exists(agent, path):
    try:
        if not path.exists():
            return False
        import torch
        sd = torch.load(path, map_location="cpu")
        if hasattr(agent, "q_net") and "q_net" in sd:
            agent.q_net.load_state_dict(sd["q_net"])
        if hasattr(agent, "target_net") and "target_net" in sd:
            agent.target_net.load_state_dict(sd["target_net"])
        if hasattr(agent, "policy") and "policy" in sd:
            agent.policy.load_state_dict(sd["policy"])
        return True
    except Exception as e:
        print("Warning loading model:", e)
        return False

# -----------------------------
# Metric extraction (uses real env attributes)
# -----------------------------
def run_single_eval(agent, env, eval_episodes=EVAL_EPISODES, max_steps=MAX_STEPS):
    """
    Runs eval_episodes episodes, returns arrays of metrics per episode.
    Uses only real environment attributes:
     - Makespan: env.base_env.current_time
     - MachineUtilization: mean(machine_utilization / current_time)
     - TotalTardiness: env.base_env.total_tardiness
     - SetupTime: sum(env.base_env.reconfig_cost)
     - Reward: summed reward in episode
    """
    makespans = []
    utils = []
    tardiness = []
    setups = []
    rewards = []

    for ep in range(eval_episodes):
        s = env.reset()
        done = False
        total_r = 0.0
        steps = 0
        while not done and steps < max_steps:
            a = agent.act(s, train=False)
            s, r, done, _ = env.step(a)
            total_r += r
            steps += 1

        base = env.base_env
        ms = float(base.current_time)
        mu = float(np.mean(base.machine_utilization / (base.current_time + 1e-12)))
        tt = float(base.total_tardiness)
        st = float(np.sum(base.reconfig_cost))

        makespans.append(ms)
        utils.append(mu)
        tardiness.append(tt)
        setups.append(st)
        rewards.append(total_r)

    return {
        "Makespan": np.array(makespans),
        "MachineUtilization": np.array(utils),
        "TotalTardiness": np.array(tardiness),
        "SetupTime": np.array(setups),
        "Reward": np.array(rewards)
    }

# -----------------------------
# TOPSIS implementation
# -----------------------------
def topsis_score(df_metrics, weights):
    # df_metrics: DataFrame with columns in order: [Makespan, MachineUtilization, TotalTardiness, SetupTime, Reward]
    # For TOPSIS we normalize and decide which are benefit/cost
    metrics = df_metrics.copy().values.astype(float)
    # normalize by Euclidean norm
    norm_den = np.sqrt((metrics ** 2).sum(axis=0))
    norm = metrics / (norm_den + 1e-12)
    # Decide directions: cost metrics (lower better): Makespan, TotalTardiness, SetupTime
    # benefit metrics (higher better): MachineUtilization, Reward
    # convert cost metrics by multiplying by -1 (so bigger is better)
    direction = np.array([-1.0, 1.0, -1.0, -1.0, 1.0])  # same order as cols
    norm = norm * direction
    # apply weights
    w = weights / (weights.sum() + 1e-12)
    weighted = norm * w
    ideal_best = weighted.max(axis=0)
    ideal_worst = weighted.min(axis=0)
    dist_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))
    score = dist_worst / (dist_best + dist_worst + 1e-12)
    return score

# -----------------------------
# Main evaluation loop
# -----------------------------
def evaluate_all():
    results_raw = defaultdict(lambda: defaultdict(list))
    # We'll keep per-instance matched arrays for paired testing
    matched_metrics = defaultdict(lambda: defaultdict(list))
    instance_records = []  # for CSV table

    for nj in N_JOBS_LIST:
        for nm in N_MACHINES_LIST:
            for seed in SEEDS:
                # Create matched environment for this instance (seed used for env randomness)
                env = EnhancedRMSWrapper(n_jobs=nj, n_machines=nm, seed=seed)
                s_dim = int(np.prod(env.observation_space.shape))
                a_dim = env.action_space.n

                for agent_name, factory in AGENTS.items():
                    print(f"Instance (jobs={nj},machines={nm},seed={seed}) -> Agent: {agent_name}")
                    agent = factory(s_dim, a_dim)

                    # Try to load model first
                    mpath = model_path(agent_name, seed, nj, nm)
                    loaded = load_model_if_exists(agent, mpath)
                    if not loaded and TRAIN_IF_MISSING:
                        # Train briefly (user can increase TRAIN_EPISODES for publication-level runs)
                        print(f"  No saved model found. Training {agent_name} for {TRAIN_EPISODES} episodes (quick run).")
                        train_agent_simple(agent, EnhancedRMSWrapper(n_jobs=nj, n_machines=nm, seed=seed), episodes=TRAIN_EPISODES, max_steps=nj*5)
                        if SAVE_MODELS:
                            try:
                                save_model_if_possible(agent, mpath)
                            except Exception:
                                pass

                    metrics = run_single_eval(agent, EnhancedRMSWrapper(n_jobs=nj, n_machines=nm, seed=seed), eval_episodes=EVAL_EPISODES, max_steps=MAX_STEPS)
                    # store aggregated statistics for this instance
                    agg = {k: float(np.mean(v)) for k, v in metrics.items()}
                    agg.update({"Agent": agent_name, "NumJobs": nj, "NumMachines": nm, "Seed": seed})
                    instance_records.append(agg)

                    # store raw arrays per metric for paired testing
                    for metric_name, arr in metrics.items():
                        matched_metrics[metric_name][agent_name].append(arr)  # list of arrays matched by (nj,nm,seed) order

    # Build DataFrame of instance-level aggregated metrics
    df_inst = pd.DataFrame(instance_records)
    csv_path = OUT_DIR / "instance_metrics.csv"
    df_inst.to_csv(csv_path, index=False)
    print("Saved instance-level metrics to:", csv_path)

    # Aggregate per-agent across instances
    agents = list(AGENTS.keys())
    metric_cols = ['Makespan','MachineUtilization','TotalTardiness','SetupTime','Reward']

    # Build per-agent mean metrics matrix for TOPSIS (rows=agents, cols=metrics)
    agent_metric_means = []
    for a in agents:
        rows = df_inst[df_inst['Agent'] == a]
        means = [rows[c].mean() for c in metric_cols]
        agent_metric_means.append(means)
    df_mean = pd.DataFrame(agent_metric_means, index=agents, columns=metric_cols)

    # Compute TOPSIS scores
    scores = topsis_score(df_mean, TOPSIS_WEIGHTS)
    df_mean['TOPSIS'] = scores
    df_mean = df_mean.sort_values('TOPSIS', ascending=False)
    df_mean.to_csv(OUT_DIR / "agent_mean_metrics_topsis.csv")
    print("Saved aggregated agent metrics + TOPSIS to:", OUT_DIR / "agent_mean_metrics_topsis.csv")

    # --------------------------
    # Paired t-tests (best vs others) per metric
    # Use matched instance arrays (flattened across instances, but matched by order)
    # --------------------------
    best_agent = df_mean.index[0]
    pvals = pd.DataFrame(index=agents, columns=metric_cols)
    tstats = pd.DataFrame(index=agents, columns=metric_cols)
    for metric in metric_cols:
        # Build list of per-instance arrays for the best agent concatenated in same order
        best_arrays = matched_metrics[metric][best_agent]
        # Flatten preserving instance alignment
        best_flat = np.concatenate(best_arrays)
        for a in agents:
            if a == best_agent:
                pvals.loc[a, metric] = np.nan
                tstats.loc[a, metric] = np.nan
                continue
            other_arrays = matched_metrics[metric][a]
            other_flat = np.concatenate(other_arrays)
            # Ensure same length; if mismatch, truncate to min length
            L = min(len(best_flat), len(other_flat))
            if L == 0:
                p, t = np.nan, np.nan
            else:
                try:
                    t_stat, p_val = ttest_rel(best_flat[:L], other_flat[:L], nan_policy='omit')
                    p, t = p_val, t_stat
                except Exception:
                    p, t = np.nan, np.nan
            pvals.loc[a, metric] = p
            tstats.loc[a, metric] = t

    pvals.to_csv(OUT_DIR / "paired_ttest_pvalues.csv")
    tstats.to_csv(OUT_DIR / "paired_ttest_tstats.csv")
    print("Saved paired t-test results to:", OUT_DIR)

    # --------------------------
    # Plots
    # --------------------------
    sns.set(style="whitegrid")
    # 1) TOPSIS bar
    plt.figure(figsize=(8,5))
    sns.barplot(x=df_mean['TOPSIS'].values, y=df_mean.index.values, palette="viridis")
    plt.title("TOPSIS Score (higher = better)")
    plt.xlabel("TOPSIS score")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "topsis_scores.png", dpi=300)
    plt.close()

    # 2) Boxplots per metric across agents (use instance-level arrays)
    for metric in metric_cols:
        plt.figure(figsize=(8,5))
        data_to_plot = []
        labels = []
        for a in agents:
            lists = matched_metrics[metric][a]
            if len(lists) == 0:
                data = np.array([])
            else:
                data = np.concatenate(lists)
            data_to_plot.append(data)
            labels.append(a)
        # remove empty
        sns.boxplot(data=data_to_plot)
        plt.xticks(range(len(labels)), labels, rotation=45)
        plt.title(f"Distribution of {metric} across instances")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"boxplot_{metric}.png", dpi=300)
        plt.close()

    # 3) p-value heatmap (best vs others)
    plt.figure(figsize=(10, 4))
    pv = pvals.astype(float)
    pv = pv.reindex(agents)  # ensure consistent order
    sns.heatmap(pv, annot=True, fmt=".3f", cmap="RdYlBu_r", cbar_kws={'label':'p-value'})
    plt.title(f"Paired t-test p-values (best = {best_agent})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pvalues_heatmap.png", dpi=300)
    plt.close()

    # 4) Save LaTeX table (means ± std)
    rows = []
    for a in agents:
        rows.append([a] + [f"{df_inst[df_inst['Agent']==a][c].mean():.3f} ± {df_inst[df_inst['Agent']==a][c].std():.3f}" for c in metric_cols] + [f"{df_mean.loc[a,'TOPSIS']:.4f}"])
    cols = ["Agent"] + metric_cols + ["TOPSIS"]
    df_table = pd.DataFrame(rows, columns=cols)
    df_table.to_csv(OUT_DIR / "agent_summary_table.csv", index=False)
    with open(OUT_DIR / "agent_summary_table.tex", "w") as f:
        f.write(df_table.to_latex(index=False, escape=False))

    print("Saved summary table and plots to:", OUT_DIR)
    return {
        "instance_df": df_inst,
        "agent_mean_df": df_mean,
        "pvalues": pvals,
        "tstats": tstats
    }

# Entry point
if __name__ == "__main__":
    res = evaluate_all()
    print("Done. Outputs in:", OUT_DIR)
