"""
evaluate_performance_metrics.py
--------------------------------
Evaluates trained RL scheduling agents using multiple operational KPIs:
Makespan, Machine Utilization, Total Tardiness, and Setup Time.
Generates research-ready performance comparison plots and rankings.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import rankdata, ttest_ind
from debug_rl_comparison import EnhancedRMSWrapper, DoubleDQNAgent, SACAgent, DuelingAttentionDQNAgent


# ==============================================================
# Evaluate each agent using the real environment metrics
# ==============================================================
def evaluate_agent_metrics(agent, env, episodes=30, max_steps=200):
    metrics = {
        'Makespan': [],
        'MachineUtilization': [],
        'TotalTardiness': [],
        'SetupTime': [],
        'Reward': []
    }

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done and steps < max_steps:
            action = agent.act(state, train=False)
            next_state, reward, done, info = env.step(action)
            state = next_state
            total_reward += reward
            steps += 1

        base = env.base_env

        # --- Compute metrics directly from environment ---
        makespan = float(base.current_time)
        mu = float(np.mean(base.machine_utilization / (base.current_time + 1e-8)))  # normalized utilization
        tt = float(base.total_tardiness)
        st = float(np.sum(base.reconfig_cost))

        metrics['Makespan'].append(makespan)
        metrics['MachineUtilization'].append(mu)
        metrics['TotalTardiness'].append(tt)
        metrics['SetupTime'].append(st)
        metrics['Reward'].append(total_reward)

    return metrics


# ==============================================================
# Composite Ranking and Visualization
# ==============================================================
def performance_ranking(episodes=500):
    os.makedirs('results', exist_ok=True)
    env = EnhancedRMSWrapper(n_jobs=40, n_machines=4, seed=42)
    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = env.action_space.n

    agents = {
        'Double DQN': DoubleDQNAgent(state_dim, action_dim),
        'SAC': SACAgent(state_dim, action_dim),
        'Dueling-Attention DQN': DuelingAttentionDQNAgent(state_dim, action_dim)
    }

    all_results = {}
    for name, agent in agents.items():
        print(f"\nEvaluating true performance metrics for {name}...")
        metrics = evaluate_agent_metrics(agent, env, episodes)
        all_results[name] = {k: np.mean(v) for k, v in metrics.items()}

    df = pd.DataFrame(all_results).T

    # Composite score: lower = better (except utilization and reward)
    df['CompositeScore'] = (
        rankdata(df['Makespan']) +
        rankdata(-df['MachineUtilization']) +
        rankdata(df['TotalTardiness']) +
        rankdata(df['SetupTime']) +
        rankdata(-df['Reward'])
    )
    df['Rank'] = rankdata(df['CompositeScore'])
    df = df.sort_values('Rank')

    print("\n=== Real Multi-Metric Performance Ranking ===")
    print(df.round(4))
    df.to_csv('results/performance_ranking_real.csv')

    # ==============================================================
    # Visualization: Radar + Bar
    # ==============================================================
    metrics_to_plot = ['Makespan','MachineUtilization','TotalTardiness','SetupTime','Reward']
    labels = metrics_to_plot
    num_vars = len(labels)
    angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # --- Radar Plot ---
    plt.figure(figsize=(8, 8))
    for name in df.index:
        values = [df.loc[name, m] for m in metrics_to_plot]
        # Normalize & invert lower-better metrics
        normalized = []
        for metric, val in zip(metrics_to_plot, values):
            if metric in ['Makespan', 'TotalTardiness', 'SetupTime']:
                val = -val
            normalized.append(val)
        normalized = np.array(normalized)
        normalized = (normalized - np.min(normalized)) / (np.max(normalized) - np.min(normalized) + 1e-8)
        normalized = np.concatenate((normalized, [normalized[0]]))
        plt.polar(angles, normalized, label=name, linewidth=2)

    plt.xticks(angles[:-1], labels, fontsize=11)
    plt.title('True Multi-Metric Performance Radar Chart', size=14, weight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig('results/multi_metric_radar_real.png', dpi=300)
    plt.close()

    # --- Composite Ranking Bar ---
    plt.figure(figsize=(8,6))
    plt.barh(df.index, df['CompositeScore'], color='darkcyan', alpha=0.8)
    plt.xlabel('Composite Score (lower = better)')
    plt.title('Overall Real Multi-Metric Ranking of Agents')
    plt.tight_layout()
    plt.savefig('results/composite_ranking_real.png', dpi=300)
    plt.close()

    print("\n✅ Saved all real performance plots and ranking CSV to results/")

if __name__ == "__main__":
    performance_ranking(episodes=500)
