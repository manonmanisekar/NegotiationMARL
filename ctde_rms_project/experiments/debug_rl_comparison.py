"""
Research-ready RL algorithm comparison for RMS scheduling
-------------------------------------------------------
Save as: debug_rl_comparison_full.py
Depends on: env/environment_fixed.py (ImprovedRMS_Env)
"""

import os
import sys
sys.path.append(r"C:\Users\91944\OneDrive\Desktop\ctde_rms_project")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import random
import json
from scipy.stats import mannwhitneyu
from env.environment_fixed import ImprovedRMS_Env

Transition = namedtuple("Transition", ["s", "a", "r", "s2", "d"])

# -----------------------------
# Replay Buffer
# -----------------------------
class ReplayBuffer:
    def __init__(self, capacity=200000):
        self.buffer = deque(maxlen=capacity)

    def add(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*samples))
        return (
            np.array(batch.s),
            np.array(batch.a),
            np.array(batch.r),
            np.array(batch.s2),
            np.array(batch.d),
        )

    def __len__(self):
        return len(self.buffer)

# -----------------------------
# Environment wrapper (fixed)
# -----------------------------
class EnhancedRMSWrapper:
    def __init__(self, n_jobs=30, n_machines=4, seed=None, difficulty='medium'):
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.seed = seed
        self.difficulty = difficulty
        self.base_env = ImprovedRMS_Env(num_jobs=n_jobs, num_machines=n_machines, seed=seed)
        # Compatibility attributes expected by other scripts
        self.actual_jobs = n_jobs
        self.actual_machines = n_machines
        self.observation_space = self.base_env.observation_space
        self.action_space = self.base_env.action_space
        self.episode_step = 0
        self.max_steps = n_jobs * 3
        self.original_processing_times = self.base_env.job_processing_times.copy()
        self.original_due_dates = self.base_env.job_due_dates.copy()
        self.original_machine_times = self.base_env.machine_base_times.copy()

    def reset(self):
        self.episode_step = 0
        state = self.base_env.reset()
        # controlled variability
        noise_factors = np.random.uniform(0.8, 1.2, size=self.n_jobs)
        self.base_env.job_processing_times = self.original_processing_times * noise_factors

        if self.difficulty == 'hard':
            due_date_factor = np.random.uniform(0.7, 0.9, size=self.n_jobs)
        elif self.difficulty == 'easy':
            due_date_factor = np.random.uniform(1.1, 1.4, size=self.n_jobs)
        else:
            due_date_factor = np.random.uniform(0.9, 1.1, size=self.n_jobs)
        self.base_env.job_due_dates = self.original_due_dates * due_date_factor

        machine_noise = np.random.uniform(0.85, 1.15, size=self.n_machines)
        self.base_env.machine_base_times = self.original_machine_times * machine_noise

        reconfig_noise = np.random.uniform(0.9, 1.1, size=self.base_env.reconfig_time_matrix.shape)
        self.base_env.reconfig_time_matrix *= reconfig_noise
        np.fill_diagonal(self.base_env.reconfig_time_matrix, 0.0)

        return self.base_env._get_state()

    def step(self, action):
        self.episode_step += 1
        if action >= self.action_space.n or action < 0:
            action = int(np.clip(action, 0, self.action_space.n - 1))
        try:
            next_state, reward, done, info = self.base_env.step(int(action))
        except Exception as e:
            print(f"⚠️ Error in environment step: {e}")
            next_state = self.base_env._get_state()
            reward = -10.0
            done = True
            info = {}
        reward -= 0.02
        if self.episode_step >= self.max_steps:
            done = True
            reward -= 2.0
        return next_state, reward, done, info

# -----------------------------
# Model building blocks
# -----------------------------
class StandardDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)

# Double DQN agent (uses StandardDQN architecture)
class DoubleDQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, batch_size=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.01
        self.soft_tau = 0.001
        self.q_net = StandardDQN(state_dim, action_dim).to(self.device)
        self.target_net = StandardDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay = ReplayBuffer()
        self.update_count = 0
        self.target_update_freq = 10

    def act(self, state, train=True):
        if train and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state)
        return int(q_values.argmax().item())

    def update(self):
        if len(self.replay) < self.batch_size:
            return None
        s, a, r, s2, d = self.replay.sample(self.batch_size)
        s = torch.tensor(s, dtype=torch.float32, device=self.device)
        s2 = torch.tensor(s2, dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.long, device=self.device)
        r = torch.tensor(r, dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            next_a = self.q_net(s2).argmax(1)
            next_q = self.target_net(s2).gather(1, next_a.unsqueeze(1)).squeeze(1)
            target = r + (1 - d) * self.gamma * next_q
        q_values = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            for tp, p in zip(self.target_net.parameters(), self.q_net.parameters()):
                tp.data.copy_(tp.data * (1.0 - self.soft_tau) + p.data * self.soft_tau)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss.item()

# -----------------------------
# SAC (discrete action adaptation)
# -----------------------------
class SACPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

class SACQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)

class SACAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, batch_size=128, alpha=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.alpha = alpha
        self.soft_tau = 0.005
        self.policy = SACPolicyNetwork(state_dim, action_dim).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.q1 = SACQNetwork(state_dim, action_dim).to(self.device)
        self.q2 = SACQNetwork(state_dim, action_dim).to(self.device)
        self.q1_target = SACQNetwork(state_dim, action_dim).to(self.device)
        self.q2_target = SACQNetwork(state_dim, action_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)
        self.replay = ReplayBuffer()

    def act(self, state, train=True):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.policy(state)
        if train:
            action = torch.multinomial(action_probs, 1).item()
        else:
            action = action_probs.argmax().item()
        return action

    def update(self):
        if len(self.replay) < self.batch_size:
            return None
        s, a, r, s2, d = self.replay.sample(self.batch_size)
        s = torch.tensor(s, dtype=torch.float32, device=self.device)
        s2 = torch.tensor(s2, dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.long, device=self.device)
        r = torch.tensor(r, dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            next_action_probs = self.policy(s2)
            next_log_probs = torch.log(next_action_probs + 1e-8)
            next_q1 = self.q1_target(s2)
            next_q2 = self.q2_target(s2)
            next_q = torch.min(next_q1, next_q2)
            next_v = (next_action_probs * (next_q - self.alpha * next_log_probs)).sum(dim=1)
            target_q = r + (1 - d) * self.gamma * next_v
        current_q1 = self.q1(s).gather(1, a.unsqueeze(1)).squeeze(1)
        current_q2 = self.q2(s).gather(1, a.unsqueeze(1)).squeeze(1)
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        self.q1_optimizer.zero_grad(); q1_loss.backward(); self.q1_optimizer.step()
        self.q2_optimizer.zero_grad(); q2_loss.backward(); self.q2_optimizer.step()
        action_probs = self.policy(s)
        log_probs = torch.log(action_probs + 1e-8)
        q1_values = self.q1(s)
        q2_values = self.q2(s)
        min_q = torch.min(q1_values, q2_values)
        policy_loss = (action_probs * (self.alpha * log_probs - min_q)).sum(dim=1).mean()
        self.policy_optimizer.zero_grad(); policy_loss.backward(); self.policy_optimizer.step()
        for tp, p in zip(self.q1_target.parameters(), self.q1.parameters()):
            tp.data.copy_(tp.data * (1.0 - self.soft_tau) + p.data * self.soft_tau)
        for tp, p in zip(self.q2_target.parameters(), self.q2.parameters()):
            tp.data.copy_(tp.data * (1.0 - self.soft_tau) + p.data * self.soft_tau)
        return (q1_loss.item() + q2_loss.item()) / 2

# -----------------------------
# Dueling-Attention DQN
# -----------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        batch_size = x.size(0)
        q = self.query(x).view(batch_size, self.num_heads, self.head_dim)
        k = self.key(x).view(batch_size, self.num_heads, self.head_dim)
        v = self.value(x).view(batch_size, self.num_heads, self.head_dim)
        scores = torch.bmm(q, k.transpose(1, 2)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.bmm(attn_weights, v)
        attn_out = attn_out.view(batch_size, self.hidden_dim)
        out = self.out(attn_out)
        out = self.layer_norm(out + attn_out)
        return out

class DuelingAttentionDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(0.1)
        )
        self.attention = MultiHeadAttention(hidden_dim, hidden_dim, num_heads=4)
        self.proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.ReLU(), nn.Linear(hidden_dim // 4, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.ReLU(), nn.Linear(hidden_dim // 4, action_dim)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.feature_net(x)
        attn_out = self.attention(features)
        enhanced = self.proj(attn_out) + features
        value = self.value_stream(enhanced)
        adv = self.advantage_stream(enhanced)
        q = value + adv - adv.mean(dim=1, keepdim=True)
        return q

class DuelingAttentionDQNAgent:
    def __init__(self, state_dim, action_dim, lr=5e-4, gamma=0.99, batch_size=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1.0
        self.epsilon_decay = 0.9997
        self.epsilon_min = 0.01
        self.soft_tau = 0.001
        self.q_net = DuelingAttentionDQN(state_dim, action_dim).to(self.device)
        self.target_net = DuelingAttentionDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=lr, weight_decay=1e-5)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay = ReplayBuffer()
        self.update_count = 0
        self.target_update_freq = 10

    def act(self, state, train=True):
        if train and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state)
        return int(q_values.argmax().item())

    def update(self):
        if len(self.replay) < self.batch_size:
            return None
        s, a, r, s2, d = self.replay.sample(self.batch_size)
        s = torch.tensor(s, dtype=torch.float32, device=self.device)
        s2 = torch.tensor(s2, dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.long, device=self.device)
        r = torch.tensor(r, dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            next_a = self.q_net(s2).argmax(1)
            next_q = self.target_net(s2).gather(1, next_a.unsqueeze(1)).squeeze(1)
            target = r + (1 - d) * self.gamma * next_q
        q_values = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0); self.optimizer.step()
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            for tp, p in zip(self.target_net.parameters(), self.q_net.parameters()):
                tp.data.copy_(tp.data * (1.0 - self.soft_tau) + p.data * self.soft_tau)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss.item()

# -----------------------------
# Training & evaluation utilities
# -----------------------------
def train_agent_simple(agent, env, episodes=200, max_steps=200):
    rewards = []
    losses = []
    for ep in range(episodes):
        s = env.reset()
        total_r = 0.0
        done = False
        step = 0
        while not done and step < max_steps:
            a = agent.act(s, train=True)
            s2, r, done, _ = env.step(a)
            agent.replay.add(s, a, r, s2, done)
            loss = agent.update()
            if loss is not None:
                losses.append(loss)
            s = s2
            total_r += r
            step += 1
        rewards.append(total_r)
        if (ep+1) % max(1, episodes//10) == 0:
            print(f"[{agent.__class__.__name__}] Ep {ep+1}/{episodes} avg(recent 10): {np.mean(rewards[-10:]):.3f}")
    return rewards, losses

def evaluate_agent(agent, env, episodes=50, max_steps=200):
    eval_rewards = []
    action_counts = np.zeros(agent.action_dim)
    for _ in range(episodes):
        s = env.reset()
        total = 0.0
        done = False
        step = 0
        while not done and step < max_steps:
            a = agent.act(s, train=False)
            action_counts[a] += 1
            s, r, done, _ = env.step(a)
            total += r
            step += 1
        eval_rewards.append(total)
    dist = action_counts / (action_counts.sum() + 1e-10)
    entropy = -np.sum(dist * np.log(dist + 1e-10))
    return eval_rewards, entropy

# -----------------------------
# Comparison runner and plotting
# -----------------------------
def compare_algorithms(episodes=500, seeds=3, episode_length=40):
    os.makedirs('results', exist_ok=True)
    print("Initializing test environment...")
    test_env = EnhancedRMSWrapper(n_jobs=episode_length, n_machines=4, seed=42)
    state_dim = int(np.prod(test_env.observation_space.shape))
    action_dim = test_env.action_space.n
    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    factories = {
        'Double DQN': lambda: DoubleDQNAgent(state_dim, action_dim, lr=1e-3, batch_size=64),
        'SAC': lambda: SACAgent(state_dim, action_dim, lr=3e-4, batch_size=128, alpha=0.2),
        'Dueling-Attention DQN': lambda: DuelingAttentionDQNAgent(state_dim, action_dim, lr=5e-4, batch_size=64),
    }

    results = {name: {'rewards': [], 'eval_rewards': [], 'entropies': []} for name in factories}

    for seed in range(seeds):
        print(f"\n=== SEED {seed+1}/{seeds} ===")
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        for name, factory in factories.items():
            print(f"\nTraining {name}...")
            env_train = EnhancedRMSWrapper(n_jobs=episode_length, n_machines=4, seed=seed*100 + hash(name) % 1000)
            agent = factory()
            rewards, losses = train_agent_simple(agent, env_train, episodes=episodes, max_steps=episode_length*2)
            results[name]['rewards'].append(rewards)
            print(f"Evaluating {name}...")
            env_eval = EnhancedRMSWrapper(n_jobs=episode_length, n_machines=4, seed=seed*1000 + hash(name) % 10000)
            eval_rewards, entropy = evaluate_agent(agent, env_eval, episodes=100, max_steps=episode_length*2)
            results[name]['eval_rewards'].append(eval_rewards)
            results[name]['entropies'].append(entropy)
            print(f" {name} eval mean: {np.mean(eval_rewards):.3f} std: {np.std(eval_rewards):.3f} entropy: {entropy:.3f}")

    # Save raw results
    with open('results/alg_comparison_results.json', 'w') as f:
        json.dump({k: v for k, v in results.items()}, f, default=lambda x: np.array(x).tolist())

    # Plotting
    plot_results(results, episodes)
    return results

def plot_results(results, episodes):
    # Learning curves (smoothed)
    plt.figure(figsize=(10, 6))
    for name, data in results.items():
        all_rewards = np.array(data['rewards'])
        mean_rewards = np.mean(all_rewards, axis=0)
        window = max(1, episodes // 20)
        if len(mean_rewards) >= window:
            smoothed = np.convolve(mean_rewards, np.ones(window)/window, mode='valid')
            plt.plot(range(len(smoothed)), smoothed, label=name)
    plt.xlabel('Episode (smoothed)')
    plt.ylabel('Reward')
    plt.title('Training Learning Curves (Smoothed)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/learning_curves.png', dpi=300)
    plt.close()

    # Violin plot for evaluation rewards
    plt.figure(figsize=(10, 6))
    evals = [np.concatenate(results[name]['eval_rewards']) for name in results]
    plt.violinplot(evals, showmeans=True)
    plt.xticks(range(1, len(results)+1), list(results.keys()))
    plt.title('Evaluation Reward Distributions')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig('results/eval_violin.png', dpi=300)
    plt.close()

    # Bar plot mean +- std
    means = [np.mean(np.concatenate(results[name]['eval_rewards'])) for name in results]
    stds = [np.std(np.concatenate(results[name]['eval_rewards'])) for name in results]
    plt.figure(figsize=(8,6))
    x = np.arange(len(results))
    plt.bar(x, means, yerr=stds, capsize=8)
    plt.xticks(x, list(results.keys()))
    plt.ylabel('Mean Eval Reward')
    plt.title('Mean Evaluation Reward ± Std')
    plt.grid(axis='y')
    plt.savefig('results/eval_mean_std.png', dpi=300)
    plt.close()

    print('Plots saved to results/')

# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    # Shorter defaults for quick runs — increase episodes/seeds for publication-level experiments
    results = compare_algorithms(episodes=200, seeds=2, episode_length=40)
    print('Done')
