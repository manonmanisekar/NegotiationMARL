import os
import sys
sys.path.append(r"C:\Users\91944\OneDrive\Desktop\ctde_rms_project")

import os
import glob
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque, namedtuple
import random
from env.environment import ImprovedRMS_Env

sns.set(style="whitegrid")

# ==========================================================
#  Attention + Dueling DQN Network
# ==========================================================
class AttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        q = self.query(x).unsqueeze(1)
        k = self.key(x).unsqueeze(1)
        v = self.value(x).unsqueeze(1)
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(self.hidden_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        out = torch.bmm(attn_weights, v)
        return out.squeeze(1)


class DuelingAttentionDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.attn = AttentionLayer(hidden_dim, hidden_dim // 2)
        self.proj = nn.Linear(hidden_dim // 2, hidden_dim)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        attn_out = self.attn(x)
        attn_proj = self.proj(attn_out)
        x = x + attn_proj
        value = self.value_stream(x)
        adv = self.adv_stream(x)
        q = value + adv - adv.mean(dim=1, keepdim=True)
        return q


# ==========================================================
#  Replay Buffer
# ==========================================================
Transition = namedtuple("Transition", ["s", "a", "r", "s2", "d"])


class ReplayBuffer:
    def __init__(self, capacity=50000):
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


# ==========================================================
#  Dueling Attention DQN Agent
# ==========================================================
class EnhancedDQNAgent:
    def __init__(self, state_dim, action_dim,
                 lr=5e-5, gamma=0.98, hidden_dim=128,
                 buffer_capacity=50000, batch_size=32,
                 epsilon_decay=0.995, soft_tau=0.005):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay = epsilon_decay
        self.soft_tau = soft_tau

        self.q_net = DuelingAttentionDQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = DuelingAttentionDQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.replay = ReplayBuffer(capacity=buffer_capacity)

    def act(self, state):
        if np.random.rand() < self.epsilon:
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

        # Soft update
        for tp, p in zip(self.target_net.parameters(), self.q_net.parameters()):
            tp.data.copy_(tp.data * (1.0 - self.soft_tau) + p.data * self.soft_tau)

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return loss.item()


# ==========================================================
#  Load Best Config from Tuning
# ==========================================================
def load_best_config_from_tuning():
    csv_files = glob.glob("results_tuning/lr_*.csv")
    if not csv_files:
        raise FileNotFoundError("❌ No tuning CSV files found. Run hyperparameter_tuning.py first.")

    best_lr, best_avg_reward = None, -float("inf")
    for path in csv_files:
        df = pd.read_csv(path)
        avg_reward = df["Reward"].mean()
        lr_str = os.path.basename(path).replace("lr_", "").replace(".csv", "")
        lr_value = float(lr_str)
        print(f"Checked {path} → Avg Reward={avg_reward:.4f}")
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            best_lr = lr_value

    print(f"\n✅ Best learning rate found: {best_lr} (Avg Reward={best_avg_reward:.4f})")

    return {
        "lr": best_lr,
        "gamma": 0.99,
        "hidden_dim": 128,
        "batch_size": 64,
        "epsilon_decay": 0.995,
        "soft_tau": 0.005
    }


# ==========================================================
#  Training Function
# ==========================================================
def train_with_best_config(config, episodes=1000):
    print("\n" + "=" * 60)
    print("Training with Best Configuration")
    print("=" * 60)

    env = ImprovedRMS_Env(num_jobs=5, num_machines=4)
    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = env.action_space.n

    agent = EnhancedDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=config['lr'],
        gamma=config['gamma'],
        hidden_dim=config['hidden_dim'],
        batch_size=config['batch_size'],
        epsilon_decay=config['epsilon_decay'],
        soft_tau=config['soft_tau']
    )

    rewards, losses, epsilons = [], [], []

    for ep in range(episodes):
        s = env.reset()
        total_reward, total_loss = 0, 0
        done = False
        step_count = 0
        max_steps = 1000
        loss_count = 0

        while not done and step_count < max_steps:
            a = agent.act(s)
            s2, r, done, _ = env.step(a)
            agent.replay.add(s, a, r, s2, done)
            loss = agent.update()
            s = s2
            total_reward += r
            if loss:
                total_loss += loss
                loss_count += 1
            step_count += 1

        avg_loss = total_loss / max(loss_count, 1)
        rewards.append(total_reward)
        losses.append(avg_loss)
        epsilons.append(agent.epsilon)

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1:03d}/{episodes} | Reward: {total_reward:8.3f} | Loss: {avg_loss:8.4f} | ε={agent.epsilon:.3f}")

    # ======================================================
    # Publication-Quality Plotting (No Reward Smoothing)
    # ======================================================
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size": 10,
        "font.family": "serif",
        "axes.labelweight": "bold",
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "axes.linewidth": 0.8,
        "legend.fontsize": 9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "savefig.dpi": 300
    })

    os.makedirs("results", exist_ok=True)
    log_path = "results/training_metrics.csv"
    pd.DataFrame({
        "Episode": np.arange(1, len(rewards) + 1),
        "Reward": rewards,
        "Loss": losses,
        "Epsilon": epsilons
    }).to_csv(log_path, index=False)
    print(f"\n📁 Training metrics saved to: {log_path}")

    # Moving average for loss only
    def moving_average(x, window=20):
        return np.convolve(x, np.ones(window) / window, mode='valid')

    ma_losses = moving_average(losses, 20)
    ma_epsilons = moving_average(epsilons, 20)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    plt.subplots_adjust(wspace=0.3, top=0.85)
    fig.suptitle("Training Performance Metrics", fontsize=13, fontweight="bold")

    # (a) Reward Trend - RAW
    axes[0].plot(rewards, color="tab:blue", linewidth=1.5)
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].set_title("(a) Episode Reward Trend", fontweight="bold", loc="left")
    axes[0].grid(alpha=0.3)

    # (b) Loss Trend - Smoothed
    axes[1].plot(losses, color="tab:orange", alpha=0.3, linewidth=1.0, label="Raw Loss")
    axes[1].plot(range(19, len(ma_losses) + 19), ma_losses,
                 color="tab:orange", linewidth=1.8, label="MA(20)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("(b) Training Loss Trend", fontweight="bold", loc="left")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.3)

    # (c) Epsilon Decay
    axes[2].plot(epsilons, color="tab:red", linewidth=1.5)
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Exploration Rate (ε)")
    axes[2].set_title("(c) Exploration Rate Decay", fontweight="bold", loc="left")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/training_summary_full.png", bbox_inches="tight", dpi=300)
    plt.savefig("results/training_summary_full.pdf", bbox_inches="tight")
    plt.show()
    print("📊 Saved high-quality plots to results/: training_summary_full.[png|pdf]")

    # Save Model
    model_path = "results/best_model.pth"
    torch.save(agent.q_net.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")

    return agent, rewards, losses


# ==========================================================
#  Main Entry Point
# ==========================================================
if __name__ == "__main__":
    best_config = load_best_config_from_tuning()
    print(f"\nUsing config: {best_config}")
    agent, rewards, losses = train_with_best_config(best_config, episodes=1000)

    print("\n🎉 Training complete!")
    print(f"📊 Final Results:")
    print(f"   - Average Reward: {np.mean(rewards):.3f}")
    print(f"   - Best Reward: {np.max(rewards):.3f}")
    print(f"   - Final Loss: {losses[-1]:.5f}")
    print(f"   - Final Epsilon: {agent.epsilon:.3f}")
