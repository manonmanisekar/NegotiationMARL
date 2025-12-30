"""
dueling_attention_dqn.py
------------------------
Enhanced DQN architecture with:
 - Nonlinear embedding layers
 - Context-aware attention
 - Dueling Value/Advantage streams
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionModule(nn.Module):
    """Simple context-aware attention mechanism."""
    def __init__(self, embed_dim):
        super().__init__()
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5

    def forward(self, x):
        # x: [batch, features]
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        attn_scores = torch.matmul(q, k.T) / self.scale  # [batch, batch]
        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        return context + x  # residual


class DuelingAttentionDQN(nn.Module):
    """
    Dueling + Attention Enhanced Q-network
    Input: state (job + machine features)
    Output: Q(s,a)
    """
    def __init__(self, state_dim, action_dim, embed_dim=128):
        super().__init__()

        # Nonlinear embedding
        self.fc1 = nn.Linear(state_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)

        # Context-aware attention
        self.attn = AttentionModule(embed_dim)

        # Dueling heads
        self.value_stream = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

        self.adv_stream = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, action_dim)
        )

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.attn(x)

        value = self.value_stream(x)
        adv = self.adv_stream(x)
        q = value + adv - adv.mean(dim=1, keepdim=True)
        return q


class DuelingAttentionAgent:
    """Wrapper with predict method for trainer compatibility."""
    def __init__(self, state_dim, action_dim, device="cpu"):
        self.device = device
        self.q_network = DuelingAttentionDQN(state_dim, action_dim).to(device)

    def predict(self, state):
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_vals = self.q_network(state_t)
            return torch.argmax(q_vals).item()
