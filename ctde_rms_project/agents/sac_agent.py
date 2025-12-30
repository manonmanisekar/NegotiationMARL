import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

class SACAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, alpha=0.2):
        self.actor = Actor(state_dim, action_dim)
        self.critic1 = Critic(state_dim, action_dim)
        self.critic2 = Critic(state_dim, action_dim)
        self.target_critic1 = Critic(state_dim, action_dim)
        self.target_critic2 = Critic(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr)
        self.gamma = gamma
        self.alpha = alpha
        self.replay_buffer = deque(maxlen=50000)
        self.batch_size = 64
        self.update_targets()

    def update_targets(self):
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            probs = self.actor(state)
            action = torch.multinomial(probs, 1).item()
        return action

    def store_transition(self, s, a, r, s_next, done):
        self.replay_buffer.append((s, a, r, s_next, done))

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = random.sample(self.replay_buffer, self.batch_size)
        s, a, r, s_next, done = zip(*batch)
        s = torch.FloatTensor(s)
        a = torch.LongTensor(a).unsqueeze(1)
        r = torch.FloatTensor(r).unsqueeze(1)
        s_next = torch.FloatTensor(s_next)
        done = torch.FloatTensor(done).unsqueeze(1)

        a_onehot = F.one_hot(a.squeeze(), num_classes=self.actor.net[-1].out_features).float()
        q1 = self.critic1(s, a_onehot)
        q2 = self.critic2(s, a_onehot)

        with torch.no_grad():
            next_probs = self.actor(s_next)
            next_a = F.one_hot(torch.multinomial(next_probs, 1).squeeze(), num_classes=next_probs.shape[1]).float()
            target_q = torch.min(self.target_critic1(s_next, next_a), self.target_critic2(s_next, next_a))
            y = r + self.gamma * (1 - done) * (target_q - self.alpha * torch.log(next_probs + 1e-10).sum(dim=1, keepdim=True))

        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        probs = self.actor(s)
        log_probs = torch.log(probs + 1e-10)
        q_min = torch.min(self.critic1(s, probs), self.critic2(s, probs))
        actor_loss = (self.alpha * log_probs - q_min).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return (critic_loss.item(), actor_loss.item())
