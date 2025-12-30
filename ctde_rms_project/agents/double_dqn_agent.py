from agents.dqn_agent import DQNAgent, QNetwork
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DoubleDQNAgent(DQNAgent):
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

        q_values = self.q_net(s).gather(1, a)
        next_actions = self.q_net(s_next).argmax(1).unsqueeze(1)
        next_q = self.target_net(s_next).gather(1, next_actions)
        target_q = r + self.gamma * next_q * (1 - done)

        loss = self.loss_fn(q_values, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
