import torch, torch.nn as nn, torch.optim as optim
import numpy as np, random
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state','action','reward','next_state','done'))

class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)
    def push(self,*args): self.buffer.append(Transition(*args))
    def sample(self,batch): samples = random.sample(self.buffer,batch); return Transition(*zip(*samples))
    def __len__(self): return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, in_dim, out_dim, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, out_dim)
        )
    def forward(self,x): return self.net(x)

class CentralDQN:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy = DQN(state_dim, action_dim).to(self.device)
        self.target = DQN(state_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.opt = optim.Adam(self.policy.parameters(), lr=lr)
        self.replay = ReplayBuffer()
        self.gamma = gamma
        self.eps = 1.0
        self.eps_min = 0.05
        self.eps_decay = 0.995
        self.steps = 0

    def act(self, state, legal_mask):
        if random.random() < self.eps:
            legal_idx = [i for i,ok in enumerate(legal_mask) if ok]
            if legal_idx:
                return random.choice(legal_idx)
            return random.randrange(len(legal_mask))
        s = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.policy(s).cpu().numpy().flatten()
        q = q - (~np.array(legal_mask)).astype(float)*1e6
        return int(q.argmax())

    def store(self, *args):
        self.replay.push(*args)

    def update(self, batch=64):
        if len(self.replay) < batch:
            return
        trans = self.replay.sample(batch)
        import torch, numpy as np, torch.nn as nn
        s = torch.FloatTensor(np.stack(trans.state)).to(self.device)
        a = torch.LongTensor(trans.action).to(self.device)
        r = torch.FloatTensor(trans.reward).to(self.device)
        ns = torch.FloatTensor(np.stack(trans.next_state)).to(self.device)
        d = torch.FloatTensor(trans.done).to(self.device)

        qvals = self.policy(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target(ns).max(1)[0]
            target = r + self.gamma * next_q * (1.0 - d)

        loss = nn.MSELoss()(qvals, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        if self.eps > self.eps_min:
            self.eps *= self.eps_decay
        self.steps += 1
        if self.steps % 200 == 0:
            self.target.load_state_dict(self.policy.state_dict())