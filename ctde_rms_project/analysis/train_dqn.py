import os
import sys
sys.path.append(r"C:\Users\91944\OneDrive\Desktop\ctde_rms_project")

from env.environment import ImprovedRMS_Env
from agents.dqn_agent import DQNAgent
import pandas as pd
import numpy as np

env = ImprovedRMS_Env()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQNAgent(state_dim, action_dim)
episodes = 10
rewards, losses = [], []

for ep in range(episodes):
    s = env.reset()
    total_r, total_l = 0, 0
    done = False
    while not done:
        a = agent.select_action(s)
        s_next, r, done, _ = env.step(a)
        agent.store_transition(s, a, r, s_next, done)
        loss = agent.train_step()
        agent.decay_epsilon()
        s = s_next
        total_r += r
        if loss:
            total_l += loss
    rewards.append(total_r)
    losses.append(total_l)
    print(f"Ep {ep+1}/{episodes} | Reward={total_r:.3f} | Loss={total_l:.4f} | ε={agent.epsilon:.3f}")

pd.DataFrame({"Reward": rewards, "Loss": losses}).to_csv("results/DQN_rewards.csv", index=False)
print("✅ Saved DQN results to results/DQN_rewards.csv")
