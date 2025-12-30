
import os
import sys
sys.path.append(r"C:\Users\91944\OneDrive\Desktop\ctde_rms_project")

import os
import numpy as np
import pandas as pd
from env.environment import ImprovedRMS_Env
from agents.sac_agent import SACAgent

# Create results folder
os.makedirs("results", exist_ok=True)

# Initialize environment
env = ImprovedRMS_Env()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Initialize SAC agent
agent = SACAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    lr=3e-4,
    gamma=0.99,
    alpha=0.2
)

episodes = 500
rewards, critic_losses, actor_losses = [], [], []

print("\n============================================================")
print("Training Soft Actor-Critic (SAC) on ImprovedRMS_Env")
print("============================================================")

for ep in range(episodes):
    state = env.reset()
    done = False
    total_reward, total_critic_loss, total_actor_loss = 0.0, 0.0, 0.0

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        loss_tuple = agent.train_step()
        state = next_state
        total_reward += reward
        if loss_tuple:
            critic_loss, actor_loss = loss_tuple
            total_critic_loss += critic_loss
            total_actor_loss += actor_loss

    rewards.append(total_reward)
    critic_losses.append(total_critic_loss)
    actor_losses.append(total_actor_loss)

    if (ep + 1) % 50 == 0:
        print(f"Episode {ep+1:03d}/{episodes} | Reward: {total_reward:.3f} | Critic Loss: {total_critic_loss:.5f} | Actor Loss: {total_actor_loss:.5f}")

# Save metrics to CSV
results_path = os.path.join("results", "SAC_rewards.csv")
pd.DataFrame({
    "Reward": rewards,
    "Critic_Loss": critic_losses,
    "Actor_Loss": actor_losses
}).to_csv(results_path, index=False)
print(f"\n✅ SAC Training complete. Results saved to {results_path}")
