import os,sys
sys.path.append(r"C:\Users\91944\OneDrive\Desktop\ctde_rms_project")

import gym, numpy as np
from gym import spaces
from env.environment import ImprovedRMS_Env
from utils.state_extraction import extract_normalized_state
class RMSGymEnv(gym.Env):
    def __init__(self, n_jobs=30, n_machines=4, seed=0):
        super().__init__()
        self.env = ImprovedRMS_Env(n_jobs=n_jobs, n_machines=n_machines, seed=seed)
        obs = extract_normalized_state(self.env.jobs[0], self.env.machines, 0.0, self.env)
        self.observation_space = spaces.Box(low=-1e9, high=1e9, shape=obs.shape, dtype=float)
        self.action_space = spaces.Discrete(n_machines)
        self.current_idx = 0
    def reset(self):
        self.env.time=0.0; self.env.stats=[]
        for j in self.env.jobs: j.completed=False
        for m in self.env.machines: m.last_family=None; m.utilization=0.0; m.reputation=1.0
        self.current_idx=0
        return extract_normalized_state(self.env.jobs[self.current_idx], self.env.machines, self.env.time, self.env)
    def step(self, action):
        job = self.env.jobs[self.current_idx]
        chosen = int(action)
        if job.process_type not in self.env.machines[chosen].capabilities:
            eligible = [i for i,m in enumerate(self.env.machines) if job.process_type in m.capabilities]
            if not eligible:
                self.env.stats.append({'job': job.id, 'setup':0,'tardiness':0,'makespan':self.env.time,'reward':-100.0,'machine':None})
                reward=-100.0; done=False
                self.current_idx+=1
                if self.current_idx>=len(self.env.jobs): done=True; return np.zeros(self.observation_space.shape), reward, done, {}
                return extract_normalized_state(self.env.jobs[self.current_idx], self.env.machines, self.env.time, self.env), reward, done, {}
            chosen = eligible[0]
        m = self.env.machines[chosen]
        setup = self.env.setup_time(m, job); ptime = job.processing_time / m.speed; total = setup+ptime
        self.env.time += total; job.completed=True; m.last_family=job.subfamily; m.utilization += job.processing_time
        tard = max(0.0, self.env.time - job.deadline)
        reward = float(self.env.reward_fn.compute(self.env, job, m, setup, tard))
        self.env.stats.append({'job': job.id, 'setup':setup, 'tardiness':tard, 'makespan':self.env.time, 'reward': reward, 'machine': m.id})
        self.current_idx += 1
        done = self.current_idx >= len(self.env.jobs)
        obs = np.zeros(self.observation_space.shape) if done else extract_normalized_state(self.env.jobs[self.current_idx], self.env.machines, self.env.time, self.env)
        return obs, reward, done, {}
