"""
environment.py
Gym-compatible environment for Reconfigurable Manufacturing Scheduling (RMS)
Used in CTDE RMS project for Enhanced DQN, PPO, and SAC training.
"""
import os
import sys
sys.path.append(r"C:\Users\91944\OneDrive\Desktop\ctde_rms_project")

import numpy as np
import random
import os
import gym
from gym import spaces
from env.reward_function import RMSRewardFunction


# ============================================================
# 🏭 Basic Machine and Job Simulation
# ============================================================
class Machine:
    def __init__(self, machine_id):
        self.id = machine_id
        self.busy_until = 0.0
        self.total_busy_time = 0.0
        self.total_setup_time = 0.0

    def assign_job(self, job, current_time, process_time, setup_time):
        """Assign job to machine, update busy and setup times."""
        start_time = max(current_time, self.busy_until) + setup_time
        finish_time = start_time + process_time
        self.total_setup_time += setup_time
        self.total_busy_time += process_time
        self.busy_until = finish_time
        return finish_time

    @property
    def utilization(self):
        total_time = max(self.busy_until, 1)
        return self.total_busy_time / total_time


class Job:
    def __init__(self, job_id, due_time, base_process_time):
        self.id = job_id
        self.due_time = due_time
        self.base_process_time = base_process_time
        self.completed = False
        self.completion_time = 0.0


# ============================================================
# 🌐 Improved RMS Environment
# ============================================================
class ImprovedRMS_Env(gym.Env):
    """
    Gym-compatible RMS environment with optional negotiation and reconfiguration.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, negotiation=True, reconfigurable=True, eval_mode=False, **kwargs):
        super(ImprovedRMS_Env, self).__init__()

        # Configurable parameters
        self.negotiation_enabled = negotiation
        self.reconfigurable_enabled = reconfigurable
        self.eval_mode = eval_mode

        self.num_jobs = kwargs.get("num_jobs", 10)
        self.num_machines = kwargs.get("num_machines", 4)
        self.max_time = kwargs.get("max_time", 500)

        # Observation: [current_time, remaining_jobs, avg_machine_util]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([self.max_time, float(self.num_jobs), 1.0]),
            dtype=np.float32,
        )

        # Action: choose a machine to assign the next job
        self.action_space = spaces.Discrete(self.num_machines)

        # Initialize components
        self.reward_fn = RMSRewardFunction()
        self.jobs = []
        self.machines = []
        self.time = 0.0
        self.current_job_index = 0
        self.done = False

        # Metrics for analysis
        self.metrics = {
            "makespan": 0,
            "tardiness": 0,
            "utilization": 0,
            "negotiation_success": 0,
            "reconfig_freq": 0,
        }

        self.init_negotiation_module()
        self.init_reconfig_module()
        self.reset()

    # ------------------------------------------------------------
    # ⚙️ Negotiation / Reconfig Modules
    # ------------------------------------------------------------
    def init_negotiation_module(self):
        self.negotiation_attempts = 0
        self.negotiation_successes = 0

    def init_reconfig_module(self):
        self.reconfig_count = 0

    # ------------------------------------------------------------
    # 🔁 Environment Reset
    # ------------------------------------------------------------
    def reset(self, negotiation=None, reconfigurable=None):
        """Reset the environment for a new episode."""
        if negotiation is not None:
            self.negotiation_enabled = negotiation
        if reconfigurable is not None:
            self.reconfigurable_enabled = reconfigurable

        self.time = 0.0
        self.done = False
        self.current_job_index = 0

        self.jobs = [
            Job(i, np.random.uniform(80, 200), np.random.uniform(5, 20))
            for i in range(self.num_jobs)
        ]
        self.machines = [Machine(i) for i in range(self.num_machines)]

        for k in self.metrics:
            self.metrics[k] = 0

        return self._get_state()

    # ------------------------------------------------------------
    # 🎯 Step Function (Main RL Logic)
    # ------------------------------------------------------------
    def step(self, action):
        """Perform one scheduling step."""
        if self.done:
            return self._get_state(), 0.0, True, {}

        job = self.jobs[self.current_job_index]
        machine = self.machines[action]

        # Negotiation (if enabled)
        negotiation_success = 0
        if self.negotiation_enabled:
            self.negotiation_attempts += 1
            negotiation_success = np.random.choice([0, 1], p=[0.3, 0.7])
            self.negotiation_successes += negotiation_success

        # Reconfiguration (if enabled)
        setup_time = np.random.uniform(0, 3)
        if self.reconfigurable_enabled and np.random.rand() < 0.4:
            setup_time *= 1.3
            self.reconfig_count += 1

        process_time = job.base_process_time * np.random.uniform(0.8, 1.2)
        finish_time = machine.assign_job(job, self.time, process_time, setup_time)
        self.time = finish_time
        job.completed = True
        job.completion_time = finish_time

        tardiness = max(0, finish_time - job.due_time)
        step_progress = np.clip(process_time / job.base_process_time, 0, 1)

        reward = self.reward_fn.compute(
            step_progress=step_progress,
            job_completed=True,
            tardiness=tardiness,
            current_time=self.time,
            setup_time=setup_time,
        )

        self.current_job_index += 1
        if self.current_job_index >= self.num_jobs:
            self.done = True

        obs = self._get_state()
        info = {"tardiness": tardiness, "negotiation_success": negotiation_success}
        return obs, reward, self.done, info

    # ------------------------------------------------------------
    # 🔍 Observation (State Representation)
    # ------------------------------------------------------------
    def _get_state(self):
        remaining_jobs = self.num_jobs - self.current_job_index
        avg_util = np.mean([m.utilization for m in self.machines])
        return np.array([self.time, remaining_jobs, avg_util], dtype=np.float32)

    # ------------------------------------------------------------
    # 🧮 Post-Episode Metrics
    # ------------------------------------------------------------
    def compute_metrics(self):
        """Aggregate metrics after an episode for analysis."""
        self.metrics["makespan"] = max(j.completion_time for j in self.jobs)
        self.metrics["tardiness"] = np.mean(
            [max(0, j.completion_time - j.due_time) for j in self.jobs]
        )
        self.metrics["utilization"] = np.mean([m.utilization for m in self.machines])
        self.metrics["negotiation_success"] = (
            self.negotiation_successes / self.negotiation_attempts
            if self.negotiation_attempts > 0
            else 0
        )
        self.metrics["reconfig_freq"] = (
            self.reconfig_count / len(self.jobs)
            if self.reconfigurable_enabled
            else 0
        )
        return self.metrics

    # ------------------------------------------------------------
    # 🖼️ Optional Render
    # ------------------------------------------------------------
    def render(self, mode="human"):
        print(
            f"Time: {self.time:.2f} | Job {self.current_job_index}/{self.num_jobs} | "
            f"Utilization: {np.mean([m.utilization for m in self.machines]):.2f}"
        )

    # ------------------------------------------------------------
    # 🧪 Batch Run for Analysis (No Agent)
    # ------------------------------------------------------------
    def run(self, training=False):
        total_reward = 0.0
        self.stats = []

        while not self.done:
            action = np.random.randint(self.num_machines)
            obs, reward, done, info = self.step(action)
            total_reward += reward
            self.stats.append(
                {
                    "job_id": self.current_job_index,
                    "reward": reward,
                    "tardiness": info["tardiness"],
                    "negotiation_success": info["negotiation_success"],
                }
            )

        self.compute_metrics()
        return self.stats, total_reward


# # ============================================================
# # 🧪 Quick test
# # ============================================================
# if __name__ == "__main__":
#     env = ImprovedRMS_Env(negotiation=True, reconfigurable=True)
#     stats, total_reward = env.run(training=False)
#     print(f"✅ Total Reward: {total_reward:.3f}")
#     print("Metrics:", env.metrics)
