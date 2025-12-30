# ctde_rms_project/env/environment.py
import os, sys
sys.path.append(r"C:\Users\91944\OneDrive\Desktop\ctde_rms_project")

import numpy as np
from gym import spaces

# ctde_rms_project/env/environment.py
import numpy as np
from gym import spaces


class ImprovedRMS_Env:
    """
    Reconfigurable Manufacturing System (RMS) Environment
    optimized for Dueling Attention DQN training.
    Higher reward scaling for stronger learning signal.
    """

    def __init__(
        self,
        num_jobs=30,
        num_machines=4,
        reconfigurable=True,
        negotiation=True,
        eval_mode=False,
        seed=42
    ):
        np.random.seed(seed)
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.reconfigurable = reconfigurable
        self.negotiation = negotiation
        self.eval_mode = eval_mode

        # --- Observation and Action spaces ---
        self.state_dim = self.num_jobs * 3 + self.num_machines * 3
        self.observation_space = spaces.Box(
            low=np.zeros(self.state_dim, dtype=np.float32),
            high=np.ones(self.state_dim, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.num_machines)

        # --- System data ---
        self.job_processing_times = np.array([8, 10, 6, 12, 9], dtype=np.float32)
        self.job_due_dates = np.array([20, 22, 18, 25, 21], dtype=np.float32)
        self.machine_base_times = np.array([1.0, 0.9, 1.2, 1.1], dtype=np.float32)
        self.reconfig_time_matrix = np.array(
            [[0, 2, 3, 4],
             [2, 0, 3, 2],
             [3, 2, 0, 1],
             [4, 3, 2, 0]], dtype=np.float32
        )

        self.reset()

    # ==========================================================
    #  Core Functions
    # ==========================================================
    def reset(self):
        """Reset environment for a new episode."""
        self.current_time = 0.0
        self.job_remaining = self.job_processing_times.copy()
        self.machine_utilization = np.zeros(self.num_machines, dtype=np.float32)
        self.reconfig_cost = np.zeros(self.num_machines, dtype=np.float32)
        self.total_tardiness = 0.0
        self.completed_jobs = 0
        return self._get_state()

    def _get_state(self):
        """Constructs the state vector (normalized for network stability)."""
        job_rem_norm = self.job_remaining / np.max(self.job_processing_times)
        job_due_norm = self.job_due_dates / np.max(self.job_due_dates)
        job_tardy_norm = np.clip((self.current_time - self.job_due_dates) / np.max(self.job_due_dates), 0, 1)
        mach_util_norm = np.clip(self.machine_utilization / 50.0, 0, 1)
        setup_norm = np.clip(self.reconfig_cost / np.max(self.reconfig_time_matrix), 0, 1)
        reconfig_norm = np.array([1 if self.reconfigurable else 0] * self.num_machines)

        state = np.concatenate([
            job_rem_norm, job_due_norm, job_tardy_norm,
            mach_util_norm, setup_norm, reconfig_norm
        ]).astype(np.float32)
        return state

    def step(self, action):
        """Simulate job assignment to a machine and compute scaled reward."""
        done = False
        job_idx = self._select_job(action)

        # --- Processing and timing ---
        base_time = self.job_processing_times[job_idx] * self.machine_base_times[action]
        reconfig_time = self.reconfig_time_matrix[action, job_idx % self.num_machines] if self.reconfigurable else 0.0
        setup_time = 0.5 * reconfig_time if self.negotiation else reconfig_time
        total_time = base_time + setup_time

        self.current_time += total_time
        self.machine_utilization[action] += total_time
        self.reconfig_cost[action] = reconfig_time

        # --- Job completion tracking ---
        self.job_remaining[job_idx] -= base_time
        completion_bonus, tardiness_penalty = 0.0, 0.0

        if self.job_remaining[job_idx] <= 0:
            self.completed_jobs += 1
            tardiness = max(0, self.current_time - self.job_due_dates[job_idx])
            self.total_tardiness += tardiness
            # Reward jobs finished on or before due date
            completion_bonus = 1.0 - (tardiness / (self.job_due_dates[job_idx] + 1))
            self.job_remaining[job_idx] = 0.0
        else:
            tardiness_penalty = -0.01 * (self.current_time / np.max(self.job_due_dates))

        # ======================================================
        #  Enhanced Reward Function (no normalization)
        # ======================================================
        util_reward = 2.0 * (np.mean(self.machine_utilization) / (self.current_time + 1))
        completion_bonus = 10.0 * completion_bonus     # scaled up
        reconfig_penalty = -0.5 * np.mean(self.reconfig_cost)
        makespan_penalty = -0.05 * total_time
        tardiness_penalty = -0.1 * (self.total_tardiness / (self.completed_jobs + 1))

        reward = completion_bonus + util_reward + reconfig_penalty + makespan_penalty + tardiness_penalty

        # ======================================================
        #  Termination Condition
        # ======================================================
        if self.completed_jobs >= self.num_jobs or self.current_time > 100:
            done = True
            reward += 5.0 * (self.completed_jobs / self.num_jobs)  # final completion bonus

        next_state = self._get_state()
        return next_state, float(reward), done, {}

    def _select_job(self, machine_idx):
        """Negotiation logic for selecting the next job."""
        if self.negotiation:
            tardiness_weight = np.clip(self.current_time - self.job_due_dates, 0, np.inf)
            priority = self.job_remaining + 0.5 * tardiness_weight
            remaining = np.where(self.job_remaining > 0)[0]
            job_idx = remaining[np.argmin(priority[remaining])] if len(remaining) else 0
        else:
            remaining = np.where(self.job_remaining > 0)[0]
            job_idx = remaining[0] if len(remaining) else 0
        return job_idx
