# ctde_rms_project/env/environment.py
import os, sys
sys.path.append(r"C:\Users\91944\OneDrive\Desktop\ctde_rms_project")

import numpy as np
from gym import spaces


class ImprovedRMS_Env:
    """
    Reconfigurable Manufacturing System (RMS) Environment
    optimized for Dueling Attention DQN training.
    NOW SUPPORTS CONFIGURABLE JOB AND MACHINE COUNTS!
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
        self.seed = seed

        # --- Observation and Action spaces ---
        self.state_dim = self.num_jobs * 3 + self.num_machines * 3
        self.observation_space = spaces.Box(
            low=np.zeros(self.state_dim, dtype=np.float32),
            high=np.ones(self.state_dim, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.num_machines)

        # --- Generate system data based on num_jobs and num_machines ---
        self._initialize_system_parameters()

        self.reset()

    def _initialize_system_parameters(self):
        """Initialize job and machine parameters based on sizes"""
        # Generate job processing times (distributed between 5 and 15 time units)
        self.job_processing_times = np.random.uniform(
            5.0, 15.0, size=self.num_jobs
        ).astype(np.float32)
        
        # Generate job due dates (based on processing times with some slack)
        # Due dates are processing time * (1.5 to 2.5) to ensure variety
        self.job_due_dates = (
            self.job_processing_times * np.random.uniform(1.5, 2.5, size=self.num_jobs)
        ).astype(np.float32)
        
        # Generate machine base speed factors (0.8 to 1.2)
        self.machine_base_times = np.random.uniform(
            0.8, 1.2, size=self.num_machines
        ).astype(np.float32)
        
        # Generate reconfiguration time matrix (machines x machines)
        # Diagonal is 0 (no reconfig when staying on same machine)
        self.reconfig_time_matrix = np.random.uniform(
            1.0, 5.0, size=(self.num_machines, self.num_machines)
        ).astype(np.float32)
        np.fill_diagonal(self.reconfig_time_matrix, 0.0)
        
        # Make matrix symmetric (reconfig time same in both directions)
        self.reconfig_time_matrix = (
            self.reconfig_time_matrix + self.reconfig_time_matrix.T
        ) / 2.0

    # ==========================================================
    #  Core Functions
    # ==========================================================
    def reset(self):
        """Reset environment for a new episode."""
        self.current_time = 0.0
        self.job_remaining = self.job_processing_times.copy()
        self.machine_utilization = np.zeros(self.num_machines, dtype=np.float32)
        self.machine_last_job = np.full(self.num_machines, -1, dtype=np.int32)  # Track last job on each machine
        self.reconfig_cost = np.zeros(self.num_machines, dtype=np.float32)
        self.total_tardiness = 0.0
        self.completed_jobs = 0
        return self._get_state()

    def _get_state(self):
        """Constructs the state vector (normalized for network stability)."""
        # Normalize job features
        max_proc = np.max(self.job_processing_times) + 1e-6
        max_due = np.max(self.job_due_dates) + 1e-6
        
        job_rem_norm = self.job_remaining / max_proc
        job_due_norm = self.job_due_dates / max_due
        job_tardy_norm = np.clip(
            (self.current_time - self.job_due_dates) / max_due, 0, 1
        )
        
        # Normalize machine features
        mach_util_norm = np.clip(self.machine_utilization / 100.0, 0, 1)
        setup_norm = np.clip(
            self.reconfig_cost / (np.max(self.reconfig_time_matrix) + 1e-6), 0, 1
        )
        reconfig_norm = np.array(
            [1 if self.reconfigurable else 0] * self.num_machines, dtype=np.float32
        )

        state = np.concatenate([
            job_rem_norm, job_due_norm, job_tardy_norm,
            mach_util_norm, setup_norm, reconfig_norm
        ]).astype(np.float32)
        
        return state

    def step(self, action):
        """Simulate job assignment to a machine and compute scaled reward."""
        done = False
        
        # Validate action
        if action < 0 or action >= self.num_machines:
            action = np.clip(action, 0, self.num_machines - 1)
        
        job_idx = self._select_job(action)
        
        # Check if no jobs remaining
        if job_idx is None or self.job_remaining[job_idx] <= 0:
            done = True
            next_state = self._get_state()
            reward = -1.0  # Small penalty for invalid action
            return next_state, float(reward), done, {}

        # --- Processing and timing ---
        base_time = self.job_processing_times[job_idx] * self.machine_base_times[action]
        
        # Calculate reconfiguration time if switching job types
        if self.reconfigurable and self.machine_last_job[action] != -1:
            prev_job = self.machine_last_job[action]
            # Use job indices modulo num_machines for reconfig matrix lookup
            reconfig_time = self.reconfig_time_matrix[
                prev_job % self.num_machines, 
                job_idx % self.num_machines
            ]
        else:
            reconfig_time = 0.0
        
        setup_time = 0.5 * reconfig_time if self.negotiation else reconfig_time
        total_time = base_time + setup_time

        self.current_time += total_time
        self.machine_utilization[action] += total_time
        self.reconfig_cost[action] = reconfig_time
        self.machine_last_job[action] = job_idx

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
        #  Enhanced Reward Function
        # ======================================================
        util_reward = 2.0 * (np.mean(self.machine_utilization) / (self.current_time + 1))
        completion_bonus = 10.0 * completion_bonus     # scaled up
        reconfig_penalty = -0.5 * np.mean(self.reconfig_cost)
        makespan_penalty = -0.05 * total_time
        tardiness_penalty = -0.1 * (self.total_tardiness / (self.completed_jobs + 1))

        reward = (
            completion_bonus + util_reward + reconfig_penalty + 
            makespan_penalty + tardiness_penalty
        )

        # ======================================================
        #  Termination Condition
        # ======================================================
        if self.completed_jobs >= self.num_jobs or self.current_time > 200:
            done = True
            # Final completion bonus based on how many jobs finished
            reward += 5.0 * (self.completed_jobs / self.num_jobs)

        next_state = self._get_state()
        return next_state, float(reward), done, {}

    def _select_job(self, machine_idx):
        """Negotiation logic for selecting the next job."""
        remaining = np.where(self.job_remaining > 0)[0]
        
        if len(remaining) == 0:
            return None
        
        if self.negotiation:
            # Priority-based selection considering tardiness
            tardiness_weight = np.clip(self.current_time - self.job_due_dates, 0, np.inf)
            priority = self.job_remaining + 0.5 * tardiness_weight
            job_idx = remaining[np.argmin(priority[remaining])]
        else:
            # Simple: first available job
            job_idx = remaining[0]
        
        return job_idx

    def render(self):
        """Optional: visualize current state"""
        print(f"\n=== Time: {self.current_time:.2f} ===")
        print(f"Completed Jobs: {self.completed_jobs}/{self.num_jobs}")
        print(f"Total Tardiness: {self.total_tardiness:.2f}")
        print(f"Machine Utilization: {self.machine_utilization}")
