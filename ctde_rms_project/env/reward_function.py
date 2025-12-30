import numpy as np

class RMSRewardFunction:
    def __init__(self):
        self.w_step = 0.4
        self.w_comp = 2.0
        self.w_tard = 0.002
        self.w_make = 0.0003
        self.w_setup = 0.0003
        self.R_min, self.R_max = -50.0, 50.0

    def compute(self, step_progress, job_completed, tardiness, current_time, setup_time):
        r_step = self.w_step * step_progress
        r_comp = self.w_comp * max(-self.w_tard, 1.0 - tardiness * self.w_tard) if job_completed else 0.0
        r_tard = -self.w_tard * tardiness
        r_make = -self.w_make * current_time
        r_setup = -self.w_setup * setup_time
        total = r_step + r_comp + r_tard + r_make + r_setup
        return float(np.clip(total, self.R_min, self.R_max))
