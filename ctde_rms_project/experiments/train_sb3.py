import argparse
import os
import pandas as pd
from env.gym_wrapper import RMSGymEnv
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback


class LossLoggingCallback(BaseCallback):
    """
    Logs PPO/SAC losses (policy, value, entropy) to CSV.
    """
    def __init__(self, log_dir="results", check_freq=1000, verbose=1):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.check_freq = check_freq
        self.records = []

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            try:
                losses = self.model.logger.name_to_value
                rec = {
                    "step": self.n_calls,
                    "policy_loss": losses.get("train/policy_gradient_loss", None),
                    "value_loss": losses.get("train/value_loss", None),
                    "entropy_loss": losses.get("train/entropy_loss", None),
                }
                self.records.append(rec)
            except Exception:
                pass
        return True

    def _on_training_end(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        df = pd.DataFrame(self.records)
        algo = type(self.model).__name__
        csv_path = os.path.join(self.log_dir, f"{algo.lower()}_loss_log.csv")
        df.to_csv(csv_path, index=False)
        if self.verbose:
            print(f"[LossLoggingCallback] Saved loss log to {csv_path}")


def train(algo="PPO", timesteps=50000, n_jobs=30, n_machines=4, seed=0, save="models/ppo_rms.zip"):
    os.makedirs(os.path.dirname(save), exist_ok=True)
    env = DummyVecEnv([lambda: RMSGymEnv(n_jobs=n_jobs, n_machines=n_machines, seed=seed)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    if algo == "PPO":
        model = PPO("MlpPolicy", env, verbose=1)
    else:
        model = SAC("MlpPolicy", env, verbose=1)

    callback = LossLoggingCallback(log_dir="results")
    model.learn(total_timesteps=timesteps, callback=callback)
    model.save(save)
    print(f"Saved {algo} model to {save}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["PPO", "SAC"], default="PPO")
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--jobs", type=int, default=30)
    parser.add_argument("--machines", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()
    save_path = args.save or f"models/{args.algo.lower()}_rms.zip"
    train(
        algo=args.algo,
        timesteps=args.timesteps,
        n_jobs=args.jobs,
        n_machines=args.machines,
        seed=args.seed,
        save=save_path,
    )
