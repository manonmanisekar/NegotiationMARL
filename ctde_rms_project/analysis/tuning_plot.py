import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def moving_average(data, w=20):
    return np.convolve(data, np.ones(w)/w, mode='valid')

for lr in [1e-3, 5e-4, 1e-4, 5e-5]:
    df = pd.read_csv(f"results_tuning/lr_{lr}.csv")
    plt.plot(moving_average(df["Reward"]), label=f"lr={lr}")

plt.xlabel("Episode")
plt.ylabel("Reward (MA20)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("results_tuning/tuning_results.png", dpi=300)
