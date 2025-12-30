import pandas as pd, matplotlib.pyplot as plt, seaborn as sns, os
def kpi_boxplot(csv='results/kpi_results.csv', save_dir='results/analysis'):
    os.makedirs(save_dir, exist_ok=True)
    # create simple plots from provided CSV(s)
    try:
        df = pd.read_csv(csv)
    except Exception as e:
        print('Missing kpi csv', e); return
    plt.figure(figsize=(6,4)); plt.plot(df['Episode'], df['TotalReward'], marker='o'); plt.title('Episode Rewards'); plt.xlabel('Episode'); plt.ylabel('TotalReward'); plt.tight_layout(); plt.savefig(os.path.join(save_dir,'episode_rewards.png')); plt.close()
if __name__=='__main__': kpi_boxplot()
