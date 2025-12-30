

import os
import sys
sys.path.append(r"C:\Users\91944\OneDrive\Desktop\ctde_rms_project")

import os, argparse, random, numpy as np, pandas as pd
from datetime import datetime
from env.envm import ImprovedRMS_Env
from agents.dqn_agent import StableEnhancedDQNAgent
from env.gym_wrapper import RMSGymEnv
from stable_baselines3 import PPO, SAC
def evaluate_all(runs=5, n_jobs=30, n_machines=4, ppo_model=None, sac_model=None, dqn_model=None, save_dir='results/compare'):
    os.makedirs(save_dir, exist_ok=True)
    results = []
    # load SB3 models if provided
    ppo = PPO.load(ppo_model) if ppo_model and os.path.exists(ppo_model) else None
    sac = SAC.load(sac_model) if sac_model and os.path.exists(sac_model) else None
    for run in range(runs):
        seed = 100 + run
        # EnhancedDQN (use exported state_dict if provided)
        env1 = ImprovedRMS_Env(n_jobs=n_jobs, n_machines=n_machines, seed=seed)
        state_dim = 5 + (7 * n_machines) + 3; action_dim = n_machines
        dqn_agent = StableEnhancedDQNAgent(state_dim, action_dim)
        if dqn_model and os.path.exists(dqn_model):
            try:
                import torch
                dqn_agent.q_network.load_state_dict(torch.load(dqn_model, map_location='cpu'))
            except Exception:
                pass
        stats, tot = env1.run(training=False)
        kpis = {'Agent':'EnhancedDQN','Run':run,'Makespan': max(s['makespan'] for s in stats) if stats else 0,'TotalTardiness': sum(s['tardiness'] for s in stats),'AvgUtilization': np.mean([m.utilization for m in env1.machines]), 'AvgSetupTime': np.mean([s['setup'] for s in stats]) if stats else 0}
        results.append(kpis)
        # PPO
        if ppo:
            gym = RMSGymEnv(n_jobs=n_jobs, n_machines=n_machines, seed=seed)
            obs = gym.reset(); done=False; while_loop=0
            while not done:
                action, _ = ppo.predict(obs, deterministic=True)
                obs, r, done, _ = gym.step(int(action))
                while_loop+=1
            stats_p = gym.env.stats
            results.append({'Agent':'PPO','Run':run,'Makespan': max(s['makespan'] for s in stats_p) if stats_p else 0,'TotalTardiness': sum(s['tardiness'] for s in stats_p),'AvgUtilization': np.mean([m.utilization for m in gym.env.machines]), 'AvgSetupTime': np.mean([s['setup'] for s in stats_p]) if stats_p else 0})
        # SAC
        if sac:
            gym = RMSGymEnv(n_jobs=n_jobs, n_machines=n_machines, seed=seed)
            obs = gym.reset(); done=False
            while not done:
                action, _ = sac.predict(obs, deterministic=True)
                obs, r, done, _ = gym.step(int(action))
            stats_s = gym.env.stats
            results.append({'Agent':'SAC','Run':run,'Makespan': max(s['makespan'] for s in stats_s) if stats_s else 0,'TotalTardiness': sum(s['tardiness'] for s in stats_s),'AvgUtilization': np.mean([m.utilization for m in gym.env.machines]), 'AvgSetupTime': np.mean([s['setup'] for s in stats_s]) if stats_s else 0})
        # EDF baseline
        env2 = ImprovedRMS_Env(n_jobs=n_jobs, n_machines=n_machines, seed=seed)
        # simple EDF selection: choose machine with smallest (setup+ptime)
        for job in env2.jobs:
            eligible = [i for i,m in enumerate(env2.machines) if job.process_type in m.capabilities]
            if not eligible:
                env2.stats.append({'job': job.id,'setup':0,'tardiness':0,'makespan':env2.time,'reward':-100,'machine':None}); continue
            best=None; best_v=1e9
            for i in eligible:
                m=env2.machines[i]; setup=env2.setup_time(m,job); p = job.processing_time/m.speed
                v = setup + p + max(0.0,(env2.time+setup+p)-job.deadline)
                if v<best_v: best_v=v; best=i
            env2.assign_job_direct(job,best)
        stats_edf = env2.stats
        results.append({'Agent':'EDF','Run':run,'Makespan': max(s['makespan'] for s in stats_edf) if stats_edf else 0,'TotalTardiness': sum(s['tardiness'] for s in stats_edf),'AvgUtilization': np.mean([m.utilization for m in env2.machines]), 'AvgSetupTime': np.mean([s['setup'] for s in stats_edf]) if stats_edf else 0})
        # Random baseline
        env3 = ImprovedRMS_Env(n_jobs=n_jobs, n_machines=n_machines, seed=seed)
        for job in env3.jobs:
            eligible = [i for i,m in enumerate(env3.machines) if job.process_type in m.capabilities]
            if not eligible:
                env3.stats.append({'job': job.id,'setup':0,'tardiness':0,'makespan':env3.time,'reward':-100,'machine':None}); continue
            import random; choice = random.choice(eligible); env3.assign_job_direct(job,choice)
        stats_rand = env3.stats
        results.append({'Agent':'Random','Run':run,'Makespan': max(s['makespan'] for s in stats_rand) if stats_rand else 0,'TotalTardiness': sum(s['tardiness'] for s in stats_rand),'AvgUtilization': np.mean([m.utilization for m in env3.machines]), 'AvgSetupTime': np.mean([s['setup'] for s in stats_rand]) if stats_rand else 0})
    df = pd.DataFrame(results)
    out = os.path.join(save_dir, 'compare_ext_results_synthetic.csv')
    df.to_csv(out, index=False)
    print('Saved', out)
if __name__=='__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--runs', type=int, default=5); parser.add_argument('--ppo_model', default=None); parser.add_argument('--sac_model', default=None); parser.add_argument('--dqn_model', default=None); args=parser.parse_args()
    evaluate_all(runs=args.runs, ppo_model=args.ppo_model, sac_model=args.sac_model, dqn_model=args.dqn_model)
