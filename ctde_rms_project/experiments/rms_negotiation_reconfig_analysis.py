"""
Comprehensive Analysis: Impact of Negotiation and Reconfigurability on RMS Performance
Compares 4 scenarios:
1. With Negotiation + Reconfigurable
2. Without Negotiation + Reconfigurable  
3. With Negotiation + Fixed
4. Without Negotiation + Fixed
"""
import os
import sys
sys.path.append(r"C:\Users\91944\OneDrive\Desktop\ctde_rms_project")

from agents.dqn_agent import CentralDQN
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import List, Dict, Tuple
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)

# ===========================
# Job and Machine Classes
# ===========================

class Job:
    def __init__(self, jid, family, subfamily, ptime, deadline, priority):
        self.id = jid
        self.family = family
        self.subfamily = subfamily
        self.processing_time = ptime
        self.deadline = deadline
        self.priority = priority
        self.urgency = 0.0
        self.assigned = False
        self.completed = False
        self.start_time = None
        self.finish_time = None
        self.negotiation_rounds = 0
        self.negotiation_strategy = random.choice(['aggressive', 'conservative', 'balanced'])

class Machine:
    def __init__(self, mid, base_failure=0.1, base_speed=1.0, flexibility=0.6, reconfigurable=True):
        self.id = mid
        self.available = True
        self.failure_rate = base_failure
        self.speed = base_speed
        self.flexibility = flexibility
        self.reconfigurable = reconfigurable
        self.last_family = None
        self.maintenance_until = 0
        self.utilization_time = 0.0
        self.reconfiguration_count = 0
        self.total_reconfiguration_time = 0.0
        self.total_bids_made = 0
        self.successful_bids = 0
        self.reputation = 1.0
        self.negotiation_strategy = random.choice(['aggressive', 'conservative', 'balanced'])

    def is_operational(self, t):
        return self.maintenance_until <= t

    def update_reputation(self, success):
        alpha = 0.1
        if success:
            self.reputation = min(1.0, self.reputation + alpha * (1.0 - self.reputation))
        else:
            self.reputation = max(0.1, self.reputation - alpha * self.reputation)

# ===========================
# Negotiation Protocol
# ===========================

class NegotiationProtocol:
    def __init__(self, max_rounds=3, enabled=True):
        self.max_rounds = max_rounds
        self.enabled = enabled
        self.negotiation_log = []
        self.market_conditions = {'competition_level': 0.5}
    
    def conduct_negotiation(self, job: Job, machines: List[Machine], 
                          current_time: float, env):
        """Conduct negotiation if enabled, otherwise simple assignment"""
        
        if not self.enabled:
            # Simple greedy assignment without negotiation
            operational = [m for m in machines if m.is_operational(current_time) and m.available]
            if not operational:
                return None, {'reason': 'no_machines', 'rounds': 0}
            
            # Select machine with lowest setup time
            best_machine = None
            best_score = -999
            for m in operational:
                setup = env.setup_time(m, job)
                score = 10.0 - setup - m.failure_rate * 5
                if score > best_score:
                    best_score = score
                    best_machine = m.id
            
            return best_machine, {'rounds': 0, 'method': 'greedy'}
        
        # Full negotiation protocol
        job.urgency = self._calculate_urgency(job, current_time)
        operational = [m for m in machines if m.is_operational(current_time)]
        
        if not operational:
            return None, {'reason': 'no_operational_machines', 'rounds': 0}
        
        self.market_conditions['competition_level'] = len(operational) / max(1, job.urgency * 10)
        
        best_bid = None
        best_machine = None
        negotiation_history = []
        
        for round_num in range(self.max_rounds):
            bids = []
            
            for machine in operational:
                bid_score, components = self._calculate_bid(job, machine, current_time, round_num, env)
                bid_score = self._apply_strategy(bid_score, machine.negotiation_strategy, round_num)
                
                if self.market_conditions['competition_level'] > 0.7:
                    bid_score *= 1.15
                elif self.market_conditions['competition_level'] < 0.3:
                    bid_score *= 0.90
                
                bids.append({
                    'machine_id': machine.id,
                    'score': bid_score,
                    'components': components,
                    'round': round_num
                })
                machine.total_bids_made += 1
            
            if not bids:
                break
            
            sorted_bids = sorted(bids, key=lambda x: x['score'], reverse=True)
            current_best = sorted_bids[0]
            
            negotiation_history.append({
                'round': round_num,
                'bids': bids,
                'best_bid': current_best
            })
            
            acceptance_threshold = self._get_acceptance_threshold(job, round_num, self.max_rounds)
            
            if current_best['score'] >= acceptance_threshold or round_num == self.max_rounds - 1:
                best_bid = current_best
                best_machine = current_best['machine_id']
                break
            
            rejection_threshold = acceptance_threshold * 0.85
            operational = [
                m for m in operational 
                if any(b['machine_id'] == m.id and b['score'] >= rejection_threshold for b in bids)
            ]
        
        job.negotiation_rounds = len(negotiation_history)
        
        self.negotiation_log.append({
            'job_id': job.id,
            'rounds': len(negotiation_history),
            'winner': best_machine,
            'final_score': best_bid['score'] if best_bid else None
        })
        
        if best_machine is not None:
            machines[best_machine].successful_bids += 1
        
        return best_machine, {
            'rounds': len(negotiation_history),
            'final_bid': best_bid,
            'method': 'negotiation'
        }
    
    def _calculate_bid(self, job: Job, machine: Machine, current_time: float, round_num: int, env):
        components = {}
        
        components['availability'] = 1.0 if machine.available else 0.3
        components['reliability'] = (1.0 - machine.failure_rate) * machine.reputation
        
        setup_time = env.setup_time(machine, job)
        components['capability'] = 1.0 - (setup_time / 10.0)
        
        avg_util = machine.utilization_time / max(1.0, current_time)
        components['workload'] = max(0.0, 1.0 - avg_util)
        
        components['speed'] = machine.speed / 1.3
        components['priority_match'] = job.priority
        components['urgency_response'] = job.urgency * machine.flexibility
        
        success_rate = machine.successful_bids / max(1, machine.total_bids_made)
        components['experience'] = success_rate
        
        weights = {
            'availability': 2.0, 'reliability': 1.5, 'capability': 1.8,
            'workload': 1.2, 'speed': 1.0, 'priority_match': 2.5,
            'urgency_response': 1.5, 'experience': 0.8
        }
        
        score = sum(components[k] * weights[k] for k in components)
        score -= setup_time * 0.15
        
        return score, components
    
    def _calculate_urgency(self, job: Job, current_time: float) -> float:
        remaining = max(0, job.deadline - current_time)
        total_time = max(1, job.deadline)
        urgency = 1.0 - (remaining / total_time)
        return min(1.0, max(0.0, urgency))
    
    def _apply_strategy(self, base_score: float, strategy: str, round_num: int) -> float:
        if strategy == 'aggressive':
            return base_score * (1.15 + 0.08 * round_num)
        elif strategy == 'conservative':
            return base_score * (0.75 + 0.15 * round_num)
        else:
            return base_score * (0.92 + 0.06 * round_num)
    
    def _get_acceptance_threshold(self, job: Job, round_num: int, max_rounds: int) -> float:
        base_threshold = 6.0
        priority_factor = 0.8 + (job.priority * 0.4)
        urgency_discount = job.urgency * 0.15 * round_num
        
        if job.negotiation_strategy == 'aggressive':
            threshold = base_threshold * priority_factor * (1.3 - 0.12 * round_num)
        elif job.negotiation_strategy == 'conservative':
            threshold = base_threshold * priority_factor * (0.9 - 0.1 * round_num)
        else:
            threshold = base_threshold * priority_factor * (1.1 - 0.11 * round_num)
        
        return threshold - urgency_discount

# ===========================
# Enhanced Environment
# ===========================

class RMS_Comprehensive_Env:
    def __init__(self, num_jobs=12, num_machines=4, families=('A','B','C'),
                 seed=None, max_time=1000, use_negotiation=True, use_reconfiguration=True):
        if seed:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.families = families
        self.max_time = max_time
        self.use_negotiation = use_negotiation
        self.use_reconfiguration = use_reconfiguration
        
        self.negotiation_protocol = NegotiationProtocol(max_rounds=3, enabled=use_negotiation)
        self.reset()
    
    def reset(self):
        self.time = 0.0
        self.jobs: List[Job] = []
        self.machines: List[Machine] = []
        self.total_reconfig_time = 0.0
        
        # Generate jobs
        for j in range(self.num_jobs):
            fam = random.choice(self.families)
            sub = fam + str(random.randint(1,3))
            ptime = random.randint(5,20)
            deadline = random.randint(int(ptime*3), int(ptime*8)+40)
            pr = round(random.uniform(0.3,1.0),3)
            self.jobs.append(Job(j, fam, sub, ptime, deadline, pr))
        
        # Generate machines
        for m in range(self.num_machines):
            self.machines.append(Machine(
                m,
                base_failure=random.uniform(0.05, 0.15),
                base_speed=random.uniform(0.8, 1.3),
                flexibility=random.uniform(0.5, 0.9),
                reconfigurable=self.use_reconfiguration
            ))
        
        return self.get_state()
    
    def setup_time(self, machine: Machine, job: Job) -> float:
        """Calculate setup time with reconfiguration logic"""
        if machine.last_family is None:
            return 0.0
        
        if machine.last_family == job.subfamily:
            return 1.0
        
        if machine.last_family == job.family:
            base_time = 2.0
        else:
            base_time = 5.0
        
        # Reconfigurable machines can adapt faster
        if machine.reconfigurable:
            reconfig_time = base_time * (1.0 - machine.flexibility * 0.5)
            machine.reconfiguration_count += 1
            machine.total_reconfiguration_time += reconfig_time
            self.total_reconfig_time += reconfig_time
            return reconfig_time
        else:
            # Fixed machines have longer, inflexible setup times
            return base_time * (1.5 - machine.flexibility * 0.3)
    
    def assign_job_via_negotiation(self, job_id: int):
        job = self.jobs[job_id]
        
        if job.assigned or job.completed:
            return self.get_state(), -3.0, self.all_done(), {'reason': 'invalid'}
        
        winner_id, neg_info = self.negotiation_protocol.conduct_negotiation(
            job, self.machines, self.time, self
        )
        
        if winner_id is None or winner_id == -1:
            self.time += 1.0
            return self.get_state(), -1.0, self.all_done(), neg_info
        
        if not (0 <= winner_id < len(self.machines)):
            self.time += 1.0
            return self.get_state(), -2.0, self.all_done(), {'reason': 'invalid_winner'}
        
        machine = self.machines[winner_id]
        setup = self.setup_time(machine, job)
        ptime = job.processing_time / machine.speed
        total = setup + ptime
        
        job.assigned = True
        job.start_time = self.time + setup
        job.finish_time = self.time + total
        
        machine.last_family = job.subfamily
        machine.available = False
        machine.utilization_time += ptime
        
        self.time += total
        
        fail = random.random() < machine.failure_rate
        if fail:
            machine.maintenance_until = self.time + random.randint(5, 12)
            machine.update_reputation(False)
            reward = -4.0 - 0.3 * setup
        else:
            job.completed = True
            machine.available = True
            machine.update_reputation(True)
            
            tardiness = max(0.0, job.finish_time - job.deadline)
            
            negotiation_bonus = 0
            if self.use_negotiation:
                if job.negotiation_rounds == 2:
                    negotiation_bonus = 1.5
                elif job.negotiation_rounds == 3:
                    negotiation_bonus = 2.0
                elif job.negotiation_rounds == 1:
                    negotiation_bonus = -0.5
            
            reconfig_penalty = 0
            if not machine.reconfigurable:
                reconfig_penalty = setup * 0.2  # Extra penalty for fixed machines
            
            reward = (job.priority * 10.0
                     - 0.25 * setup
                     - 0.2 * tardiness
                     + negotiation_bonus
                     + machine.reputation * 2.5
                     - reconfig_penalty)
        
        done = self.all_done() or self.time >= self.max_time
        
        neg_info.update({
            'machine_id': winner_id,
            'setup': setup,
            'fail': fail,
            'negotiation_rounds': job.negotiation_rounds if self.use_negotiation else 0
        })
        
        return self.get_state(), reward, done, neg_info
    
    def all_done(self):
        return all(j.completed for j in self.jobs)
    
    def get_state(self):
        job_part = []
        max_deadline = max([j.deadline for j in self.jobs]) if self.jobs else 1.0
        max_pt = max([j.processing_time for j in self.jobs]) if self.jobs else 1.0
        
        for j in self.jobs:
            urgency = self.time / max(1, j.deadline)
            job_part.extend([
                1.0 if j.completed else 0.0,
                1.0 if j.assigned else 0.0,
                j.priority,
                j.deadline / max_deadline,
                j.processing_time / max_pt,
                urgency
            ])
        
        machine_part = []
        for m in self.machines:
            machine_part.extend([
                1.0 if (m.available and m.is_operational(self.time)) else 0.0,
                m.failure_rate,
                m.flexibility,
                m.speed / 1.3,
                1.0 if m.reconfigurable else 0.0
            ])
        
        return np.array(job_part + machine_part, dtype=np.float32)
    
    def get_metrics(self):
        """Calculate comprehensive performance metrics"""
        completed = [j for j in self.jobs if j.completed]
        
        if not completed:
            return {
                'makespan': self.time,
                'total_tardiness': 0,
                'avg_utilization': 0,
                'total_reconfig_time': self.total_reconfig_time,
                'avg_reconfig_time_per_machine': 0,
                'completion_rate': 0,
                'avg_negotiation_rounds': 0
            }
        
        makespan = max([j.finish_time for j in completed])
        total_tardiness = sum([max(0, j.finish_time - j.deadline) for j in completed])
        
        total_util = sum([m.utilization_time for m in self.machines])
        avg_utilization = total_util / (len(self.machines) * makespan) * 100
        
        total_reconfig = sum([m.total_reconfiguration_time for m in self.machines])
        avg_reconfig_per_machine = total_reconfig / len(self.machines)
        
        avg_neg_rounds = np.mean([j.negotiation_rounds for j in completed]) if self.use_negotiation else 0
        
        return {
            'makespan': makespan,
            'total_tardiness': total_tardiness,
            'avg_utilization': avg_utilization,
            'total_reconfig_time': total_reconfig,
            'avg_reconfig_time_per_machine': avg_reconfig_per_machine,
            'completion_rate': len(completed) / len(self.jobs) * 100,
            'avg_negotiation_rounds': avg_neg_rounds
        }

# ===========================
# Training Function
# ===========================

def train_scenario(scenario_name, use_negotiation, use_reconfiguration, 
                   episodes=100, num_jobs=12, num_machines=4, seed=42):
    """Train a single scenario"""
    print(f"\n{'='*60}")
    print(f"Training: {scenario_name}")
    print(f"{'='*60}")
    
    env = RMS_Comprehensive_Env(
        num_jobs=num_jobs,
        num_machines=num_machines,
        seed=seed,
        use_negotiation=use_negotiation,
        use_reconfiguration=use_reconfiguration
    )
    
    state_dim = len(env.get_state())
    action_dim = env.num_jobs
    
    agent = CentralDQN(state_dim, action_dim, lr=5e-4)
    
    episode_metrics = []
    
    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        
        while not done and steps < 400:
            unassigned = [j for j in env.jobs if not j.assigned and not j.completed]
            if not unassigned:
                break
            
            legal = [(not j.assigned and not j.completed) for j in env.jobs]
            action = agent.act(state, legal)
            
            next_state, reward, done, info = env.assign_job_via_negotiation(action)
            
            agent.store(state, action, reward, next_state, float(done))
            agent.update(batch=64)
            
            state = next_state
            total_reward += reward
            steps += 1
        
        metrics = env.get_metrics()
        metrics['episode'] = ep
        metrics['total_reward'] = total_reward
        episode_metrics.append(metrics)
        
        if ep % 10 == 0:
            print(f"Ep {ep}/{episodes} | Reward: {total_reward:.2f} | "
                  f"Makespan: {metrics['makespan']:.1f} | "
                  f"Tardiness: {metrics['total_tardiness']:.1f} | "
                  f"Util: {metrics['avg_utilization']:.1f}% | "
                  f"Reconfig: {metrics['total_reconfig_time']:.1f}")
    
    return episode_metrics, env

# ===========================
# Main Analysis
# ===========================

def run_comprehensive_analysis():
    """Run all 4 scenarios and compare results"""
    
    scenarios = {
        'With Negotiation + Reconfigurable': (True, True),
        'Without Negotiation + Reconfigurable': (False, True),
        'With Negotiation + Fixed': (True, False),
        'Without Negotiation + Fixed': (False, False)
    }
    
    all_results = {}
    
    for scenario_name, (use_neg, use_reconfig) in scenarios.items():
        metrics, final_env = train_scenario(
            scenario_name, 
            use_neg, 
            use_reconfig,
            episodes=500,
            num_jobs=12,
            num_machines=4,
            seed=42
        )
        all_results[scenario_name] = {
            'metrics': metrics,
            'env': final_env
        }
    
    # Create comprehensive visualizations
    create_comparison_plots(all_results)
    create_detailed_analysis(all_results)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

def create_comparison_plots(all_results):
    """Create comprehensive comparison plots"""
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    scenarios = list(all_results.keys())
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    # Plot 1: Makespan Box Plot
    ax1 = fig.add_subplot(gs[0, 0])
    makespan_data = []
    for scenario in scenarios:
        metrics = all_results[scenario]['metrics']
        makespan_data.append([m['makespan'] for m in metrics])
    
    bp1 = ax1.boxplot(makespan_data, labels=[s.replace(' + ', '\n') for s in scenarios],
                       patch_artist=True)
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_title('Makespan', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Time Units', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Total Tardiness Box Plot
    ax2 = fig.add_subplot(gs[0, 1])
    tardiness_data = []
    for scenario in scenarios:
        metrics = all_results[scenario]['metrics']
        tardiness_data.append([m['total_tardiness'] for m in metrics])
    
    bp2 = ax2.boxplot(tardiness_data, labels=[s.replace(' + ', '\n') for s in scenarios],
                       patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_title('Total Tardiness', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Time Units', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Average Utilization Box Plot
    ax3 = fig.add_subplot(gs[0, 2])
    util_data = []
    for scenario in scenarios:
        metrics = all_results[scenario]['metrics']
        util_data.append([m['avg_utilization'] for m in metrics])
    
    bp3 = ax3.boxplot(util_data, labels=[s.replace(' + ', '\n') for s in scenarios],
                       patch_artist=True)
    for patch, color in zip(bp3['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_title('Avg Utilization', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Percentage (%)', fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Reconfiguration Time Comparison
    ax4 = fig.add_subplot(gs[1, 0])
    reconfig_data = []
    for scenario in scenarios:
        metrics = all_results[scenario]['metrics']
        reconfig_data.append([m['total_reconfig_time'] for m in metrics])
    
    bp4 = ax4.boxplot(reconfig_data, labels=[s.replace(' + ', '\n') for s in scenarios],
                       patch_artist=True)
    for patch, color in zip(bp4['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax4.set_title('Total Reconfiguration Time', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Time Units', fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Negotiation Rounds (where applicable)
    ax5 = fig.add_subplot(gs[1, 1])
    neg_scenarios = [s for s in scenarios if 'With Negotiation' in s]
    neg_data = []
    neg_labels = []
    for scenario in neg_scenarios:
        metrics = all_results[scenario]['metrics']
        rounds = [m['avg_negotiation_rounds'] for m in metrics if m['avg_negotiation_rounds'] > 0]
        if rounds:
            neg_data.append(rounds)
            neg_labels.append(scenario.replace(' + ', '\n'))
    
    if neg_data:
        bp5 = ax5.boxplot(neg_data, labels=neg_labels, patch_artist=True)
        for patch in bp5['boxes']:
            patch.set_facecolor('#9b59b6')
            patch.set_alpha(0.7)
    ax5.set_title('Negotiation Rounds\n(Negotiation-Enabled Only)', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Avg Rounds', fontsize=11)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Completion Rate
    ax6 = fig.add_subplot(gs[1, 2])
    completion_data = []
    for scenario in scenarios:
        metrics = all_results[scenario]['metrics']
        completion_data.append([m['completion_rate'] for m in metrics])
    
    bp6 = ax6.boxplot(completion_data, labels=[s.replace(' + ', '\n') for s in scenarios],
                       patch_artist=True)
    for patch, color in zip(bp6['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax6.set_title('Completion Rate', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Percentage (%)', fontsize=11)
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Learning Curves - Reward
    ax7 = fig.add_subplot(gs[2, :2])
    for scenario, color in zip(scenarios, colors):
        metrics = all_results[scenario]['metrics']
        episodes = [m['episode'] for m in metrics]
        rewards = [m['total_reward'] for m in metrics]
        
        # Smooth with moving average
        window = 5
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax7.plot(episodes[:len(smoothed)], smoothed, color=color, 
                    label=scenario, linewidth=2, alpha=0.8)
        else:
            ax7.plot(episodes, rewards, color=color, label=scenario, linewidth=2, alpha=0.8)
    
    ax7.set_title('Learning Curves: Total Reward Over Episodes', fontsize=14, fontweight='bold')
    ax7.set_xlabel('Episode', fontsize=11)
    ax7.set_ylabel('Total Reward', fontsize=11)
    ax7.legend(loc='best', fontsize=9)
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Summary Statistics Table
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    table_data = []
    table_data.append(['Metric'] + [s.split(' + ')[0][:12] for s in scenarios])
    
    for metric_name, metric_key in [('Makespan', 'makespan'), 
                                     ('Tardiness', 'total_tardiness'),
                                     ('Utilization', 'avg_utilization'),
                                     ('Reconfig', 'total_reconfig_time')]:
        row = [metric_name]
        for scenario in scenarios:
            metrics = all_results[scenario]['metrics']
            values = [m[metric_key] for m in metrics]
            row.append(f"{np.mean(values):.1f}")
        table_data.append(row)
    
    table = ax8.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25] + [0.19]*4)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax8.set_title('Average Performance Metrics', fontsize=12, fontweight='bold', pad=20)
    
    plt.suptitle('Impact of Negotiation and Reconfigurability on RMS Performance', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig('rms_negotiation_reconfig_analysis_part1.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: rms_negotiation_reconfig_analysis_part1.png")
    plt.close()

def create_detailed_analysis(all_results):
    """Create detailed analysis with additional metrics"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    scenarios = list(all_results.keys())
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    # Plot 1: Makespan vs Reconfiguration Time
    ax = axes[0, 0]
    for scenario, color in zip(scenarios, colors):
        metrics = all_results[scenario]['metrics']
        makespans = [m['makespan'] for m in metrics]
        reconfigs = [m['total_reconfig_time'] for m in metrics]
        ax.scatter(reconfigs, makespans, c=color, label=scenario.split(' + ')[0], 
                  s=80, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Total Reconfiguration Time', fontsize=11, fontweight='bold')
    ax.set_ylabel('Makespan', fontsize=11, fontweight='bold')
    ax.set_title('Makespan vs Reconfiguration Time', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Utilization vs Tardiness Trade-off
    ax = axes[0, 1]
    for scenario, color in zip(scenarios, colors):
        metrics = all_results[scenario]['metrics']
        utils = [m['avg_utilization'] for m in metrics]
        tardiness = [m['total_tardiness'] for m in metrics]
        ax.scatter(utils, tardiness, c=color, label=scenario.replace(' + ', '\n'), 
                  s=80, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Average Utilization (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Total Tardiness', fontsize=11, fontweight='bold')
    ax.set_title('Utilization vs Tardiness Trade-off', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Performance Improvement Over Episodes
    ax = axes[0, 2]
    for scenario, color in zip(scenarios, colors):
        metrics = all_results[scenario]['metrics']
        episodes = [m['episode'] for m in metrics]
        makespans = [m['makespan'] for m in metrics]
        
        # Calculate moving average
        window = 5
        if len(makespans) >= window:
            smoothed = np.convolve(makespans, np.ones(window)/window, mode='valid')
            ax.plot(episodes[:len(smoothed)], smoothed, color=color, 
                   label=scenario.split(' + ')[0], linewidth=2.5, alpha=0.8)
    ax.set_xlabel('Episode', fontsize=11, fontweight='bold')
    ax.set_ylabel('Makespan (smoothed)', fontsize=11, fontweight='bold')
    ax.set_title('Makespan Improvement Over Training', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Reconfiguration Efficiency
    ax = axes[1, 0]
    reconfig_scenarios = []
    avg_reconfig_times = []
    scenario_labels = []
    
    for scenario in scenarios:
        metrics = all_results[scenario]['metrics']
        avg_reconfig = np.mean([m['avg_reconfig_time_per_machine'] for m in metrics])
        avg_reconfig_times.append(avg_reconfig)
        scenario_labels.append(scenario.replace(' + ', '\n'))
    
    bars = ax.bar(range(len(scenarios)), avg_reconfig_times, color=colors, 
                  alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenario_labels, fontsize=9)
    ax.set_ylabel('Avg Reconfig Time per Machine', fontsize=11, fontweight='bold')
    ax.set_title('Reconfiguration Efficiency Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, avg_reconfig_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 5: Statistical Summary - Violin Plot for Tardiness
    ax = axes[1, 1]
    tardiness_data = []
    for scenario in scenarios:
        metrics = all_results[scenario]['metrics']
        tardiness_data.append([m['total_tardiness'] for m in metrics])
    
    parts = ax.violinplot(tardiness_data, positions=range(len(scenarios)), 
                          showmeans=True, showmedians=True)
    
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([s.replace(' + ', '\n') for s in scenarios], fontsize=9)
    ax.set_ylabel('Total Tardiness', fontsize=11, fontweight='bold')
    ax.set_title('Tardiness Distribution (Violin Plot)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Comprehensive Performance Radar Chart
    ax = axes[1, 2]
    
    # Prepare data for radar chart
    categories = ['Makespan\n(inv)', 'Tardiness\n(inv)', 'Utilization', 
                  'Completion', 'Reconfig\n(inv)']
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    ax = plt.subplot(2, 3, 6, projection='polar')
    
    for scenario, color in zip(scenarios[:2], colors[:2]):  # Show only 2 for clarity
        metrics = all_results[scenario]['metrics']
        
        # Normalize metrics (0-1, higher is better)
        makespan_norm = 1 - (np.mean([m['makespan'] for m in metrics]) / 200)  # Inverse
        tardiness_norm = 1 - (np.mean([m['total_tardiness'] for m in metrics]) / 150)  # Inverse
        util_norm = np.mean([m['avg_utilization'] for m in metrics]) / 100
        completion_norm = np.mean([m['completion_rate'] for m in metrics]) / 100
        reconfig_norm = 1 - (np.mean([m['total_reconfig_time'] for m in metrics]) / 100)  # Inverse
        
        values = [makespan_norm, tardiness_norm, util_norm, completion_norm, reconfig_norm]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=scenario.split(' + ')[0], 
                color=color, markersize=6)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title('Performance Radar\n(Normalized, Higher=Better)', 
                fontsize=11, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('rms_negotiation_reconfig_analysis_part2.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: rms_negotiation_reconfig_analysis_part2.png")
    plt.close()
    
    # Print detailed statistics
    print_detailed_statistics(all_results)

def print_detailed_statistics(all_results):
    """Print comprehensive statistical analysis"""
    
    print("\n" + "="*80)
    print(" "*20 + "DETAILED PERFORMANCE ANALYSIS")
    print("="*80)
    
    scenarios = list(all_results.keys())
    
    for scenario in scenarios:
        print(f"\n{'─'*80}")
        print(f"📊 {scenario}")
        print(f"{'─'*80}")
        
        metrics = all_results[scenario]['metrics']
        env = all_results[scenario]['env']
        
        # Calculate statistics
        makespans = [m['makespan'] for m in metrics]
        tardiness = [m['total_tardiness'] for m in metrics]
        utilizations = [m['avg_utilization'] for m in metrics]
        reconfig_times = [m['total_reconfig_time'] for m in metrics]
        completion_rates = [m['completion_rate'] for m in metrics]
        
        print(f"\n  MAKESPAN:")
        print(f"    Mean:   {np.mean(makespans):8.2f}  │  Std: {np.std(makespans):8.2f}")
        print(f"    Median: {np.median(makespans):8.2f}  │  Range: [{np.min(makespans):.1f}, {np.max(makespans):.1f}]")
        
        print(f"\n  TARDINESS:")
        print(f"    Mean:   {np.mean(tardiness):8.2f}  │  Std: {np.std(tardiness):8.2f}")
        print(f"    Median: {np.median(tardiness):8.2f}  │  Range: [{np.min(tardiness):.1f}, {np.max(tardiness):.1f}]")
        
        print(f"\n  UTILIZATION (%):")
        print(f"    Mean:   {np.mean(utilizations):8.2f}  │  Std: {np.std(utilizations):8.2f}")
        print(f"    Median: {np.median(utilizations):8.2f}  │  Range: [{np.min(utilizations):.1f}, {np.max(utilizations):.1f}]")
        
        print(f"\n  RECONFIGURATION TIME:")
        print(f"    Total Mean:      {np.mean(reconfig_times):8.2f}")
        print(f"    Per Machine Avg: {np.mean(reconfig_times)/env.num_machines:8.2f}")
        print(f"    Std:             {np.std(reconfig_times):8.2f}")
        
        print(f"\n  COMPLETION RATE:")
        print(f"    Mean:   {np.mean(completion_rates):8.2f}%  │  Std: {np.std(completion_rates):8.2f}%")
        
        # Machine-specific statistics
        print(f"\n  MACHINE STATISTICS:")
        for m in env.machines:
            print(f"    Machine {m.id}: Reconfig Count={m.reconfiguration_count:3d}, "
                  f"Total Reconfig Time={m.total_reconfiguration_time:6.2f}, "
                  f"Utilization={m.utilization_time:6.2f}")
        
        if 'With Negotiation' in scenario:
            neg_rounds = [m['avg_negotiation_rounds'] for m in metrics 
                         if m['avg_negotiation_rounds'] > 0]
            if neg_rounds:
                print(f"\n  NEGOTIATION ROUNDS:")
                print(f"    Mean:   {np.mean(neg_rounds):8.2f}  │  Std: {np.std(neg_rounds):8.2f}")
                print(f"    Median: {np.median(neg_rounds):8.2f}  │  Range: [{np.min(neg_rounds):.1f}, {np.max(neg_rounds):.1f}]")
    
    # Comparative analysis
    print(f"\n{'='*80}")
    print(" "*25 + "COMPARATIVE ANALYSIS")
    print(f"{'='*80}")
    
    # Compare negotiation impact
    print(f"\n  📈 NEGOTIATION IMPACT:")
    
    neg_reconfig = [m['makespan'] for m in all_results['With Negotiation + Reconfigurable']['metrics']]
    no_neg_reconfig = [m['makespan'] for m in all_results['Without Negotiation + Reconfigurable']['metrics']]
    
    improvement_reconfig = ((np.mean(no_neg_reconfig) - np.mean(neg_reconfig)) / 
                           np.mean(no_neg_reconfig) * 100)
    
    print(f"    Reconfigurable Machines:")
    print(f"      With Negotiation:    {np.mean(neg_reconfig):.2f}")
    print(f"      Without Negotiation: {np.mean(no_neg_reconfig):.2f}")
    print(f"      Improvement:         {improvement_reconfig:+.2f}%")
    
    neg_fixed = [m['makespan'] for m in all_results['With Negotiation + Fixed']['metrics']]
    no_neg_fixed = [m['makespan'] for m in all_results['Without Negotiation + Fixed']['metrics']]
    
    improvement_fixed = ((np.mean(no_neg_fixed) - np.mean(neg_fixed)) / 
                        np.mean(no_neg_fixed) * 100)
    
    print(f"\n    Fixed Machines:")
    print(f"      With Negotiation:    {np.mean(neg_fixed):.2f}")
    print(f"      Without Negotiation: {np.mean(no_neg_fixed):.2f}")
    print(f"      Improvement:         {improvement_fixed:+.2f}%")
    
    # Compare reconfigurability impact
    print(f"\n  🔧 RECONFIGURABILITY IMPACT:")
    
    reconfig_neg = [m['makespan'] for m in all_results['With Negotiation + Reconfigurable']['metrics']]
    fixed_neg = [m['makespan'] for m in all_results['With Negotiation + Fixed']['metrics']]
    
    improvement_with_neg = ((np.mean(fixed_neg) - np.mean(reconfig_neg)) / 
                           np.mean(fixed_neg) * 100)
    
    print(f"    With Negotiation:")
    print(f"      Reconfigurable: {np.mean(reconfig_neg):.2f}")
    print(f"      Fixed:          {np.mean(fixed_neg):.2f}")
    print(f"      Improvement:    {improvement_with_neg:+.2f}%")
    
    reconfig_no_neg = [m['makespan'] for m in all_results['Without Negotiation + Reconfigurable']['metrics']]
    fixed_no_neg = [m['makespan'] for m in all_results['Without Negotiation + Fixed']['metrics']]
    
    improvement_without_neg = ((np.mean(fixed_no_neg) - np.mean(reconfig_no_neg)) / 
                              np.mean(fixed_no_neg) * 100)
    
    print(f"\n    Without Negotiation:")
    print(f"      Reconfigurable: {np.mean(reconfig_no_neg):.2f}")
    print(f"      Fixed:          {np.mean(fixed_no_neg):.2f}")
    print(f"      Improvement:    {improvement_without_neg:+.2f}%")
    
    # Best configuration
    print(f"\n  🏆 BEST CONFIGURATION:")
    avg_makespans = {scenario: np.mean([m['makespan'] for m in all_results[scenario]['metrics']]) 
                     for scenario in scenarios}
    best_scenario = min(avg_makespans, key=avg_makespans.get)
    print(f"    {best_scenario}")
    print(f"    Average Makespan: {avg_makespans[best_scenario]:.2f}")
    
    print(f"\n{'='*80}\n")
    
    # Save to CSV
    save_results_to_csv(all_results)

def save_results_to_csv(all_results):
    """Save detailed results to CSV files"""
    
    # Summary statistics
    summary_data = []
    for scenario, data in all_results.items():
        metrics = data['metrics']
        summary_data.append({
            'Scenario': scenario,
            'Avg_Makespan': np.mean([m['makespan'] for m in metrics]),
            'Std_Makespan': np.std([m['makespan'] for m in metrics]),
            'Avg_Tardiness': np.mean([m['total_tardiness'] for m in metrics]),
            'Std_Tardiness': np.std([m['total_tardiness'] for m in metrics]),
            'Avg_Utilization': np.mean([m['avg_utilization'] for m in metrics]),
            'Std_Utilization': np.std([m['avg_utilization'] for m in metrics]),
            'Avg_Reconfig_Time': np.mean([m['total_reconfig_time'] for m in metrics]),
            'Std_Reconfig_Time': np.std([m['total_reconfig_time'] for m in metrics]),
            'Avg_Completion_Rate': np.mean([m['completion_rate'] for m in metrics])
        })
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv('rms_analysis_summary.csv', index=False)
    print("✓ Saved: rms_analysis_summary.csv")
    
    # Detailed episode data
    all_episode_data = []
    for scenario, data in all_results.items():
        for m in data['metrics']:
            row = m.copy()
            row['Scenario'] = scenario
            all_episode_data.append(row)
    
    df_detailed = pd.DataFrame(all_episode_data)
    df_detailed.to_csv('rms_analysis_detailed.csv', index=False)
    print("✓ Saved: rms_analysis_detailed.csv")

# ===========================
# Main Execution
# ===========================

if __name__ == "__main__":
    print("\n" + "="*80)
    print(" "*15 + "RMS NEGOTIATION & RECONFIGURATION ANALYSIS")
    print("="*80)
    print("\nThis analysis compares 4 scenarios:")
    print("  1. With Negotiation + Reconfigurable")
    print("  2. Without Negotiation + Reconfigurable")
    print("  3. With Negotiation + Fixed")
    print("  4. Without Negotiation + Fixed")
    print("\nMetrics tracked:")
    print("  • Makespan")
    print("  • Total Tardiness")
    print("  • Machine Utilization")
    print("  • Reconfiguration Time")
    print("  • Completion Rate")
    print("  • Negotiation Rounds (where applicable)")
    print("="*80 + "\n")
    
    run_comprehensive_analysis()
    
    print("\n" + "="*80)
    print(" "*25 + "ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  📊 rms_negotiation_reconfig_analysis_part1.png")
    print("  📊 rms_negotiation_reconfig_analysis_part2.png")
    print("  📄 rms_analysis_summary.csv")
    print("  📄 rms_analysis_detailed.csv")
    print("="*80 + "\n")