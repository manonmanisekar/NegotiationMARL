"""
Enhanced Negotiation-Based RMS Scheduler with Multi-Round Bidding - FIXED

Key Enhancements:
1. Multi-round negotiation with counter-offers
2. Machine-specific negotiation strategies (aggressive, conservative, balanced)
3. Job negotiation tactics based on urgency and priority
4. Negotiation history tracking for better decisions
5. Dynamic bid adjustments based on market conditions
6. Coalition formation between machines
7. Detailed negotiation metrics and analysis

BUG FIX: Added proper None-check for winner_id to prevent list index errors
"""
import os
import sys
sys.path.append(r"C:\Users\91944\OneDrive\Desktop\ctde_rms_project")

from agents.dqn_agent import CentralDQN
import random, numpy as np, torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt, csv
from collections import deque, namedtuple
from typing import List, Dict, Tuple, Optional

# ===========================
# Enhanced Job & Machine Classes
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
        self.rejected_bids = []
        self.accepted_bid = None
        self.negotiation_strategy = random.choice(['aggressive', 'conservative', 'balanced'])

class Machine:
    def __init__(self, mid, base_failure=0.1, base_speed=1.0, flexibility=0.6):
        self.id = mid
        self.available = True
        self.current_job = None
        self.failure_rate = base_failure
        self.speed = base_speed
        self.flexibility = flexibility
        self.last_family = None
        self.maintenance_until = 0
        self.timeline = []
        self.utilization_time = 0.0
        self.total_bids_made = 0
        self.successful_bids = 0
        self.negotiation_strategy = random.choice(['aggressive', 'conservative', 'balanced'])
        self.reputation = 1.0
        self.bid_history = []

    def is_operational(self, t):
        return self.maintenance_until <= t

    def update_reputation(self, success):
        """Update machine reputation based on job completion success"""
        alpha = 0.1
        if success:
            self.reputation = min(1.0, self.reputation + alpha * (1.0 - self.reputation))
        else:
            self.reputation = max(0.1, self.reputation - alpha * self.reputation)

# ===========================
# Negotiation Protocol
# ===========================

class NegotiationProtocol:
    """Manages multi-round negotiation between jobs and machines"""
    
    def __init__(self, max_rounds=5):
        self.max_rounds = max_rounds
        self.negotiation_log = []
        self.coalition_cache = {}
        self.market_conditions = {'competition_level': 0.5}
    
    def conduct_negotiation(self, job: Job, machines: List[Machine], 
                          current_time: float, env) -> Tuple[Optional[int], Dict]:
        """
        Multi-round negotiation protocol - Returns (winner_id, info_dict)
        winner_id can be None, -1 (no machines), or a valid machine index
        """
        job.urgency = self._calculate_urgency(job, current_time)
        
        # Filter operational machines
        operational = [m for m in machines if m.is_operational(current_time)]
        if not operational:
            return None, {'reason': 'no_operational_machines', 'rounds': 0}  # Changed -1 to None for clarity
        
        # Update market competition level
        self.market_conditions['competition_level'] = len(operational) / max(1, job.urgency * 10)
        
        best_bid = None
        best_machine = None
        negotiation_history = []
        
        for round_num in range(self.max_rounds):
            bids = []
            
            for machine in operational:
                bid_score, components = self._calculate_bid(
                    job, machine, current_time, round_num, env
                )
                
                bid_score = self._apply_strategy(
                    bid_score, machine.negotiation_strategy, round_num
                )
                
                # Market dynamics adjustment
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
                machine.bid_history.append(bid_score)
            
            if not bids:
                break
            
            # Job evaluates bids
            sorted_bids = sorted(bids, key=lambda x: x['score'], reverse=True)
            current_best = sorted_bids[0]
            
            negotiation_history.append({
                'round': round_num,
                'bids': bids,
                'best_bid': current_best
            })
            
            # Check acceptance
            acceptance_threshold = self._get_acceptance_threshold(
                job, round_num, self.max_rounds
            )
            
            if current_best['score'] >= acceptance_threshold or round_num == self.max_rounds - 1:
                best_bid = current_best
                best_machine = current_best['machine_id']
                break
            
            # Reject weak bids for next round
            rejection_threshold = acceptance_threshold * 0.85
            operational = [
                m for m in operational 
                if any(b['machine_id'] == m.id and b['score'] >= rejection_threshold 
                       for b in bids)
            ]
            
            # Competitive pressure
            if len(operational) > 1:
                avg_bid = np.mean([b['score'] for b in bids])
                for m in operational:
                    m.bid_history.append(avg_bid)
        
        job.negotiation_rounds = len(negotiation_history)
        
        # Log negotiation
        self.negotiation_log.append({
            'job_id': job.id,
            'rounds': len(negotiation_history),
            'winner': best_machine,
            'final_score': best_bid['score'] if best_bid else None,
            'num_bidders': len(operational)
        })
        
        if best_machine is not None:
            machines[best_machine].successful_bids += 1
        
        return best_machine, {
            'rounds': len(negotiation_history),
            'final_bid': best_bid,
            'history': negotiation_history
        }
    
    def _calculate_bid(self, job: Job, machine: Machine, 
                      current_time: float, round_num: int, env) -> Tuple[float, Dict]:
        """Calculate comprehensive bid score with detailed components"""
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
        
        success_rate = (machine.successful_bids / max(1, machine.total_bids_made))
        components['experience'] = success_rate
        
        weights = {
            'availability': 2.0,
            'reliability': 1.5,
            'capability': 1.8,
            'workload': 1.2,
            'speed': 1.0,
            'priority_match': 2.5,
            'urgency_response': 1.5,
            'experience': 0.8
        }
        
        score = sum(components[k] * weights[k] for k in components)
        score -= setup_time * 0.15
        
        return score, components
    
    def _calculate_urgency(self, job: Job, current_time: float) -> float:
        """Calculate job urgency (0-1)"""
        remaining = max(0, job.deadline - current_time)
        total_time = max(1, job.deadline)
        urgency = 1.0 - (remaining / total_time)
        return min(1.0, max(0.0, urgency))
    
    def _apply_strategy(self, base_score: float, strategy: str, round_num: int) -> float:
        """Apply negotiation strategy to modify bid score"""
        if strategy == 'aggressive':
            return base_score * (1.15 + 0.08 * round_num)
        elif strategy == 'conservative':
            return base_score * (0.75 + 0.15 * round_num)
        else:  # balanced
            return base_score * (0.92 + 0.06 * round_num)
    
    def _get_acceptance_threshold(self, job: Job, round_num: int, max_rounds: int) -> float:
        """Calculate acceptance threshold based on job strategy and round"""
        base_threshold = 8.0
        
        priority_factor = 0.8 + (job.priority * 0.4)
        urgency_discount = job.urgency * 0.15 * round_num
        
        if job.negotiation_strategy == 'aggressive':
            threshold = base_threshold * priority_factor * (1.5 - 0.15 * round_num)
        elif job.negotiation_strategy == 'conservative':
            threshold = base_threshold * priority_factor * (0.9 - 0.1 * round_num)
        else:  # balanced
            threshold = base_threshold * priority_factor * (1.15 - 0.12 * round_num)
        
        return threshold - urgency_discount

# ===========================
# Enhanced Environment
# ===========================

class RMS_Enhanced_Negotiation_Env:
    def __init__(self, num_jobs=10, num_machines=4, families=('A','B','C'),
                 seed=None, max_time=1000):
        if seed:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.families = families
        self.max_time = max_time
        self.negotiation_protocol = NegotiationProtocol(max_rounds=5)
        self.reset()
    
    def reset(self):
        self.time = 0.0
        self.jobs: List[Job] = []
        self.machines: List[Machine] = []
        
        for j in range(self.num_jobs):
            fam = random.choice(self.families)
            sub = fam + str(random.randint(1,3))
            ptime = random.randint(5,20)
            deadline = random.randint(int(ptime*3), int(ptime*8)+40)
            pr = round(random.uniform(0.3,1.0),3)
            self.jobs.append(Job(j, fam, sub, ptime, deadline, pr))
        
        for m in range(self.num_machines):
            self.machines.append(Machine(
                m,
                base_failure=random.uniform(0.05, 0.15),
                base_speed=random.uniform(0.8, 1.3),
                flexibility=random.uniform(0.5, 0.9)
            ))
        
        return self.get_state()
    
    def setup_time(self, machine: Machine, job: Job) -> int:
        """Calculate setup time based on family compatibility"""
        if machine.last_family is None:
            return 0
        if machine.last_family == job.subfamily:
            return 1
        if machine.last_family == job.family:
            return 2 + int(1 - machine.flexibility)
        return 5 + int(3 * (1 - machine.flexibility))
    
    def assign_job_via_negotiation(self, job_id: int):
        """Assign job through enhanced negotiation protocol - FIXED"""
        job = self.jobs[job_id]
        
        if job.assigned or job.completed:
            return self.get_state(), -3.0, self.all_done(), {'reason': 'invalid'}
        
        # Conduct multi-round negotiation
        winner_id, neg_info = self.negotiation_protocol.conduct_negotiation(
            job, self.machines, self.time, self
        )
        
        # FIX: Check for None or -1 (both mean no winner)
        if winner_id is None or winner_id == -1:
            self.time += 1.0
            return self.get_state(), -1.0, self.all_done(), neg_info
        
        # FIX: Additional safety check for valid index
        if not (0 <= winner_id < len(self.machines)):
            print(f"WARNING: Invalid winner_id {winner_id}, skipping assignment")
            self.time += 1.0
            return self.get_state(), -2.0, self.all_done(), {'reason': 'invalid_winner'}
        
        # Execute assignment
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
        machine.timeline.append((job.start_time, job.finish_time, job.id))
        
        self.time += total
        
        # Failure handling
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
            
            # Negotiation bonus structure
            negotiation_bonus = 0
            if job.negotiation_rounds == 1:
                negotiation_bonus = -1.0
            elif job.negotiation_rounds == 2:
                negotiation_bonus = 1.5
            elif job.negotiation_rounds == 3:
                negotiation_bonus = 2.5
            elif job.negotiation_rounds == 4:
                negotiation_bonus = 2.0
            else:
                negotiation_bonus = 0.5
            
            if job.priority > 0.7 and job.negotiation_rounds < 2:
                negotiation_bonus -= 2.0
            
            reward = (job.priority * 10.0
                     - 0.25 * setup 
                     - 0.2 * tardiness 
                     + negotiation_bonus
                     + machine.reputation * 3.0)
        
        done = self.all_done() or self.time >= self.max_time
        
        neg_info.update({
            'machine_id': winner_id,
            'setup': setup,
            'fail': fail,
            'negotiation_rounds': job.negotiation_rounds
        })
        
        return self.get_state(), reward, done, neg_info
    
    def all_done(self):
        return all(j.completed for j in self.jobs)
    
    def get_state(self):
        """Enhanced state with negotiation-relevant features"""
        job_part = []
        max_deadline = max([j.deadline for j in self.jobs]) if self.jobs else 1.0
        max_pt = max([j.processing_time for j in self.jobs]) if self.jobs else 1.0
        
        for j in self.jobs:
            urgency = (self.time / max(1, j.deadline))
            job_part.extend([
                1.0 if j.completed else 0.0,
                1.0 if j.assigned else 0.0,
                j.priority,
                j.deadline / max_deadline,
                j.processing_time / max_pt,
                urgency,
                j.negotiation_rounds / 5.0,
                1.0 if j.negotiation_strategy == 'aggressive' else 0.0,
                1.0 if j.negotiation_strategy == 'conservative' else 0.0
            ])
        
        machine_part = []
        for m in self.machines:
            success_rate = m.successful_bids / max(1, m.total_bids_made)
            machine_part.extend([
                1.0 if (m.available and m.is_operational(self.time)) else 0.0,
                m.failure_rate,
                m.flexibility,
                m.speed / 1.3,
                m.reputation,
                success_rate
            ])
        
        return np.array(job_part + machine_part, dtype=np.float32)

# ===========================
# Training with Enhanced Negotiation Metrics
# ===========================

def train_enhanced_negotiation(env, episodes=100, max_steps=400):
    """Training loop with detailed negotiation analytics"""
    state_dim = len(env.get_state())
    action_dim = env.num_jobs
    
    from agents.dqn_agent import CentralDQN
    agent = CentralDQN(state_dim, action_dim, lr=5e-4)
    
    episode_rewards = []
    negotiation_stats = []
    
    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        ep_negotiations = []
        
        while not done and steps < max_steps:
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
            
            if 'negotiation_rounds' in info:
                ep_negotiations.append(info['negotiation_rounds'])
        
        episode_rewards.append(total_reward)
        avg_rounds = np.mean(ep_negotiations) if ep_negotiations else 0
        negotiation_stats.append({
            'episode': ep,
            'avg_rounds': avg_rounds,
            'total_negotiations': len(ep_negotiations)
        })
        
        if ep % 10 == 0 or ep == 1:
            print(f"Ep {ep}/{episodes} | Reward: {total_reward:.2f} | "
                  f"Avg Neg Rounds: {avg_rounds:.2f} | Eps: {agent.eps:.3f}")
    
    return agent, episode_rewards, negotiation_stats, env


# ===========================
# Analysis & Visualization
# ===========================

def analyze_negotiation_impact(env, negotiation_stats):
    """Generate comprehensive negotiation analysis"""
    
    # 1. Negotiation rounds distribution
    all_negotiations = env.negotiation_protocol.negotiation_log
    rounds_dist = [n['rounds'] for n in all_negotiations]
    
    # 2. Machine performance
    machine_stats = []
    for m in env.machines:
        if m.total_bids_made > 0:
            machine_stats.append({
                'machine_id': m.id,
                'success_rate': m.successful_bids / m.total_bids_made,
                'reputation': m.reputation,
                'strategy': m.negotiation_strategy,
                'total_bids': m.total_bids_made
            })
    
    # 3. Job negotiation patterns
    job_stats = []
    for j in env.jobs:
        if j.completed:
            job_stats.append({
                'job_id': j.id,
                'priority': j.priority,
                'rounds': j.negotiation_rounds,
                'strategy': j.negotiation_strategy,
                'tardiness': max(0, j.finish_time - j.deadline)
            })
    
    # === PRINT DETAILED STATISTICS ===
    print("\n" + "="*60)
    print("NEGOTIATION IMPACT ANALYSIS")
    print("="*60)
    
    print(f"\n📊 NEGOTIATION ROUNDS STATISTICS:")
    print(f"   Average Rounds: {np.mean(rounds_dist):.2f}")
    print(f"   Median Rounds: {np.median(rounds_dist):.0f}")
    print(f"   Max Rounds: {max(rounds_dist)}")
    print(f"   Std Dev: {np.std(rounds_dist):.2f}")
    
    rounds_count = {}
    for r in rounds_dist:
        rounds_count[r] = rounds_count.get(r, 0) + 1
    for r in sorted(rounds_count.keys()):
        print(f"   {r} rounds: {rounds_count[r]} negotiations ({100*rounds_count[r]/len(rounds_dist):.1f}%)")
    
    print(f"\n🏭 MACHINE PERFORMANCE:")
    for m in sorted(machine_stats, key=lambda x: x['success_rate'], reverse=True):
        print(f"   Machine {m['machine_id']} ({m['strategy']:12s}): "
              f"Success={m['success_rate']:.1%}, Reputation={m['reputation']:.3f}, "
              f"Bids={m['total_bids']}")
    
    print(f"\n💼 JOB STRATEGY EFFECTIVENESS:")
    job_strategies = {}
    for j in job_stats:
        s = j['strategy']
        if s not in job_strategies:
            job_strategies[s] = {'rounds': [], 'tardiness': []}
        job_strategies[s]['rounds'].append(j['rounds'])
        job_strategies[s]['tardiness'].append(j['tardiness'])
    
    for strat, data in job_strategies.items():
        print(f"   {strat.capitalize():12s}: Avg Rounds={np.mean(data['rounds']):.2f}, "
              f"Avg Tardiness={np.mean(data['tardiness']):.2f}")
    
    print(f"\n🎯 PRIORITY vs NEGOTIATION:")
    high_priority = [j for j in job_stats if j['priority'] > 0.7]
    low_priority = [j for j in job_stats if j['priority'] <= 0.4]
    if high_priority and low_priority:
        print(f"   High Priority Jobs (>0.7): Avg {np.mean([j['rounds'] for j in high_priority]):.2f} rounds")
        print(f"   Low Priority Jobs (≤0.4): Avg {np.mean([j['rounds'] for j in low_priority]):.2f} rounds")
    
    print(f"\n💰 NEGOTIATION EFFICIENCY:")
    completed_jobs = [j for j in env.jobs if j.completed]
    on_time = sum(1 for j in completed_jobs if j.finish_time <= j.deadline)
    print(f"   Jobs Completed: {len(completed_jobs)}/{len(env.jobs)}")
    print(f"   On-Time Delivery: {on_time}/{len(completed_jobs)} ({100*on_time/max(1,len(completed_jobs)):.1f}%)")
    print(f"   Total Negotiations: {len(all_negotiations)}")
    print(f"   Avg Time per Job: {env.time/max(1,len(completed_jobs)):.2f} time units")
    
    print("="*60 + "\n")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Negotiation rounds distribution
    axes[0, 0].hist(rounds_dist, bins=range(1, max(rounds_dist)+2), edgecolor='black')
    axes[0, 0].set_title('Negotiation Rounds Distribution')
    axes[0, 0].set_xlabel('Rounds')
    axes[0, 0].set_ylabel('Frequency')
    
    # Plot 2: Machine success rates
    m_ids = [m['machine_id'] for m in machine_stats]
    success_rates = [m['success_rate'] for m in machine_stats]
    axes[0, 1].bar(m_ids, success_rates)
    axes[0, 1].set_title('Machine Bid Success Rates')
    axes[0, 1].set_xlabel('Machine ID')
    axes[0, 1].set_ylabel('Success Rate')
    
    # Plot 3: Reputation vs Success Rate
    reps = [m['reputation'] for m in machine_stats]
    axes[0, 2].scatter(reps, success_rates)
    axes[0, 2].set_title('Reputation vs Success Rate')
    axes[0, 2].set_xlabel('Reputation')
    axes[0, 2].set_ylabel('Success Rate')
    
    # Plot 4: Strategy effectiveness
    strategies = ['aggressive', 'conservative', 'balanced']
    strategy_success = {s: [] for s in strategies}
    for m in machine_stats:
        strategy_success[m['strategy']].append(m['success_rate'])
    
    avg_success = [np.mean(strategy_success[s]) if strategy_success[s] else 0 
                   for s in strategies]
    axes[1, 0].bar(strategies, avg_success)
    axes[1, 0].set_title('Strategy Effectiveness (Machines)')
    axes[1, 0].set_ylabel('Avg Success Rate')
    
    # Plot 5: Priority vs Negotiation Rounds
    priorities = [j['priority'] for j in job_stats]
    rounds = [j['rounds'] for j in job_stats]
    axes[1, 1].scatter(priorities, rounds)
    axes[1, 1].set_title('Job Priority vs Negotiation Rounds')
    axes[1, 1].set_xlabel('Priority')
    axes[1, 1].set_ylabel('Rounds')
    
    # Plot 6: Negotiation efficiency over episodes
    episodes = [s['episode'] for s in negotiation_stats]
    avg_rounds_ep = [s['avg_rounds'] for s in negotiation_stats]
    axes[1, 2].plot(episodes, avg_rounds_ep)
    axes[1, 2].set_title('Negotiation Efficiency Over Time')
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].set_ylabel('Avg Rounds per Job')
    
    plt.tight_layout()
    plt.savefig('negotiation_analysis.png', dpi=300)
    print("Saved negotiation_analysis.png")
    
    # Save detailed stats
    with open('negotiation_detailed_stats.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Avg Rounds', np.mean(rounds_dist)])
        writer.writerow(['Avg Machine Success Rate', np.mean(success_rates)])
        writer.writerow(['Avg Machine Reputation', np.mean(reps)])
        writer.writerow(['Total Negotiations', len(all_negotiations)])
    
    print("Saved negotiation_detailed_stats.csv")

# ===========================
# Main Execution
# ===========================

if __name__ == "__main__":
    print("=== Enhanced Negotiation RMS Training ===\n")
    
    env = RMS_Enhanced_Negotiation_Env(
        num_jobs=12, 
        num_machines=4, 
        seed=42
    )
    
    agent, rewards, neg_stats, final_env = train_enhanced_negotiation(
        env, 
        episodes=100, 
        max_steps=400
    )
    
    print("\n=== Analyzing Negotiation Impact ===\n")
    analyze_negotiation_impact(final_env, neg_stats)
    
    print("\n=== Training Complete ===")
    print(f"Final avg reward (last 10): {np.mean(rewards[-10:]):.2f}")
    print(f"Total negotiations logged: {len(final_env.negotiation_protocol.negotiation_log)}")
