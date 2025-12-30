import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple
import random

@dataclass
class Machine:
    id: int
    native_processes: Set[int]  # Processes machine can do natively
    reconfigurable_processes: Set[int]  # Processes machine can do after reconfiguration
    is_broken: bool = False
    reconfiguration_time: float = 5.0  # Time to reconfigure
    current_process: int = None
    negotiation_flexibility: float = 0.8  # Willingness to negotiate (0-1)
    
    def can_process(self, process_id: int, allow_reconfig: bool) -> bool:
        if self.is_broken:
            return False
        if process_id in self.native_processes:
            return True
        if allow_reconfig and process_id in self.reconfigurable_processes:
            return True
        return False
    
    def get_processing_time(self, process_id: int, base_time: float, allow_reconfig: bool) -> float:
        if self.is_broken:
            return float('inf')
        if process_id in self.native_processes:
            return base_time
        if allow_reconfig and process_id in self.reconfigurable_processes:
            # Add reconfiguration time if switching process
            if self.current_process != process_id:
                return base_time + self.reconfiguration_time
            return base_time
        return float('inf')
    
    def negotiate_priority(self, job_priority: int, job_urgency: float) -> float:
        """Calculate negotiated priority score"""
        # Higher score = more willing to take the job
        base_score = job_priority * job_urgency
        negotiated_score = base_score * self.negotiation_flexibility
        return negotiated_score

@dataclass
class Job:
    id: int
    processes: List[int]  # Sequence of processes needed
    process_times: List[float]  # Time for each process
    priority: int = 1  # Job priority (1=low, 5=high)
    due_date: float = float('inf')  # Job due date
    current_process_idx: int = 0
    completion_time: float = 0
    waited_time: float = 0  # Track waiting time
    
    def is_complete(self) -> bool:
        return self.current_process_idx >= len(self.processes)
    
    def get_current_process(self) -> int:
        if self.is_complete():
            return None
        return self.processes[self.current_process_idx]
    
    def get_current_process_time(self) -> float:
        if self.is_complete():
            return 0
        return self.process_times[self.current_process_idx]
    
    def get_urgency(self, current_time: float) -> float:
        """Calculate job urgency based on due date and waiting time"""
        if self.due_date == float('inf'):
            urgency = 0.5
        else:
            time_remaining = self.due_date - current_time
            if time_remaining <= 0:
                urgency = 1.0  # Overdue - highest urgency
            else:
                # More urgent as deadline approaches
                urgency = max(0.1, 1.0 - (time_remaining / self.due_date))
        
        # Increase urgency based on waiting time
        waiting_factor = min(0.5, self.waited_time / 100)
        return min(1.0, urgency + waiting_factor)

class NegotiationProtocol:
    """Handles negotiation between jobs and machines"""
    
    @staticmethod
    def negotiate_allocation(job: Job, machines: List[Machine], current_time: float,
                           machine_available_time: Dict, allow_reconfig: bool) -> Tuple[Machine, float]:
        """
        Negotiate which machine should process the job
        Returns: (selected_machine, negotiated_start_time)
        """
        process_id = job.get_current_process()
        process_time = job.get_current_process_time()
        job_urgency = job.get_urgency(current_time)
        
        candidates = []
        
        for machine in machines:
            if not machine.can_process(process_id, allow_reconfig):
                continue
            
            available_time = machine_available_time[machine.id]
            actual_proc_time = machine.get_processing_time(
                process_id, process_time, allow_reconfig
            )
            
            if actual_proc_time == float('inf'):
                continue
            
            # Machine evaluates the job
            machine_score = machine.negotiate_priority(job.priority, job_urgency)
            
            # Calculate completion time if this machine takes the job
            start_time = max(current_time, available_time)
            completion_time = start_time + actual_proc_time
            
            # Job evaluates the machine (prefers faster completion)
            wait_time = start_time - current_time
            job_score = job.priority / (1 + wait_time + actual_proc_time)
            
            # Negotiated score considers both perspectives
            negotiated_score = (machine_score + job_score) / 2
            
            candidates.append({
                'machine': machine,
                'start_time': start_time,
                'completion_time': completion_time,
                'score': negotiated_score,
                'wait_time': wait_time
            })
        
        if not candidates:
            return None, float('inf')
        
        # Select machine with best negotiated score
        best_candidate = max(candidates, key=lambda x: x['score'])
        return best_candidate['machine'], best_candidate['start_time']

class ProductionScheduler:
    def __init__(self, machines: List[Machine], jobs: List[Job], 
                 allow_reconfiguration: bool, allow_negotiation: bool):
        self.machines = machines
        self.jobs = jobs
        self.allow_reconfiguration = allow_reconfiguration
        self.allow_negotiation = allow_negotiation
        self.current_time = 0
        self.machine_available_time = {m.id: 0 for m in machines}
        self.negotiation_protocol = NegotiationProtocol()
        
    def simulate(self, breakdown_machine_id: int = None) -> Dict:
        # Apply breakdown if specified
        if breakdown_machine_id is not None:
            for machine in self.machines:
                if machine.id == breakdown_machine_id:
                    machine.is_broken = True
        
        total_completion_time = 0
        job_completion_times = []
        total_wait_time = 0
        failed_negotiations = 0
        total_tardiness = 0
        tardy_jobs = 0
        total_setup_time = 0
        num_reconfigurations = 0
        
        # Track machine utilization
        machine_busy_time = {m.id: 0 for m in self.machines}
        machine_setup_time = {m.id: 0 for m in self.machines}
        machine_idle_time = {m.id: 0 for m in self.machines}
        
        for job in self.jobs:
            job_start_time = self.current_time
            
            while not job.is_complete():
                process_id = job.get_current_process()
                process_time = job.get_current_process_time()
                
                if self.allow_negotiation:
                    # Use negotiation protocol
                    best_machine, start_time = self.negotiation_protocol.negotiate_allocation(
                        job, self.machines, self.current_time, 
                        self.machine_available_time, self.allow_reconfiguration
                    )
                    
                    if best_machine is None:
                        failed_negotiations += 1
                        return self._create_failed_result(failed_negotiations)
                    
                    actual_proc_time = best_machine.get_processing_time(
                        process_id, process_time, self.allow_reconfiguration
                    )
                    
                else:
                    # Traditional scheduling (earliest available machine)
                    best_machine = None
                    earliest_time = float('inf')
                    
                    for machine in self.machines:
                        if machine.can_process(process_id, self.allow_reconfiguration):
                            available_time = self.machine_available_time[machine.id]
                            
                            if available_time < earliest_time:
                                earliest_time = available_time
                                best_machine = machine
                    
                    if best_machine is None:
                        return self._create_failed_result(0)
                    
                    start_time = max(self.current_time, self.machine_available_time[best_machine.id])
                    actual_proc_time = best_machine.get_processing_time(
                        process_id, process_time, self.allow_reconfiguration
                    )
                
                # Track setup/reconfiguration time
                setup_time_for_this_op = 0
                if self.allow_reconfiguration and process_id not in best_machine.native_processes:
                    if best_machine.current_process != process_id:
                        setup_time_for_this_op = best_machine.reconfiguration_time
                        total_setup_time += setup_time_for_this_op
                        num_reconfigurations += 1
                        machine_setup_time[best_machine.id] += setup_time_for_this_op
                
                # Track waiting time
                wait_time = start_time - self.current_time
                job.waited_time += wait_time
                total_wait_time += wait_time
                
                # Calculate idle time for this machine
                machine_idle = start_time - self.machine_available_time[best_machine.id]
                if machine_idle > 0:
                    machine_idle_time[best_machine.id] += machine_idle
                
                # Schedule on selected machine
                end_time = start_time + actual_proc_time
                
                # Track productive (busy) time (excluding setup)
                productive_time = actual_proc_time - setup_time_for_this_op
                machine_busy_time[best_machine.id] += productive_time
                
                self.machine_available_time[best_machine.id] = end_time
                self.current_time = end_time
                best_machine.current_process = process_id
                
                job.current_process_idx += 1
            
            job.completion_time = self.current_time
            job_completion_times.append(self.current_time - job_start_time)
            total_completion_time = self.current_time
            
            # Calculate tardiness
            if job.completion_time > job.due_date:
                tardiness = job.completion_time - job.due_date
                total_tardiness += tardiness
                tardy_jobs += 1
        
        # Calculate machine utilization metrics
        makespan = total_completion_time
        total_utilization = 0
        utilization_by_machine = {}
        active_machines = 0
        
        for machine in self.machines:
            if not machine.is_broken:
                active_machines += 1
                busy_time = machine_busy_time[machine.id]
                setup_time = machine_setup_time[machine.id]
                idle_time = machine_idle_time[machine.id]
                
                # Utilization = (Busy Time + Setup Time) / Makespan
                utilization = ((busy_time + setup_time) / makespan * 100) if makespan > 0 else 0
                utilization_by_machine[machine.id] = {
                    'busy_time': busy_time,
                    'setup_time': setup_time,
                    'idle_time': idle_time,
                    'utilization': utilization,
                    'productive_utilization': (busy_time / makespan * 100) if makespan > 0 else 0
                }
                total_utilization += utilization
        
        avg_utilization = total_utilization / active_machines if active_machines > 0 else 0
        
        # Calculate multi-objective performance
        avg_completion = np.mean(job_completion_times)
        avg_tardiness = total_tardiness / len(self.jobs)
        avg_setup = total_setup_time / len(self.jobs) if len(self.jobs) > 0 else 0
        
        # Objective function: weighted sum (lower is better)
        # Weight: 40% makespan, 30% tardiness, 20% setup time, 10% wait time
        objective_value = (0.4 * total_completion_time + 
                          0.3 * total_tardiness + 
                          0.2 * total_setup_time + 
                          0.1 * total_wait_time)
        
        return {
            'success': True,
            'total_time': total_completion_time,
            'avg_completion_time': avg_completion,
            'job_times': job_completion_times,
            'makespan': total_completion_time,
            'total_wait_time': total_wait_time,
            'avg_wait_time': total_wait_time / len(self.jobs) if len(self.jobs) > 0 else 0,
            'failed_negotiations': failed_negotiations,
            'total_tardiness': total_tardiness,
            'avg_tardiness': avg_tardiness,
            'tardy_jobs': tardy_jobs,
            'tardy_job_percentage': (tardy_jobs / len(self.jobs) * 100) if len(self.jobs) > 0 else 0,
            'total_setup_time': total_setup_time,
            'avg_setup_time': avg_setup,
            'num_reconfigurations': num_reconfigurations,
            'objective_value': objective_value,
            'avg_machine_utilization': avg_utilization,
            'utilization_by_machine': utilization_by_machine,
            'total_busy_time': sum(machine_busy_time.values()),
            'total_idle_time': sum(machine_idle_time.values()),
            'active_machines': active_machines
        }
    
    def _create_failed_result(self, failed_negotiations):
        """Helper to create failed result dictionary"""
        return {
            'success': False,
            'total_time': float('inf'),
            'avg_completion_time': float('inf'),
            'job_times': [],
            'makespan': float('inf'),
            'total_wait_time': float('inf'),
            'avg_wait_time': float('inf'),
            'failed_negotiations': failed_negotiations,
            'total_tardiness': float('inf'),
            'avg_tardiness': float('inf'),
            'tardy_jobs': 0,
            'tardy_job_percentage': 0,
            'total_setup_time': float('inf'),
            'avg_setup_time': float('inf'),
            'num_reconfigurations': 0,
            'objective_value': float('inf'),
            'avg_machine_utilization': 0,
            'utilization_by_machine': {},
            'total_busy_time': 0,
            'total_idle_time': 0,
            'active_machines': 0
        }

def generate_scenario(num_machines=8, num_jobs=50, num_process_types=6):
    """Generate a random production scenario"""
    machines = []
    
    for i in range(num_machines):
        # Each machine natively handles 2-3 random processes
        num_native = random.randint(2, 3)
        native = set(random.sample(range(num_process_types), num_native))
        # Can reconfigure for 2-3 additional processes
        remaining = list(set(range(num_process_types)) - native)
        num_reconfig = min(random.randint(2, 3), len(remaining))
        reconfigurable = set(random.sample(remaining, num_reconfig)) if remaining else set()
        
        machines.append(Machine(
            id=i,
            native_processes=native,
            reconfigurable_processes=reconfigurable,
            reconfiguration_time=random.uniform(3, 7),
            negotiation_flexibility=random.uniform(0.6, 1.0)
        ))
    
    jobs = []
    for i in range(num_jobs):
        # Each job needs 3-5 processes
        num_processes = random.randint(3, 5)
        processes = [random.randint(0, num_process_types-1) for _ in range(num_processes)]
        process_times = [random.uniform(5, 15) for _ in range(num_processes)]
        
        # Assign priority and due date
        priority = random.randint(1, 5)
        total_process_time = sum(process_times)
        due_date = total_process_time * random.uniform(2.0, 4.0)
        
        jobs.append(Job(
            id=i,
            processes=processes,
            process_times=process_times,
            priority=priority,
            due_date=due_date
        ))
    
    return machines, jobs

def analyze_negotiation_and_reconfigurability():
    """Main analysis function with negotiation"""
    print("=" * 80)
    print("MACHINE RECONFIGURABILITY & NEGOTIATION IMPACT ANALYSIS")
    print("Scenario: Job Shop with Machine Breakdowns")
    print("=" * 80)
    
    # Generate scenario with more machines and jobs
    machines, jobs = generate_scenario(num_machines=8, num_jobs=50, num_process_types=6)
    
    print(f"\n📊 Scenario Configuration:")
    print(f"   - Machines: {len(machines)}")
    print(f"   - Jobs: {len(jobs)}")
    print(f"   - Process Types: 6")
    
    print("\n🔧 Machine Capabilities (Sample - First 5 machines):")
    for m in machines[:5]:
        print(f"   Machine {m.id}:")
        print(f"      Native Processes: {sorted(m.native_processes)}")
        print(f"      Reconfigurable For: {sorted(m.reconfigurable_processes)}")
        print(f"      Reconfiguration Time: {m.reconfiguration_time:.1f}")
        print(f"      Negotiation Flexibility: {m.negotiation_flexibility:.2f}")
    print(f"   ... and {len(machines)-5} more machines")
    
    print("\n📋 Job Requirements (Sample - First 5 jobs):")
    for j in jobs[:5]:
        print(f"   Job {j.id}: Processes {j.processes} | Priority: {j.priority} | Due: {j.due_date:.1f}")
    print(f"   ... and {len(jobs)-5} more jobs")
    
    # Test all combinations
    results = {
        'no_breakdown': {},
        'with_breakdown': {}
    }
    
    configurations = [
        ("No Reconfig, No Negotiation", False, False),
        ("With Reconfig, No Negotiation", True, False),
        ("No Reconfig, With Negotiation", False, True),
        ("With Reconfig + Negotiation", True, True)
    ]
    
    # Scenario 1: No breakdown
    print("\n" + "=" * 80)
    print("SCENARIO 1: Normal Operation (No Breakdown)")
    print("=" * 80)
    
    for name, reconfig, negotiate in configurations:
        print(f"\nRunning: {name}...", end=" ")
        machines_copy = [Machine(m.id, m.native_processes.copy(), 
                                m.reconfigurable_processes.copy(), 
                                False, m.reconfiguration_time, m.current_process,
                                m.negotiation_flexibility) 
                        for m in machines]
        jobs_copy = [Job(j.id, j.processes.copy(), j.process_times.copy(),
                        j.priority, j.due_date) 
                    for j in jobs]
        
        scheduler = ProductionScheduler(machines_copy, jobs_copy, reconfig, negotiate)
        result = scheduler.simulate()
        
        results['no_breakdown'][name] = result
        
        if result['success']:
            print("✓ COMPLETED")
            print(f"   Makespan: {result['makespan']:.2f} | " +
                  f"Tardiness: {result['total_tardiness']:.2f} | " +
                  f"Setup Time: {result['total_setup_time']:.2f} | " +
                  f"Utilization: {result['avg_machine_utilization']:.1f}%")
        else:
            print("✗ FAILED")
    
    # Scenario 2: With breakdown
    print("\n" + "=" * 80)
    print("SCENARIO 2: Machine Breakdown (Machine 3 breaks down)")
    print("=" * 80)
    
    breakdown_machine = 3
    
    for name, reconfig, negotiate in configurations:
        print(f"\nRunning: {name}...", end=" ")
        machines_copy = [Machine(m.id, m.native_processes.copy(), 
                                m.reconfigurable_processes.copy(), 
                                False, m.reconfiguration_time, m.current_process,
                                m.negotiation_flexibility) 
                        for m in machines]
        jobs_copy = [Job(j.id, j.processes.copy(), j.process_times.copy(),
                        j.priority, j.due_date) 
                    for j in jobs]
        
        scheduler = ProductionScheduler(machines_copy, jobs_copy, reconfig, negotiate)
        result = scheduler.simulate(breakdown_machine_id=breakdown_machine)
        
        results['with_breakdown'][name] = result
        
        if result['success']:
            print("✓ COMPLETED")
            print(f"   Makespan: {result['makespan']:.2f} | " +
                  f"Tardiness: {result['total_tardiness']:.2f} | " +
                  f"Setup Time: {result['total_setup_time']:.2f} | " +
                  f"Utilization: {result['avg_machine_utilization']:.1f}%")
        else:
            print("✗ FAILED")
    
    # Comparative Analysis
    print("\n" + "=" * 80)
    print("COMPARATIVE ANALYSIS - ALL PERFORMANCE METRICS")
    print("=" * 80)
    
    print("\n📈 Normal Operation:")
    baseline_result = results['no_breakdown']['No Reconfig, No Negotiation']
    if baseline_result['success']:
        baseline = baseline_result['makespan']
        print(f"\n{'Configuration':<30} {'Makespan':<12} {'Tardiness':<12} {'Setup':<10} {'Util%':<10} {'Objective':<12}")
        print("-" * 90)
        for name, result in results['no_breakdown'].items():
            if result['success']:
                short_name = name.replace('With Reconfig + Negotiation', 'Combined').replace('With Reconfig, No Negotiation', 'Reconfig Only').replace('No Reconfig, With Negotiation', 'Negotiation Only').replace('No Reconfig, No Negotiation', 'Baseline')
                print(f"{short_name:<30} {result['makespan']:<12.2f} {result['total_tardiness']:<12.2f} {result['total_setup_time']:<10.2f} {result['avg_machine_utilization']:<10.1f} {result['objective_value']:<12.2f}")
            else:
                print(f"{name:<30} {'FAILED':<12}")
    else:
        print("   Baseline failed - cannot compute comparisons")
    
    print(f"\n🔥 Breakdown Scenario (Machine {breakdown_machine} down):")
    print(f"\n{'Configuration':<30} {'Makespan':<12} {'Tardiness':<12} {'Setup':<10} {'Util%':<10} {'Objective':<12} {'Status':<10}")
    print("-" * 100)
    for name, result in results['with_breakdown'].items():
        short_name = name.replace('With Reconfig + Negotiation', 'Combined').replace('With Reconfig, No Negotiation', 'Reconfig Only').replace('No Reconfig, With Negotiation', 'Negotiation Only').replace('No Reconfig, No Negotiation', 'Baseline')
        if result['success']:
            print(f"{short_name:<30} {result['makespan']:<12.2f} {result['total_tardiness']:<12.2f} {result['total_setup_time']:<10.2f} {result['avg_machine_utilization']:<10.1f} {result['objective_value']:<12.2f} {'✓':<10}")
        else:
            print(f"{short_name:<30} {'FAILED':<12} {'-':<12} {'-':<10} {'-':<10} {'-':<12} {'✗':<10}")
    
    # Machine utilization breakdown for breakdown scenario
    print("\n📊 Machine Utilization Breakdown (Breakdown Scenario):")
    for name, result in results['with_breakdown'].items():
        if result['success']:
            short_name = name.replace('With Reconfig + Negotiation', 'Combined').replace('With Reconfig, No Negotiation', 'Reconfig Only')
            print(f"\n  {short_name}:")
            print(f"    Active Machines: {result['active_machines']}/8")
            print(f"    Average Utilization: {result['avg_machine_utilization']:.1f}%")
            if result['utilization_by_machine']:
                print(f"    Per-Machine Utilization:")
                for m_id, util_data in sorted(result['utilization_by_machine'].items()):
                    print(f"      Machine {m_id}: {util_data['utilization']:.1f}% " +
                          f"(Productive: {util_data['productive_utilization']:.1f}%, " +
                          f"Setup: {util_data['setup_time']:.1f})")
    
    # Find best configuration
    successful_configs = [(name, r) for name, r in results['with_breakdown'].items() if r['success']]
    if successful_configs:
        print("\n🏆 BEST CONFIGURATION (Breakdown Scenario - Based on Objective Function):")
        best_config = min(successful_configs, key=lambda x: x[1]['objective_value'])
        print(f"   Configuration: {best_config[0]}")
        print(f"   Makespan: {best_config[1]['makespan']:.2f}")
        print(f"   Total Tardiness: {best_config[1]['total_tardiness']:.2f}")
        print(f"   Setup Time: {best_config[1]['total_setup_time']:.2f}")
        print(f"   Machine Utilization: {best_config[1]['avg_machine_utilization']:.1f}%")
        print(f"   Reconfigurations: {best_config[1]['num_reconfigurations']}")
        print(f"   Tardy Jobs: {best_config[1]['tardy_jobs']} ({best_config[1]['tardy_job_percentage']:.1f}%)")
        print(f"   Objective Value: {best_config[1]['objective_value']:.2f}")
    else:
        print("\n⚠️  All configurations failed in breakdown scenario!")
    
    # Visualization
    create_visualization(results)
    
    print("\n" + "=" * 80)
    print("KEY FINDINGS - MULTI-OBJECTIVE PERFORMANCE")
    print("=" * 80)
    print("✓ Makespan: Total time to complete all jobs")
    print("✓ Tardiness: Delay beyond due dates (on-time delivery performance)")
    print("✓ Setup Time: Reconfiguration overhead (flexibility cost)")
    print("✓ Objective Function: Weighted combination (40% makespan, 30% tardiness, 20% setup, 10% wait)")
    print("\n✓ Reconfigurability: Provides flexibility but adds setup time")
    print("✓ Negotiation: Optimizes allocation, reduces tardiness")
    print("✓ Combined Approach: Best overall objective value")
    print("✓ Trade-off: Setup time vs breakdown resilience")
    print("=" * 80)

def create_figure1_main_comparison(results, config_names):
    """Figure 1: Main Comparison with Machine Utilization Focus"""
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    colors_dict = {
        'No Reconfig, No Negotiation': '#e74c3c',
        'With Reconfig, No Negotiation': '#f39c12',
        'No Reconfig, With Negotiation': '#3498db',
        'With Reconfig + Negotiation': '#27ae60'
    }
    
    short_names = ['Baseline', 'Reconfig\nOnly', 'Negotiation\nOnly', 'Combined']
    bar_colors = [colors_dict[name] for name in config_names]
    
    x_pos = np.arange(len(config_names))
    width = 0.35
    
    # Chart 1: Makespan Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    normal_makespan = [results['no_breakdown'][name]['makespan'] if results['no_breakdown'][name]['success'] else 0 
                       for name in config_names]
    breakdown_makespan = [results['with_breakdown'][name]['makespan'] if results['with_breakdown'][name]['success'] else 0 
                          for name in config_names]
    
    bars1 = ax1.bar(x_pos - width/2, normal_makespan, width, label='Normal', 
                    color='#3498db', alpha=0.85, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x_pos + width/2, breakdown_makespan, width, label='Breakdown', 
                    color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.5)
    
    ax1.set_ylabel('Makespan (time units)', fontsize=11, fontweight='bold')
    ax1.set_title('(a) Makespan Comparison', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(short_names, fontsize=10)
    ax1.legend(fontsize=10, framealpha=0.9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add values and FAILED markers
    for i, (bar, name) in enumerate(zip(bars1, config_names)):
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + 50,
                    f'{height:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    for i, (bar, name) in enumerate(zip(bars2, config_names)):
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + 50,
                    f'{height:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        else:
            if not results['with_breakdown'][name]['success']:
                ax1.text(bar.get_x() + bar.get_width()/2., 100,
                        'FAILED', ha='center', va='center', fontsize=9, fontweight='bold',
                        color='white', bbox=dict(boxstyle='round,pad=0.4', facecolor='#c0392b', 
                                                edgecolor='black', linewidth=2))
    
    # Chart 2: Machine Utilization - THE KEY CHART
    ax2 = fig.add_subplot(gs[0, 1])
    normal_util = [results['no_breakdown'][name]['avg_machine_utilization'] if results['no_breakdown'][name]['success'] else 0 
                   for name in config_names]
    breakdown_util = [results['with_breakdown'][name]['avg_machine_utilization'] if results['with_breakdown'][name]['success'] else 0 
                      for name in config_names]
    
    bars_util1 = ax2.bar(x_pos - width/2, normal_util, width, label='Normal (8 machines)', 
                         color='#2ecc71', alpha=0.85, edgecolor='black', linewidth=1.5)
    bars_util2 = ax2.bar(x_pos + width/2, breakdown_util, width, label='Breakdown (7 machines)', 
                         color='#e67e22', alpha=0.85, edgecolor='black', linewidth=1.5)
    
    ax2.set_ylabel('Machine Utilization (%)', fontsize=11, fontweight='bold')
    ax2.set_title('(b) Machine Utilization Impact\n(Higher is Better)', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(short_names, fontsize=10)
    ax2.legend(fontsize=9, framealpha=0.9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim([0, 75])
    ax2.axhline(y=60, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='Good Utilization')
    
    # Add values
    for i, (bar, name) in enumerate(zip(bars_util1, config_names)):
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color='green')
    
    for i, (bar, name) in enumerate(zip(bars_util2, config_names)):
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color='orange')
        else:
            if not results['with_breakdown'][name]['success']:
                ax2.text(bar.get_x() + bar.get_width()/2., 5,
                        'N/A', ha='center', va='center', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#95a5a6', alpha=0.7))
    
    # Chart 3: Utilization Change Analysis
    ax3 = fig.add_subplot(gs[0, 2])
    util_change = []
    util_colors = []
    for name in config_names:
        normal = results['no_breakdown'][name]['avg_machine_utilization'] if results['no_breakdown'][name]['success'] else 0
        breakdown = results['with_breakdown'][name]['avg_machine_utilization'] if results['with_breakdown'][name]['success'] else 0
        
        if normal > 0 and breakdown > 0:
            change = breakdown - normal
            util_change.append(change)
            util_colors.append('#27ae60' if change >= -5 else '#e74c3c')
        else:
            util_change.append(0)
            util_colors.append('#95a5a6')
    
    bars_change = ax3.bar(range(len(config_names)), util_change, color=util_colors,
                          alpha=0.85, edgecolor='black', linewidth=1.5)
    ax3.set_xticks(range(len(config_names)))
    ax3.set_xticklabels(short_names, fontsize=10)
    ax3.set_ylabel('Utilization Change (%)', fontsize=11, fontweight='bold')
    ax3.set_title('(c) Breakdown Impact on Utilization\n(Closer to 0 = Better Resilience)', 
                  fontsize=13, fontweight='bold', pad=10)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=2)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, (bar, val) in enumerate(zip(bars_change, util_change)):
        if val != 0:
            ax3.text(bar.get_x() + bar.get_width()/2., val + (0.5 if val > 0 else -0.5),
                    f'{val:+.1f}%', ha='center', va='bottom' if val > 0 else 'top',
                    fontweight='bold', fontsize=10)
        else:
            if not results['with_breakdown'][config_names[i]]['success']:
                ax3.text(bar.get_x() + bar.get_width()/2., -2,
                        'System\nFailure', ha='center', va='center', fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#c0392b', alpha=0.9))
    
    # Chart 4: Total Tardiness
    ax4 = fig.add_subplot(gs[1, 0])
    normal_tardiness = [results['no_breakdown'][name]['total_tardiness'] if results['no_breakdown'][name]['success'] else 0 
                        for name in config_names]
    breakdown_tardiness = [results['with_breakdown'][name]['total_tardiness'] if results['with_breakdown'][name]['success'] else 0 
                           for name in config_names]
    
    bars4 = ax4.bar(x_pos - width/2, normal_tardiness, width, label='Normal', 
                    color='#9b59b6', alpha=0.85, edgecolor='black', linewidth=1.5)
    bars5 = ax4.bar(x_pos + width/2, breakdown_tardiness, width, label='Breakdown', 
                    color='#e67e22', alpha=0.85, edgecolor='black', linewidth=1.5)
    
    ax4.set_ylabel('Total Tardiness (time units)', fontsize=11, fontweight='bold')
    ax4.set_title('(d) Tardiness Comparison', fontsize=13, fontweight='bold', pad=10)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(short_names, fontsize=10)
    ax4.legend(fontsize=9)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bars in [bars4, bars5]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax4.text(bar.get_x() + bar.get_width()/2., height + 1000,
                        f'{height:.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Chart 5: Setup Time
    ax5 = fig.add_subplot(gs[1, 1])
    normal_setup = [results['no_breakdown'][name]['total_setup_time'] if results['no_breakdown'][name]['success'] else 0 
                    for name in config_names]
    breakdown_setup = [results['with_breakdown'][name]['total_setup_time'] if results['with_breakdown'][name]['success'] else 0 
                       for name in config_names]
    
    bars6 = ax5.bar(x_pos - width/2, normal_setup, width, label='Normal', 
                    color='#16a085', alpha=0.85, edgecolor='black', linewidth=1.5)
    bars7 = ax5.bar(x_pos + width/2, breakdown_setup, width, label='Breakdown', 
                    color='#d35400', alpha=0.85, edgecolor='black', linewidth=1.5)
    
    ax5.set_ylabel('Total Setup Time (time units)', fontsize=11, fontweight='bold')
    ax5.set_title('(e) Reconfiguration Overhead', fontsize=13, fontweight='bold', pad=10)
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(short_names, fontsize=10)
    ax5.legend(fontsize=9)
    ax5.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bars in [bars6, bars7]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax5.text(bar.get_x() + bar.get_width()/2., height + 10,
                        f'{height:.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Chart 6: Number of Reconfigurations
    ax6 = fig.add_subplot(gs[1, 2])
    normal_reconfig = [results['no_breakdown'][name]['num_reconfigurations'] if results['no_breakdown'][name]['success'] else 0 
                       for name in config_names]
    breakdown_reconfig = [results['with_breakdown'][name]['num_reconfigurations'] if results['with_breakdown'][name]['success'] else 0 
                          for name in config_names]
    
    bars8 = ax6.bar(x_pos - width/2, normal_reconfig, width, label='Normal', 
                    color='#f39c12', alpha=0.85, edgecolor='black', linewidth=1.5)
    bars9 = ax6.bar(x_pos + width/2, breakdown_reconfig, width, label='Breakdown', 
                     color='#8e44ad', alpha=0.85, edgecolor='black', linewidth=1.5)
    
    ax6.set_ylabel('Number of Reconfigurations', fontsize=11, fontweight='bold')
    ax6.set_title('(f) Reconfiguration Frequency', fontsize=13, fontweight='bold', pad=10)
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(short_names, fontsize=10)
    ax6.legend(fontsize=9)
    ax6.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bars in [bars8, bars9]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax6.text(bar.get_x() + bar.get_width()/2., height + 2,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Chart 7: Objective Function (spans bottom)
    ax7 = fig.add_subplot(gs[2, :])
    normal_obj = [results['no_breakdown'][name]['objective_value'] if results['no_breakdown'][name]['success'] else 0 
                  for name in config_names]
    breakdown_obj = [results['with_breakdown'][name]['objective_value'] if results['with_breakdown'][name]['success'] else 0 
                     for name in config_names]
    
    bars10 = ax7.bar(x_pos - width/2, normal_obj, width, label='Normal Operation', 
                     color='#27ae60', alpha=0.85, edgecolor='black', linewidth=1.5)
    bars11 = ax7.bar(x_pos + width/2, breakdown_obj, width, label='Breakdown Scenario', 
                     color='#c0392b', alpha=0.85, edgecolor='black', linewidth=1.5)
    
    ax7.set_ylabel('Objective Function Value\n(Lower is Better)', fontsize=13, fontweight='bold')
    ax7.set_title('(g) Multi-Objective Performance: 40% Makespan + 30% Tardiness + 20% Setup + 10% Wait Time', 
                  fontsize=14, fontweight='bold', pad=15)
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(short_names, fontsize=12)
    ax7.legend(fontsize=12, loc='upper left', framealpha=0.9)
    ax7.grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, (bar, name) in enumerate(zip(bars10, config_names)):
        height = bar.get_height()
        if height > 0:
            ax7.text(bar.get_x() + bar.get_width()/2., height + 300,
                    f'{height:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    for i, (bar, name) in enumerate(zip(bars11, config_names)):
        height = bar.get_height()
        if height > 0:
            ax7.text(bar.get_x() + bar.get_width()/2., height + 300,
                    f'{height:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        else:
            if not results['with_breakdown'][name]['success']:
                ax7.text(bar.get_x() + bar.get_width()/2., 500,
                        'System Failed', ha='center', va='center', fontsize=10, fontweight='bold',
                        color='white', bbox=dict(boxstyle='round,pad=0.5', facecolor='#c0392b', 
                                                edgecolor='black', linewidth=2))
    
    plt.tight_layout()
    plt.savefig('Figure1_Comprehensive_Metrics.png', dpi=400, bbox_inches='tight', facecolor='white')
    plt.savefig('Figure1_Comprehensive_Metrics.pdf', bbox_inches='tight', facecolor='white')
    print("\n✓ Figure 1 saved: Figure1_Comprehensive_Metrics.png & .pdf")
    plt.close()
    
    colors_dict = {
        'No Reconfig, No Negotiation': '#e74c3c',
        'With Reconfig, No Negotiation': '#f39c12',
        'No Reconfig, With Negotiation': '#3498db',
        'With Reconfig + Negotiation': '#27ae60'
    }
    
    short_names = ['Baseline\n(No R, No N)', 'Reconfig\nOnly', 'Negotiation\nOnly', 'Combined\n(R + N)']
    
    # LEFT: Side-by-side comparison
    x_pos = np.arange(len(config_names))
    width = 0.38
    
    normal_times = [results['no_breakdown'][name]['makespan'] if results['no_breakdown'][name]['success'] else 0 
                    for name in config_names]
    breakdown_times = [results['with_breakdown'][name]['makespan'] if results['with_breakdown'][name]['success'] else 0 
                       for name in config_names]
    
    bars1 = ax1.bar(x_pos - width/2, normal_times, width, 
                    label='Normal Operation', color='#3498db', alpha=0.85, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x_pos + width/2, breakdown_times, width, 
                    label='Machine Breakdown', color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.5)
    
    ax1.set_ylabel('Makespan (time units)', fontsize=13, fontweight='bold')
    ax1.set_title('(a) Makespan Comparison:\nNormal vs Breakdown Scenarios', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(short_names, fontsize=10)
    ax1.legend(loc='upper left', fontsize=11, frameon=True, shadow=True, fancybox=True)
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
    ax1.set_ylim(0, max(max(normal_times), max(breakdown_times)) * 1.15)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height + 30,
                        f'{height:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
            else:
                if not results['with_breakdown'][config_names[i]]['success']:
                    ax1.text(bar.get_x() + bar.get_width()/2., 150,
                            'FAILED', ha='center', va='center', fontsize=10, fontweight='bold',
                            color='white', bbox=dict(boxstyle='round,pad=0.5', facecolor='#c0392b', 
                                                    edgecolor='black', linewidth=2))
    
    # RIGHT: Breakdown Impact
    impact_increase = []
    impact_colors = []
    for name in config_names:
        normal = results['no_breakdown'][name]['makespan'] if results['no_breakdown'][name]['success'] else 0
        breakdown = results['with_breakdown'][name]['makespan'] if results['with_breakdown'][name]['success'] else float('inf')
        if normal > 0 and breakdown != float('inf'):
            increase = ((breakdown - normal) / normal) * 100
            impact_increase.append(increase)
            impact_colors.append(colors_dict[name])
        else:
            impact_increase.append(0)
            impact_colors.append('#95a5a6')
    
    bars3 = ax2.bar(range(len(config_names)), impact_increase, color=impact_colors,
                    alpha=0.85, edgecolor='black', linewidth=1.5)
    ax2.set_xticks(range(len(config_names)))
    ax2.set_xticklabels(short_names, fontsize=10)
    ax2.set_ylabel('Makespan Increase (%)', fontsize=13, fontweight='bold')
    ax2.set_title('(b) Breakdown Impact Analysis\n(Lower = More Resilient)', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=2)
    
    for i, (bar, val) in enumerate(zip(bars3, impact_increase)):
        if val > 0 and val < 200:
            ax2.text(bar.get_x() + bar.get_width()/2., val + 0.5,
                    f'+{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
        elif not results['with_breakdown'][config_names[i]]['success']:
            ax2.text(bar.get_x() + bar.get_width()/2., 2,
                    'System\nFailure', ha='center', va='bottom', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#c0392b', 
                             edgecolor='black', linewidth=2, alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('Figure1_Main_Comparison.png', dpi=400, bbox_inches='tight', facecolor='white')
    plt.savefig('Figure1_Main_Comparison.pdf', bbox_inches='tight', facecolor='white')
    print("\n✓ Figure 1 saved: Figure1_Main_Comparison.png & .pdf")
    plt.close()

def create_figure2_breakdown_resilience(results, config_names):
    """Figure 2: Breakdown Resilience & Success Rate"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    colors_dict = {
        'No Reconfig, No Negotiation': '#e74c3c',
        'With Reconfig, No Negotiation': '#f39c12',
        'No Reconfig, With Negotiation': '#3498db',
        'With Reconfig + Negotiation': '#27ae60'
    }
    
    short_names = ['Baseline', 'Reconfig\nOnly', 'Negotiation\nOnly', 'Combined']
    bar_colors = [colors_dict[name] for name in config_names]
    
    # LEFT: Success Rate
    success_rates = []
    for name in config_names:
        normal_success = 1 if results['no_breakdown'][name]['success'] else 0
        breakdown_success = 1 if results['with_breakdown'][name]['success'] else 0
        success_rate = (normal_success + breakdown_success) / 2 * 100
        success_rates.append(success_rate)
    
    bars1 = ax1.bar(range(len(config_names)), success_rates, color=bar_colors,
                    alpha=0.85, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(len(config_names)))
    ax1.set_xticklabels(short_names, fontsize=11)
    ax1.set_ylabel('Success Rate (%)', fontsize=13, fontweight='bold')
    ax1.set_title('(a) System Reliability\n(Both Scenarios Combined)', fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylim([0, 110])
    ax1.axhline(y=100, color='#27ae60', linestyle='--', linewidth=2.5, alpha=0.6, label='Perfect Reliability')
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
    ax1.legend(loc='lower right', fontsize=10)
    
    for bar, val in zip(bars1, success_rates):
        color = 'green' if val == 100 else 'red'
        ax1.text(bar.get_x() + bar.get_width()/2., val + 2,
                f'{val:.0f}%', ha='center', va='bottom', fontweight='bold', fontsize=12, color=color)
    
    # RIGHT: Performance Improvement
    baseline_result = results['no_breakdown']['No Reconfig, No Negotiation']
    improvements = []
    
    if baseline_result['success']:
        baseline_normal = baseline_result['makespan']
        for name in config_names:
            result = results['no_breakdown'][name]
            if result['success']:
                improvement = ((baseline_normal - result['makespan']) / baseline_normal) * 100
                improvements.append(improvement)
            else:
                improvements.append(-100)
    else:
        improvements = [0, 0, 0, 0]
    
    bars2 = ax2.barh(range(len(config_names)), improvements, color=bar_colors,
                     alpha=0.85, edgecolor='black', linewidth=1.5, height=0.6)
    ax2.set_yticks(range(len(config_names)))
    ax2.set_yticklabels(short_names, fontsize=11)
    ax2.set_xlabel('Performance Improvement (%)', fontsize=13, fontweight='bold')
    ax2.set_title('(b) Makespan Improvement\n(vs Baseline - Normal Operation)', fontsize=14, fontweight='bold', pad=15)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=2)
    ax2.grid(axis='x', alpha=0.3, linestyle='--', linewidth=1)
    
    for i, (bar, val) in enumerate(zip(bars2, improvements)):
        x_pos = val + (1 if val > 0 else -1)
        ax2.text(x_pos, i, f'{val:+.1f}%', 
                va='center', ha='left' if val > 0 else 'right', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('Figure2_Reliability_Performance.png', dpi=400, bbox_inches='tight', facecolor='white')
    plt.savefig('Figure2_Reliability_Performance.pdf', bbox_inches='tight', facecolor='white')
    print("✓ Figure 2 saved: Figure2_Reliability_Performance.png & .pdf")
    plt.close()

def create_figure3_performance_metrics(results, config_names):
    """Figure 3: Detailed Performance Metrics"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    colors_dict = {
        'No Reconfig, No Negotiation': '#e74c3c',
        'With Reconfig, No Negotiation': '#f39c12',
        'No Reconfig, With Negotiation': '#3498db',
        'With Reconfig + Negotiation': '#27ae60'
    }
    
    short_names = ['Baseline', 'Reconfig\nOnly', 'Negotiation\nOnly', 'Combined']
    bar_colors = [colors_dict[name] for name in config_names]
    
    # Chart 1: Wait Time Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    wait_times_normal = [results['no_breakdown'][name]['avg_wait_time'] if results['no_breakdown'][name]['success'] else 0 
                        for name in config_names]
    wait_times_breakdown = [results['with_breakdown'][name]['avg_wait_time'] if results['with_breakdown'][name]['success'] else 0 
                           for name in config_names]
    
    x_pos = np.arange(len(config_names))
    width = 0.38
    bars1 = ax1.bar(x_pos - width/2, wait_times_normal, width, 
                    label='Normal', color='#2ecc71', alpha=0.85, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x_pos + width/2, wait_times_breakdown, width, 
                    label='Breakdown', color='#e67e22', alpha=0.85, edgecolor='black', linewidth=1.5)
    
    ax1.set_ylabel('Avg Wait Time (time units)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Average Wait Time Analysis', fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(short_names, fontsize=10)
    ax1.legend(loc='upper right', fontsize=11, frameon=True, shadow=True)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Chart 2: Completion Time Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    completion_normal = [results['no_breakdown'][name]['avg_completion_time'] if results['no_breakdown'][name]['success'] else 0 
                        for name in config_names]
    completion_breakdown = [results['with_breakdown'][name]['avg_completion_time'] if results['with_breakdown'][name]['success'] else 0 
                           for name in config_names]
    
    bars3 = ax2.bar(x_pos - width/2, completion_normal, width, 
                    label='Normal', color='#9b59b6', alpha=0.85, edgecolor='black', linewidth=1.5)
    bars4 = ax2.bar(x_pos + width/2, completion_breakdown, width, 
                    label='Breakdown', color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.5)
    
    ax2.set_ylabel('Avg Completion Time (time units)', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Job Completion Time Analysis', fontsize=13, fontweight='bold', pad=15)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(short_names, fontsize=10)
    ax2.legend(loc='upper left', fontsize=11, frameon=True, shadow=True)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Chart 3: Summary Table
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    
    table_data = []
    for i, name in enumerate(config_names):
        normal_result = results['no_breakdown'][name]
        breakdown_result = results['with_breakdown'][name]
        
        row = [
            short_names[i],
            f"{normal_result['makespan']:.0f}" if normal_result['success'] else "FAIL",
            f"{breakdown_result['makespan']:.0f}" if breakdown_result['success'] else "FAIL",
            f"{normal_result['avg_wait_time']:.2f}" if normal_result['success'] else "-",
            f"{breakdown_result['avg_wait_time']:.2f}" if breakdown_result['success'] else "-",
            "✓" if breakdown_result['success'] else "✗"
        ]
        table_data.append(row)
    
    table = ax3.table(cellText=table_data,
                     colLabels=['Configuration', 'Normal\nMakespan', 'Breakdown\nMakespan', 
                               'Normal\nWait Time', 'Breakdown\nWait Time', 'Resilient'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0.05, 0.1, 0.9, 0.8])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3)
    
    # Style the table
    for i in range(len(config_names) + 1):
        for j in range(6):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#34495e')
                cell.set_text_props(weight='bold', color='white', fontsize=12)
            else:
                cell.set_facecolor(bar_colors[i-1] if i > 0 else 'white')
                cell.set_alpha(0.3)
                cell.set_edgecolor('black')
                cell.set_linewidth(1.5)
                # Highlight resilient column
                if j == 5:
                    text = cell.get_text().get_text()
                    if text == "✓":
                        cell.get_text().set_color('green')
                        cell.get_text().set_fontsize(16)
                        cell.get_text().set_weight('bold')
                    elif text == "✗":
                        cell.get_text().set_color('red')
                        cell.get_text().set_fontsize(16)
                        cell.get_text().set_weight('bold')
    
    ax3.set_title('(c) Comprehensive Performance Summary', fontsize=14, fontweight='bold', y=0.95)
    
    plt.savefig('Figure3_Detailed_Metrics.png', dpi=400, bbox_inches='tight', facecolor='white')
    plt.savefig('Figure3_Detailed_Metrics.pdf', bbox_inches='tight', facecolor='white')
    print("✓ Figure 3 saved: Figure3_Detailed_Metrics.png & .pdf")
    plt.close()

def create_figure4_conceptual_framework():
    """Figure 4: Conceptual Framework"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    fig.suptitle('Conceptual Framework: Manufacturing System Evolution with Reconfigurability and Negotiation',
                 fontsize=17, fontweight='bold', y=0.96)
    
    # Stage 1: Traditional System (Baseline)
    ax.add_patch(plt.Rectangle((0.5, 5), 2.2, 3.5, facecolor='#e74c3c', alpha=0.25, 
                               edgecolor='#c0392b', linewidth=3))
    ax.text(1.6, 7.8, 'TRADITIONAL\nSYSTEM', ha='center', va='center', 
            fontsize=13, fontweight='bold', color='#c0392b')
    ax.text(1.6, 7.0, 'Characteristics:', ha='center', va='center', 
            fontsize=11, fontweight='bold', style='italic')
    ax.text(1.6, 6.5, '• Fixed machine\n  capabilities', ha='center', va='center', fontsize=10)
    ax.text(1.6, 5.9, '• No negotiation', ha='center', va='center', fontsize=10)
    ax.text(1.6, 5.4, '• Breakdown\n  vulnerable', ha='center', va='center', fontsize=10)
    
    # Arrow 1
    ax.annotate('', xy=(3.0, 6.8), xytext=(2.8, 6.8),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    ax.text(2.9, 7.2, 'Add\nReconfig', ha='center', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Stage 2: Reconfigurable System
    ax.add_patch(plt.Rectangle((3.2, 5), 2.2, 3.5, facecolor='#f39c12', alpha=0.25, 
                               edgecolor='#e67e22', linewidth=3))
    ax.text(4.3, 7.8, 'RECONFIGURABLE\nSYSTEM', ha='center', va='center', 
            fontsize=13, fontweight='bold', color='#d68910')
    ax.text(4.3, 7.0, 'Improvements:', ha='center', va='center', 
            fontsize=11, fontweight='bold', style='italic')
    ax.text(4.3, 6.5, '• Process\n  flexibility', ha='center', va='center', fontsize=10)
    ax.text(4.3, 5.9, '• Breakdown\n  resilience', ha='center', va='center', fontsize=10)
    ax.text(4.3, 5.4, '• Higher\n  utilization', ha='center', va='center', fontsize=10)
    
    # Arrow 2
    ax.annotate('', xy=(5.7, 6.8), xytext=(5.5, 6.8),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    ax.text(5.6, 7.2, 'Add\nNegotiation', ha='center', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Stage 3: Negotiation System
    ax.add_patch(plt.Rectangle((5.9, 5), 2.2, 3.5, facecolor='#3498db', alpha=0.25, 
                               edgecolor='#2980b9', linewidth=3))
    ax.text(7.0, 7.8, 'NEGOTIATION\nSYSTEM', ha='center', va='center', 
            fontsize=13, fontweight='bold', color='#21618c')
    ax.text(7.0, 7.0, 'Additional:', ha='center', va='center', 
            fontsize=11, fontweight='bold', style='italic')
    ax.text(7.0, 6.5, '• Smart job\n  allocation', ha='center', va='center', fontsize=10)
    ax.text(7.0, 5.9, '• Priority-based\n  scheduling', ha='center', va='center', fontsize=10)
    ax.text(7.0, 5.4, '• Reduced\n  wait times', ha='center', va='center', fontsize=10)
    
    # Arrow 3
    ax.annotate('', xy=(8.4, 6.8), xytext=(8.2, 6.8),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    ax.text(8.3, 7.2, 'Combine\nBoth', ha='center', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Stage 4: Combined System (Optimal)
    ax.add_patch(plt.Rectangle((8.6, 5), 2.2, 3.5, facecolor='#27ae60', alpha=0.25, 
                               edgecolor='#229954', linewidth=3))
    ax.text(9.7, 7.8, 'INTEGRATED\nSYSTEM ★', ha='center', va='center', 
            fontsize=13, fontweight='bold', color='#186a3b')
    ax.text(9.7, 7.0, 'Optimal Result:', ha='center', va='center', 
            fontsize=11, fontweight='bold', style='italic')
    ax.text(9.7, 6.5, '✓ Best makespan', ha='center', va='center', fontsize=10, color='green')
    ax.text(9.7, 5.9, '✓ Highest\n   reliability', ha='center', va='center', fontsize=10, color='green')
    ax.text(9.7, 5.3, '✓ Maximum\n   resilience', ha='center', va='center', fontsize=10, color='green')
    
    # Bottom comparison boxes
    y_bottom = 3.5
    
    # Breakdown vulnerability comparison
    ax.text(6.0, y_bottom + 0.5, 'Machine Breakdown Resilience:', ha='center', 
            fontsize=12, fontweight='bold', style='italic')
    
    vulnerabilities = ['High\nVulnerability', 'Medium\nVulnerability', 'Low\nVulnerability', 'Maximum\nResilience']
    colors_vuln = ['#e74c3c', '#f39c12', '#3498db', '#27ae60']
    x_starts = [0.8, 3.5, 6.2, 8.9]
    
    for i, (vuln, color, x_start) in enumerate(zip(vulnerabilities, colors_vuln, x_starts)):
        intensity = 1.0 - (i * 0.2)
        ax.add_patch(plt.Rectangle((x_start, y_bottom - 1.2), 1.8, 0.8, 
                                   facecolor=color, alpha=0.4, edgecolor='black', linewidth=2))
        ax.text(x_start + 0.9, y_bottom - 0.8, vuln, ha='center', va='center', 
                fontsize=9, fontweight='bold')
    
    # Performance metrics comparison
    y_metric = 1.0
    metrics = ['Baseline\n(Lowest)', 'Improved', 'Enhanced', 'Optimal\n(Highest)']
    
    ax.text(6.0, y_metric + 0.4, 'Overall System Performance:', ha='center', 
            fontsize=12, fontweight='bold', style='italic')
    
    for i, (metric, color, x_start) in enumerate(zip(metrics, colors_vuln, x_starts)):
        bar_height = 0.2 + (i * 0.15)
        ax.add_patch(plt.Rectangle((x_start + 0.3, y_metric - 0.8), 1.2, bar_height, 
                                   facecolor=color, alpha=0.6, edgecolor='black', linewidth=2))
        ax.text(x_start + 0.9, y_metric - 1.1, metric, ha='center', va='top', 
                fontsize=8, fontweight='bold')
    
    # Legend
    legend_y = 0.2
    ax.text(1.0, legend_y, 'R = Reconfigurability', fontsize=10, style='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f39c12', alpha=0.3, edgecolor='black'))
    ax.text(3.5, legend_y, 'N = Negotiation', fontsize=10, style='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#3498db', alpha=0.3, edgecolor='black'))
    ax.text(6.0, legend_y, '★ = Recommended Configuration', fontsize=10, style='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#27ae60', alpha=0.3, edgecolor='black'))
    
    plt.tight_layout()
    plt.savefig('Figure4_Conceptual_Framework.png', dpi=400, bbox_inches='tight', facecolor='white')
    plt.savefig('Figure4_Conceptual_Framework.pdf', bbox_inches='tight', facecolor='white')
    print("✓ Figure 4 saved: Figure4_Conceptual_Framework.png & .pdf")
    plt.close()

def create_visualization(results):
    """Create publication-ready multiple focused figures"""
    
    # Set publication-quality style
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.linewidth'] = 1.5
    
    config_names = list(results['no_breakdown'].keys())
    
    # Create all figures
    create_figure1_main_comparison(results, config_names)
    create_figure2_breakdown_resilience(results, config_names)
    create_figure3_performance_metrics(results, config_names)
    create_figure4_conceptual_framework()
    
    print("\n" + "="*80)
    print("📊 ALL PUBLICATION-READY FIGURES GENERATED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated files:")
    print("  • Figure1_Main_Comparison.png & .pdf")
    print("  • Figure2_Reliability_Performance.png & .pdf")
    print("  • Figure3_Detailed_Metrics.png & .pdf")
    print("  • Figure4_Conceptual_Framework.png & .pdf")
    print("="*80)
    
    plt.show()
    
    # Create main figure with better layout
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3, 
                          left=0.08, right=0.95, top=0.94, bottom=0.06)
    
    # Main title
    fig.suptitle('Impact of Reconfigurability and Negotiation on Manufacturing System Performance\n' + 
                 'Under Normal and Breakdown Conditions',
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Color scheme
    colors_dict = {
        'No Reconfig, No Negotiation': '#e74c3c',
        'With Reconfig, No Negotiation': '#f39c12',
        'No Reconfig, With Negotiation': '#3498db',
        'With Reconfig + Negotiation': '#27ae60'
    }
    
    # ===== CHART 1: Comparative Makespan (Side by Side) =====
    ax1 = fig.add_subplot(gs[0, :])
    
    x_pos = np.arange(len(config_names))
    width = 0.35
    
    normal_times = [results['no_breakdown'][name]['makespan'] if results['no_breakdown'][name]['success'] else 0 
                    for name in config_names]
    breakdown_times = [results['with_breakdown'][name]['makespan'] if results['with_breakdown'][name]['success'] else 0 
                       for name in config_names]
    
    bars1_normal = ax1.bar(x_pos - width/2, normal_times, width, 
                          label='Normal Operation', color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars1_breakdown = ax1.bar(x_pos + width/2, breakdown_times, width, 
                             label='Machine Breakdown', color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax1.set_ylabel('Makespan (time units)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Makespan Comparison: Normal vs Breakdown Scenarios', fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(['Baseline\n(No R, No N)', 'Reconfig\nOnly', 'Negotiation\nOnly', 'Combined\n(R + N)'])
    ax1.legend(loc='upper left', fontsize=11, frameon=True, shadow=True)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bars in [bars1_normal, bars1_breakdown]:
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height + 50,
                        f'{height:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
            else:
                result = results['with_breakdown'][config_names[i]]
                if not result['success']:
                    ax1.text(bar.get_x() + bar.get_width()/2., 100,
                            'FAILED', ha='center', va='bottom', fontsize=9, fontweight='bold',
                            color='white', bbox=dict(boxstyle='round', facecolor='#c0392b', alpha=0.9))
    
    # ===== CHART 2: Performance Improvement Matrix =====
    ax2 = fig.add_subplot(gs[1, 0])
    
    baseline_normal = results['no_breakdown']['No Reconfig, No Negotiation']['makespan']
    improvements_normal = []
    for name in config_names:
        result = results['no_breakdown'][name]
        if result['success']:
            improvement = ((baseline_normal - result['makespan']) / baseline_normal) * 100
            improvements_normal.append(improvement)
        else:
            improvements_normal.append(-100)
    
    bar_colors = [colors_dict[name] for name in config_names]
    bars2 = ax2.barh(range(len(config_names)), improvements_normal, color=bar_colors, 
                     alpha=0.8, edgecolor='black', linewidth=1.2)
    ax2.set_yticks(range(len(config_names)))
    ax2.set_yticklabels(['Baseline', 'Reconfig\nOnly', 'Negotiation\nOnly', 'Combined'])
    ax2.set_xlabel('Improvement (%)', fontsize=11, fontweight='bold')
    ax2.set_title('(b) Performance Improvement\n(Normal Operation)', fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    for i, (bar, val) in enumerate(zip(bars2, improvements_normal)):
        ax2.text(val + 0.5 if val > 0 else val - 0.5, i, f'{val:+.1f}%', 
                va='center', ha='left' if val > 0 else 'right', fontweight='bold', fontsize=9)
    
    # ===== CHART 3: Wait Time Comparison =====
    ax3 = fig.add_subplot(gs[1, 1])
    
    wait_times_normal = [results['no_breakdown'][name]['avg_wait_time'] if results['no_breakdown'][name]['success'] else 0 
                        for name in config_names]
    wait_times_breakdown = [results['with_breakdown'][name]['avg_wait_time'] if results['with_breakdown'][name]['success'] else 0 
                           for name in config_names]
    
    x_pos2 = np.arange(len(config_names))
    bars3_normal = ax3.bar(x_pos2 - width/2, wait_times_normal, width, 
                          label='Normal', color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars3_breakdown = ax3.bar(x_pos2 + width/2, wait_times_breakdown, width, 
                             label='Breakdown', color='#e67e22', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax3.set_ylabel('Avg Wait Time (time units)', fontsize=11, fontweight='bold')
    ax3.set_title('(c) Average Wait Time\nComparison', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_pos2)
    ax3.set_xticklabels(['Baseline', 'Reconfig', 'Negotiation', 'Combined'], rotation=45, ha='right')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bars in [bars3_normal, bars3_breakdown]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    # ===== CHART 4: Success Rate =====
    ax4 = fig.add_subplot(gs[1, 2])
    
    success_rates = []
    for name in config_names:
        normal_success = 1 if results['no_breakdown'][name]['success'] else 0
        breakdown_success = 1 if results['with_breakdown'][name]['success'] else 0
        success_rate = (normal_success + breakdown_success) / 2 * 100
        success_rates.append(success_rate)
    
    bars4 = ax4.bar(range(len(config_names)), success_rates, color=bar_colors,
                    alpha=0.8, edgecolor='black', linewidth=1.2)
    ax4.set_xticks(range(len(config_names)))
    ax4.set_xticklabels(['Baseline', 'Reconfig', 'Negotiation', 'Combined'], rotation=45, ha='right')
    ax4.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
    ax4.set_title('(d) System Reliability\n(Success Rate)', fontsize=12, fontweight='bold')
    ax4.set_ylim([0, 105])
    ax4.axhline(y=100, color='green', linestyle='--', linewidth=1.5, alpha=0.5)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, val in zip(bars4, success_rates):
        ax4.text(bar.get_x() + bar.get_width()/2., val + 2,
                f'{val:.0f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # ===== CHART 5: Breakdown Impact Analysis =====
    ax5 = fig.add_subplot(gs[2, :2])
    
    impact_increase = []
    for name in config_names:
        normal = results['no_breakdown'][name]['makespan'] if results['no_breakdown'][name]['success'] else 0
        breakdown = results['with_breakdown'][name]['makespan'] if results['with_breakdown'][name]['success'] else float('inf')
        if normal > 0 and breakdown != float('inf'):
            increase = ((breakdown - normal) / normal) * 100
            impact_increase.append(increase)
        else:
            impact_increase.append(0 if breakdown == float('inf') else 100)
    
    bars5 = ax5.bar(range(len(config_names)), impact_increase, color=bar_colors,
                    alpha=0.8, edgecolor='black', linewidth=1.2)
    ax5.set_xticks(range(len(config_names)))
    ax5.set_xticklabels(['Baseline\n(No R, No N)', 'Reconfigurability\nOnly', 
                        'Negotiation\nOnly', 'Combined\n(R + N)'], fontsize=11)
    ax5.set_ylabel('Makespan Increase (%)', fontsize=12, fontweight='bold')
    ax5.set_title('(e) Breakdown Impact: Makespan Increase Due to Machine Failure\n(Lower is Better - Shows Resilience)', 
                  fontsize=13, fontweight='bold', pad=15)
    ax5.grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, (bar, val) in enumerate(zip(bars5, impact_increase)):
        if val > 0 and val < 200:
            ax5.text(bar.get_x() + bar.get_width()/2., val + 1,
                    f'+{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
        elif results['with_breakdown'][config_names[i]]['success']:
            ax5.text(bar.get_x() + bar.get_width()/2., 5,
                    'Complete\nFailure', ha='center', va='bottom', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='#c0392b', edgecolor='black', alpha=0.9))
    
    # ===== CHART 6: Key Insights Table =====
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    insights_data = []
    for name in config_names:
        normal_result = results['no_breakdown'][name]
        breakdown_result = results['with_breakdown'][name]
        
        row = [
            name.replace(', ', '\n').replace('With ', '').replace('No ', '✗'),
            f"{normal_result['makespan']:.0f}" if normal_result['success'] else "FAIL",
            f"{breakdown_result['makespan']:.0f}" if breakdown_result['success'] else "FAIL",
            "✓" if breakdown_result['success'] else "✗"
        ]
        insights_data.append(row)
    
    table = ax6.table(cellText=insights_data,
                     colLabels=['Configuration', 'Normal\nMakespan', 'Breakdown\nMakespan', 'Resilient'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Style the table
    for i in range(len(config_names) + 1):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#34495e')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
                cell.set_edgecolor('black')
                cell.set_linewidth(1)
    
    ax6.set_title('(f) Summary Table', fontsize=12, fontweight='bold', pad=20)
    
    # ===== CHART 7: Conceptual Diagram =====
    ax7 = fig.add_subplot(gs[3, :])
    ax7.axis('off')
    ax7.set_xlim(0, 10)
    ax7.set_ylim(0, 3)
    
    # Draw conceptual framework
    # Baseline box
    ax7.add_patch(plt.Rectangle((0.5, 1.2), 1.8, 1.2, facecolor='#e74c3c', alpha=0.3, edgecolor='black', linewidth=2))
    ax7.text(1.4, 1.8, 'BASELINE\n(No R, No N)', ha='center', va='center', fontsize=10, fontweight='bold')
    ax7.text(1.4, 1.4, '• Limited flexibility\n• Breakdown vulnerable', ha='center', va='center', fontsize=8)
    
    # Reconfig box
    ax7.add_patch(plt.Rectangle((3, 1.2), 1.8, 1.2, facecolor='#f39c12', alpha=0.3, edgecolor='black', linewidth=2))
    ax7.text(3.9, 1.8, 'RECONFIGURABLE\n(R, No N)', ha='center', va='center', fontsize=10, fontweight='bold')
    ax7.text(3.9, 1.4, '• Process flexibility\n• Breakdown resilient', ha='center', va='center', fontsize=8)
    
    # Negotiation box
    ax7.add_patch(plt.Rectangle((5.5, 1.2), 1.8, 1.2, facecolor='#3498db', alpha=0.3, edgecolor='black', linewidth=2))
    ax7.text(6.4, 1.8, 'NEGOTIATION\n(No R, N)', ha='center', va='center', fontsize=10, fontweight='bold')
    ax7.text(6.4, 1.4, '• Smart allocation\n• Reduced waiting', ha='center', va='center', fontsize=8)
    
    # Combined box
    ax7.add_patch(plt.Rectangle((8, 1.2), 1.8, 1.2, facecolor='#27ae60', alpha=0.3, edgecolor='black', linewidth=2))
    ax7.text(8.9, 1.8, 'COMBINED\n(R + N)', ha='center', va='center', fontsize=10, fontweight='bold')
    ax7.text(8.9, 1.4, '• BEST performance\n• Maximum resilience', ha='center', va='center', fontsize=8)
    
    # Arrows showing progression
    ax7.arrow(2.4, 1.8, 0.5, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')
    ax7.arrow(4.9, 1.8, 0.5, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')
    ax7.arrow(7.4, 1.8, 0.5, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')
    
    ax7.text(5, 0.5, '(g) Conceptual Framework: Evolution of Manufacturing System Capabilities', 
             ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Add legend for R and N
    ax7.text(5, 0.1, 'R = Reconfigurability    N = Negotiation', 
             ha='center', va='center', fontsize=9, style='italic',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Save figure
    plt.savefig('publication_ready_analysis.png', dpi=400, bbox_inches='tight', facecolor='white')
    plt.savefig('publication_ready_analysis.pdf', dpi=400, bbox_inches='tight', facecolor='white')
    print("\n📊 Publication-ready visualizations saved:")
    print("   - publication_ready_analysis.png (400 DPI)")
    print("   - publication_ready_analysis.pdf (Vector format)")
    plt.show()

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    analyze_negotiation_and_reconfigurability()