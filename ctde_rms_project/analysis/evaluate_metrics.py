import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import math

# Define the Manufacturing Environment
class Job:
    def __init__(self, id, processing_time, deadline, setup_time):
        self.id = id
        self.processing_time = processing_time
        self.deadline = deadline
        self.setup_time = setup_time
        self.start_time = None
        self.completion_time = None

    def __repr__(self):
        return f"Job({self.id}, PT:{self.processing_time}, DL:{self.deadline}, ST:{self.setup_time})"

class Machine:
    def __init__(self, id):
        self.id = id
        self.status = 'idle'  # 'idle', 'busy', 'setup'
        self.current_job = None
        self.time_remaining = 0
        self.setup_time_remaining = 0

    def assign_job(self, job):
        self.current_job = job
        self.status = 'setup'
        self.setup_time_remaining = job.setup_time
        # The job's start_time should be set when it actually starts processing after setup
        # job.start_time = env.current_time # Assuming env is accessible - this is better handled in the step method

    def process(self, env): # Pass env here
        if self.status == 'setup':
            self.setup_time_remaining -= 1
            if self.setup_time_remaining == 0:
                self.status = 'busy'
                self.time_remaining = self.current_job.processing_time
                self.current_job.start_time = env.current_time # Set start time when processing begins
        elif self.status == 'busy':
            self.time_remaining -= 1
            if self.time_remaining == 0:
                self.status = 'idle'
                self.current_job.completion_time = env.current_time + 1 # Completion is at the end of the current time step
                completed_job = self.current_job
                self.current_job = None
                return completed_job
        return None

    def __repr__(self):
        return f"Machine({self.id}, Status:{self.status}, Job:{self.current_job.id if self.current_job else 'None'})"


class ManufacturingEnvironment:
    def __init__(self, num_machines=3, num_jobs=15, max_processing_time=5, max_deadline_offset=10, max_setup_time=2):
        self.num_machines = num_machines
        self.num_jobs = num_jobs
        self.machines = [Machine(i) for i in range(num_machines)]
        self.completed_jobs = []
        self.current_time = 0 # Initialize current_time before generating jobs
        self.jobs = self.generate_jobs(num_jobs, max_processing_time, max_deadline_offset, max_setup_time)
        self.job_pool = list(self.jobs) # Copy for availability check

    def generate_jobs(self, num_jobs, max_processing_time, max_deadline_offset, max_setup_time):
        jobs = []
        for i in range(num_jobs):
            processing_time = random.randint(1, max_processing_time)
            deadline = self.current_time + processing_time + random.randint(0, max_deadline_offset)
            setup_time = random.randint(0, max_setup_time)
            jobs.append(Job(i, processing_time, deadline, setup_time))
        return jobs

    def step(self):
        self.current_time += 1
        newly_completed = []
        for machine in self.machines:
            completed_job = machine.process(self) # Pass env (self) here
            if completed_job:
                newly_completed.append(completed_job)
        self.completed_jobs.extend(newly_completed)
        # Remove completed jobs from the active job list
        self.jobs = [job for job in self.jobs if job.completion_time is None]

    def get_state(self):
        # Simple state representation: job processing times and machine states
        job_states = [(job.processing_time, job.deadline, job.setup_time) for job in self.jobs]
        machine_states = [(m.status == 'busy' or m.status == 'setup', m.time_remaining, m.setup_time_remaining) for m in self.machines]
        return {'jobs': job_states, 'machines': machine_states}

    def is_done(self):
        return not self.jobs and all(m.status == 'idle' for m in self.machines)


class DQNAgent:
    def __init__(self, id, input_size, output_size):
        self.id = id
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def train(self, state, action, reward, next_state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
        q_values = self.model(state_tensor)
        next_q_values = self.model(next_state_tensor)
        target_q_values = q_values.clone()
        target_q_values[action] = reward + 0.9 * torch.max(next_q_values)

        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class PPOAgent:
    def __init__(self, id, input_size=3, action_size=1): # Assuming action is a continuous value for simplicity
        self.id = id
        self.actor = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_size)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action = self.actor(state_tensor)
        return action.item() # Return a scalar action

    def evaluate(self, state):
        return self.critic(torch.tensor(state, dtype=torch.float32)).item()

class A2CAgent:
    def __init__(self, id, input_size=3, action_size=1): # Assuming action is a continuous value for simplicity
        self.id = id
        self.actor = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        return self.actor(state_tensor).item() # Return a scalar action

    def evaluate(self, state):
        return self.critic(torch.tensor(state, dtype=torch.float32)).item()


class MultiAgentHybridScheduler:
    def __init__(self, env, agent_type='DQN'):
        self.env = env
        self.agent_type = agent_type
        self.agents = {} # Agent id -> Agent instance
        self.initialize_agents()

    def initialize_agents(self):
        if self.agent_type == 'DQN':
            # DQN agent expects a state representation of a single job and outputs a machine index
            input_size = 3 # processing_time, deadline, setup_time
            output_size = self.env.num_machines
            for i in range(self.env.num_jobs):
                self.agents[i] = DQNAgent(i, input_size, output_size)
        elif self.agent_type == 'PPO' or self.agent_type == 'A2C':
             # For simplicity, PPO and A2C agents take a simplified job state and output a machine preference/score
             input_size = 3 # processing_time, deadline, setup_time
             for i in range(self.env.num_jobs):
                 if self.agent_type == 'PPO':
                     self.agents[i] = PPOAgent(i, input_size)
                 elif self.agent_type == 'A2C':
                     self.agents[i] = A2CAgent(i, input_size)
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")

    def assign_jobs(self):
        available_machines = [m for m in self.env.machines if m.status == 'idle']
        available_jobs = list(self.env.jobs) # Consider all active jobs

        if not available_machines or not available_jobs:
            return # No machines or jobs to assign

        if self.agent_type == 'DQN':
            for job in available_jobs:
                if job.id in self.agents:
                    job_state = (job.processing_time, job.deadline, job.setup_time)
                    chosen_machine_idx = self.agents[job.id].select_action(job_state)
                    if chosen_machine_idx < len(self.env.machines) and self.env.machines[chosen_machine_idx].status == 'idle':
                        self.env.machines[chosen_machine_idx].assign_job(job)
                        # available_machines.remove(self.env.machines[chosen_machine_idx]) # Machine is now busy/setup - no need to remove, just check status
                        # We won't immediately train the DQN agent here in this simplified setup.
                        # Training would typically happen after experiencing a reward from a step.
                        break # Assign one job per time step for simplicity
        elif self.agent_type in ['PPO', 'A2C']:
            job_machine_preferences = []
            for job in available_jobs:
                 if job.id in self.agents:
                     job_state = (job.processing_time, job.deadline, job.setup_time)
                     # Agent provides a preference for a machine or a score to prioritize
                     preference = self.agents[job.id].select_action(job_state)
                     job_machine_preferences.append((job, preference))

            # Sort jobs based on the agent's preference (higher preference first)
            job_machine_preferences.sort(key=lambda x: x[1], reverse=True)

            # Greedily assign jobs to available machines based on sorted preference
            for job, _ in job_machine_preferences:
                if available_machines:
                    # In a real scenario, the agent would output a machine index or a distribution.
                    # Here, we'll just assign the highest-preferred job to the first available machine.
                    machine_to_assign = available_machines.pop(0)
                    machine_to_assign.assign_job(job)
                else:
                    break # No more machines


def compute_objective_score(completions, deadlines, setup_times, weights):
    w_c, w_t, w_s = weights
    scores = []
    for i in range(len(completions)):
        C_i = completions[i]
        d_i = deadlines[i]
        S_i = setup_times[i] if i < len(setup_times) else 0 # Handle cases where setup time data might not be available for all completed jobs
        T_i = max(0, C_i - d_i)
        Z_i = w_c * C_i + w_t * T_i + w_s * S_i
        scores.append(Z_i)
    return scores

def plot_objective_scores(agent_scores, labels):
    plt.figure()
    # Ensure all score lists have the same length for plotting
    max_len = max(len(scores) for scores in agent_scores)
    padded_scores = [scores + [scores[-1]] * (max_len - len(scores)) for scores in agent_scores] # Pad with last value

    for scores, label in zip(padded_scores, labels):
        plt.plot(scores, label=label)
    plt.title("Objective Function Score Comparison")
    plt.xlabel("Job Index")
    plt.ylabel("Objective Score (Z)")
    plt.legend()
    plt.grid(True)
    plt.savefig("objective_function_comparison.png")
    plt.show()


def simulate(agent_type, episodes=10):
    utilization_record = []
    completion_record = []
    all_completed_jobs = [] # To store completed jobs for objective score calculation
    for episode in range(episodes):
        env = ManufacturingEnvironment()
        scheduler = MultiAgentHybridScheduler(env, agent_type)
        jobs_completed_episode = 0
        machine_used_episode = 0
        max_time_steps = 100 # Limit simulation time to avoid infinite loops

        for t in range(max_time_steps):
            scheduler.assign_jobs()
            initial_jobs_len = len(env.jobs)
            env.step()
            jobs_completed_step = initial_jobs_len - len(env.jobs)
            jobs_completed_episode += jobs_completed_step
            machine_used_episode += sum(1 for m in env.machines if m.status in ['busy', 'setup'])

            if env.is_done():
                break

        utilization = machine_used_episode / (env.current_time * env.num_machines if env.current_time > 0 else 1)
        utilization_record.append(utilization)
        completion_record.append(jobs_completed_episode)
        all_completed_jobs.append(env.completed_jobs) # Store completed jobs for this episode

    # Calculate objective scores for the last episode of each agent type
    # Assuming objective score is calculated based on the set of jobs completed in a run
    last_episode_completed_jobs = all_completed_jobs[-1]
    completions = [job.completion_time for job in last_episode_completed_jobs if job.completion_time is not None]
    deadlines = [job.deadline for job in last_episode_completed_jobs if job.completion_time is not None]
    setup_times = [job.setup_time for job in last_episode_completed_jobs if job.completion_time is not None] # Assuming original setup time
    weights = (1, 1, 1) # Example weights
    objective_scores = compute_objective_score(completions, deadlines, setup_times, weights)


    return utilization_record, completion_record, objective_scores

agents = ['DQN', 'PPO', 'A2C']
colors = ['blue', 'green', 'red']

agent_utilization = {}
agent_completion = {}
agent_objective_scores = {}

for agent in agents:
    util, comp, obj_scores = simulate(agent, episodes=20) # Increased episodes for potentially better training if agents were trainable
    agent_utilization[agent] = util
    agent_completion[agent] = comp
    agent_objective_scores[agent] = obj_scores

plt.figure()
for agent, color in zip(agents, colors):
    plt.plot(agent_utilization[agent], label=f'{agent}', color=color)
plt.title("Machine Utilization Comparison")
plt.xlabel("Episode")
plt.ylabel("Utilization")
plt.legend()
plt.savefig("agent_utilization_comparison.png")
plt.show()

plt.figure()
for agent, color in zip(agents, colors):
    plt.plot(agent_completion[agent], label=f'{agent}', color=color)
plt.title("Job Completion Comparison")
plt.xlabel("Episode")
plt.ylabel("Jobs Completed")
plt.legend()
plt.savefig("agent_job_completion_comparison.png")
plt.show()

# Plot Objective Function Score Comparison for the last episode
plot_objective_scores(list(agent_objective_scores.values()), list(agent_objective_scores.keys()))

# Prepare data for the bar chart
# We'll use the average utilization and average jobs completed over episodes for each agent
average_utilization = [sum(agent_utilization[agent]) / len(agent_utilization[agent]) for agent in agents]
average_completion = [sum(agent_completion[agent]) / len(agent_completion[agent]) for agent in agents]

x = np.arange(len(agents)) # the label locations
width = 0.35 # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, average_utilization, width, label='Average Utilization', color='skyblue')
rects2 = ax.bar(x + width/2, average_completion, width, label='Average Jobs Completed', color='lightcoral')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Score')
ax.set_title('Average Performance Metrics by Agent Type')
ax.set_xticks(x)
ax.set_xticklabels(agents)
ax.legend()

# Add value labels on top of the bars
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('%.2f' % height,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.show()


class Negotiation(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x is expected to be at least 2D: (batch_size, ..., embed_dim)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Simple dot-product attention-like mechanism for "negotiation"
        # Ensure k has at least 2 dimensions for transpose
        if k.dim() < 2:
             k = k.unsqueeze(0) # Add a dummy dimension if it's 1D

        # Calculate negotiation scores based on query and key similarity
        negotiation_scores = torch.matmul(q, k.transpose(-2, -1))
        # Apply softmax to get weights representing influence or agreement
        negotiation_weights = self.softmax(negotiation_scores)
        # Combine values based on negotiation weights
        output = torch.matmul(negotiation_weights, v)
        return output, negotiation_weights

class DQNAgentWithNegotiation(DQNAgent):
    def __init__(self, id, input_size, output_size, embed_dim=64):
        super().__init__(id, input_size, output_size)
        self.embed_dim = embed_dim
        # Simple embedding layer for the input state
        self.embedding = nn.Linear(input_size, embed_dim)
        self.negotiation = Negotiation(embed_dim)
        # Redefine the model to use negotiation
        self.model = nn.Sequential(
            self.embedding,
            nn.ReLU(),
            NegotiationLayer(embed_dim), # Use the new NegotiationLayer
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0) # Add batch dimension
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values.squeeze(0)).item() # Remove batch dimension before argmax

    def train(self, state, action, reward, next_state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        q_values = self.model(state_tensor).squeeze(0)
        next_q_values = self.model(next_state_tensor).squeeze(0)
        target_q_values = q_values.clone()
        target_q_values[action] = reward + 0.9 * torch.max(next_q_values)

        loss = self.criterion(q_values.unsqueeze(0), target_q_values.unsqueeze(0)) # Add batch dimension back for loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class PPOAgentWithNegotiation(PPOAgent):
    def __init__(self, id, input_size=3, action_size=1, embed_dim=64):
        super().__init__(id, input_size, action_size)
        self.embed_dim = embed_dim
        self.embedding = nn.Linear(input_size, embed_dim)
        self.negotiation = Negotiation(embed_dim)

        # Redefine actor and critic to use negotiation
        self.actor = nn.Sequential(
            self.embedding,
            nn.Tanh(),
            NegotiationLayer(embed_dim), # Use NegotiationLayer
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, action_size)
        )
        self.critic = nn.Sequential(
            self.embedding,
            nn.ReLU(),
            NegotiationLayer(embed_dim), # Use NegotiationLayer
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )
        self.optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0) # Add batch dimension
        action = self.actor(state_tensor).squeeze(0) # Remove batch dimension
        return action.item() # Return a scalar action


class A2CAgentWithNegotiation(A2CAgent):
    def __init__(self, id, input_size=3, action_size=1, embed_dim=64):
        super().__init__(id, input_size, action_size)
        self.embed_dim = embed_dim
        self.embedding = nn.Linear(input_size, embed_dim)
        self.negotiation = Negotiation(embed_dim)

        # Redefine actor and critic to use negotiation
        self.actor = nn.Sequential(
            self.embedding,
            nn.ReLU(),
            NegotiationLayer(embed_dim), # Use NegotiationLayer
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, action_size)
        )
        self.critic = nn.Sequential(
            self.embedding,
            nn.ReLU(),
            NegotiationLayer(embed_dim), # Use NegotiationLayer
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0) # Add batch dimension
        return self.actor(state_tensor).squeeze(0).item() # Remove batch dimension and return scalar


class NegotiationLayer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.negotiation = Negotiation(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Apply negotiation
        negotiation_output, _ = self.negotiation(x)
        # Add and Norm
        output = self.norm(x + negotiation_output)
        return output

class MultiAgentHybridSchedulerWithNegotiation(MultiAgentHybridScheduler):
    def initialize_agents(self):
        if self.agent_type == 'DQN':
            input_size = 3
            output_size = self.env.num_machines
            for i in range(self.env.num_jobs):
                self.agents[i] = DQNAgentWithNegotiation(i, input_size, output_size)
        elif self.agent_type == 'PPO' or self.agent_type == 'A2C':
             input_size = 3
             for i in range(self.env.num_jobs):
                 if self.agent_type == 'PPO':
                     self.agents[i] = PPOAgentWithNegotiation(i, input_size)
                 elif self.agent_type == 'A2C':
                     self.agents[i] = A2CAgentWithNegotiation(i, input_size)
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")

def simulate_with_negotiation(agent_type, episodes=10):
    utilization_record = []
    completion_record = []
    all_completed_jobs = [] # To store completed jobs for objective score calculation
    for episode in range(episodes):
        env = ManufacturingEnvironment()
        # Use the scheduler with negotiation agents
        scheduler = MultiAgentHybridSchedulerWithNegotiation(env, agent_type)
        jobs_completed_episode = 0
        machine_used_episode = 0
        max_time_steps = 100 # Limit simulation time

        for t in range(max_time_steps):
            scheduler.assign_jobs()
            initial_jobs_len = len(env.jobs)
            env.step()
            jobs_completed_step = initial_jobs_len - len(env.jobs)
            jobs_completed_episode += jobs_completed_step
            machine_used_episode += sum(1 for m in env.machines if m.status in ['busy', 'setup'])

            if env.is_done():
                break

        utilization = machine_used_episode / (env.current_time * env.num_machines if env.current_time > 0 else 1)
        utilization_record.append(utilization)
        completion_record.append(jobs_completed_episode)
        all_completed_jobs.append(env.completed_jobs) # Store completed jobs for this episode

    last_episode_completed_jobs = all_completed_jobs[-1]
    completions = [job.completion_time for job in last_episode_completed_jobs if job.completion_time is not None]
    deadlines = [job.deadline for job in last_episode_completed_jobs if job.completion_time is not None]
    setup_times = [job.setup_time for job in last_episode_completed_jobs if job.completion_time is not None]
    weights = (1, 1, 1)
    objective_scores = compute_objective_score(completions, deadlines, setup_times, weights)

    return utilization_record, completion_record, objective_scores

agents_with_negotiation = ['DQN', 'PPO', 'A2C']
colors_negotiation = ['cyan', 'magenta', 'yellow'] # Different colors for negotiation versions

agent_utilization_neg = {}
agent_completion_neg = {}
agent_objective_scores_neg = {}

print("Simulating agents with Negotiation mechanism...")
for agent in agents_with_negotiation:
    print(f"Simulating {agent} with Negotiation...")
    # Pass the agent type string, the simulation function handles using the Negotiation versions
    util_neg, comp_neg, obj_scores_neg = simulate_with_negotiation(agent, episodes=20)
    agent_utilization_neg[agent + ' (Neg)'] = util_neg
    agent_completion_neg[agent + ' (Neg)'] = comp_neg
    agent_objective_scores_neg[agent + ' (Neg)'] = obj_scores_neg

# Combine data for plotting
all_utilization = {**agent_utilization, **agent_utilization_neg}
all_completion = {**agent_completion, **agent_completion_neg}
all_objective_scores = {**agent_objective_scores, **agent_objective_scores_neg}
all_agents_labels = list(all_utilization.keys())
all_colors = colors + colors_negotiation

# Plot Machine Utilization Comparison
plt.figure(figsize=(10, 6))
for i, label in enumerate(all_agents_labels):
    plt.plot(all_utilization[label], label=label, color=all_colors[i % len(all_colors)])
plt.title("Machine Utilization Comparison (Original vs. Negotiation)")
plt.xlabel("Episode")
plt.ylabel("Utilization")
plt.legend()
plt.grid(True)
plt.savefig("agent_utilization_comparison_with_negotiation.png")
plt.show()

# Plot Job Completion Comparison
plt.figure(figsize=(10, 6))
for i, label in enumerate(all_agents_labels):
    plt.plot(all_completion[label], label=label, color=all_colors[i % len(all_colors)])
plt.title("Job Completion Comparison (Original vs. Negotiation)")
plt.xlabel("Episode")
plt.ylabel("Jobs Completed")
plt.legend()
plt.grid(True)
plt.savefig("agent_job_completion_comparison_with_negotiation.png")
plt.show()

# Plot Objective Function Score Comparison for the last episode
plot_objective_scores(list(all_objective_scores.values()), list(all_objective_scores.keys()))

# Prepare data for the combined bar chart
average_utilization_combined = [sum(all_utilization[agent]) / len(all_utilization[agent]) for agent in all_agents_labels]
average_completion_combined = [sum(all_completion[agent]) / len(all_completion[agent]) for agent in all_agents_labels]

x_combined = np.arange(len(all_agents_labels))
width_combined = 0.35

fig_combined, ax_combined = plt.subplots(figsize=(12, 7))
rects1_combined = ax_combined.bar(x_combined - width_combined/2, average_utilization_combined, width_combined, label='Average Utilization', color='skyblue')
rects2_combined = ax_combined.bar(x_combined + width_combined/2, average_completion_combined, width_combined, label='Average Jobs Completed', color='lightcoral')

ax_combined.set_ylabel('Score')
ax_combined.set_title('Average Performance Metrics by Agent Type (Original vs. Negotiation)')
ax_combined.set_xticks(x_combined)
ax_combined.set_xticklabels(all_agents_labels)
ax_combined.legend()

# Add value labels on top of the bars
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax_combined.annotate('%.2f' % height,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1_combined)
autolabel(rects2_combined)

fig_combined.tight_layout()
plt.show()


# Function to simulate with varying number of machines and jobs
def simulate_varying_params(agent_type, episodes=10, num_machines=3, num_jobs=15):
    utilization_record = []
    completion_record = []
    all_completed_jobs = [] # To store completed jobs for objective score calculation
    for episode in range(episodes):
        env = ManufacturingEnvironment(num_machines=num_machines, num_jobs=num_jobs)
        # Determine which scheduler to use based on agent_type name
        if 'Neg' in agent_type:
            scheduler = MultiAgentHybridSchedulerWithNegotiation(env, agent_type.replace(' (Neg)', ''))
        else:
            scheduler = MultiAgentHybridScheduler(env, agent_type)

        jobs_completed_episode = 0
        machine_used_episode = 0
        max_time_steps = 200 # Increased max time steps for potentially more jobs

        for t in range(max_time_steps):
            scheduler.assign_jobs()
            initial_jobs_len = len(env.jobs)
            env.step()
            jobs_completed_step = initial_jobs_len - len(env.jobs)
            jobs_completed_episode += jobs_completed_step
            machine_used_episode += sum(1 for m in env.machines if m.status in ['busy', 'setup'])

            if env.is_done():
                break

        utilization = machine_used_episode / (env.current_time * env.num_machines if env.current_time > 0 else 1)
        utilization_record.append(utilization)
        completion_record.append(jobs_completed_episode)
        all_completed_jobs.append(env.completed_jobs) # Store completed jobs for this episode

    # Calculate objective scores for the last episode
    last_episode_completed_jobs = all_completed_jobs[-1]
    completions = [job.completion_time for job in last_episode_completed_jobs if job.completion_time is not None]
    deadlines = [job.deadline for job in last_episode_completed_jobs if job.completion_time is not None]
    setup_times = [job.setup_time for job in last_episode_completed_jobs if job.completion_time is not None]
    weights = (1, 1, 1)
    objective_scores = compute_objective_score(completions, deadlines, setup_times, weights)

    return utilization_record, completion_record, objective_scores

# Define the range of machines and jobs to test
machine_range = range(5, 51, 15) # Vary machines from 3 to 50 with a step of 5
job_range = range(15, 226, 35)   # Vary jobs from 15 to 100 with a step of 10

agents_to_test = ['DQN', 'PPO', 'A2C', 'DQN (Neg)', 'PPO (Neg)', 'A2C (Neg)']

# Store results
results = {}

print("Simulating with varying machines and jobs...")
for num_machines in machine_range:
    for num_jobs in job_range:
        print(f"\nSimulating with {num_machines} machines and {num_jobs} jobs...")
        results[(num_machines, num_jobs)] = {}
        for agent_type in agents_to_test:
            print(f"  Simulating {agent_type}...")
            # For performance evaluation, a smaller number of episodes might be sufficient
            util, comp, obj_scores = simulate_varying_params(agent_type, episodes=5, num_machines=num_machines, num_jobs=num_jobs)
            results[(num_machines, num_jobs)][agent_type] = {
                'avg_utilization': sum(util) / len(util),
                'avg_completion': sum(comp) / len(comp),
                # Take the average or some aggregate of objective scores if there are multiple completed jobs
                'avg_objective_score': sum(obj_scores) / len(obj_scores) if obj_scores else 0
            }

# Analyze and visualize results
# We can create plots showing how performance metrics change with the number of machines and jobs for each agent type.


# Plotting Average Utilization vs. Number of Machines for different Job counts
num_job_plots = len(job_range)
num_cols_job = math.ceil(num_job_plots / 2)
plt.figure(figsize=(15, 8))
for j_idx, num_jobs in enumerate(job_range):
    plt.subplot(2, num_cols_job, j_idx + 1)
    for agent_type in agents_to_test:
        util_values = [results[(nm, num_jobs)][agent_type]['avg_utilization'] for nm in machine_range]
        plt.plot(list(machine_range), util_values, label=agent_type, marker='o')
    plt.title(f'Avg Utilization (Jobs={num_jobs})')
    plt.xlabel("Number of Machines")
    plt.ylabel("Average Utilization")
    plt.grid(True)
    if j_idx == 0: # Add legend to the first subplot
        plt.legend()
plt.tight_layout()
plt.savefig("avg_utilization_vs_machines_varying_jobs.png")
plt.show()

# Plotting Average Completion vs. Number of Machines for different Job counts
plt.figure(figsize=(15, 8))
for j_idx, num_jobs in enumerate(job_range):
    plt.subplot(2, num_cols_job, j_idx + 1)
    for agent_type in agents_to_test:
        comp_values = [results[(nm, num_jobs)][agent_type]['avg_completion'] for nm in machine_range]
        plt.plot(list(machine_range), comp_values, label=agent_type, marker='o')
    plt.title(f'Avg Completion (Jobs={num_jobs})')
    plt.xlabel("Number of Machines")
    plt.ylabel("Average Jobs Completed")
    plt.grid(True)
    if j_idx == 0: # Add legend to the first subplot
        plt.legend()
plt.tight_layout()
plt.savefig("avg_completion_vs_machines_varying_jobs.png")
plt.show()


# Plotting Average Utilization vs. Number of Jobs for different Machine counts
num_machine_plots = len(machine_range)
num_cols_machine = math.ceil(num_machine_plots / 2)
plt.figure(figsize=(15, 8))
for m_idx, num_machines in enumerate(machine_range):
    plt.subplot(2, num_cols_machine, m_idx + 1) # Adjust layout for odd number of machines
    for agent_type in agents_to_test:
        util_values = [results[(num_machines, nj)][agent_type]['avg_utilization'] for nj in job_range]
        plt.plot(list(job_range), util_values, label=agent_type, marker='o')
    plt.title(f'Avg Utilization (Machines={num_machines})')
    plt.xlabel("Number of Jobs")
    plt.ylabel("Average Utilization")
    plt.grid(True)
    if m_idx == 0: # Add legend to the first subplot
        plt.legend()
plt.tight_layout()
plt.savefig("avg_utilization_vs_jobs_varying_machines.png")
plt.show()


# Plotting Average Completion vs. Number of Jobs for different Machine counts
plt.figure(figsize=(15, 8))
for m_idx, num_machines in enumerate(machine_range):
    plt.subplot(2, num_cols_machine, m_idx + 1) # Adjust layout
    for agent_type in agents_to_test:
        comp_values = [results[(num_machines, nj)][agent_type]['avg_completion'] for nj in job_range]
        plt.plot(list(job_range), comp_values, label=agent_type, marker='o')
    plt.title(f'Avg Completion (Machines={num_machines})')
    plt.xlabel("Number of Jobs")
    plt.ylabel("Average Jobs Completed")
    plt.grid(True)
    if m_idx == 0: # Add legend to the first subplot
        plt.legend()
plt.tight_layout()
plt.savefig("avg_completion_vs_jobs_varying_machines.png")
plt.show()

# You can similarly plot average objective scores
plt.figure(figsize=(15, 8))
for j_idx, num_jobs in enumerate(job_range):
    plt.subplot(2, num_cols_job, j_idx + 1)
    for agent_type in agents_to_test:
        obj_values = [results[(nm, num_jobs)][agent_type]['avg_objective_score'] for nm in machine_range]
        plt.plot(list(machine_range), obj_values, label=agent_type, marker='o')
    plt.title(f'Avg Objective Score (Jobs={num_jobs})')
    plt.xlabel("Number of Machines")
    plt.ylabel("Average Objective Score")
    plt.grid(True)
    if j_idx == 0: # Add legend to the first subplot
        plt.legend()
plt.tight_layout()
plt.savefig("avg_objective_vs_machines_varying_jobs.png")
plt.show()

plt.figure(figsize=(15, 8))
for m_idx, num_machines in enumerate(machine_range):
    plt.subplot(2, num_cols_machine, m_idx + 1)
    for agent_type in agents_to_test:
        obj_values = [results[(num_machines, nj)][agent_type]['avg_objective_score'] for nj in job_range]
        plt.plot(list(job_range), obj_values, label=agent_type, marker='o')
    plt.title(f'Avg Objective Score (Machines={num_machines})')
    plt.xlabel("Number of Jobs")
    plt.ylabel("Average Objective Score")
    plt.grid(True)
    if m_idx == 0: # Add legend to the first subplot
        plt.legend()
plt.tight_layout()
plt.savefig("avg_objective_vs_jobs_varying_machines.png")
plt.show()

# Filter the results dictionary to include only the specified job counts and machine counts
filtered_results = {}
jobs_to_plot = [15, 100, 200] # Including 200 as requested, even though it's not in the job_range from the previous code
machines_to_plot = [5, 20, 35, 50]

# Ensure that we only attempt to access keys that exist in the original results dictionary
# The previous simulation only covered job counts up to 100. We cannot plot for 200 jobs with the current data.
# Let's adjust jobs_to_plot based on the actual job_range simulated.
simulated_job_counts = list(job_range) # [15, 25, 50, 100,200]
simulated_machine_counts = list(machine_range) # [3, 15,25,50 ]

# Filter jobs_to_plot and machines_to_plot to only include those that were actually simulated
jobs_to_plot_actual = [job for job in jobs_to_plot if job in simulated_job_counts]
machines_to_plot_actual = [machine for machine in machines_to_plot if machine in simulated_machine_counts]

# Store data for the specific plots
plot_data = {}

for num_jobs in jobs_to_plot_actual:
    for num_machines in machines_to_plot_actual:
        if (num_machines, num_jobs) in results:
            plot_data[(num_machines, num_jobs)] = results[(num_machines, num_jobs)]

# Now create individual plots based on the filtered data

# Plot Average Utilization for specified (machines, jobs) combinations
for (num_machines, num_jobs), data in plot_data.items():
    plt.figure(figsize=(8, 5))
    agents = list(data.keys())
    avg_utilization_values = [data[agent]['avg_utilization'] for agent in agents]
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow'] # Ensure enough colors
    plt.bar(agents, avg_utilization_values, color=colors[:len(agents)])
    plt.title(f'Avg Utilization (Machines={num_machines}, Jobs={num_jobs})')
    plt.xlabel("Agent Type")
    plt.ylabel("Average Utilization")
    plt.grid(axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"avg_utilization_m{num_machines}_j{num_jobs}.png")
    plt.show()

# Plot Average Completion for specified (machines, jobs) combinations
for (num_machines, num_jobs), data in plot_data.items():
    plt.figure(figsize=(8, 5))
    agents = list(data.keys())
    avg_completion_values = [data[agent]['avg_completion'] for agent in agents]
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']
    plt.bar(agents, avg_completion_values, color=colors[:len(agents)])
    plt.title(f'Avg Completion (Machines={num_machines}, Jobs={num_jobs})')
    plt.xlabel("Agent Type")
    plt.ylabel("Average Jobs Completed")
    plt.grid(axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"avg_completion_m{num_machines}_j{num_jobs}.png")
    plt.show()

# Plot Average Objective Score for specified (machines, jobs) combinations
for (num_machines, num_jobs), data in plot_data.items():
    plt.figure(figsize=(8, 5))
    agents = list(data.keys())
    avg_objective_values = [data[agent]['avg_objective_score'] for agent in agents]
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']
    plt.bar(agents, avg_objective_values, color=colors[:len(agents)])
    plt.title(f'Avg Objective Score (Machines={num_machines}, Jobs={num_jobs})')
    plt.xlabel("Agent Type")
    plt.ylabel("Average Objective Score")
    plt.grid(axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"avg_objective_m{num_machines}_j{num_jobs}.png")
    plt.show()

