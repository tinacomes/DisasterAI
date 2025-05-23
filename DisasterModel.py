!pip install mesa
#!/usr/bin/env python
import random
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.stats as stats
import itertools 
import pickle      


from mesa import Agent, Model
from mesa.space import MultiGrid
#from mesa.time import RandomActivation

#########################################
# Disaster Model Definition (Updated Trust and Accuracy Version)
#########################################
class DisasterModel(Model):
    def __init__(self,
                 share_exploitative,          # Fraction of humans that are exploitative.
                 share_of_disaster,           # Fraction of grid cells affected initially.
                 initial_trust,               # Baseline trust for human agents.
                 initial_ai_trust,            # Baseline trust for AI agents.
                 number_of_humans,
                 share_confirming,            # Fraction of humans that are "confirming"
                 disaster_dynamics=2,         # Maximum change in disaster per tick.
                 shock_probability=0.1,       # Probability that a shock occurs.
                 shock_magnitude=2,           # Maximum shock magnitude.
                 trust_update_mode="average", # (Not used further here)
                 ai_alignment_level=0.3,      # ai alignment
                 exploitative_correction_factor=1.0,  # (Not used further)
                 width=30, height=30,
                 lambda_parameter=0.5,
                 learning_rate=0.05,
                 epsilon=0.2):
        super().__init__()
        self.share_exploitative = share_exploitative
        self.share_of_disaster = share_of_disaster
        self.base_trust = initial_trust
        self.base_ai_trust = initial_ai_trust
        self.num_humans = number_of_humans
        self.num_ai = 5
        self.share_confirming = share_confirming
        self.width = width
        self.height = height
        self.disaster_dynamics = disaster_dynamics
        self.shock_probability = shock_probability
        self.shock_magnitude = shock_magnitude
        self.trust_update_mode = trust_update_mode
        self.exploitative_correction_factor = exploitative_correction_factor
        self.ai_alignment_level = ai_alignment_level
        self.lambda_parameter = lambda_parameter
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        self.grid = MultiGrid(width, height, torus=False)
        self.tick = 0
        self.tokens_this_tick = {}  
        self.global_variance_data = []  # variance tracking for seci 
        self.friend_variance_data = []  

        # For tracking assistance:
        self.assistance_exploit = {}
        self.assistance_explor = {}
        self.assistance_incorrect_exploit = {}
        self.assistance_incorrect_explor = {}
        self.unmet_needs_evolution = []

        #ECHO chamber effects for humans and AI: SECI and AECI
        self.seci_data = []  # (tick, avg_seci_exp, avg_seci_expl) #add metric for social echo chamber
        self.aeci_data = [] 
        self.correlation_data = []  # (tick, corr_exp, corr_expl) track correlation
        
        # Disaster Grid as numpy arrays
        self.disaster_grid = np.zeros((width, height), dtype=int)
        self.baseline_grid = np.zeros((width, height), dtype=int)
        self.epicenter = (random.randint(0, width - 1), random.randint(0, height - 1))
        total_cells = width * height
        self.disaster_radius = math.sqrt(self.share_of_disaster * total_cells / math.pi)

        # Precompute baseline levels based on distance from epicenter
        x, y = np.indices((width, height))
        distances = np.sqrt((x - self.epicenter[0])**2 + (y - self.epicenter[1])**2)
        self.baseline_grid = np.where(distances < self.disaster_radius / 3, 5,
                                     np.where(distances < 2 * self.disaster_radius / 3, 4,
                                             np.where(distances < self.disaster_radius, 3, 0)))
        self.disaster_grid[...] = self.baseline_grid

        # Create a Watts–Strogatz network for friend selection.
        self.social_network = nx.watts_strogatz_graph(self.num_humans, 4, 0.1)

        # Create human agents.
        self.humans = {}
        for i in range(self.num_humans):
            agent_type = "exploitative" if random.random() < self.share_exploitative else "exploratory"
            a = HumanAgent(unique_id=f"H_{i}", model=self, id_num=i, agent_type=agent_type, share_confirming=self.share_confirming, learning_rate=self.learning_rate, epsilon=self.epsilon)
            self.humans[f"H_{i}"] = a
            self.agents.add(a)  # Add to AgentSet instead of schedule
            x = random.randrange(width)
            y = random.randrange(height)
            self.grid.place_agent(a, (x, y))   
            a.pos = (x, y)  # explicitly set the agent's position

        # Initialize trust and info_accuracy for each human.
        self.network_trust_data = []  # (tick, avg_trust_in_friends, avg_trust_outside)
        
        for i in range(self.num_humans):
            agent_id = f"H_{i}"
            agent = self.humans[agent_id]
            # Set friends based on social network (Step 2)
            agent.friends = set(f"H_{j}" for j in self.social_network.neighbors(i) if f"H_{j}" in self.humans)
            for j in range(self.num_humans):
                if agent_id == f"H_{j}":
                    continue
                agent.trust[f"H_{j}"] = random.uniform(self.base_trust - 0.05, self.base_trust + 0.05)
                agent.info_accuracy[f"H_{j}"] = random.uniform(0.3, 0.7)
            for friend_id in agent.friends:
                agent.trust[friend_id] = min(1, agent.trust[friend_id] + 0.1)
            for k in range(self.num_ai):
                ai_trust = initial_ai_trust if agent.agent_type == "exploitative" else initial_ai_trust - 0.1  # Lower initial AI trust for exploratory
                agent.trust[f"A_{k}"] = random.uniform(self.base_ai_trust - 0.1, self.base_ai_trust + 0.1)
                agent.info_accuracy[f"A_{k}"] = random.uniform(0.4, 0.7)

        # Create AI agents.
        self.ais = {}
        for k in range(self.num_ai):
            a = AIAgent(unique_id=f"A_{k}", model=self)
            self.ais[f"A_{k}"] = a
            self.agents.add(a) 
            x = random.randrange(width)
            y = random.randrange(height)
            self.grid.place_agent(a, (x, y))
            a.pos = (x, y)  # explicitly set the agent's position

        # Data tracking.
        self.trust_data = []
        self.calls_data = []
        self.rewards_data = []
   
    def update_disaster(self):
        # Calculate difference from baseline
        diff = self.baseline_grid - self.disaster_grid
    
        # Compute gradual changes toward baseline
        change = np.zeros_like(self.disaster_grid)
        change[diff > 0] = np.random.randint(1, int(self.disaster_dynamics) + 1, size=np.sum(diff > 0))
        change[diff < 0] = -np.random.randint(1, int(self.disaster_dynamics) + 1, size=np.sum(diff < 0))
    
        # Apply random shocks
        shock_mask = np.random.random(self.disaster_grid.shape) < self.shock_probability
        shocks = np.random.randint(-self.shock_magnitude, self.shock_magnitude + 1, size=self.disaster_grid.shape)
        shocks[~shock_mask] = 0  # Zero out shocks where mask is False
    
        # Update grid in-place
        self.disaster_grid = np.clip(self.disaster_grid + change + shocks, 0, 5)

    def step(self):
        self.tick += 1
        self.tokens_this_tick = {}
        self.update_disaster()  # Update every tick now
        self.agents.shuffle_do("step")

        # Track unmet needs
        height, width = self.disaster_grid.shape
        token_array = np.zeros((height, width), dtype=int)
        for pos, count in self.tokens_this_tick.items():
            x, y = pos
            token_array[x, y] = count
        need_mask = self.disaster_grid >= 4
        unmet = np.sum(need_mask & (token_array == 0))
        self.unmet_needs_evolution.append(unmet)

        # Process rewards
        total_reward_exploit = 0
        total_reward_explor = 0
        for agent in self.humans.values():
            agent.process_relief_actions(self.tick, self.disaster_grid)
            if agent.agent_type == "exploitative":
                total_reward_exploit += agent.total_reward
            else:
                total_reward_explor += agent.total_reward

        # Trust and call tracking
        called_sources = set()
        exp_human_trust = []
        exp_ai_trust = []
        expl_human_trust = []
        expl_ai_trust = []
        calls_exp_human = calls_exp_ai = calls_expl_human = calls_expl_ai = 0
        exp_trust_in = []
        exp_trust_out = []
        expl_trust_in = []
        expl_trust_out = []
        for agent in self.humans.values():
            human_vals = [v for key, v in agent.trust.items() if key.startswith("H_")]
            ai_vals = [v for key, v in agent.trust.items() if key.startswith("A_")]
            trust_in = [agent.trust[f] for f in agent.friends if f in agent.trust]
            trust_out = [agent.trust[h] for h in agent.trust if h.startswith("H_") and h not in agent.friends]
            if agent.agent_type == "exploitative":
                if human_vals:
                    exp_human_trust.append(np.mean(human_vals))
                if ai_vals:
                    exp_ai_trust.append(np.mean(ai_vals))
                calls_exp_human += agent.calls_human
                calls_exp_ai += agent.calls_ai
                exp_trust_in.append(np.mean(trust_in) if trust_in else 0)
                exp_trust_out.append(np.mean(trust_out) if trust_out else 0)
            else:
                if human_vals:
                    expl_human_trust.append(np.mean(human_vals))
                if ai_vals:
                    expl_ai_trust.append(np.mean(ai_vals))
                calls_expl_human += agent.calls_human
                calls_exp_ai += agent.calls_ai
                expl_trust_in.append(np.mean(trust_in) if trust_in else 0)
                expl_trust_out.append(np.mean(trust_out) if trust_out else 0)
            for entry in agent.pending_relief:
                if len(entry) >= 2 and entry[1] is not None:
                    called_sources.add(entry[1])
            agent.calls_human = 0
            agent.calls_ai = 0
            agent.total_reward = 0

        self.network_trust_data.append((
            self.tick,
            np.mean(exp_trust_in),
            np.mean(exp_trust_out),
            np.mean(expl_trust_in),
            np.mean(expl_trust_out)
        ))

        # SECI calculation (every 5 ticks)
        if self.tick % 5 == 0:
            all_beliefs = []
            for agent in self.humans.values():
                all_beliefs.extend(agent.beliefs.values())
            global_variance = np.var(all_beliefs) if all_beliefs else 1e-6

            seci_exp = []
            seci_expl = []
            exp_friend_vars = []
            expl_friend_vars = []
            for agent in self.humans.values():
                friend_ids = agent.friends
                friend_beliefs = []
                for friend_id in friend_ids:
                    friend = self.humans.get(friend_id)
                    if friend:
                        friend_beliefs.extend(friend.beliefs.values())
                friend_variance = np.var(friend_beliefs) if friend_beliefs else global_variance
                seci = max(0, 1 - (friend_variance / global_variance)) if global_variance > 0 else 0
                if agent.agent_type == "exploitative":
                    seci_exp.append(seci)
                    exp_friend_vars.append(friend_variance)
                else:
                    seci_expl.append(seci)
                    expl_friend_vars.append(friend_variance)

            # Append to seci_data inside the if block where seci_exp and seci_expl are defined
            self.seci_data.append((
                self.tick,
                np.mean(seci_exp) if seci_exp else 0,
                np.mean(seci_expl) if seci_expl else 0
            ))
            self.global_variance_data.append((self.tick, global_variance))
            self.friend_variance_data.append((
                self.tick,
                np.mean(exp_friend_vars) if exp_friend_vars else 0,
                np.mean(expl_friend_vars) if expl_friend_vars else 0
            ))

            # Debug output (optional)
        #    if self.tick % 50 == 0:
         #   print(f"Tick {self.tick}:")
          #  print(f"  Global Variance: {global_variance}")
           # print(f"  Exploitative Friend Variance: {np.mean(exp_friend_vars) if exp_friend_vars else 0}")
           # print(f"  Exploratory Friend Variance: {np.mean(expl_friend_vars) if expl_friend_vars else 0}")
           # print(f"  SECI Exp: {np.mean(seci_exp) if seci_exp else 0}, SECI Expl: {np.mean(seci_expl) if seci_expl else 0}")

        # AECI calculation (every 5 ticks)
        if self.tick % 5 == 0:
            aeci_exp = []
            aeci_expl = []
            for agent in self.humans.values():
                total_calls = (agent.calls_human + agent.calls_ai) or 1
                ai_contribution = 0
                for ai_id in [f"A_{k}" for k in range(self.num_ai)]:
                    calls = sum(1 for entry in agent.pending_relief if len(entry) >= 2 and entry[1] == ai_id)
                    trust = agent.trust.get(ai_id, 0)
                    alignment = agent.ai_alignment_scores.get(ai_id, 0.5)
                    ai_contribution += calls * trust * alignment
                aeci = ai_contribution / total_calls
                if agent.agent_type == "exploitative":
                    aeci_exp.append(aeci)
                else:
                    aeci_expl.append(aeci)
            self.aeci_data.append((self.tick, np.mean(aeci_exp) if aeci_exp else 0, np.mean(aeci_expl) if aeci_expl else 0))

        # Correlation calculation (sampled, with adjusted window)
        if self.tick % 5 == 0 and self.tick >= 50:
            seci_exp_window = [d[1] for d in self.seci_data[-10:]]
            aeci_exp_window = [d[1] for d in self.aeci_data[-10:]]
            seci_expl_window = [d[2] for d in self.seci_data[-10:]]
            aeci_expl_window = [d[2] for d in self.aeci_data[-10:]]
            corr_exp, p_exp = stats.pearsonr(seci_exp_window, aeci_exp_window) if len(seci_exp_window) > 1 else (0, 1)
            corr_expl, p_expl = stats.pearsonr(seci_expl_window, aeci_expl_window) if len(seci_expl_window) > 1 else (0, 1)
            self.correlation_data.append((self.tick, corr_exp, corr_expl, p_exp, p_exp))
    
        # Trust decay (every tick for now)
        for agent in self.humans.values():
            for source in agent.trust:
                if source not in called_sources:
                    if agent.agent_type == "exploitative":
                        decay = 0.0005 if (source.startswith("H_") and source in agent.friends) else \
                            0.02 if source.startswith("H_") else 0.05
                    else:
                        decay = 0.05
                    agent.trust[source] = max(0, agent.trust[source] - decay)

        # Store data
        self.trust_data.append((
            self.tick,
            np.mean(exp_ai_trust) if exp_ai_trust else 0,
            np.mean(expl_ai_trust) if expl_ai_trust else 0,
            np.mean(exp_trust_in) if exp_trust_in else 0,
            np.mean(exp_trust_out) if exp_trust_out else 0,
            np.mean(expl_trust_in) if expl_trust_in else 0,
            np.mean(expl_trust_out) if expl_trust_out else 0
        ))

        self.calls_data.append((calls_exp_human, calls_exp_ai, calls_expl_human, calls_expl_ai))
        self.rewards_data.append((total_reward_exploit, total_reward_explor))
    
#########################################
# Agent Definitions
#########################################
class HumanAgent(Agent):
    def __init__(self, unique_id, model, id_num, agent_type, share_confirming, learning_rate=0.05, epsilon=0.2):
      
        super().__init__(model)
        self.unique_id = unique_id
        self.model = model
        self.id_num = id_num
        self.pos = None
        self.agent_type = agent_type
        self.tokens = 5
        self.share_confirming = share_confirming
        # Determine if this agent is confirming (only for exploitative)
        self.is_confirming = (agent_type == "exploitative" and random.random() < share_confirming)
        # Set D and delta based on confirming status
        if self.is_confirming:
            self.D = 1.0  # Stricter for confirming agents
            self.delta = 3
        else:
            self.D = 2.0  # More lenient for non-confirming
            self.delta = 2
        self.trust = {}
        self.info_accuracy = {}
        self.friends = set()
        self.Q = {}
        self.q_parameter = 0.95
        self.lambda_parameter = 0.5
        self.learning_rate = learning_rate  
        self.calls_human = 0
        self.calls_ai = 0
        self.total_reward = 0
        self.pending_relief = []
        self.ai_reported = {}
        self.delayed_reports = []
        self.ai_alignment_scores = {}
        width, height = self.model.grid.width, self.model.grid.height
        self.beliefs = {(x, y): 0 for x in range(width) for y in range(height)}
        self.epsilon = epsilon         # initial exploration probability
        self.last_call_tick = 0    # track tick of the last call
        self.last_human_call_tick = 0 #track last call to human

    def smooth_friend_trust(self):
        if self.friends:
            # Average the agent's trust toward all friends
            friend_trust_values = [self.trust[f] for f in self.friends]
            avg_friend_trust = sum(friend_trust_values) / len(friend_trust_values)
            # Converge each friend’s trust toward this average
            for friend_id in self.friends:
                # Weighted update to not jump too abruptly
                self.trust[friend_id] = 0.5 * self.trust[friend_id] + 0.5 * avg_friend_trust


    def sense_environment(self):
        pos = self.pos
        radius = 1 if self.agent_type == "exploitative" else 5
        cells = self.model.grid.get_neighborhood(pos, moore=True, radius=radius, include_center=True)
        for cell in cells:
            x, y = cell
            actual = self.model.disaster_grid[x, y]
            if random.random() < 0.3:
                self.beliefs[cell] = max(0, min(5, actual + random.choice([-1, 1])))
            else:
                self.beliefs[cell] = actual


    def request_information(self):
        # Initialize candidate lists and tracking dictionaries.
        human_candidates = []
        ai_candidates = []
        accepted_counts = {}
        aggregated_reports = {}

        # --- Collect Candidates ---
        if self.agent_type == "exploitative":
            all_human_candidates = []
            network_human_candidates = []
            non_network_candidates = []
            for candidate in self.trust:
                if candidate.startswith("H_"):
                    bonus = 1.0 if candidate in self.friends else 0.0
                    base_q = ((self.info_accuracy.get(candidate, 0.5) * 0.2) +
                          (self.trust[candidate] * 0.8) + bonus)
                    noise = random.uniform(0, 0.1)
                    if candidate not in self.Q:
                        self.Q[candidate] = (base_q + noise) * self.q_parameter
                    candidate_tuple = (candidate, self.Q[candidate])
                    all_human_candidates.append(candidate_tuple)
                    if candidate in self.friends:
                        network_human_candidates.append(candidate_tuple)
                    else:
                        non_network_candidates.append(candidate_tuple)
                elif candidate.startswith("A_"):
                    if candidate not in self.Q:
                        coverage_bonus = 0.3  # lower bonus so AI is not dominating
                        self.Q[candidate] = (
                            (self.info_accuracy.get(candidate, 0.5) * 0.6) +
                            (self.trust[candidate] * 0.4)
                        ) * self.q_parameter * coverage_bonus
                    ai_candidates.append((candidate, self.Q[candidate]))
            # Prefer friends (network) if available.
            human_candidates = network_human_candidates if network_human_candidates else all_human_candidates
            if non_network_candidates and random.random() < 0.05:
                human_candidates.append(random.choice(non_network_candidates))
        else:
            # Exploratory branch: use a milder bonus structure.
            network_neighbors = set(f"H_{j}" for j in self.model.social_network.neighbors(self.id_num)
                                if f"H_{j}" in self.model.humans)
            extended_network = set()
            for neighbor in network_neighbors:
                neighbor_id = int(neighbor.split("_")[1])
                extended_network.update(f"H_{j}" for j in self.model.social_network.neighbors(neighbor_id)
                                    if f"H_{j}" in self.model.humans and f"H_{j}" != self.unique_id)
            all_human_candidates = []
            network_human_candidates = []
            non_network_candidates = []
            for candidate in self.trust:
                if candidate.startswith("H_"):
                    bonus = 0.5 if candidate in network_neighbors else (0.3 if candidate in extended_network else 0.1)
                    if candidate not in self.Q:
                        self.Q[candidate] = (
                            (self.info_accuracy.get(candidate, 0.5) * 0.5) +
                            (self.trust[candidate] * 0.5) + bonus
                        ) * self.q_parameter
                    candidate_tuple = (candidate, self.Q[candidate])
                    all_human_candidates.append(candidate_tuple)
                    if candidate in network_neighbors or candidate in extended_network:
                        network_human_candidates.append(candidate_tuple)
                    else:
                        non_network_candidates.append(candidate_tuple)
                elif candidate.startswith("A_"):
                    if candidate not in self.Q:
                        coverage_bonus = 0.8
                        self.Q[candidate] = (
                            (self.info_accuracy.get(candidate, 0.5) * 0.7) +
                            (self.trust[candidate] * 0.3)
                        ) * self.q_parameter * coverage_bonus
                    ai_candidates.append((candidate, self.Q[candidate]))
            human_candidates = network_human_candidates if network_human_candidates else all_human_candidates

        # --- Evaluate Candidates ---
        best_human = max([q for _, q in human_candidates]) if human_candidates else 0
        best_ai = max([q for _, q in ai_candidates]) if ai_candidates else 0
        multiplier = 4.0 if self.agent_type == "exploitative" else 1.5
        lambda_param = 0.15 if self.agent_type == "exploitative" else 0.4

        # --- Decide Mode (Human vs AI) ---
        if self.agent_type == "exploitative":
            if self.model.tick - self.last_human_call_tick > 10:
                mode_choice = "human"
            elif random.random() < 0.3:
                mode_choice = "human"
            else:
                mode_choice = "human" if (random.random() >= lambda_param and best_human * multiplier > best_ai) else "ai"
        else:
            if random.random() < 0.2:
                mode_choice = "human"
            else:
                mode_choice = "human" if (random.random() >= lambda_param and best_human * multiplier > best_ai) else "ai"

        # --- Process Selected Candidates Based on Mode ---
        if mode_choice == "human":
            # Update timestamp for human calls.
            self.last_human_call_tick = self.model.tick
            candidate_pool = human_candidates.copy()
            num_calls = 5 if self.agent_type == "exploitative" else 7
            selected = []
            for _ in range(min(num_calls, len(candidate_pool))):
                if random.random() < self.epsilon:
                    choice = random.choice(candidate_pool)
                else:
                    choice = max(candidate_pool, key=lambda x: x[1])
                selected.append(choice)
                candidate_pool.remove(choice)

            # Process each selected human candidate.
            for candidate, q_val in selected:
                self.calls_human += 1
                accepted = 0
                confirmations = 0

                # Get the "other" human's report
                other = self.model.humans.get(candidate)
                if other is not None:
                    rep = other.provide_information_full()
                    if rep is not None:
                        # For each cell in the report, decide if we accept it
                        for cell, reported_value in rep.items():
                            # Store the info in aggregated_reports
                            aggregated_reports.setdefault(cell, []).append((reported_value, candidate in self.friends))
                            old_belief = self.beliefs[cell]
                            d = abs(reported_value - old_belief)

                            # Acceptance formula
                            P_accept = 1.0 if d == 0 else (self.D ** self.delta) / ((d ** self.delta) + (self.D ** self.delta))
                            if random.random() < P_accept:
                                accepted += 1
                                # If exploitative and the reported_value matches old_belief
                                if reported_value == old_belief and self.agent_type == "exploitative":
                                    confirmations += 1
                                    # Higher boost if self.is_confirming
                                    trust_boost = 0.3 if self.is_confirming else 0.25
                                    self.trust[candidate] = min(1, self.trust[candidate] + trust_boost)
                                elif self.agent_type == "exploitative":
                                    # Smaller positive update
                                    self.trust[candidate] = min(1, self.trust[candidate] + 0.1)
                                else:
                                    # Exploratory minor boost
                                    self.trust[candidate] = min(1, self.trust[candidate] + 0.05)
                            else:
                                # Negative update if exploitative
                                if self.agent_type == "exploitative":
                                    self.trust[candidate] = max(0, self.trust[candidate] - 0.02)
                                else:
                                    self.trust[candidate] = max(0, self.trust[candidate] - 0.03)

                # Store acceptance info for this candidate
                accepted_counts[candidate] = (accepted, confirmations)
                # Q-value update
                self.Q[candidate] = (1 - self.q_parameter) * self.Q[candidate] + self.q_parameter * accepted

        else:
            # AI branch
            candidate_pool = ai_candidates.copy()
          
            if candidate_pool:
                if random.random() >= lambda_param:
                    selected = [max(candidate_pool, key=lambda x: x[1])]
                else:
                    selected = [random.choice(candidate_pool)]
            else:
                selected = []

            for candidate, q_val in selected:
                self.calls_ai += 1
                accepted = 0
                confirmations = 0

                other = self.model.ais.get(candidate)
                if other is not None:
                    rep = other.provide_information_full(self.beliefs, trust=self.trust[candidate], agent_type=self.agent_type)
                    if rep is not None:
                        for cell, reported_value in rep.items():
                            aggregated_reports.setdefault(cell, []).append((reported_value, False))  # AI is not friend
                            old_belief = self.beliefs[cell]
                            d = abs(reported_value - old_belief)
                            P_accept = 1.0 if d == 0 else (self.D ** self.delta) / ((d ** self.delta) + (self.D ** self.delta))
                            if random.random() < P_accept:
                                accepted += 1
                                if reported_value == old_belief and self.agent_type == "exploitative":
                                    confirmations += 1
                                    # Increased trust boost for exploitative agents
                                    trust_boost = 0.25 if self.is_confirming else 0.15
                                    self.trust[candidate] = min(1, self.trust[candidate] + trust_boost)
                                elif self.agent_type == "exploitative":
                                    self.trust[candidate] = min(1, self.trust[candidate] + 0.15)
                                else:
                                    # Increased trust boost for exploratory agents
                                    self.trust[candidate] = min(1, self.trust[candidate] + 0.10)
                            else:
                                # Increased trust penalty
                                if self.agent_type == "exploitative":
                                    self.trust[candidate] = max(0, self.trust[candidate] - 0.05)
                                else:
                                    self.trust[candidate] = max(0, self.trust[candidate] - 0.10)

                # Store acceptance info
                accepted_counts[candidate] = (accepted, confirmations)
                # Lower learning rate for AI Q-values
                learning_rate_ai = 0.02
                self.Q[candidate] = (1 - learning_rate_ai) * self.Q[candidate] + learning_rate_ai * accepted
            

        # --- Belief Update (using aggregated_reports) ---
        if aggregated_reports:
            for cell, reports in aggregated_reports.items():
                if self.agent_type == "exploitative":
                    friend_reports = [r[0] for r in reports if isinstance(r, tuple) and r[1]]
                    other_reports = [r[0] for r in reports if isinstance(r, tuple) and not r[1]]
                    avg_report = (0.9 * (sum(friend_reports) / len(friend_reports) if friend_reports else 0) +
                              0.1 * (sum(other_reports) / len(other_reports) if other_reports else 0)) if reports else 0
                    current_value = self.beliefs[cell]
                    difference = avg_report - current_value
                    scaling = 1 + (0.3 if self.is_confirming else 0.2) * (len(reports) - 1)
                    self.beliefs[cell] = max(0, min(5, current_value + self.learning_rate * scaling * difference))
                else:
                    friend_reports = [r[0] for r in reports if isinstance(r, tuple) and r[1]]
                    other_reports = [r[0] for r in reports if isinstance(r, tuple) and not r[1]]
                    avg_report = (0.7 * (sum(friend_reports) / len(friend_reports) if friend_reports else 0) +
                              0.3 * (sum(other_reports) / len(other_reports) if other_reports else 0)) if reports else 0
                    current_value = self.beliefs[cell]
                    difference = avg_report - current_value
                    scaling = 0.5 if any(isinstance(r, tuple) and r[1] for r in reports) else 1.0
                    self.beliefs[cell] = max(0, min(5, current_value + self.learning_rate * scaling * difference))

        # Extend pending_relief with acceptance info
        self.pending_relief.extend([
            (self.model.tick, cand, accepted_counts[cand][0], accepted_counts[cand][1]) 
            for cand in accepted_counts
        ])

    
    def update_delayed_beliefs(self):
        new_buffer = []
        for t, reports in self.delayed_reports:
            if self.model.tick - t >= 2:
                for cell, rep_list in reports.items():
                    avg_report = sum(rep_list) / len(rep_list)
                    current_value = self.beliefs[cell]
                    difference = avg_report - current_value
                    scaling = 1 + 0.1 * (len(rep_list) - 1)
                    self.beliefs[cell] = max(0, min(5, current_value + self.learning_rate * scaling * difference))
            else:
                new_buffer.append((t, reports))
        self.delayed_reports = new_buffer


    def send_relief(self):
        self.tokens = 10
        tokens_to_send = self.tokens * 0.7 if self.agent_type == "exploitative" else self.tokens * 0.7

        if self.agent_type == "exploitative":
            raw_cells = self.model.grid.get_neighborhood(self.pos, moore=True, radius=2, include_center=True)
            # Force each cell into a tuple of ints and remove duplicates manually.
            cells = []
            for cell in raw_cells:
                cell_t = tuple(int(v) for v in cell)
                if cell_t not in cells:
                    cells.append(cell_t)
        else:
            height, width = self.model.disaster_grid.shape
            cells = [(x, y) for x in range(width) for y in range(height)]

        friend_positions = {self.model.humans[friend_id].pos for friend_id in self.friends if friend_id in self.model.humans}

        def cell_score(cell):
            x, y = cell
            belief = self.beliefs.get(cell, 0)
            score = belief  # Base score on belief
            if self.agent_type == "exploitative":
                if cell in friend_positions:
                    score += 0.5  # Bonus for friend positions
            else:
                if cell in friend_positions:
                    score += 0.2  # Smaller bonus for exploratory agents
            # Subtract diversity penalty
            existing_tokens = self.model.tokens_this_tick.get(cell, 0)
            diversity_penalty = existing_tokens * 0.5
            score -= diversity_penalty
            # Add small random noise for exploration
            score += random.uniform(0, 0.2)
            return score

        # Compute candidate scores for all cells
        candidate_scores = [(cell, cell_score(cell)) for cell in cells]
        # Ensure non-negative weights by shifting if necessary
        min_score = min(score for _, score in candidate_scores)
        adjusted_scores = [score - min_score + 0.1 for _, score in candidate_scores]  # +0.1 to avoid zeros
        total_score = sum(adjusted_scores)
        probabilities = [score / total_score for score in adjusted_scores]

        candidate_list = [tuple(cell) for cell, _ in candidate_scores]
        candidate_cells = np.empty(len(candidate_list), dtype=object)
        for i, cell in enumerate(candidate_list):
            candidate_cells[i] = cell

        num_cells_to_send = min(int(tokens_to_send), len(candidate_scores))
        probabilities = np.array(probabilities)

        selected = np.random.choice(
            candidate_cells,
            size=num_cells_to_send,
            replace=False,
            p=probabilities
        )

        for cell in selected:
            self.pending_relief.append((self.model.tick, None, 0, 0, cell))
            self.model.tokens_this_tick[cell] = self.model.tokens_this_tick.get(cell, 0) + 1
        self.tokens -= num_cells_to_send
    
    
    def process_relief_actions(self, current_tick, disaster_grid):
        new_pending = []
        for entry in self.pending_relief:
            if len(entry) == 5:
                t, source_id, accepted_count, confirmations, target_cell = entry
            else:
                t, source_id, accepted_count, confirmations = entry
                target_cell = random.choice(list(self.beliefs.keys()))
            if current_tick - t >= 2:
                x, y = target_cell
                level = self.model.disaster_grid[x, y]
                reward = 2 if level == 4 else (5 if level == 5 else 0)
                if level >= 4 and (self.model.assistance_exploit.get(target_cell, 0) + self.model.assistance_explor.get(target_cell, 0)) == 0:
                    reward = 10
                if level <= 2:
                    reward = -0.2 * accepted_count
                    if source_id:
                        penalty = 0.1 if self.agent_type == "exploitative" else 0.15
                        self.trust[source_id] = max(0, self.trust[source_id] - penalty)
                        self.Q[source_id] = max(0, self.Q[source_id] - penalty * self.q_parameter)
                self.total_reward += reward

                if source_id and self.agent_type == "exploratory":
                    actual_diff = abs(self.beliefs[target_cell] - level)
                    self.info_accuracy[source_id] = max(0, min(1, self.info_accuracy.get(source_id, 0.5) - 0.05 * actual_diff))

                # Update per-agent token counts
                agent_type_key = "exploit" if self.agent_type == "exploitative" else "explor"
                if level >= 4:
                    self.model.per_agent_tokens[agent_type_key][self.unique_id]["correct"] += 1
                    self.model.per_agent_tokens[agent_type_key][self.unique_id]["positions"].append(target_cell)
                    self.model.assistance_exploit[target_cell] = self.model.assistance_exploit.get(target_cell, 0) + 1 if self.agent_type == "exploitative" else self.model.assistance_exploit.get(target_cell, 0)
                    self.model.assistance_explor[target_cell] = self.model.assistance_explor.get(target_cell, 0) + 1 if self.agent_type == "exploratory" else self.model.assistance_explor.get(target_cell, 0)
                elif level <= 2:
                    self.model.per_agent_tokens[agent_type_key][self.unique_id]["incorrect"] += 1
                    self.model.per_agent_tokens[agent_type_key][self.unique_id]["positions"].append(target_cell)
                    self.model.assistance_incorrect_exploit[target_cell] = self.model.assistance_incorrect_exploit.get(target_cell, 0) + 1 if self.agent_type == "exploitative" else self.model.assistance_incorrect_exploit.get(target_cell, 0)
                    self.model.assistance_incorrect_explor[target_cell] = self.model.assistance_incorrect_explor.get(target_cell, 0) + 1 if self.agent_type == "exploratory" else self.model.assistance_incorrect_explor.get(target_cell, 0)
                self.model.tokens_this_tick[target_cell] = self.model.tokens_this_tick.get(target_cell, 0) + 1
            else:
                new_pending.append(entry)
        self.pending_relief = new_pending

    def provide_information_full(self):
        pos = self.pos
        cells = self.model.grid.get_neighborhood(pos, moore=True, include_center=True)
        info = {}
        for cell in cells:
            level = self.model.disaster_grid[cell]
            if random.random() < 0.1:
                level = max(0, min(5, level + random.choice([-1, 1])))
            info[cell] = level
        return info

    def step(self):
        self.sense_environment()
        self.request_information()
        self.send_relief()
        if self.agent_type == "exploratory":
            self.update_delayed_beliefs()
        if self.agent_type == "exploitative":
            self.smooth_friend_trust()


class AIAgent(Agent):

    def __init__(self, unique_id, model):
        super().__init__(model)
        self.unique_id = unique_id
        self.model = model
        self.memory = {}
        self.sensed = {}
        self.sense_radius = 10

    def sense_environment(self):
        cells = self.model.grid.get_neighborhood(self.pos, moore=True, radius=self.sense_radius, include_center=True)
        num_cells = min(int(0.1 * self.model.width * self.model.height), len(cells))
        if num_cells < len(cells):
            cells_array = np.array(cells, dtype=object)
            indices = np.random.choice(len(cells_array), size=num_cells, replace=False)
            cells = [tuple(cells_array[i]) for i in indices]
        self.sensed = {}
        current_tick = self.model.tick
        for cell in cells:
            x, y = cell
            memory_key = (current_tick - 1, cell)
            if memory_key in self.memory and np.random.random() < 0.8:
                self.sensed[cell] = self.memory[memory_key]
            else:
                value = self.model.disaster_grid[x, y]
                self.sensed[cell] = value
                self.memory[(current_tick, cell)] = value

    def provide_information_full(self, human_beliefs, trust, agent_type=None):
        if not self.sensed:
            return {}
        cells = list(self.sensed.keys())
        sensed_vals = np.array([self.sensed[cell] for cell in cells])
        human_vals = np.array([human_beliefs.get(cell, sensed_vals[i]) for i, cell in enumerate(cells)])
        diff = np.abs(sensed_vals - human_vals)
    
        # New alignment factor: same for both agent types, more impactful
        trust_factor = 1 - min(1, trust)
        alignment_factor = self.model.ai_alignment_level * (1 + trust_factor)
    
        # Nonlinear adjustment: amplify alignment when differences are large
        adjustment = alignment_factor * (human_vals - sensed_vals) * (1 + diff / 5)
        corrected = np.round(sensed_vals + adjustment)
        corrected = np.clip(corrected, 0, 5)
    
        return {cell: int(corrected[i]) for i, cell in enumerate(cells)}

    def step(self):
        self.sense_environment()

##################### Simulations #########

def run_simulation(params):
    model = DisasterModel(
        share_exploitative=params.get("share_exploitative", 0.5),
        share_of_disaster=params.get("share_of_disaster", 0.2),
        initial_trust=params.get("initial_trust", 0.5),
        initial_ai_trust=params.get("initial_ai_trust", 0.5),
        number_of_humans=params.get("number_of_humans", 50),
        share_confirming=params.get("share_confirming", 0.5),
        disaster_dynamics=params.get("disaster_dynamics", 2),
        shock_probability=params.get("shock_probability", 0.1),
        shock_magnitude=params.get("shock_magnitude", 2),
        trust_update_mode=params.get("trust_update_mode", "average"),
        ai_alignment_level=params["ai_alignment_level"],
        exploitative_correction_factor=params.get("exploitative_correction_factor", 1.0),
        width=params.get("width", 50),
        height=params.get("height", 50),
        lambda_parameter=params.get("lambda_parameter", 0.5),
        learning_rate=params.get("learning_rate", 0.05),
        epsilon=params.get("epsilon", 0.2)
    )
    ticks = params.get("ticks", 150)
    for _ in range(ticks):
        model.step()
    return model

# Define the simulation generator 
import gc
def simulation_generator(num_runs, base_params):
    for seed in range(num_runs):
        random.seed(seed)
        np.random.seed(seed)
        model = run_simulation(base_params)
        result = {
            "trust": np.array(model.trust_data),
            "seci": np.array(model.seci_data),
            "aeci": np.array(model.aeci_data),
            "assist_exploit": sum(model.assistance_exploit.values()),
            "assist_explor": sum(model.assistance_explor.values()),
            "assist_incorrect_exploit": sum(model.assistance_incorrect_exploit.values()),
            "assist_incorrect_explor": sum(model.assistance_incorrect_explor.values()),
            "per_agent_tokens": model.per_agent_tokens 
        }
        yield result, model
        del model
        gc.collect()

# Define aggregation function
def aggregate_simulation_results(num_runs, base_params):
    trust_list = []
    seci_list = []
    aeci_list = []
    assist_exploit_list = []
    assist_explor_list = []
    assist_incorrect_exploit_list = []
    assist_incorrect_explor_list = []
    
    for result in simulation_generator(num_runs, base_params):
        trust_list.append(result["trust"])
        seci_list.append(result["seci"])
        aeci_list.append(result["aeci"])
        assist_exploit_list.append(result["assist_exploit"])
        assist_explor_list.append(result["assist_explor"])
        assist_incorrect_exploit_list.append(result["assist_incorrect_exploit"])
        assist_incorrect_explor_list.append(result["assist_incorrect_explor"])
        per_agent_tokens_list.append(result["per_agent_tokens"])
    
    trust_array = np.stack(trust_list, axis=0)
    seci_array = np.stack(seci_list, axis=0)
    aeci_array = np.stack(aeci_list, axis=0)
    
    trust_mean = np.mean(trust_array, axis=0)
    trust_lower = np.percentile(trust_array, 25, axis=0)
    trust_upper = np.percentile(trust_array, 75, axis=0)
    
    seci_mean = np.mean(seci_array, axis=0)
    seci_lower = np.percentile(seci_array, 25, axis=0)
    seci_upper = np.percentile(seci_array, 75, axis=0)
    
    aeci_mean = np.mean(aeci_array, axis=0)
    aeci_lower = np.percentile(aeci_array, 25, axis=0)
    aeci_upper = np.percentile(aeci_array, 75, axis=0)

    # Compute per-agent averages
    exploit_correct = []
    exploit_incorrect = []
    explor_correct = []
    explor_incorrect = []
    for per_agent_tokens in per_agent_tokens_list:
        num_exploit = len(per_agent_tokens["exploit"])
        num_explor = len(per_agent_tokens["explor"])
        exploit_correct.append(np.mean([data["correct"] for data in per_agent_tokens["exploit"].values()]) if num_exploit > 0 else 0)
        exploit_incorrect.append(np.mean([data["incorrect"] for data in per_agent_tokens["exploit"].values()]) if num_exploit > 0 else 0)
        explor_correct.append(np.mean([data["correct"] for data in per_agent_tokens["explor"].values()]) if num_explor > 0 else 0)
        explor_incorrect.append(np.mean([data["incorrect"] for data in per_agent_tokens["explor"].values()]) if num_explor > 0 else 0)
    
    
    assist_stats = {
        "exploit": {
            "mean": np.mean(assist_exploit_list),
            "lower": np.percentile(assist_exploit_list, 25),
            "upper": np.percentile(assist_exploit_list, 75)
        },
        "explor": {
            "mean": np.mean(assist_explor_list),
            "lower": np.percentile(assist_explor_list, 25),
            "upper": np.percentile(assist_explor_list, 75)
        },
        "incorrect_exploit": {
            "mean": np.mean(assist_incorrect_exploit_list),
            "lower": np.percentile(assist_incorrect_exploit_list, 25),
            "upper": np.percentile(assist_incorrect_exploit_list, 75)
        },
        "incorrect_explor": {
            "mean": np.mean(assist_incorrect_explor_list),
            "lower": np.percentile(assist_incorrect_explor_list, 25),
            "upper": np.percentile(assist_incorrect_explor_list, 75)
        },
        "per_agent": {
            "exploit_correct": {"mean": np.mean(exploit_correct), "lower": np.percentile(exploit_correct, 25), "upper": np.percentile(exploit_correct, 75)},
            "exploit_incorrect": {"mean": np.mean(exploit_incorrect), "lower": np.percentile(exploit_incorrect, 25), "upper": np.percentile(exploit_incorrect, 75)},
            "explor_correct": {"mean": np.mean(explor_correct), "lower": np.percentile(explor_correct, 25), "upper": np.percentile(explor_correct, 75)},
            "explor_incorrect": {"mean": np.mean(explor_incorrect), "lower": np.percentile(explor_incorrect, 25), "upper": np.percentile(explor_incorrect, 75)}
        }
    }
    
    return {
        "trust": {"mean": trust_mean, "lower": trust_lower, "upper": trust_upper},
        "seci": {"mean": seci_mean, "lower": seci_lower, "upper": seci_upper},
        "aeci": {"mean": aeci_mean, "lower": aeci_lower, "upper": aeci_upper},
        "assist": assist_stats
    }

####################################################################
###################### Define Experiments #######################
####################################################################

##### Experiment 1: vary share of exploitative agents ######
def experiment_share_exploitative(base_params, share_values, num_runs=20):
    results = {}
    for share in share_values:
        params = base_params.copy()
        params["share_exploitative"] = share
        print("Running share_exploitative =", share)
        results[share] = aggregate_simulation_results(num_runs, params)
    return results

##### Experiment 2: vary alignment level, by which AI adapts to human beliefs ######
def experiment_ai_alignment(base_params, alignment_values, num_runs=20):
    results = {}
    for align in alignment_values:
        params = base_params.copy()
        params["ai_alignment_level"] = align
        print("Running ai_alignment_level =", align)
        results[align] = aggregate_simulation_results(num_runs, params)
    return results

    import itertools

##### Experiment 3: vary alignment disaster dynamics ######
def experiment_disaster_dynamics(base_params, dynamics_values, shock_values, num_runs=20):
    results = {}
    for dd, sm in itertools.product(dynamics_values, shock_values):
        params = base_params.copy()
        params["disaster_dynamics"] = dd
        params["shock_magnitude"] = sm
        print("Running disaster_dynamics =", dd, "shock_magnitude =", sm)
        results[(dd, sm)] = aggregate_simulation_results(num_runs, params)
    return results


##### Experiment 4: vary learning and trust ######
def experiment_learning_trust(base_params, learning_rate_values, epsilon_values, num_runs=20):
    results = {}
    for lr, eps in itertools.product(learning_rate_values, epsilon_values):
        params = base_params.copy()
        params["learning_rate"] = lr
        params["epsilon"] = eps
        print("Running learning_rate =", lr, "epsilon =", eps)
        results[(lr, eps)] = aggregate_simulation_results(num_runs, params)
    return results


#########################################################################################
### MAIN ####
#########################################################################################

if __name__ == "__main__":
    # Define base parameters.
    base_params = {
        "share_exploitative": 0.5,  # default, but will be overwritten in experiment (a)
        "share_of_disaster": 0.2,
        "initial_trust": 0.5,
        "initial_ai_trust": 0.5,
        "number_of_humans": 30,
        "share_confirming": 0.5,
        "disaster_dynamics": 2,  # default, to be varied in experiment (c)
        "shock_probability": 0.1,
        "shock_magnitude": 2,    # default, to be varied in experiment (c)
        "trust_update_mode": "average",
        "ai_alignment_level": 0.3,  # default, to be varied in experiment (b)
        "exploitative_correction_factor": 1.0,
        "width": 30,
        "height": 30,
        "lambda_parameter": 0.5,
        "learning_rate": 0.05,  # default, to be varied in experiment (d)
        "epsilon": 0.2,         # default, to be varied in experiment (d)
        "ticks": 150
    }
    
    num_runs = 20  # simulations per parameter combination

    # Save results
    from google.colab import drive
    drive.mount('/content/drive')

    ### Experiment (a): Vary share_exploitative
    share_values = [0.3, 0.5, 0.7]
    results_a = experiment_share_exploitative(base_params, share_values, num_runs)
    # (Now you can plot outcomes from results_a, e.g., final assistance delivered vs. share_exploitative)
    with open('/content/drive/My Drive/results_a.pkl', 'wb') as f:
        pickle.dump(results_a, f)
    print("Experiment (a) completed and saved.")
    
    ### Experiment (b): Vary AI alignment level
    alignment_values = [0.1, 0.3, 0.5]
    results_b = experiment_ai_alignment(base_params, alignment_values, num_runs)
    # (Plot outcomes vs. ai_alignment_level)
    with open('/content/drive/My Drive/results_b.pkl', 'wb') as f:
        pickle.dump(results_b, f)
    print("Experiment (b) completed and saved.")

    ### Experiment (c): Vary disaster dynamics and shock magnitude
    dynamics_values = [1, 2, 3]
    shock_values = [1, 2, 3]
    results_c = experiment_disaster_dynamics(base_params, dynamics_values, shock_values, num_runs)
    # (Plot outcomes vs. disaster parameters; you might create a 3D plot or multiple subplots)
    with open('/content/drive/My Drive/results_c.pkl', 'wb') as f:
        pickle.dump(results_c, f)
    print("Experiment (c) completed and saved.")

    ### Experiment (d): Vary learning_rate and epsilon
    learning_rate_values = [0.03, 0.05, 0.07]
    epsilon_values = [0.2, 0.3]
    results_d = experiment_learning_trust(base_params, learning_rate_values, epsilon_values, num_runs)
    # (Plot outcomes vs. learning/trust parameters)
    with open('/content/drive/My Drive/results_d.pkl', 'wb') as f:
        pickle.dump(results_d, f)
    print("Experiment (d) completed and saved.")

    # For each experiment you can then extract and plot your time series (trust, SECI, AECI) and final assistance values.
    # For example, for experiment (a):
    import matplotlib.pyplot as plt
    shares = sorted(results_a.keys())
    assist_exploit_means = [results_a[s]["assist"]["exploit"]["mean"] for s in shares]
    assist_explor_means = [results_a[s]["assist"]["explor"]["mean"] for s in shares]
    
    plt.figure(figsize=(8, 6))
    plt.plot(shares, assist_exploit_means, marker="o", label="Exploitative Assistance")
    plt.plot(shares, assist_explor_means, marker="s", label="Exploratory Assistance")
    plt.xlabel("Share Exploitative")
    plt.ylabel("Final Total Tokens Delivered")
    plt.title("Effect of Share Exploitative on Assistance Delivered")
    plt.legend()
    plt.show()
    
    # (You can similarly produce plots for experiments (b), (c), and (d).)
    
    # Optionally, force garbage collection between experiments:
    import gc
    gc.collect()
