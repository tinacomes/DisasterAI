# Install mesa if not already installed
# !pip install mesa

import os
import random
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import itertools
import pickle
import gc
import csv

from mesa import Agent, Model
from mesa.space import MultiGrid

#########################################
# Helper Classes and Agent Definitions
#########################################

class Candidate:
    def __init__(self, cell):
        self.cell = cell

    def __repr__(self):
        return f"Candidate({self.cell})"

class HumanAgent(Agent):
    def __init__(self, unique_id, model, id_num, agent_type, share_confirming, learning_rate=0.05, epsilon=0.2):
        # Use workaround: call parent initializer with model only, then set attributes.
        super(HumanAgent, self).__init__(model)
        self.unique_id = unique_id  
        self.model = model

        self.id_num = id_num
        self.agent_type = agent_type
        self.share_confirming = share_confirming
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.pos = None
        # self.tokens = 5
        self.beliefs = {} # Will store dictionaries: {(x, y): {'level': L, 'confidence': C}}
        self.info_accuracy = {}  # For info accuracy of other agents
        self.trust = {f"A_{k}": model.base_ai_trust for k in range(5)}
        self.friends = []
        self.D = 2.5 if agent_type == "exploitative" else 1.0
        self.delta = 4 if agent_type == "exploitative" else 2
        self.trust_update_mode = model.trust_update_mode
        self.multiplier = 2.0 if agent_type == "exploitative" else 1.0
        
        # Counters for calls:
        self.accum_calls_total = 0
        self.accum_calls_ai = 0
        self.accum_calls_human = 0
        # Counters for accepted information:
        self.accepted_human = 0
        self.accepted_friend = 0
        self.accepted_ai = 0
        
        self.tokens_this_tick = {}
        self.pending_rewards = []  # List of tuples: (tick_due, mode, list_of_cells)
        self.q_table = {f"A_{k}": 0.0 for k in range(5)}
        self.q_table["human"] = 0.0

    def initialize_beliefs(self):
        height, width = self.model.disaster_grid.shape
        for x in range(width):
            for y in range(height):
                self.beliefs[(x, y)] = 0
        self.sense_environment()

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

    def report_beliefs(self, caller_pos):
        if self.model.disaster_grid[self.pos] >= 4 and random.random() < 0.1: #10% chance of not reporting if on a disaster affected grid cell
            return {}
        radius = 1 if self.agent_type == "exploitative" else 5
        cells = self.model.grid.get_neighborhood(caller_pos, moore=True, radius=radius, include_center=True)
        report = {}
        for cell in cells:
            value = self.beliefs.get(cell, 0)
            if random.random() < 0.1:
                value = max(0, min(5, value + random.choice([-1, 1])))
            report[cell] = value
        return report

    def choose_information_mode(self):
        if random.random() < self.epsilon:
            return random.choice(list(self.q_table.keys()))
        return max(self.q_table, key=self.q_table.get)

    def request_information(self):
        mode_choice = self.choose_information_mode()
        reports = {}
        if mode_choice.startswith("A_"):
            ai_id = mode_choice
            chosen_ai = self.model.ais[ai_id]
            reports = chosen_ai.respond(self.beliefs, self.trust.get(ai_id, 0))
            # Weight AI call as 1
            self.accum_calls_ai += 1
            self.accum_calls_total += 1
        else:
            candidates = [agent for agent in self.model.humans.values() if agent.unique_id != self.unique_id]
            top_candidates = sorted(candidates, key=lambda x: self.trust.get(x.unique_id, 0), reverse=True)[:5]
            for candidate in top_candidates:
                rep = candidate.report_beliefs(self.pos)
                for cell, value in rep.items():
                    if cell not in reports or (candidate.unique_id in self.friends and random.random() < 0.7):
                        reports[cell] = value
            # Weight human call as 5
            self.accum_calls_human += 5
            self.accum_calls_total += 5

        self.tokens_this_tick[mode_choice] = self.tokens_this_tick.get(mode_choice, 0) + 1

        trust_reward = 0
        for cell, reported_value in reports.items():
            old_belief = self.beliefs.get(cell, 0)
            d = abs(reported_value - old_belief)
            friend_weight = 1.5 if (mode_choice == "human" and any(c.unique_id in self.friends for c in top_candidates)) else 1.0
            P_accept = 1.0 if d == 0 else (self.D ** self.delta) / ((d ** self.delta) + (self.D ** self.delta)) * friend_weight
            
            if random.random() < P_accept:
                self.beliefs[cell] = reported_value
                if self.agent_type == "exploitative":
                    delta = 0.075 if self.trust_update_mode == "average" else 0.25
                    target = mode_choice if mode_choice.startswith("A_") else top_candidates[0].unique_id
                    if target not in self.trust: self.trust[target] = self.model.base_ai_trust if target.startswith("A_") else self.model.base_trust
                    trust_reward += delta * friend_weight
                    self.trust[target] = min(1.0, self.trust.get(target, 0) + delta * friend_weight)
                if mode_choice == "human":
                    self.accepted_human += 1
                    if any(c.unique_id in self.friends for c in top_candidates):
                        self.accepted_friend += 1
                else:
                    self.accepted_ai += 1

        if self.agent_type == "exploitative" and trust_reward > 0:
            old_q = self.q_table[mode_choice]
            self.q_table[mode_choice] = old_q + self.learning_rate * (trust_reward - old_q)

    def send_relief(self):
        tokens_to_send = 5
        height, width = self.model.disaster_grid.shape
        cells = [(x, y) for x in range(width) for y in range(height)]
        need_cells = sorted(
            [(cell, self.beliefs.get(cell, 0)) for cell in cells if self.beliefs.get(cell, 0) >= 4],
            key=lambda x: x[1], reverse=True
        )[:5]
        if need_cells:
            for cell, _ in need_cells:
                self.model.tokens_this_tick[cell] = self.model.tokens_this_tick.get(cell, 0) + 1
            # Append pending reward: (due tick, mode, cells)
            self.pending_rewards.append((self.model.tick + 2, list(self.tokens_this_tick.keys())[0], need_cells))

    def send_relief(self):
        max_target_cells = 5 # <-- Clearer name
        height, width = self.model.disaster_grid.shape
        cells = [(x, y) for x in range(width) for y in range(height)]
        # Find up to max_target_cells cells with highest perceived need (>=4)
        need_cells = sorted(
            [(cell, self.beliefs.get(cell, 0)) for cell in cells if self.beliefs.get(cell, 0) >= 4],
            key=lambda x: x[1], reverse=True
        )[:max_target_cells] # <-- Use new name
        if need_cells:
            # Record which cells were targeted this tick in the model's central tracker
            for cell, _ in need_cells:
                self.model.tokens_this_tick[cell] = self.model.tokens_this_tick.get(cell, 0) + 1

            # Append pending reward: (due tick, mode, cells targeted)
            # Requires knowing which information mode led to this relief decision.
            # The agent's self.tokens_this_tick stores modes used *in this step*.
            if self.tokens_this_tick: # Check if any info was requested this tick
                # If multiple modes used due to epsilon, which one gets credit/blame?
                # Original logic takes the first key: list(self.tokens_this_tick.keys())[0].
                # Let's assume this simplistic credit assignment is acceptable for now.
             responsible_mode = list(self.tokens_this_tick.keys())[0]
             self.pending_rewards.append((self.model.tick + 2, responsible_mode, need_cells))

            # Reset the agent's record of modes used for info this tick, ready for next step.
            self.tokens_this_tick = {} # Moved reset here

    def process_reward(self):
        """Compute and return numeric reward for expired pending rewards."""
        current_tick = self.model.tick
        expired = [r for r in self.pending_rewards if r[0] <= current_tick]
        self.pending_rewards = [r for r in self.pending_rewards if r[0] > current_tick]
        total_agent_reward = 0
        for tick, mode, cells in expired:
            reward = 0
            for cell, _ in cells:
                # Ensure cell is valid before accessing grid
                if 0 <= cell[0] < self.model.width and 0 <= cell[1] < self.model.height:
                    level = self.model.disaster_grid[cell[0], cell[1]] # Use (x, y) indexing
                    if level == 5:
                        reward += 5
                    elif level == 4:
                        reward += 2
                    elif level <= 2:
                        reward -= 2
                else:
                    # Handle invalid cell if needed, e.g., print warning or skip
                    # print(f"Warning: Agent {self.unique_id} pending reward for invalid cell {cell}")
                    pass # Skipping reward calculation for invalid cells for now

            total_agent_reward += reward

            # Check if mode exists in q_table, initialize if not (safety check)
            if mode not in self.q_table:
                 self.q_table[mode] = 0.0 # Initialize Q-value if mode is new

            old_q = self.q_table[mode]

            # --- Start of Agent Type Specific Logic ---
            if self.agent_type == "exploitative":
                # Ensure mode exists in trust dict before getting/updating
                if mode not in self.trust: self.trust[mode] = self.model.base_ai_trust if mode.startswith("A_") else self.model.base_trust
                trust = self.trust.get(mode, 0)

                # Increase weight on trust for Q-value update (Goal a.1)
                combined_reward = 0.9 * trust + 0.1 * reward # <-- Modified line (Exploitative)
                self.q_table[mode] = old_q + self.learning_rate * (combined_reward - old_q)

                # Trust update based on accuracy (Exploitative) - Kept original factors for exploitative
                reward_norm = reward / 5 # Normalize roughly
                trust_change_factor = 0.05 # Original factor for exploitative
                if reward > 0: # Note: Original used > 0, let's stick to that for exploitative
                    self.trust[mode] = min(1.0, self.trust.get(mode, 0) + trust_change_factor * reward_norm)
                elif reward < 0: # Use elif for clarity, handles negative reward
                    self.trust[mode] = max(0, self.trust.get(mode, 0) + trust_change_factor * reward_norm)
                # else reward == 0, no trust change for exploitative

            else: # Exploratory agent
                # Ensure mode exists in trust dict before getting/updating
                if mode not in self.trust: self.trust[mode] = self.model.base_ai_trust if mode.startswith("A_") else self.model.base_trust
                trust = self.trust.get(mode, 0)

                # Increase weight on actual reward accuracy for Q-value update (Goal b.1)
                combined_reward = 0.1 * trust + 0.9 * reward   # <-- Modified line (Exploratory)
                self.q_table[mode] = old_q + self.learning_rate * (combined_reward - old_q)

                # --- Definition of trust factors for Exploratory ---
                # Update trust based on accuracy, with stronger penalty for negative reward (Goal b.2)
                reward_norm = reward / 5 # Normalize reward roughly
                trust_change_factor_pos = 0.02 # Factor for positive reward  <-- ENSURE THIS IS HERE
                trust_change_factor_neg = 0.05 # Factor for negative reward <-- ENSURE THIS IS HERE
                # --- End Definition ---

                if reward >= 0: # Includes reward == 0 -> no change or slight increase if reward > 0
                    self.trust[mode] = min(1.0, self.trust.get(mode, 0) + trust_change_factor_pos * reward_norm) # <-- Uses the defined variable
                else: # reward < 0
                    # reward_norm is negative, so adding a positive factor * negative reward norm decreases trust
                    self.trust[mode] = max(0, self.trust.get(mode, 0) + trust_change_factor_neg * reward_norm) # <-- Uses the defined variable

        return total_agent_reward

    def step(self):
        self.sense_environment()
        self.request_information()
        self.send_relief()
        reward = self.process_reward()
        if self.agent_type == "exploitative":
            self.smooth_friend_trust()
        return reward

    def smooth_friend_trust(self):
        if self.friends:
            friend_values = [self.trust.get(f, 0) for f in self.friends]
            avg_friend = sum(friend_values) / len(friend_values)
            for friend in self.friends:
                self.trust[friend] = 0.7 * self.trust[friend] + 0.3 * avg_friend

    def decay_trust(self, candidate):
        decay_rate = 0.01 if candidate not in self.friends else 0.002
        self.trust[candidate] = max(0, self.trust[candidate] - decay_rate)
        if "ai" not in self.tokens_this_tick:
            self.trust["ai"] = max(0, self.trust["ai"] - 0.005)


class AIAgent(Agent):
    def __init__(self, unique_id, model):
        super(AIAgent, self).__init__(model)
        self.unique_id = unique_id
        self.model = model
        self.memory = {}
        self.sensed = {}
        self.total_cells = model.width * model.height
        self.cells_to_sense = int(0.2 * self.total_cells)

    def sense_environment(self):
        height, width = self.model.disaster_grid.shape
        all_cells = [(x, y) for x in range(width) for y in range(height)]
        cells = random.sample(all_cells, self.cells_to_sense)
        self.sensed = {}
        current_tick = self.model.tick
        for cell in cells:
            x, y = cell
            memory_key = (current_tick - 1, cell)
            if memory_key in self.memory and random.random() < 0.8:
                self.sensed[cell] = self.memory[memory_key]
            else:
                value = self.model.disaster_grid[x, y]
                self.sensed[cell] = value
                self.memory[(current_tick, cell)] = value

    def respond(self, human_beliefs, trust, requester_type="exploitative"):
        if not self.sensed:
            return {}
        cells = list(self.sensed.keys())
        sensed_vals = np.array([self.sensed[cell] for cell in cells])
        human_vals = np.array([human_beliefs.get(cell, sensed_vals[i]) for i, cell in enumerate(cells)])
        diff = np.abs(sensed_vals - human_vals)
        trust_factor = 1 - min(1, trust)

        alignment_factor = self.model.ai_alignment_level * (1 + trust_factor)
        # Increase sensitivity to difference for alignment adjustment
    
        adjustment = alignment_factor * (human_vals - sensed_vals) * (1 + diff)
        corrected = np.round(sensed_vals + adjustment)
        corrected = np.clip(corrected, 0, 5)
        return {cell: int(corrected[i]) for i, cell in enumerate(cells)}

    def step(self):
        self.sense_environment()


#########################################
# Disaster Model Definition
#########################################

class DisasterModel(Model):
    def __init__(self,
                 share_exploitative,
                 share_of_disaster,
                 initial_trust,
                 initial_ai_trust,
                 number_of_humans,
                 share_confirming,
                 disaster_dynamics=2,
                 shock_probability=0.1,
                 shock_magnitude=2,
                 trust_update_mode="average",
                 ai_alignment_level=0.3,
                 exploitative_correction_factor=1.0,
                 width=30, height=30,
                 lambda_parameter=0.5,
                 learning_rate=0.05,
                 epsilon=0.2,
                 ticks=150):
        super(DisasterModel, self).__init__()
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
        self.ticks = ticks

        self.grid = MultiGrid(width, height, torus=False)
        self.tick = 0
        self.tokens_this_tick = {}
        self.unmet_needs_evolution = []
        self.trust_stats = []       # (tick, AI_trust_exploit, Friend_trust_exploit, Nonfriend_trust_exploit,
                                   #         AI_trust_explor, Friend_trust_explor, Nonfriend_trust_explor)
        self.calls_data = []
        self.rewards_data = []
        self.seci_data = []         # (tick, avg_SECI_exploit, avg_SECI_explor)
        self.aeci_data = []         # (tick, avg_AECI_exploit, avg_AECI_explor)
        self.retain_aeci_data = []  # (tick, avg_retain_AECI_exploit, avg_retain_AECI_explor)
        self.retain_seci_data = []  # (tick, avg_retain_SECI_exploit, avg_retain_SECI_explor)

        # Initialize disaster grid with Gaussian decay around an epicenter.
        self.disaster_grid = np.zeros((width, height), dtype=int)
        self.baseline_grid = np.zeros((width, height), dtype=int)
        self.epicenter = (random.randint(0, width - 1), random.randint(0, height - 1))
        total_cells = width * height
        self.disaster_radius = math.sqrt(self.share_of_disaster * total_cells / math.pi)
        x, y = np.indices((width, height))
        distances = np.sqrt((x - self.epicenter[0])**2 + (y - self.epicenter[1])**2)
        sigma = self.disaster_radius / 3
        gaussian = 5 * np.exp(- (distances**2) / (2 * sigma**2))
        self.baseline_grid = np.clip(np.round(gaussian), 0, 5).astype(int)
        self.baseline_grid[self.epicenter] = 5
        self.disaster_grid[...] = self.baseline_grid

        self.social_network = nx.watts_strogatz_graph(self.num_humans, 4, 0.1)
        self.agent_list = []
        self.humans = {}

        # Create human agents.
        for i in range(self.num_humans):
            agent_type = "exploitative" if i < int(self.num_humans * self.share_exploitative) else "exploratory"
            agent = HumanAgent(unique_id=f"H_{i}", model=self, id_num=i, agent_type=agent_type,
                               share_confirming=self.share_confirming, learning_rate=self.learning_rate, epsilon=self.epsilon)
            self.humans[f"H_{i}"] = agent
            self.agent_list.append(agent)
            pos = (random.randrange(width), random.randrange(height))
            self.grid.place_agent(agent, pos)
            agent.pos = pos

        for agent in self.agent_list:
            if isinstance(agent, HumanAgent):
                agent.initialize_beliefs()

        # Set up trust, info accuracy, and friends.
        for i in range(self.num_humans):
            agent_id = f"H_{i}"
            agent = self.humans[agent_id]
            agent.friends = set(f"H_{j}" for j in self.social_network.neighbors(i) if f"H_{j}" in self.humans)
            for j in range(self.num_humans):
                if agent_id == f"H_{j}":
                    continue
                agent.trust[f"H_{j}"] = random.uniform(self.base_trust - 0.05, self.base_trust + 0.05)
                agent.info_accuracy[f"H_{j}"] = random.uniform(0.3, 0.7)
            for friend_id in agent.friends:
                agent.trust[friend_id] = min(1, agent.trust[friend_id] + 0.1)
            for k in range(self.num_ai):
                ai_trust = self.base_ai_trust if agent.agent_type == "exploitative" else self.base_ai_trust - 0.1
                agent.trust[f"A_{k}"] = random.uniform(ai_trust - 0.1, ai_trust + 0.1)
                agent.info_accuracy[f"A_{k}"] = random.uniform(0.4, 0.7)

        # Create AI agents.
        self.ais = {}
        for k in range(self.num_ai):
            ai_agent = AIAgent(unique_id=f"A_{k}", model=self)
            self.ais[f"A_{k}"] = ai_agent
            self.agent_list.append(ai_agent)
            pos = (random.randrange(width), random.randrange(height))
            self.grid.place_agent(ai_agent, pos)
            ai_agent.pos = pos

    def update_disaster(self):
        diff = self.baseline_grid - self.disaster_grid
        change = np.zeros_like(self.disaster_grid)
        change[diff > 0] = np.random.randint(1, int(self.disaster_dynamics) + 1, size=np.sum(diff > 0))
        change[diff < 0] = -np.random.randint(1, int(self.disaster_dynamics) + 1, size=np.sum(diff < 0))
        shock_mask = np.random.random(self.disaster_grid.shape) < self.shock_probability
        shocks = np.random.randint(-self.shock_magnitude, self.shock_magnitude + 1, size=self.disaster_grid.shape)
        shocks[~shock_mask] = 0
        self.disaster_grid = np.clip(self.disaster_grid + change + shocks, 0, 5)

    def step(self):
        self.tick += 1
        self.tokens_this_tick = {}
        self.update_disaster()
        random.shuffle(self.agent_list)
        
        total_reward_exploit = 0
        total_reward_explor = 0
        for agent in self.agent_list:
            r = agent.step()  # Process rewards and return numeric reward.
            if isinstance(agent, HumanAgent):
                if agent.agent_type == "exploitative":
                    total_reward_exploit += r
                else:
                    total_reward_explor += r
        self.rewards_data.append((total_reward_exploit, total_reward_explor))
        
        token_array = np.zeros((self.height, self.width), dtype=int)
        for pos, count in self.tokens_this_tick.items():
            x, y = pos
            token_array[x, y] = count
        need_mask = self.disaster_grid >= 4
        unmet = np.sum(need_mask & (token_array == 0))
        self.unmet_needs_evolution.append(unmet)
        
        # Every 5 ticks, compute additional metrics.
        if self.tick % 5 == 0:
            # --- SECI Calculation ---
            all_beliefs = []
            for agent in self.humans.values():
                all_beliefs.extend(list(agent.beliefs.values()))
            global_var = np.var(all_beliefs) if all_beliefs else 1e-6
            seci_exp_list = []
            seci_expl_list = []
            for agent in self.humans.values():
                friend_beliefs = []
                for fid in agent.friends:
                    friend = self.humans.get(fid)
                    if friend:
                        friend_beliefs.extend(list(friend.beliefs.values()))
                friend_var = np.var(friend_beliefs) if friend_beliefs else global_var
                seci_val = max(0, 1 - (friend_var / global_var))
                if agent.agent_type == "exploitative":
                    seci_exp_list.append(seci_val)
                else:
                    seci_expl_list.append(seci_val)
            self.seci_data.append((self.tick,
                                   np.mean(seci_exp_list) if seci_exp_list else 0,
                                   np.mean(seci_expl_list) if seci_expl_list else 0))
            
            # --- AECI Calculation ---
            aeci_exp = [agent.accum_calls_ai / agent.accum_calls_total for agent in self.humans.values()
                        if agent.agent_type == "exploitative" and agent.accum_calls_total > 0]
            aeci_expl = [agent.accum_calls_ai / agent.accum_calls_total for agent in self.humans.values()
                         if agent.agent_type == "exploratory" and agent.accum_calls_total > 0]
            self.aeci_data.append((self.tick,
                                   np.mean(aeci_exp) if aeci_exp else 0,
                                   np.mean(aeci_expl) if aeci_expl else 0))
            
            # --- Retainment Metrics ---
            retain_aeci_exp_list = []
            retain_aeci_expl_list = []
            retain_seci_exp_list = []
            retain_seci_expl_list = []
            for agent in self.humans.values():
                total_accepted = agent.accepted_human + agent.accepted_ai
                total_accepted = total_accepted if total_accepted > 0 else 1
                retain_aeci_val = agent.accepted_ai / total_accepted
                retain_seci_val = agent.accepted_friend / total_accepted
                if agent.agent_type == "exploitative":
                    retain_aeci_exp_list.append(retain_aeci_val)
                    retain_seci_exp_list.append(retain_seci_val)
                else:
                    retain_aeci_expl_list.append(retain_aeci_val)
                    retain_seci_expl_list.append(retain_seci_val)
            self.retain_aeci_data.append((self.tick,
                                          np.mean(retain_aeci_exp_list) if retain_aeci_exp_list else 0,
                                          np.mean(retain_aeci_expl_list) if retain_aeci_expl_list else 0))
            self.retain_seci_data.append((self.tick,
                                          np.mean(retain_seci_exp_list) if retain_seci_exp_list else 0,
                                          np.mean(retain_seci_expl_list) if retain_seci_expl_list else 0))
            
            # --- Trust Statistics ---
            trust_exp = []
            trust_expl = []
            for agent in self.humans.values():
                ai_vals = [agent.trust[k] for k in agent.trust if k.startswith("A_")]
                friend_vals = [agent.trust[k] for k in agent.trust if k.startswith("H_") and k in agent.friends]
                nonfriend_vals = [agent.trust[k] for k in agent.trust if k.startswith("H_") and k not in agent.friends]
                ai_mean = np.mean(ai_vals) if ai_vals else 0
                friend_mean = np.mean(friend_vals) if friend_vals else 0
                nonfriend_mean = np.mean(nonfriend_vals) if nonfriend_vals else 0
                if agent.agent_type == "exploitative":
                    trust_exp.append((ai_mean, friend_mean, nonfriend_mean))
                else:
                    trust_expl.append((ai_mean, friend_mean, nonfriend_mean))
            if trust_exp:
                ai_exp_mean = np.mean([x[0] for x in trust_exp])
                friend_exp_mean = np.mean([x[1] for x in trust_exp])
                nonfriend_exp_mean = np.mean([x[2] for x in trust_exp])
            else:
                ai_exp_mean, friend_exp_mean, nonfriend_exp_mean = 0, 0, 0
            if trust_expl:
                ai_expl_mean = np.mean([x[0] for x in trust_expl])
                friend_expl_mean = np.mean([x[1] for x in trust_expl])
                nonfriend_expl_mean = np.mean([x[2] for x in trust_expl])
            else:
                ai_expl_mean, friend_expl_mean, nonfriend_expl_mean = 0, 0, 0
            self.trust_stats.append((self.tick, ai_exp_mean, friend_exp_mean, nonfriend_exp_mean,
                                     ai_expl_mean, friend_expl_mean, nonfriend_expl_mean))
            
            # Reset call counters after computing AECI metrics.
            for agent in self.humans.values():
                agent.accum_calls_ai = 0
                agent.accum_calls_total = 0

#########################################
# Simulation and Experiment Functions
#########################################

def run_simulation(params):
    model = DisasterModel(**params)
    for _ in range(params.get("ticks", 150)):
        model.step()
    return model

def simulation_generator(num_runs, base_params):
    for seed in range(num_runs):
        random.seed(seed)
        np.random.seed(seed)
        model = run_simulation(base_params)
        result = {
            "trust_stats": np.array(model.trust_stats),
            "seci": np.array(model.seci_data),
            "aeci": np.array(model.aeci_data),
            "retain_aeci": np.array(model.retain_aeci_data),
            "retain_seci": np.array(model.retain_seci_data),
            "per_agent_tokens": model.unmet_needs_evolution,
            "assistance_exploit": {},  # Placeholder
            "assistance_explor": {},   # Placeholder
            "assistance_incorrect_exploit": {},  # Placeholder
            "assistance_incorrect_explor": {}    # Placeholder
        }
        yield result, model
        del model
        gc.collect()

def aggregate_simulation_results(num_runs, base_params):
    trust_list = []
    seci_list = []
    aeci_list = []
    retain_aeci_list = []
    retain_seci_list = []
    per_agent_tokens_list = []
    # Placeholders for assistance metrics:
    exploit_correct = []
    exploit_incorrect = []
    explor_correct = []
    explor_incorrect = []
    
    for result, _ in simulation_generator(num_runs, base_params):
        trust_list.append(result["trust_stats"])
        seci_list.append(result["seci"])
        aeci_list.append(result["aeci"])
        retain_aeci_list.append(result["retain_aeci"])
        retain_seci_list.append(result["retain_seci"])
        per_agent_tokens_list.append(result["per_agent_tokens"])
        exploit_correct.append(0)
        exploit_incorrect.append(0)
        explor_correct.append(0)
        explor_incorrect.append(0)
    
    trust_array = np.stack(trust_list, axis=0)
    seci_array = np.stack(seci_list, axis=0)
    aeci_array = np.stack(aeci_list, axis=0)
    retain_aeci_array = np.stack(retain_aeci_list, axis=0)
    retain_seci_array = np.stack(retain_seci_list, axis=0)
    
    assist_stats = {
        "exploit_correct": {"mean": np.mean(exploit_correct), "lower": np.percentile(exploit_correct, 25), "upper": np.percentile(exploit_correct, 75)},
        "exploit_incorrect": {"mean": np.mean(exploit_incorrect), "lower": np.percentile(exploit_incorrect, 25), "upper": np.percentile(exploit_incorrect, 75)},
        "explor_correct": {"mean": np.mean(explor_correct), "lower": np.percentile(explor_correct, 25), "upper": np.percentile(explor_correct, 75)},
        "explor_incorrect": {"mean": np.mean(explor_incorrect), "lower": np.percentile(explor_incorrect, 25), "upper": np.percentile(explor_incorrect, 75)}
    }
    ratio_stats = {
        "exploit_ratio": {"mean": 0, "lower": 0, "upper": 0},
        "explor_ratio": {"mean": 0, "lower": 0, "upper": 0}
    }
    
    return {
        "trust_stats": trust_array,
        "seci": seci_array,
        "aeci": aeci_array,
        "retain_aeci": retain_aeci_array,
        "retain_seci": retain_seci_array,
        "assist": assist_stats,
        "assist_ratio": ratio_stats,
        "per_agent_tokens": per_agent_tokens_list
    }

def experiment_share_exploitative(base_params, share_values, num_runs=20):
    results = {}
    for share in share_values:
        params = base_params.copy()
        params["share_exploitative"] = share
        print(f"Running share_exploitative = {share}")
        results[share] = aggregate_simulation_results(num_runs, params)
    return results

def experiment_ai_alignment(base_params, alignment_values, num_runs=20):
    results = {}
    for align in alignment_values:
        params = base_params.copy()
        params["ai_alignment_level"] = align
        print(f"Running ai_alignment_level = {align}")
        results[align] = aggregate_simulation_results(num_runs, params)
    return results

def experiment_disaster_dynamics(base_params, dynamics_values, shock_values, num_runs=20):
    results = {}
    for dd, sm in itertools.product(dynamics_values, shock_values):
        params = base_params.copy()
        params["disaster_dynamics"] = dd
        params["shock_magnitude"] = sm
        print(f"Running disaster_dynamics = {dd}, shock_magnitude = {sm}")
        results[(dd, sm)] = aggregate_simulation_results(num_runs, params)
    return results

def experiment_learning_trust(base_params, learning_rate_values, epsilon_values, num_runs=20):
    results = {}
    for lr, eps in itertools.product(learning_rate_values, epsilon_values):
        params = base_params.copy()
        params["learning_rate"] = lr
        params["epsilon"] = eps
        print(f"Running learning_rate = {lr}, epsilon = {eps}")
        results[(lr, eps)] = aggregate_simulation_results(num_runs, params)
    return results

#########################################
# Plotting Functions
#########################################

def plot_trust_evolution(trust_stats):
    num_runs, T, _ = trust_stats.shape
    ticks = trust_stats[0, :, 0]
    ai_exp = trust_stats[:, :, 1]
    friend_exp = trust_stats[:, :, 2]
    nonfriend_exp = trust_stats[:, :, 3]
    ai_expl = trust_stats[:, :, 4]
    friend_expl = trust_stats[:, :, 5]
    nonfriend_expl = trust_stats[:, :, 6]

    def compute_stats(arr):
        mean = np.mean(arr, axis=0)
        lower = np.percentile(arr, 25, axis=0)
        upper = np.percentile(arr, 75, axis=0)
        return mean, lower, upper

    ai_exp_mean, ai_exp_lower, ai_exp_upper = compute_stats(ai_exp)
    friend_exp_mean, friend_exp_lower, friend_exp_upper = compute_stats(friend_exp)
    nonfriend_exp_mean, nonfriend_exp_lower, nonfriend_exp_upper = compute_stats(nonfriend_exp)
    ai_expl_mean, ai_expl_lower, ai_expl_upper = compute_stats(ai_expl)
    friend_expl_mean, friend_expl_lower, friend_expl_upper = compute_stats(friend_expl)
    nonfriend_expl_mean, nonfriend_expl_lower, nonfriend_expl_upper = compute_stats(nonfriend_expl)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(ticks, ai_exp_mean, label="AI Trust", color='blue')
    axes[0].fill_between(ticks, ai_exp_lower, ai_exp_upper, color='blue', alpha=0.2)
    axes[0].plot(ticks, friend_exp_mean, label="Friend Trust", color='green')
    axes[0].fill_between(ticks, friend_exp_lower, friend_exp_upper, color='green', alpha=0.2)
    axes[0].plot(ticks, nonfriend_exp_mean, label="Non-Friend Trust", color='red')
    axes[0].fill_between(ticks, nonfriend_exp_lower, nonfriend_exp_upper, color='red', alpha=0.2)
    axes[0].set_title("Exploitative Agents Trust Evolution")
    axes[0].set_ylabel("Trust Level")
    axes[0].legend()

    axes[1].plot(ticks, ai_expl_mean, label="AI Trust", color='blue')
    axes[1].fill_between(ticks, ai_expl_lower, ai_expl_upper, color='blue', alpha=0.2)
    axes[1].plot(ticks, friend_expl_mean, label="Friend Trust", color='green')
    axes[1].fill_between(ticks, friend_expl_lower, friend_expl_upper, color='green', alpha=0.2)
    axes[1].plot(ticks, nonfriend_expl_mean, label="Non-Friend Trust", color='red')
    axes[1].fill_between(ticks, nonfriend_expl_lower, nonfriend_expl_upper, color='red', alpha=0.2)
    axes[1].set_title("Exploratory Agents Trust Evolution")
    axes[1].set_xlabel("Tick")
    axes[1].set_ylabel("Trust Level")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def plot_seci_aeci_evolution(seci_array, aeci_array):
    if seci_array.ndim == 2:
        seci_array = seci_array[np.newaxis, ...]
    if aeci_array.ndim == 2:
        aeci_array = aeci_array[np.newaxis, ...]
    if aeci_array.shape[2] < 3:
        print("Warning: AECI data is empty; plotting SECI data only.")
        ticks = seci_array[0, :, 0]
        def compute_stats(arr):
            mean = np.mean(arr, axis=0)
            lower = np.percentile(arr, 25, axis=0)
            upper = np.percentile(arr, 75, axis=0)
            return mean, lower, upper
        seci_exp_mean, seci_exp_lower, seci_exp_upper = compute_stats(seci_array[:, :, 1])
        seci_expl_mean, seci_expl_lower, seci_expl_upper = compute_stats(seci_array[:, :, 2])
        plt.figure(figsize=(8, 6))
        plt.plot(ticks, seci_exp_mean, label="Exploitative SECI", color="blue")
        plt.fill_between(ticks, seci_exp_lower, seci_exp_upper, color="blue", alpha=0.2)
        plt.plot(ticks, seci_expl_mean, label="Exploratory SECI", color="orange")
        plt.fill_between(ticks, seci_expl_lower, seci_expl_upper, color="orange", alpha=0.2)
        plt.xlabel("Tick")
        plt.ylabel("SECI")
        plt.title("SECI Evolution")
        plt.legend()
        plt.tight_layout()
        plt.show()
        return

    ticks = seci_array[0, :, 0]
    
    def compute_stats(arr):
        mean = np.mean(arr, axis=0)
        lower = np.percentile(arr, 25, axis=0)
        upper = np.percentile(arr, 75, axis=0)
        return mean, lower, upper

    seci_exp_mean, seci_exp_lower, seci_exp_upper = compute_stats(seci_array[:, :, 1])
    seci_expl_mean, seci_expl_lower, seci_expl_upper = compute_stats(seci_array[:, :, 2])
    aeci_exp_mean, aeci_exp_lower, aeci_exp_upper = compute_stats(aeci_array[:, :, 1])
    aeci_expl_mean, aeci_expl_lower, aeci_expl_upper = compute_stats(aeci_array[:, :, 2])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    axes[0].plot(ticks, seci_exp_mean, label="Exploitative", color="blue")
    axes[0].fill_between(ticks, seci_exp_lower, seci_exp_upper, color="blue", alpha=0.2)
    axes[0].plot(ticks, seci_expl_mean, label="Exploratory", color="orange")
    axes[0].fill_between(ticks, seci_expl_lower, seci_expl_upper, color="orange", alpha=0.2)
    axes[0].set_title("SECI Evolution")
    axes[0].set_xlabel("Tick")
    axes[0].set_ylabel("SECI")
    axes[0].legend()

    axes[1].plot(ticks, aeci_exp_mean, label="Exploitative", color="blue")
    axes[1].fill_between(ticks, aeci_exp_lower, aeci_exp_upper, color="blue", alpha=0.2)
    axes[1].plot(ticks, aeci_expl_mean, label="Exploratory", color="orange")
    axes[1].fill_between(ticks, aeci_expl_lower, aeci_expl_upper, color="orange", alpha=0.2)
    axes[1].set_title("AECI Evolution")
    axes[1].set_xlabel("Tick")
    axes[1].set_ylabel("AECI")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def plot_assistance(assist_stats, title_suffix=""):
    labels = ['Exploitative', 'Exploratory']
    total_exploit = assist_stats["exploit_correct"]["mean"] + assist_stats["exploit_incorrect"]["mean"]
    total_explor = assist_stats["explor_correct"]["mean"] + assist_stats["explor_incorrect"]["mean"]
    share_correct = [
        assist_stats["exploit_correct"]["mean"] / total_exploit if total_exploit > 0 else 0,
        assist_stats["explor_correct"]["mean"] / total_explor if total_explor > 0 else 0
    ]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, share_correct, width, label='Share Correct', color='skyblue')
    ax.set_ylabel("Share of Correct Tokens")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(f"Token Assistance {title_suffix}")
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_retainment_comparison(seci_data, aeci_data, retain_seci_data, retain_aeci_data, title_suffix=""):
    ticks = seci_data[0, :, 0]
    seci_exp_mean = np.mean(seci_data[:, :, 1], axis=0)
    seci_expl_mean = np.mean(seci_data[:, :, 2], axis=0)
    aeci_exp_mean = np.mean(aeci_data[:, :, 1], axis=0)
    aeci_expl_mean = np.mean(aeci_data[:, :, 2], axis=0)
    retain_seci_exp_mean = np.mean(retain_seci_data[:, :, 1], axis=0)
    retain_seci_expl_mean = np.mean(retain_seci_data[:, :, 2], axis=0)
    retain_aeci_exp_mean = np.mean(retain_aeci_data[:, :, 1], axis=0)
    retain_aeci_expl_mean = np.mean(retain_aeci_data[:, :, 2], axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    axes[0, 0].plot(ticks, seci_exp_mean, label="SECI", color="blue")
    axes[0, 0].plot(ticks, retain_seci_exp_mean, label="Retain SECI", color="green")
    axes[0, 0].set_title("Exploitative SECI vs Retainment")
    axes[0, 0].set_ylabel("Value")
    axes[0, 0].legend()

    axes[0, 1].plot(ticks, seci_expl_mean, label="SECI", color="blue")
    axes[0, 1].plot(ticks, retain_seci_expl_mean, label="Retain SECI", color="green")
    axes[0, 1].set_title("Exploratory SECI vs Retainment")
    axes[0, 1].set_ylabel("Value")
    axes[0, 1].legend()

    axes[1, 0].plot(ticks, aeci_exp_mean, label="AECI", color="orange")
    axes[1, 0].plot(ticks, retain_aeci_exp_mean, label="Retain AECI", color="red")
    axes[1, 0].set_title("Exploitative AECI vs Retainment")
    axes[1, 0].set_xlabel("Tick")
    axes[1, 0].set_ylabel("Value")
    axes[1, 0].legend()

    axes[1, 1].plot(ticks, aeci_expl_mean, label="AECI", color="orange")
    axes[1, 1].plot(ticks, retain_aeci_expl_mean, label="Retain AECI", color="red")
    axes[1, 1].set_title("Exploratory AECI vs Retainment")
    axes[1, 1].set_xlabel("Tick")
    axes[1, 1].set_ylabel("Value")
    axes[1, 1].legend()

    plt.suptitle(f"Retainment Comparison {title_suffix}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_correct_token_shares(results, share_values):
    correct_shares = []
    for share in share_values:
        assist = results[share]["assist"]
        total = assist["exploit_correct"]["mean"] + assist["exploit_incorrect"]["mean"]
        share_val = assist["exploit_correct"]["mean"] / total if total > 0 else 0
        correct_shares.append(share_val)
    plt.figure(figsize=(8,6))
    plt.plot(share_values, correct_shares, marker='o')
    plt.xlabel("Share Exploitative")
    plt.ylabel("Correct Token Share")
    plt.title("Correct Token Share vs Share Exploitative")
    plt.show()

#########################################
# Utility Function for CSV Export
#########################################

def export_results_to_csv(results, share_values, filename, experiment_name):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Experiment", "Share", "Results"])
        for share in share_values:
            writer.writerow([experiment_name, share, str(results.get(share))])


#########################################
# Main: Run Experiments and Plot Results
#########################################
if __name__ == "__main__":
    base_params = {
        "share_exploitative": 0.5,
        "share_of_disaster": 0.2,
        "initial_trust": 0.5,
        "initial_ai_trust": 0.5,
        "number_of_humans": 30,
        "share_confirming": 0.5,
        "disaster_dynamics": 2,
        "shock_probability": 0.1,
        "shock_magnitude": 2,
        "trust_update_mode": "average",
        "ai_alignment_level": 0.3,
        "exploitative_correction_factor": 1.0,
        "width": 30,
        "height": 30,
        "lambda_parameter": 0.5,
        "learning_rate": 0.05,
        "epsilon": 0.2,
        "ticks": 150
    }
    num_runs = 20
    save_dir = "agent_model_results"
    os.makedirs(save_dir, exist_ok=True)

    ##############################################
    # Experiment A: Vary share_exploitative
    ##############################################
    share_values = [0.2, 0.4, 0.6, 0.8]
    file_a_pkl = os.path.join(save_dir, "results_experiment_A.pkl")
    file_a_csv = os.path.join(save_dir, "results_experiment_A.csv")
    
    print("Running Experiment A...")
    results_a = experiment_share_exploitative(base_params, share_values, num_runs)
    with open(file_a_pkl, "wb") as f:
        pickle.dump(results_a, f)
    export_results_to_csv(results_a, share_values, file_a_csv, "Experiment A")

    for share in share_values:
        print(f"Share Exploitative = {share}")
        plot_seci_aeci_evolution(results_a[share]["seci"], results_a[share]["aeci"])
        plot_trust_evolution(results_a[share]["trust_stats"])
        plot_retainment_comparison(results_a[share]["seci"], results_a[share]["aeci"],
                                   results_a[share]["retain_seci"], results_a[share]["retain_aeci"],
                                   f"(Share={share})")
    plot_correct_token_shares(results_a, share_values)

    ##############################################
    # Experiment B: Vary AI Alignment Level
    ##############################################
    alignment_values = [0.1, 0.5, 0.9]
    results_b = experiment_ai_alignment(base_params, alignment_values, num_runs)

    aligns = sorted(results_b.keys())
    assist_exploit_means_b = [results_b[a]["assist"]["exploit_correct"]["mean"] for a in aligns]
    assist_explor_means_b = [results_b[a]["assist"]["explor_correct"]["mean"] for a in aligns]
    plt.figure(figsize=(8, 6))
    plt.plot(aligns, assist_exploit_means_b, marker="o", label="Exploitative Assistance")
    plt.plot(aligns, assist_explor_means_b, marker="s", label="Exploratory Assistance")
    plt.xlabel("AI Alignment Level")
    plt.ylabel("Final Total Correct Tokens Delivered")
    plt.title("Experiment B: Effect of AI Alignment")
    plt.legend()
    plt.show()

    for align in alignment_values:
        print(f"AI Alignment Level = {align}")
        plot_retainment_comparison(results_b[align]["seci"], results_b[align]["aeci"],
                                   results_b[align]["retain_seci"], results_b[align]["retain_aeci"],
                                   f"(Alignment={align})")

    ##############################################
    # Experiment C: Vary Disaster Dynamics and Shock Magnitude
    ##############################################
    dynamics_values = [1, 2, 3]
    shock_values = [1, 2, 3]
    results_c = experiment_disaster_dynamics(base_params, dynamics_values, shock_values, num_runs)

    exploit_matrix = np.zeros((len(dynamics_values), len(shock_values)))
    explor_matrix = np.zeros((len(dynamics_values), len(shock_values)))

    for i, dd in enumerate(dynamics_values):
        for j, sm in enumerate(shock_values):
            res = results_c[(dd, sm)]
            exploit_matrix[i, j] = res["assist"]["exploit_correct"]["mean"]
            explor_matrix[i, j] = res["assist"]["explor_correct"]["mean"]
            plot_assistance(res["assist"], f"(Dynamics={dd}, Shock={sm})")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(exploit_matrix, cmap='viridis', origin='lower')
    plt.colorbar(label='Exploitative Correct Tokens')
    plt.xticks(ticks=range(len(shock_values)), labels=shock_values)
    plt.yticks(ticks=range(len(dynamics_values)), labels=dynamics_values)
    plt.xlabel("Shock Magnitude")
    plt.ylabel("Disaster Dynamics")
    plt.title("Exploitative Assistance Heatmap")

    plt.subplot(1, 2, 2)
    plt.imshow(explor_matrix, cmap='viridis', origin='lower')
    plt.colorbar(label='Exploratory Correct Tokens')
    plt.xticks(ticks=range(len(shock_values)), labels=shock_values)
    plt.yticks(ticks=range(len(dynamics_values)), labels=dynamics_values)
    plt.xlabel("Shock Magnitude")
    plt.ylabel("Disaster Dynamics")
    plt.title("Exploratory Assistance Heatmap")
    plt.tight_layout()
    plt.show()

    ##############################################
    # Experiment D: Vary Learning Rate and Epsilon
    ##############################################
    learning_rate_values = [0.03, 0.05, 0.07]
    epsilon_values = [0.2, 0.3]
    results_d = experiment_learning_trust(base_params, learning_rate_values, epsilon_values, num_runs)

    plt.figure(figsize=(8,6))
    for eps in epsilon_values:
        means_exploit = []
        means_explor = []
        for lr in learning_rate_values:
            res = results_d[(lr, eps)]["assist"]
            means_exploit.append(res["exploit_correct"]["mean"])
            means_explor.append(res["explor_correct"]["mean"])
        plt.plot(learning_rate_values, means_exploit, marker='o', label=f"Exploitative (epsilon={eps})")
        plt.plot(learning_rate_values, means_explor, marker='s', label=f"Exploratory (epsilon={eps})")
    plt.xlabel("Learning Rate")
    plt.ylabel("Final Total Correct Tokens Delivered")
    plt.title("Experiment D: Effect of Learning Rate & Epsilon")
    plt.legend()
    plt.show()

    gc.collect()
