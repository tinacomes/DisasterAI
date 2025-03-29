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
import gc

from mesa import Agent, Model
from mesa.space import MultiGrid

#########################################
# Disaster Model Definition
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
                 ai_alignment_level=0.3,      # AI alignment level.
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
        self.global_variance_data = []  # For SECI variance tracking.
        self.friend_variance_data = []

        # For tracking assistance outcomes.
        self.assistance_exploit = {}
        self.assistance_explor = {}
        self.assistance_incorrect_exploit = {}
        self.assistance_incorrect_explor = {}
        self.unmet_needs_evolution = []

        # Data storage for echo chamber measures.
        self.seci_data = []  
        self.aeci_data = []
        self.correlation_data = []

        # Create disaster grid.
        self.disaster_grid = np.zeros((width, height), dtype=int)
        self.baseline_grid = np.zeros((width, height), dtype=int)
        self.epicenter = (random.randint(0, width - 1), random.randint(0, height - 1))
        total_cells = width * height
        self.disaster_radius = math.sqrt(self.share_of_disaster * total_cells / math.pi)

        # Compute baseline levels based on distance from epicenter.
        x, y = np.indices((width, height))
        distances = np.sqrt((x - self.epicenter[0])**2 + (y - self.epicenter[1])**2)
        self.baseline_grid = np.where(distances < self.disaster_radius / 3, 5,
                                     np.where(distances < 2 * self.disaster_radius / 3, 4,
                                              np.where(distances < self.disaster_radius, 3, 0)))
        self.disaster_grid[...] = self.baseline_grid

        # Create a Watts–Strogatz network for friend selection.
        self.social_network = nx.watts_strogatz_graph(self.num_humans, 4, 0.1)

        # Container for all agents.
        self.agents = []

        # Create human agents.
        self.humans = {}
        for i in range(self.num_humans):
            agent_type = "exploitative" if random.random() < self.share_exploitative else "exploratory"
            a = HumanAgent(unique_id=f"H_{i}", model=self, id_num=i, agent_type=agent_type, 
                           share_confirming=self.share_confirming, learning_rate=self.learning_rate, epsilon=self.epsilon)
            self.humans[f"H_{i}"] = a
            self.agents.append(a)
            x_coord = random.randrange(width)
            y_coord = random.randrange(height)
            self.grid.place_agent(a, (x_coord, y_coord))
            a.pos = (x_coord, y_coord)

        # Initialize per-agent token tracking.
        self.per_agent_tokens = {
            "exploit": {agent_id: {"correct": 0, "incorrect": 0, "positions": []} 
                        for agent_id in self.humans if self.humans[agent_id].agent_type == "exploitative"},
            "explor": {agent_id: {"correct": 0, "incorrect": 0, "positions": []} 
                       for agent_id in self.humans if self.humans[agent_id].agent_type == "exploratory"}
        }

        # Initialize trust and information accuracy for each human.
        self.network_trust_data = []
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
                agent.trust[f"A_{k}"] = random.uniform(self.base_ai_trust - 0.1, self.base_ai_trust + 0.1)
                agent.info_accuracy[f"A_{k}"] = random.uniform(0.4, 0.7)

        # Create AI agents.
        self.ais = {}
        for k in range(self.num_ai):
            a = AIAgent(unique_id=f"A_{k}", model=self)
            self.ais[f"A_{k}"] = a
            self.agents.append(a)
            x_coord = random.randrange(width)
            y_coord = random.randrange(height)
            self.grid.place_agent(a, (x_coord, y_coord))
            a.pos = (x_coord, y_coord)

        # Data tracking for outcomes.
        self.trust_data = []
        self.calls_data = []
        self.rewards_data = []

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

        # Shuffle agents and let them step.
        random.shuffle(self.agents)
        for agent in self.agents:
            agent.step()

        # Track unmet needs.
        token_array = np.zeros((self.height, self.width), dtype=int)
        for pos, count in self.tokens_this_tick.items():
            x, y = pos
            token_array[x, y] = count
        need_mask = self.disaster_grid >= 4
        unmet = np.sum(need_mask & (token_array == 0))
        self.unmet_needs_evolution.append(unmet)

        # Process rewards.
        total_reward_exploit = 0
        total_reward_explor = 0
        called_sources = set()
        for agent in self.humans.values():
            agent.process_relief_actions(self.tick, self.disaster_grid)
            if agent.agent_type == "exploitative":
                total_reward_exploit += agent.total_reward
            else:
                total_reward_explor += agent.total_reward
            for entry in agent.pending_relief:
                if len(entry) >= 2 and entry[1] is not None:
                    called_sources.add(entry[1])
            agent.calls_human = 0
            agent.calls_ai = 0
            agent.total_reward = 0

        # Decay trust for sources not contacted this tick.
        for agent in self.humans.values():
            for candidate in agent.trust:
                if candidate not in called_sources:
                    agent.decay_trust(candidate)

        self.trust_data.append((
            self.tick,
            np.mean([np.mean(list(agent.trust.values())) for agent in self.humans.values()])
        ))
        self.calls_data.append(0)
        self.rewards_data.append((total_reward_exploit, total_reward_explor))

#########################################
# Human Agent with Q-learning and Trust Dynamics
#########################################
class HumanAgent(Agent):
    def __init__(self, unique_id, model, id_num, agent_type, share_confirming, learning_rate=0.05, epsilon=0.2):
        super().__init__(model)
        self.unique_id = unique_id
        self.id_num = id_num
        self.agent_type = agent_type
        self.share_confirming = share_confirming
        self.learning_rate = learning_rate      # Q-learning update parameter.
        self.epsilon = epsilon                  # Exploration probability.
        self.q_parameter = 0.95                 # Scaling factor for initial Q-values.
        self.human_call_probability = 0.3       # Extra chance to force a human call.

        # Trust parameters (exposed for sensitivity analysis).
        self.trust_decay_friends = 0.0005
        self.trust_decay_nonfriends = 0.02
        self.trust_decay_ai = 0.05
        self.trust_boost_confirmation = 0.3
        self.trust_boost_acceptance = 0.1

        self.Q = {}                           # Q-values for each candidate.
        self.trust = {}                       # Trust levels for each candidate.
        self.info_accuracy = {}               # Estimated info–accuracy.
        self.friends = set()
        self.calls_human = 0
        self.calls_ai = 0
        self.last_human_call_tick = 0

        width, height = self.model.grid.width, self.model.grid.height
        self.beliefs = {(x, y): 0 for x in range(width) for y in range(height)}

        # Parameters for belief update.
        if self.agent_type == "exploitative":
            self.D = 1.0
            self.delta = 3
        else:
            self.D = 2.0
            self.delta = 2

        self.pending_relief = []
        self.total_reward = 0

    def update_q_value(self, candidate, reward):
        old_q = self.Q.get(candidate, 0)
        new_q = old_q + self.learning_rate * (reward - old_q)
        self.Q[candidate] = new_q

    def choose_information_mode(self, best_human, best_ai, lambda_param, multiplier):
        if self.model.tick - self.last_human_call_tick > 10:
            return "human"
        if random.random() < self.human_call_probability:
            return "human"
        if random.random() >= lambda_param and best_human * multiplier > best_ai:
            return "human"
        else:
            return "ai"

    def update_trust(self, candidate, accepted, is_confirmation=False):
        if accepted:
            boost = self.trust_boost_confirmation if is_confirmation else self.trust_boost_acceptance
            self.trust[candidate] = min(1, self.trust[candidate] + boost)

    def decay_trust(self, candidate):
        if candidate.startswith("H_"):
            decay = self.trust_decay_friends if candidate in self.friends else self.trust_decay_nonfriends
        else:
            decay = self.trust_decay_ai
        self.trust[candidate] = max(0, self.trust[candidate] - decay)

    def select_candidates(self, candidates, num_calls):
        selected = []
        candidate_pool = candidates.copy()
        for _ in range(min(num_calls, len(candidate_pool))):
            if random.random() < self.epsilon:
                choice = random.choice(candidate_pool)
            else:
                choice = max(candidate_pool, key=lambda x: x[1])
            selected.append(choice)
            candidate_pool.remove(choice)
        return selected

    def request_information(self):
        human_candidates = []
        ai_candidates = []
        for candidate in self.trust:
            if candidate.startswith("H_"):
                bonus = 1.0 if candidate in self.friends else 0.0
                base_q = ((self.info_accuracy.get(candidate, 0.5) * 0.2) +
                          (self.trust[candidate] * 0.8) + bonus)
                noise = random.uniform(0, 0.1)
                if candidate not in self.Q:
                    self.Q[candidate] = (base_q + noise) * self.q_parameter
                human_candidates.append((candidate, self.Q[candidate]))
            elif candidate.startswith("A_"):
                if candidate not in self.Q:
                    coverage_bonus = 0.3
                    self.Q[candidate] = ((self.info_accuracy.get(candidate, 0.5) * 0.6) +
                                         (self.trust[candidate] * 0.4)) * self.q_parameter * coverage_bonus
                ai_candidates.append((candidate, self.Q[candidate]))
        best_human = max([q for _, q in human_candidates]) if human_candidates else 0
        best_ai = max([q for _, q in ai_candidates]) if ai_candidates else 0
        multiplier = 4.0 if self.agent_type == "exploitative" else 1.5
        lambda_param = 0.15 if self.agent_type == "exploitative" else 0.4

        mode_choice = self.choose_information_mode(best_human, best_ai, lambda_param, multiplier)

        if mode_choice == "human":
            self.last_human_call_tick = self.model.tick
            num_calls = 5 if self.agent_type == "exploitative" else 7
            selected_candidates = self.select_candidates(human_candidates, num_calls)
            for candidate, q_val in selected_candidates:
                self.calls_human += 1
                accepted = 0
                confirmations = 0
                other = self.model.humans.get(candidate)
                if other is not None:
                    rep = other.provide_information_full()
                    if rep is not None:
                        for cell, reported_value in rep.items():
                            old_belief = self.beliefs[cell]
                            d = abs(reported_value - old_belief)
                            P_accept = 1.0 if d == 0 else (self.D ** self.delta) / ((d ** self.delta) + (self.D ** self.delta))
                            if random.random() < P_accept:
                                accepted += 1
                                if reported_value == old_belief and self.agent_type == "exploitative":
                                    confirmations += 1
                                self.update_trust(candidate, accepted=True, is_confirmation=(reported_value==old_belief))
                self.update_q_value(candidate, accepted)
        else:
            selected_candidates = self.select_candidates(ai_candidates, num_calls=1)
            for candidate, q_val in selected_candidates:
                self.calls_ai += 1
                accepted = 0
                confirmations = 0
                other = self.model.ais.get(candidate)
                if other is not None:
                    rep = other.provide_information_full(self.beliefs, trust=self.trust[candidate], agent_type=self.agent_type)
                    if rep is not None:
                        for cell, reported_value in rep.items():
                            old_belief = self.beliefs[cell]
                            d = abs(reported_value - old_belief)
                            P_accept = 1.0 if d == 0 else (self.D ** self.delta) / ((d ** self.delta) + (self.D ** self.delta))
                            if random.random() < P_accept:
                                accepted += 1
                                if reported_value == old_belief and self.agent_type == "exploitative":
                                    confirmations += 1
                                self.update_trust(candidate, accepted=True, is_confirmation=(reported_value==old_belief))
                            else:
                                self.update_trust(candidate, accepted=False)
                self.update_q_value(candidate, accepted)

    def smooth_friend_trust(self):
        if self.friends:
            friend_trust_values = [self.trust[f] for f in self.friends]
            avg_friend_trust = sum(friend_trust_values) / len(friend_trust_values)
            for friend in self.friends:
                self.trust[friend] = 0.5 * self.trust[friend] + 0.5 * avg_friend_trust

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

    def send_relief(self):
        self.tokens = 10
        tokens_to_send = self.tokens * 0.7
        if self.agent_type == "exploitative":
            raw_cells = self.model.grid.get_neighborhood(self.pos, moore=True, radius=2, include_center=True)
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
            score = belief
            if self.agent_type == "exploitative":
                if cell in friend_positions:
                    score += 0.5
            else:
                if cell in friend_positions:
                    score += 0.2
            existing_tokens = self.model.tokens_this_tick.get(cell, 0)
            diversity_penalty = existing_tokens * 0.5
            score -= diversity_penalty
            score += random.uniform(0, 0.2)
            return score
        candidate_scores = [(cell, cell_score(cell)) for cell in cells]
        min_score = min(score for _, score in candidate_scores)
        adjusted_scores = [score - min_score + 0.1 for _, score in candidate_scores]
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
                agent_type_key = "exploit" if self.agent_type == "exploitative" else "explor"
                if level >= 4:
                    self.model.per_agent_tokens[agent_type_key][self.unique_id]["correct"] += 1
                    self.model.per_agent_tokens[agent_type_key][self.unique_id]["positions"].append(target_cell)
                    if self.agent_type == "exploitative":
                        self.model.assistance_exploit[target_cell] = self.model.assistance_exploit.get(target_cell, 0) + 1
                    else:
                        self.model.assistance_explor[target_cell] = self.model.assistance_explor.get(target_cell, 0) + 1
                elif level <= 2:
                    self.model.per_agent_tokens[agent_type_key][self.unique_id]["incorrect"] += 1
                    self.model.per_agent_tokens[agent_type_key][self.unique_id]["positions"].append(target_cell)
                    if self.agent_type == "exploitative":
                        self.model.assistance_incorrect_exploit[target_cell] = self.model.assistance_incorrect_exploit.get(target_cell, 0) + 1
                    else:
                        self.model.assistance_incorrect_explor[target_cell] = self.model.assistance_incorrect_explor.get(target_cell, 0) + 1
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
            pass  # (Optional: update delayed beliefs.)
        if self.agent_type == "exploitative":
            self.smooth_friend_trust()

#########################################
# AI Agent Definition (similar to original)
#########################################
class AIAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(model)
        self.unique_id = unique_id
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
        trust_factor = 1 - min(1, trust)
        alignment_factor = self.model.ai_alignment_level * (1 + trust_factor)
        adjustment = alignment_factor * (human_vals - sensed_vals) * (1 + diff / 5)
        corrected = np.round(sensed_vals + adjustment)
        corrected = np.clip(corrected, 0, 5)
        return {cell: int(corrected[i]) for i, cell in enumerate(cells)}

    def step(self):
        self.sense_environment()

#########################################
# Simulation and Experiment Functions
#########################################
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

def simulation_generator(num_runs, base_params):
    for seed in range(num_runs):
        random.seed(seed)
        np.random.seed(seed)
        model = run_simulation(base_params)
        result = {
            "trust": np.array(model.trust_data),
            "seci": np.array(model.seci_data) if model.seci_data else np.empty((0,)),
            "aeci": np.array(model.aeci_data) if model.aeci_data else np.empty((0,)),
            "assist_exploit": sum(model.assistance_exploit.values()),
            "assist_explor": sum(model.assistance_explor.values()),
            "assist_incorrect_exploit": sum(model.assistance_incorrect_exploit.values()),
            "assist_incorrect_explor": sum(model.assistance_incorrect_explor.values()),
            "per_agent_tokens": model.per_agent_tokens
        }
        yield result, model
        del model
        gc.collect()

def aggregate_simulation_results(num_runs, base_params):
    trust_list = []
    seci_list = []
    aeci_list = []
    assist_exploit_list = []
    assist_explor_list = []
    assist_incorrect_exploit_list = []
    assist_incorrect_explor_list = []
    per_agent_tokens_list = []
    for result, model in simulation_generator(num_runs, base_params):
        trust_list.append(result["trust"])
        seci_list.append(result["seci"])
        aeci_list.append(result["aeci"])
        assist_exploit_list.append(result["assist_exploit"])
        assist_explor_list.append(result["assist_explor"])
        assist_incorrect_exploit_list.append(result["assist_incorrect_exploit"])
        assist_incorrect_explor_list.append(result["assist_incorrect_explor"])
        per_agent_tokens_list.append(result["per_agent_tokens"])
    trust_array = np.stack(trust_list, axis=0)
    trust_mean = np.mean(trust_array, axis=0)
    trust_lower = np.percentile(trust_array, 25, axis=0)
    trust_upper = np.percentile(trust_array, 75, axis=0)
    seci_mean = np.mean(seci_list, axis=0) if seci_list and seci_list[0].size > 0 else np.array([])
    seci_lower = np.percentile(seci_list, 25, axis=0) if seci_list and seci_list[0].size > 0 else np.array([])
    seci_upper = np.percentile(seci_list, 75, axis=0) if seci_list and seci_list[0].size > 0 else np.array([])
    aeci_mean = np.mean(aeci_list, axis=0) if aeci_list and aeci_list[0].size > 0 else np.array([])
    aeci_lower = np.percentile(aeci_list, 25, axis=0) if aeci_list and aeci_list[0].size > 0 else np.array([])
    aeci_upper = np.percentile(aeci_list, 75, axis=0) if aeci_list and aeci_list[0].size > 0 else np.array([])
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

#########################################
# Experiment Definitions
#########################################
def experiment_share_exploitative(base_params, share_values, num_runs=20):
    results = {}
    for share in share_values:
        params = base_params.copy()
        params["share_exploitative"] = share
        print("Running share_exploitative =", share)
        results[share] = aggregate_simulation_results(num_runs, params)
    return results

def experiment_ai_alignment(base_params, alignment_values, num_runs=20):
    results = {}
    for align in alignment_values:
        params = base_params.copy()
        params["ai_alignment_level"] = align
        print("Running ai_alignment_level =", align)
        results[align] = aggregate_simulation_results(num_runs, params)
    return results

def experiment_disaster_dynamics(base_params, dynamics_values, shock_values, num_runs=20):
    results = {}
    for dd, sm in itertools.product(dynamics_values, shock_values):
        params = base_params.copy()
        params["disaster_dynamics"] = dd
        params["shock_magnitude"] = sm
        print("Running disaster_dynamics =", dd, "shock_magnitude =", sm)
        results[(dd, sm)] = aggregate_simulation_results(num_runs, params)
    return results

def experiment_learning_trust(base_params, learning_rate_values, epsilon_values, num_runs=20):
    results = {}
    for lr, eps in itertools.product(learning_rate_values, epsilon_values):
        params = base_params.copy()
        params["learning_rate"] = lr
        params["epsilon"] = eps
        print("Running learning_rate =", lr, "epsilon =", eps)
        results[(lr, eps)] = aggregate_simulation_results(num_runs, params)
    return results

#########################################
# MAIN: Run Experiments and Plot Results
#########################################
if __name__ == "__main__":
    # Define base parameters.
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
    num_runs = 20  # Adjust for quicker runs if needed.

    # -----------------------
    # Experiment A: Vary share_exploitative
    # -----------------------
    share_values = [0.3, 0.5, 0.7]
    results_a = experiment_share_exploitative(base_params, share_values, num_runs)
    
    # Plot: Final Assistance Delivered vs. Share Exploitative.
    shares = sorted(results_a.keys())
    assist_exploit_means = [results_a[s]["assist"]["exploit"]["mean"] for s in shares]
    assist_explor_means = [results_a[s]["assist"]["explor"]["mean"] for s in shares]
    plt.figure(figsize=(8, 6))
    plt.plot(shares, assist_exploit_means, marker="o", label="Exploitative Assistance")
    plt.plot(shares, assist_explor_means, marker="s", label="Exploratory Assistance")
    plt.xlabel("Share Exploitative")
    plt.ylabel("Final Total Tokens Delivered")
    plt.title("Experiment A: Effect of Share Exploitative")
    plt.legend()
    plt.show()

    # -----------------------
    # Experiment B: Vary AI Alignment Level
    # -----------------------
    alignment_values = [0.1, 0.3, 0.5]
    results_b = experiment_ai_alignment(base_params, alignment_values, num_runs)
    
    # Plot: Final Assistance Delivered vs. AI Alignment Level.
    aligns = sorted(results_b.keys())
    assist_exploit_means_b = [results_b[a]["assist"]["exploit"]["mean"] for a in aligns]
    assist_explor_means_b = [results_b[a]["assist"]["explor"]["mean"] for a in aligns]
    plt.figure(figsize=(8, 6))
    plt.plot(aligns, assist_exploit_means_b, marker="o", label="Exploitative Assistance")
    plt.plot(aligns, assist_explor_means_b, marker="s", label="Exploratory Assistance")
    plt.xlabel("AI Alignment Level")
    plt.ylabel("Final Total Tokens Delivered")
    plt.title("Experiment B: Effect of AI Alignment")
    plt.legend()
    plt.show()

    # -----------------------
    # Experiment C: Vary Disaster Dynamics and Shock Magnitude
    # -----------------------
    dynamics_values = [1, 2, 3]
    shock_values = [1, 2, 3]
    results_c = experiment_disaster_dynamics(base_params, dynamics_values, shock_values, num_runs)
    
    # Plot: Heatmaps for Final Assistance Delivered.
    exploit_matrix = np.zeros((len(dynamics_values), len(shock_values)))
    explor_matrix = np.zeros((len(dynamics_values), len(shock_values)))
    for i, dd in enumerate(dynamics_values):
        for j, sm in enumerate(shock_values):
            res = results_c[(dd, sm)]
            exploit_matrix[i, j] = res["assist"]["exploit"]["mean"]
            explor_matrix[i, j] = res["assist"]["explor"]["mean"]
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(exploit_matrix, cmap='viridis', origin='lower')
    plt.colorbar(label='Exploitative Assistance')
    plt.xticks(ticks=range(len(shock_values)), labels=shock_values)
    plt.yticks(ticks=range(len(dynamics_values)), labels=dynamics_values)
    plt.xlabel("Shock Magnitude")
    plt.ylabel("Disaster Dynamics")
    plt.title("Exploitative Assistance")
    
    plt.subplot(1, 2, 2)
    plt.imshow(explor_matrix, cmap='viridis', origin='lower')
    plt.colorbar(label='Exploratory Assistance')
    plt.xticks(ticks=range(len(shock_values)), labels=shock_values)
    plt.yticks(ticks=range(len(dynamics_values)), labels=dynamics_values)
    plt.xlabel("Shock Magnitude")
    plt.ylabel("Disaster Dynamics")
    plt.title("Exploratory Assistance")
    plt.tight_layout()
    plt.show()

    # -----------------------
    # Experiment D: Vary Learning Rate and Epsilon
    # -----------------------
    learning_rate_values = [0.03, 0.05, 0.07]
    epsilon_values = [0.2, 0.3]
    results_d = experiment_learning_trust(base_params, learning_rate_values, epsilon_values, num_runs)
    
    # Plot: Final Assistance Delivered vs. Learning Rate for each Epsilon.
    plt.figure(figsize=(8, 6))
    for eps in epsilon_values:
        means_exploit = []
        means_explor = []
        for lr in learning_rate_values:
            res = results_d[(lr, eps)]
            means_exploit.append(res["assist"]["exploit"]["mean"])
            means_explor.append(res["assist"]["explor"]["mean"])
        plt.plot(learning_rate_values, means_exploit, marker='o', label=f"Exploitative (epsilon={eps})")
        plt.plot(learning_rate_values, means_explor, marker='s', label=f"Exploratory (epsilon={eps})")
    plt.xlabel("Learning Rate")
    plt.ylabel("Final Total Tokens Delivered")
    plt.title("Experiment D: Effect of Learning Rate & Epsilon")
    plt.legend()
    plt.show()

    gc.collect()
