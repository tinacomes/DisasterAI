!pip install mesa

import os
import random
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.stats as stats
import itertools
import pickle
import gc
import csv

from mesa import Agent, Model
from mesa.space import MultiGrid

# --- Helper class to wrap candidate cells ---
class Candidate:
    def __init__(self, cell):
        self.cell = cell
    def __repr__(self):
        return f"Candidate({self.cell})"

#########################################
# Full Agent Definitions
#########################################

class HumanAgent(Agent):
    def __init__(self, unique_id, model, id_num, agent_type, share_confirming, learning_rate=0.05, epsilon=0.2):
        # Call the Mesa Agent initializer with only the model
        super(HumanAgent, self).__init__(model)
        self.unique_id = unique_id  
        self.model = model
        self.id_num = id_num
        self.agent_type = agent_type
        self.share_confirming = share_confirming
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.q_parameter = 0.95

        # Initialize trust, Q-values, beliefs, etc.
        self.trust = {}
        self.Q = {}
        self.info_accuracy = {}
        self.friends = set()
        width, height = self.model.grid.width, self.model.grid.height
        self.beliefs = {(x, y): 0 for x in range(width) for y in range(height)}
        self.tokens = 10
        
        if self.agent_type == "exploitative":
            self.D = 2.5
            self.delta = 4
        else:
            self.D = 1.0
            self.delta = 2
        
        self.pending_relief = []
        self.total_reward = 0
        self.calls_human = 0
        self.calls_ai = 0
        self.accum_calls_ai = 0
        self.accum_calls_total = 0
        
        # Missing attributes added:
        self.accepted_human = 0
        self.accepted_ai = 0
        self.accepted_friend = 0

    def choose_information_mode(self, best_human, best_ai, lambda_param, multiplier):
        total = best_human + best_ai + 1e-6
        prob_ai = (best_ai / total) * multiplier
        # For exploratory agents, choose AI with probability lambda_param if best_ai is very low
        if self.agent_type == "exploratory" and (best_ai < 1e-6 or random.random() < lambda_param):
            return "ai"
        if random.random() < prob_ai:
            return "ai"
        return "human"

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

    def sense_environment(self):
        pos = self.pos
        cells = self.model.grid.get_neighborhood(pos, moore=True, radius=1, include_center=True)
        for cell in cells:
            self.beliefs[cell] = self.model.disaster_grid[cell]
            
    def send_relief(self):
        tokens_to_send = int(self.tokens * 0.7)
        if self.agent_type == "exploitative":
            raw_cells = self.model.grid.get_neighborhood(self.pos, moore=True, radius=2, include_center=True)
            cells = list(set(tuple(int(v) for v in cell) for cell in raw_cells))
        else:
            height, width = self.model.disaster_grid.shape
            cells = [(x, y) for x in range(width) for y in range(height)]
        friend_positions = {self.model.humans[friend_id].pos for friend_id in self.friends if friend_id in self.model.humans}

        def cell_score(cell):
            x, y = cell
            belief = self.beliefs.get(cell, 0)
            score = belief
            if self.agent_type == "exploitative" and cell in friend_positions:
                score += 1.0
            elif self.agent_type == "exploratory":
                score = belief * 1.5 if belief >= 4 else belief
            existing = self.model.tokens_this_tick.get(cell, 0)
            score -= existing * 0.5
            score += random.uniform(0, 0.2)
            return score

        candidate_scores = [(cell, cell_score(cell)) for cell in cells]
        if candidate_scores:
            min_score = min(score for _, score in candidate_scores)
        else:
            min_score = 0
        adjusted_scores = [score - min_score + 0.1 for _, score in candidate_scores]
        total_score = sum(adjusted_scores) if adjusted_scores else 1
        probabilities = [score / total_score for score in adjusted_scores]
        # Wrap candidate cells in Candidate objects
        candidate_cells = [Candidate(cell) for cell, _ in candidate_scores]
        num_cells = min(tokens_to_send, len(candidate_cells))
        if num_cells > 0 and probabilities:
            selected = np.random.choice(candidate_cells, size=num_cells, replace=False, p=probabilities)
            for candidate in selected:
                cell = candidate.cell
                self.pending_relief.append((self.model.tick, None, 0, 0, cell))
                self.model.tokens_this_tick[cell] = self.model.tokens_this_tick.get(cell, 0) + 1
            self.tokens -= num_cells

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
                    diff = abs(self.beliefs[target_cell] - level)
                    self.info_accuracy[source_id] = max(0, min(1, self.info_accuracy.get(source_id, 0.5) - 0.05 * diff))
                agent_key = "exploit" if self.agent_type == "exploitative" else "explor"
                if level >= 4:
                    self.model.per_agent_tokens[agent_key][self.unique_id]["correct"] += 1
                    self.model.per_agent_tokens[agent_key][self.unique_id]["positions"].append(target_cell)
                    if self.agent_type == "exploitative":
                        self.model.assistance_exploit[target_cell] = self.model.assistance_exploit.get(target_cell, 0) + 1
                    else:
                        self.model.assistance_explor[target_cell] = self.model.assistance_explor.get(target_cell, 0) + 1
                elif level <= 2:
                    self.model.per_agent_tokens[agent_key][self.unique_id]["incorrect"] += 1
                    self.model.per_agent_tokens[agent_key][self.unique_id]["positions"].append(target_cell)
                    if self.agent_type == "exploitative":
                        self.model.assistance_incorrect_exploit[target_cell] = self.model.assistance_incorrect_exploit.get(target_cell, 0) + 1
                    else:
                        self.model.assistance_incorrect_explor[target_cell] = self.model.assistance_incorrect_explor.get(target_cell, 0) + 1
            else:
                new_pending.append(entry)
        self.pending_relief = new_pending

    def request_information(self):
        human_candidates = []
        ai_candidates = []
        if self.agent_type == "exploitative":
            for candidate in self.trust:
                if candidate.startswith("H_"):
                    bonus = 0.2 if candidate in self.friends else 0.0
                    base_q = ((self.info_accuracy.get(candidate, 0.5) * 0.2) +
                              (self.trust[candidate] * 0.8) + bonus)
                    noise = random.uniform(0, 0.1)
                    if candidate not in self.Q:
                        self.Q[candidate] = (base_q + noise) * self.q_parameter
                    human_candidates.append((candidate, self.Q[candidate]))
                elif candidate.startswith("A_"):
                    if candidate not in self.Q:
                        coverage_bonus = 1.0 + self.model.ai_alignment_level * 2
                        self.Q[candidate] = ((self.info_accuracy.get(candidate, 0.5) * 0.2) +
                                             (self.trust[candidate] * 0.8)) * self.q_parameter * coverage_bonus
                    ai_candidates.append((candidate, self.Q[candidate]))
            num_calls = 5
            multiplier = 4.0
            lambda_param = 0.15
        else:
            for candidate in self.trust:
                if candidate.startswith("H_"):
                    bonus = 0.02 if candidate in self.friends else 0.0
                    base_q = ((self.info_accuracy.get(candidate, 0.5) * 0.8) +
                              (self.trust[candidate] * 0.2) + bonus)
                    noise = random.uniform(0, 0.1)
                    if candidate not in self.Q:
                        self.Q[candidate] = (base_q + noise) * self.q_parameter
                    human_candidates.append((candidate, self.Q[candidate]))
                elif candidate.startswith("A_"):
                    if candidate not in self.Q:
                        coverage_bonus = max(0.2, 1.0 - self.model.ai_alignment_level * 2)
                        self.Q[candidate] = ((self.info_accuracy.get(candidate, 0.5) * 0.8) +
                                             (self.trust[candidate] * 0.2)) * self.q_parameter * coverage_bonus
                    ai_candidates.append((candidate, self.Q[candidate]))
            num_calls = 7
            multiplier = 1.5
            lambda_param = 0.4

        best_human = max([q for _, q in human_candidates]) if human_candidates else 0
        best_ai = max([q for _, q in ai_candidates]) if ai_candidates else 0
        mode_choice = self.choose_information_mode(best_human, best_ai, lambda_param, multiplier)

        if mode_choice == "human":
            self.last_human_call_tick = self.model.tick
            selected_candidates = self.select_candidates(human_candidates, num_calls)
            for candidate, q_val in selected_candidates:
                self.calls_human += 1
                accepted = 0
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
                                self.accepted_human += 1
                                if candidate in self.friends:
                                    self.accepted_friend += 1
                                self.update_trust(candidate, accepted=True, is_confirmation=(reported_value == old_belief))
                                self.beliefs[cell] = reported_value
                self.update_q_value(candidate, accepted)
        else:
            selected_candidates = self.select_candidates(ai_candidates, num_calls=1)
            for candidate, q_val in selected_candidates:
                self.calls_ai += 1
                accepted = 0
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
                                self.accepted_ai += 1
                                self.update_trust(candidate, accepted=True, is_confirmation=(reported_value == old_belief))
                                self.beliefs[cell] = reported_value
                            else:
                                self.update_trust(candidate, accepted=False)
                self.update_q_value(candidate, accepted)

    def update_q_value(self, candidate, reward):
        old_q = self.Q.get(candidate, 0)
        if self.agent_type == "exploitative":
            trust_factor = self.trust[candidate] * 0.7
            new_q = old_q + self.learning_rate * (reward * 0.3 + trust_factor - old_q)
        else:
            new_q = old_q + self.learning_rate * (reward * 0.8 - old_q)
        self.Q[candidate] = new_q

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

    def smooth_friend_trust(self):
        if self.friends:
            friend_values = [self.trust[f] for f in self.friends]
            avg_friend = sum(friend_values) / len(friend_values)
            for friend in self.friends:
                self.trust[friend] = 0.5 * self.trust[friend] + 0.5 * avg_friend

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
        tokens_to_send = int(self.tokens * 0.7)
        if self.agent_type == "exploitative":
            raw_cells = self.model.grid.get_neighborhood(self.pos, moore=True, radius=2, include_center=True)
            cells = list(set(tuple(int(v) for v in cell) for cell in raw_cells))
        else:
            height, width = self.model.disaster_grid.shape
            cells = [(x, y) for x in range(width) for y in range(height)]
        friend_positions = {self.model.humans[friend_id].pos for friend_id in self.friends if friend_id in self.model.humans}

        def cell_score(cell):
            x, y = cell
            belief = self.beliefs.get(cell, 0)
            score = belief
            if self.agent_type == "exploitative" and cell in friend_positions:
                score += 1.0
            elif self.agent_type == "exploratory":
                score = belief * 1.5 if belief >= 4 else belief
            existing = self.model.tokens_this_tick.get(cell, 0)
            score -= existing * 0.5
            score += random.uniform(0, 0.2)
            return score

        candidate_scores = [(cell, cell_score(cell)) for cell in cells]
        if candidate_scores:
            min_score = min(score for _, score in candidate_scores)
        else:
            min_score = 0
        adjusted_scores = [score - min_score + 0.1 for _, score in candidate_scores]
        total_score = sum(adjusted_scores) if adjusted_scores else 1
        probabilities = [score / total_score for score in adjusted_scores]
        candidate_cells = [Candidate(cell) for cell, _ in candidate_scores]
        num_cells = min(tokens_to_send, len(candidate_cells))
        if num_cells > 0 and probabilities:
            selected = np.random.choice(candidate_cells, size=num_cells, replace=False, p=probabilities)
            for candidate in selected:
                cell = candidate.cell
                self.pending_relief.append((self.model.tick, None, 0, 0, cell))
                self.model.tokens_this_tick[cell] = self.model.tokens_this_tick.get(cell, 0) + 1
            self.tokens -= num_cells

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
                    diff = abs(self.beliefs[target_cell] - level)
                    self.info_accuracy[source_id] = max(0, min(1, self.info_accuracy.get(source_id, 0.5) - 0.05 * diff))
                agent_key = "exploit" if self.agent_type == "exploitative" else "explor"
                if level >= 4:
                    self.model.per_agent_tokens[agent_key][self.unique_id]["correct"] += 1
                    self.model.per_agent_tokens[agent_key][self.unique_id]["positions"].append(target_cell)
                    if self.agent_type == "exploitative":
                        self.model.assistance_exploit[target_cell] = self.model.assistance_exploit.get(target_cell, 0) + 1
                    else:
                        self.model.assistance_explor[target_cell] = self.model.assistance_explor.get(target_cell, 0) + 1
                elif level <= 2:
                    self.model.per_agent_tokens[agent_key][self.unique_id]["incorrect"] += 1
                    self.model.per_agent_tokens[agent_key][self.unique_id]["positions"].append(target_cell)
                    if self.agent_type == "exploitative":
                        self.model.assistance_incorrect_exploit[target_cell] = self.model.assistance_incorrect_exploit.get(target_cell, 0) + 1
                    else:
                        self.model.assistance_incorrect_explor[target_cell] = self.model.assistance_incorrect_explor.get(target_cell, 0) + 1
            else:
                new_pending.append(entry)
        self.pending_relief = new_pending

    def step(self):
        self.sense_environment()
        self.request_information()
        self.send_relief()
        if self.agent_type == "exploitative":
            self.smooth_friend_trust()

class AIAgent(Agent):
    def __init__(self, unique_id, model):
        super(AIAgent, self).__init__(model)
        self.unique_id = unique_id
        self.model = model
        self.memory = {}
        self.sensed = {}
        self.sense_radius = 10

    def sense_environment(self):
        pos = self.pos
        cells = self.model.grid.get_neighborhood(pos, moore=True, radius=self.sense_radius, include_center=True)
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
# Simulation & Experiment Functions
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
            "per_agent_tokens": model.per_agent_tokens,
            "assistance_exploit": model.assistance_exploit,
            "assistance_explor": model.assistance_explor,
            "assistance_incorrect_exploit": model.assistance_incorrect_exploit,
            "assistance_incorrect_explor": model.assistance_incorrect_explor
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
        exploit_correct.append(sum(result["assistance_exploit"].values()))
        exploit_incorrect.append(sum(result["assistance_incorrect_exploit"].values()))
        explor_correct.append(sum(result["assistance_explor"].values()))
        explor_incorrect.append(sum(result["assistance_incorrect_explor"].values()))

    trust_array = np.stack(trust_list, axis=0)
    seci_array = np.stack(seci_list, axis=0)
    aeci_array = np.stack(aeci_list, axis=0)
    retain_aeci_array = np.stack(retain_aeci_list, axis=0)
    retain_seci_array = np.stack(retain_seci_list, axis=0)

    exploit_ratio = [c / (c + i) if (c + i) > 0 else 0 for c, i in zip(exploit_correct, exploit_incorrect)]
    explor_ratio = [c / (c + i) if (c + i) > 0 else 0 for c, i in zip(explor_correct, explor_incorrect)]

    assist_stats = {
        "exploit_correct": {"mean": np.mean(exploit_correct), "lower": np.percentile(exploit_correct, 25), "upper": np.percentile(exploit_correct, 75)},
        "exploit_incorrect": {"mean": np.mean(exploit_incorrect), "lower": np.percentile(exploit_incorrect, 25), "upper": np.percentile(exploit_incorrect, 75)},
        "explor_correct": {"mean": np.mean(explor_correct), "lower": np.percentile(explor_correct, 25), "upper": np.percentile(explor_correct, 75)},
        "explor_incorrect": {"mean": np.mean(explor_incorrect), "lower": np.percentile(explor_incorrect, 25), "upper": np.percentile(explor_incorrect, 75)}
    }
    ratio_stats = {
        "exploit_ratio": {"mean": np.mean(exploit_ratio), "lower": np.percentile(exploit_ratio, 25), "upper": np.percentile(exploit_ratio, 75)},
        "explor_ratio": {"mean": np.mean(explor_ratio), "lower": np.percentile(explor_ratio, 25), "upper": np.percentile(explor_ratio, 75)}
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

#########################################
# Full DisasterModel Definition
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
        Model.__init__(self)
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
        self.global_variance_data = []
        self.friend_variance_data = []

        self.assistance_exploit = {}
        self.assistance_explor = {}
        self.assistance_incorrect_exploit = {}
        self.assistance_incorrect_explor = {}
        self.unmet_needs_evolution = []
        self.per_agent_tokens = {
            "exploit": {f"H_{i}": {"correct": 0, "incorrect": 0, "positions": []}
                        for i in range(number_of_humans) if i < int(number_of_humans * share_exploitative)},
            "explor": {f"H_{i}": {"correct": 0, "incorrect": 0, "positions": []}
                       for i in range(number_of_humans) if i >= int(number_of_humans * share_exploitative)}
        }

        self.seci_data = []   # (tick, avg_SECI_exploit, avg_SECI_explor)
        self.aeci_data = []   # (tick, avg_AECI_exploit, avg_AECI_explor)
        self.retain_aeci_data = []  # (tick, avg_retain_AECI_exploit, avg_retain_AECI_explor)
        self.retain_seci_data = []  # (tick, avg_retain_SECI_exploit, avg_retain_SECI_explor)
        self.correlation_data = []
        self.trust_data = []
        self.trust_stats = []  # (tick, AI_exp, Friend_exp, NonFriend_exp, AI_expl, Friend_expl, NonFriend_expl)
        self.calls_data = []   # (tick, placeholder)
        self.rewards_data = [] # (tick, total_reward_exploit, total_reward_explor)

        self.disaster_grid = np.zeros((width, height), dtype=int)
        self.baseline_grid = np.zeros((width, height), dtype=int)
        self.epicenter = (random.randint(0, width - 1), random.randint(0, height - 1))
        total_cells = width * height
        self.disaster_radius = math.sqrt(self.share_of_disaster * total_cells / math.pi)
        x, y = np.indices((width, height))
        distances = np.sqrt((x - self.epicenter[0])**2 + (y - self.epicenter[1])**2)
        self.baseline_grid = np.where(distances < self.disaster_radius / 3, 5,
                                      np.where(distances < 2 * self.disaster_radius / 3, 4,
                                               np.where(distances < self.disaster_radius, 3, 0)))
        self.disaster_grid[...] = self.baseline_grid

        self.social_network = nx.watts_strogatz_graph(self.num_humans, 4, 0.1)
        
        self.agent_list = []
        self.humans = {}
        for i in range(self.num_humans):
            agent_type = "exploitative" if i < int(self.num_humans * self.share_exploitative) else "exploratory"
            agent = HumanAgent(unique_id=f"H_{i}", model=self, id_num=i, agent_type=agent_type,
                                share_confirming=self.share_confirming, learning_rate=self.learning_rate, epsilon=self.epsilon)
            self.humans[f"H_{i}"] = agent
            self.agent_list.append(agent)
            pos = (random.randrange(width), random.randrange(height))
            self.grid.place_agent(agent, pos)
            agent.pos = pos

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
        for agent in self.agent_list:
            agent.step()
        token_array = np.zeros((self.height, self.width), dtype=int)
        for pos, count in self.tokens_this_tick.items():
            x, y = pos
            token_array[x, y] = count
        need_mask = self.disaster_grid >= 4
        unmet = np.sum(need_mask & (token_array == 0))
        self.unmet_needs_evolution.append(unmet)

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
            if not hasattr(agent, "accum_calls_ai"):
                agent.accum_calls_ai = 0
                agent.accum_calls_total = 0
            agent.accum_calls_ai += agent.calls_ai
            agent.accum_calls_total += (agent.calls_human + agent.calls_ai)
            agent.calls_human = 0
            agent.calls_ai = 0
            agent.total_reward = 0

        for agent in self.humans.values():
            for candidate in agent.trust:
                if candidate not in called_sources:
                    agent.decay_trust(candidate)

        self.trust_data.append((self.tick, np.mean([np.mean(list(agent.trust.values())) for agent in self.humans.values()])))
        self.calls_data.append(0)
        self.rewards_data.append((total_reward_exploit, total_reward_explor))

        if self.tick % 5 == 0:
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
                seci_val = max(0, 1 - (friend_var / global_var)) if global_var > 0 else 0
                if agent.agent_type == "exploitative":
                    seci_exp_list.append(seci_val)
                else:
                    seci_expl_list.append(seci_val)
            self.seci_data.append((self.tick,
                                   np.mean(seci_exp_list) if seci_exp_list else 0,
                                   np.mean(seci_expl_list) if seci_expl_list else 0))

            aeci_exp = [agent.accum_calls_ai / agent.accum_calls_total for agent in self.humans.values()
                        if agent.agent_type == "exploitative" and agent.accum_calls_total > 0]
            aeci_expl = [agent.accum_calls_ai / agent.accum_calls_total for agent in self.humans.values()
                         if agent.agent_type == "exploratory" and agent.accum_calls_total > 0]
            self.aeci_data.append((self.tick,
                                   np.mean(aeci_exp) if aeci_exp else 0,
                                   np.mean(aeci_expl) if aeci_expl else 0))

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

            for agent in self.humans.values():
                agent.accum_calls_ai = 0
                agent.accum_calls_total = 0

    def step(self):
        self.tick += 1
        self.tokens_this_tick = {}
        self.update_disaster()
        random.shuffle(self.agent_list)
        for agent in self.agent_list:
            agent.step()

#########################################
# Simulation & Experiment Functions
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
            "per_agent_tokens": model.per_agent_tokens,
            "assistance_exploit": model.assistance_exploit,
            "assistance_explor": model.assistance_explor,
            "assistance_incorrect_exploit": model.assistance_incorrect_exploit,
            "assistance_incorrect_explor": model.assistance_incorrect_explor
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
        exploit_correct.append(sum(result["assistance_exploit"].values()))
        exploit_incorrect.append(sum(result["assistance_incorrect_exploit"].values()))
        explor_correct.append(sum(result["assistance_explor"].values()))
        explor_incorrect.append(sum(result["assistance_incorrect_explor"].values()))

    trust_array = np.stack(trust_list, axis=0)
    seci_array = np.stack(seci_list, axis=0)
    aeci_array = np.stack(aeci_list, axis=0)
    retain_aeci_array = np.stack(retain_aeci_list, axis=0)
    retain_seci_array = np.stack(retain_seci_list, axis=0)

    exploit_ratio = [c / (c + i) if (c + i) > 0 else 0 for c, i in zip(exploit_correct, exploit_incorrect)]
    explor_ratio = [c / (c + i) if (c + i) > 0 else 0 for c, i in zip(explor_correct, explor_incorrect)]

    assist_stats = {
        "exploit_correct": {"mean": np.mean(exploit_correct), "lower": np.percentile(exploit_correct, 25), "upper": np.percentile(exploit_correct, 75)},
        "exploit_incorrect": {"mean": np.mean(exploit_incorrect), "lower": np.percentile(exploit_incorrect, 25), "upper": np.percentile(exploit_incorrect, 75)},
        "explor_correct": {"mean": np.mean(explor_correct), "lower": np.percentile(explor_correct, 25), "upper": np.percentile(explor_correct, 75)},
        "explor_incorrect": {"mean": np.mean(explor_incorrect), "lower": np.percentile(explor_incorrect, 25), "upper": np.percentile(explor_incorrect, 75)}
    }
    ratio_stats = {
        "exploit_ratio": {"mean": np.mean(exploit_ratio), "lower": np.percentile(exploit_ratio, 25), "upper": np.percentile(exploit_ratio, 75)},
        "explor_ratio": {"mean": np.mean(explor_ratio), "lower": np.percentile(explor_ratio, 25), "upper": np.percentile(explor_ratio, 75)}
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

#########################################
# Simulation & Experiment Functions
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
            "per_agent_tokens": model.per_agent_tokens,
            "assistance_exploit": model.assistance_exploit,
            "assistance_explor": model.assistance_explor,
            "assistance_incorrect_exploit": model.assistance_incorrect_exploit,
            "assistance_incorrect_explor": model.assistance_incorrect_explor
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
        exploit_correct.append(sum(result["assistance_exploit"].values()))
        exploit_incorrect.append(sum(result["assistance_incorrect_exploit"].values()))
        explor_correct.append(sum(result["assistance_explor"].values()))
        explor_incorrect.append(sum(result["assistance_incorrect_explor"].values()))

    trust_array = np.stack(trust_list, axis=0)
    seci_array = np.stack(seci_list, axis=0)
    aeci_array = np.stack(aeci_list, axis=0)
    retain_aeci_array = np.stack(retain_aeci_list, axis=0)
    retain_seci_array = np.stack(retain_seci_list, axis=0)

    exploit_ratio = [c / (c + i) if (c + i) > 0 else 0 for c, i in zip(exploit_correct, exploit_incorrect)]
    explor_ratio = [c / (c + i) if (c + i) > 0 else 0 for c, i in zip(explor_correct, explor_incorrect)]

    assist_stats = {
        "exploit_correct": {"mean": np.mean(exploit_correct), "lower": np.percentile(exploit_correct, 25), "upper": np.percentile(exploit_correct, 75)},
        "exploit_incorrect": {"mean": np.mean(exploit_incorrect), "lower": np.percentile(exploit_incorrect, 25), "upper": np.percentile(exploit_incorrect, 75)},
        "explor_correct": {"mean": np.mean(explor_correct), "lower": np.percentile(explor_correct, 25), "upper": np.percentile(explor_correct, 75)},
        "explor_incorrect": {"mean": np.mean(explor_incorrect), "lower": np.percentile(explor_incorrect, 25), "upper": np.percentile(explor_incorrect, 75)}
    }
    ratio_stats = {
        "exploit_ratio": {"mean": np.mean(exploit_ratio), "lower": np.percentile(exploit_ratio, 25), "upper": np.percentile(exploit_ratio, 75)},
        "explor_ratio": {"mean": np.mean(explor_ratio), "lower": np.percentile(explor_ratio, 25), "upper": np.percentile(explor_ratio, 75)}
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

    # Experiment A: Vary share_exploitative
    share_values = [0.2, 0.4, 0.6, 0.8]
    file_a_pkl = os.path.join(save_dir, "results_experiment_A.pkl")
    file_a_csv = os.path.join(save_dir, "results_experiment_A.csv")

    if os.path.exists(file_a_pkl):
        with open(file_a_pkl, "rb") as f:
            results_a = pickle.load(f)
        print("Loaded Experiment A results.")
    else:
        print("Running Experiment A...")
        results_a = experiment_share_exploitative(base_params, share_values, num_runs)
        with open(file_a_pkl, "wb") as f:
            pickle.dump(results_a, f)
        with open(file_a_csv, "w", newline='') as csvfile:
            fieldnames = ["share", "seci_exp_mean", "seci_expl_mean", "aeci_exp_mean", "aeci_expl_mean",
                          "retain_seci_exp_mean", "retain_seci_expl_mean", "retain_aeci_exp_mean", "retain_aeci_expl_mean"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for share in share_values:
                writer.writerow({
                    "share": share,
                    "seci_exp_mean": np.mean(results_a[share]["seci"][:, 1]),
                    "seci_expl_mean": np.mean(results_a[share]["seci"][:, 2]),
                    "aeci_exp_mean": np.mean(results_a[share]["aeci"][:, 1]),
                    "aeci_expl_mean": np.mean(results_a[share]["aeci"][:, 2]),
                    "retain_seci_exp_mean": np.mean(results_a[share]["retain_seci"][:, 1]),
                    "retain_seci_expl_mean": np.mean(results_a[share]["retain_seci"][:, 2]),
                    "retain_aeci_exp_mean": np.mean(results_a[share]["retain_aeci"][:, 1]),
                    "retain_aeci_expl_mean": np.mean(results_a[share]["retain_aeci"][:, 2])
                })
        print("Experiment A saved.")

    for share in share_values:
        print(f"Share Exploitative = {share}")
        plot_seci_aeci_evolution(results_a[share]["seci"], results_a[share]["aeci"])
        plot_trust_evolution(results_a[share]["trust_stats"])
        plot_retainment_comparison(results_a[share]["seci"], results_a[share]["aeci"],
                                   results_a[share]["retain_seci"], results_a[share]["retain_aeci"],
                                   f"(Share={share})")
    
    # Experiment B: Vary AI Alignment Level
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

    # Experiment C: Vary Disaster Dynamics and Shock Magnitude
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

    # Experiment D: Vary Learning Rate and Epsilon
    learning_rate_values = [0.03, 0.05, 0.07]
    epsilon_values = [0.2, 0.3]
    results_d = experiment_learning_trust(base_params, learning_rate_values, epsilon_values, num_runs)

    plt.figure(figsize=(8,6))
    for eps in epsilon_values:
        means_exploit = []
        means_explor = []
        for lr in learning_rate_values:
            res = results_d[(lr, eps)]["assist"]
            plot_assistance(res["assist"], f"(LR={lr}, Eps={eps})")
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
