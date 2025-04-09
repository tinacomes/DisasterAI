# Install mesa if not already installed
!pip install mesa

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
import os
import math

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
    def __init__(self, unique_id, model, id_num, agent_type, share_confirming,
                 learning_rate=0.05, epsilon=0.2,
                 # --- Q-learning & Behavior Tuning Parameters ---
                 exploit_trust_weight=0.7,    # Q-target: For 'human' mode, weight for avg FRIEND trust (exploiter)
                 self_confirm_weight=0.9,     # Q-target: For 'self_action', weight for confirmation value (exploiter)
                 self_confirm_boost=1.5,      # Multiplier for avg confidence to get confirmation value V_C (tune)
                 confirmation_q_lr=0.2,       # Learning rate for Q-boost on confirmation
                 exploit_ai_trust_weight=0.95,# Q-target: For 'A_k' modes, trust weight (exploiter) - very high
                 explore_reward_weight=0.8,   # Q-target: For all modes, reward weight (exploratory)
                 exploit_friend_bias=1.0,     # Action Selection: Bias added to 'human' score (exploiter) (tune)
                 exploit_self_bias=0.5):      # Action Selection: Bias added to 'self_action' score (exploiter) (tune)

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

        # --- Store Q-learning & Behavior Parameters ---
        self.exploit_trust_weight = exploit_trust_weight
        self.self_confirm_weight = self_confirm_weight
        self.self_confirm_boost = self_confirm_boost
        self.confirmation_q_lr = confirmation_q_lr
        self.exploit_ai_trust_weight = exploit_ai_trust_weight
        self.explore_reward_weight = explore_reward_weight
        self.exploit_friend_bias = exploit_friend_bias
        self.exploit_self_bias = exploit_self_bias


        # --- Agent State ---
        self.beliefs = {} # {(x, y): {'level': L, 'confidence': C}}
        self.trust = {f"A_{k}": model.base_ai_trust for k in range(model.num_ai)} # Trust in AI agents
        # Human trust initialized later in model setup
        self.friends = set() # Use set for efficient checking ('H_j' format)
        self.pending_rewards = [] # [(tick_due, mode, [(cell, belief_level), ...]), ...]
        self.tokens_this_tick = {} # Tracks mode choice leading to send_relief THIS tick

        # --- Q-Table for Source Values ---
        self.q_table = {f"A_{k}": 0.0 for k in range(model.num_ai)}
        self.q_table["human"] = 0.05 # Represents generic value of querying humans
        self.q_table["self_action"] = 0.0

        # --- Belief Update Parameters ---
        # These control how beliefs change when info is ACCEPTED (separate from Q-learning)
        self.D = 3.0 if agent_type == "exploitative" else 1.5 # Acceptance threshold parameter
        self.delta = 3.0 if agent_type == "exploitative" else 1.5 # Acceptance sensitivity parameter
        self.belief_learning_rate = 0.8 if agent_type == "exploratory" else 0.5 # How much belief shifts towards accepted info

        # --- Other Parameters & Counters ---
        self.trust_update_mode = model.trust_update_mode # Affects trust increment size? Seems used in removed code? Check usage.
        # self.multiplier = 2.0 if agent_type == "exploitative" else 1.0 # Seems unused? Review if needed.
        # self.info_accuracy = {} # Seems unused? Review if needed.

        # Call/Acceptance Counters
        self.accum_calls_total = 0
        self.accum_calls_ai = 0
        self.accum_calls_human = 0
        self.accepted_human = 0
        self.accepted_friend = 0
        self.accepted_ai = 0

        # Performance Counters
        self.correct_targets = 0
        self.incorrect_targets = 0

    def initialize_beliefs(self):
        height, width = self.model.disaster_grid.shape
        for x in range(width):
            for y in range(height):
                self.beliefs[(x, y)] = {'level': 0, 'confidence': 0.1}  # Initialize with level 0 and low confidence
        self.sense_environment()


    def sense_environment(self):
        pos = self.pos
        radius = 2 if self.agent_type == "exploitative" else 3 #exploratory agents 'look' further
        cells = self.model.grid.get_neighborhood(pos, moore=True, radius=radius, include_center=True)
        for cell in cells:
            if 0 <= cell[0] < self.model.width and 0 <= cell[1] < self.model.height:
                actual = self.model.disaster_grid[cell[0], cell[1]] # Use (x,y)
                noise_roll = random.random()
                if noise_roll < 0.1: # 10% chance slight error
                    belief_level = max(0, min(5, actual + random.choice([-1, 1])))
                    belief_conf = 0.7 # Lower confidence if noisy read
                elif noise_roll < 0.3: # 20% chance larger error (original logic) - REMOVED, simplified noise
                    # Let's simplify: 80% accurate read, 20% noisy read by +/- 1
                    belief_level = max(0, min(5, actual + random.choice([-1, 1])))
                    belief_conf = 0.7 # Lower confidence if noisy read
                else: # 70% Accurate read
                    belief_level = actual
                    belief_conf = 0.99 # High confidence for direct sensing

            # Update belief with level and confidence
            self.beliefs[cell] = {'level': belief_level, 'confidence': belief_conf}
        # else: cell is outside grid, ignore



    def report_beliefs(self, caller_pos):
        # Ensure self.pos access is correct if it's a tuple
        try:
            current_pos_level = self.model.disaster_grid[self.pos[0], self.pos[1]]
        except IndexError:
            # Handle cases where pos might be invalid temporarily (e.g., during setup)
            current_pos_level = 0 # Assume safe if position is invalid

        # Check disaster level at the agent's own position
        if current_pos_level >= 4 and random.random() < 0.1: # 10% chance of not reporting if on a disaster affected grid cell
            return {}

        radius = 2 if self.agent_type == "exploitative" else 3
        cells = self.model.grid.get_neighborhood(caller_pos, moore=True, radius=radius, include_center=True)
        report = {}
        for cell in cells:
            # Retrieve the belief dictionary for the cell, provide default dict if unknown
            belief_info = self.beliefs.get(cell, {'level': 0, 'confidence': 0.1})

            # Ensure belief_info is a dictionary before accessing 'level'
            current_level = belief_info.get('level', 0) if isinstance(belief_info, dict) else belief_info # Handles default or potential old int format

            # Apply noise to the level (not the dictionary)
            if random.random() < 0.1: # 10% chance to report noisy value
                noisy_level = max(0, min(5, current_level + random.choice([-1, 1])))
                # Store the integer level in the report
                report[cell] = noisy_level
            else:
                # Store the current believed integer level in the report
                report[cell] = current_level

        return report # Report dictionary contains integer levels as values

    def apply_trust_decay(self):
         """Applies a slow decay to all trust relationships."""
         decay_rate = 0.001 if self.agent_type == "exploitative" else 0.001# Slow general decay rate per step
         friend_decay_rate = 0 if self.agent_type == "exploitative" else 0.001# # Slower decay for friends
         # Iterate over a copy of keys because dictionary size might change (though unlikely here)
         for source_id in list(self.trust.keys()):
              rate = friend_decay_rate if source_id in self.friends else decay_rate
              self.trust[source_id] = max(0, self.trust[source_id] - rate)

    def choose_information_mode(self):
        """
        Chooses an information source mode (e.g., 'A_k', 'human', 'self_action')
        based on Q-values, exploration rate (epsilon), and agent type biases.
        """
        available_modes = list(self.q_table.keys())
        if not available_modes:
            # print(f"Warning: Agent {self.unique_id} has no available modes in Q-table. Defaulting to self_action.")
            return "self_action" # Fallback

        if random.random() < self.epsilon:
            # Exploration: Choose randomly from available modes
            chosen_mode = random.choice(available_modes)
            # print(f"Agent {self.unique_id} exploring mode: chooses random {chosen_mode}") # Debug
            return chosen_mode
        else:
            # Exploitation: Choose based on Q-value + bias
            mode_scores = {}
            for mode in available_modes:
                q_value = self.q_table.get(mode, 0.0)
                bias = 0.0

                # --- Apply Bias for Exploitative Agents during Exploitation ---
                # This encourages choosing humans (friends) or self-action over AI,
                # unless the AI's learned Q-value significantly overcomes the bias.
                if self.agent_type == "exploitative":
                    if mode == "human":
                        bias = self.exploit_friend_bias # Add positive bias
                    elif mode == "self_action":
                        bias = self.exploit_self_bias # Add smaller positive bias
                    # No explicit bias for AI modes (A_k)

                mode_scores[mode] = q_value + bias
                # print(f"  Agent {self.unique_id} ({self.agent_type[:4]}): Mode={mode}, Q={q_value:.2f}, Bias={bias:.2f}, Score={mode_scores[mode]:.2f}") # Debug scores

            # Choose the mode with the highest score
            best_mode = max(mode_scores, key=mode_scores.get)
            # print(f"Agent {self.unique_id} exploiting mode: chooses {best_mode} (Score={mode_scores[best_mode]:.2f})") # Debug
            return best_mode

    def request_information(self):
        """
        Requests information based on chosen mode, updates beliefs if accepted,
        and updates trust based on acceptance. NO Q-UPDATE HERE.
        """
        mode_choice = self.choose_information_mode()
        reports = {} # Dictionary to store received info {cell: value}
        source_agent_id_for_credit = None # Tracks specific agent ID for trust/confidence updates

        report_sources = {} # Ensures variable exists regardless of mode
        # Track the chosen mode if external info is sought
        if mode_choice != "self_action":
             self.tokens_this_tick[mode_choice] = self.tokens_this_tick.get(mode_choice, 0) + 1

        # --- Handle Mode Choice ---
        if mode_choice == "self_action":
            # No external request needed, exit. send_relief will handle 'self_action'.
            return

        elif mode_choice.startswith("A_"):
            # Request from specific AI
            ai_id = mode_choice
            if ai_id in self.model.ais:
                chosen_ai = self.model.ais[ai_id]
                current_trust_in_ai = self.trust.get(ai_id, self.model.base_ai_trust)
                reports = chosen_ai.respond(self.beliefs, current_trust_in_ai, self.agent_type)
                self.accum_calls_ai += 1
                source_agent_id_for_credit = mode_choice # Credit the specific AI
            else:
                # print(f"Warning: Agent {self.unique_id} chose AI {ai_id} but it doesn't exist.")
                return # Skip if AI not found

        elif mode_choice == "human":
            # Query multiple humans based on agent type strategy
            all_humans = [agent for agent in self.model.humans.values() if agent.unique_id != self.unique_id]
            if not all_humans: return

            # Sort humans by trust (handle missing trust with low default)
            sorted_humans = sorted(all_humans, key=lambda x: self.trust.get(x.unique_id, 0.1), reverse=True)
            humans_to_query = set()

            # --- Agent-Type Specific Human Querying Strategy ---
            if self.agent_type == "exploitative":
                num_trusted = 4 # Query top trusted (likely includes friends)
                for agent in sorted_humans[:num_trusted]: humans_to_query.add(agent)
                if random.random() < 0.15: # Chance to query a random non-friend
                    non_friends = [h for h in all_humans if h.unique_id not in self.friends]
                    if non_friends: humans_to_query.add(random.choice(non_friends))
            else: # Exploratory: Query diverse set
                num_trusted = 2; num_random_others = 2; num_least_trusted = 1
                for agent in sorted_humans[:num_trusted]: humans_to_query.add(agent)
                others = [h for h in sorted_humans[num_trusted:] if h not in humans_to_query]
                random.shuffle(others)
                for agent in others[:num_random_others]: humans_to_query.add(agent)
                if len(sorted_humans) > num_trusted + num_random_others:
                    least_trusted_candidates = sorted_humans[-(num_least_trusted + len(humans_to_query)):]
                    for agent in least_trusted_candidates:
                         if agent not in humans_to_query:
                             humans_to_query.add(agent)
                             if len(humans_to_query) >= num_trusted + num_random_others + num_least_trusted: break

            query_list = list(humans_to_query)
            random.shuffle(query_list)

            # --- Aggregate Reports (Example: Use most trusted source per cell) ---
            aggregated_reports = {}
            report_sources = {} # Tracks source for each cell's final reported value
            for candidate in query_list:
                 rep = candidate.report_beliefs(self.pos)
                 for cell, value in rep.items():
                     current_source_trust = self.trust.get(candidate.unique_id, 0.1)
                     # If cell not seen OR new source is more trusted than previous source for this cell
                     if cell not in aggregated_reports or current_source_trust > self.trust.get(report_sources.get(cell), 0.0):
                         aggregated_reports[cell] = value
                         report_sources[cell] = candidate.unique_id

            reports = aggregated_reports

            # Assign credit for trust updates to the single most trusted human queried overall
            if query_list:
                 query_list_sorted_by_trust = sorted(query_list, key=lambda x: self.trust.get(x.unique_id, 0.1), reverse=True)
                 source_agent_id_for_credit = query_list_sorted_by_trust[0].unique_id

            self.accum_calls_human += 1

        else:
             # print(f"Warning: Agent {self.unique_id} chose invalid mode '{mode_choice}' in request_information")
             return

        # Common counter
        if mode_choice != "self_action":
             self.accum_calls_total += 1

        # --- Acceptance & Belief Update Logic ---
        # (This block determines if info is accepted and how beliefs/confidence change)
        # Kept mostly as is from your original code + previous refinement structure.
        # Trust update based on ACCEPTANCE happens here.
        if not reports: return # Skip if no reports received

        for cell, reported_value in reports.items():
             belief_info = self.beliefs.get(cell, {'level': 0, 'confidence': 0.1})
             old_belief_level = belief_info.get('level', 0)
             reported_level_int = int(round(reported_value)) # Ensure integer comparison
             d = abs(reported_level_int - old_belief_level)

             # Determine acceptance probability
             responsible_source_id = report_sources.get(cell, source_agent_id_for_credit) # Use cell-specific source if available, else overall credit ID
             friend_weight = 1.0
             if self.agent_type == "exploitative" and responsible_source_id and responsible_source_id in self.friends:
                 friend_weight = 2.5 # Friend boost for acceptance

             P_accept = 1.0 if d == 0 else (self.D ** self.delta) / ((d ** self.delta) + (self.D ** self.delta)) * friend_weight
             P_accept = max(0.0, min(1.0, P_accept)) # Ensure probability is valid

             if random.random() < P_accept:
                 # --- Info Accepted ---
                 if responsible_source_id:
                     if responsible_source_id not in self.trust: # Safety init
                         base = self.model.base_ai_trust if responsible_source_id.startswith("A_") else self.model.base_trust
                         self.trust[responsible_source_id] = base

                     # --- Belief & Confidence Update ---
                     # (Using the detailed logic from previous refinement which includes biases, intensity, high-trust overwrite)
                     old_belief_info = self.beliefs.get(cell, {'level': 0, 'confidence': 0.1})
                     old_level = old_belief_info.get('level', 0)
                     old_confidence = old_belief_info.get('confidence', 0.1)
                     intensity_factor = 1.0 if d == 0 else (self.D ** self.delta) / ((d ** self.delta) + (self.D ** self.delta))
                     level_diff = reported_level_int - old_level
                     min_intensity_for_correction = 0.4
                     effective_intensity = max(intensity_factor, min_intensity_for_correction) if level_diff > 1 else intensity_factor
                     base_adjustment = self.belief_learning_rate * effective_intensity * level_diff
                     final_adjustment = base_adjustment
                     is_confirming_high_need = (old_level >= 4 and reported_level_int >= 4)
                     is_disconfirming_high_need = (old_level >= 4 and reported_level_int < 3)
                     is_suggesting_increase_from_low = (old_level < 3 and reported_level_int >= 3)
                     if self.agent_type == "exploitative":
                         if is_confirming_high_need and d <= 1: final_adjustment *= 1.2
                         elif is_disconfirming_high_need and old_confidence > 0.5: final_adjustment *= 0.7
                         elif is_suggesting_increase_from_low and level_diff > 0: final_adjustment *= 2.0 # Ensure level_diff > 0 too
                     else: # Exploratory
                         if d >= 2 and old_confidence < 0.5: final_adjustment *= 1.3
                         if is_suggesting_increase_from_low and level_diff > 0: final_adjustment *= 1.5 # Ensure level_diff > 0 too
                     new_level_float = old_level + final_adjustment
                     source_trust = self.trust.get(responsible_source_id, 0.1)
                     trust_threshold = 0.80; low_confidence_threshold = 0.5
                     if source_trust >= trust_threshold and old_confidence < low_confidence_threshold:
                         new_level_float = old_level + 0.75 * (reported_level_int - old_level)
                     new_level = max(0, min(5, int(round(new_level_float))))
                     # Confidence Update
                     confidence_target = source_trust
                     if is_confirming_high_need and self.agent_type == "exploitative": confidence_target = min(1.0, source_trust + 0.2)
                     elif is_disconfirming_high_need and self.agent_type == "exploitative" and old_confidence > 0.5: confidence_target = max(0.0, source_trust - 0.2)
                     if self.agent_type == "exploitative": confidence_alpha = 0.3 * intensity_factor; base_confidence_bump = 0.005
                     else: confidence_alpha = 0.2; base_confidence_bump = 0.02
                     if abs(new_level - old_level) >= 2:
                         if self.agent_type == "exploitative": confidence_alpha *= 0.9
                         else: confidence_alpha *= 0.95
                     new_confidence = old_confidence + base_confidence_bump + (confidence_target - old_confidence) * confidence_alpha
                     if d == 0 and old_level > 1: new_confidence += 0.05 * source_trust
                     new_confidence = max(0.01, min(0.99, new_confidence))
                     self.beliefs[cell] = {'level': new_level, 'confidence': new_confidence}
                     # --- End Belief & Confidence Update ---


                     # --- Trust Update Based on Acceptance ---
                     trust_increment = 0.0
                     if self.agent_type == "exploitative":
                     # --- Exploiter-Specific Updates on Acceptance ---
                        confirmation_score = 1.0 / (1.0 + d) # Score = 1 if d=0, decreases as d increases

                        # 1. Immediate Q-Value Boost based on Confirmation (for AI sources)
                        if responsible_source_id.startswith("A_"):
                            current_q = self.q_table.get(responsible_source_id, 0.0)
                            # Target for this update is the confirmation score itself
                            q_target_confirm = confirmation_score
                            q_boost = self.confirmation_q_lr * (q_target_confirm - current_q)
                            self.q_table[responsible_source_id] = current_q + q_boost
                            # print(f"  Ag {self.unique_id} (Exp) Q-Boost AI {responsible_source_id}: d={d}, score={confirmation_score:.2f}, boost={q_boost:.3f}") # Debug

                        # 2. Trust Increment based on Confirmation (Both Human & AI)
                        # Base increment size depends on trust_update_mode (as before)
                        delta = 0.075 if self.trust_update_mode == "average" else 0.25
                        # Apply friend weight ONLY if source is human friend
                        friend_weight_accept = 1.0
                        if responsible_source_id.startswith("H_") and responsible_source_id in self.friends:
                             friend_weight_accept = 2.5 # Use original friend weight logic here
                        # Scale increment by confirmation score
                        trust_increment = delta * friend_weight_accept * (confirmation_score ** 2)

                     else: # Exploratory
                         delta = 0.04 if self.trust_update_mode == "average" else 0.05
                         trust_increment = delta # No friend weight boost for explorers

                     if trust_increment > 0: # Only update if there's a positive increment to mimic trust inertia
                        new_trust_target = min(1.0, self.trust.get(responsible_source_id, 0.5) + trust_increment) # Default 0.5
                        self.trust[responsible_source_id] = 0.9 * self.trust.get(responsible_source_id, 0.5) + 0.1 * new_trust_target


                     # Update acceptance counters
                     if responsible_source_id.startswith("H_"):
                         self.accepted_human += 1
                         if responsible_source_id in self.friends: self.accepted_friend += 1
                     elif responsible_source_id.startswith("A_"):
                         self.accepted_ai += 1
                 else:
                     # print(f"Warning: Accepted info for cell {cell} but no responsible source ID.")
                     pass # Cannot update trust or counters without source ID

        # Note: Q-update based on outcome happens in process_reward, not here.

    def send_relief(self):
        """Determines where to send relief based on beliefs and agent type,
           then queues the action for reward processing, crediting the responsible source.
           Resets the mode choice tracker for the next tick."""

        max_target_cells = 5
        min_forced_tokens = 1 # If no good targets found, send at least this many

        # --- Calculate Scores for potential target cells based on agent type ---
        cell_scores = []
        max_believed_level_this_agent = -1
        for cell, belief_info in self.beliefs.items():
             if isinstance(belief_info, dict):
                 level = belief_info.get('level', 0)
                 confidence = belief_info.get('confidence', 0.1)
                 if level > max_believed_level_this_agent: max_believed_level_this_agent = level

                 if level >= 3: # Only consider targeting cells believed to be level 3+
                     score = 0.0
                     if self.agent_type == "exploitative":
                         min_exploit_conf = 0.6 # Confidence threshold
                         if confidence >= min_exploit_conf:
                             # Favor high level AND high confidence
                             score = (level / 5.0) * (confidence**1.5)
                     else: # Exploratory
                         level_weight = level / 5.0
                         uncertainty_weight = (1.0 - confidence)
                         # Favor uncertainty and distance
                         score = (uncertainty_weight * 0.6) + (level_weight * 0.4) # Adjusted weights
                         distance_from_agent = math.sqrt((cell[0] - self.pos[0])**2 + (cell[1] - self.pos[1])**2)
                         max_dist = math.sqrt(self.model.width**2 + self.model.height**2)
                         normalized_distance = distance_from_agent / max_dist if max_dist > 0 else 0
                         spatial_bonus_factor = 0.2
                         score += spatial_bonus_factor * normalized_distance

                     if score > 0.01: # Threshold to consider targeting
                         cell_scores.append({'cell': cell, 'score': score, 'level': level})

        # --- Select Top Targets based on score ---
        need_cells_sorted_by_score = sorted(cell_scores, key=lambda x: x['score'], reverse=True)
        top_cells_info = need_cells_sorted_by_score[:max_target_cells]
        targeted_cells_coords = [item['cell'] for item in top_cells_info]
        final_need_cells_for_reward = [(item['cell'], item['level']) for item in top_cells_info] # (cell, believed_level)

        # --- Fallback Mechanism if no targets scored high enough ---
        if not targeted_cells_coords and min_forced_tokens > 0:
            best_guess_level = max_believed_level_this_agent
            best_guess_cells = []
            if best_guess_level > -1: # Find cells matching max believed level
                 for cell, belief_info in self.beliefs.items():
                      if isinstance(belief_info, dict) and belief_info.get('level', -1) == best_guess_level:
                           best_guess_cells.append(cell)

            if best_guess_cells: # If found cells at max level
                 random.shuffle(best_guess_cells)
                 forced_targets = best_guess_cells[:min_forced_tokens]
                 targeted_cells_coords = forced_targets # Target these cells
                 # Use the guessed level for the reward list
                 final_need_cells_for_reward = [(cell, best_guess_level) for cell in forced_targets]
            else: # Fallback failed completely
                 targeted_cells_coords = []
                 final_need_cells_for_reward = []

        # --- Record Tokens & Queue Pending Rewards ---
        if targeted_cells_coords: # Only if targets were chosen (scored or fallback)
            # Determine the 'mode' responsible for this action
            responsible_mode = None
            if self.tokens_this_tick: # Was info requested this tick?
                # Use the mode recorded during request_information (e.g., "A_1", "human")
                responsible_mode = list(self.tokens_this_tick.keys())[0]
            else:
                # No info requested, agent acted based on own beliefs
                responsible_mode = "self_action"

            # Record tokens sent for model-level tracking/visualization
            agent_type_key = 'exploit' if self.agent_type == 'exploitative' else 'explor'
            for cell in targeted_cells_coords:
                 if 0 <= cell[0] < self.model.width and 0 <= cell[1] < self.model.height:
                     if cell not in self.model.tokens_this_tick:
                         self.model.tokens_this_tick[cell] = {'exploit': 0, 'explor': 0}
                     self.model.tokens_this_tick[cell][agent_type_key] += 1

            # Add to this agent's list of rewards to be processed later
            # The 'responsible_mode' links this outcome back to the Q-value update
            self.pending_rewards.append((self.model.tick + 2, responsible_mode, final_need_cells_for_reward))

        # --- Reset the agent's mode choice tracker for the next tick ---
        # This dictionary is only used to link request_information choice to send_relief outcome within a single tick.
        self.tokens_this_tick = {}

    def process_reward(self):
        """
        Processes expired rewards, calculates reward based on actual outcomes,
        updates Q-table for the responsible source mode, and updates trust based on reward.
        """
        current_tick = self.model.tick
        expired = [r for r in self.pending_rewards if r[0] <= current_tick]
        self.pending_rewards = [r for r in self.pending_rewards if r[0] > current_tick]
        total_agent_reward_this_tick = 0

        for reward_tick, mode, cells_and_beliefs in expired:
            batch_reward = 0 # Reward accumulated for this specific batch/outcome
            if not cells_and_beliefs: continue # Skip if no cells were targeted

            correct_in_batch = 0
            incorrect_in_batch = 0

            # --- Calculate Reward based on true state vs action ---
            for cell, belief_at_action_time in cells_and_beliefs:
                 if 0 <= cell[0] < self.model.width and 0 <= cell[1] < self.model.height:
                     actual_level = self.model.disaster_grid[cell[0], cell[1]]
                     is_correct_target = actual_level >= 3
                     if is_correct_target: correct_in_batch += 1
                     else: incorrect_in_batch += 1
                     # Calculate points
                     cell_reward = 0
                     if actual_level == 5: cell_reward = 5
                     elif actual_level == 4: cell_reward = 2
                     elif actual_level == 3: cell_reward = 1
                     elif actual_level <= 2: cell_reward = -1.5
                     batch_reward += cell_reward
                     # Confidence boost for correct identification (optional)
                     if is_correct_target and cell in self.beliefs and isinstance(self.beliefs[cell], dict):
                         correct_high_need_boost = 0.15
                         current_conf = self.beliefs[cell].get('confidence', 0.1)
                         self.beliefs[cell]['confidence'] = min(0.99, current_conf + correct_high_need_boost)
                 else: pass # Invalid cell coord

            # Update agent's overall performance counters
            self.correct_targets += correct_in_batch
            self.incorrect_targets += incorrect_in_batch
            total_agent_reward_this_tick += batch_reward # Add batch reward to agent's total for the tick

            # --- Q-Table Update ---
            # Update the Q-value associated with the 'mode' (source) that led to this outcome
            old_q = self.q_table.get(mode, 0.0) # Get current Q-value, default 0
            target_q_value = 0.0 # Initialize target value for the Q-update

            # Calculate target based on agent type and responsible mode
            if self.agent_type == "exploitative":
                if mode == "self_action":
                    # Calculate current average confidence as proxy for confirmation value
                    confidences = [b.get('confidence', 0.1) for b in self.beliefs.values() if isinstance(b, dict)]
                    avg_confidence = np.mean(confidences) if confidences else 0.1
                    confirmation_value = self.self_confirm_boost * avg_confidence # Scale avg confidence
                    # Target heavily weights confirmation, lightly weights reward
                    target_q_value = (self.self_confirm_weight * confirmation_value +
                                      (1 - self.self_confirm_weight) * batch_reward)

                elif mode == "human":
                    # Calculate average trust in current friends
                    friend_trusts = [self.trust.get(f_id, 0.1) for f_id in self.friends if f_id in self.trust]
                    avg_friend_trust = np.mean(friend_trusts) if friend_trusts else 0.1 # Default low
                    # Target weights average friend trust + reward
                    target_q_value = (self.exploit_trust_weight * avg_friend_trust +
                                      (1 - self.exploit_trust_weight) * batch_reward)

                elif mode.startswith("A_"): # AI source
                     trust = self.trust.get(mode, 0.3) # Default lowish trust if missing
                     # Target very heavily weights trust, minimal reward influence
                     adjusted_ai_trust_weight = 0.70 # NEW - Tune this (e.g., 0.7 to 0.9)
                     target_q_value = (adjusted_ai_trust_weight * trust +
                                       (1 - adjusted_ai_trust_weight) * batch_reward)

                else: # Fallback for unexpected modes
                      trust = self.trust.get(mode, 0.3)
                      # Default to trust-weighted target
                      target_q_value = (self.exploit_trust_weight * trust + (1 - self.exploit_trust_weight) * batch_reward)

            else: # Exploratory Agent
                 # Target weights environmental reward heavily, trust lightly
                 trust = self.trust.get(mode, 0.5) # Default neutral trust
                 target_q_value = ((1 - self.explore_reward_weight) * trust +
                                   self.explore_reward_weight * batch_reward)

            # Apply the Q-learning update rule
            effective_learning_rate = max(0, self.learning_rate) # Ensure non-negative LR
            new_q = old_q + effective_learning_rate * (target_q_value - old_q)
            self.q_table[mode] = new_q # Update the Q-table

            # --- Trust Update Based on Reward Outcome ---
            # Update trust in the specific AI or the generic 'human' mode based on reward
            if mode != "self_action":
                 # Normalize reward
                 avg_reward_per_cell = batch_reward / len(cells_and_beliefs) # Avoid div by zero checked earlier
                 max_reward = 5; min_reward = -1.5 # Define reward range
                 if avg_reward_per_cell > 0: scaled_norm = avg_reward_per_cell / max_reward
                 elif avg_reward_per_cell < 0: scaled_norm = avg_reward_per_cell / abs(min_reward)
                 else: scaled_norm = 0
                 scaled_norm = max(-1.0, min(1.0, scaled_norm)) # Clip normalization

                 current_trust = self.trust.get(mode, 0.5) # Default neutral if missing

                 # Determine trust change factor based on agent type
                 if self.agent_type == "exploitative":
                    trust_change_factor_pos = 0.05 # For positive reward
                    trust_change_factor_neg = 0.1 # Example: MUCH larger for negative reward
                 else: #exploratory
                    trust_change_factor_pos = 0.05 # For positive reward
                    trust_change_factor_neg = 0.25 # Example: larger for negative reward
                 if scaled_norm >= 0:
                    trust_change = trust_change_factor_pos * scaled_norm
                 else:
                    trust_change = trust_change_factor_neg * scaled_norm # Larger negative impact
                
                 self.trust[mode] = max(0.0, min(1.0, current_trust + trust_change)) # Apply update with bounds

        return total_agent_reward_this_tick

    def step(self):
        self.sense_environment()
        self.request_information()
        self.send_relief()
        reward = self.process_reward()
        if self.agent_type == "exploitative":
            self.smooth_friend_trust()

        self.apply_trust_decay() # Apply slow decay to all relationships

        confidence_decay_rate = 0.005 # Start very small and tune

        #confidence decay
        for cell in self.beliefs:
            if isinstance(self.beliefs[cell], dict):
              current_conf = self.beliefs[cell].get('confidence', 0.1)
              # Prevent decay below a minimum floor, maybe slightly above initial
              min_conf_floor = 0.05
              self.beliefs[cell]['confidence'] = max(min_conf_floor, current_conf - confidence_decay_rate)
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
        self.cells_to_sense = int(0.15 * self.total_cells)

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
                if random.random() < 0.05: # 5% chance of AI sensing noise
                    value = max(0, min(5, value + random.choice([-1, 1])))
                self.sensed[cell] = value
                self.memory[(current_tick, cell)] = value

    def respond(self, human_beliefs, trust_in_ai, requester_agent_type):
        """
        Responds with potentially adjusted beliefs based on AI sensing, human beliefs,
        trust level, and alignment parameter.
        """
        if not self.sensed: return {} # Return empty if AI hasn't sensed anything

        cells = list(self.sensed.keys())
        # Ensure sensed_vals are integers
        sensed_vals = np.array([int(self.sensed.get(cell, 0)) for cell in cells])

        # Extract human belief levels for the cells the AI sensed
        human_levels_list = []
        for i, cell in enumerate(cells):
            belief_info = human_beliefs.get(cell) # Get human belief dict/value
            if isinstance(belief_info, dict):
                # Get level from dict, default to AI's sensed value if key missing
                human_level = belief_info.get('level', sensed_vals[i])
            else:
                # Default to AI's sensed value if human has no belief or it's not a dict
                human_level = sensed_vals[i]
            human_levels_list.append(int(human_level)) # Ensure integer
        human_vals = np.array(human_levels_list)

        # --- Simplified AI Alignment Logic ---
        alignment_strength = self.model.ai_alignment_level # Base alignment (0 to 1)

        # Factor in trust: Less alignment adjustment if trust is high
        clipped_trust = max(0.0, min(1.0, trust_in_ai)) # Ensure trust is [0, 1]
        trust_influence = (1.0 - clipped_trust) # Scales adjustment (0=no adjust, 1=full align)

        # Calculate difference between human belief and AI sensing
        belief_difference = human_vals - sensed_vals

        # Calculate adjustment, scaled by alignment strength and inverse trust
        adjustment = alignment_strength * trust_influence * belief_difference

        # Apply adjustment to AI's sensed values
        corrected = np.round(sensed_vals + adjustment)
        # Clip results to valid belief range [0, 5]
        corrected = np.clip(corrected, 0, 5)

        # Return adjusted beliefs as dictionary {cell: adjusted_level}
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
                 epsilon=0.3, #q-learning / exploration rate
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
        self.belief_error_data = [] # (tick, avg_MAE_exploit, avg_MAE_explor)
        self.belief_variance_data = [] # (tick, var_exploit, var_explor)

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
                # agent.info_accuracy[f"H_{j}"] = random.uniform(0.3, 0.7)
            for friend_id in agent.friends:
                agent.trust[friend_id] = min(1, agent.trust[friend_id] + 0.1)
            for k in range(self.num_ai):
                ai_trust = self.base_ai_trust if agent.agent_type == "exploitative" else self.base_ai_trust - 0.1
                agent.trust[f"A_{k}"] = random.uniform(ai_trust - 0.1, ai_trust + 0.1)
                # agent.info_accuracy[f"A_{k}"] = random.uniform(0.4, 0.7)

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

        token_array = np.zeros((self.width, self.height), dtype=int) # Array expects integers
        for pos, count_dict in self.tokens_this_tick.items():
            x, y = pos
            # Check bounds before accessing array
            if 0 <= x < self.width and 0 <= y < self.height:
                # Calculate TOTAL tokens for the cell from the dictionary
                total_tokens_for_cell = count_dict.get('exploit', 0) + count_dict.get('explor', 0)
                # Assign the integer sum to the array
                token_array[x, y] = total_tokens_for_cell
            else:
                print(f"Warning: Invalid position key in tokens_this_tick: {pos}") # Safety check

        need_mask = self.disaster_grid >= 4
        unmet = np.sum(need_mask & (token_array == 0))
        self.unmet_needs_evolution.append(unmet)

        # Call mapping function periodically and/or at the end
        if self.tick == 1 or self.tick % 25 == 0 or self.tick == self.ticks:
              plot_grid_state(self, self.tick, save_dir="agent_model_results/grid_plots") # Save to subfolder

        if self.tick % 50 == 0 or self.tick == self.ticks: # Example frequency
            plot_belief_histogram(self, self.tick)
        #   Choose representative agents based on ID or type
            exploit_sample = [a.unique_id for a in self.humans.values() if a.agent_type=='exploitative'][:1]
            explor_sample = [a.unique_id for a in self.humans.values() if a.agent_type=='exploratory'][:1]
            if exploit_sample and explor_sample:
                plot_individual_beliefs(self, [exploit_sample[0], explor_sample[0]], self.tick)

        # Every 5 ticks, compute additional metrics.
        if self.tick % 5 == 0:
            # --- SECI Calculation ---
            all_belief_levels = []
            for agent in self.humans.values():
                friend_belief_levels = [] # Initialize HERE

                for belief_info in agent.beliefs.values():
                    if isinstance(belief_info, dict):
                        all_belief_levels.append(belief_info.get('level', 0))

            global_var = np.var(all_belief_levels) if all_belief_levels else 1e-6
            seci_exp_list = []
            seci_expl_list = []
            for agent in self.humans.values():
                friend_beliefs = []
                for fid in agent.friends:
                    friend = self.humans.get(fid)
                    if friend:
                        for belief_info in friend.beliefs.values():
                             # Ensure it's a dictionary and get the level, default 0
                            if isinstance(belief_info, dict):
                                friend_belief_levels.append(belief_info.get('level', 0))

                friend_var = np.var(friend_belief_levels) if friend_belief_levels else global_var

                # Add safety check for division by (near) zero
                if global_var < 1e-9:
                     seci_val = 0 # Assign 0 if global variance is effectively zero
                else:
                     seci_val = max(0, 1 - (friend_var / global_var))


                if agent.agent_type == "exploitative":
                    seci_exp_list.append(seci_val)
                else:
                    seci_expl_list.append(seci_val)
            self.seci_data.append((self.tick,
                                   np.mean(seci_exp_list) if seci_exp_list else 0,
                                   np.mean(seci_expl_list) if seci_expl_list else 0))

            # --- Belief Accuracy Metric (MAE) ---
            total_mae_exploit = 0
            total_mae_explor = 0
            count_exploit = 0
            count_explor = 0
            num_cells = self.width * self.height

            if num_cells > 0: # Avoid division by zero
                for agent in self.humans.values():
                     agent_error_sum = 0
                     valid_belief_count = 0
                     for x in range(self.width):
                         for y in range(self.height):
                             cell = (x, y)
                             actual = self.disaster_grid[x, y]
                             belief_info = agent.beliefs.get(cell, {}) # Get belief dict or empty dict
                             if isinstance(belief_info, dict):
                                 belief = belief_info.get('level', -1) # Use -1 if level missing
                                 if belief != -1: # Only count cells where agent has a level belief
                                     agent_error_sum += abs(actual - belief)
                                     valid_belief_count += 1

                     avg_agent_mae = agent_error_sum / valid_belief_count if valid_belief_count > 0 else 0

                     if agent.agent_type == "exploitative":
                         total_mae_exploit += avg_agent_mae
                         count_exploit += 1
                     else:
                         total_mae_explor += avg_agent_mae
                         count_explor += 1

            avg_mae_exploit = total_mae_exploit / count_exploit if count_exploit > 0 else 0
            avg_mae_explor = total_mae_explor / count_explor if count_explor > 0 else 0
            self.belief_error_data.append((self.tick, avg_mae_exploit, avg_mae_explor))
            # --- End Belief Accuracy ---

            # --- Within-Type Belief Variance ---
            exploit_beliefs_levels = []
            explor_beliefs_levels = []
            for agent in self.humans.values():
                target_list = exploit_beliefs_levels if agent.agent_type == "exploitative" else explor_beliefs_levels
                for belief_info in agent.beliefs.values():
                     if isinstance(belief_info, dict):
                          target_list.append(belief_info.get('level', 0)) # Add level

            var_exploit = np.var(exploit_beliefs_levels) if exploit_beliefs_levels else 0
            var_explor = np.var(explor_beliefs_levels) if explor_beliefs_levels else 0
            self.belief_variance_data.append((self.tick, var_exploit, var_explor))
            # --- End Within-Type Belief Variance ---


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
            "belief_error": np.array(model.belief_error_data),
            "belief_variance": np.array(model.belief_variance_data),
            "unmet_needs_evolution": model.unmet_needs_evolution
            #"per_agent_tokens": model.unmet_needs_evolution,
            #"assistance_exploit": {},  # Placeholder
            #"assistance_explor": {},   # Placeholder
            #"assistance_incorrect_exploit": {},  # Placeholder
            #"assistance_incorrect_explor": {}    # Placeholder
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
    unmet_needs_evolution_list = []
    belief_error_list = []       # Belief tracking
    belief_variance_list = []    # variance

    # --- Lists to store TOTAL correct/incorrect targets PER RUN ---
    exploit_correct_per_run = []
    exploit_incorrect_per_run = []
    explor_correct_per_run = []
    explor_incorrect_per_run = []
    # ---

    # Ensure simulation_generator yields the model object

    for result, model in simulation_generator(num_runs, base_params):
        trust_list.append(result["trust_stats"])
        seci_list.append(result["seci"])
        aeci_list.append(result["aeci"])
        retain_aeci_list.append(result["retain_aeci"])
        retain_seci_list.append(result["retain_seci"])
        unmet_needs_evolution_list.append(result.get("unmet_needs_evolution", [])) # Use .get for safety
        belief_error_list.append(result.get("belief_error", []))     # ADDED .get for safety
        belief_variance_list.append(result.get("belief_variance", [])) # ADDED .get for safety

         # --- Aggregate correct/incorrect targets for THIS run ---
        run_exploit_correct = 0
        run_exploit_incorrect = 0
        run_explor_correct = 0
        run_explor_incorrect = 0

        # Iterate through agents in the completed model run
        for agent in model.humans.values():
            if agent.agent_type == "exploitative":
                run_exploit_correct += agent.correct_targets
                run_exploit_incorrect += agent.incorrect_targets
            else: # Exploratory
                run_explor_correct += agent.correct_targets
                run_explor_incorrect += agent.incorrect_targets

        # Append the totals for THIS run to the overall lists
        exploit_correct_per_run.append(run_exploit_correct)
        exploit_incorrect_per_run.append(run_exploit_incorrect)
        explor_correct_per_run.append(run_explor_correct)
        explor_incorrect_per_run.append(run_explor_incorrect)
        # --- End Aggregation for Run ---

        # Clean up memory
        del model
        gc.collect()


       # --- Calculate Stats based on aggregated data ---
    # Stack arrays for metrics that evolve over time (add empty checks)
    trust_array = np.stack(trust_list, axis=0) if trust_list else np.array([])
    seci_array = np.stack(seci_list, axis=0) if seci_list else np.array([])
    aeci_array = np.stack(aeci_list, axis=0) if aeci_list else np.array([])
    retain_aeci_array = np.stack(retain_aeci_list, axis=0) if retain_aeci_list else np.array([])
    retain_seci_array = np.stack(retain_seci_list, axis=0) if retain_seci_list else np.array([])
    belief_error_array = np.stack(belief_error_list) if belief_error_list and all(isinstance(i, np.ndarray) for i in belief_error_list) else np.array([]) # ADDED
    belief_variance_array = np.stack(belief_variance_list) if belief_variance_list and all(isinstance(i, np.ndarray) for i in belief_variance_list) else np.array([]) # ADDED
    # Note: unmet_needs_evolution_list remains a list of lists/arrays


    # Helper function for calculating mean/percentiles
    def calculate_metric_stats(data_list):
        # Handle case where simulation fails or produces no data
        valid_data = [d for d in data_list if d is not None]
        if not valid_data: return {"mean": 0, "lower": 0, "upper": 0}
        return {
            "mean": np.mean(valid_data),
            "lower": np.percentile(valid_data, 25),
            "upper": np.percentile(valid_data, 75)
        }

    # Calculate stats for assistance (using the lists populated with actual run totals)
    assist_stats = {
        "exploit_correct": calculate_metric_stats(exploit_correct_per_run),
        "exploit_incorrect": calculate_metric_stats(exploit_incorrect_per_run),
        "explor_correct": calculate_metric_stats(explor_correct_per_run),
        "explor_incorrect": calculate_metric_stats(explor_incorrect_per_run)
    }

    # Calculate ratio stats (Share of Correct Tokens) based on the *means*
    total_exploit_mean = assist_stats["exploit_correct"]["mean"] + assist_stats["exploit_incorrect"]["mean"]
    total_explor_mean = assist_stats["explor_correct"]["mean"] + assist_stats["explor_incorrect"]["mean"]
    ratio_stats = {
        "exploit_ratio": {
            "mean": assist_stats["exploit_correct"]["mean"] / total_exploit_mean if total_exploit_mean > 0 else 0,
            # Calculating percentiles for ratios is complex; report mean ratio for now.
            "lower": 0, "upper": 0
        },
        "explor_ratio": {
            "mean": assist_stats["explor_correct"]["mean"] / total_explor_mean if total_explor_mean > 0 else 0,
            "lower": 0, "upper": 0
        }
    }
    # --- End Calculate Stats ---

    return {
        "trust_stats": trust_array,
        "seci": seci_array,
        "aeci": aeci_array,
        "retain_aeci": retain_aeci_array,
        "retain_seci": retain_seci_array,
        "belief_error": belief_error_array,         # ADDED
        "belief_variance": belief_variance_array,   # ADDED
        "assist": assist_stats,         # Contains mean/percentiles of raw counts
        "assist_ratio": ratio_stats,   # Contains mean share of correct tokens
        "unmet_needs_evol": unmet_needs_evolution_list, # Pass the list of lists
        # --- ADD RAW ASSIST COUNTS ---
        "raw_assist_counts": {
             "exploit_correct": exploit_correct_per_run,
             "exploit_incorrect": exploit_incorrect_per_run,
             "explor_correct": explor_correct_per_run,
             "explor_incorrect": explor_incorrect_per_run
        }
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

def plot_grid_state(model, tick, save_dir="grid_plots"):
    """Plots the disaster grid state, agent locations, and tokens sent."""
    os.makedirs(save_dir, exist_ok=True) # Create directory if it doesn't exist
    # ---2x3 subplots ---
    fig, ax = plt.subplots(2, 3, figsize=(21, 12)) # Increased width for 3 columns
    fig.suptitle(f"Model State at Tick {tick}", fontsize=16)

    # --- Plot 1: Disaster Grid ---
    # Transpose grid if needed for imshow orientation (depends on grid creation)
    # Assuming model.disaster_grid shape is (width, height) -> needs transpose for imshow
    grid_display = model.disaster_grid.T
    im = ax[0, 0].imshow(grid_display, cmap='viridis', origin='lower', vmin=0, vmax=5, interpolation='nearest')
    ax[0, 0].set_title("Disaster Level")
    ax[0, 0].set_xticks([]) # Remove ticks for cleaner look
    ax[0, 0].set_yticks([])
    fig.colorbar(im, ax=ax[0, 0], label="Disaster Level", shrink=0.8)

    # --- Plot 2: Average Belief (Top Right - ax[0, 1]) ---
    avg_belief_grid = np.zeros((model.width, model.height))
    num_agents = model.num_humans
    if num_agents > 0:
        belief_sum_grid = np.zeros((model.width, model.height))
        for agent in model.humans.values():
            for cell, belief_info in agent.beliefs.items():
                 if isinstance(belief_info, dict) and (0 <= cell[0] < model.width and 0 <= cell[1] < model.height):
                      belief_sum_grid[cell[0], cell[1]] += belief_info.get('level', 0)
        avg_belief_grid = belief_sum_grid / num_agents
    im_avg = ax[0, 1].imshow(avg_belief_grid.T, cmap='coolwarm', origin='lower', vmin=0, vmax=5, interpolation='nearest')
    ax[0, 1].set_title("Avg. Agent Belief Level")
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    fig.colorbar(im_avg, ax=ax[0, 1], label="Avg. Belief Level", shrink=0.8)


    # --- Prepare Token Grids ---
    exploit_token_grid = np.zeros((model.width, model.height))
    explor_token_grid = np.zeros((model.width, model.height))
    max_exploit_tokens = 0
    max_explor_tokens = 0
    for pos, counts in model.tokens_this_tick.items():
         if 0 <= pos[0] < model.width and 0 <= pos[1] < model.height:
              exploit_count = counts.get('exploit', 0)
              explor_count = counts.get('explor', 0)
              exploit_token_grid[pos[0], pos[1]] = exploit_count
              explor_token_grid[pos[0], pos[1]] = explor_count
              max_exploit_tokens = max(max_exploit_tokens, exploit_count)
              max_explor_tokens = max(max_explor_tokens, explor_count)

    # Get agent positions (needed for overlay)
    exploit_x, exploit_y = [], []
    explor_x, explor_y = [], []
    for agent in model.humans.values():
        if agent.pos:
             x, y = agent.pos
             if agent.agent_type == "exploitative": exploit_x.append(x); exploit_y.append(y)
             else: explor_x.append(x); explor_y.append(y)

    # --- Plot 3: Average Confidence Level (ax[0, 2]) ---
    avg_confidence_grid = np.full((model.width, model.height), np.nan) # Use NaN for cells with no data
    confidence_sum_grid = np.zeros((model.width, model.height))
    confidence_count_grid = np.zeros((model.width, model.height), dtype=int) # Initialize count grid

    if model.num_humans > 0:
        for agent in model.humans.values():
            for cell, belief_info in agent.beliefs.items():
                 # Ensure coordinates are valid
                 if 0 <= cell[0] < model.width and 0 <= cell[1] < model.height:
                    # Explicitly get confidence, default to a known value (e.g., 0.1) if not dict or key missing
                    if isinstance(belief_info, dict):
                         confidence_val = belief_info.get('confidence', 0.1) # Use 0.1 if key missing
                    else:
                         confidence_val = 0.1 # Treat non-dict belief as low confidence

                    # Check for NaN just in case, although .get() should handle it
                    if not np.isnan(confidence_val):
                         confidence_sum_grid[cell[0], cell[1]] += confidence_val
                         # *** FIX: Increment the correct count grid ***
                         confidence_count_grid[cell[0], cell[1]] += 1

        # Avoid division by zero using the specific confidence counts
        valid_counts = confidence_count_grid > 0
        # *** FIX: Use confidence_count_grid for division ***
        avg_confidence_grid[valid_counts] = confidence_sum_grid[valid_counts] / confidence_count_grid[valid_counts]

        # Optional: Print stats just before plotting (for debugging)
        # print(f"\n--- Tick {tick} Confidence Plot Stats ---")
        # if np.isnan(avg_confidence_grid).any():
        #     print(f"  Min: {np.nanmin(avg_confidence_grid):.4f}, Max: {np.nanmax(avg_confidence_grid):.4f}, Mean: {np.nanmean(avg_confidence_grid):.4f}")
        # elif avg_confidence_grid.size > 0:
        #     print(f"  Min: {np.min(avg_confidence_grid):.4f}, Max: {np.max(avg_confidence_grid):.4f}, Mean: {np.mean(avg_confidence_grid):.4f}")
        # else: print("  Grid is empty!")
        # print(f"--- End Confidence Plot Stats ---")


    # Use a sequential colormap like 'magma' or 'plasma' for confidence (0 to 1)
    im_conf = ax[0, 2].imshow(avg_confidence_grid.T, cmap='magma', origin='lower', vmin=0, vmax=1, interpolation='nearest')
    ax[0, 2].set_title("Avg. Agent Confidence")
    ax[0, 2].set_xticks([])
    ax[0, 2].set_yticks([])
    fig.colorbar(im_conf, ax=ax[0, 2], label="Avg. Confidence", shrink=0.8)
    # --- Plot 4: Exploit Tokens (Bottom Left - ax[1, 0]) ---
    exploit_token_grid = np.zeros((model.width, model.height))
    explor_token_grid = np.zeros((model.width, model.height))
    max_exploit_tokens = 0
    max_explor_tokens = 0
    # Check if tokens_this_tick is populated before iterating
    if hasattr(model, 'tokens_this_tick') and model.tokens_this_tick:
        for pos, counts in model.tokens_this_tick.items():
            if 0 <= pos[0] < model.width and 0 <= pos[1] < model.height:
                exploit_count = counts.get('exploit', 0)
                explor_count = counts.get('explor', 0)
                exploit_token_grid[pos[0], pos[1]] = exploit_count
                explor_token_grid[pos[0], pos[1]] = explor_count
                max_exploit_tokens = max(max_exploit_tokens, exploit_count)
                max_explor_tokens = max(max_explor_tokens, explor_count)

    vmax_exploit = max(1, max_exploit_tokens)
    im_tok_exp = ax[1, 0].imshow(exploit_token_grid.T, cmap='Blues', origin='lower', vmin=0, vmax=vmax_exploit, interpolation='nearest')
    ax[1, 0].set_title("Exploitative Tokens Sent")
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])
    fig.colorbar(im_tok_exp, ax=ax[1, 0], label="# Exploit Tokens", shrink=0.8)
    # Optional overlay removed for brevity, can be added back

    # --- Plot 5: Exploratory Tokens (Bottom Right - ax[1, 1]) ---
    vmax_explor = max(1, max_explor_tokens) # Ensure vmax >= 1
    im_tok_er = ax[1, 1].imshow(explor_token_grid.T, cmap='Oranges', origin='lower', vmin=0, vmax=vmax_explor, interpolation='nearest')
    ax[1, 1].set_title("Exploratory Tokens Sent")
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    fig.colorbar(im_tok_er, ax=ax[1, 1], label="# Explor Tokens", shrink=0.8)
    # Optional overlay removed

    # --- Plot 6: Placeholder or Belief Variance (ax[1, 2]) ---
    # Example: Belief Level Variance Per Cell
    belief_levels_per_cell = {} # cell -> list of levels
    if model.num_humans > 0:
        for agent in model.humans.values():
             for cell, belief_info in agent.beliefs.items():
                 if isinstance(belief_info, dict) and (0 <= cell[0] < model.width and 0 <= cell[1] < model.height):
                    level = belief_info.get('level', np.nan) # Use NaN if level missing
                    if cell not in belief_levels_per_cell:
                        belief_levels_per_cell[cell] = []
                    if not np.isnan(level):
                         belief_levels_per_cell[cell].append(level)

    variance_grid = np.full((model.width, model.height), np.nan)
    for cell, levels in belief_levels_per_cell.items():
        if len(levels) > 1: # Need at least 2 beliefs to calculate variance
            variance_grid[cell[0], cell[1]] = np.var(levels)
        elif len(levels) == 1:
             variance_grid[cell[0], cell[1]] = 0 # Variance is 0 if only one belief

    im_var = ax[1, 2].imshow(variance_grid.T, cmap='plasma', origin='lower', vmin=0, interpolation='nearest') # Vmax adapts
    ax[1, 2].set_title("Belief Level Variance (Per Cell)")
    ax[1, 2].set_xticks([])
    ax[1, 2].set_yticks([])
    fig.colorbar(im_var, ax=ax[1, 2], label="Variance", shrink=0.8)


    # --- Final Touches for Layout ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    filepath = os.path.join(save_dir, f"grid_state_tick_{tick:04d}.png")
    plt.savefig(filepath)
    plt.close(fig) # Close figure to free memory

def plot_individual_beliefs(model, agent_ids, tick, save_dir="grid_plots/individual"):
    """Plots the belief map (level) for specific agent(s)."""
    if not isinstance(agent_ids, list):
        agent_ids = [agent_ids] # Ensure it's a list

    os.makedirs(save_dir, exist_ok=True)
    num_agents_to_plot = len(agent_ids)
    if num_agents_to_plot == 0:
        print("Warning: No agent IDs provided for plot_individual_beliefs.")
        return

    # Adjust layout based on number of agents
    ncols = min(3, num_agents_to_plot) # Max 3 columns
    nrows = math.ceil(num_agents_to_plot / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5), squeeze=False)
    fig.suptitle(f"Individual Belief Maps at Tick {tick}", fontsize=16)
    axes_flat = axes.flatten()

    plot_idx = 0
    for agent_id in agent_ids:
        agent = model.humans.get(agent_id)
        if agent is None:
            print(f"Warning: Agent {agent_id} not found in model.humans.")
            continue

        belief_grid = np.zeros((model.width, model.height)) # Default to 0
        for cell, belief_info in agent.beliefs.items():
            if isinstance(belief_info, dict) and (0 <= cell[0] < model.width and 0 <= cell[1] < model.height):
                belief_grid[cell[0], cell[1]] = belief_info.get('level', 0) # Use level

        ax = axes_flat[plot_idx]
        im = ax.imshow(belief_grid.T, cmap='coolwarm', origin='lower', vmin=0, vmax=5, interpolation='nearest')
        ax.set_title(f"Agent {agent.unique_id} ({agent.agent_type[:5]})") # Short type
        ax.set_xticks([])
        ax.set_yticks([])
        # Add a colorbar to the last plot or individually if needed
        if plot_idx == num_agents_to_plot - 1:
             fig.colorbar(im, ax=axes_flat[:plot_idx+1].tolist(), shrink=0.6, label="Believed Level")

        plot_idx += 1

    # Hide any unused subplots
    for i in range(plot_idx, len(axes_flat)):
        axes_flat[i].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filepath = os.path.join(save_dir, f"individual_beliefs_tick_{tick:04d}.png")
    plt.savefig(filepath)
    plt.close(fig)

def plot_belief_histogram(model, tick, save_dir="grid_plots/histograms"):
    """Plots histograms of belief levels for exploitative and exploratory agents."""
    os.makedirs(save_dir, exist_ok=True)

    exploit_levels = []
    explor_levels = []

    for agent in model.humans.values():
        target_list = exploit_levels if agent.agent_type == "exploitative" else explor_levels
        for belief_info in agent.beliefs.values():
            if isinstance(belief_info, dict):
                target_list.append(belief_info.get('level', -1)) # Use -1 or NaN for missing?

    # Filter out potential missing values if needed (e.g., -1)
    exploit_levels = [lvl for lvl in exploit_levels if lvl >= 0]
    explor_levels = [lvl for lvl in explor_levels if lvl >= 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle(f"Distribution of Agent Belief Levels at Tick {tick}", fontsize=16)

    # Define bins centered around integers 0 to 5
    bins = np.arange(-0.5, 6.5, 1) # Bins are -0.5, 0.5, 1.5, ..., 5.5

    # Plot Exploitative Histogram
    axes[0].hist(exploit_levels, bins=bins, rwidth=0.8, color='skyblue', alpha=0.7)
    axes[0].set_title(f"Exploitative Agents (N={len(exploit_levels)})")
    axes[0].set_xlabel("Believed Level")
    axes[0].set_ylabel("Frequency (Cell Beliefs)")
    axes[0].set_xticks(range(6)) # Ticks at 0, 1, 2, 3, 4, 5
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # Plot Exploratory Histogram
    axes[1].hist(explor_levels, bins=bins, rwidth=0.8, color='lightcoral', alpha=0.7)
    axes[1].set_title(f"Exploratory Agents (N={len(explor_levels)})")
    axes[1].set_xlabel("Believed Level")
    # axes[1].set_ylabel("Frequency (Cell Beliefs)") # Optional shared Y
    axes[1].set_xticks(range(6))
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filepath = os.path.join(save_dir, f"belief_histogram_tick_{tick:04d}.png")
    plt.savefig(filepath)
    plt.close(fig)

def plot_trust_evolution(trust_stats_array, title_suffix=""):
    """Plots trust evolution with Mean +/- IQR bands."""
    # Input validation
    if trust_stats_array is None or trust_stats_array.ndim < 3 or trust_stats_array.shape[0] == 0:
         print(f"Warning: Invalid or empty data for plot_trust_evolution {title_suffix}")
         return
    num_runs, T, num_metrics = trust_stats_array.shape
    if num_metrics < 7:
        print(f"Warning: Expected 7 metrics in trust_stats_array, found {num_metrics} for {title_suffix}")
        return
    ticks = trust_stats_array[0, :, 0] # Get ticks from first run

    # Helper to get stats for a specific trust type index
    def compute_plot_stats(data_index):
         # Ensure index is valid
         if data_index >= num_metrics:
             print(f"Warning: Invalid data index {data_index} requested.")
             return np.zeros(T), np.zeros(T), np.zeros(T) # Return zeros matching tick length
         mean = np.mean(trust_stats_array[:, :, data_index], axis=0)
         lower = np.percentile(trust_stats_array[:, :, data_index], 25, axis=0)
         upper = np.percentile(trust_stats_array[:, :, data_index], 75, axis=0)
         return mean, lower, upper

    # Indices: 1:AI_exp, 2:Friend_exp, 3:NonFriend_exp, 4:AI_expl, 5:Friend_expl, 6:NonFriend_expl
    ai_exp_mean, ai_exp_lower, ai_exp_upper = compute_plot_stats(1)
    friend_exp_mean, friend_exp_lower, friend_exp_upper = compute_plot_stats(2)
    nonfriend_exp_mean, nonfriend_exp_lower, nonfriend_exp_upper = compute_plot_stats(3)
    ai_expl_mean, ai_expl_lower, ai_expl_upper = compute_plot_stats(4)
    friend_expl_mean, friend_expl_lower, friend_expl_upper = compute_plot_stats(5)
    nonfriend_expl_mean, nonfriend_expl_lower, nonfriend_expl_upper = compute_plot_stats(6)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"Trust Evolution {title_suffix} (Mean +/- IQR)")

    # Exploitative Plot
    axes[0].plot(ticks, ai_exp_mean, label="AI Trust", color='blue')
    axes[0].fill_between(ticks, ai_exp_lower, ai_exp_upper, color='blue', alpha=0.2)
    axes[0].plot(ticks, friend_exp_mean, label="Friend Trust", color='green')
    axes[0].fill_between(ticks, friend_exp_lower, friend_exp_upper, color='green', alpha=0.2)
    axes[0].plot(ticks, nonfriend_exp_mean, label="Non-Friend Trust", color='red')
    axes[0].fill_between(ticks, nonfriend_exp_lower, nonfriend_exp_upper, color='red', alpha=0.2)
    axes[0].set_title("Exploitative Agents")
    axes[0].set_ylabel("Trust Level")
    axes[0].legend(fontsize='small')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].set_ylim(bottom=0, top=1.05) # Set Y limits for trust

    # Exploratory Plot
    axes[1].plot(ticks, ai_expl_mean, label="AI Trust", color='blue')
    axes[1].fill_between(ticks, ai_expl_lower, ai_expl_upper, color='blue', alpha=0.2)
    axes[1].plot(ticks, friend_expl_mean, label="Friend Trust", color='green')
    axes[1].fill_between(ticks, friend_expl_lower, friend_expl_upper, color='green', alpha=0.2)
    axes[1].plot(ticks, nonfriend_expl_mean, label="Non-Friend Trust", color='red')
    axes[1].fill_between(ticks, nonfriend_expl_lower, nonfriend_expl_upper, color='red', alpha=0.2)
    axes[1].set_title("Exploratory Agents")
    axes[1].set_xlabel("Tick")
    axes[1].set_ylabel("Trust Level")
    axes[1].legend(fontsize='small')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].set_ylim(bottom=0, top=1.05) # Set Y limits for trust

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    plt.show()


def plot_seci_aeci_evolution(seci_array, aeci_array, title_suffix=""):
    """Plots SECI and AECI evolution with Mean +/- IQR bands."""

    # Helper to get stats
    def get_stats(data_array, index):
        if data_array is None or data_array.ndim < 3 or data_array.shape[0] == 0: return None, None, None
        mean = np.mean(data_array[:, :, index], axis=0)
        lower = np.percentile(data_array[:, :, index], 25, axis=0)
        upper = np.percentile(data_array[:, :, index], 75, axis=0)
        return mean, lower, upper

    # Get SECI Stats (Index 1: Exploit, Index 2: Explor)
    seci_exp_mean, seci_exp_lower, seci_exp_upper = get_stats(seci_array, 1)
    seci_expl_mean, seci_expl_lower, seci_expl_upper = get_stats(seci_array, 2)

    # Get AECI Stats (Index 1: Exploit, Index 2: Explor)
    aeci_exp_mean, aeci_exp_lower, aeci_exp_upper = get_stats(aeci_array, 1)
    aeci_expl_mean, aeci_expl_lower, aeci_expl_upper = get_stats(aeci_array, 2)

    # Check if data is valid before plotting
    if seci_exp_mean is None or aeci_exp_mean is None:
         print(f"Warning: Insufficient data to plot SECI/AECI evolution {title_suffix}")
         return
    if seci_array.shape[1] == 0: # Check if there are any ticks recorded
        print(f"Warning: No ticks recorded in SECI/AECI data for {title_suffix}")
        return
    ticks = seci_array[0, :, 0] # Assuming seci_array is valid

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True) # *** Add sharey=True ***
    fig.suptitle(f"SECI & AECI Evolution {title_suffix} (Mean +/- IQR)")

    # SECI Plot
    axes[0].plot(ticks, seci_exp_mean, label="Exploitative", color="blue")
    axes[0].fill_between(ticks, seci_exp_lower, seci_exp_upper, color="blue", alpha=0.2)
    axes[0].plot(ticks, seci_expl_mean, label="Exploratory", color="orange")
    axes[0].fill_between(ticks, seci_expl_lower, seci_expl_upper, color="orange", alpha=0.2)
    axes[0].set_title("SECI Evolution")
    axes[0].set_xlabel("Tick")
    axes[0].set_ylabel("SECI")
    axes[0].legend(fontsize='small')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].set_ylim(bottom=0)

    # AECI Plot
    axes[1].plot(ticks, aeci_exp_mean, label="Exploitative", color="blue")
    axes[1].fill_between(ticks, aeci_exp_lower, aeci_exp_upper, color="blue", alpha=0.2)
    axes[1].plot(ticks, aeci_expl_mean, label="Exploratory", color="orange")
    axes[1].fill_between(ticks, aeci_expl_lower, aeci_expl_upper, color="orange", alpha=0.2)
    axes[1].set_title("AECI Evolution")
    axes[1].set_xlabel("Tick")
    axes[1].set_ylabel("AECI")
    axes[1].legend(fontsize='small')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].set_ylim(bottom=0)

    #*** FIX: Explicitly set shared Y limits AFTER plotting all data ***
    # Find overall min/max across all relevant data for these plots (usually 0 to 1, but check)
    all_means = np.concatenate([d for d in [seci_exp_mean, seci_expl_mean, aeci_exp_mean, aeci_expl_mean] if d is not None])
    all_uppers = np.concatenate([d for d in [seci_exp_upper, seci_expl_upper, aeci_exp_upper, aeci_expl_upper] if d is not None])

    # Check if we have valid data before setting limits
    if all_means.size > 0 and all_uppers.size > 0:
        ymin = 0 # Usually starts at 0
        ymax = max(1.0, np.nanmax(all_uppers) * 1.05) # Ensure at least 1.0, add 5% padding
        axes[0].set_ylim(ymin, ymax)
        axes[1].set_ylim(ymin, ymax) # Apply same limits to second plot

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_assistance_bars(assist_stats, raw_assist_counts, title_suffix=""):
    """Plots mean cumulative correct/incorrect tokens as bars with IQR error bars."""
    labels = ['Exploitative', 'Exploratory']

    # Data for bars (means) - use .get for safety
    mean_correct = [
        assist_stats.get("exploit_correct", {}).get("mean", 0),
        assist_stats.get("explor_correct", {}).get("mean", 0)
    ]
    mean_incorrect = [
        assist_stats.get("exploit_incorrect", {}).get("mean", 0),
        assist_stats.get("explor_incorrect", {}).get("mean", 0)
    ]

    # Data for error bars (IQR relative to mean)
    def get_iqr_errors(count_list):
        valid_counts = [c for c in count_list if isinstance(c, (int, float, np.number))] # Allow numpy numbers
        if not valid_counts: return [0, 0] # [lower_error_len, upper_error_len]
        # Calculate percentiles FIRST
        p25 = np.percentile(valid_counts, 25)
        p75 = np.percentile(valid_counts, 75)
        mean_val = np.mean(valid_counts) # Can calculate mean anytime

        lower_error_length = max(0, mean_val - p25)
        upper_error_length = max(0, p75 - mean_val)
        # Ensure list contains numbers
        valid_counts = [c for c in count_list if isinstance(c, (int, float))]


        return [lower_error_length, upper_error_length]

    errors_correct = [
        get_iqr_errors(raw_assist_counts.get("exploit_correct", [])),
        get_iqr_errors(raw_assist_counts.get("explor_correct", []))
    ]
    errors_incorrect = [
        get_iqr_errors(raw_assist_counts.get("exploit_incorrect", [])),
        get_iqr_errors(raw_assist_counts.get("explor_incorrect", []))
    ]
    # Transpose error lists for yerr format: [[lower1, lower2], [upper1, upper2]]
    yerr_correct = np.array(errors_correct).T
    yerr_incorrect = np.array(errors_incorrect).T

    x = np.arange(len(labels))  # label locations
    width = 0.1  # bar width

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, mean_correct, width, yerr=yerr_correct, capsize=4, label='Mean Correct Tokens', color='forestgreen', error_kw=dict(alpha=0.5))
    rects2 = ax.bar(x + width/2, mean_incorrect, width, yerr=yerr_incorrect, capsize=4, label='Mean Incorrect Tokens', color='firebrick', error_kw=dict(alpha=0.5))

    ax.set_ylabel('Mean Cumulative Tokens Sent per Run (Total)')
    ax.set_title(f'Token Assistance Summary {title_suffix} (Mean & IQR)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    # Add value labels on top of bars - check bar container type
    if hasattr(rects1, 'patches'): # Newer matplotlib versions
        ax.bar_label(rects1, padding=3, fmt='%.1f')
        ax.bar_label(rects2, padding=3, fmt='%.1f')
    else: # Older fallback - might not work perfectly
         for rect in rects1 + rects2:
              height = rect.get_height()
              ax.annotate('%.1f' % height,
                           xy=(rect.get_x() + rect.get_width() / 2, height),
                           xytext=(0, 3), # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom')

    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.set_ylim(bottom=0) # Ensure y-axis starts at 0
    fig.tight_layout()
    plt.show()

def plot_retainment_comparison(seci_data, aeci_data, retain_seci_data, retain_aeci_data, title_suffix=""):
    """Plots SECI/AECI vs Retainment metrics with Mean +/- IQR bands."""

    # Helper to get stats for a specific metric type index (1=exploit, 2=explor)
    def get_stats(data_array, index):
        # Add more robust check for valid data array
        if data_array is None or not isinstance(data_array, np.ndarray) or data_array.ndim < 3 or data_array.shape[0] == 0 or data_array.shape[1] == 0 or index >= data_array.shape[2]:
             num_ticks = 1 # Default tick count
             # Try to get actual tick count from a potentially valid array like seci_data
             if seci_data is not None and isinstance(seci_data, np.ndarray) and seci_data.ndim >=3 and seci_data.shape[1] > 0:
                 num_ticks = seci_data.shape[1]
             print(f"Warning: Invalid data for index {index} in plot_retainment_comparison {title_suffix}")
             return np.zeros(num_ticks), np.zeros(num_ticks), np.zeros(num_ticks) # Return zeros

        mean = np.mean(data_array[:, :, index], axis=0)
        lower = np.percentile(data_array[:, :, index], 25, axis=0)
        upper = np.percentile(data_array[:, :, index], 75, axis=0)
        return mean, lower, upper

    # Get ticks from a valid array
    if seci_data is None or seci_data.shape[0] == 0 or seci_data.shape[1] == 0:
        print(f"Warning: No data to plot for {title_suffix}")
        return
    ticks = seci_data[0, :, 0]

    # Get stats for each metric and agent type
    seci_exp_mean, seci_exp_lower, seci_exp_upper = get_stats(seci_data, 1)
    seci_expl_mean, seci_expl_lower, seci_expl_upper = get_stats(seci_data, 2)
    aeci_exp_mean, aeci_exp_lower, aeci_exp_upper = get_stats(aeci_data, 1)
    aeci_expl_mean, aeci_expl_lower, aeci_expl_upper = get_stats(aeci_data, 2)
    retain_seci_exp_mean, retain_seci_exp_lower, retain_seci_exp_upper = get_stats(retain_seci_data, 1)
    retain_seci_expl_mean, retain_seci_expl_lower, retain_seci_expl_upper = get_stats(retain_seci_data, 2)
    retain_aeci_exp_mean, retain_aeci_exp_lower, retain_aeci_exp_upper = get_stats(retain_aeci_data, 1)
    retain_aeci_expl_mean, retain_aeci_expl_lower, retain_aeci_expl_upper = get_stats(retain_aeci_data, 2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)

    # Plot Exploitative SECI
    axes[0, 0].plot(ticks, seci_exp_mean, label="SECI", color="blue")
    axes[0, 0].fill_between(ticks, seci_exp_lower, seci_exp_upper, color="blue", alpha=0.2)
    axes[0, 0].plot(ticks, retain_seci_exp_mean, label="Retain SECI", color="green")
    axes[0, 0].fill_between(ticks, retain_seci_exp_lower, retain_seci_exp_upper, color="green", alpha=0.2)
    axes[0, 0].set_title("Exploitative SECI vs Retainment")
    axes[0, 0].set_ylabel("Value")
    axes[0, 0].legend(fontsize='small')
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)
    axes[0, 0].set_ylim(bottom=0)

    # Plot Exploratory SECI
    axes[0, 1].plot(ticks, seci_expl_mean, label="SECI", color="blue")
    axes[0, 1].fill_between(ticks, seci_expl_lower, seci_expl_upper, color="blue", alpha=0.2)
    axes[0, 1].plot(ticks, retain_seci_expl_mean, label="Retain SECI", color="green")
    axes[0, 1].fill_between(ticks, retain_seci_expl_lower, retain_seci_expl_upper, color="green", alpha=0.2)
    axes[0, 1].set_title("Exploratory SECI vs Retainment")
    axes[0, 1].legend(fontsize='small')
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)
    axes[0, 1].set_ylim(bottom=0)

    # Plot Exploitative AECI
    axes[1, 0].plot(ticks, aeci_exp_mean, label="AECI", color="orange")
    axes[1, 0].fill_between(ticks, aeci_exp_lower, aeci_exp_upper, color="orange", alpha=0.2)
    axes[1, 0].plot(ticks, retain_aeci_exp_mean, label="Retain AECI", color="red")
    axes[1, 0].fill_between(ticks, retain_aeci_exp_lower, retain_aeci_exp_upper, color="red", alpha=0.2)
    axes[1, 0].set_title("Exploitative AECI vs Retainment")
    axes[1, 0].set_xlabel("Tick")
    axes[1, 0].set_ylabel("Value")
    axes[1, 0].legend(fontsize='small')
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)
    axes[1, 0].set_ylim(bottom=0)

    # Plot Exploratory AECI
    axes[1, 1].plot(ticks, aeci_expl_mean, label="AECI", color="orange")
    axes[1, 1].fill_between(ticks, aeci_expl_lower, aeci_expl_upper, color="orange", alpha=0.2)
    axes[1, 1].plot(ticks, retain_aeci_expl_mean, label="Retain AECI", color="red")
    axes[1, 1].fill_between(ticks, retain_aeci_expl_lower, retain_aeci_expl_upper, color="red", alpha=0.2)
    axes[1, 1].set_title("Exploratory AECI vs Retainment")
    axes[1, 1].set_xlabel("Tick")
    axes[1, 1].legend(fontsize='small')
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)
    axes[1, 1].set_ylim(bottom=0)

    plt.suptitle(f"Retainment Comparison {title_suffix} (Mean +/- IQR)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_correct_token_shares_bars(results, share_values):
    """Plots mean correct token share as bars with IQR error bars."""
    mean_shares_exploit = []
    lower_errors_exploit = [] # Length below mean
    upper_errors_exploit = [] # Length above mean
    mean_shares_explor = []
    lower_errors_explor = [] # Length below mean
    upper_errors_explor = []
    # Add lists for exploratory if you want to plot them too

    num_param_values = len(share_values)
    x_pos = np.arange(num_param_values)
    width = 0.35 # Define the width for the bars


    for share in share_values:
        # Access raw counts per run safely using .get()
        results_dict = results.get(share, {})
        raw_counts = results_dict.get("raw_assist_counts")
        if not raw_counts:
            # Append default values if data missing for this parameter setting
             mean_shares_exploit.append(0); lower_errors_exploit.append(0); upper_errors_exploit.append(0)
             mean_shares_explor.append(0); lower_errors_explor.append(0); upper_errors_explor.append(0)

             continue

        exploit_correct = raw_counts.get("exploit_correct", [])
        exploit_incorrect = raw_counts.get("exploit_incorrect", [])
        explor_correct = raw_counts.get("explor_correct", []) # If plotting
        explor_incorrect = raw_counts.get("explor_incorrect", []) # If plotting

        run_shares_exploit = []
        run_shares_explor = []
        num_runs = len(exploit_correct)
        if num_runs == 0: # Handle case where lists might be empty even if key exists
             mean_shares_exploit.append(0); lower_errors_exploit.append(0); upper_errors_exploit.append(0)
             mean_shares_explor.append(0); lower_errors_explor.append(0); upper_errors_explor.append(0)
             continue
        # Check if exploratory data has the same number of runs
        if len(explor_correct) != num_runs or len(explor_incorrect) != num_runs:
             print(f"Warning: Mismatch in run count for exploit/explor data for share={share}. Skipping explor share calculation.")
             # Handle mismatch - maybe skip explor calculation for this share?
             # For now, let's pad means/errors for explor if mismatch occurs
             calculating_explor = False
        else:
             calculating_explor = True

        for i in range(num_runs):
            # Exploitative Share
            run_correct_exp = exploit_correct[i]
            run_incorrect_exp = exploit_incorrect[i]
            run_total_exp = run_correct_exp + run_incorrect_exp
            run_share_exp = run_correct_exp / run_total_exp if run_total_exp > 0 else 0
            run_shares_exploit.append(run_share_exp)


            #  explor share if needed
            run_correct_expr = explor_correct[i]
            run_incorrect_expr = explor_incorrect[i]
            run_total_expr = run_correct_expr + run_incorrect_expr
            run_share_expr = run_correct_expr / run_total_expr if run_total_expr > 0 else 0
            run_shares_explor.append(run_share_expr)


        # Calculate stats for exploit shares
        mean_val = np.mean(run_shares_exploit)
        p25 = np.percentile(run_shares_exploit, 25)
        p75 = np.percentile(run_shares_exploit, 75)
        mean_shares_exploit.append(mean_val)
        lower_errors_exploit.append(max(0, mean_val - p25)) # Error bar length below mean
        upper_errors_exploit.append(max(0, p75 - mean_val)) # Error bar length above mean
        # Calculate explor stats
        if calculating_explor and run_shares_explor: # Check if list has dat
            mean_val_expr = np.mean(run_shares_explor)
            p25r = np.percentile(run_shares_explor, 25)
            p75r = np.percentile(run_shares_explor, 75)
            mean_shares_explor.append(mean_val_expr)
            lower_errors_explor.append(max(0, mean_val - p25)) # Error bar length below mean
            upper_errors_explor.append(max(0, p75 - mean_val)) # Error bar length above mean
        else:
            mean_shares_explor.append(0)
            lower_errors_explor.append(0)
            upper_errors_explor.append(0)

    # Asymmetric error bars for IQR relative to mean
    error_bars_exploit = [lower_errors_exploit, upper_errors_exploit]
    error_bars_explor = [lower_errors_explor, upper_errors_explor]


    plt.figure(figsize=(8, 6))
    # Plot bars for Exploitative
    plt.bar(x_pos - width/2, mean_shares_exploit, yerr=error_bars_exploit, capsize=5, color='skyblue', label='Mean Exploit Share') # Grouped bar position
    # Plot bars for Exploratory
    plt.bar(x_pos + width/2, mean_shares_explor, yerr=error_bars_explor, capsize=5, color='springgreen', label='Mean Explor Share') # Grouped bar position

    # Add bars for Exploratory if calculated (use x_pos + width/2 etc. for grouping)

    plt.xlabel("Share Exploitative Agents")
    plt.ylabel("Share of Correctly Targeted Tokens")
    plt.title("Correct Token Share vs. Agent Mix (Mean & IQR)") # Combined title
    plt.xticks(x_pos, share_values) # Set x-axis labels to be the share values
    plt.legend()
    plt.ylim(0, 1)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.show()


def plot_belief_error_evolution(belief_error_array, title_suffix=""):
    """Plots Avg. Belief MAE evolution with Mean +/- IQR bands."""
    # Input validation (similar to plot_trust_evolution)
    if belief_error_array is None or not isinstance(belief_error_array, np.ndarray) or belief_error_array.ndim < 3 or belief_error_array.shape[0] == 0 or belief_error_array.shape[1] == 0:
         print(f"Warning: Invalid or empty data for plot_belief_error_evolution {title_suffix}")
         return
    num_runs, T, num_metrics = belief_error_array.shape
    if num_metrics < 3:
        print(f"Warning: Expected 3 metrics in belief_error_array, found {num_metrics} for {title_suffix}")
        return
    ticks = belief_error_array[0, :, 0]

    def get_stats(index): # index 1 for exploit, 2 for explor
         if index >= num_metrics: return np.zeros(T), np.zeros(T), np.zeros(T)
         mean = np.mean(belief_error_array[:, :, index], axis=0)
         lower = np.percentile(belief_error_array[:, :, index], 25, axis=0)
         upper = np.percentile(belief_error_array[:, :, index], 75, axis=0)
         return mean, lower, upper

    mae_exp_mean, mae_exp_lower, mae_exp_upper = get_stats(1)
    mae_expl_mean, mae_expl_lower, mae_expl_upper = get_stats(2)

    plt.figure(figsize=(8, 5))
    plt.plot(ticks, mae_exp_mean, label="Exploitative", color="blue")
    plt.fill_between(ticks, mae_exp_lower, mae_exp_upper, color="blue", alpha=0.2, label="Exploit IQR")
    plt.plot(ticks, mae_expl_mean, label="Exploratory", color="orange")
    plt.fill_between(ticks, mae_expl_lower, mae_expl_upper, color="orange", alpha=0.2, label="Explor IQR")

    plt.title(f"Average Belief MAE {title_suffix} (Mean +/- IQR)")
    plt.xlabel("Tick")
    plt.ylabel("Avg. Belief MAE")
    plt.legend(fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.show()

# --- New Plot Function 5 ---
def plot_belief_variance_evolution(belief_variance_array, title_suffix=""):
    """Plots Within-Type Belief Variance evolution with Mean +/- IQR bands."""
    # Input validation
    if belief_variance_array is None or not isinstance(belief_variance_array, np.ndarray) or belief_variance_array.ndim < 3 or belief_variance_array.shape[0] == 0 or belief_variance_array.shape[1] == 0:
         print(f"Warning: Invalid or empty data for plot_belief_variance_evolution {title_suffix}")
         return
    num_runs, T, num_metrics = belief_variance_array.shape
    if num_metrics < 3:
         print(f"Warning: Expected 3 metrics in belief_variance_array, found {num_metrics} for {title_suffix}")
         return
    ticks = belief_variance_array[0, :, 0]

    def get_stats(index): # index 1 for exploit, 2 for explor
         if index >= num_metrics: return np.zeros(T), np.zeros(T), np.zeros(T)
         mean = np.mean(belief_variance_array[:, :, index], axis=0)
         lower = np.percentile(belief_variance_array[:, :, index], 25, axis=0)
         upper = np.percentile(belief_variance_array[:, :, index], 75, axis=0)
         return mean, lower, upper

    var_exp_mean, var_exp_lower, var_exp_upper = get_stats(1)
    var_expl_mean, var_expl_lower, var_expl_upper = get_stats(2)

    plt.figure(figsize=(8, 5))
    plt.plot(ticks, var_exp_mean, label="Exploitative", color="blue")
    plt.fill_between(ticks, var_exp_lower, var_exp_upper, color="blue", alpha=0.2, label="Exploit IQR")
    plt.plot(ticks, var_expl_mean, label="Exploratory", color="orange")
    plt.fill_between(ticks, var_expl_lower, var_expl_upper, color="orange", alpha=0.2, label="Explor IQR")

    plt.title(f"Within-Type Belief Variance {title_suffix} (Mean +/- IQR)")
    plt.xlabel("Tick")
    plt.ylabel("Belief Variance")
    plt.legend(fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.show()

# --- New Plot Function 6 ---
def plot_unmet_need_evolution(unmet_needs_data, title_suffix=""):
    """Plots Unmet Need count evolution with Mean +/- IQR bands."""
    # unmet_needs_data is expected to be a list of lists/arrays (one per run)
    if not unmet_needs_data or not isinstance(unmet_needs_data, list) or not all(isinstance(run_data, (list, np.ndarray)) for run_data in unmet_needs_data):
        print(f"Warning: Invalid or empty data for plot_unmet_need_evolution {title_suffix}")
        return

    # Pad shorter runs with NaN if lengths differ, then convert to array
    try:
        # Find max length across all runs, ensure it's > 0
        T = max(len(run_data) for run_data in unmet_needs_data if run_data is not None and len(run_data)>0)
        if T == 0: raise ValueError("No non-empty runs found")

        unmet_array = np.full((len(unmet_needs_data), T), np.nan)
        for i, run_data in enumerate(unmet_needs_data):
             if run_data is not None and len(run_data) > 0:
                  unmet_array[i, :len(run_data)] = run_data
        num_runs = unmet_array.shape[0]
    except Exception as e:
        print(f"Warning: Could not process unmet needs data for {title_suffix}: {e}")
        return

    ticks = np.arange(T) # Simple tick count (0 to T-1)

    mean = np.nanmean(unmet_array, axis=0) # Use nanmean to ignore padding
    try:
         lower = np.nanpercentile(unmet_array, 25, axis=0)
         upper = np.nanpercentile(unmet_array, 75, axis=0)
    except (AttributeError, TypeError): # Fallback for older numpy or if NaNs cause issues
         print("Warning: np.nanpercentile failed. Plotting mean only for unmet need.")
         lower = mean
         upper = mean
         # Replace NaNs in mean if any resulted from all-NaN columns
         mean = np.nan_to_num(mean)
         lower = np.nan_to_num(lower)
         upper = np.nan_to_num(upper)

    plt.figure(figsize=(8, 5))
    plt.plot(ticks, mean, label="Mean Unmet Need", color="purple")
    plt.fill_between(ticks, lower, upper, color="purple", alpha=0.2, label="IQR")

    plt.title(f"Unmet Need (Count) {title_suffix} (Mean +/- IQR)")
    plt.xlabel("Tick")
    plt.ylabel("Number of High-Need Cells with 0 Tokens")
    plt.legend(fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(bottom=0)
    plt.tight_layout()
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
    num_runs = 10
    save_dir = "agent_model_results"
    os.makedirs(save_dir, exist_ok=True)

    ##############################################
    # Experiment A: Vary share_exploitative
    ##############################################
    share_values = [0.5]
    file_a_pkl = os.path.join(save_dir, "results_experiment_A.pkl")
    file_a_csv = os.path.join(save_dir, "results_experiment_A.csv")

    print("Running Experiment A...")
    results_a = experiment_share_exploitative(base_params, share_values, num_runs)
    with open(file_a_pkl, "wb") as f:
        pickle.dump(results_a, f)
    export_results_to_csv(results_a, share_values, file_a_csv, "Experiment A")

    for share in share_values:
        print(f"Share Exploitative = {share}")
        results_dict = results_a[share] # Get results for this parameter setting

        # Call existing plots (now updated to show IQR)
        plot_trust_evolution(results_dict["trust_stats"], f"(Share={share})")
        plot_seci_aeci_evolution(results_dict["seci"], results_dict["aeci"], f"(Share={share})")
        plot_retainment_comparison(results_dict["seci"], results_dict["aeci"],
                               results_dict["retain_seci"], results_dict["retain_aeci"],
                               f"(Share={share})")

        # Call NEW plots
        plot_belief_error_evolution(results_dict["belief_error"], f"(Share={share})")
        plot_belief_variance_evolution(results_dict["belief_variance"], f"(Share={share})")
        plot_unmet_need_evolution(results_dict["unmet_needs_evol"], f"(Share={share})")

        # Call updated bar chart plots
        plot_assistance_bars(results_dict["assist"], results_dict["raw_assist_counts"], f"(Share={share})")

    # After Experiment A loop, call the parameter sweep bar chart
    plot_correct_token_shares_bars(results_a, share_values)

    ##############################################
    # Experiment B: Vary AI Alignment Level
    ##############################################
    alignment_values = [0.1, 0.5, 0.9]
    results_b = experiment_ai_alignment(base_params, alignment_values, num_runs)

    print("-" * 20) # Separator
    if 'results_b' in locals() or 'results_b' in globals():
        print(f"Experiment B check: 'results_b' variable exists. Type: {type(results_b)}")
        if isinstance(results_b, dict):
            print(f"Keys found: {list(results_b.keys())}")
        else:
            print(f"Value: {results_b}")
    else:
        print("Experiment B check: 'results_b' variable DOES NOT EXIST.")
    print("-" * 20) # 

    # --- Bar Chart Plotting for Exp B  ---
    aligns = sorted(results_b.keys())
    num_aligns = len(aligns)
    x_pos = np.arange(num_aligns)
    width = 0.35 # Width for grouped bars

    means_exploit_b = []
    errors_exploit_b = [[],[]] # [lower_errors, upper_errors]
    means_explor_b = []
    errors_explor_b = [[],[]]

    for align in aligns:
        res = results_b[align]
        raw_counts = res.get("raw_assist_counts")
        assist_stats = res.get("assist") # Get pre-calculated means if needed, though raw is better for error bars

        if not raw_counts or not assist_stats: continue # Skip if data missing

        # Calculate stats from raw counts for error bars
        def get_iqr_errors_exp_b(count_list): # Helper specific to this plot
            valid_counts = [c for c in count_list if isinstance(c, (int, float, np.number))]
            if not valid_counts: return [0, 0]
            mean_val = np.mean(valid_counts)
            p25 = np.percentile(valid_counts, 25)
            p75 = np.percentile(valid_counts, 75)
            return [max(0, mean_val - p25), max(0, p75 - mean_val)] # CORRECTED

        # Exploitative stats
        exploit_correct_runs = raw_counts.get("exploit_correct", [])
        means_exploit_b.append(assist_stats["exploit_correct"]["mean"]) # Use pre-calculated mean for bar height
        errors_exp = get_iqr_errors_exp_b(exploit_correct_runs)
        errors_exploit_b[0].append(errors_exp[0]); errors_exploit_b[1].append(errors_exp[1])

        # Exploratory stats
        explor_correct_runs = raw_counts.get("explor_correct", [])
        means_explor_b.append(assist_stats["explor_correct"]["mean"]) # Use pre-calculated mean for bar height
        errors_er = get_iqr_errors_exp_b(explor_correct_runs)
        errors_explor_b[0].append(errors_er[0]); errors_explor_b[1].append(errors_er[1])

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot grouped bars with corrected error bars
    rects1 = ax.bar(x_pos - width/2, means_exploit_b, width, yerr=errors_exploit_b, capsize=4, label='Exploitative Correct Tokens', color='tab:blue', error_kw=dict(alpha=0.5))
    rects2 = ax.bar(x_pos + width/2, means_explor_b, width, yerr=errors_explor_b, capsize=4, label='Exploratory Correct Tokens', color='tab:orange', error_kw=dict(alpha=0.5))

    ax.set_xlabel("AI Alignment Level")
    ax.set_ylabel("Mean Cumulative Correct Tokens per Run")
    ax.set_title("Experiment B: Effect of AI Alignment (Mean & IQR)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(aligns)
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.set_ylim(bottom=0) # Start y-axis at 0
    fig.tight_layout()
    plt.show()

    # --- Time Evolution Plots for Exp B  ---
    print("\n--- Plotting Time Evolution for Experiment B ---")

    for align in alignment_values:
        print(f"AI Alignment Level = {align}")
        # Get the dictionary of results for this specific alignment level
        # Use .get(align, {}) to safely get the dict or an empty one if key missing
        results_dict_b = results_b.get(align, {})
        title_suffix_b = f"(Align={align})" # Use align in title

        # Check if keys exist before plotting, access via results_dict_b
        if "trust_stats" in results_dict_b:
            plot_trust_evolution(results_dict_b["trust_stats"], title_suffix_b)
        else: print(f"  Skipping plot_trust_evolution for Align={align} (missing data)")

        if "seci" in results_dict_b and "aeci" in results_dict_b:
            plot_seci_aeci_evolution(results_dict_b["seci"], results_dict_b["aeci"], title_suffix_b)
        else: print(f"  Skipping plot_seci_aeci_evolution for Align={align} (missing data)")

        if "seci" in results_dict_b and "aeci" in results_dict_b and "retain_seci" in results_dict_b and "retain_aeci" in results_dict_b:
            plot_retainment_comparison(results_dict_b["seci"], results_dict_b["aeci"],
                                         results_dict_b["retain_seci"], results_dict_b["retain_aeci"],
                                         title_suffix_b)
        else: print(f"  Skipping plot_retainment_comparison for Align={align} (missing data)")

        if "belief_error" in results_dict_b:
                plot_belief_error_evolution(results_dict_b["belief_error"], title_suffix_b)
        else: print(f"  Skipping plot_belief_error_evolution for Align={align} (missing data)")

        if "belief_variance" in results_dict_b:
                plot_belief_variance_evolution(results_dict_b["belief_variance"], title_suffix_b)
        else: print(f"  Skipping plot_belief_variance_evolution for Align={align} (missing data)")

        if "unmet_needs_evol" in results_dict_b:
            # Ensure unmet_needs_evol data is correctly formatted (list of lists/arrays)
            if isinstance(results_dict_b["unmet_needs_evol"], list):
                 plot_unmet_need_evolution(results_dict_b["unmet_needs_evol"], title_suffix_b)
            else: print(f"  Skipping plot_unmet_need_evolution for Align={align} (invalid data format)")
        else: print(f"  Skipping plot_unmet_need_evolution for Align={align} (missing data)")

        if "assist" in results_dict_b and "raw_assist_counts" in results_dict_b:
              # Ensure assist and raw_assist_counts have expected structure
              if isinstance(results_dict_b["assist"], dict) and isinstance(results_dict_b["raw_assist_counts"], dict):
                  plot_assistance_bars(results_dict_b["assist"], results_dict_b["raw_assist_counts"], title_suffix_b)
              else: print(f"  Skipping plot_assistance_bars for Align={align} (invalid data format)")
        else: print(f"  Skipping plot_assistance_bars for Align={align} (missing data)")

    else:
        print("Warning: results_b dictionary not found. Skipping Experiment B plotting.")



    ##############################################
    # Experiment C: Vary Disaster Dynamics and Shock Magnitude
    ##############################################
    dynamics_values = [1, 2, 3]
    shock_values = [1, 2, 3]
    results_c = experiment_disaster_dynamics(base_params, dynamics_values, shock_values, num_runs)

    # Initialize matrices for heatmaps
    exploit_correct_matrix = np.zeros((len(dynamics_values), len(shock_values)))
    explor_correct_matrix = np.zeros((len(dynamics_values), len(shock_values)))
    final_mae_exploit_matrix = np.zeros((len(dynamics_values), len(shock_values))) # ADDED
    final_mae_explor_matrix = np.zeros((len(dynamics_values), len(shock_values)))  # ADDED
    final_unmet_need_matrix = np.zeros((len(dynamics_values), len(shock_values))) # ADDED

    for i, dd in enumerate(dynamics_values):
        for j, sm in enumerate(shock_values):
            res_key = (dd, sm)
            # Ensure results exist for this key
            if res_key not in results_c:
                print(f"Warning: Missing results for {res_key}")
                continue
            res = results_c[res_key]

            # --- Populate Heatmap Matrices ---
            exploit_correct_matrix[i, j] = res["assist"]["exploit_correct"]["mean"]
            explor_correct_matrix[i, j] = res["assist"]["explor_correct"]["mean"]

            # Get final belief error (mean across runs)
            # Assumes belief_error array shape is (runs, ticks_recorded, 3) -> [tick, exploit, explor]
            if res["belief_error"].ndim >= 3 and res["belief_error"].shape[1] > 0:
                final_mae_exploit_matrix[i, j] = np.mean(res["belief_error"][:, -1, 1]) # Mean of last tick's exploit MAE
                final_mae_explor_matrix[i, j] = np.mean(res["belief_error"][:, -1, 2]) # Mean of last tick's explor MAE
            else:
                final_mae_exploit_matrix[i, j] = np.nan # Use NaN if data missing/invalid
                final_mae_explor_matrix[i, j] = np.nan

            # Get final unmet need (mean across runs)
            # unmet_needs_evol is a list of lists/arrays (runs, ticks)
            final_unmet_counts = []
            if "unmet_needs_evol" in res and res["unmet_needs_evol"]:
                for run_data in res["unmet_needs_evol"]:
                    if run_data is not None and len(run_data) > 0:
                        final_unmet_counts.append(run_data[-1]) # Get last value from each run
            final_unmet_need_matrix[i, j] = np.mean(final_unmet_counts) if final_unmet_counts else np.nan
            # --- End Populate ---

            # --- Call Bar Chart Plot (Update this call) ---
            # plot_assistance(res["assist"], f"(Dynamics={dd}, Shock={sm})") # OLD Call
            if "raw_assist_counts" in res: # Check if raw data exists
                 plot_assistance_bars(res["assist"], res["raw_assist_counts"], f"(Dynamics={dd}, Shock={sm})") # NEW Call
            else:
                 print(f"Warning: Raw assist counts missing for {res_key}, skipping assistance bar plot.")
            # --- End Bar Chart Call ---

    # --- Plotting Heatmaps ---
    fig_c, axes_c = plt.subplots(2, 2, figsize=(12, 10)) # Create 2x2 layout for 4 heatmaps
    fig_c.suptitle("Experiment C: Impact of Disaster Dynamics & Shock (Mean Final Values)")

    # Heatmap 1: Exploit Correct Tokens (Keep Existing)
    im1 = axes_c[0, 0].imshow(exploit_correct_matrix, cmap='viridis', origin='lower', aspect='auto')
    axes_c[0, 0].set_title("Exploitative Correct Tokens")
    fig_c.colorbar(im1, ax=axes_c[0, 0], label='Mean Total Correct Tokens')

    # Heatmap 2: Explor Correct Tokens (Keep Existing)
    im2 = axes_c[0, 1].imshow(explor_correct_matrix, cmap='viridis', origin='lower', aspect='auto')
    axes_c[0, 1].set_title("Exploratory Correct Tokens")
    fig_c.colorbar(im2, ax=axes_c[0, 1], label='Mean Total Correct Tokens')

    # Heatmap 3: Final Belief MAE (Exploit) (NEW)
    im3 = axes_c[1, 0].imshow(final_mae_exploit_matrix, cmap='magma', origin='lower', aspect='auto')
    axes_c[1, 0].set_title("Exploitative Final Belief MAE")
    fig_c.colorbar(im3, ax=axes_c[1, 0], label='Mean Final MAE')

    # Heatmap 4: Final Unmet Need (NEW)
    im4 = axes_c[1, 1].imshow(final_unmet_need_matrix, cmap='cividis', origin='lower', aspect='auto')
    axes_c[1, 1].set_title("Final Unmet Need Count")
    fig_c.colorbar(im4, ax=axes_c[1, 1], label='Mean Final Unmet Count')

    # Set ticks and labels for all heatmaps
    for ax_row in axes_c:
        for ax in ax_row:
            ax.set_xticks(ticks=range(len(shock_values)), labels=shock_values)
            ax.set_yticks(ticks=range(len(dynamics_values)), labels=dynamics_values)
            ax.set_xlabel("Shock Magnitude")
            ax.set_ylabel("Disaster Dynamics")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    # --- End Heatmap Plotting ---

    ##############################################
    # Experiment D: Vary Learning Rate and Epsilon
    ##############################################
    learning_rate_values = [0.03, 0.05, 0.07]
    epsilon_values = [0.2, 0.3]
    results_d = experiment_learning_trust(base_params, learning_rate_values, epsilon_values, num_runs)

    # --- Plot 1: Final SECI vs LR/Epsilon (Bar Chart) ---
    fig_d_seci, ax_d_seci = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig_d_seci.suptitle("Experiment D: Final SECI vs Learning Rate / Epsilon (Mean & IQR)")
    bar_width = 0.35

    for idx, eps in enumerate(epsilon_values):
        means_exploit = []; errors_exploit = [[],[]] # [lower_err, upper_err]
        means_explor = []; errors_explor = [[],[]]

        for lr in learning_rate_values:
            res_key = (lr, eps)
            if res_key not in resubase_confidence_bumplts_d: continue
            res = results_d[res_key]

            # Extract Final SECI values per run
            # Assumes seci array shape (runs, ticks_recorded, 3) -> [tick, exploit, explor]
            if res["seci"].ndim >= 3 and res["seci"].shape[1] > 0:
                seci_exploit_final = res["seci"][:, -1, 1] # Last tick's exploit SECI for all runs
                seci_explor_final = res["seci"][:, -1, 2] # Last tick's explor SECI for all runs

                # Calculate stats
                mean_exp = np.mean(seci_exploit_final); p25_exp = np.percentile(seci_exploit_final, 25); p75_exp = np.percentile(seci_exploit_final, 75)
                mean_er = np.mean(seci_explor_final); p25_er = np.percentile(seci_explor_final, 25); p75_er = np.percentile(seci_explor_final, 75)

                means_exploit.append(mean_exp); errors_exploit[0].append(mean_exp-p25_exp); errors_exploit[1].append(p75_exp-mean_exp)
                means_explor.append(mean_er); errors_explor[0].append(mean_er-p25_er); errors_explor[1].append(p75_er-mean_er)
            else:
                 means_exploit.append(0); errors_exploit[0].append(0); errors_exploit[1].append(0)
                 means_explor.append(0); errors_explor[0].append(0); errors_explor[1].append(0)

        x_pos = np.arange(len(learning_rate_values))
        ax = ax_d_seci[idx] # Use subplot for each epsilon
        rects1 = ax.bar(x_pos - bar_width/2, means_exploit, bar_width, yerr=errors_exploit, capsize=4, label='Exploitative', color='tab:blue', error_kw=dict(alpha=0.5))
        rects2 = ax.bar(x_pos + bar_width/2, means_explor, bar_width, yerr=errors_explor, capsize=4, label='Exploratory', color='tab:orange', error_kw=dict(alpha=0.5))

        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Mean Final SECI")
        ax.set_title(f"Epsilon = {eps}")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(learning_rate_values)
        ax.legend()
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        ax.set_ylim(bottom=0)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


    # --- Plot 2: Final AECI vs LR/Epsilon (Bar Chart) ---
    # (Create a similar figure and axes (fig_d_aeci, ax_d_aeci)
    # Loop through eps and lr as above
    # Extract final AECI values: res["aeci"][:, -1, 1] and res["aeci"][:, -1, 2]
    # Calculate mean/IQR errors
    # Use ax.bar(...) to plot grouped bars for final AECI
    # Set appropriate labels and titles
    # ... (code structure similar to SECI plot) ...
    # plt.show()


    # --- Plot 3: Correct Tokens vs LR/Epsilon (Bar Chart - Optional Replacement) ---
    # (Create a similar figure and axes (fig_d_tokens, ax_d_tokens)
    # Loop through eps and lr
    # Extract raw correct token counts: res["raw_assist_counts"]["exploit_correct"] etc.
    # Calculate mean/IQR errors for total correct tokens per run
    # Use ax.bar(...) to plot grouped bars
    # Set appropriate labels and titles
    # ... (code structure similar to SECI plot) ...
    # plt.show()

    # --- (Delete or comment out the old line plot code for Exp D) ---
    # plt.figure(figsize=(8,6))
    # ... (old plt.plot loops) ...
    # plt.show()

    gc.collect()
