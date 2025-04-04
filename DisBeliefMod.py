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

        self.correct_targets = 0
        self.incorrect_targets = 0

        self.tokens_this_tick = {}
        self.pending_rewards = []  # List of tuples: (tick_due, mode, list_of_cells)
        self.q_table = {f"A_{k}": 0.0 for k in range(5)}
        self.q_table["human"] = 0.0

        self.belief_learning_rate = 0.6 # Tunable: 1.0 = immediate update, < 1.0 = slower update

    def initialize_beliefs(self):
        height, width = self.model.disaster_grid.shape
        for x in range(width):
            for y in range(height):
                self.beliefs[(x, y)] = {'level': 0, 'confidence': 0.1}  # Initialize with level 0 and low confidence
        self.sense_environment()



    def sense_environment(self):
        pos = self.pos
        radius = 1 #if self.agent_type == "exploitative" else 5
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
                    belief_conf = 0.95 # High confidence for direct sensing

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

        radius = 1 #if self.agent_type == "exploitative" else 5
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
         decay_rate = 0.005 # Slow general decay rate per step
         friend_decay_rate = 0.001 # Slower decay for friends
         # Iterate over a copy of keys because dictionary size might change (though unlikely here)
         for source_id in list(self.trust.keys()):
              rate = friend_decay_rate if source_id in self.friends else decay_rate
              self.trust[source_id] = max(0, self.trust[source_id] - rate)

    def choose_information_mode(self):
        if random.random() < self.epsilon:
            return random.choice(list(self.q_table.keys()))
        return max(self.q_table, key=self.q_table.get)

    def request_information(self):
        mode_choice = self.choose_information_mode()
        reports = {}

        top_candidates = [] # Define scope for potential use later if needed
        source_agent_id_for_credit = None # Define scope, used for trust/Q updates later


        if mode_choice.startswith("A_"):
            ai_id = mode_choice
            chosen_ai = self.model.ais[ai_id]
            reports = chosen_ai.respond(self.beliefs, self.trust.get(ai_id, 0))
            # Weight AI call as 1
            self.accum_calls_ai += 1
            self.accum_calls_total += 1
            source_agent_id_for_credit = mode_choice # AI is the source for credit assignment

        else: #Human
            # --- Modified Human Source Selection Logic ---
            all_humans = [agent for agent in self.model.humans.values() if agent.unique_id != self.unique_id]
            if not all_humans: # Skip if no other humans
                return # Or handle appropriately

            # Sort all potential human sources by current trust
            sorted_humans = sorted(all_humans, key=lambda x: self.trust.get(x.unique_id, 0), reverse=True)

            humans_to_query = set() # Use a set to avoid duplicates easily

            if self.agent_type == "exploitative":
                 # Query top N trusted + chance for 1 random non-friend
                 num_trusted = 4
                 # Add top trusted agents to query set
                 for agent in sorted_humans[:num_trusted]:
                     humans_to_query.add(agent)
                 # Small chance to query a non-friend
                 if random.random() < 0.15: # e.g., 15% chance (tune this)
                     non_friends = [h for h in all_humans if h.unique_id not in self.friends]
                     if non_friends:
                         humans_to_query.add(random.choice(non_friends))

            else: # Exploratory agent - query a diverse set
                 num_trusted = 2
                 num_random_others = 2
                 num_least_trusted = 1
                 # Add top trusted
                 for agent in sorted_humans[:num_trusted]:
                     humans_to_query.add(agent)
                 # Add some random others (not in top trusted)
                 others = [h for h in sorted_humans[num_trusted:] if h not in humans_to_query]
                 random.shuffle(others)
                 for agent in others[:num_random_others]:
                      humans_to_query.add(agent)
                 # Add some least trusted (if list long enough and not already included)
                 if len(sorted_humans) > num_trusted + num_random_others:
                      least_trusted_candidates = sorted_humans[-num_least_trusted:]
                      for agent in least_trusted_candidates:
                           humans_to_query.add(agent)

            # Convert set back to list for querying
            query_list = list(humans_to_query)
            random.shuffle(query_list) # Query in random order

            # Query selected humans
            for candidate in query_list:
                 rep = candidate.report_beliefs(self.pos)
                 # Take first report received for a cell (simplest)
                 for cell, value in rep.items():
                     if cell not in reports:
                         reports[cell] = value

            # Determine source ID for credit assignment (Crucial for updates below)
            # Simplification: Assign credit to the *most trusted* agent that was in the query list.
            if query_list: # Check if any humans were queried
                  query_list_sorted_by_trust = sorted(query_list, key=lambda x: self.trust.get(x.unique_id, 0), reverse=True)
                  source_agent_id_for_credit = query_list_sorted_by_trust[0].unique_id
            # --- End Modified Human Source Selection ---

            # Weight human call 

            self.accum_calls_human += 5
            self.accum_calls_total += 5

        self.tokens_this_tick[mode_choice] = self.tokens_this_tick.get(mode_choice, 0) + 1

        trust_reward = 0

        # Acceptance Logic
        for cell, reported_value in reports.items():
             belief_info = self.beliefs.get(cell, {'level': 0, 'confidence': 0.1})
             old_belief_level = belief_info.get('level', 0) if isinstance(belief_info, dict) else belief_info
             d = abs(reported_value - old_belief_level)

             # Use source_agent_id_for_credit to check if the 'responsible' source is a friend
             friend_weight = 1.5 if (source_agent_id_for_credit and source_agent_id_for_credit in self.friends) else 1.0
             P_accept = 1.0 if d == 0 else (self.D ** self.delta) / ((d ** self.delta) + (self.D ** self.delta)) * friend_weight

             if random.random() < P_accept:
                  # Check if we determined a source for credit assignment
                 if source_agent_id_for_credit:
                     if source_agent_id_for_credit not in self.trust: # Safety check
                         base = self.model.base_ai_trust if source_agent_id_for_credit.startswith("A_") else self.model.base_trust
                         self.trust[source_agent_id_for_credit] = base
                    # --- Belief Update with Inertia ---
                      # Get old belief level and confidence
                     old_belief_info = self.beliefs.get(cell, {'level': 0, 'confidence': 0.1})
                     old_level = old_belief_info.get('level', 0) if isinstance(old_belief_info, dict) else old_belief_info
                     #old_confidence = old_belief_info.get('confidence', 0.1) if isinstance(old_belief_info, dict) else 0.
                     # Calculate difference 'd' (already calculated for P_accept, but recalculate for clarity/safety)
                     d = abs(reported_value - old_level)
                     
                     # Calculate intensity factor based on difference, D, and delta
                     if d == 0:
                        intensity_factor = 1.0
                     else:
                      # Use the agent's existing D and delta for acceptance probability parameters
                      # These define how strongly the update magnitude decays with difference
                        intensity_factor = (self.D ** self.delta) / ((d ** self.delta) + (self.D ** self.delta))
                      # Note: This factor is <= 1.0

                      # Calculate the adjustment, scaled by belief_learning_rate and intensity_factor
                      adjustment = self.belief_learning_rate * intensity_factor * (reported_value - old_level)

                      # Calculate new level
                      new_level = old_level + adjustment
                      new_level = int(round(new_level)) # Round to nearest integer level
                      new_level = max(0, min(5, new_level)) # Clip to bounds [0, 5]

                      # Confidence update (keeping simple version for now: based on source trust)
                      new_confidence = self.trust.get(source_agent_id_for_credit, 0.5) * 0.9

                      # Store updated belief
                      self.beliefs[cell] = {'level': new_level, 'confidence': new_confidence}           
                     

                     # Trust update logic using source_agent_id_for_credit
                     if self.agent_type == "exploitative":
                         delta = 0.075 if self.trust_update_mode == "average" else 0.25
                         trust_reward += delta * friend_weight
                         self.trust[source_agent_id_for_credit] = min(1.0, self.trust.get(source_agent_id_for_credit, 0) + delta * friend_weight)

                     # Update acceptance counters (check if source was human or AI)
                     if source_agent_id_for_credit.startswith("H_"):
                         self.accepted_human += 1
                         if source_agent_id_for_credit in self.friends:
                             self.accepted_friend += 1
                     else: # AI source
                         self.accepted_ai += 1
                 else:
                     # This case should ideally not happen if mode_choice was valid
                     print(f"Warning: Could not determine source for credit for accepted info at cell {cell}")


        # Q-Update based on trust_reward for exploitative agents
        if self.agent_type == "exploitative" and trust_reward > 0:
              if mode_choice not in self.q_table: self.q_table[mode_choice] = 0.0 # Safety
              old_q = self.q_table[mode_choice]
              if isinstance(trust_reward, (int, float)): # Safety
                  self.q_table[mode_choice] = old_q + self.learning_rate * (trust_reward - old_q)

   
    def send_relief(self):
        max_target_cells = 5 # <-- Max cells to target
        height, width = self.model.disaster_grid.shape
        cells = [(x, y) for x in range(width) for y in range(height)]

        # Calculate expected need score for cells believed to be potentially needy
        cell_scores = []
        for cell, belief_info in self.beliefs.items():
            # Ensure belief_info is a dictionary before accessing
            if isinstance(belief_info, dict):
                level = belief_info.get('level', 0)
                confidence = belief_info.get('confidence', 0.1) # Use default if confidence missing

                # Filter: Consider cells believed to be level 4 or 5
                if level >= 4:
                    # Calculate score = level * confidence (adjust level weight if needed, e.g., level^2 * conf)
                    score = level * confidence
                    cell_scores.append({'cell': cell, 'score': score, 'level': level}) # Store level too
            # else: belief_info might be old format (int) or invalid, ignore

        # Sort potential target cells by descending score
        need_cells_sorted_by_score = sorted(cell_scores, key=lambda x: x['score'], reverse=True)

        # Select the top 'max_target_cells' based on the score
        top_cells_info = need_cells_sorted_by_score[:max_target_cells]
        # Extract just the cell coordinates for the model's tracking
        targeted_cells_coords = [item['cell'] for item in top_cells_info]

        if targeted_cells_coords:
            # Record which cells were targeted this tick in the model's central tracker
            agent_type_key = 'exploit' if self.agent_type == 'exploitative' else 'explor'
            for cell in targeted_cells_coords:
                # Ensure cell is valid before adding to model tracker
                if 0 <= cell[0] < self.model.width and 0 <= cell[1] < self.model.height:
                # Initialize cell entry if it doesn't exist
                    if cell not in self.model.tokens_this_tick:
                        self.model.tokens_this_tick[cell] = {'exploit': 0, 'explor': 0}
                    # Increment the count for the specific agent type
                    self.model.tokens_this_tick[cell][agent_type_key] += 1
                # else: ignore invalid cell target

            # Prepare data for pending reward: needs list of (cell, level) tuples
            # The reward logic needs the belief level at the time of sending relief
            need_cells_for_reward = [(item['cell'], item['level']) for item in top_cells_info]

            # Append pending reward: (due tick, mode, list_of_cells_with_level)
            if self.tokens_this_tick: # Check if any info was requested this tick
                responsible_mode = list(self.tokens_this_tick.keys())[0] # Still using simple credit assignment
                self.pending_rewards.append((self.model.tick + 2, responsible_mode, need_cells_for_reward)) # Pass list of (cell, level) tuples

            # Reset the agent's record of modes used for info this tick, ready for next step.
            self.tokens_this_tick = {}

    def process_reward(self):

        """Compute and return numeric reward for expired pending rewards."""
        current_tick = self.model.tick
        expired = [r for r in self.pending_rewards if r[0] <= current_tick]
        self.pending_rewards = [r for r in self.pending_rewards if r[0] > current_tick]
        total_agent_reward = 0

        # Outer loop unpacks: (tick, mode, list_of_tuples)
        # The list_of_tuples is assigned to the variable 'cells'
        for tick, mode, cells in expired: # 'cells' holds [(cell, belief_level), ...]
            reward = 0
            # --- Temporary counters for this reward batch ---
            correct_in_batch = 0
            incorrect_in_batch = 0

            # --- FIX 1: Use the correct variable 'cells' from the outer loop ---
            for cell, belief_level in cells: # belief_level is belief at time of sending
                # Check cell validity
                if 0 <= cell[0] < self.model.width and 0 <= cell[1] < self.model.height:
                   # This is the TRUE level at the time reward is processed
                   actual_level = self.model.disaster_grid[cell[0], cell[1]]

                   # --- Correctness Counting (using actual_level) ---
                   if actual_level >= 4:
                      correct_in_batch += 1
                   else: # actual_level is 0, 1, 2, or 3
                        incorrect_in_batch += 1
                   if actual_level == 5:       # Use actual_level here
                        reward += 5
                   elif actual_level == 4:     # Use actual_level here
                        reward += 2
                   elif actual_level <= 2:     # Use actual_level here (Level 3 gives 0 reward)
                        reward -= 2
                   # --- End Reward Calculation ---

                else:
                   # Handle invalid cell if needed
                   pass # Skipping reward calculation for invalid cells

           # --- Accumulate totals for the agent ---
            self.correct_targets += correct_in_batch
            self.incorrect_targets += incorrect_in_batch
            # --- End Accumulate ---

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
                trust_change_factor_neg = 0.03 # Factor for negative reward <-- ENSURE THIS IS HERE
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

        self.apply_trust_decay() # Apply slow decay to all relationships

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
        # sensed_vals is an array of integer levels sensed by the AI
        sensed_vals = np.array([self.sensed.get(cell, 0) for cell in cells]) # Ensure default 0 if key somehow missing

        human_levels_list = []
        for i, cell in enumerate(cells):
            # Get the human's belief about the cell (might be dict or None/default)
            belief_info = human_beliefs.get(cell)

            # Extract the level if it's a dictionary, otherwise use default
            if isinstance(belief_info, dict):
                # Get level from dict, default to AI's sensed value if level key missing (unlikely)
                human_level = belief_info.get('level', sensed_vals[i])
            # elif belief_info is not None: # Fallback: If belief is somehow still int?
            #     human_level = belief_info
            else:
                # Default: If human has no belief OR belief_info wasn't a dict, use AI's sensed value
                human_level = sensed_vals[i]

            human_levels_list.append(human_level)

        # human_vals is now an array of integer levels from human beliefs (or defaults)
        human_vals = np.array(human_levels_list)


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
    # ---2x2 subplots ---
    fig, ax = plt.subplots(2, 2, figsize=(14, 12)) # Adjust figsize as needed
    fig.suptitle(f"Model State at Tick {tick}", fontsize=16) # Add overall title

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


    # --- Plot 3: Exploit Tokens (Bottom Left - ax[1, 0]) ---
    vmax_exploit = max(1, max_exploit_tokens) # Ensure vmax >= 1
    im_tok_exp = ax[1, 0].imshow(exploit_token_grid.T, cmap='Blues', origin='lower', vmin=0, vmax=vmax_exploit, interpolation='nearest')
    ax[1, 0].set_title("Exploitative Tokens Sent")
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])
    fig.colorbar(im_tok_exp, ax=ax[1, 0], label="# Exploit Tokens", shrink=0.8)
    # Optional: Overlay exploit agent locations lightly
    ax[1, 0].scatter(exploit_x, exploit_y, color='darkblue', marker='o', s=20, alpha=0.3)
    ax[1, 0].set_xlim(-0.5, model.width - 0.5)
    ax[1, 0].set_ylim(-0.5, model.height - 0.5)


    # --- Plot 4: Exploratory Tokens (Bottom Right - ax[1, 1]) ---
    vmax_explor = max(1, max_explor_tokens) # Ensure vmax >= 1
    im_tok_er = ax[1, 1].imshow(explor_token_grid.T, cmap='Oranges', origin='lower', vmin=0, vmax=vmax_explor, interpolation='nearest')
    ax[1, 1].set_title("Exploratory Tokens Sent")
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    fig.colorbar(im_tok_er, ax=ax[1, 1], label="# Explor Tokens", shrink=0.8)
    # Optional: Overlay explor agent locations lightly
    ax[1, 1].scatter(explor_x, explor_y, color='darkorange', marker='s', s=20, alpha=0.3)
    ax[1, 1].set_xlim(-0.5, model.width - 0.5)
    ax[1, 1].set_ylim(-0.5, model.height - 0.5)


    # --- Final Touches for Layout ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    filepath = os.path.join(save_dir, f"grid_state_tick_{tick:04d}.png")
    plt.savefig(filepath)
    plt.close(fig) # Close figure to free memory

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
