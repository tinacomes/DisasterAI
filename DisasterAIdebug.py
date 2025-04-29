
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
                 learning_rate=0.1, epsilon=0.3,
                 # --- Q-learning & Behavior Tuning Parameters ---
                 #exploit_trust_weight=0.7,    # Q-target: For 'human' mode, weight for avg FRIEND trust (exploiter)
                 self_confirm_weight=0.9,     # Q-target: For 'self_action', weight for confirmation value (exploiter)
                 self_confirm_boost=1.5,      # Multiplier for avg confidence to get confirmation value V_C (tune)
                 confirmation_q_lr=0.05,       # Learning rate for Q-boost on confirmation
                 #exploit_ai_trust_weight=0.95, # Q-target: For 'A_k' modes, trust weight (exploiter) - very high
                 trust_learning_rate=0.05,  # Learning rate for trust updates
                 explore_reward_weight=0.8,   # Q-target: For all modes, reward weight (exploratory)
                 exploit_trust_lr=0.03, # Low trust LR for exploiters
                 explor_trust_lr=0.05,  # Higher trust LR for explorers
                 exploit_friend_bias=0.1,     # Action Selection: Bias added to 'human' score (exploiter) (tune)
                 exploit_self_bias=0.1):
                 #exploiter_trust_lr=0.1):
                 #exploit_reward_weight=0.2,  # Reward weight for exploitative agents
                 #onfirmation_weight=0.2):     # Action Selection: Bias added to 'self_action' score (exploiter) (tune)

        # Use workaround: call parent initializer with model only, then set attributes.
        super(HumanAgent, self).__init__(model)
        self.unique_id = unique_id
        self.model = model

        self.id_num = id_num
        self.agent_type = agent_type
        self.share_confirming = share_confirming
        self.trust_learning_rate = exploit_trust_lr if self.agent_type == "exploitative" else explor_trust_lr
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.pos = None

        self.exploration_targets = []

        # --- Store Q-learning & Behavior Parameters ---
        # self.exploit_trust_weight = exploit_trust_weight
        self.self_confirm_weight = self_confirm_weight
        self.self_confirm_boost = self_confirm_boost
        self.confirmation_q_lr = confirmation_q_lr
        # self.exploit_ai_trust_weight = exploit_ai_trust_weight
        self.explore_reward_weight = explore_reward_weight
        self.exploit_friend_bias = exploit_friend_bias
        self.exploit_self_bias = exploit_self_bias
        self.exploit_trust_lr = exploit_trust_lr
        # self.exploit_reward_weight = exploit_reward_weight
        # self.confirmation_weight = confirmation_weight
        # self.min_accept_chance = min_accept_chance


        # --- Agent State ---
        self.beliefs = {} # {(x, y): {'level': L, 'confidence': C}}
        self.trust = {}# {f"A_{k}": model.base_ai_trust for k in range(model.num_ai)} # Trust in AI agents
        self.q_table = {}
        # Human trust initialized later in model setup
        self.friends = set() # Use set for efficient checking ('H_j' format)
        self.pending_rewards = [] # [(tick_due, mode, [(cell, belief_level), ...]), ...]
        self.tokens_this_tick = {} # Tracks mode choice leading to send_relief THIS tick
        self.last_queried_source_ids = [] # Temp store for source IDs

        # --- Q-Table for Source Values ---
        self.q_table = {f"A_{k}": 0.0 for k in range(model.num_ai)}
        self.q_table["human"] = 0.05 # Represents generic value of querying humans
        self.q_table["self_action"] = 0.0

        # --- Belief Update Parameters ---
        # These control how beliefs change when info is ACCEPTED (separate from Q-learning)
        self.D = 2.0 if agent_type == "exploitative" else 4 # Acceptance threshold parameter
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


    def initialize_beliefs(self, assigned_rumor=None):
        """
        Initializes agent beliefs with default values, applies assigned rumor if provided,
        and senses the local environment. Ensures all cells have valid belief dictionaries.
        """
        height, width = self.model.disaster_grid.shape
        rumor_epicenter = None
        rumor_intensity = 0
        rumor_conf = 0.6
        rumor_radius = 0

        if assigned_rumor:
            rumor_epicenter, rumor_intensity, rumor_conf, rumor_radius = assigned_rumor

        # Initialize belief grid for ALL cells in the grid
        for x in range(width):
            for y in range(height):
                cell = (x, y)

                # Default initialization - all cells start with minimal belief
                initial_level = 0
                initial_conf = 0.1

                # Calculate distance from agent for sensing
                distance_from_agent = math.sqrt((x - self.pos[0])**2 + (y - self.pos[1])**2)
                sense_radius = 3 if self.agent_type == "exploratory" else 2

                # If cell is within sensing range, initialize with noisy perception of actual disaster
                if distance_from_agent <= sense_radius:
                    try:
                        # Get actual level with bounds checking
                        if 0 <= x < self.model.width and 0 <= y < self.model.height:
                            actual_level = self.model.disaster_grid[x, y]  #

                            # Add small noise to initial sensing
                            if random.random() < 0.2:  # 20% chance of noisy reading
                                initial_level = max(0, min(5, actual_level + random.choice([-1, 0, 1])))
                            else:
                                initial_level = actual_level

                            initial_conf = 0.7 if self.agent_type == "exploratory" else 0.5  # Explorers more confident
                    except (IndexError, TypeError) as e:
                        # Log the error for debugging
                        if self.model.debug_mode:
                            print(f"Warning: Error accessing disaster grid at {x},{y} for agent {self.unique_id}: {e}")
                        # Keep default values set above

                # Apply rumor effects (overlay on top of sensed info)
                if rumor_epicenter:
                    dist_from_rumor = math.sqrt((x - rumor_epicenter[0])**2 + (y - rumor_epicenter[1])**2)
                    if dist_from_rumor < rumor_radius:
                        rumor_level = min(5, int(round(3 + rumor_intensity)))
                        # Only override if rumor level is higher or agent has low confidence
                        if rumor_level > initial_level or initial_conf < rumor_conf:
                            initial_level = rumor_level
                            initial_conf = rumor_conf

                # Make sure to set belief for every cell
                self.beliefs[cell] = {'level': initial_level, 'confidence': initial_conf}

        # Debug logging for key agents
       # if self.unique_id in [f"H_0", f"H_{self.model.num_humans // 2}"]:
          #  print(f"\n--- Initial Beliefs for Agent {self.unique_id} ({self.agent_type}) ---")
           # high_belief_cells = [(cell, info['level']) for cell, info in self.beliefs.items() if info['level'] >= 3]
           # if high_belief_cells:
            #    print(f"  Found {len(high_belief_cells)} initial high-belief (L3+) cells.")
             #   max_lvl = max(info['level'] for info in self.beliefs.values())
          #      print(f"  Max initial believed level: {max_lvl}")
         #   else:
           #     print("  WARNING: No initial high-belief (L3+) cells found!")
           #     max_lvl = max(info['level'] for info in self.beliefs.values())
          #      print(f"  Max initial believed level: {max_lvl}")
          #  print("------------------------------------\n")

        # Verify that all cells have been initialized
        if len(self.beliefs) != width * height:
            print(f"WARNING: Agent {self.unique_id} beliefs not fully initialized! Expected {width*height}, got {len(self.beliefs)}")
            # Add missing beliefs if any
            for x in range(width):
                for y in range(height):
                    cell = (x, y)
                    if cell not in self.beliefs:
                        self.beliefs[cell] = {'level': 0, 'confidence': 0.1}
                        if self.model.debug_mode:
                            print(f"  Added missing belief for cell {cell}")

        # Additional sensing after initialization
        self.sense_environment()

    def apply_confidence_decay(self):
        decay_rate = 0.0005
        min_confidence_floor = 0.02
        for cell, belief in self.beliefs.items():
            if isinstance(belief, dict): # Ensure it's a belief dict
                confidence = belief.get('confidence', 0.1)
                if confidence > min_confidence_floor:
                    new_confidence = max(min_confidence_floor, confidence - decay_rate)
                    self.beliefs[cell]['confidence'] = new_confidence

    def query_source(self, source_id, interest_point, query_radius):
        source_agent = self.model.humans.get(source_id) or self.model.ais.get(source_id)
        if source_agent:
            if hasattr(source_agent, 'report_beliefs'):
                return source_agent.report_beliefs(interest_point, query_radius)
        return {}

    def sense_environment(self):
        pos = self.pos
        radius = 2 if self.agent_type == "exploitative" else 3
        cells = self.model.grid.get_neighborhood(pos, moore=True, radius=radius, include_center=True)
        cells = self.model.grid.get_neighborhood(pos, moore=True, radius=radius, include_center=True)
        for cell in cells:
            if 0 <= cell[0] < self.model.width and 0 <= cell[1] < self.model.height:
                actual = self.model.disaster_grid[cell[0], cell[1]]
                noise_roll = random.random()
                noise_threshold = 0.1

                belief_level = 0
                belief_conf = 0.1

                if noise_roll < noise_threshold: # Noisy Read
                    belief_level = max(0, min(5, actual + random.choice([-1, 1])))
                    belief_conf = 0.5
                else: # Accurate Read
                    belief_level = actual
                    if self.agent_type == "exploitative":
                        # Keep slightly lower base confidence, but boost strongly for high levels
                        belief_conf = 0.80 # Lower base than explorer
                        if belief_level >= 3:
                            belief_conf = 0.95 # HIGH confidence if accurately sensing L3+
                        elif belief_level > 0:
                            belief_conf = 0.90 # Moderate confidence for L1/L2
                    else: # Exploratory (Keep previous boost: 0.90 base, 0.98 for L>0)
                        belief_conf = 0.90
                        if belief_level > 0:
                            belief_conf = 0.98

                # Blend confidence
                old_belief_info = self.beliefs.get(cell, {'confidence': 0.1})
                old_confidence = old_belief_info.get('confidence', 0.1)
                sense_weight = 0.8 if noise_roll >= noise_threshold else 0.6
                final_confidence = (sense_weight * belief_conf) + ((1 - sense_weight) * old_confidence)
                final_confidence = max(0.01, min(0.99, final_confidence))

                self.beliefs[cell] = {'level': belief_level, 'confidence': final_confidence}



    def report_beliefs(self, interest_point, query_radius):  # New Arguments
        """Reports own beliefs about cells within query_radius of the interest_point."""
        report = {}
        # Check if agent can report (e.g., not in high disaster zone)
        try:
            current_pos_level = self.model.disaster_grid[self.pos[0], self.pos[1]]
            if current_pos_level >= 4 and random.random() < 0.1:
                return {}
        except (TypeError, IndexError):
            return {}

        # Get neighborhood around the specified interest_point
        cells_to_report_on = self.model.grid.get_neighborhood(
            interest_point,  # Use the point the caller is interested in
            moore=True,
            radius=1,  # Use the radius requested by the caller
            include_center=True
        )

        for cell in cells_to_report_on:
            # Check if cell is valid before getting belief
            if 0 <= cell[0] < self.model.width and 0 <= cell[1] < self.model.height:
                # Report own belief about the requested cell
                belief_info = self.beliefs.get(cell, {'level': 0, 'confidence': 0.1})
                current_level = belief_info.get('level', 0)

                # Apply reporting noise (keep this for realism)
                if random.random() < 0.05:  # 0.1
                    noisy_level = max(0, min(5, current_level + random.choice([-1, 1])))
                    report[cell] = noisy_level
                else:
                    report[cell] = current_level

        return report  # Report contains beliefs about the queried area

    def find_believed_epicenter(self):
        """Finds the cell with the highest believed disaster level."""
        max_level = -1
        best_cells = []
        # Check own beliefs first
        for cell, belief_info in self.beliefs.items():
            if isinstance(belief_info, dict):
                level = belief_info.get('level', -1)
                if level > max_level:
                    max_level = level
                    best_cells = [cell]
                elif level == max_level:
                    best_cells.append(cell)

        # If no beliefs > 0 return None
        if best_cells and max_level > 0: # Only consider if found something >= L1
            self.believed_epicenter = random.choice(best_cells) # Return one coordinate tuple (x,y)
        else:
            self.believed_epicenter = None # Indicate no clear epicenter believed yet


    def find_exploration_targets(self, num_targets=1):
        """Finds cells scoring high on a weighted sum of believed level and uncertainty."""
        candidates = []
        min_level_to_explore = 1  # Consider L1+ cells

        # Check if we have any beliefs to work with
        if not self.beliefs:
            # Handle case with no beliefs
            if self.model.debug_mode:
                print(f"Agent {self.unique_id}: No beliefs available for exploration targeting")
            self.exploration_targets = []
            return

        # Gather all valid candidates with their scores
        for cell, belief_info in self.beliefs.items():
            if isinstance(belief_info, dict):
                level = belief_info.get('level', 0)

                # Skip L0 cells for primary exploration targets
                if level < min_level_to_explore:
                    continue

                # Check if cell coordinates are valid
                if not (0 <= cell[0] < self.model.width and 0 <= cell[1] < self.model.height):
                    continue

                confidence = belief_info.get('confidence', 0.1)
                uncertainty = 1.0 - confidence

                # --- Scoring Logic: Weighted Sum ---
                level_weight = 0.7  # Focus heavily on finding higher levels
                uncertainty_weight = 0.3  # Uncertainty is secondary, but still relevant

                # Normalize level (0-5 -> 0-1)
                normalized_level = level / 5.0

                score = (level_weight * normalized_level) + (uncertainty_weight * uncertainty)

                # Add to candidates
                candidates.append({'cell': cell, 'score': score, 'level': level, 'conf': confidence})

        # --- Fallback if no L1+ candidates found ---
        if not candidates:
            if self.model.debug_mode:
                print(f"Agent {self.unique_id} ({self.agent_type}): No L{min_level_to_explore}+ targets found. Using fallback.")

            # Look for cells with lowest confidence
            min_conf = 1.1
            lowest_conf_cells = []

            for cell, belief_info in self.beliefs.items():
                if isinstance(belief_info, dict):
                    # Check if cell coordinates are valid
                    if not (0 <= cell[0] < self.model.width and 0 <= cell[1] < self.model.height):
                        continue

                    conf = belief_info.get('confidence', 1.0)
                    if conf < min_conf:
                        min_conf = conf
                        lowest_conf_cells = [cell]
                    elif conf == min_conf:
                        lowest_conf_cells.append(cell)

            if lowest_conf_cells:
                fallback_cell = random.choice(lowest_conf_cells)
                candidates = [{'cell': fallback_cell, 'score': -1, 'level': '?', 'conf': min_conf}]
            else:
                # Ultimate fallback: pick some random cells
                random_cells = []
                for _ in range(3):  # Try to find 3 valid random cells
                    x = random.randrange(self.model.width)
                    y = random.randrange(self.model.height)
                    cell = (x, y)
                    if cell in self.beliefs and isinstance(self.beliefs[cell], dict):
                        random_cells.append(cell)

                if random_cells:
                    random_cell = random.choice(random_cells)
                    belief_info = self.beliefs.get(random_cell, {'level': 0, 'confidence': 0.1})
                    candidates = [{'cell': random_cell, 'score': -2, 'level': belief_info.get('level', 0),
                                  'conf': belief_info.get('confidence', 0.1)}]
                elif self.model.debug_mode:
                    print(f"SEVERE: Agent {self.unique_id} couldn't find any valid exploration targets")

        # Sort by score (highest score first) and select top candidates
        if candidates:
            candidates.sort(key=lambda x: x['score'], reverse=True)
            # Debug print top candidates periodically
            if self.model.debug_mode and self.model.tick % 10 == 1 and random.random() < 0.2:
                print(f"DEBUG Tick {self.model.tick} Agt {self.unique_id} Top Explore Candidates:")
                for cand in candidates[:min(5, len(candidates))]:
                    print(f"  Cell:{cand.get('cell')} Lvl:{cand.get('level')} Conf:{cand.get('conf'):.2f} Score:{cand.get('score'):.3f}")

            self.exploration_targets = [c['cell'] for c in candidates[:num_targets]]
        else:
            # Final fallback if all else fails
            if self.model.debug_mode:
                print(f"Agent {self.unique_id}: No exploration candidates found at all, using position as target")
            self.exploration_targets = [self.pos]  # Use current position as a last resort

    def apply_trust_decay(self):
         """Applies a slow decay to all trust relationships."""
         decay_rate = 0.002 if self.agent_type == "exploitative" else 0.002 # Slow general decay rate per step
         friend_decay_rate = 0.0005 if self.agent_type == "exploitative" else 0.001# # Slower decay for friends
         # Iterate over a copy of keys because dictionary size might change (though unlikely here)
         for source_id in list(self.trust.keys()):
              rate = friend_decay_rate if source_id in self.friends else decay_rate
              self.trust[source_id] = max(0, self.trust[source_id] - rate)

    def update_belief_bayesian(self, cell, reported_level, source_trust):
        """
        Update agent's belief about a cell using Bayesian principles.

        Parameters:
        - cell: The (x,y) coordinates of the cell
        - reported_level: The disaster level reported by the source
        - source_trust: The trust level for the source [0,1]
        """
        # Get current belief
        if cell not in self.beliefs:
            # Initialize if not present
            self.beliefs[cell] = {'level': 0, 'confidence': 0.1}

        current_belief = self.beliefs[cell]
        prior_level = current_belief.get('level', 0)
        prior_confidence = current_belief.get('confidence', 0.1)

        # Convert confidence to precision (inverse variance)
        # Higher confidence = lower variance = higher precision
        prior_precision = prior_confidence / (1 - prior_confidence + 1e-6)

        # Source precision is based on trust
        # Scale to make reasonable variance values
        source_precision = 4.0 * source_trust / (1 - source_trust + 1e-6)

        # Special case for low trust sources - reduce precision further
        if source_trust < 0.3:
            source_precision *= (source_trust / 0.3)

        # Special case for completely untrusted sources
        if source_trust < 0.05:
            # Almost ignore this information
            source_precision *= 0.1

        # Combine information using precision weighting (Bayesian update)
        # For normally distributed beliefs, optimal combination weights by precision
        posterior_precision = prior_precision + source_precision

        # Calculate the weighted update of belief level
        posterior_level = (prior_precision * prior_level + source_precision * reported_level) / posterior_precision

        # Convert precision back to confidence [0,1]
        posterior_confidence = posterior_precision / (1 + posterior_precision)

        # Constrain to valid ranges
        posterior_level = max(0, min(5, round(posterior_level)))
        posterior_confidence = max(0.01, min(0.99, posterior_confidence))

        # Apply agent-type-specific adjustments
        if self.agent_type == "exploitative":
            # Exploitative agents give more weight to consistent information
            if abs(posterior_level - prior_level) <= 1:
                # Information roughly confirms existing belief
                confirmation_boost = 0.1 * prior_confidence
                posterior_confidence = min(0.99, posterior_confidence + confirmation_boost)
        else:  # exploratory
            # Exploratory agents are more accepting of new information
            if abs(posterior_level - prior_level) >= 2:
                # Information significantly differs from prior
                # Increase confidence less for exploratory agents when beliefs change a lot
                posterior_confidence = max(0.05, posterior_confidence * 0.9)

        # Update the belief
        self.beliefs[cell] = {
            'level': int(posterior_level),  # Ensure integer
            'confidence': posterior_confidence
        }

        # Return whether this was a significant belief change
        return abs(posterior_level - prior_level) >= 1


    def seek_information(self):
        """
        Queries a single source for information about an interest point, processes the report,
        and updates beliefs if accepted. Includes robust error handling and logging.
        """
        reports = {}
        source_agent_id = None
        interest_point = None
        query_radius = 0
        eports = {}
        source_id = None  # Initialize source_id here, at the beginning of the method
        chosen_mode = None  # Initialize chosen_mode as well

        # init counters
        belief_updates = 0
        source_calls_human = 0
        source_calls_ai = 0

        # Add diagnostic logging for source selection
        decision_factors = {'q_values': {}, 'biases': {}, 'final_scores': {}}

        try:
            # Determine interest point based on agent type
            if self.agent_type == "exploitative":
                self.find_believed_epicenter()
                interest_point = self.believed_epicenter
                query_radius = 2

                # Fix: Ensure we always have a valid interest point
                if not interest_point or self.beliefs.get(interest_point, {}).get('level', 0) <= 0:
                    # First try: Ask most trusted friend
                    valid_friends = [f_id for f_id in self.friends if f_id in self.model.humans]
                    best_friend_id = None
                    max_friend_trust = -1.0
                    for f_id in valid_friends:
                        trust = self.trust.get(f_id, 0.0)
                        if trust > max_friend_trust:
                            max_friend_trust = trust
                            best_friend_id = f_id

                    if best_friend_id:
                        friend = self.model.humans[best_friend_id]
                        friend.find_believed_epicenter()
                        if friend.believed_epicenter and friend.beliefs.get(friend.believed_epicenter, {}).get('level', 0) >= 1:
                            interest_point = friend.believed_epicenter

                    # Second try: Use highest confidence cell
                    if not interest_point or self.beliefs.get(interest_point, {}).get('level', 0) <= 0:
                        # FIX: Initialize max_conf and highest_conf_cells properly
                        max_conf = -1
                        highest_conf_cells = []

                        # Ensure we have valid beliefs to search through
                        if len(self.beliefs) > 0:
                            for cell, belief_info in self.beliefs.items():
                                if isinstance(belief_info, dict):  # Make sure it's a valid belief dictionary
                                    conf = belief_info.get('confidence', 0.0)
                                    if conf > max_conf:
                                        max_conf = conf
                                        highest_conf_cells = [cell]
                                    elif conf == max_conf:
                                        highest_conf_cells.append(cell)

                        # FIX: Ensure we have at least one valid cell before choosing
                        if highest_conf_cells:
                            interest_point = random.choice(highest_conf_cells)
                        else:
                            # Absolute fallback: pick a random cell in the grid
                            interest_point = (random.randrange(self.model.width), random.randrange(self.model.height))
                            if self.model.debug_mode:
                                print(f"Agent {self.unique_id}: Using random fallback interest point {interest_point}")

            else:  # Exploratory
                self.find_exploration_targets()
                interest_point = self.exploration_targets[0] if self.exploration_targets else None
                query_radius = 3

                # FIX: Add robust fallback mechanism
                if not interest_point:
                    # Try finding highest confidence cells
                    max_conf = -1
                    highest_conf_cells = []

                    if len(self.beliefs) > 0:
                        for cell, belief_info in self.beliefs.items():
                            if isinstance(belief_info, dict):
                                conf = belief_info.get('confidence', 0.0)
                                if conf > max_conf:
                                    max_conf = conf
                                    highest_conf_cells = [cell]
                                elif conf == max_conf:
                                    highest_conf_cells.append(cell)

                    if highest_conf_cells:
                        interest_point = random.choice(highest_conf_cells)
                    else:
                        # Absolute fallback: pick a random cell
                        interest_point = (random.randrange(self.model.width), random.randrange(self.model.height))
                        if self.model.debug_mode:
                            print(f"Agent {self.unique_id}: Using random fallback interest point {interest_point}")

            # Final safety check
            if not interest_point:
                if self.model.debug_mode:
                    print(f"Agent {self.unique_id}: No valid interest point, skipping seek_information.")
                return

            # Validate coordinates are within grid bounds
            if not (0 <= interest_point[0] < self.model.width and 0 <= interest_point[1] < self.model.height):
                if self.model.debug_mode:
                    print(f"Agent {self.unique_id}: Interest point {interest_point} out of bounds, skipping seek_information.")
                return

            # Debug logging
            if self.model.debug_mode and random.random() < 0.05:  # Only log ~5% of decisions
                print(f"Tick {self.model.tick} Agent {self.unique_id} ({self.agent_type}) selected interest_point: {interest_point}")
                print(f" > My belief: {self.beliefs.get(interest_point, {})}")
                print(f" > Ground truth: {self.model.disaster_grid[interest_point[0], interest_point[1]]}")


            if not interest_point:
                print(f"Agent {self.unique_id}: No valid interest point, skipping seek_information.")
                return

            # Debug logging
            if self.model.debug_mode and random.random() < 0.05:  # Only log ~5% of decisions to avoid spam
                print(f"Tick {self.model.tick} Agent {self.unique_id} ({self.agent_type}) selected interest_point: {interest_point}")
                print(f" > My belief: {self.beliefs.get(interest_point, {})}")
                print(f" > Ground truth: {self.model.disaster_grid[interest_point[0], interest_point[1]]}")

            # Source selection (epsilon-greedy with type-specific biases)
            possible_modes = ["self_action", "human"] + [f"A_{k}" for k in range(self.model.num_ai)]

            # DIAGNOSTIC: Store Q-values
            for mode in possible_modes:
                decision_factors['q_values'][mode] = self.q_table.get(mode, 0.0)

            # Exploration case - record randomly chosen mode
            if random.random() < self.epsilon:
                chosen_mode = random.choice(possible_modes)
                decision_factors['selection_type'] = 'exploration'
                decision_factors['chosen_mode'] = chosen_mode
            else:
                # Exploitation case - record all factors in decision
                decision_factors['selection_type'] = 'exploitation'

                # Base scores are from Q-table
                scores = {mode: self.q_table.get(mode, 0.0) for mode in possible_modes}
                decision_factors['base_scores'] = scores.copy()

                # Add biases based on agent type
                decision_factors['biases'] = {}

                if self.agent_type == "exploitative":
                    # Exploitative agents prefer friends and self-confirmation
                    scores["human"] += self.exploit_friend_bias
                    scores["self_action"] += self.exploit_self_bias

                    decision_factors['biases']["human"] = self.exploit_friend_bias
                    decision_factors['biases']["self_action"] = self.exploit_self_bias

                    # AI alignment effect on exploitative agents
                    ai_alignment_factor = self.model.ai_alignment_level * 0.2
                    for k in range(self.model.num_ai):
                        ai_id = f"A_{k}"
                        ai_bias = ai_alignment_factor * self.trust.get(ai_id, 0.1)
                        scores[ai_id] += ai_bias
                        decision_factors['biases'][ai_id] = ai_bias

                else:  # exploratory
                    # Exploratory agents have a slight bias against self-confirmation
                    scores["self_action"] -= 0.05
                    decision_factors['biases']["self_action"] = -0.05

                    # Strengthen inverse alignment effect for exploratory agents
                    inverse_alignment_factor = (1.0 - self.model.ai_alignment_level) * 0.3  # Doubled from 0.15
                    baseline_ai_factor = 0.15  # Increased from 0.1

                    for k in range(self.model.num_ai):
                        ai_id = f"A_{k}"
                        # Combine baseline with inverse alignment for more stable behavior
                        ai_bias = baseline_ai_factor + inverse_alignment_factor
                        scores[ai_id] += ai_bias
                        decision_factors['biases'][ai_id] = ai_bias

                    # Reduce bias for human consultation when alignment is low
                    human_bias = self.model.ai_alignment_level * 0.1
                    scores["human"] -= human_bias
                    decision_factors['biases']["human"] = -human_bias
                    # Exploratory agents have a slight bias against self-confirmation
                    scores["self_action"] -= 0.05
                    decision_factors['biases']["self_action"] = -0.05


                # Add small random noise to break ties
                for mode in scores:
                    noise = random.uniform(-0.01, 0.01)
                    scores[mode] += noise
                    if 'noise' not in decision_factors:
                        decision_factors['noise'] = {}
                    decision_factors['noise'][mode] = noise

                decision_factors['final_scores'] = scores

                # Choose highest scoring mode
                chosen_mode = max(scores, key=scores.get)
                decision_factors['chosen_mode'] = chosen_mode

            # Log decision factors periodically
            if self.model.tick % 10 == 0 and random.random() < 0.2:  # Log ~20% of decisions every 10 ticks
                if self.model.debug_mode:
                    print(f"\nAgent {self.unique_id} ({self.agent_type}) source selection:")
                    print(f"  Decision type: {decision_factors['selection_type']}")
                    print(f"  Chosen mode: {decision_factors['chosen_mode']}")
                    if decision_factors['selection_type'] == 'exploitation':
                        print(f"  Base Q-values: {decision_factors['base_scores']}")
                        print(f"  Applied biases: {decision_factors['biases']}")
                        print(f"  Final scores: {decision_factors['final_scores']}")
                    print(f"  AI alignment level: {self.model.ai_alignment_level}")

            self.tokens_this_tick = {chosen_mode: 1}
            self.last_queried_source_ids = []

            # Query source

            if chosen_mode == "self_action":
                reports = self.report_beliefs(interest_point, query_radius)
                source_id = None  # No external source used
            elif chosen_mode == "human":
                valid_sources = [h for h in self.model.humans if h != self.unique_id]
                if not valid_sources:
                    return
                if not self.friends:
                    source_id = random.choice(valid_sources)
                else:
                    # Get friend with highest trust
                    friend_trust_pairs = [(fid, self.trust.get(fid, 0.1)) for fid in self.friends]
                    if friend_trust_pairs:
                        source_id = max(friend_trust_pairs, key=lambda x: x[1])[0]
                    else:
                        source_id = random.choice(valid_sources)

                source_agent = self.model.humans.get(source_id)
                if source_agent:
                    reports = source_agent.report_beliefs(interest_point, query_radius)
                    self.last_queried_source_ids = [source_id]
                    self.accum_calls_human += 1
                    self.accum_calls_total += 1
                else:
                    # Invalid source agent
                    source_id = None
            else:  # AI
                source_id = chosen_mode
                source_agent = self.model.ais.get(source_id)
                if source_agent:
                    reports = source_agent.report_beliefs(interest_point, query_radius, self.beliefs, self.trust.get(source_id, 0.1))
                    self.last_queried_source_ids = [source_id]
                    self.accum_calls_ai += 1
                    self.accum_calls_total += 1
                else:
                    # Invalid AI agent
                    source_id = None

            # Process reports and update beliefs
            belief_updates = 0
            source_trust = self.trust.get(source_id, 0.1) if source_id else 0.1

            for cell, reported_value in reports.items():
                if cell not in self.beliefs:
                    continue

                # Convert to integer level if it's not already
                reported_level = int(round(reported_value))

                # Use the Bayesian update function instead of the old P_accept logic
                significant_update = self.update_belief_bayesian(cell, reported_level, source_trust)

                # Track if this was a significant belief update
                if significant_update:
                    belief_updates += 1

                    # Track source acceptance (for metrics)
                    if source_id:
                        if source_id.startswith("H_"):
                            self.accepted_human += 1
                            if source_id in self.friends:
                                self.accepted_friend += 1
                        elif source_id.startswith("A_"):
                            self.accepted_ai += 1

        except Exception as e:
            print(f"ERROR in Agent {self.unique_id} seek_information at tick {self.model.tick}: {e}")
            import traceback
            traceback.print_exc()

    def send_relief(self):
        """
        Sends relief to high-belief cells, queues rewards, and tracks responsible sources.
        Ensures robust bounds checking and fallback handling.
        """
        max_target_cells = 5
        min_forced_tokens = 1

        try:
            cell_scores = []
            max_believed_level = -1
            for cell, belief_info in self.beliefs.items():
                if not isinstance(belief_info, dict):
                    continue
                level = belief_info.get('level', 0)
                confidence = belief_info.get('confidence', 0.1)
                max_believed_level = max(max_believed_level, level)

                if level >= 3:
                    score = (level / 5.0) * (confidence ** 1.5) if self.agent_type == "exploitative" else (
                        (1.0 - confidence) * 0.6 + (level / 5.0) * 0.4 + 0.2 * (
                            math.sqrt((cell[0] - self.pos[0])**2 + (cell[1] - self.pos[1])**2) / math.sqrt(self.model.width**2 + self.model.height**2)
                        )
                    )
                    if score > 0.01:
                        cell_scores.append({'cell': cell, 'score': score, 'level': level})

            # Select top targets
            top_cells = sorted(cell_scores, key=lambda x: x['score'], reverse=True)[:max_target_cells]
            targeted_cells = [item['cell'] for item in top_cells]
            reward_cells = [(item['cell'], item['level']) for item in top_cells]

            # Fallback if no targets
            if not targeted_cells and min_forced_tokens > 0 and max_believed_level > -1:
                best_cells = [cell for cell, belief_info in self.beliefs.items() if isinstance(belief_info, dict) and belief_info.get('level', -1) == max_believed_level]
                if best_cells:
                    random.shuffle(best_cells)
                    targeted_cells = best_cells[:min_forced_tokens]
                    reward_cells = [(cell, max_believed_level) for cell in targeted_cells]

            if targeted_cells:
                responsible_mode = list(self.tokens_this_tick.keys())[0] if self.tokens_this_tick else "self_action"
                agent_type_key = 'exploit' if self.agent_type == 'exploitative' else 'explor'
                for cell in targeted_cells:
                    if 0 <= cell[0] < self.model.width and 0 <= cell[1] < self.model.height:
                        if cell not in self.model.tokens_this_tick:
                            self.model.tokens_this_tick[cell] = {'exploit': 0, 'explor': 0}
                        self.model.tokens_this_tick[cell][agent_type_key] += 1

                self.pending_rewards.append((
                    self.model.tick + 2,
                    responsible_mode,
                    reward_cells,
                    self.last_queried_source_ids
                ))

            self.tokens_this_tick = {}

        except Exception as e:
            print(f"ERROR in Agent {self.unique_id} send_relief at tick {self.model.tick}: {e}")


    def process_reward(self):
        """
        Processes expired rewards, updates Q-table and trust based on outcomes.
        Includes robust error handling and normalized reward signals.
        """
        current_tick = self.model.tick
        expired = [r for r in self.pending_rewards if r[0] <= current_tick]
        self.pending_rewards = [r for r in self.pending_rewards if r[0] > current_tick]
        total_reward = 0

        try:
            for reward_tick, mode, cells_and_beliefs, source_ids in expired:
                if not cells_and_beliefs:
                    continue

                batch_reward = 0
                correct_in_batch = 0
                incorrect_in_batch = 0
                cell_rewards = []

                for cell, belief_level in cells_and_beliefs:
                    if not (0 <= cell[0] < self.model.width and 0 <= cell[1] < self.model.height):
                        continue
                    actual_level = self.model.disaster_grid[cell[0], cell[1]]
                    is_correct = actual_level >= 3
                    if is_correct:
                        correct_in_batch += 1
                    else:
                        incorrect_in_batch += 1

                    cell_reward = 5 if actual_level == 5 else 3 if actual_level == 4 else 2 if actual_level == 3 else -1
                    cell_rewards.append(cell_reward)
                    if is_correct and cell in self.beliefs and isinstance(self.beliefs[cell], dict):
                        self.beliefs[cell]['confidence'] = min(0.99, self.beliefs[cell].get('confidence', 0.2) + 0.2)

                batch_reward = max(cell_rewards) if correct_in_batch > 0 else sum(cell_rewards) if cell_rewards else -1.5
                total_reward += batch_reward

                self.correct_targets += correct_in_batch
                self.incorrect_targets += incorrect_in_batch

                # Normalize reward to [-1, 1]
                scaled_reward = 1.0 if batch_reward >= 5 else 0.6 + 0.6 * (batch_reward - 2) / 3 if batch_reward >= 2 else 0.4 + 0.4 * (batch_reward - 1) if batch_reward >= 1 else max(-1.0, batch_reward / 1.5)
                target_trust = (scaled_reward + 1.0) / 2.0

                # Update Q-table and trust
                if mode == "self_action":
                    old_q = self.q_table.get("self_action", 0.0)
                    self.q_table["self_action"] = old_q + self.learning_rate * (scaled_reward - old_q)
                elif source_ids:
                    for source_id in source_ids:
                        if source_id in self.q_table:
                            old_q = self.q_table[source_id]
                            self.q_table[source_id] = old_q + self.learning_rate * (scaled_reward - old_q)
                        if source_id in self.trust:
                            old_trust = self.trust[source_id]

                            # Enhanced trust update logic based on agent type
                            trust_change = self.trust_learning_rate * (target_trust - old_trust)

                            # Exploitative agents increase trust more for confirmatory info
                            if self.agent_type == "exploitative" and source_id.startswith("A_"):
                                # Check if AI's report aligned with agent's existing beliefs
                                confirmation_bonus = 0.0
                                for cell, _ in cells_and_beliefs:
                                    if cell in self.beliefs:
                                        agent_belief = self.beliefs[cell].get('level', 0)
                                        # Alignment is measured by closeness of beliefs
                                        confirmation_bonus += (1.0 - abs(agent_belief - actual_level) / 5.0) * 0.1

                                # Higher alignment levels make exploitative agents trust AI more when it confirms beliefs
                                ai_alignment = self.model.ai_alignment_level
                                trust_change *= (1.0 + confirmation_bonus * ai_alignment * 2.0)

                            # Exploratory agents increase trust more for correct information
                            elif self.agent_type == "exploratory" and source_id.startswith("A_"):
                                # Boost trust change for correct information proportionally to accuracy
                                accuracy_bonus = correct_in_batch / max(1, len(cells_and_beliefs))

                                # Lower alignment makes exploratory agents trust AI more for accurate info
                                inverse_alignment = 1.0 - self.model.ai_alignment_level
                                trust_change *= (1.0 + accuracy_bonus * inverse_alignment * 1.5)

                            self.trust[source_id] = max(0.0, min(1.0, old_trust + trust_change))

        except Exception as e:
            print(f"ERROR in Agent {self.unique_id} process_reward at tick {current_tick}: {e}")

        return total_reward


    def step(self):
        self.sense_environment()
        self.seek_information()
        self.send_relief()
        reward = self.process_reward()


        self.apply_trust_decay() # Apply slow trust decay to all relationships
        self.apply_confidence_decay() # Apply slow decay to confidence
        #confidence_decay_rate = 0.005 # Start very small and tune

        #confidence decay
        #for cell in self.beliefs:
           # if isinstance(self.beliefs[cell], dict):
            #  current_conf = self.beliefs[cell].get('confidence', 0.1)
              # Prevent decay below a minimum floor, maybe slightly above initial
           #   min_conf_floor = 0.05
            #  self.beliefs[cell]['confidence'] = max(min_conf_floor, current_conf - confidence_decay_rate)
        return reward

    def smooth_friend_trust(self):

        if self.friends:
          # --- Trust Smoothing (Keep, maybe reduce weight) ---
            friend_ids = [f for f in self.friends if f in self.trust] # Ensure friend exists
            if not friend_ids: return

            friend_values = [self.trust.get(f, 0) for f in self.friends]
            avg_friend = sum(friend_values) / len(friend_values)
            smoothing_factor = 0.1
            for friend in self.friends:
                self.trust[friend] = (1-smoothing_factor) * self.trust[friend] + smoothing_factor * avg_friend


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
        self.total_cells = self.model.width * self.model.height
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
                if random.random() < 0.1: # 10% chance of AI sensing noise
                    value = max(0, min(5, value + random.choice([-1, 1])))
                self.sensed[cell] = value
                self.memory[(current_tick, cell)] = value


    def report_beliefs(self, interest_point, query_radius, caller_beliefs, caller_trust_in_ai):
        """
        Reports AI's beliefs about cells within query_radius of interest_point,
        applying alignment based on caller's trust and beliefs.
        """
        report = {}
        # Determine the cells the caller is asking about
        cells_to_report_on = self.model.grid.get_neighborhood(
            interest_point,
            moore=True,
            radius=query_radius,
            include_center=True
        )

        valid_cells_in_query = []
        sensed_vals_list = []
        human_vals_list = []
        human_confidence_list = []  # Track confidence to weight alignment

        # Prepare data needed for alignment calculation
        for cell in cells_to_report_on:
            if 0 <= cell[0] < self.model.width and 0 <= cell[1] < self.model.height:
                # Get AI's sensed value for this cell (might be None if not sensed)
                ai_sensed_val = self.sensed.get(cell)

                # Only proceed if AI actually sensed this cell
                if ai_sensed_val is not None:
                    valid_cells_in_query.append(cell)
                    sensed_vals_list.append(int(ai_sensed_val))

                    # Get the CALLER'S belief for this cell
                    caller_belief_info = caller_beliefs.get(cell, {'level': 0, 'confidence': 0.1})
                    human_level = caller_belief_info.get('level', 0)
                    human_confidence = caller_belief_info.get('confidence', 0.1)
                    human_vals_list.append(int(human_level))
                    human_confidence_list.append(human_confidence)

        # If no relevant sensed data, return empty report
        if not valid_cells_in_query:
            return {}

        sensed_vals = np.array(sensed_vals_list)
        human_vals = np.array(human_vals_list)
        human_conf = np.array(human_confidence_list)

        # --- Apply "Confidence-Weighted Alignment" Logic ---
        alignment_strength = self.model.ai_alignment_level

        # Formula:
        # - For high human confidence: AI reports shift more toward human beliefs
        # - For low trust: Shift more to match human's beliefs (attempting to build trust)

        # Base amplification inversely proportional to trust (low trust → higher alignment)
        low_trust_amplification = getattr(self.model, 'low_trust_amplification_factor', 0.3)
        clipped_trust = max(0.0, min(1.0, caller_trust_in_ai))

        # Calculate personalized alignment for each cell based on confidence
        alignment_factors = alignment_strength * (1.0 + human_conf) + low_trust_amplification * (1.0 - clipped_trust)
        alignment_factors = np.clip(alignment_factors, 0.0, 2.0)  # Cap the max alignment factor

        # Calculate adjustments
        belief_differences = human_vals - sensed_vals
        adjustments = alignment_factors * belief_differences

        # Apply adjustments to AI's sensed values
        corrected = np.round(sensed_vals + adjustments)
        corrected = np.clip(corrected, 0, 5)

        # Build the report dictionary with aligned values
        for i, cell in enumerate(valid_cells_in_query):
            report[cell] = int(corrected[i])

        return report


    def step(self):
        self.sense_environment()
        return 0


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
                 low_trust_amplification_factor=0.3,
                 exploitative_correction_factor=1.0,
                 width=30, height=30,
                 lambda_parameter=0.5,
                 learning_rate=0.15,
                 epsilon=0.3, #q-learning / exploration rate
                 ticks=150,
                 rumor_probability=0.3, # 30%
                 rumor_intensity=1.0,   # Default peak ~L4 if active
                 rumor_confidence=0.6,  # Default moderate confidence
                 rumor_radius_factor=0.5, # Default half radius of real disaster
                 min_rumor_separation_factor=0.5,
                 exploit_trust_lr=0.03, # Default value matching base_params
                 explor_trust_lr=0.05,
                 exploit_friend_bias=0.1, # Default value matching base_params
                 exploit_self_bias=0.1  # Default value matching base_params
                 ):
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
        self.low_trust_amplification_factor = low_trust_amplification_factor

        # Learning rates and biases
        self.exploit_trust_lr = exploit_trust_lr
        self.explor_trust_lr = explor_trust_lr
        self.exploit_friend_bias = exploit_friend_bias
        self.exploit_self_bias = exploit_self_bias

        self.grid = MultiGrid(width, height, torus=False)
        self.tick = 0
        self.tokens_this_tick = {}
        self.unmet_needs_evolution = []
        self.trust_stats = []       # (tick, AI_trust_exploit, Friend_trust_exploit, Nonfriend_trust_exploit,
        self.rumor_probability = rumor_probability
        self.rumor_intensity = rumor_intensity
        self.rumor_confidence = rumor_confidence
        self.rumor_radius_factor = rumor_radius_factor
        self.min_rumor_separation_factor = min_rumor_separation_factor
         #         AI_trust_explor, Friend_trust_explor, Nonfriend_trust_explor)
        self.calls_data = []
        self.rewards_data = []
        self.previous_grid = None
        self.event_ticks = []
        self.event_threshold = 2.0

        self.seci_data = []         # (tick, avg_SECI_exploit, avg_SECI_explor)
        self.aeci_data = []         # (tick, avg_AECI_exploit, avg_AECI_explor)
        self.retain_aeci_data = []  # (tick, avg_retain_AECI_exploit, avg_retain_AECI_explor)
        self.retain_seci_data = []  # (tick, avg_retain_SECI_exploit, avg_retain_SECI_explor)
        self.belief_error_data = [] # (tick, avg_MAE_exploit, avg_MAE_explor)
        self.belief_variance_data = [] # (tick, var_exploit, var_explor)
        self.running_aeci_exp = 0.0  # Running average AECI for exploitative agents
        self.running_aeci_expl = 0.0  # Running average AECI for exploratory agents
        self.running_aeci_data = []  # Store (tick, running_aeci_exp, running_aeci_expl)
        self.component_seci_data = []
        self.aeci_variance_data = []   # AI Echo Chamber Index (Variance-based)
        self.component_aeci_data = []  # Component-level AI Engagement
        self.component_ai_trust_variance_data = []  # AI Trust Clustering

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

        # In DisasterModel.__init__
        self.debug_mode = True  # Set to False for production runs

        # Create human agents.
        for i in range(self.num_humans):
            agent_type = "exploitative" if i < int(self.num_humans * self.share_exploitative) else "exploratory"
            current_trust_lr = self.exploit_trust_lr if agent_type == "exploitative" else self.explor_trust_lr

            agent = HumanAgent(unique_id=f"H_{i}", model=self, id_num=i, agent_type=agent_type,
                             share_confirming=self.share_confirming,
                             learning_rate=self.learning_rate, # Q rate from model
                             epsilon=self.epsilon,           # Epsilon from model
                             # pass trust rates and biases
                             trust_learning_rate=current_trust_lr,
                             exploit_friend_bias=self.exploit_friend_bias, # From model
                             exploit_self_bias=self.exploit_self_bias     # From model
                             )
            self.humans[f"H_{i}"] = agent
            self.agent_list.append(agent)
            pos = (random.randrange(width), random.randrange(height))
            self.grid.place_agent(agent, pos)
            agent.pos = pos

        self.agent_rumors = {} # Store assigned rumor details {agent_id: (epicenter, intensity, confidence)}
        rumor_prob = getattr(self, 'rumor_probability', 0.0)
        rumor_intensity = getattr(self, 'rumor_intensity', 0.0)
        rumor_conf = getattr(self, 'rumor_confidence', 0.6)
        min_sep_dist_sq = (self.disaster_radius * getattr(self, 'min_rumor_separation_factor', 0.7))**2
        rumor_radius = self.disaster_radius * getattr(self, 'rumor_radius_factor', 0.5) # Needed for init

        # Use integer node IDs from network for components
        # Map node ID back to agent ID string 'H_i'
        node_to_agent_id = {i: f"H_{i}" for i in range(self.num_humans)}

        # Find connected components in the social network
        # Note: This assumes node IDs 0..N-1 correspond to agent indices
        components = list(nx.connected_components(self.social_network))
        print(f"Found {len(components)} network components.") # Debug print

        for i, component_nodes in enumerate(components):
            # Decide if this component gets a rumor
            if random.random() < rumor_prob:
                # Generate ONE rumor epicenter for the entire component
                rumor_epicenter = None
                attempts = 0
                max_attempts = 100
                while attempts < max_attempts:
                    potential_rumor_epicenter = (random.randrange(self.width), random.randrange(self.height))
                    dist_sq = (potential_rumor_epicenter[0] - self.epicenter[0])**2 + \
                          (potential_rumor_epicenter[1] - self.epicenter[1])**2
                    if dist_sq >= min_sep_dist_sq:
                        rumor_epicenter = potential_rumor_epicenter
                        # print(f"  Assigning rumor at {rumor_epicenter} to component {i} (size {len(component_nodes)})") # Debug
                        break
                    attempts += 1

                # Assign the SAME rumor details to all agents in this component
                if rumor_epicenter:
                    rumor_details = (rumor_epicenter, rumor_intensity, rumor_conf, rumor_radius)
                    for node_id in component_nodes:
                        agent_id = node_to_agent_id.get(node_id)
                        if agent_id: # Check if agent exists
                            self.agent_rumors[agent_id] = rumor_details

        for agent in self.agent_list:
            if isinstance(agent, HumanAgent):
                agent_rumor = self.agent_rumors.get(agent.unique_id, None)
                agent.initialize_beliefs(assigned_rumor=agent_rumor) # Pass rumor details

       # Create AI agents.
        self.ais = {}
        for k in range(self.num_ai):
            ai_agent = AIAgent(unique_id=f"A_{k}", model=self)
            self.ais[f"A_{k}"] = ai_agent
            self.agent_list.append(ai_agent)
            pos = (random.randrange(width), random.randrange(height))
            self.grid.place_agent(ai_agent, pos)
            ai_agent.pos = pos

        # --- Initialize Trust and Q-tables for INDIVIDUALS ---
        all_human_ids = list(self.humans.keys())
        all_ai_ids = list(self.ais.keys())
        default_q_value = 0.0 # Initial Q-value for unknown sources
        base_human_trust = self.base_trust
        base_ai_trust_val = self.base_ai_trust

        for agent in self.humans.values(): # Iterate through all human agents
            agent.friends = set(f"H_{j}" for j in self.social_network.neighbors(agent.id_num) if f"H_{j}" in self.humans)
            agent.q_table['self_action'] = 0.0 # Q-value for acting on own belief

            # Initialize trust/Q for other humans
            for other_id in all_human_ids:
                if agent.unique_id == other_id: continue # Skip self

                initial_t = random.uniform(base_human_trust - 0.05, base_human_trust + 0.05)
                if other_id in agent.friends:
                    initial_t = min(1.0, initial_t + 0.1) # Friend boost
                agent.trust[other_id] = initial_t
                agent.q_table[other_id] = default_q_value # Initialize Q for this specific human

            # Initialize trust/Q for AI agents
            for ai_id in all_ai_ids:
                initial_ai_t = random.uniform(base_ai_trust_val - 0.1, base_ai_trust_val + 0.1)
                agent.trust[ai_id] = max(0.0, min(1.0, initial_ai_t))
                agent.q_table[ai_id] = default_q_value # Initialize Q for this specific AI


           # Add this method to DisasterModel
    def debug_log(self, message, force=False):
        """Log debug messages if debug mode is enabled or forced."""
        if self.debug_mode or force:
            print(f"[DEBUG] Tick {self.tick}: {message}")

    def track_ai_usage_patterns(model, tick_interval=10, save_dir="analysis_plots"):
        """Track and plot AI usage patterns over time."""
        os.makedirs(save_dir, exist_ok=True)

        if model.tick % tick_interval != 0:
            return  # Only run at specified intervals

        # Gather data on AI usage ratio
        exploit_ai_ratio = []
        explor_ai_ratio = []

        for agent in model.humans.values():
            if agent.accum_calls_total > 0:
                ai_ratio = agent.accum_calls_ai / agent.accum_calls_total
                if agent.agent_type == "exploitative":
                    exploit_ai_ratio.append(ai_ratio)
                else:
                    explor_ai_ratio.append(ai_ratio)

        # Save data for later analysis
        if not hasattr(model, 'ai_usage_history'):
            model.ai_usage_history = {'tick': [], 'exploit_mean': [], 'exploit_std': [],
                                    'explor_mean': [], 'explor_std': []}

        model.ai_usage_history['tick'].append(model.tick)
        model.ai_usage_history['exploit_mean'].append(np.mean(exploit_ai_ratio) if exploit_ai_ratio else 0)
        model.ai_usage_history['exploit_std'].append(np.std(exploit_ai_ratio) if exploit_ai_ratio else 0)
        model.ai_usage_history['explor_mean'].append(np.mean(explor_ai_ratio) if explor_ai_ratio else 0)
        model.ai_usage_history['explor_std'].append(np.std(explor_ai_ratio) if explor_ai_ratio else 0)

        # Plot current state
        if model.tick > 0 and len(model.ai_usage_history['tick']) > 1:
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(model.ai_usage_history['tick'], model.ai_usage_history['exploit_mean'],
                    'r-', label='Exploitative')
            ax.fill_between(model.ai_usage_history['tick'],
                            np.array(model.ai_usage_history['exploit_mean']) - np.array(model.ai_usage_history['exploit_std']),
                            np.array(model.ai_usage_history['exploit_mean']) + np.array(model.ai_usage_history['exploit_std']),
                            color='r', alpha=0.2)

            ax.plot(model.ai_usage_history['tick'], model.ai_usage_history['explor_mean'],
                    'b-', label='Exploratory')
            ax.fill_between(model.ai_usage_history['tick'],
                            np.array(model.ai_usage_history['explor_mean']) - np.array(model.ai_usage_history['explor_std']),
                            np.array(model.ai_usage_history['explor_mean']) + np.array(model.ai_usage_history['explor_std']),
                            color='b', alpha=0.2)

            ax.set_xlabel('Tick')
            ax.set_ylabel('AI Usage Ratio')
            ax.set_title(f'AI Usage Over Time (Alignment = {model.ai_alignment_level})')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"ai_usage_tick_{model.tick}.png"))
            plt.close()

    def update_disaster(self):
        # Store a copy of the current grid before updating
        if self.disaster_grid is not None:
            self.previous_grid = self.disaster_grid.copy()
        else:
            self.previous_grid = None

        # Replace this with your actual update_disaster logic
        # Example placeholder: Create a hotspot with some probability
        if random.random() < 0.1:  # 10% chance each tick
            x, y = np.random.randint(0, self.disaster_grid.shape[0]), np.random.randint(0, self.disaster_grid.shape[1])
            self.disaster_grid[x, y] = min(5, self.disaster_grid[x, y] + 3)

        # Detect significant changes
        if self.previous_grid is not None:
            grid_change = np.abs(self.disaster_grid - self.previous_grid)
            max_change = np.max(grid_change)
            if max_change >= self.event_threshold:
                self.event_ticks.append(self.tick)
                # print(f"Tick {self.tick}: Significant disaster event detected (max change: {max_change})")


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
                #print(f"Agent {agent.unique_id} step returned: {r}")
                if r is None:
                    print(f"ERROR: Agent {agent.unique_id} returned None from process_reward")
                reward = r if r is not None else 0  # Fallback to 0 if None
                if agent.agent_type == "exploitative":
                    total_reward_exploit += reward
                else:
                    total_reward_explor += reward
        self.rewards_data.append((total_reward_exploit, total_reward_explor))

        # Create a proper token array counting all tokens
        token_array = np.zeros((self.width, self.height), dtype=int)

        # Debug totals for verification
        total_exploit_tokens = 0
        total_explor_tokens = 0

        # Fill the token array
        for pos, count_dict in self.tokens_this_tick.items():
            x, y = pos
            # Check bounds before accessing array
            if 0 <= x < self.width and 0 <= y < self.height:
                # Get token counts by type
                exploit_tokens = count_dict.get('exploit', 0)
                explor_tokens = count_dict.get('explor', 0)

                # Track totals for debugging
                total_exploit_tokens += exploit_tokens
                total_explor_tokens += explor_tokens

                # Assign the sum to the token array
                token_array[x, y] = exploit_tokens + explor_tokens
            else:
                print(f"Warning: Invalid position key in tokens_this_tick: {pos}")

        # Log token totals periodically
        if self.tick % 10 == 0 or self.debug_mode:
            print(f"Tick {self.tick} token counts - Exploit: {total_exploit_tokens}, Explor: {total_explor_tokens}, Total: {total_exploit_tokens + total_explor_tokens}")

        # Identify high-need cells (L4+) that received no tokens
        need_mask = self.disaster_grid >= 4
        tokens_mask = token_array == 0
        unmet = np.sum(need_mask & tokens_mask)

        # Log details about the unmet needs
        if self.tick % 10 == 0 or self.debug_mode:
            high_need_count = np.sum(need_mask)
            print(f"Tick {self.tick} high-need cells: {high_need_count}, unmet: {unmet} ({unmet/max(1,high_need_count)*100:.1f}%)")

        self.unmet_needs_evolution.append(unmet)

        if self.tick % 10 == 0:
            self.track_ai_usage_patterns()

                       # Every 5 ticks, compute additional metrics.
        # This goes in the DisasterModel.step method
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
                # FIXED: Properly collect friend beliefs
                friend_belief_levels = []
                for fid in agent.friends:
                    friend = self.humans.get(fid)
                    if friend:
                        for belief_info in friend.beliefs.values():
                            if isinstance(belief_info, dict):
                                friend_belief_levels.append(belief_info.get('level', 0))

                # Calculate friend variance with safety check
                if len(friend_belief_levels) > 1:
                    friend_var = np.var(friend_belief_levels)
                else:
                    friend_var = global_var  # Default to global if insufficient friend data

                # FIXED: Ensure SECI is properly bounded
                seci_val = max(0, min(1, (global_var - friend_var) / global_var))

                if agent.agent_type == "exploitative":
                    seci_exp_list.append(seci_val)
                else:
                    seci_expl_list.append(seci_val)

            self.seci_data.append((
                self.tick,
                np.mean(seci_exp_list) if seci_exp_list else 0,
                np.mean(seci_expl_list) if seci_expl_list else 0
            ))

            # Component SECI calculation
            component_seci_list = []
            for component_nodes in nx.connected_components(self.social_network):
                component_beliefs = []
                for node_id in component_nodes:
                    agent_id = f"H_{node_id}"
                    agent = self.humans.get(agent_id)
                    if agent:
                        for belief_info in agent.beliefs.values():
                            if isinstance(belief_info, dict):
                                component_beliefs.append(belief_info.get('level', 0))
                comp_var = np.var(component_beliefs) if component_beliefs else global_var
                comp_seci = max(0, min(1, (global_var - comp_var) / (global_var + 1e-6)))
                component_seci_list.append(comp_seci)
            self.component_seci_data.append((self.tick, np.mean(component_seci_list) if component_seci_list else 0))

            # --- AECI-Variance (AI Echo Chamber Index) ---
            aeci_variance = 0.0

            # Define AI-reliant agents with a more inclusive threshold
            min_calls_threshold = 3  # Minimum calls to be considered active
            min_ai_ratio = 0.3      # Lowered from 0.5 to be more inclusive

            ai_reliant_agents = [agent for agent in self.humans.values()
                                if agent.accum_calls_total >= min_calls_threshold and
                                    (agent.accum_calls_ai / max(1, agent.accum_calls_total)) >= min_ai_ratio]

            # Log counts for debugging
            if self.debug_mode:
                print(f"Tick {self.tick}: Found {len(ai_reliant_agents)} AI-reliant agents out of {len(self.humans)}")
                num_active = sum(1 for a in self.humans.values() if a.accum_calls_total >= min_calls_threshold)
                print(f"    Active agents: {num_active}, AI threshold: {min_ai_ratio}")

            # Only compute variance if we have AI-reliant agents
            if ai_reliant_agents:
                # Get global variance first
                all_belief_levels = []
                for agent in self.humans.values():
                    for belief_info in agent.beliefs.values():
                        if isinstance(belief_info, dict):
                            all_belief_levels.append(belief_info.get('level', 0))

                global_var = np.var(all_belief_levels) if all_belief_levels else 1e-6

                # Now get AI-reliant agents' belief variance
                ai_reliant_beliefs = []
                for agent in ai_reliant_agents:
                    for belief_info in agent.beliefs.values():
                        if isinstance(belief_info, dict):
                            ai_reliant_beliefs.append(belief_info.get('level', 0))

                if ai_reliant_beliefs and global_var > 1e-9:
                    ai_reliant_var = np.var(ai_reliant_beliefs)
                    aeci_variance = max(0, min(1, (global_var - ai_reliant_var) / global_var))
                elif ai_reliant_beliefs:
                    aeci_variance = 0.0  # If global_var is essentially zero, no reduction is possible

            # Store the value (aeci_variance is guaranteed to be defined now)
            self.aeci_variance_data.append((self.tick, aeci_variance))

            # Additional debug to log individual agent AI usage
            if self.debug_mode and self.tick % 20 == 0:
                print("\nAI Usage Breakdown:")
                for i, agent in enumerate(self.humans.values()):
                    if i < 10:  # Limit to first 10 agents for brevity
                        ratio = agent.accum_calls_ai / max(1, agent.accum_calls_total)
                        print(f"Agent {agent.unique_id} ({agent.agent_type[:4]}): AI calls {agent.accum_calls_ai}/{agent.accum_calls_total} = {ratio:.2f}")

            # --- New Metric: Component-AECI ---
            component_aeci_list = []
            for component_nodes in nx.connected_components(self.social_network):
                component_aeci = []
                for node_id in component_nodes:
                    agent_id = f"H_{node_id}"
                    agent = self.humans.get(agent_id)
                    if agent and agent.accum_calls_total > 0:
                        aeci = agent.accum_calls_ai / agent.accum_calls_total
                        component_aeci.append(aeci)
                avg_component_aeci = np.mean(component_aeci) if component_aeci else 0.0
                component_aeci_list.append(avg_component_aeci)
            self.component_aeci_data.append((self.tick, np.mean(component_aeci_list) if component_aeci_list else 0))

            # --- New Metric: Component AI Trust Variance ---
            component_ai_trust_var_list = []
            for component_nodes in nx.connected_components(self.social_network):
                component_ai_trusts = []
                for node_id in component_nodes:
                    agent_id = f"H_{node_id}"
                    agent = self.humans.get(agent_id)
                    if agent:
                        ai_trust = np.mean([agent.trust[k] for k in agent.trust if k.startswith("A_")]) if any(k.startswith("A_") for k in agent.trust) else 0.0
                        component_ai_trusts.append(ai_trust)
                ai_trust_var = np.var(component_ai_trusts) if component_ai_trusts else 0.0
                component_ai_trust_var_list.append(ai_trust_var)
            self.component_ai_trust_variance_data.append((self.tick, np.mean(component_ai_trust_var_list) if component_ai_trust_var_list else 0))

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
            aeci_exp = []
            aeci_expl = []

            for agent in self.humans.values():
                if agent.accum_calls_total > 0:
                    # Ensure ratio is properly bounded between 0 and 1
                    ratio = max(0.0, min(1.0, agent.accum_calls_ai / agent.accum_calls_total))

                    if agent.agent_type == "exploitative":
                        aeci_exp.append(ratio)
                    else:  # exploratory
                        aeci_expl.append(ratio)

            # Debug logging of raw values
            if self.debug_mode and self.tick % 20 == 0:
                print(f"\nTick {self.tick} AECI (AI Call Ratio) Raw Values:")
                print(f"  Exploitative: {sorted(aeci_exp)[:5]}...{sorted(aeci_exp)[-5:] if len(aeci_exp) > 5 else []}")
                print(f"  Exploratory: {sorted(aeci_expl)[:5]}...{sorted(aeci_expl)[-5:] if len(aeci_expl) > 5 else []}")

            # Calculate means with added safety
            avg_aeci_exp = np.mean(aeci_exp) if aeci_exp else 0.0
            avg_aeci_expl = np.mean(aeci_expl) if aeci_expl else 0.0

            # Ensure averages are also properly bounded
            avg_aeci_exp = max(0.0, min(1.0, avg_aeci_exp))
            avg_aeci_expl = max(0.0, min(1.0, avg_aeci_expl))

            self.aeci_data.append((self.tick, avg_aeci_exp, avg_aeci_expl))

            self.running_aeci_exp = self.running_aeci_exp * 0.8 + avg_aeci_exp * 0.2
            self.running_aeci_expl = self.running_aeci_expl * 0.8 + avg_aeci_expl * 0.2
            self.running_aeci_data.append((self.tick, self.running_aeci_exp, self.running_aeci_expl))

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
                # print(f"Tick {self.tick} Agent {agent.unique_id}: accepted_ai={agent.accepted_ai}, accepted_human={agent.accepted_human}, retain_aeci={retain_aeci_val:.3f}, retain_seci={retain_seci_val:.3f}")

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
                agent.accum_calls_human = 0
                agent.accum_calls_total = 0
        else:
            # Store zeros if we have insufficient global data
            self.seci_data.append((self.tick, 0, 0))
            self.component_seci_data.append((self.tick, 0))
            self.aeci_variance_data.append((self.tick, 0))
            self.component_aeci_data.append((self.tick, 0))
            self.component_ai_trust_variance_data.append((self.tick, 0))
            self.belief_error_data.append((self.tick, 0, 0))
            self.belief_variance_data.append((self.tick, 0, 0))
            self.aeci_data.append((self.tick, 0, 0))
            self.running_aeci_data.append((self.tick, 0, 0))
            self.retain_aeci_data.append((self.tick, 0, 0))
            self.retain_seci_data.append((self.tick, 0, 0))
            self.trust_stats.append((self.tick, 0, 0, 0, 0, 0, 0))

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
            "unmet_needs_evolution": model.unmet_needs_evolution,
            "component_seci": np.array(getattr(model, 'component_seci_data', [])), # Use getattr for safety
            "aeci_variance": np.array(getattr(model, 'aeci_variance_data', [])),
            "component_aeci": np.array(getattr(model, 'component_aeci_data', [])),
            "component_ai_trust_variance": np.array(getattr(model, 'component_ai_trust_variance_data', [])),
            "event_ticks": list(getattr(model, 'event_ticks', [])) # Store as list
        }
            #"per_agent_tokens": model.unmet_needs_evolution,
            #"assistance_exploit": {},  # Placeholder
            #"assistance_explor": {},   # Placeholder
            #"assistance_incorrect_exploit": {},  # Placeholder
            #"assistance_incorrect_explor": {}    # Placeholder

        yield result, model
        del model
        gc.collect()

    component_seci_list, aeci_variance_list, component_aeci_list, component_ai_trust_variance_list = [], [], [], []
    event_ticks_list = [] # Collect event ticks from each run

    # --- Lists for Assistance Counts ---
    exploit_correct_per_run, exploit_incorrect_per_run = [], []
    explor_correct_per_run, explor_incorrect_per_run = [], []

    for result, model in simulation_generator(num_runs, base_params):
        # --- Append Original Metrics ---
        trust_list.append(result.get("trust_stats", np.array([]))) # Use .get with default empty array
        seci_list.append(result.get("seci", np.array([])))
        aeci_list.append(result.get("aeci", np.array([])))
        retain_aeci_list.append(result.get("retain_aeci", np.array([])))
        retain_seci_list.append(result.get("retain_seci", np.array([])))
        unmet_needs_evolution_list.append(result.get("unmet_needs_evolution", []))
        belief_error_list.append(result.get("belief_error", np.array([])))
        belief_variance_list.append(result.get("belief_variance", np.array([])))

        # <<< APPEND New Metrics >>>
        component_seci_list.append(result.get("component_seci", np.array([])))
        aeci_variance_list.append(result.get("aeci_variance", np.array([])))
        component_aeci_list.append(result.get("component_aeci", np.array([])))
        component_ai_trust_variance_list.append(result.get("component_ai_trust_variance", np.array([])))
        event_ticks_list.append(result.get("event_ticks", [])) # Append the list of ticks

        # --- Aggregate Assistance Counts ---
        run_exploit_correct, run_exploit_incorrect = 0, 0
        run_explor_correct, run_explor_incorrect = 0, 0
        for agent in model.humans.values():
            if agent.agent_type == "exploitative":
                run_exploit_correct += agent.correct_targets
                run_exploit_incorrect += agent.incorrect_targets
            else:
                run_explor_correct += agent.correct_targets
                run_explor_incorrect += agent.incorrect_targets
        exploit_correct_per_run.append(run_exploit_correct)
        exploit_incorrect_per_run.append(run_exploit_incorrect)
        explor_correct_per_run.append(run_explor_correct)
        explor_incorrect_per_run.append(run_explor_incorrect)

        del model
        gc.collect()

    # --- Stack Arrays ---
    # Helper function (ensure this is defined globally or locally)
    def safe_stack(data_list):
        """Safely stacks a list of numpy arrays, handling empty lists/arrays and shape inconsistencies."""
        if not data_list:
            return np.array([])

        # Filter out None values and empty arrays
        valid_arrays = [item for item in data_list if isinstance(item, np.ndarray) and item.size > 0]

        if not valid_arrays:
            return np.array([])

        # Check if all arrays have the same shape
        first_shape = valid_arrays[0].shape
        if all(arr.shape == first_shape for arr in valid_arrays):
            return np.stack(valid_arrays, axis=0)

        # Handle case where arrays might have different numbers of time steps
        # Find arrays with same number of dimensions but possibly different lengths
        same_ndim = [arr for arr in valid_arrays if arr.ndim == valid_arrays[0].ndim]

        if not same_ndim:
            return np.array([])

        # For time series data (assumed to be shape (time_steps, metrics))
        # Find the minimum number of time steps across all arrays
        if valid_arrays[0].ndim >= 2:
            min_time_steps = min(arr.shape[1] for arr in same_ndim)
            # Truncate all arrays to this minimum length
            truncated = [arr[:, :min_time_steps, ...] for arr in same_ndim]
            return np.stack(truncated, axis=0)
        else:
            # For 1D arrays, just stack them if they have the same shape
            if all(arr.shape == same_ndim[0].shape for arr in same_ndim):
                return np.stack(same_ndim, axis=0)

        # If we can't stack, return empty array
        print("Warning: Could not stack arrays with inconsistent shapes")
        return np.array([])

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
        },
        "component_seci": component_seci_array,
        "aeci_variance": aeci_variance_array,
        "component_aeci": component_aeci_array,
        "component_ai_trust_variance": component_ai_trust_variance_array,
        "event_ticks_list": event_ticks_list # Pass list of lists for event ticks
    }

def experiment_alignment_tipping_point(base_params, alignment_values=None, num_runs=10):
    """Run a fine-grained sweep of AI alignment values to find tipping points."""
    if alignment_values is None:
        # Create a fine-grained list of alignment values
        alignment_values = [round(x * 0.1, 1) for x in range(11)]  # 0.0 to 1.0 by 0.1

    results = {}
    crossover_detected = False

    for align in alignment_values:
        params = base_params.copy()
        params["ai_alignment_level"] = align
        print(f"Running ai_alignment_level = {align}")

        result = aggregate_simulation_results(num_runs, params)
        results[align] = result

        # Check for tipping point where exploratory agents prefer AI more than exploitative agents
        if align > 0 and align-0.1 in results:
            prev_result = results[align-0.1]

            # Extract AECI values (AI Call Ratio) for each agent type
            curr_aeci_exploit = result.get("aeci", np.array([]))
            curr_aeci_explor = result.get("aeci", np.array([]))
            prev_aeci_exploit = prev_result.get("aeci", np.array([]))
            prev_aeci_explor = prev_result.get("aeci", np.array([]))

            # Check if crossover occurred
            if (prev_aeci_explor.size > 0 and prev_aeci_exploit.size > 0 and
                curr_aeci_explor.size > 0 and curr_aeci_exploit.size > 0):

                prev_explor_mean = np.mean(prev_aeci_explor[:, -1, 2])  # Last tick, explor column
                prev_exploit_mean = np.mean(prev_aeci_exploit[:, -1, 1])  # Last tick, exploit column
                curr_explor_mean = np.mean(curr_aeci_explor[:, -1, 2])
                curr_exploit_mean = np.mean(curr_aeci_exploit[:, -1, 1])

                if ((prev_explor_mean > prev_exploit_mean and curr_explor_mean < curr_exploit_mean) or
                    (prev_explor_mean < prev_exploit_mean and curr_explor_mean > curr_exploit_mean)):
                    crossover_detected = True
                    print(f"TIPPING POINT DETECTED between alignment {align-0.1} and {align}")
                    print(f"Previous: Explor={prev_explor_mean:.3f}, Exploit={prev_exploit_mean:.3f}")
                    print(f"Current: Explor={curr_explor_mean:.3f}, Exploit={curr_exploit_mean:.3f}")

                    # Run a finer-grained search between these values
                    fine_alignment = [round((align-0.1) + x * 0.02, 2) for x in range(1, 5)]  # 5 points between
                    for fine_align in fine_alignment:
                        params = base_params.copy()
                        params["ai_alignment_level"] = fine_align
                        print(f"Running fine-grained ai_alignment_level = {fine_align}")
                        results[fine_align] = aggregate_simulation_results(num_runs, params)

    if not crossover_detected:
        print("No tipping point detected across the alignment range.")

    return results

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


# Helper function
def safe_stack(data_list):
    """Safely stacks a list of numpy arrays, handling empty lists/arrays."""
    if not data_list: return np.array([])
    valid_arrays = [item for item in data_list if isinstance(item, np.ndarray) and item.size > 0]
    if not valid_arrays: return np.array([])
    # Basic stacking, assumes consistent shapes (requires metrics to be recorded for same # of ticks)
    try:
        # Check expected dimensions - most time series here are (ticks, n_metrics_in_tuple) -> stack to (runs, ticks, n_metrics)
        # Find expected shape from first valid array
        expected_ndim = valid_arrays[0].ndim
        expected_shape_after_tick_col = valid_arrays[0].shape[1:] # Shape excluding the tick dimension

        processed_list = []
        for item in valid_arrays:
             # Only include arrays matching expected dimensions
             if item.ndim == expected_ndim and item.shape[1:] == expected_shape_after_tick_col:
                  processed_list.append(item)
             else:
                  print(f"Warning: Skipping array with shape {item.shape} during stacking (expected ndim={expected_ndim}, shape[1:]={expected_shape_after_tick_col})")

        if not processed_list: return np.array([])
        return np.stack(processed_list, axis=0)
    except ValueError as e:
        print(f"Error during stacking: {e}. Returning empty array.")
        # More sophisticated padding could be added here if tick counts vary significantly
        return np.array([])

def calculate_metric_stats(data_list):
    """Helper function for calculating mean/percentiles for assistance counts."""
    valid_data = [d for d in data_list if d is not None and isinstance(d, (int, float, np.number))] # Filter non-numeric
    if not valid_data: return {"mean": 0, "lower": 0, "upper": 0}
    return {
        "mean": float(np.mean(valid_data)), # Ensure float
        "lower": float(np.percentile(valid_data, 25)), # Ensure float
        "upper": float(np.percentile(valid_data, 75))  # Ensure float
    }

# --- Aggregate results ---
def aggregate_simulation_results(num_runs, base_params):
    """
    Runs multiple simulations, aggregates results, and calculates summary statistics.

    Args:
        num_runs (int): Number of simulation runs to perform.
        base_params (dict): Base dictionary of parameters for the model.

    Returns:
        dict: A dictionary containing aggregated results (stacked arrays for time-series,
              summary stats for assistance, raw assistance counts, etc.).
    """
    # --- Initialize Lists for ALL metrics ---
    trust_list, seci_list, aeci_list, retain_aeci_list, retain_seci_list = [], [], [], [], []
    unmet_needs_evolution_list, belief_error_list, belief_variance_list = [], [], []
    # <<< Lists for New Metrics >>>
    component_seci_list, aeci_variance_list, component_aeci_list, component_ai_trust_variance_list = [], [], [], []
    event_ticks_list = [] # Collect event ticks from each run
    max_aeci_variance_per_run = [] #max aeci var

    # --- Lists for Assistance Counts ---
    exploit_correct_per_run, exploit_incorrect_per_run = [], []
    explor_correct_per_run, explor_incorrect_per_run = [], []

    print(f"Starting aggregation for {num_runs} runs...")
    for run_num in range(num_runs):
        print(f"  Starting Run {run_num + 1}/{num_runs}...")
        try:
            # Assuming simulation_generator yields result dict and model object
            result, model = next(simulation_generator(1, base_params)) # Run one sim at a time

            # --- Append All Metrics (using .get for safety) ---
            trust_list.append(result.get("trust_stats", np.array([])))
            seci_list.append(result.get("seci", np.array([])))
            aeci_list.append(result.get("aeci", np.array([]))) # AI Call Ratio
            aeci_variance_data = result.get("aeci_variance", np.array([]))
            retain_aeci_list.append(result.get("retain_aeci", np.array([])))
            retain_seci_list.append(result.get("retain_seci", np.array([])))
            unmet_needs_evolution_list.append(result.get("unmet_needs_evolution", [])) # List
            belief_error_list.append(result.get("belief_error", np.array([])))
            belief_variance_list.append(result.get("belief_variance", np.array([])))
            # <<< APPEND New Metrics >>>
            component_seci_list.append(result.get("component_seci", np.array([])))
            aeci_variance_list.append(result.get("aeci_variance", np.array([])))
            component_aeci_list.append(result.get("component_aeci", np.array([])))
            component_ai_trust_variance_list.append(result.get("component_ai_trust_variance", np.array([])))
            event_ticks_list.append(result.get("event_ticks", [])) # List


            # Find maximum AECI variance value for this run
            if isinstance(aeci_variance_data, np.ndarray) and aeci_variance_data.size > 0:
                if aeci_variance_data.ndim >= 3 and aeci_variance_data.shape[1] > 0 and aeci_variance_data.shape[2] > 1:
                    # Extract values from column 1 (value column)
                    aeci_variance_values = aeci_variance_data[:, :, 1]
                    # Get maximum value across all time steps
                    max_variance = np.max(aeci_variance_values)
                    max_aeci_variance_per_run.append(max_variance)
                else:
                    print(f"Warning: Invalid AECI variance data shape for run {run_num}")
            else:
                print(f"Warning: No AECI variance data for run {run_num}")
            # --- Aggregate Assistance Counts ---
            run_exploit_correct, run_exploit_incorrect = 0, 0
            run_explor_correct, run_explor_incorrect = 0, 0
            if model and hasattr(model, 'humans'): # Check if model and humans exist
                 for agent in model.humans.values():
                     if agent.agent_type == "exploitative":
                         run_exploit_correct += agent.correct_targets
                         run_exploit_incorrect += agent.incorrect_targets
                     else: # Exploratory
                         run_explor_correct += agent.correct_targets
                         run_explor_incorrect += agent.incorrect_targets
            exploit_correct_per_run.append(run_exploit_correct)
            exploit_incorrect_per_run.append(run_exploit_incorrect)
            explor_correct_per_run.append(run_explor_correct)
            explor_incorrect_per_run.append(run_explor_incorrect)

            print(f"  Finished Run {run_num + 1}/{num_runs}")

        except StopIteration:
            print(f"Error: simulation_generator did not yield data for run {run_num + 1}")
            # Optionally append default empty data or handle error differently
            continue # Skip to next run
        except Exception as e:
            print(f"Error during simulation run {run_num + 1}: {e}")
            # Optionally add more details:
            # import traceback
            # traceback.print_exc()
            continue # Skip to next run
        finally:
             # Ensure model object is deleted even if errors occur
             if 'model' in locals() and model is not None:
                 del model
             gc.collect()

    print("Finished all runs. Aggregating results...")

    # --- Stack Arrays ---
    trust_array = safe_stack(trust_list)
    seci_array = safe_stack(seci_list)
    aeci_array = safe_stack(aeci_list)
    retain_aeci_array = safe_stack(retain_aeci_list)
    retain_seci_array = safe_stack(retain_seci_list)
    belief_error_array = safe_stack(belief_error_list)
    belief_variance_array = safe_stack(belief_variance_list)
    # <<< STACK New Data >>>
    component_seci_array = safe_stack(component_seci_list)
    aeci_variance_array = safe_stack(aeci_variance_list)
    component_aeci_array = safe_stack(component_aeci_list)
    component_ai_trust_variance_array = safe_stack(component_ai_trust_variance_list)

    # --- Calculate Assistance Stats ---
    assist_stats = {
        "exploit_correct": calculate_metric_stats(exploit_correct_per_run),
        "exploit_incorrect": calculate_metric_stats(exploit_incorrect_per_run),
        "explor_correct": calculate_metric_stats(explor_correct_per_run),
        "explor_incorrect": calculate_metric_stats(explor_incorrect_per_run)
    }
    # --- Calculate Ratio Stats ---
    total_exploit_mean = assist_stats["exploit_correct"]["mean"] + assist_stats["exploit_incorrect"]["mean"]
    total_explor_mean = assist_stats["explor_correct"]["mean"] + assist_stats["explor_incorrect"]["mean"]
    ratio_stats = {
        "exploit_ratio": {
            "mean": assist_stats["exploit_correct"]["mean"] / total_exploit_mean if total_exploit_mean > 0 else 0,
            "lower": 0, "upper": 0 # Percentiles complex for ratios
        },
        "explor_ratio": {
            "mean": assist_stats["explor_correct"]["mean"] / total_explor_mean if total_explor_mean > 0 else 0,
            "lower": 0, "upper": 0
        }
    }

    print("Aggregation complete.")

    # --- Return Dictionary ---
    return {
        # Original metrics
        "trust_stats": trust_array,
        "seci": seci_array,
        "aeci": aeci_array, # AI Call Ratio
        "retain_aeci": retain_aeci_array,
        "retain_seci": retain_seci_array,
        "belief_error": belief_error_array,
        "belief_variance": belief_variance_array,
        "unmet_needs_evol": unmet_needs_evolution_list, # Keep raw list of lists for plotting
        # Assistance metrics
        "assist": assist_stats,
        "assist_ratio": ratio_stats,
        "raw_assist_counts": {
            "exploit_correct": exploit_correct_per_run,
            "exploit_incorrect": exploit_incorrect_per_run,
            "explor_correct": explor_correct_per_run,
            "explor_incorrect": explor_incorrect_per_run
        },
        # <<<  Aggregated Arrays >>>
        "component_seci": component_seci_array,
        "aeci_variance": aeci_variance_array,
        "max_aeci_variance": max_aeci_variance_per_run,
        "component_aeci": component_aeci_array,
        "component_ai_trust_variance": component_ai_trust_variance_array,
        "event_ticks_list": event_ticks_list # Pass list of lists for event ticks
    }
#########################################
# Plotting Functions
#########################################

# Make sure this helper function is defined correctly in the global scope
def _plot_mean_iqr(ax, ticks, data_array, data_index, label, color, linestyle='-'):
    """Helper to plot mean and IQR band."""
    # --- Robust Check ---
    if data_array is None or data_array.size == 0 or data_array.ndim < 2 or data_array.shape[1] == 0 or data_index >= data_array.shape[2]:
        print(f"Warning: Skipping plot for {label} due to invalid data shape {data_array.shape if data_array is not None else 'None'} or index {data_index}.")
        return
    T = data_array.shape[1] # Get number of ticks from data
    # Ensure ticks array matches data length
    if ticks is None or len(ticks) != T:
        print(f"Warning: Tick length mismatch for {label} ({len(ticks) if ticks is not None else 'None'} vs {T}). Using default range.")
        ticks = np.arange(T) # Default ticks if mismatch

    # --- Calculate Stats ---
    try:
        mean = np.mean(data_array[:, :, data_index], axis=0)
        lower = np.percentile(data_array[:, :, data_index], 25, axis=0)
        upper = np.percentile(data_array[:, :, data_index], 75, axis=0)
    except IndexError: # Handle cases where percentile calculation might fail on edge cases
         print(f"Warning: Could not calculate percentiles for {label}. Plotting mean only.")
         mean = np.mean(data_array[:, :, data_index], axis=0)
         lower = mean
         upper = mean
    except Exception as e:
         print(f"Error calculating stats for {label}: {e}")
         return # Skip plotting this line

    # --- Plot Mean and IQR Band ---
    ax.plot(ticks, mean, label=label, color=color, linestyle=linestyle)
    ax.fill_between(ticks, lower, upper, color=color, alpha=0.4) # Draws the shaded band

def plot_grid_state(model, tick, save_dir="grid_plots"):
    """Plots the disaster grid state, agent locations, and tokens sent."""
    os.makedirs(save_dir, exist_ok=True) # Create directory if it doesn't exist
    # ---2x3 subplots ---
    fig, ax = plt.subplots(2, 3, figsize=(21, 12), constrained_layout=True)# Increased width for 3 columns
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
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    filepath = os.path.join(save_dir, f"grid_state_tick_{tick:04d}.png")
    plt.savefig(filepath)
    plt.close(fig) # Close figure to free memory

def plot_max_aeci_variance_by_alignment(results_dict, alignment_values, title_suffix=""):
    """Plots the maximum AECI variance observed in each run for different alignment levels."""
    # Initialize lists to store mean and IQR of max values for each alignment level
    max_var_means = []
    max_var_errors = [[],[]]  # For lower and upper error bars

    # Process each alignment level
    for align in alignment_values:
        result = results_dict.get(align, {})
        max_values = result.get("max_aeci_variance", [])

        if max_values:
            # Calculate statistics for this alignment level
            mean_max = np.mean(max_values)
            p25 = np.percentile(max_values, 25)
            p75 = np.percentile(max_values, 75)

            max_var_means.append(mean_max)
            max_var_errors[0].append(max(0, mean_max - p25))  # Lower error
            max_var_errors[1].append(max(0, p75 - mean_max))  # Upper error
        else:
            # If no data, append zeros
            max_var_means.append(0)
            max_var_errors[0].append(0)
            max_var_errors[1].append(0)

    # Create plot
    plt.figure(figsize=(10, 6))
    x = np.arange(len(alignment_values))

    plt.bar(x, max_var_means, width=0.6, yerr=max_var_errors, capsize=5,
            color='magenta', alpha=0.7, label='Max AECI Variance')

    # Add value labels on top of bars
    for i, v in enumerate(max_var_means):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

    plt.xlabel('AI Alignment Level')
    plt.ylabel('Maximum AECI Variance Reduction')
    plt.title(f'Maximum AI Belief Variance Reduction by Alignment Level {title_suffix}')
    plt.xticks(x, alignment_values)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()

    # Save and show the plot
    save_path = f"agent_model_results/max_aeci_variance_{title_suffix.replace('(','').replace(')','').replace('=','_')}.png"
    plt.savefig(save_path)
    plt.tight_layout()
    plt.show()

#########################################
# Echo chamber plotting functions
def plot_final_echo_indices_vs_alignment(results_b, alignment_values, title_suffix="Exp B"):
    """Plots final mean values of echo chamber indices vs AI alignment."""
    num_params = len(alignment_values)
    bar_width = 0.15

    index = np.arange(num_params)

    # Initialize data structures
    final_seci_exploit_mean, final_seci_exploit_err = [], [[],[]]
    final_seci_explor_mean, final_seci_explor_err = [], [[],[]]
    final_aeci_var_mean, final_aeci_var_err = [], [[],[]] # AI Echo Chamber
    final_aeci_call_exploit_mean, final_aeci_call_exploit_err = [], [[],[]] # AI Call ratio
    final_aeci_call_explor_mean, final_aeci_call_explor_err = [], [[],[]]
    final_comp_ai_trust_var_mean, final_comp_ai_trust_var_err = [], [[],[]]

    for align in alignment_values:
        res = results_b.get(align)
        if not res: continue

        # Helper to extract final mean and IQR error across runs
        def get_final_stats(data_array, col_index):
            if data_array is None or not isinstance(data_array, np.ndarray) or data_array.size == 0:
                return 0, [0, 0]

            if data_array.ndim < 3 or data_array.shape[1] == 0 or col_index >= data_array.shape[2]:
                return 0, [0, 0]

            # Get final values for all runs
            final_vals = data_array[:, -1, col_index]

            # Handle ratio metrics - clip to [0,1]
            if any(x in str(col_index) for x in ['aeci', 'seci']) or col_index in [1, 2]:
                final_vals = np.clip(final_vals, 0.0, 1.0)

            # Calculate mean and percentiles
            mean = np.mean(final_vals)
            p25 = np.percentile(final_vals, 25)
            p75 = np.percentile(final_vals, 75)

            # Return mean and error lengths [lower, upper]
            return mean, [max(0, mean - p25), max(0, p75 - mean)]

        # Extract final values with percentiles
        mean, err = get_final_stats(res.get("seci"), 1)
        final_seci_exploit_mean.append(mean)
        final_seci_exploit_err[0].append(err[0])
        final_seci_exploit_err[1].append(err[1])

        mean, err = get_final_stats(res.get("seci"), 2)
        final_seci_explor_mean.append(mean)
        final_seci_explor_err[0].append(err[0])
        final_seci_explor_err[1].append(err[1])

        mean, err = get_final_stats(res.get("aeci_variance"), 1)
        final_aeci_var_mean.append(mean)
        final_aeci_var_err[0].append(err[0])
        final_aeci_var_err[1].append(err[1])

        mean, err = get_final_stats(res.get("aeci"), 1)
        final_aeci_call_exploit_mean.append(mean)
        final_aeci_call_exploit_err[0].append(err[0])
        final_aeci_call_exploit_err[1].append(err[1])

        mean, err = get_final_stats(res.get("aeci"), 2)
        final_aeci_call_explor_mean.append(mean)
        final_aeci_call_explor_err[0].append(err[0])
        final_aeci_call_explor_err[1].append(err[1])

        mean, err = get_final_stats(res.get("component_ai_trust_variance"), 1)
        final_comp_ai_trust_var_mean.append(mean)
        final_comp_ai_trust_var_err[0].append(err[0])
        final_comp_ai_trust_var_err[1].append(err[1])

    # Setup plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Final Echo Chamber Indices vs AI Alignment ({title_suffix})", fontsize=16)

    # Plot Final SECI with percentile error bars
    axes[0, 0].bar(index - bar_width/2, final_seci_exploit_mean, bar_width,
                   yerr=final_seci_exploit_err, capsize=4, label='Exploit', color='maroon')
    axes[0, 0].bar(index + bar_width/2, final_seci_explor_mean, bar_width,
                   yerr=final_seci_explor_err, capsize=4, label='Explor', color='salmon')
    axes[0, 0].set_ylabel("Final Mean SECI")
    axes[0, 0].set_title("Social Echo Chamber (SECI)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, axis='y', linestyle='--', alpha=0.6)
    axes[0, 0].set_ylim(0, 1)  # Set bounds for SECI

    # Plot Final AI Echo Chamber (AECI-Variance) with percentile error bars
    axes[0, 1].bar(index, final_aeci_var_mean, bar_width*1.5,
                   yerr=final_aeci_var_err, capsize=4, label='AI Reliant Group', color='magenta')
    axes[0, 1].set_ylabel("Final Mean AI Bel. Var. Reduction")
    axes[0, 1].set_title("AI Echo Chamber (AECI-Var)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, axis='y', linestyle='--', alpha=0.6)
    axes[0, 1].set_ylim(0, 1)  # Set bounds for variance reduction

    # Plot Final AI Call Ratio with percentile error bars
    axes[1, 0].bar(index - bar_width/2, final_aeci_call_exploit_mean, bar_width,
                   yerr=final_aeci_call_exploit_err, capsize=4, label='Exploit', color='darkblue')
    axes[1, 0].bar(index + bar_width/2, final_aeci_call_explor_mean, bar_width,
                   yerr=final_aeci_call_explor_err, capsize=4, label='Explor', color='skyblue')
    axes[1, 0].set_ylabel("Final Mean AI Call Ratio")
    axes[1, 0].set_title("AI Usage (AECI Call Ratio)")
    axes[1, 0].set_xlabel("AI Alignment Level")
    axes[1, 0].legend()
    axes[1, 0].grid(True, axis='y', linestyle='--', alpha=0.6)
    # Ensure y-limits are appropriate - AI call ratio should be between 0 and 1
    axes[1, 0].set_ylim(0, 1)

    # Plot Component AI Trust Variance with percentile error bars
    axes[1, 1].bar(index, final_comp_ai_trust_var_mean, bar_width*1.5,
                   yerr=final_comp_ai_trust_var_err, capsize=4, label='Comp. AI Trust Var.', color='cyan')
    axes[1, 1].set_ylabel("Final Mean Variance")
    axes[1, 1].set_title("Component AI Trust Variance")
    axes[1, 1].set_xlabel("AI Alignment Level")
    axes[1, 1].legend()
    axes[1, 1].grid(True, axis='y', linestyle='--', alpha=0.6)
    axes[1, 1].set_ylim(bottom=0)  # Variance should be non-negative

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks(index)
            ax.set_xticklabels(alignment_values)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = f"agent_model_results/summary_echo_{title_suffix}.png"
    plt.savefig(save_path.replace('(','').replace(')','').replace('=','_'))
    plt.close(fig)

def plot_final_performance_vs_alignment(results_b, alignment_values, title_suffix="Exp B"):
    """Plots final mean performance metrics vs AI alignment with percentile bands."""
    num_params = len(alignment_values)
    bar_width = 0.2
    index = np.arange(num_params)

    # Empty lists to store metrics
    final_mae_exploit_mean, final_mae_exploit_err = [], [[],[]]
    final_mae_explor_mean, final_mae_explor_err = [], [[],[]]
    final_unmet_mean, final_unmet_err = [], [[],[]]
    final_correct_ratio_exploit_mean, final_correct_ratio_exploit_err = [], [[],[]]
    final_correct_ratio_explor_mean, final_correct_ratio_explor_err = [], [[],[]]

    for align in alignment_values:
        res = results_b.get(align)
        if not res: continue

        # Helper to extract final stats with percentiles
        def get_final_stats(data_array, col_index):
            if data_array is None or not isinstance(data_array, np.ndarray) or data_array.size == 0:
                return 0, [0, 0]

            if data_array.ndim < 3 or data_array.shape[1] == 0 or col_index >= data_array.shape[2]:
                return 0, [0, 0]

            # Get the final values for all runs
            final_vals = data_array[:, -1, col_index]

            # Calculate mean and percentiles
            mean = np.mean(final_vals)
            p25 = np.percentile(final_vals, 25)
            p75 = np.percentile(final_vals, 75)

            # Return mean and error lengths [lower, upper]
            return mean, [max(0, mean - p25), max(0, p75 - mean)]

        # Helper for unmet needs (list of lists)
        def get_final_unmet_stats(unmet_list):
            if not unmet_list:
                return 0, [0, 0]

            # Extract final values from each run
            final_vals = []
            for run_data in unmet_list:
                if run_data and len(run_data) > 0:
                    final_vals.append(run_data[-1])

            if not final_vals:
                return 0, [0, 0]

            # Calculate mean and percentiles
            mean = np.mean(final_vals)
            p25 = np.percentile(final_vals, 25)
            p75 = np.percentile(final_vals, 75)

            # Return mean and error lengths [lower, upper]
            return mean, [max(0, mean - p25), max(0, p75 - mean)]

        # Helper for correct token ratios with percentiles
        def get_ratio_stats(raw_counts, correct_key, incorrect_key):
            correct_list = raw_counts.get(correct_key, [])
            incorrect_list = raw_counts.get(incorrect_key, [])

            # Skip if either list is empty
            if not correct_list or not incorrect_list:
                return 0, [0, 0]

            # Calculate ratio for each run
            ratios = []
            for i in range(min(len(correct_list), len(incorrect_list))):
                total = correct_list[i] + incorrect_list[i]
                if total > 0:
                    ratios.append(correct_list[i] / total)
                else:
                    ratios.append(0)

            if not ratios:
                return 0, [0, 0]

            # Ensure ratios are in [0,1]
            ratios = np.clip(ratios, 0.0, 1.0)

            # Calculate mean and percentiles
            mean = np.mean(ratios)
            p25 = np.percentile(ratios, 25)
            p75 = np.percentile(ratios, 75)

            # Return mean and error lengths [lower, upper]
            return mean, [max(0, mean - p25), max(0, p75 - mean)]

        # Extract belief error stats
        mean, err = get_final_stats(res.get("belief_error"), 1)
        final_mae_exploit_mean.append(mean)
        final_mae_exploit_err[0].append(err[0])
        final_mae_exploit_err[1].append(err[1])

        mean, err = get_final_stats(res.get("belief_error"), 2)
        final_mae_explor_mean.append(mean)
        final_mae_explor_err[0].append(err[0])
        final_mae_explor_err[1].append(err[1])

        # Extract unmet needs stats
        mean, err = get_final_unmet_stats(res.get("unmet_needs_evol"))
        final_unmet_mean.append(mean)
        final_unmet_err[0].append(err[0])
        final_unmet_err[1].append(err[1])

        # Extract correct token ratios with percentiles
        raw_counts = res.get("raw_assist_counts", {})
        mean, err = get_ratio_stats(raw_counts, "exploit_correct", "exploit_incorrect")
        final_correct_ratio_exploit_mean.append(mean)
        final_correct_ratio_exploit_err[0].append(err[0])
        final_correct_ratio_exploit_err[1].append(err[1])

        mean, err = get_ratio_stats(raw_counts, "explor_correct", "explor_incorrect")
        final_correct_ratio_explor_mean.append(mean)
        final_correct_ratio_explor_err[0].append(err[0])
        final_correct_ratio_explor_err[1].append(err[1])

    # Setup plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    fig.suptitle(f"Final Performance vs AI Alignment ({title_suffix})", fontsize=16)

    # Plot Final MAE with percentile error bars
    axes[0].bar(index - bar_width/2, final_mae_exploit_mean, bar_width,
                yerr=final_mae_exploit_err, capsize=4, label='Exploit', color='red')
    axes[0].bar(index + bar_width/2, final_mae_explor_mean, bar_width,
                yerr=final_mae_explor_err, capsize=4, label='Explor', color='blue')
    axes[0].set_ylabel("Final Mean MAE")
    axes[0].set_title("Belief Error (MAE)")
    axes[0].legend()
    axes[0].grid(True, axis='y', linestyle='--', alpha=0.6)
    axes[0].set_ylim(bottom=0)  # MAE cannot be negative

    # Plot Final Unmet Needs with percentile error bars
    axes[1].bar(index, final_unmet_mean, bar_width*1.5,
                yerr=final_unmet_err, capsize=4, label='Overall', color='purple')
    axes[1].set_ylabel("Final Mean Unmet Needs")
    axes[1].set_title("Unmet Needs")
    axes[1].set_xlabel("AI Alignment Level")
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.6)
    axes[1].set_ylim(bottom=0)  # Unmet needs cannot be negative

    # Plot Final Correct Token Share with percentile error bars
    axes[2].bar(index - bar_width/2, final_correct_ratio_exploit_mean, bar_width,
                yerr=final_correct_ratio_exploit_err, capsize=4, label='Exploit', color='red')
    axes[2].bar(index + bar_width/2, final_correct_ratio_explor_mean, bar_width,
                yerr=final_correct_ratio_explor_err, capsize=4, label='Explor', color='blue')
    axes[2].set_ylabel("Final Mean Correct Token Share")
    axes[2].set_title("Assistance Quality (Correct Ratio)")
    axes[2].legend()
    axes[2].grid(True, axis='y', linestyle='--', alpha=0.6)
    axes[2].set_ylim(0, 1)  # Ratio must be between 0 and 1

    for ax in axes:
        ax.set_xticks(index)
        ax.set_xticklabels(alignment_values)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = f"agent_model_results/summary_perf_{title_suffix}.png"
    plt.savefig(save_path.replace('(','').replace(')','').replace('=','_'))
    plt.close(fig)

def plot_final_echo_indices_vs_alignment(results_b, alignment_values, title_suffix="Exp B"):
    """Plots final mean values of echo chamber indices vs AI alignment with percentile bands."""
    num_params = len(alignment_values)
    bar_width = 0.15
    index = np.arange(num_params)

    # Initialize data structures
    final_seci_exploit_mean, final_seci_exploit_err = [], [[],[]]
    final_seci_explor_mean, final_seci_explor_err = [], [[],[]]
    final_aeci_var_mean, final_aeci_var_err = [], [[],[]] # AI Echo Chamber
    final_aeci_call_exploit_mean, final_aeci_call_exploit_err = [], [[],[]] # AI Call ratio
    final_aeci_call_explor_mean, final_aeci_call_explor_err = [], [[],[]]
    final_comp_ai_trust_var_mean, final_comp_ai_trust_var_err = [], [[],[]]

    for align in alignment_values:
        res = results_b.get(align)
        if not res: continue

        # Helper to extract final mean and IQR error across runs
        def get_final_stats(data_array, col_index):
            if data_array is None or not isinstance(data_array, np.ndarray) or data_array.size == 0:
                return 0, [0, 0]

            if data_array.ndim < 3 or data_array.shape[1] == 0 or col_index >= data_array.shape[2]:
                return 0, [0, 0]

            # Get final values for all runs
            final_vals = data_array[:, -1, col_index]

            # Calculate mean and percentiles
            mean = np.mean(final_vals)
            p25 = np.percentile(final_vals, 25)
            p75 = np.percentile(final_vals, 75)

            # Return mean and error lengths [lower, upper]
            return mean, [max(0, mean - p25), max(0, p75 - mean)]

        # Extract final values with percentiles
        mean, err = get_final_stats(res.get("seci"), 1)
        final_seci_exploit_mean.append(mean)
        final_seci_exploit_err[0].append(err[0])
        final_seci_exploit_err[1].append(err[1])

        mean, err = get_final_stats(res.get("seci"), 2)
        final_seci_explor_mean.append(mean)
        final_seci_explor_err[0].append(err[0])
        final_seci_explor_err[1].append(err[1])

        mean, err = get_final_stats(res.get("aeci_variance"), 1)
        final_aeci_var_mean.append(mean)
        final_aeci_var_err[0].append(err[0])
        final_aeci_var_err[1].append(err[1])

        mean, err = get_final_stats(res.get("aeci"), 1)
        final_aeci_call_exploit_mean.append(mean)
        final_aeci_call_exploit_err[0].append(err[0])
        final_aeci_call_exploit_err[1].append(err[1])

        mean, err = get_final_stats(res.get("aeci"), 2)
        final_aeci_call_explor_mean.append(mean)
        final_aeci_call_explor_err[0].append(err[0])
        final_aeci_call_explor_err[1].append(err[1])

        mean, err = get_final_stats(res.get("component_ai_trust_variance"), 1)
        final_comp_ai_trust_var_mean.append(mean)
        final_comp_ai_trust_var_err[0].append(err[0])
        final_comp_ai_trust_var_err[1].append(err[1])

    # Setup plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Final Echo Chamber Indices vs AI Alignment ({title_suffix})", fontsize=16)

    # Plot Final SECI with percentile error bars
    axes[0, 0].bar(index - bar_width/2, final_seci_exploit_mean, bar_width,
                   yerr=final_seci_exploit_err, capsize=4, label='Exploit', color='maroon')
    axes[0, 0].bar(index + bar_width/2, final_seci_explor_mean, bar_width,
                   yerr=final_seci_explor_err, capsize=4, label='Explor', color='salmon')
    axes[0, 0].set_ylabel("Final Mean SECI")
    axes[0, 0].set_title("Social Echo Chamber (SECI)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, axis='y', linestyle='--', alpha=0.6)

    # Plot Final AI Echo Chamber (AECI-Variance) with percentile error bars
    axes[0, 1].bar(index, final_aeci_var_mean, bar_width*1.5,
                   yerr=final_aeci_var_err, capsize=4, label='AI Reliant Group', color='magenta')
    axes[0, 1].set_ylabel("Final Mean AI Bel. Var. Reduction")
    axes[0, 1].set_title("AI Echo Chamber (AECI-Var)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, axis='y', linestyle='--', alpha=0.6)

    # Plot Final AI Call Ratio with percentile error bars
    axes[1, 0].bar(index - bar_width/2, final_aeci_call_exploit_mean, bar_width,
                   yerr=final_aeci_call_exploit_err, capsize=4, label='Exploit', color='darkblue')
    axes[1, 0].bar(index + bar_width/2, final_aeci_call_explor_mean, bar_width,
                   yerr=final_aeci_call_explor_err, capsize=4, label='Explor', color='skyblue')
    axes[1, 0].set_ylabel("Final Mean AI Call Ratio")
    axes[1, 0].set_title("AI Usage (AECI Call Ratio)")
    axes[1, 0].set_xlabel("AI Alignment Level")
    axes[1, 0].legend()
    axes[1, 0].grid(True, axis='y', linestyle='--', alpha=0.6)
    # Ensure y-limits are appropriate - AI call ratio should be between 0 and 1
    axes[1, 0].set_ylim(0, 1)


    # Plot Component AI Trust Variance with percentile error bars
    axes[1, 1].bar(index, final_comp_ai_trust_var_mean, bar_width*1.5,
                   yerr=final_comp_ai_trust_var_err, capsize=4, label='Comp. AI Trust Var.', color='cyan')
    axes[1, 1].set_ylabel("Final Mean Variance")
    axes[1, 1].set_title("Component AI Trust Variance")
    axes[1, 1].set_xlabel("AI Alignment Level")
    axes[1, 1].legend()
    axes[1, 1].grid(True, axis='y', linestyle='--', alpha=0.6)
    axes[1, 1].set_ylim(bottom=0)  # Variance should be non-negative

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks(index)
            ax.set_xticklabels(alignment_values)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = f"agent_model_results/summary_echo_{title_suffix}.png"
    plt.savefig(save_path.replace('(','').replace(')','').replace('=','_'))
    plt.close(fig)

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

    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
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
    if trust_stats_array is None or not isinstance(trust_stats_array, np.ndarray) or trust_stats_array.ndim < 3 or trust_stats_array.shape[0] == 0:
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

        # Extract data slice and handle infinities/NaNs
        data_slice = trust_stats_array[:, :, data_index]
        data_slice = np.where(np.isinf(data_slice), np.nan, data_slice)

        # Trust values must be in [0,1]
        data_slice = np.clip(data_slice, 0.0, 1.0)

        # Calculate statistics
        mean = np.nanmean(data_slice, axis=0)
        lower = np.nanpercentile(data_slice, 25, axis=0)
        upper = np.nanpercentile(data_slice, 75, axis=0)

        # Replace NaNs and ensure bounds
        mean = np.clip(np.nan_to_num(mean), 0.0, 1.0)
        lower = np.clip(np.nan_to_num(lower), 0.0, 1.0)
        upper = np.clip(np.nan_to_num(upper), 0.0, 1.0)

        # Ensure percentiles don't cross mean
        lower = np.minimum(mean, lower)
        upper = np.maximum(mean, upper)

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
    axes[0].set_ylim(0, 1)  # Trust must be in [0,1]

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
    axes[1].set_ylim(0, 1)  # Trust must be in [0,1]

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def _plot_mean_iqr(ax, ticks, data_array, data_index, label, color, linestyle='-'):
    """Helper to plot mean and IQR band."""
    # --- Robust Check ---
    if data_array is None or data_array.size == 0 or data_array.ndim < 2 or data_index >= data_array.shape[2] or data_array.shape[1] == 0:
        print(f"Warning: Skipping plot for {label} due to invalid data shape {data_array.shape if data_array is not None else 'None'} or index {data_index}.")
        return
    T = data_array.shape[1] # Get number of ticks from data
    # Ensure ticks array matches data length
    if ticks is None or len(ticks) != T:
        print(f"Warning: Tick length mismatch for {label} ({len(ticks) if ticks is not None else 'None'} vs {T}). Using default range.")
        ticks = np.arange(T) # Default ticks if mismatch

    # --- Calculate Stats ---
    try:
        # Handle potential NaNs and infinities
        data_slice = data_array[:, :, data_index]
        data_slice = np.where(np.isinf(data_slice), np.nan, data_slice)

        mean = np.nanmean(data_slice, axis=0)
        lower = np.nanpercentile(data_slice, 25, axis=0)
        upper = np.nanpercentile(data_slice, 75, axis=0)

        # Replace NaNs with zeros
        mean = np.nan_to_num(mean)
        lower = np.nan_to_num(lower)
        upper = np.nan_to_num(upper)

        # Ensure percentiles don't cross mean
        lower = np.minimum(mean, lower)
        upper = np.maximum(mean, upper)

        # Special handling for ratios and indices (must be in [0,1])
        if 'ratio' in label.lower() or any(x in label.lower() for x in ['aeci', 'seci', 'trust']):
            mean = np.clip(mean, 0.0, 1.0)
            lower = np.clip(lower, 0.0, 1.0)
            upper = np.clip(upper, 0.0, 1.0)
    except IndexError: # Handle cases where percentile calculation might fail on edge cases
        print(f"Warning: Could not calculate percentiles for {label}. Plotting mean only.")
        mean = np.nanmean(data_array[:, :, data_index], axis=0)
        mean = np.nan_to_num(mean)
        lower = mean
        upper = mean
    except Exception as e:
        print(f"Error calculating stats for {label}: {e}")
        return # Skip plotting this line

    # --- Plot Mean and IQR Band ---
    ax.plot(ticks, mean, label=label, color=color, linestyle=linestyle)
    ax.fill_between(ticks, lower, upper, color=color, alpha=0.4) # Draws the shaded band

# ---  CONSOLIDATED PLOT 1: SIMULATION INDICES  ---
def plot_simulation_overview(results_dict, title_suffix=""):
    """Plots key performance and belief metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Simulation Overview {title_suffix}", fontsize=16)

    # --- Data Extraction ---
    belief_error = results_dict.get("belief_error")
    belief_variance = results_dict.get("belief_variance")
    unmet_needs_list = results_dict.get("unmet_needs_evol") # List of lists/arrays
    assist_stats = results_dict.get("assist", {})
    raw_counts = results_dict.get("raw_assist_counts", {})

    # Determine Ticks (use a reliable source like belief_error if possible)
    ticks = np.array([])
    if belief_error is not None and belief_error.ndim >=3 and belief_error.shape[1]>0:
        ticks = belief_error[0, :, 0] # Assumes tick is column 0
        if len(ticks) != belief_error.shape[1]: # Check if ticks are actually stored
             ticks = np.arange(belief_error.shape[1])
    if ticks.size == 0: # Fallback
         if unmet_needs_list and unmet_needs_list[0]: ticks = np.arange(len(unmet_needs_list[0]))
         else: print("Warning: Cannot determine ticks for overview plot."); return

    # --- Subplot 1: Belief Error (MAE) ---
    ax = axes[0, 0]
    _plot_mean_iqr(ax, ticks, belief_error, 1, "MAE Exploit", "red")
    _plot_mean_iqr(ax, ticks, belief_error, 2, "MAE Explor", "blue")
    ax.set_title("Avg. Belief MAE")
    ax.set_ylabel("MAE")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize='small')
    ax.set_ylim(bottom=0)

    # --- Subplot 2: Belief Variance ---
    ax = axes[0, 1]
    _plot_mean_iqr(ax, ticks, belief_variance, 1, "Var Exploit", "red")
    _plot_mean_iqr(ax, ticks, belief_variance, 2, "Var Explor", "blue")
    ax.set_title("Within-Type Belief Variance")
    ax.set_ylabel("Variance")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize='small')
    ax.set_ylim(bottom=0)

    # --- Subplot 3: Unmet Needs ---
    ax = axes[1, 0]
    if unmet_needs_list:
         try: # Handle potential length mismatches and NaNs
             T_needs = max(len(run_data) for run_data in unmet_needs_list if run_data is not None and len(run_data)>0)
             if T_needs > 0:
                 unmet_array = np.full((len(unmet_needs_list), T_needs), np.nan)
                 for i, run_data in enumerate(unmet_needs_list):
                     if run_data is not None and len(run_data) > 0: unmet_array[i, :len(run_data)] = run_data

                 mean = np.nanmean(unmet_array, axis=0)
                 lower = np.nanpercentile(unmet_array, 25, axis=0)
                 upper = np.nanpercentile(unmet_array, 75, axis=0)
                 plot_ticks_needs = np.arange(T_needs) # Use length determined from this data
                 ax.plot(plot_ticks_needs, mean, label="Unmet Need", color="purple")
                 ax.fill_between(plot_ticks_needs, lower, upper, color="purple", alpha=0.2)
             else: ax.text(0.5, 0.5, 'No unmet needs data', ha='center', va='center')
         except Exception as e: print(f"Warning: Could not plot unmet needs: {e}")
    ax.set_title("Unmet Need Count")
    ax.set_ylabel("Count")
    ax.set_xlabel("Tick")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize='small')
    ax.set_ylim(bottom=0)

    # --- Subplot 4: Assistance Quality (Bar plot of final totals) ---
    ax = axes[1, 1]
    if assist_stats and raw_counts:
        labels = ['Exploitative', 'Exploratory']
        mean_correct = [assist_stats.get("exploit_correct", {}).get("mean", 0), assist_stats.get("explor_correct", {}).get("mean", 0)]
        mean_incorrect = [assist_stats.get("exploit_incorrect", {}).get("mean", 0), assist_stats.get("explor_incorrect", {}).get("mean", 0)]
        x = np.arange(len(labels))
        width = 0.35
        rects1 = ax.bar(x - width/2, mean_correct, width, label='Correct Tokens', color='forestgreen')
        rects2 = ax.bar(x + width/2, mean_incorrect, width, label='Incorrect Tokens', color='firebrick')
        ax.set_ylabel('Mean Total Tokens per Run')
        ax.set_title('Final Assistance Summary')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(fontsize='small')
        ax.set_ylim(bottom=0)
        if hasattr(rects1, 'patches'): # Add labels if possible
            ax.bar_label(rects1, padding=3, fmt='%.1f')
            ax.bar_label(rects2, padding=3, fmt='%.1f')
    else:
        ax.text(0.5, 0.5, 'Assistance data missing', ha='center', va='center')
    ax.set_xlabel("Agent Type") # X-axis is Type, not Tick

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Save or show
    save_path = f"agent_model_results/overview_{title_suffix}.png"
    plt.savefig(save_path.replace('(','').replace(')','').replace('=','_')) # Clean filename
    plt.close(fig)


# ---  PLOT 2: ECHO CHAMBERS ---
def plot_echo_chamber_indices(results_dict, title_suffix=""):
    """Plots various echo chamber and information flow metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    fig.suptitle(f"Information Dynamics & Echo Chambers {title_suffix}", fontsize=16)

    # --- Data Extraction ---
    seci = results_dict.get("seci")
    aeci = results_dict.get("aeci")  # AI Call Ratio
    retain_seci = results_dict.get("retain_seci")
    retain_aeci = results_dict.get("retain_aeci")
    comp_seci = results_dict.get("component_seci")
    comp_aeci = results_dict.get("component_aeci")
    aeci_var = results_dict.get("aeci_variance")  # AI Belief Variance Reduction
    comp_ai_trust_var = results_dict.get("component_ai_trust_variance")
    event_ticks_list = results_dict.get("event_ticks_list", [])
    avg_event_ticks = event_ticks_list[0] if event_ticks_list else []

    # Determine Ticks - more robust check
    ticks = np.array([])
    for data_array in [seci, aeci, retain_seci, comp_seci]:
        if data_array is not None and isinstance(data_array, np.ndarray) and data_array.ndim >= 3 and data_array.shape[0] > 0 and data_array.shape[1] > 0:
            if data_array[0, 0, 0] != 0:  # Check if first element looks like a tick
                ticks = data_array[0, :, 0]
                break
            else:
                ticks = np.arange(data_array.shape[1])
                break

    if ticks.size == 0:
        print("Warning: Cannot determine ticks for echo plot.")
        ticks = np.arange(100)  # Default fallback

    # Helper to add event lines
    def add_event_lines(ax, events):
        first = True
        for et in events:
            if et <= ticks[-1]:  # Only plot if within range
                ax.axvline(et, color='gray', linestyle='--', alpha=0.5, label='Disaster Event' if first else "")
                first = False

    # --- Fix _plot_mean_iqr function to be more robust ---
    def _robust_plot_mean_iqr(ax, ticks, data_array, data_index, label, color, linestyle='-'):
        """Helper to plot mean and IQR band with better error handling."""
        if data_array is None or not isinstance(data_array, np.ndarray) or data_array.size == 0:
            print(f"Warning: Empty data for {label}")
            return

        if data_array.ndim < 3:
            print(f"Warning: Data for {label} has wrong dimensions: {data_array.ndim}")
            return

        if data_array.shape[1] == 0:
            print(f"Warning: No time points for {label}")
            return

        if data_index >= data_array.shape[2]:
            print(f"Warning: Invalid data index {data_index} for {label} (max: {data_array.shape[2]-1})")
            return

        # Ensure ticks match data length
        plot_ticks = ticks
        if len(ticks) != data_array.shape[1]:
            print(f"Warning: Tick length mismatch for {label} ({len(ticks)} vs {data_array.shape[1]})")
            plot_ticks = np.arange(data_array.shape[1])

        try:
            # Try with nan-safe functions to handle potential NaNs
            data_slice = data_array[:, :, data_index]
            data_slice = np.where(np.isinf(data_slice), np.nan, data_slice)

            mean = np.nanmean(data_slice, axis=0)
            lower = np.nanpercentile(data_slice, 25, axis=0)
            upper = np.nanpercentile(data_slice, 75, axis=0)

            # Replace any remaining NaNs
            mean = np.nan_to_num(mean)
            lower = np.nan_to_num(lower)
            upper = np.nan_to_num(upper)

            # Ensure percentiles are properly ordered
            lower = np.minimum(mean, lower)
            upper = np.maximum(mean, upper)

            # Clip values for indices and ratios
            if is_ratio_metric:
                mean = np.clip(mean, 0.0, 1.0)
                lower = np.clip(lower, 0.0, 1.0)
                upper = np.clip(upper, 0.0, 1.0)

                # Safety check - print warnings for any out-of-bounds values
                if (mean < 0).any() or (mean > 1).any() or (lower < 0).any() or (lower > 1).any() or (upper < 0).any() or (upper > 1).any():
                    print(f"Warning: After clipping, {label} still has out-of-bounds values")

            ax.plot(plot_ticks, mean, label=label, color=color, linestyle=linestyle)
            ax.fill_between(plot_ticks, lower, upper, color=color, alpha=0.2)

            # For ratio metrics, explicitly set y-axis limits
            if is_ratio_metric:
                ax.set_ylim(0, 1)

        except Exception as e:
            print(f"Error plotting {label}: {e}")

    # --- Subplot 1: SECI & AECI (AI Call Ratio) ---
    ax = axes[0, 0]
    _robust_plot_mean_iqr(ax, ticks, seci, 1, "SECI Exploit", "maroon")
    _robust_plot_mean_iqr(ax, ticks, seci, 2, "SECI Explor", "salmon")
    _robust_plot_mean_iqr(ax, ticks, aeci, 1, "AI Call Ratio Exploit", "darkblue")
    _robust_plot_mean_iqr(ax, ticks, aeci, 2, "AI Call Ratio Explor", "skyblue", linestyle='--')
    add_event_lines(ax, avg_event_ticks)
    ax.set_title("Social Homogeneity (SECI) & AI Call Ratio (AECI)")
    ax.set_ylabel("Index / Ratio")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize='small')
    ax.set_ylim(bottom=0, top=1.05)

    # --- Subplot 2: Retainment ---
    ax = axes[0, 1]
    _robust_plot_mean_iqr(ax, ticks, retain_seci, 1, "Retain Friend (Exploit)", "green")
    _robust_plot_mean_iqr(ax, ticks, retain_seci, 2, "Retain Friend (Explor)", "lightgreen", linestyle='--')
    _robust_plot_mean_iqr(ax, ticks, retain_aeci, 1, "Retain AI (Exploit)", "purple")
    _robust_plot_mean_iqr(ax, ticks, retain_aeci, 2, "Retain AI (Explor)", "plum", linestyle='--')
    add_event_lines(ax, avg_event_ticks)
    ax.set_title("Information Retainment (Share Accepted)")
    ax.set_ylabel("Share of Accepted Info")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize='small')
    ax.set_ylim(bottom=0, top=1.05)

    # --- Subplot 3: Component & AI Variance Indices ---
    ax = axes[1, 0]
    _robust_plot_mean_iqr(ax, ticks, comp_seci, 1, "Component SECI", "black")
    _robust_plot_mean_iqr(ax, ticks, comp_aeci, 1, "Component AI Call Ratio", "grey")
    _robust_plot_mean_iqr(ax, ticks, aeci_var, 1, "AI Bel. Var. Reduction", "magenta")  # AI Echo Chamber
    add_event_lines(ax, avg_event_ticks)
    ax.set_title("Component & AI Echo Chamber Indices")
    ax.set_ylabel("Index / Variance Reduction")
    ax.set_xlabel("Tick")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize='small')
    ax.set_ylim(bottom=0, top=1.05)  # Set explicit limits for consistency

    # --- Subplot 4: AI Trust Clustering ---
    ax = axes[1, 1]
    _robust_plot_mean_iqr(ax, ticks, comp_ai_trust_var, 1, "Component AI Trust Var.", "cyan")
    add_event_lines(ax, avg_event_ticks)
    ax.set_title("AI Trust Clustering (Variance within Components)")
    ax.set_ylabel("Variance")
    ax.set_xlabel("Tick")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize='small')
    ax.set_ylim(bottom=0)  # Only set bottom for variance

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Save or show
    save_path = f"agent_model_results/echo_indices_{title_suffix}.png"
    plt.savefig(save_path.replace('(','').replace(')','').replace('=','_'))
    plt.close(fig)

def plot_seci_aeci_evolution(seci_array, aeci_array, title_suffix=""):
    """Plots SECI and AECI evolution with Mean +/- IQR bands."""

    # Helper to get stats with better error handling
    def get_stats(data_array, index):
        if data_array is None or not isinstance(data_array, np.ndarray) or data_array.size == 0 or data_array.ndim < 3:
            print(f"Warning: Invalid data array for index {index}")
            return None, None, None

        if index >= data_array.shape[2]:
            print(f"Warning: Index {index} out of bounds for data array with shape {data_array.shape}")
            return None, None, None

        try:
            # Extract data slice and handle infinities
            data_slice = data_array[:, :, index]
            data_slice = np.where(np.isinf(data_slice), np.nan, data_slice)

            # For SECI/AECI, ensure values are bounded in [0,1]
            data_slice = np.clip(data_slice, 0.0, 1.0)

            # Calculate stats
            mean = np.nanmean(data_slice, axis=0)
            lower = np.nanpercentile(data_slice, 25, axis=0)
            upper = np.nanpercentile(data_slice, 75, axis=0)

            # Replace NaNs and ensure bounds
            mean = np.clip(np.nan_to_num(mean), 0.0, 1.0)
            lower = np.clip(np.nan_to_num(lower), 0.0, 1.0)
            upper = np.clip(np.nan_to_num(upper), 0.0, 1.0)

            # Ensure percentiles don't cross mean
            lower = np.minimum(mean, lower)
            upper = np.maximum(mean, upper)

            return mean, lower, upper
        except Exception as e:
            print(f"Error calculating stats for index {index}: {e}")
            return None, None, None

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

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
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
    axes[0].set_ylim(0, 1)  # Force y-axis limits to [0,1]

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
    axes[1].set_ylim(0, 1)  # Force y-axis limits to [0,1]

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_ai_trust_vs_alignment(model, save_dir="analysis_plots"):
    """Plot AI trust by agent type with respect to AI alignment level."""
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Gather trust data by agent type
    trust_by_type = {"exploitative": [], "exploratory": []}

    for agent in model.humans.values():
        # Calculate average trust in AI agents
        ai_trust_vals = [agent.trust.get(f"A_{k}", 0) for k in range(model.num_ai)]
        avg_ai_trust = sum(ai_trust_vals) / len(ai_trust_vals) if ai_trust_vals else 0
        trust_by_type[agent.agent_type].append(avg_ai_trust)

    # Calculate statistics
    exploit_mean = np.mean(trust_by_type["exploitative"]) if trust_by_type["exploitative"] else 0
    exploit_std = np.std(trust_by_type["exploitative"]) if trust_by_type["exploitative"] else 0
    explor_mean = np.mean(trust_by_type["exploratory"]) if trust_by_type["exploratory"] else 0
    explor_std = np.std(trust_by_type["exploratory"]) if trust_by_type["exploratory"] else 0

    # Plot
    x = ["Exploitative", "Exploratory"]
    means = [exploit_mean, explor_mean]
    stds = [exploit_std, explor_std]

    ax.bar(x, means, yerr=stds, capsize=10, color=['red', 'blue'], alpha=0.7)
    ax.set_ylabel("Average AI Trust")
    ax.set_title(f"AI Trust by Agent Type (AI Alignment = {model.ai_alignment_level})")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels
    for i, v in enumerate(means):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"ai_trust_alignment_{model.ai_alignment_level:.1f}.png"))
    plt.close()

def plot_assistance_bars(assist_stats, raw_assist_counts, title_suffix=""):
    """Plots mean cumulative correct/incorrect tokens as bars with IQR error bars."""
    labels = ['Exploitative', 'Exploratory']

    # Data for bars (means)
    mean_correct = [
        assist_stats.get("exploit_correct", {}).get("mean", 0),
        assist_stats.get("explor_correct", {}).get("mean", 0)
    ]
    mean_incorrect = [
        assist_stats.get("exploit_incorrect", {}).get("mean", 0),
        assist_stats.get("explor_incorrect", {}).get("mean", 0)
    ]

    # FIXED: Error calculation with better handling
    errors_correct = [[0, 0], [0, 0]]  # Default empty errors
    errors_incorrect = [[0, 0], [0, 0]]  # Default empty errors

    # Calculate errors if data exists
    if "exploit_correct" in raw_assist_counts and raw_assist_counts["exploit_correct"]:
        data = raw_assist_counts["exploit_correct"]
        mean = np.mean(data)
        p25 = np.percentile(data, 25)
        p75 = np.percentile(data, 75)
        errors_correct[0] = [max(0, mean - p25), max(0, p75 - mean)]

    if "explor_correct" in raw_assist_counts and raw_assist_counts["explor_correct"]:
        data = raw_assist_counts["explor_correct"]
        mean = np.mean(data)
        p25 = np.percentile(data, 25)
        p75 = np.percentile(data, 75)
        errors_correct[1] = [max(0, mean - p25), max(0, p75 - mean)]

    if "exploit_incorrect" in raw_assist_counts and raw_assist_counts["exploit_incorrect"]:
        data = raw_assist_counts["exploit_incorrect"]
        mean = np.mean(data)
        p25 = np.percentile(data, 25)
        p75 = np.percentile(data, 75)
        errors_incorrect[0] = [max(0, mean - p25), max(0, p75 - mean)]

    if "explor_incorrect" in raw_assist_counts and raw_assist_counts["explor_incorrect"]:
        data = raw_assist_counts["explor_incorrect"]
        mean = np.mean(data)
        p25 = np.percentile(data, 25)
        p75 = np.percentile(data, 75)
        errors_incorrect[1] = [max(0, mean - p25), max(0, p75 - mean)]

    # FIXED: Properly format error bars for matplotlib
    yerr_correct = np.array(errors_correct).T
    yerr_incorrect = np.array(errors_incorrect).T

    x = np.arange(len(labels))
    width = 0.35

    # FIXED: Create new figure with more space
    fig, ax = plt.subplots(figsize=(12, 8))

    # FIXED: Properly position bars to avoid overlap
    rects1 = ax.bar(x - width/2, mean_correct, width*0.7, yerr=yerr_correct, capsize=4,
                   label='Mean Correct Tokens', color='forestgreen')
    rects2 = ax.bar(x + width/2, mean_incorrect, width*0.7, yerr=yerr_incorrect, capsize=4,
                   label='Mean Incorrect Tokens', color='firebrick')

    ax.set_ylabel('Mean Cumulative Tokens Sent per Run (Total)')
    ax.set_title(f'Token Assistance Summary {title_suffix} (Mean & IQR)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper left')

    # FIXED: Add value labels with position adjustment
    for i, rect in enumerate(rects1):
        height = rect.get_height()
        ax.annotate(f'{mean_correct[i]:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    for i, rect in enumerate(rects2):
        height = rect.get_height()
        ax.annotate(f'{mean_incorrect[i]:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.set_ylim(bottom=0)

    # FIXED: More space in layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = f"agent_model_results/assistance_bars_{title_suffix.replace('(','').replace(')','').replace('=','_')}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

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
    if not unmet_needs_data or not isinstance(unmet_needs_data, list):
        print(f"Warning: Invalid or empty data for plot_unmet_need_evolution {title_suffix}")
        return

    # Filter out None values and empty lists
    valid_runs = [run_data for run_data in unmet_needs_data if run_data is not None and len(run_data) > 0]
    if not valid_runs:
        print(f"Warning: No valid run data for unmet needs evolution {title_suffix}")
        return

    # Pad shorter runs with NaN if lengths differ, then convert to array
    try:
        # Find max length across all runs
        T = max(len(run_data) for run_data in valid_runs)
        if T == 0:
            print(f"Warning: All runs have empty data for {title_suffix}")
            return

        unmet_array = np.full((len(valid_runs), T), np.nan)
        for i, run_data in enumerate(valid_runs):
            unmet_array[i, :len(run_data)] = run_data
    except Exception as e:
        print(f"Warning: Could not process unmet needs data for {title_suffix}: {e}")
        return

    ticks = np.arange(T) # Simple tick count (0 to T-1)

    # Calculate statistics with nan-safe functions
    try:
        mean = np.nanmean(unmet_array, axis=0)
        lower = np.nanpercentile(unmet_array, 25, axis=0)
        upper = np.nanpercentile(unmet_array, 75, axis=0)

        # Replace NaNs with zeros
        mean = np.nan_to_num(mean)
        lower = np.nan_to_num(lower)
        upper = np.nan_to_num(upper)

        # Ensure percentiles don't cross mean
        lower = np.minimum(mean, lower)
        upper = np.maximum(mean, upper)
    except Exception as e:
        print(f"Warning: Error calculating statistics for unmet needs: {e}")
        mean = np.nanmean(unmet_array, axis=0)
        mean = np.nan_to_num(mean)
        lower = mean
        upper = mean

    plt.figure(figsize=(8, 5))
    plt.plot(ticks, mean, label="Mean Unmet Need", color="purple")
    plt.fill_between(ticks, lower, upper, color="purple", alpha=0.2, label="IQR")

    plt.title(f"Unmet Need (Count) {title_suffix} (Mean +/- IQR)")
    plt.xlabel("Tick")
    plt.ylabel("Number of High-Need Cells with 0 Tokens")
    plt.legend(fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(bottom=0)  # Unmet needs cannot be negative
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
        "ai_alignment_level": 0.0,
        "exploitative_correction_factor": 1.0,
        "width": 30,
        "height": 30,
        "lambda_parameter": 0.5,
        "learning_rate": 0.05,
        "epsilon": 0.2,
        "ticks": 150,
        "rumor_probability": 0.7,   # Higher chance components get a rumor
        "rumor_intensity": 2.0,     # Target L5 belief at rumor epicenter
        "rumor_confidence": 0.75,   # Higher confidence in initial rumor
        "rumor_radius_factor": 0.9, # Wider rumor spread
        "min_rumor_separation_factor": 0.5,
        "exploit_trust_lr": 0.03,
        "explor_trust_lr": 0.08,
        "exploit_friend_bias": 0.1,
        "exploit_self_bias": 0.1
    }
    num_runs = 20
    save_dir = "agent_model_results"
    os.makedirs(save_dir, exist_ok=True)

    ##############################################
    # Experiment A: Vary share_exploitative
    ##############################################
    share_values = [0.3, 0.6]
    file_a_pkl = os.path.join(save_dir, "results_experiment_A.pkl")
    file_a_csv = os.path.join(save_dir, "results_experiment_A.csv")

    param_name_a = "Share Exploitative"
    print("Running Experiment A...")
    results_a = experiment_share_exploitative(base_params, share_values, num_runs)
    with open(file_a_pkl, "wb") as f:
        pickle.dump(results_a, f)
    export_results_to_csv(results_a, share_values, file_a_csv, "Experiment A")

    print("\n--- Plotting Aggregated Time Evolution for Experiment A ---")
    for share in share_values:
        print(f"{param_name_a} = {share}")
        # Load data for this parameter setting from the saved dictionary
        results_dict = results_a.get(share, {})
        title_suffix = f"({param_name_a}={share})"

        if results_dict:
            # Call NEW consolidated plot functions
            plot_simulation_overview(results_dict, title_suffix)
            # Ensure your aggregation collects component data for this one:
            plot_echo_chamber_indices(results_dict, title_suffix)
            # Call the original trust plot function (it's already well-structured)
            plot_trust_evolution(results_dict["trust_stats"], title_suffix)

            # Optional: Call final state bar plot for assistance
            # if "assist" in results_dict and "raw_assist_counts" in results_dict:
            #    plot_assistance_bars(results_dict["assist"], results_dict["raw_assist_counts"], title_suffix)

        else:
            print(f"  Skipping plots for {param_name_a}={share} (missing data)")

    # --- Plot SUMMARY Comparisons Across Parameters (AFTER LOOP) ---
    print("\n--- Plotting Summary Comparisons for Experiment A ---")
    # Plot how correct token share changes with the parameter
    if results_a: # Check if results exist before plotting summary
        plot_correct_token_shares_bars(results_a, share_values)

    ##############################################
    # Experiment B: Vary AI Alignment Level
    ##############################################

    alignment_values = [0.0, 0.3, 0.6, 0.9]  # Initial scan
    param_name_b = "AI Alignment Tipping Point"
    file_b_pkl = os.path.join(save_dir, f"results_{param_name_b.replace(' ','_')}.pkl")

    print(f"\nRunning {param_name_b} Experiment...")
    results_b = experiment_alignment_tipping_point(base_params, alignment_values, num_runs=10)

    # Save and plot results as before
    with open(file_b_pkl, "wb") as f:
        pickle.dump(results_b, f)

    # Still plot the regular summaries
    print(f"\n--- Plotting Summary Comparisons for {param_name_b} ---")
    if results_b:
        plot_final_echo_indices_vs_alignment(results_b, sorted(list(results_b.keys())), title_suffix="Tipping Points")
        plot_final_performance_vs_alignment(results_b, sorted(list(results_b.keys())), title_suffix="Tipping Points")

    # --- Plot Aggregated Time Evolution for EACH Alignment Level ---
    print(f"\n--- Plotting Aggregated Time Evolution for {param_name_b} ---")
    for align in alignment_values:
        print(f"{param_name_b} = {align}")
        results_dict = results_b.get(align, {})
        title_suffix = f"({param_name_b}={align})"

        if results_dict:
            # Call the consolidated plotting functions
            plot_simulation_overview(results_dict, title_suffix)
            plot_trust_evolution(results_dict.get("trust_stats"), title_suffix)
            plot_echo_chamber_indices(results_dict, title_suffix)

            # To generate the trust vs alignment plot, we need to run a model with this alignment setting
            # since we need the actual agent objects, not just the aggregated statistics
            model_params = base_params.copy()
            model_params["ai_alignment_level"] = align
            temp_model = run_simulation(model_params)
            plot_ai_trust_vs_alignment(temp_model, save_dir=f"{save_dir}/trust_plots")
            del temp_model

        else:
            print(f"  Skipping plots for {param_name_b}={align} (missing data)")

    # --- Plot SUMMARY Comparisons Across Alignment Levels (AFTER LOOP) ---
    print(f"\n--- Plotting Summary Comparisons for {param_name_b} ---")
    if results_b: # Check if results exist before plotting summary
         # Call the specific summary plot functions for this experiment
         plot_final_echo_indices_vs_alignment(results_b, alignment_values, title_suffix="Exp B")
         plot_final_performance_vs_alignment(results_b, alignment_values, title_suffix="Exp B")
         
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

    gc.collect()

from google.colab import drive
drive.mount('/content/drive')
