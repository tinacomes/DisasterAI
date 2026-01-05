
# Install mesa if not already installed
# !pip install mesa  # Commented for non-Colab usage

# Mount Google Drive FIRST (for Colab)
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
    print("✓ Running in Google Colab - results will be saved to Drive")
except:
    IN_COLAB = False
    print("✓ Running locally - results will be saved to local directory")

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

# Set save directory (Drive if in Colab, local otherwise)
if IN_COLAB:
    save_dir = "/content/drive/MyDrive/DisasterAI_Results"
else:
    save_dir = "agent_model_results"
os.makedirs(save_dir, exist_ok=True)
print(f"✓ Results will be saved to: {save_dir}")

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
                 exploit_trust_lr=0.03, # Low trust Learning Rate for exploiters
                 explor_trust_lr=0.06,  # Higher trust Learning Rate for explorers
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
        self.pending_info_evaluations = [] # [(tick_received, source_id, cell, reported_level), ...] for information quality feedback
        self.tokens_this_tick = {} # Tracks mode choice leading to send_relief THIS tick
        self.last_queried_source_ids = [] # Temp store for source IDs
        self.last_belief_update = {}  # Tracks when each cell was last updated
                     
        # --- Q-Table for Source Values ---
        # Use high-level modes: self_action, human, ai (not individual AIs)
        # This allows agents to learn about source CATEGORIES, not just individuals
        self.q_table = {}
        self.q_table["self_action"] = 0.0
        self.q_table["human"] = 0.0  # Generic value of querying humans as a category
        self.q_table["ai"] = 0.0     # Generic value of querying AI as a category

        # Track individual sources separately for selection within each mode
        # These are updated alongside mode Q-values for granular tracking
        for k in range(model.num_ai):
            self.q_table[f"A_{k}"] = 0.0

        # --- Belief Update Parameters ---
        # These control how beliefs change when info is ACCEPTED (separate from Q-learning)
        self.D = 2.0 if agent_type == "exploitative" else 4 # Acceptance threshold parameter
        self.delta = 3.5 if agent_type == "exploitative" else 1.2 # Acceptance sensitivity parameter
        self.belief_learning_rate = 0.9 if agent_type == "exploratory" else 0.4 # How much belief shifts towards accepted info

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
                # Explorers have wider sensing radius (3) to verify uncertain cells, exploiters narrower (2)
                sense_radius = 3 if self.agent_type == "exploratory" else 2

                # If cell is within sensing range, initialize with noisy perception of actual disaster
                if distance_from_agent <= sense_radius:
                    try:
                        # Get actual level with bounds checking
                        if 0 <= x < self.model.width and 0 <= y < self.model.height:
                            actual_level = self.model.disaster_grid[x, y]

                            # Add significant noise to initial sensing to create diversity
                            noise_chance = 0.4  # 40% chance of noisy reading (increased from 20%)
                            if random.random() < noise_chance:
                                # More diverse noise (-2 to +2)
                                noise = random.choice([-2, -1, -1, 0, 0, 1, 1, 2])
                                initial_level = max(0, min(5, actual_level + noise))
                            else:
                                initial_level = actual_level

                            # Add diversity in confidence levels
                            if self.agent_type == "exploratory":
                                initial_conf = random.uniform(0.5, 0.7)  # Range for explorers
                            else:
                                initial_conf = random.uniform(0.3, 0.6)  # Range for exploiters
                    except (IndexError, TypeError) as e:
                        # Log the error for debugging
                        if self.model.debug_mode:
                            print(f"Warning: Error accessing disaster grid at {x},{y} for agent {self.unique_id}: {e}")
                        # Keep default values set above

                # Apply rumor effects (overlay on top of sensed info)
                if rumor_epicenter:
                    dist_from_rumor = math.sqrt((x - rumor_epicenter[0])**2 + (y - rumor_epicenter[1])**2)
                    if dist_from_rumor < rumor_radius:
                        # Add random variation to rumor level too
                        base_rumor_level = min(5, int(round(3 + rumor_intensity)))
                        rumor_level = max(0, min(5, base_rumor_level + random.choice([-1, 0, 1])))

                        # Only override if rumor level is higher or agent has low confidence
                        if rumor_level > initial_level or initial_conf < rumor_conf:
                            initial_level = rumor_level
                            # Add some variation to rumor confidence too
                            initial_conf = rumor_conf + random.uniform(-0.1, 0.1)

                # Make sure to set belief for every cell
                self.beliefs[cell] = {'level': initial_level, 'confidence': initial_conf}

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

    def choose_best_ai_for_query(self, interest_point, query_radius):
        """Choose the AI with the most knowledge about the area of interest."""
        if not hasattr(self.model, 'ai_knowledge_maps'):
            # If knowledge maps aren't initialized, return random AI
            return f"A_{random.randrange(self.model.num_ai)}"

        # Get cells in the area of interest
        cells_of_interest = self.model.grid.get_neighborhood(
            interest_point, moore=True, radius=query_radius, include_center=True
        )

        # Count how many cells each AI knows about
        ai_knowledge_counts = {}
        for ai_id, knowledge_map in self.model.ai_knowledge_maps.items():
            count = 0
            for cell in cells_of_interest:
                if 0 <= cell[0] < self.model.width and 0 <= cell[1] < self.model.height:
                    if knowledge_map[cell[0], cell[1]] == 1:
                        count += 1
            ai_knowledge_counts[ai_id] = count

        # Find AI with most knowledge (with some randomness)
        if not ai_knowledge_counts:
            return f"A_{random.randrange(self.model.num_ai)}"

        # Sort AIs by knowledge count (descending)
        sorted_ais = sorted(ai_knowledge_counts.items(), key=lambda x: x[1], reverse=True)

        # Choose among top 2 AIs (add randomness)
        if len(sorted_ais) >= 2 and sorted_ais[0][1] > 0 and sorted_ais[1][1] > 0:
            # If top AI has significantly more knowledge, pick it
            if sorted_ais[0][1] >= 2 * sorted_ais[1][1]:
                return sorted_ais[0][0]
            # Otherwise, randomly choose between top 2
            return random.choice([sorted_ais[0][0], sorted_ais[1][0]])
        elif sorted_ais and sorted_ais[0][1] > 0:
            # Only one AI has knowledge
            return sorted_ais[0][0]

        # No AI has knowledge - pick random
        return f"A_{random.randrange(self.model.num_ai)}"

    def apply_confidence_decay(self):
        """Apply confidence decay with more stability and cell-specific rates."""
        base_decay_rate = 0.0003 if self.agent_type == "exploitative" else 0.0005  # Higher for exploratory
        min_confidence_floor = 0.1 if self.agent_type == "exploitative" else 0.15  # Higher floor for exploratory


        # Add position-based and level-based decay adjustments
        for cell, belief in self.beliefs.items():
            if isinstance(belief, dict):
                confidence = belief.get('confidence', 0.1)
                level = belief.get('level', 0)

                # Calculate initial adaptive decay rate from base rate
                adaptive_decay_rate = base_decay_rate

                if self.agent_type == "exploratory" and belief.get('confidence', 0) > 0.8:
                    adaptive_decay_rate *= 1.5  # Increase decay for high confidence beliefs

                # Skip decay for high confidence in high-level disaster cells
                if level >= 4 and confidence > 0.8:
                    continue

                # Apply level-based decay adjustment
                # Less decay for important cells (higher levels)
                level_factor = max(0.5, 1.0 - (level / 10.0))

                # Apply distance-based decay adjustment
                # Cells further from agent decay faster
                if self.pos and cell:
                    distance = math.sqrt((cell[0] - self.pos[0])**2 + (cell[1] - self.pos[1])**2)
                    # Scale by sensing radius (same for both agent types)
                    radius = 2
                    distance_factor = min(1.5, 1.0 + (distance / (2 * radius)))
                else:
                    distance_factor = 1.0

                # Calculate final adaptive decay rate
                adaptive_decay_rate = base_decay_rate * level_factor * distance_factor

                # Apply decay with higher floor
                if confidence > min_confidence_floor:
                    # Apply smaller decay for recently acquired high-confidence beliefs
                    if confidence > 0.9 and hasattr(self, 'last_belief_update') and cell in self.last_belief_update:
                        ticks_since_update = self.model.tick - self.last_belief_update[cell]
                        if ticks_since_update < 5:  # Recently updated
                            adaptive_decay_rate *= 0.2  # Much slower decay

                    new_confidence = max(min_confidence_floor, confidence - adaptive_decay_rate)
                    self.beliefs[cell]['confidence'] = new_confidence

    def query_source(self, source_id, interest_point, query_radius):
        source_agent = self.model.humans.get(source_id) or self.model.ais.get(source_id)
        if source_agent:
            if hasattr(source_agent, 'report_beliefs'):
                return source_agent.report_beliefs(interest_point, query_radius)
        return {}

    def sense_environment(self):
        pos = self.pos
        # CRITICAL FIX #1: Explorers need wider sensing radius to verify uncertain cells they query about
        # This enables the info verification mechanism (pending_info_evaluations) to work for explorers
        # Exploiters have narrower focus (radius=2), explorers cast wider net (radius=3)
        radius = 3 if self.agent_type == "exploratory" else 2
        cells = self.model.grid.get_neighborhood(pos, moore=True, radius=radius, include_center=True)
        for cell in cells:
            if 0 <= cell[0] < self.model.width and 0 <= cell[1] < self.model.height:
                actual = self.model.disaster_grid[cell[0], cell[1]]
                noise_roll = random.random()
                noise_threshold = 0.08

                belief_level = 0
                belief_conf = 0.1

                if noise_roll < noise_threshold: # Noisy Read
                    belief_level = max(0, min(5, actual + random.choice([-1, 1])))
                    belief_conf = 0.5
                else: # Accurate Read
                    belief_level = actual
                    if self.agent_type == "exploitative":
                        # Keep slightly lower base confidence, but boost strongly for high levels
                        belief_conf = 0.6 # Lower base than explorer
                        if belief_level >= 3:
                            belief_conf = 0.85 # HIGH confidence if accurately sensing L3+
                        elif belief_level > 0:
                            belief_conf = 0.75 # Moderate confidence for L1/L2
                    else: # Exploratory (Keep previous boost: 0.90 base, 0.98 for L>0)
                        belief_conf = 0.9
                        if belief_level > 0:
                            belief_conf = 0.98

                if cell in self.beliefs and isinstance(self.beliefs[cell], dict):
                    old_belief = self.beliefs[cell]
                    old_level = old_belief.get('level', 0)
                    old_confidence = old_belief.get('confidence', 0.1)

                    # FIXED: Calculate distance-based weight - cells further away get less weight
                    distance = math.sqrt((cell[0] - pos[0])**2 + (cell[1] - pos[1])**2)
                    distance_factor = max(0.1, 1.0 - (distance / (radius + 1)))

                    # FIXED: More conservative blending weights
                    # Direct perception has more weight for nearby cells and accurate readings
                    accuracy_factor = 0.7 if noise_roll >= noise_threshold else 0.5
                    sense_weight = accuracy_factor * distance_factor

                    # FIXED: Blend both level and confidence
                    # For level, use weighted average with rounding
                    final_level = int(round(sense_weight * belief_level + (1 - sense_weight) * old_level))

                    # For confidence, use weighted average with bounds
                    final_confidence = (sense_weight * belief_conf) + ((1 - sense_weight) * old_confidence)
                    final_confidence = max(0.1, min(0.98, final_confidence))

                    self.beliefs[cell] = {'level': final_level, 'confidence': final_confidence}
                else:
                    # No existing belief, create new one
                    self.beliefs[cell] = {'level': belief_level, 'confidence': belief_conf}

                # Evaluate information quality feedback for this sensed cell
                self.evaluate_information_quality(cell, actual)

    def evaluate_information_quality(self, cell, actual_level):
        """
        Evaluate pending information about a cell when directly observed.
        Provides fast feedback (3-15 tick window) on information accuracy.
        Wider window to accommodate different agent movement patterns.
        """
        current_tick = self.model.tick
        evaluated = []

        for tick_received, source_id, eval_cell, reported_level in self.pending_info_evaluations:
            if eval_cell != cell:
                continue

            # Check if within evaluation window (3-15 ticks) - wider for better coverage
            ticks_elapsed = current_tick - tick_received
            if ticks_elapsed > 15:
                # Too old, remove from pending
                evaluated.append((tick_received, source_id, eval_cell, reported_level))
                continue

            if ticks_elapsed < 3:
                # Too soon, wait a bit for potential disaster evolution
                continue

            # Within window: evaluate information quality
            # Calculate accuracy based on how close reported level was to actual
            level_error = abs(reported_level - actual_level)

            # Accuracy-based reward - scaled to be comparable to relief outcomes
            # Info quality should be FAST and STRONG signal for learning
            if level_error == 0:
                accuracy_reward = 0.5  # Perfect accuracy - strong positive
            elif level_error == 1:
                accuracy_reward = 0.2  # Close - moderate positive
            elif level_error == 2:
                accuracy_reward = -0.3  # Moderate error - significant penalty
            else:
                accuracy_reward = -0.7  # Large error - strong penalty

            # Determine mode from source_id (map to high-level modes)
            if source_id.startswith("H_"):
                mode = "human"
            elif source_id.startswith("A_"):
                mode = "ai"  # Generic AI mode (not individual AI)
            else:
                mode = None

            # Update mode Q-value (what's used in action selection)
            if mode and mode in self.q_table:
                old_mode_q = self.q_table[mode]
                # Info quality feedback should be FAST and STRONG
                # Use higher learning rate for exploratory agents to learn quickly from info
                info_learning_rate = 0.25 if self.agent_type == "exploratory" else 0.12
                # Use standard Q-learning update: Q += lr * (reward - Q)
                new_mode_q = old_mode_q + info_learning_rate * (accuracy_reward - old_mode_q)
                self.q_table[mode] = new_mode_q

            # Also update specific source Q-value (for tracking individuals)
            if source_id in self.q_table:
                old_q = self.q_table[source_id]
                info_learning_rate = 0.25 if self.agent_type == "exploratory" else 0.12
                # Use standard Q-learning update: Q += lr * (reward - Q)
                new_q = old_q + info_learning_rate * (accuracy_reward - old_q)
                self.q_table[source_id] = new_q

            # Update trust similarly - use stronger updates for bad info
            if source_id in self.trust:
                old_trust = self.trust[source_id]
                # Map accuracy reward [-0.7, +0.5] to trust target [0.15, 0.75]
                # Bad info should drop trust low, good info should increase moderately
                trust_target = 0.5 + 0.5 * accuracy_reward  # Maps -0.7->0.15, 0->0.5, +0.5->0.75
                # Use higher learning rate for trust updates from info quality
                trust_lr = 0.15 if self.agent_type == "exploratory" else 0.08
                trust_change = trust_lr * (trust_target - old_trust)
                new_trust = max(0.0, min(1.0, old_trust + trust_change))
                self.trust[source_id] = new_trust

            # Mark as evaluated
            evaluated.append((tick_received, source_id, eval_cell, reported_level))

            # DEBUG: Track info feedback
            if self.model.debug_mode and hasattr(self, 'id_num') and (self.id_num < 2 or (50 <= self.id_num < 52)):
                print(f"[DEBUG] Agent {self.unique_id} ({self.agent_type}) INFO FEEDBACK: source={source_id}, mode={mode}, error={level_error}, reward={accuracy_reward:.3f}")

        # Remove evaluated items from pending list
        self.pending_info_evaluations = [
            item for item in self.pending_info_evaluations
            if item not in evaluated
        ]

        # DEBUG: Track pending list size
        if self.model.debug_mode and hasattr(self, 'id_num') and (self.id_num < 2 or (50 <= self.id_num < 52)) and self.model.tick % 10 == 0:
            print(f"[DEBUG] Agent {self.unique_id} ({self.agent_type}) tick={self.model.tick}: {len(self.pending_info_evaluations)} pending evals")

    def report_beliefs(self, interest_point, query_radius):
        """Reports human agent's beliefs about cells within query_radius of the interest_point."""
        report = {}

        # Check if agent can report (e.g., not in high disaster zone)
        try:
            current_pos_level = self.model.disaster_grid[self.pos[0], self.pos[1]]
            if current_pos_level >= 4 and random.random() < 0.1:
                # Agent in severe danger might not respond
                return {}
        except (TypeError, IndexError):
            return {}

        # Get neighborhood around the specified interest_point
        cells_to_report_on = self.model.grid.get_neighborhood(
            interest_point,
            moore=True,
            radius=query_radius,
            include_center=True
        )

        # Debug info similar to AI agent
        # if hasattr(self.model, 'debug_mode') and self.model.debug_mode and random.random() < 0.05:
            #print(f"DEBUG: Human {self.unique_id} asked to report on {len(cells_to_report_on)} cells around {interest_point}")

        # Track how many cells had direct beliefs vs guesses
        known_cells = 0
        guessed_cells = 0

        for cell in cells_to_report_on:
            # Check if cell is valid
            if 0 <= cell[0] < self.model.width and 0 <= cell[1] < self.model.height:
                # Check if agent has belief about this cell
                has_belief = cell in self.beliefs and isinstance(self.beliefs[cell], dict)

                if has_belief:
                    # Report own belief about the cell
                    belief_info = self.beliefs[cell]
                    current_level = belief_info.get('level', 0)
                    current_conf = belief_info.get('confidence', 0.1)

                    # Apply a small chance of reporting noise
                    if random.random() < 0.05:
                        noisy_level = max(0, min(5, current_level + random.choice([-1, 1])))
                        report[cell] = noisy_level
                    else:
                        report[cell] = current_level

                    known_cells += 1
                else:
                    # HUMAN GUESSING MECHANISM - similar to AI but less sophisticated
                    # Humans guess less than AI (50% vs 75% for AI)
                    if random.random() < 0.5:
                        # Look at surrounding cells with beliefs
                        nearby_values = []

                        for dx in range(-1, 2):
                            for dy in range(-1, 2):
                                nearby = (cell[0] + dx, cell[1] + dy)
                                if (nearby in self.beliefs and
                                    isinstance(self.beliefs[nearby], dict)):
                                    nearby_values.append(self.beliefs[nearby].get('level', 0))

                        if nearby_values:
                            # Simple average with random noise
                            avg_value = sum(nearby_values) / len(nearby_values)
                            noise = random.choice([-1, 0, 0, 1])
                            guessed_value = max(0, min(5, int(round(avg_value + noise))))
                            report[cell] = guessed_value
                            guessed_cells += 1

        # Debug output for comparison with AI
        #if hasattr(self.model, 'debug_mode') and self.model.debug_mode and random.random() < 0.05:
           # total_reported = len(report)
            #print(f"DEBUG: Human {self.unique_id} reported on {total_reported} cells "
                  #f"({known_cells} known, {guessed_cells} guessed)")

        return report

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
            #if self.model.debug_mode:
                #print(f"Agent {self.unique_id} ({self.agent_type}): No L{min_level_to_explore}+ targets found. Using fallback.")

            # NEW APPROACH: Look for L0 cells with highest uncertainty (lowest confidence)
            l0_candidates = []

            for cell, belief_info in self.beliefs.items():
                if isinstance(belief_info, dict):
                    # Check if cell coordinates are valid
                    if not (0 <= cell[0] < self.model.width and 0 <= cell[1] < self.model.height):
                        continue

                    level = belief_info.get('level', 0)
                    conf = belief_info.get('confidence', 1.0)

                    # Focus on L0 cells now
                    if level == 0:
                        uncertainty = 1.0 - conf
                        # For L0 cells, score is purely based on uncertainty
                        l0_candidates.append({
                            'cell': cell,
                            'score': uncertainty,  # Higher uncertainty = higher score
                            'level': level,
                            'conf': conf
                        })

            # Sort L0 candidates by uncertainty (highest first)
            if l0_candidates:
                l0_candidates.sort(key=lambda x: x['score'], reverse=True)
                # Take the top candidate(s)
                candidates = l0_candidates[:num_targets]

                #if self.model.debug_mode:
                 #   print(f"  Found {len(l0_candidates)} L0 candidates, using highest uncertainty ones")
                  #  for c in candidates[:min(3, len(candidates))]:
                   #     print(f"  L0 Candidate: Cell:{c['cell']} Uncertainty:{c['score']:.3f} Conf:{c['conf']:.2f}")
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
            #if self.model.debug_mode and self.model.tick % 10 == 1 and random.random() < 0.2:
               # print(f"DEBUG Tick {self.model.tick} Agt {self.unique_id} Top Explore Candidates:")
                #for cand in candidates[:min(5, len(candidates))]:
                    #print(f"  Cell:{cand.get('cell')} Lvl:{cand.get('level')} Conf:{cand.get('conf'):.2f} Score:{cand.get('score'):.3f}")

            self.exploration_targets = [c['cell'] for c in candidates[:num_targets]]
        else:
            # Final fallback if all else fails
            if self.model.debug_mode:
                print(f"Agent {self.unique_id}: No exploration candidates found at all, using position as target")
            self.exploration_targets = [self.pos]  # Use current position as a last resort

    def apply_trust_decay(self):
        """Applies a slow decay to all trust relationships."""
        #  Make decay rates more distinct and agent-specific
        if self.agent_type == "exploitative":
            base_decay_rate = 0.002     # Standard decay
            friend_decay_rate = 0.0005  # Very slow decay for friends - exploiters value stable relationships
        else:  # exploratory
            base_decay_rate = 0.003     # Slightly faster standard decay - more critical
            friend_decay_rate = 0.0015  # Faster friend decay - less friend bias

        # Apply uniform decay rates (no alignment peeking)
        for source_id in list(self.trust.keys()):
            # Friend decay rate
            if source_id in self.friends:
                decay_rate = friend_decay_rate
            # Standard decay for all other sources (AI and non-friends)
            else:
                decay_rate = base_decay_rate

            # Apply the calculated decay rate
            self.trust[source_id] = max(0, self.trust[source_id] - decay_rate)

    def update_belief_bayesian(self, cell, reported_level, source_trust, source_id=None):
        """Update agent's belief about a cell using Bayesian principles."""
        try:
            # Get current belief
            if cell not in self.beliefs:
                # Initialize if not present
                self.beliefs[cell] = {'level': 0, 'confidence': 0.1}

            current_belief = self.beliefs[cell]
            prior_level = current_belief.get('level', 0)
            prior_confidence = current_belief.get('confidence', 0.1)

            # Apply a minimum confidence threshold to prevent wild swings
            prior_confidence = max(0.1, prior_confidence)

            # Convert confidence to precision with agent-specific scaling
            if self.agent_type == "exploitative":
                # Exploiters have higher prior precision (stronger resistance to change)
                prior_precision = 1.8 * prior_confidence / (1 - prior_confidence + 1e-6)
            else:
                # Explorers have lower prior precision (more open to new information)
                prior_precision = 0.8 * prior_confidence / (1 - prior_confidence + 1e-6)

            # Source precision calculation - Agent-type dependent
            if self.agent_type == "exploitative":
                # Exploiters highly value trusted sources (trust has more impact)
                source_precision_base = 4.0 * source_trust / (1 - source_trust + 1e-6)
            else:
                # Explorers more moderately weigh trust
                source_precision_base = 2.5 * source_trust / (1 - source_trust + 1e-6)

            source_precision = source_precision_base

            # Apply conditional adjustments to source_precision

            is_ai_source = hasattr(self, 'ai_info_sources') and cell in self.ai_info_sources

            if source_trust < 0.3:
                # Smoother transition for low trust
                trust_factor = (source_trust / 0.3) ** 0.5  # Square root for smoother curve
                source_precision = source_precision_base * trust_factor

            # Lower threshold for ignoring extremely low trust sources
            if source_trust < 0.03:
                source_precision *= 0.05  # Almost entirely ignore

            if is_ai_source and source_trust > 0.3:
                # Give extra weight to AI information from somewhat trusted sources
                # No alignment peeking - trust should capture source quality
                source_precision = source_precision * 1.5

            # Apply a maximum to source precision to prevent overwhelming prior
            # Different maximums by agent type
            if self.agent_type == "exploitative":
                source_precision = min(source_precision, 8.0)  # Lower cap (more resistant)
            else:
                source_precision = min(source_precision, 12.0)  # Higher cap (more adaptive)

            # Combine information using precision weighting
            posterior_precision = prior_precision + source_precision

            # Calculate the weighted update of belief level
            posterior_level = (prior_precision * prior_level + source_precision * reported_level) / posterior_precision

            # Convert precision back to confidence [0,1]
            posterior_confidence = posterior_precision / (1 + posterior_precision)

            # Constrain to valid ranges
            posterior_level = max(0, min(5, round(posterior_level)))
            posterior_confidence = max(0.1, min(0.98, posterior_confidence))

            # Agent-type-specific adjustments
            if self.agent_type == "exploitative":
                # Exploitative agents give more weight to consistent information
                if abs(posterior_level - prior_level) <= 1:
                    # Information confirms existing belief - stronger boost
                    confirmation_boost = min(0.3, 0.35 * prior_confidence)
                    posterior_confidence = min(0.98, posterior_confidence + confirmation_boost)
            else:  # exploratory
                # Exploratory agents are more accepting of new information
                if abs(posterior_level - prior_level) >= 2:
                    # Information significantly differs from prior
                    # Don't reduce confidence as much - they value the new information
                    posterior_confidence = max(0.2, posterior_confidence * 0.95)

                # Explorers gain extra confidence when source is trusted and reported level is high
                if source_trust > 0.6 and reported_level >= 3:
                    info_value_boost = min(0.3, 0.4 * source_trust)
                    posterior_confidence = min(0.97, posterior_confidence + info_value_boost)

            # Apply a smoothing factor to reduce large jumps in level for both agent types
            if abs(posterior_level - prior_level) >= 2:
                # Apply 20% smoothing for large changes (weighted average)
                smoothing_factor = 0.2
                smoothed_level = int(round((1-smoothing_factor) * posterior_level + smoothing_factor * prior_level))
                posterior_level = smoothed_level

            # Update the belief
            self.beliefs[cell] = {
                'level': int(posterior_level),
                'confidence': posterior_confidence
            }

            # Determine if this was a significant belief change
            # FIXED: More nuanced threshold - count if level OR confidence changes significantly
            level_change = abs(posterior_level - prior_level)
            confidence_change = abs(posterior_confidence - prior_confidence)
            significant_change = (level_change >= 1 or confidence_change >= 0.1)

            # Track information for quality feedback
            # CRITICAL FIX: Track ALL external queries (human/AI), not just significant changes
            # This ensures exploratory agents get feedback even when querying about
            # cells they've already sensed with high confidence
            if source_id:  # source_id is None for self_action
                self.pending_info_evaluations.append((
                    self.model.tick,
                    source_id,
                    cell,
                    int(reported_level)  # The level reported by the source (NOT posterior)
                ))
                # DEBUG: Track pending info evaluations
                if self.model.debug_mode and hasattr(self, 'id_num') and (self.id_num < 2 or (50 <= self.id_num < 52)):
                    print(f"[DEBUG] Agent {self.unique_id} ({self.agent_type}) added pending info eval: tick={self.model.tick}, source={source_id}, cell={cell}, reported={reported_level}")

            # Track AI source information for later trust updates
            is_ai_source = hasattr(self, 'ai_info_sources') and cell in self.ai_info_sources
            if is_ai_source and significant_change:
                if not hasattr(self, 'ai_acceptances'):
                    self.ai_acceptances = {}
                self.ai_acceptances[cell] = self.model.tick

            # BUG FIX: Removed duplicate acceptance tracking from here
            # Acceptance is now ONLY tracked in seek_information() (lines ~1070-1077)
            # to avoid double-counting. Previously this incremented counters,
            # then seek_information incremented them again = 2x inflation!

            # Return whether this update caused a significant belief change
            return significant_change

        except Exception as e:
            print(f"ERROR in Agent {self.unique_id} update_belief_bayesian: {e}")
            return False

    def seek_information(self):
        """
        Queries a single source for information about an interest point, processes the report,
        and updates beliefs if accepted. Includes robust error handling and logging.
        """
        reports = {}
        source_agent_id = None
        interest_point = None
        query_radius = 0
        reports = {}
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

                        # Ensure we have at least one valid cell before choosing
                        if highest_conf_cells:
                            interest_point = random.choice(highest_conf_cells)
                        else:
                            # Absolute fallback: pick a random cell in the grid
                            interest_point = (random.randrange(self.model.width), random.randrange(self.model.height))
                            if self.model.debug_mode:
                                print(f"Agent {self.unique_id}: Using random fallback interest point {interest_point}")

            else:  # Exploratory
                # CRITICAL FIX: Explorers should seek information about UNCERTAIN cells
                # This is their core defining characteristic per requirements
                # Find cells with low confidence (high uncertainty)
                self.find_exploration_targets(num_targets=3)

                if self.exploration_targets:
                    # Query about the most uncertain cell (first in exploration targets)
                    interest_point = self.exploration_targets[0]
                else:
                    # Fallback: find cells with lowest confidence manually
                    min_conf = float('inf')
                    lowest_conf_cells = []

                    if len(self.beliefs) > 0:
                        for cell, belief_info in self.beliefs.items():
                            if isinstance(belief_info, dict):
                                conf = belief_info.get('confidence', 1.0)
                                if conf < min_conf:
                                    min_conf = conf
                                    lowest_conf_cells = [cell]
                                elif conf == min_conf:
                                    lowest_conf_cells.append(cell)

                    if lowest_conf_cells:
                        interest_point = random.choice(lowest_conf_cells)
                    else:
                        # Absolute fallback: use current position
                        interest_point = self.pos
                        if self.model.debug_mode:
                            print(f"Agent {self.unique_id}: Using position as fallback interest point {interest_point}")

                # Use wider query radius for exploration (explorers cast wider net)
                query_radius = 3

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
            #if self.model.debug_mode and random.random() < 0.05:  # Only log ~5% of decisions
                #print(f"Tick {self.model.tick} Agent {self.unique_id} ({self.agent_type}) selected interest_point: {interest_point}")
               # print(f" > My belief: {self.beliefs.get(interest_point, {})}")
                #print(f" > Ground truth: {self.model.disaster_grid[interest_point[0], interest_point[1]]}")


            if not interest_point:
                print(f"Agent {self.unique_id}: No valid interest point, skipping seek_information.")
                return

            # Debug logging
            #if self.model.debug_mode and random.random() < 0.05:  # Only log ~5% of decisions to avoid spam
               # print(f"Tick {self.model.tick} Agent {self.unique_id} ({self.agent_type}) selected interest_point: {interest_point}")
               # print(f" > My belief: {self.beliefs.get(interest_point, {})}")
                #print(f" > Ground truth: {self.model.disaster_grid[interest_point[0], interest_point[1]]}")

            # Source selection (epsilon-greedy with type-specific biases)
            # Use 3-mode structure: self_action, human, ai
            # Then select specific source within chosen mode
            possible_modes = ["self_action", "human", "ai"]

            # Store Q-values
            for mode in possible_modes:
                decision_factors['q_values'][mode] = self.q_table.get(mode, 0.0)

            # Epsilon greedy strategy - not to be confused with the agent types :)
            # Exploration case - record randomly chosen mode
            if random.random() < self.epsilon: #epsilon parameter for randomness
                chosen_mode = random.choice(possible_modes)
                decision_factors['selection_type'] = 'exploration'
                decision_factors['chosen_mode'] = chosen_mode
            else:
                # Exploitation case - record all factors in decision
                decision_factors['selection_type'] = 'exploitation'

                # Base scores are from Q-table
                scores = {mode: self.q_table.get(mode, 0.0) for mode in possible_modes}
                decision_factors['base_scores'] = scores.copy()

                # Agent-type specific biases (preferences, not alignment-based)
                decision_factors['biases'] = {}

                if self.agent_type == "exploitative":
                    # Exploitative agents prefer friends and self-confirmation
                    scores["human"] += self.exploit_friend_bias
                    scores["self_action"] += self.exploit_self_bias

                    decision_factors['biases']["human"] = self.exploit_friend_bias
                    decision_factors['biases']["self_action"] = self.exploit_self_bias

                    # NO AI bias - let Q-learning determine AI value through experience

                else:  # exploratory
                    # Exploratory agents seek diverse information sources
                    # Bias toward querying to get info quality feedback
                    scores["human"] += 0.2   # Encourage querying humans
                    scores["ai"] += 0.2      # Encourage querying AI
                    scores["self_action"] -= 0.1  # Discourage pure self-reliance

                    decision_factors['biases']["human"] = 0.2
                    decision_factors['biases']["ai"] = 0.2
                    decision_factors['biases']["self_action"] = -0.1

                    # NO alignment-based biases - let Q-learning determine which sources are good
                    # Exploratory agents will naturally prefer accurate sources through feedback

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
            #if self.model.tick % 10 == 0 and random.random() < 0.2:  # Log ~20% of decisions every 10 ticks
               # if self.model.debug_mode:
                   # print(f"\nAgent {self.unique_id} ({self.agent_type}) source selection:")
                    #print(f"  Decision type: {decision_factors['selection_type']}")
                    #print(f"  Chosen mode: {decision_factors['chosen_mode']}")
                    #if decision_factors['selection_type'] == 'exploitation':
                        #print(f"  Base Q-values: {decision_factors['base_scores']}")
                        #print(f"  Applied biases: {decision_factors['biases']}")
                        #print(f"  Final scores: {decision_factors['final_scores']}")
                    #print(f"  AI alignment level: {self.model.ai_alignment_level}")

            self.tokens_this_tick = {chosen_mode: 1}
            self.last_queried_source_ids = []

            # Query source based on chosen mode

            if chosen_mode == "self_action":
                reports = self.report_beliefs(interest_point, query_radius)
                source_id = None  # No external source used

            elif chosen_mode == "human":
                # Select specific human source within "human" mode
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

            elif chosen_mode == "ai":
                # Select specific AI source within "ai" mode
                # Pick best AI for this query based on knowledge coverage
                source_id = self.choose_best_ai_for_query(interest_point, query_radius)

                source_agent = self.model.ais.get(source_id)

                if source_agent:
                    reports = source_agent.report_beliefs(interest_point, query_radius, self.beliefs, self.trust.get(source_id, 0.1))

                    # track AI source
                    self.last_queried_source_ids = [source_id]

                    if not hasattr(self, 'ai_info_sources'):
                        self.ai_info_sources = {}

                    # Track which cells got info from which AI
                    for cell, reported_value in reports.items():
                        self.ai_info_sources[cell] = source_id

                    # DEBUG PRINT to track AI source queries
                    #if self.model.debug_mode and random.random() < 0.1:  # 10% of the time
                        #print(f"DEBUG: Agent {self.unique_id} queried AI {source_id} for {len(reports)} cells")

                    self.accum_calls_ai += 1
                    self.accum_calls_total += 1
                else:
                    # Invalid source agent
                    source_id = None


            # Process reports and update beliefs
            belief_updates = 0
            source_trust = self.trust.get(source_id, 0.1) if source_id else 0.1

            for cell, reported_value in reports.items():
                if cell not in self.beliefs:
                    continue

                # Convert to integer level if it's not already
                reported_level = int(round(reported_value))

                # Use the Bayesian update function
                significant_update = self.update_belief_bayesian(cell, reported_level, source_trust, source_id)
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
                            # INCREMENT AI ACCEPTANCE COUNTER HERE
                            self.accepted_ai += 1

                            # DEBUG PRINT to track AI info acceptance
                            #if self.model.debug_mode and random.random() < 0.05:  # 5% of the time
                                #print(f"DEBUG: Agent {self.unique_id} accepted info from AI {source_id} for cell {cell}")

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
                        (level / 5.0) * 0.7 + (1.0 - confidence) * 0.3  # Prioritize level (70%) and exploration (30%)
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

                # Relief outcome delay: 15-25 ticks (realistic logistics delay)
                # Random variation models variable communication/logistics times
                relief_delay = random.randint(15, 25)
                self.pending_rewards.append((
                    self.model.tick + relief_delay,
                    responsible_mode,
                    reward_cells,
                    self.last_queried_source_ids
                ))

            # NOTE: Don't clear tokens_this_tick here! It's cleared at the START of model.step()
            # to ensure calculate_info_diversity() can read it at the END of the tick

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

                # possible diagnostic for reward processing
                #if self.model.debug_mode and random.random() < 0.1:
                    #print(f"Agent {self.unique_id} processing rewards for {len(cells_and_beliefs)} cells")

                # CRITICAL FIX #4: Separate ACCURACY vs CONFIRMATION rewards
                # Exploratory: accuracy = distance from ground truth
                # Exploitative: confirmation = how well reality matched confident prior beliefs

                accuracy_scores = []  # For exploratory agents
                confirmation_scores = []  # For exploitative agents

                for cell, belief_level in cells_and_beliefs:
                    if not (0 <= cell[0] < self.model.width and 0 <= cell[1] < self.model.height):
                        continue

                    # Get the ACTUAL disaster level for this cell
                    actual_level = self.model.disaster_grid[cell[0], cell[1]]

                    # Define correctness criteria (for metrics tracking)
                    is_correct = actual_level >= 3
                    if is_correct:
                        correct_in_batch += 1
                    else:
                        incorrect_in_batch += 1

                    # Get prior belief before updating (needed for confirmation reward)
                    prior_belief_level = belief_level  # This is what we believed when sending relief
                    prior_confidence = self.beliefs[cell].get('confidence', 0.5) if cell in self.beliefs else 0.5

                    if self.agent_type == "exploratory":
                        # ACCURACY REWARD: How close was belief to ground truth?
                        # Range: 1.0 (perfect match) to 0.0 (maximum error of 5)
                        accuracy = 1.0 - abs(prior_belief_level - actual_level) / 5.0
                        accuracy_scores.append(accuracy)
                    else:  # exploitative
                        # CONFIRMATION REWARD: Did reality match our confident beliefs?
                        # Match quality: how close belief was to reality
                        match_quality = 1.0 - abs(prior_belief_level - actual_level) / 5.0

                        # Weight by prior confidence:
                        # - High confidence + good match = high reward
                        # - High confidence + poor match = high penalty
                        # - Low confidence = low impact either way
                        if match_quality >= 0.6:  # Reasonably accurate (error <= 2)
                            # Reward confirming strong beliefs
                            confirmation = match_quality * prior_confidence
                        else:  # Inaccurate (error > 2)
                            # Penalty for strong wrong beliefs
                            confirmation = -(1.0 - match_quality) * prior_confidence

                        confirmation_scores.append(confirmation)

                    # Update beliefs based on ground truth (same for both agent types)
                    if cell in self.beliefs and isinstance(self.beliefs[cell], dict):
                        old_belief = self.beliefs[cell].copy()

                        # Direct update with ground truth (with some noise)
                        noise = random.choice([-1, 0, 0, 0, 1]) if random.random() < 0.2 else 0
                        corrected_level = max(0, min(5, actual_level + noise))

                        # Blend with existing belief
                        update_weight = 0.7  # Strong update toward reality
                        blended_level = int(round(update_weight * corrected_level +
                                                (1 - update_weight) * old_belief.get('level', 0)))

                        # Update confidence based on accuracy (agent-type dependent)
                        if abs(old_belief.get('level', 0) - actual_level) <= 1:
                            # Belief was accurate
                            if self.agent_type == "exploratory":
                                # Explorers moderately increase confidence when correct
                                new_conf = min(0.9, old_belief.get('confidence', 0.5) + 0.15)
                            else:
                                # Exploiters strongly increase confidence when correct (confirmation bias)
                                new_conf = min(0.98, old_belief.get('confidence', 0.5) + 0.25)
                        else:
                            # Belief was inaccurate
                            if self.agent_type == "exploratory":
                                # Explorers dramatically reduce confidence when wrong (value accuracy)
                                new_conf = max(0.1, old_belief.get('confidence', 0.5) - 0.3)
                            else:
                                # Exploiters only slightly reduce confidence when wrong (resist change)
                                new_conf = max(0.2, old_belief.get('confidence', 0.5) - 0.1)

                        # Update the belief
                        self.beliefs[cell] = {'level': blended_level, 'confidence': new_conf}

                # Calculate batch reward - ENTIRELY DIFFERENT for each agent type
                if self.agent_type == "exploratory":
                    if accuracy_scores:
                        # ACCURACY REWARD: Average accuracy across targeted cells
                        # Scale from [0, 1] to [-3, +5] reward range
                        avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
                        batch_reward = avg_accuracy * 8.0 - 3.0  # 1.0 -> +5, 0.0 -> -3
                    else:
                        batch_reward = -1.0
                else:  # exploitative
                    if confirmation_scores:
                        # CONFIRMATION REWARD: Average confirmation across targeted cells
                        # Already in range ~[-1, +1], scale to [-3, +5]
                        avg_confirmation = sum(confirmation_scores) / len(confirmation_scores)
                        batch_reward = avg_confirmation * 4.0 + 1.0  # +1.0 -> +5, -1.0 -> -3
                    else:
                        batch_reward = -1.0

                # Cap the reward range
                batch_reward = max(-3.0, min(5.0, batch_reward))

                total_reward += batch_reward

                self.correct_targets += correct_in_batch
                self.incorrect_targets += incorrect_in_batch

                # Normalize reward to [-1, 1] for Q-learning
                scaled_reward = max(-1.0, min(1.0, batch_reward / 5.0))
                target_trust = (scaled_reward + 1.0) / 2.0  # Map to [0,1] for trust

                # Update Q-table and trust - KEY CHANGE: Adjust learning rates by agent type
                if mode == "self_action":
                    old_q = self.q_table.get("self_action", 0.0)
                    # Explorers learn faster from self-action outcomes
                    effective_learning_rate = self.learning_rate * (1.5 if self.agent_type == "exploratory" else 1.0)
                    new_q = old_q + effective_learning_rate * (scaled_reward - old_q)
                    self.q_table["self_action"] = new_q

                elif source_ids:
                    # Update generic mode Q-value (fixes mode vs source ID mismatch)
                    # mode is what's used in action selection (e.g., "human", "A_0")
                    if mode in self.q_table:
                        old_mode_q = self.q_table[mode]
                        if self.agent_type == "exploratory":
                            effective_learning_rate = self.learning_rate * 1.5
                        else:
                            effective_learning_rate = self.learning_rate
                        new_mode_q = old_mode_q + effective_learning_rate * (scaled_reward - old_mode_q)
                        self.q_table[mode] = new_mode_q

                    # Also update specific source Q-values (for tracking individual sources)
                    for source_id in source_ids:
                        if source_id in self.q_table:
                            old_q = self.q_table[source_id]
                            # Pure Q-learning: same learning rate for all sources
                            # Let feedback naturally teach agents which sources are valuable
                            if self.agent_type == "exploratory":
                                effective_learning_rate = self.learning_rate * 1.5  # Explorers learn faster
                            else:
                                effective_learning_rate = self.learning_rate

                            new_q = old_q + effective_learning_rate * (scaled_reward - old_q)
                            self.q_table[source_id] = new_q

                        if source_id in self.trust:
                            old_trust = self.trust[source_id]

                            # Pure feedback-based trust update: no alignment peeking
                            # Trust naturally increases for sources that lead to good outcomes
                            # and decreases for sources that lead to bad outcomes
                            trust_change = self.trust_learning_rate * (target_trust - old_trust)

                            new_trust = max(0.0, min(1.0, old_trust + trust_change))
                            self.trust[source_id] = new_trust

        except Exception as e:
            print(f"ERROR in Agent {self.unique_id} process_reward at tick {current_tick}: {e}")
            import traceback
            traceback.print_exc()

        return total_reward

    def step(self):
        self.sense_environment()
        self.seek_information()
        self.send_relief()
        reward = self.process_reward()

        self.update_trust_for_accuracy() # Add direct trust update for accurate information
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

    def update_trust_for_accuracy(self):
        """Directly updates trust based on observed accuracy of previous information."""
        # Skip if we don't have both tracking measures
        if not hasattr(self, 'ai_acceptances') or not hasattr(self, 'ai_info_sources'):
            return

        # Check each cell where we accepted AI information
        for cell, tick in list(self.ai_acceptances.items()):
            # Only process recently accepted information (past 5 ticks)
            if self.model.tick - tick > 5:
                # Remove old entries to prevent dict from growing too large
                self.ai_acceptances.pop(cell, None)
                continue

            # Get the AI source that provided this information
            ai_source = self.ai_info_sources.get(cell)
            if not ai_source:
                continue

            # Get the AI's reported value and the actual value
            if cell in self.beliefs and isinstance(self.beliefs[cell], dict):
                believed_level = self.beliefs[cell].get('level', 0)

                # Get actual disaster level
                actual_level = None
                try:
                    if 0 <= cell[0] < self.model.width and 0 <= cell[1] < self.model.height:
                        actual_level = self.model.disaster_grid[cell[0], cell[1]]
                except (IndexError, TypeError):
                    continue

                if actual_level is not None:
                    # Calculate accuracy (how close belief is to reality)
                    accuracy = 1.0 - (abs(believed_level - actual_level) / 5.0)

                    # Apply direct trust update based on accuracy
                    if accuracy > 0.8:  # High accuracy
                        if ai_source in self.trust:
                            old_trust = self.trust[ai_source]

                            # Pure feedback-based trust boost - no alignment peeking
                            boost = 0.05  # Small boost for accurate information

                            # Apply trust update with limits
                            new_trust = min(1.0, old_trust + boost)
                            self.trust[ai_source] = new_trust

                            # Debug logging
                            #if self.model.debug_mode and random.random() < 0.1:
                               # print(f"Agent {self.unique_id} direct trust update for {ai_source}:")
                               # print(f"  - Accuracy: {accuracy:.2f}, Trust: {old_trust:.2f} -> {new_trust:.2f}")


class AIAgent(Agent):
    def __init__(self, unique_id, model):
        super(AIAgent, self).__init__(model)
        self.unique_id = unique_id
        self.model = model
        self.memory = {}
        self.sensed = {}
        self.total_cells = self.model.width * self.model.height
        self.cells_to_sense = int(0.15 * self.total_cells)
        self.sense_environment()

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
                # CRITICAL FIX #3: AI should sense TRUTH accurately
                # Alignment bias happens in report_beliefs(), not during sensing
                # This ensures: alignment=0 reports truth, alignment=1 confirms beliefs
                value = self.model.disaster_grid[x, y]

                self.sensed[cell] = value
                self.memory[(current_tick, cell)] = value

        # Update knowledge map in the model - FIXED INDENTATION
        if hasattr(self.model, 'ai_knowledge_maps'):
            knowledge_map = np.zeros((self.model.width, self.model.height))
            for cell in self.sensed:
                if 0 <= cell[0] < self.model.width and 0 <= cell[1] < self.model.height:
                    knowledge_map[cell[0], cell[1]] = 1
            self.model.ai_knowledge_maps[self.unique_id] = knowledge_map

    def report_beliefs(self, interest_point, query_radius, caller_beliefs, caller_trust_in_ai):
        """
        Reports AI's beliefs about cells within query_radius of interest_point,
        applying alignment based on caller's trust and beliefs.
        """
        report = {}

        # Safety checks for interest_point
        if interest_point is None:
            print(f"WARNING: AI {self.unique_id} received None interest_point in report_beliefs!")
            return {}

        if not (0 <= interest_point[0] < self.model.width and 0 <= interest_point[1] < self.model.height):
            print(f"WARNING: AI {self.unique_id} received out-of-bounds interest_point {interest_point}!")
            adjusted_x = max(0, min(self.model.width-1, interest_point[0]))
            adjusted_y = max(0, min(self.model.height-1, interest_point[1]))
            interest_point = (adjusted_x, adjusted_y)

        # Ensure query_radius is positive
        query_radius = max(1, query_radius)

        # Get cells to report on
        cells_to_report_on = self.model.grid.get_neighborhood(
            interest_point, moore=True, radius=query_radius, include_center=True
        )

        # Debug info
        #if hasattr(self.model, 'debug_mode') and self.model.debug_mode and random.random() < 0.05:
            #print(f"DEBUG: AI {self.unique_id} asked to report on area around {interest_point} with radius {query_radius}")
            #print(f"DEBUG: This gives {len(cells_to_report_on)} potential cells to report on")

        valid_cells_in_query = []
        sensed_vals_list = []
        human_vals_list = []
        human_confidence_list = []

        # First pass: collect all directly sensed values
        directly_sensed_cells = {}
        for cell in cells_to_report_on:
            if 0 <= cell[0] < self.model.width and 0 <= cell[1] < self.model.height:
                # Get AI's sensed value for this cell
                ai_sensed_val = self.sensed.get(cell)
                if ai_sensed_val is not None:
                    directly_sensed_cells[cell] = ai_sensed_val

        # Second pass: for each cell, either use directly sensed value or make an informed guess
        for cell in cells_to_report_on:
            if 0 <= cell[0] < self.model.width and 0 <= cell[1] < self.model.height:
                value_to_use = None
                is_guessed = False

                # If directly sensed, use that value
                if cell in directly_sensed_cells:
                    value_to_use = directly_sensed_cells[cell]
                else:
                    # IMPROVED GUESSING MECHANISM
                    # Increase guessing chance from 30% to 75%
                    if random.random() < 0.75:  # 75% chance to guess
                        # 1. Check nearby cells that are directly sensed
                        nearby_values = []
                        nearby_weights = []

                        # Look at surrounding cells with distance weighting
                        max_search_dist = query_radius + 1
                        for dx in range(-max_search_dist, max_search_dist+1):
                            for dy in range(-max_search_dist, max_search_dist+1):
                                nearby = (cell[0] + dx, cell[1] + dy)
                                if nearby in directly_sensed_cells:
                                    # Calculate distance for weighting (inverse square)
                                    dist = max(1, dx*dx + dy*dy)  # Avoid division by zero
                                    weight = 1.0 / dist

                                    nearby_values.append(directly_sensed_cells[nearby])
                                    nearby_weights.append(weight)

                        # 2. If we have nearby sensed cells, make a weighted guess
                        if nearby_values:
                            # Create a weighted average, rounded to nearest integer
                            total_weight = sum(nearby_weights)
                            weighted_sum = sum(v * w for v, w in zip(nearby_values, nearby_weights))
                            avg_value = weighted_sum / total_weight

                            # Add some noise to the weighted average
                            noise = random.choice([-1, 0, 0, 0, 1])  # Bias toward accuracy
                            guessed_value = int(round(avg_value + noise * 0.5))  # Reduced noise impact
                            value_to_use = max(0, min(5, guessed_value))
                        else:
                            # 3. If no nearby sensed cells, use caller's belief as a hint
                            caller_belief_info = caller_beliefs.get(cell, {'level': None, 'confidence': 0})
                            caller_level = caller_belief_info.get('level')
                            caller_conf = caller_belief_info.get('confidence', 0)

                            if caller_level is not None and caller_conf > 0.3:
                                # If caller has reasonable confidence, use their belief with noise
                                noise = random.choice([-1, 0, 0, 1])
                                value_to_use = max(0, min(5, caller_level + noise))
                            else:
                                # 4. Last resort: make educated random guess based on position
                                # Cells closer to center of disaster tend to have higher values
                                # Use interest_point as a proxy for potential disaster center
                                dist_to_interest = math.sqrt(
                                    (cell[0] - interest_point[0])**2 +
                                    (cell[1] - interest_point[1])**2
                                )

                                # Scale distance to range [0,1] based on query radius
                                scaled_dist = min(1.0, dist_to_interest / (query_radius + 1))

                                # Higher probability of higher values closer to interest point
                                # Further away = more likely to be 0-1
                                # Closer = more likely to be 2-4
                                if scaled_dist < 0.3:  # Very close to interest point
                                    value_to_use = random.choice([2, 3, 3, 4, 4])
                                elif scaled_dist < 0.6:  # Moderately close
                                    value_to_use = random.choice([1, 2, 2, 3, 3])
                                else:  # Far away
                                    value_to_use = random.choice([0, 0, 1, 1, 2])

                        is_guessed = True

                # If we have a value to use (either sensed or guessed), process it
                if value_to_use is not None:
                    valid_cells_in_query.append(cell)
                    sensed_vals_list.append(int(value_to_use))

                    # Track guessed values for debugging
                    #if is_guessed and hasattr(self.model, 'debug_mode') and self.model.debug_mode and random.random() < 0.02:
                        #print(f"DEBUG: AI {self.unique_id} guessed value {value_to_use} for cell {cell}")

                    # Get the CALLER'S belief for this cell (for alignment)
                    caller_belief_info = caller_beliefs.get(cell, {'level': 0, 'confidence': 0.1})
                    human_level = caller_belief_info.get('level', 0)
                    human_confidence = caller_belief_info.get('confidence', 0.1)
                    human_vals_list.append(int(human_level))
                    human_confidence_list.append(human_confidence)

        # If no cells to report on, return empty
        if not valid_cells_in_query:
            if hasattr(self.model, 'debug_mode') and self.model.debug_mode and random.random() < 0.1:
                print(f"DEBUG: AI {self.unique_id} has no data to report!")
            return {}

        # Debug output
        #if hasattr(self.model, 'debug_mode') and self.model.debug_mode and random.random() < 0.1:
            #print(f"DEBUG: AI {self.unique_id} has data for {len(valid_cells_in_query)} cells out of {len(cells_to_report_on)} requested")

        # Convert to numpy arrays for alignment logic
        sensed_vals = np.array(sensed_vals_list)
        human_vals = np.array(human_vals_list)
        human_conf = np.array(human_confidence_list)

        # --- Alignment Logic - Stronger effect ---
        alignment_strength = self.model.ai_alignment_level
        low_trust_amplification = getattr(self.model, 'low_trust_amplification_factor', 0.5)
        clipped_trust = max(0.0, min(1.0, caller_trust_in_ai))

        # If alignment is 0, report pure truth (no adjustments)
        if alignment_strength == 0:
            # Ground truth - no adjustments at all
            corrected = sensed_vals
        else:
            # Calculate adjustments based on alignment level
            alignment_factors = alignment_strength * (1.0 + human_conf * 2.0)

            # Add trust-based effect
            alignment_factors += alignment_strength * low_trust_amplification * (1.0 - clipped_trust)

            # Cap the maximum alignment factor
            alignment_factors = np.clip(alignment_factors, 0.0, 3.0)

            # Calculate the difference between human beliefs and AI sensed values
            belief_differences = human_vals - sensed_vals

            # Apply proportional adjustments based on alignment factors
            adjustments = alignment_factors * belief_differences

            # Apply adjustments to sensed values
            corrected = np.round(sensed_vals + adjustments)
            corrected = np.clip(corrected, 0, 5)  # Keep values in valid range

        # Build the report dictionary with aligned values
        for i, cell in enumerate(valid_cells_in_query):
            report[cell] = int(corrected[i])

        return report

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
        self.info_diversity_data = []  # Information Diversity (Shannon Entropy)

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

        # Construct social network
        components = self.initialize_social_network()  # Call only once and store result
        num_components = len(components)  # Get the count from the returned components list

        if num_components < 2:
            print("WARNING: Failed to create multiple components in social network")
        else:
            print(f"Successfully created social network with {num_components} components")

        self.agent_list = []
        self.humans = {}

        # Debugging mode
        self.debug_mode = False  # Set to False for production runs

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

        # validate_social_network(self, save_dir="analysis_plots") #debug

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
        for ai_agent in self.ais.values():
            # Make sure all AI agents sense the environment
            ai_agent.sense_environment()

        # Init AI knowledge
        self.initialize_ai_knowledge_maps()

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



    def debug_log(self, message, force=False):
        """Log debug messages if debug mode is enabled or forced."""
        #if self.debug_mode or force:
           # print(f"[DEBUG] Tick {self.tick}: {message}")

    def calculate_aeci_variance(self):
        """Calculate AI Echo Chamber Index variance on a [-1, +1] scale."""
        print(f"\nDEBUG: Starting AECI variance calculation at tick {self.tick}")
        
        aeci_variance = 0.0  # Default neutral value
        
        # Define AI-reliant agents with adjusted threshold for observed behavior
        # With equal +0.2 biases for human/ai and friend selection effects,
        # agents query ~70% human / ~30% AI in practice (not 50/50)
        # Threshold set to 25% to capture agents using AI significantly
        min_calls_threshold = 10   # Need stable sample size
        min_ai_ratio = 0.25        # Above 25% indicates meaningful AI usage
        
        ai_reliant_agents = []
        for agent in self.humans.values():
            # Safety checks for valid counters
            if not hasattr(agent, 'accum_calls_total') or not hasattr(agent, 'accum_calls_ai'):
                continue
                
            total_calls = max(1, agent.accum_calls_total)  # Prevent division by zero
            ai_ratio = agent.accum_calls_ai / total_calls
            
            if total_calls >= min_calls_threshold and ai_ratio >= min_ai_ratio:
                ai_reliant_agents.append(agent)
        
        # Debug print
        print(f"  Found {len(ai_reliant_agents)}/{len(self.humans)} AI-reliant agents")
        
        # Get global belief variance
        all_beliefs = []
        for agent in self.humans.values():
            for belief_info in agent.beliefs.values():
                if isinstance(belief_info, dict):
                    level = belief_info.get('level', 0)
                    if not np.isnan(level):  # Filter out NaN values
                        all_beliefs.append(level)
        
        # Calculate global variance with safety check
        if len(all_beliefs) > 1:
            global_var = np.var(all_beliefs)
            print(f"  Global belief variance: {global_var:.4f}")
        else:
            global_var = 0.0
            print("  WARNING: Not enough global beliefs to calculate variance")
            
        # Only proceed if we have a valid global variance and AI-reliant agents
        if global_var > 0 and ai_reliant_agents:
            # Get AI-reliant agents' beliefs
            ai_reliant_beliefs = []
            for agent in ai_reliant_agents:
                for belief_info in agent.beliefs.values():
                    if isinstance(belief_info, dict):
                        level = belief_info.get('level', 0)
                        if not np.isnan(level):  # Filter out NaN values
                            ai_reliant_beliefs.append(level)
            
            # Calculate AI-reliant variance with safety check
            if len(ai_reliant_beliefs) > 1:
                ai_reliant_var = np.var(ai_reliant_beliefs)
                print(f"  AI-reliant beliefs variance: {ai_reliant_var:.4f}")
                
                # Calculate variance effect
                # Negative means AI reduces variance (echo chamber)
                # Positive means AI increases variance (diversification)
                var_diff = ai_reliant_var - global_var
                
                # Normalize to [-1, +1] range
                if var_diff < 0:  # Variance reduction (echo chamber)
                    aeci_variance = max(-1, var_diff / global_var)  # Normalize by global variance
                else:  # Variance increase (diversification)
                    # Find a reasonable upper bound for normalization
                    max_possible_var = 5.0  # Given belief levels are 0-5, max variance is around 5
                    aeci_variance = min(1, var_diff / (max_possible_var - global_var))
                
                print(f"  AECI variance effect: {aeci_variance:.4f}")
            else:
                print("  WARNING: Not enough AI-reliant beliefs to calculate variance")
        else:
            print(f"  WARNING: Invalid global variance ({global_var}) or no AI-reliant agents")
        
        # Create a CORRECTLY formatted tuple
        aeci_variance_tuple = (self.tick, aeci_variance)
        print(f"  Returning AECI variance tuple: {aeci_variance_tuple}")
        
        # Update metrics dictionary with consistent format
        self._last_metrics['aeci_variance'] = {
            'tick': self.tick,
            'value': aeci_variance
        }
        
        # Store in the array with consistent format
        self.aeci_variance_data.append(aeci_variance_tuple)
        
        # Debug: print all values in the array
        print(f"  Current aeci_variance_data length: {len(self.aeci_variance_data)}")
        print(f"  Last 3 entries: {self.aeci_variance_data[-3:] if len(self.aeci_variance_data) >= 3 else self.aeci_variance_data}")
        
        return aeci_variance_tuple

    def calculate_info_diversity(self):
        """Calculate Information Diversity (Shannon Entropy of source usage).

        Returns tuple: (tick, exploit_entropy, explor_entropy)
        - Higher entropy = diverse information sources (anti-bubble)
        - Lower entropy = concentrated sources (echo chamber)
        - Range: 0 (single source) to log2(num_sources) (perfectly distributed)
        """
        from math import log2

        # Separate agents by type
        exploitative_agents = [a for a in self.humans.values() if a.agent_type == "exploitative"]
        exploratory_agents = [a for a in self.humans.values() if a.agent_type == "exploratory"]

        def calculate_entropy(agents):
            """Calculate Shannon entropy of source usage for a group of agents."""
            if not agents:
                return 0.0

            # Count source usage across all agents in group
            source_counts = {}
            total_queries = 0

            for agent in agents:
                # Check if agent has tracked source usage
                if hasattr(agent, 'tokens_this_tick'):
                    for source_mode in agent.tokens_this_tick.keys():
                        source_counts[source_mode] = source_counts.get(source_mode, 0) + 1
                        total_queries += 1

            # If no queries made, return 0 (no diversity)
            if total_queries == 0:
                return 0.0

            # Calculate Shannon entropy: H = -sum(p_i * log2(p_i))
            entropy = 0.0
            for count in source_counts.values():
                if count > 0:
                    p = count / total_queries
                    entropy -= p * log2(p)

            return entropy

        exploit_entropy = calculate_entropy(exploitative_agents)
        explor_entropy = calculate_entropy(exploratory_agents)

        return (self.tick, exploit_entropy, explor_entropy)

    def initialize_social_network(self):
        """Initialize social network with multiple components and meaningful homophily."""
        # Calculate how many exploitative and exploratory agents we have
        num_exploitative = int(self.num_humans * self.share_exploitative)
        num_exploratory = self.num_humans - num_exploitative

        print(f"Initializing social network with {num_exploitative} exploitative and {num_exploratory} exploratory agents")

        # Create an empty graph
        self.social_network = nx.Graph()

        # Add all nodes
        for i in range(self.num_humans):
            self.social_network.add_node(i)

        # Determine number of communities (2-3 for typical agent counts)
        num_communities = min(3, max(2, self.num_humans // 15))
        print(f"Creating {num_communities} substantial communities")

        # Divide agents into communities
        # Ensure each community has at least 5 agents (or appropriate minimum)
        min_community_size = max(5, self.num_humans // (num_communities * 2))

        all_agents = list(range(self.num_humans))
        random.shuffle(all_agents)  # Randomize agent assignment

        # Create the communities with minimum size constraints
        communities = []
        remaining_agents = all_agents.copy()

        for i in range(num_communities - 1):  # Allocate all but the last community
            # Ensure we leave enough agents for remaining communities
            agents_needed = min_community_size * (num_communities - i)
            max_size = len(remaining_agents) - agents_needed + min_community_size

            # Select size between min and max
            size = random.randint(min_community_size, max(min_community_size, max_size))

            # Create community
            community = remaining_agents[:size]
            communities.append(community)

            # Remove assigned agents
            remaining_agents = remaining_agents[size:]

        # Last community gets all remaining agents
        if remaining_agents:
            communities.append(remaining_agents)

        # Ensure type diversity within each community (some mixture of types)
        for community in communities:
            exploit_count = sum(1 for a in community if a < num_exploitative)
            explor_count = len(community) - exploit_count

            # If community is heavily biased to one type, adjust it
            if exploit_count == 0 and num_exploitative > 0:
                # Add at least one exploitative agent
                exploit_to_add = min(2, num_exploitative)
                for _ in range(exploit_to_add):
                    # Find an exploitative agent from another community with many exploit agents
                    for other_comm in communities:
                        if other_comm != community:
                            other_exploit = [a for a in other_comm if a < num_exploitative]
                            if len(other_exploit) > 2:  # Can spare one
                                agent_to_move = random.choice(other_exploit)
                                community.append(agent_to_move)
                                other_comm.remove(agent_to_move)
                                break

            elif explor_count == 0 and num_exploratory > 0:
                # Add at least one exploratory agent
                explor_to_add = min(2, num_exploratory)
                for _ in range(explor_to_add):
                    # Similar logic for exploratory agents
                    for other_comm in communities:
                        if other_comm != community:
                            other_explor = [a for a in other_comm if a >= num_exploitative]
                            if len(other_explor) > 2:  # Can spare one
                                agent_to_move = random.choice(other_explor)
                                community.append(agent_to_move)
                                other_comm.remove(agent_to_move)
                                break

        # Report community compositions
        print(f"Created {len(communities)} communities:")
        for i, community in enumerate(communities):
            exploit_count = sum(1 for a in community if a < num_exploitative)
            explor_count = len(community) - exploit_count
            print(f"  Community {i}: {len(community)} agents ({exploit_count} exploit, {explor_count} explor)")

        # Create dense connections WITHIN communities
        p_within = 0.7  # 70% connection probability within community

        for comm_idx, community in enumerate(communities):
            print(f"  Adding within-community edges for community {comm_idx} (size {len(community)})")
            for i in range(len(community)):
                for j in range(i+1, len(community)):
                    if random.random() < p_within:
                        self.social_network.add_edge(community[i], community[j])

        # NO connections BETWEEN communities to ensure separate components

        # Final verification
        components = list(nx.connected_components(self.social_network))
        print(f"Final network: {len(components)} connected components")
        for i, comp in enumerate(components):
            comp_size = len(comp)
            comp_exploit_count = sum(1 for a in comp if a < num_exploitative)
            comp_explor_count = comp_size - comp_exploit_count
            print(f"  Component {i}: {comp_size} agents ({comp_exploit_count} exploit, {comp_explor_count} explor)")

            # Check for too-small components
            if comp_size < 3:
                print(f"    WARNING: Component {i} is too small ({comp_size} agents)")

                # Find the largest component
                largest_comp = max(components, key=len)

                # Connect this small component to the largest one
                if comp != largest_comp:
                    small_node = random.choice(list(comp))
                    large_node = random.choice(list(largest_comp))
                    self.social_network.add_edge(small_node, large_node)
                    print(f"    Connected small component node {small_node} to large component node {large_node}")

        # Final check after fixes
        components = list(nx.connected_components(self.social_network))
        print(f"Final network after fixes: {len(components)} connected components")
        for i, comp in enumerate(components):
            comp_size = len(comp)
            comp_exploit_count = sum(1 for a in comp if a < num_exploitative)
            comp_explor_count = comp_size - comp_exploit_count
            print(f"  Component {i}: {comp_size} agents ({comp_exploit_count} exploit, {comp_explor_count} explor)")

        return components

    def initialize_ai_knowledge_maps(self):
        """Initialize maps showing which areas each AI agent has knowledge about."""
        self.ai_knowledge_maps = {}

        for ai_id, ai_agent in self.ais.items():
            # Create a binary knowledge map (0=no knowledge, 1=has knowledge)
            knowledge_map = np.zeros((self.width, self.height))

            # Mark cells that the AI has sensed
            for cell in ai_agent.sensed:
                if 0 <= cell[0] < self.width and 0 <= cell[1] < self.height:
                    knowledge_map[cell[0], cell[1]] = 1

            # Store the knowledge map
            self.ai_knowledge_maps[ai_id] = knowledge_map

            #if self.debug_mode:
                #coverage = np.sum(knowledge_map) / (self.width * self.height) * 100
                #print(f"DEBUG: {ai_id} has knowledge of {coverage:.1f}% of the grid")

    def track_ai_usage_patterns(model, tick_interval=10, save_dir="analysis_plots"):
        """Track and plot AI usage patterns over time with proper bounds."""
        os.makedirs(save_dir, exist_ok=True)

        if model.tick % tick_interval != 0:
            return  # Only run at specified intervals

        # Gather data on AI usage ratio
        exploit_ai_ratio = []
        explor_ai_ratio = []

        for agent in model.humans.values():
            if agent.accum_calls_total > 0:
                # FIXED: Ensure ratio is properly bounded
                ai_ratio = max(0.0, min(1.0, agent.accum_calls_ai / agent.accum_calls_total))
                if agent.agent_type == "exploitative":
                    exploit_ai_ratio.append(ai_ratio)
                else:
                    explor_ai_ratio.append(ai_ratio)

        # Save data for later analysis
        if not hasattr(model, 'ai_usage_history'):
            model.ai_usage_history = {'tick': [], 'exploit_mean': [], 'exploit_std': [],
                                    'explor_mean': [], 'explor_std': []}

        # FIXED: Ensure means and bounds are properly calculated and clipped
        model.ai_usage_history['tick'].append(model.tick)

        # Calculate exploit stats with bounds
        if exploit_ai_ratio:
            exploit_mean = np.mean(exploit_ai_ratio)
            exploit_std = np.std(exploit_ai_ratio)
        else:
            exploit_mean = 0
            exploit_std = 0

        # Calculate explor stats with bounds
        if explor_ai_ratio:
            explor_mean = np.mean(explor_ai_ratio)
            explor_std = np.std(explor_ai_ratio)
        else:
            explor_mean = 0
            explor_std = 0

        # Ensure all values are properly bounded
        exploit_mean = max(0.0, min(1.0, exploit_mean))
        explor_mean = max(0.0, min(1.0, explor_mean))

        # Cap standard deviations to prevent out-of-bounds fill areas
        exploit_std = min(exploit_std, min(exploit_mean, 1.0 - exploit_mean))
        explor_std = min(explor_std, min(explor_mean, 1.0 - explor_mean))

        model.ai_usage_history['exploit_mean'].append(exploit_mean)
        model.ai_usage_history['exploit_std'].append(exploit_std)
        model.ai_usage_history['explor_mean'].append(explor_mean)
        model.ai_usage_history['explor_std'].append(explor_std)

        # Plot current state with proper bounds
        if model.tick > 0 and len(model.ai_usage_history['tick']) > 1:
            fig, ax = plt.subplots(figsize=(10, 6))

            # FIXED: Ensure fill_between values stay within bounds
            exploit_mean_array = np.array(model.ai_usage_history['exploit_mean'])
            exploit_std_array = np.array(model.ai_usage_history['exploit_std'])
            explor_mean_array = np.array(model.ai_usage_history['explor_mean'])
            explor_std_array = np.array(model.ai_usage_history['explor_std'])

            # Clip bounds to ensure they stay within [0,1]
            exploit_lower = np.clip(exploit_mean_array - exploit_std_array, 0.0, 1.0)
            exploit_upper = np.clip(exploit_mean_array + exploit_std_array, 0.0, 1.0)
            explor_lower = np.clip(explor_mean_array - explor_std_array, 0.0, 1.0)
            explor_upper = np.clip(explor_mean_array + explor_std_array, 0.0, 1.0)

            ax.plot(model.ai_usage_history['tick'], exploit_mean_array,
                    'r-', label='Exploitative')
            ax.fill_between(model.ai_usage_history['tick'],
                            exploit_lower, exploit_upper,
                            color='r', alpha=0.2)

            ax.plot(model.ai_usage_history['tick'], explor_mean_array,
                    'b-', label='Exploratory')
            ax.fill_between(model.ai_usage_history['tick'],
                            explor_lower, explor_upper,
                            color='b', alpha=0.2)

            ax.set_xlabel('Tick')
            ax.set_ylabel('AI Usage Ratio')
            ax.set_title(f'AI Usage Over Time (Alignment = {model.ai_alignment_level})')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)

            # Set explicit y-axis limits
            ax.set_ylim(0, 1)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"ai_usage_tick_{model.tick}.png"))
            plt.close()

    def update_disaster(self):
        """
        Update disaster grid based on disaster_dynamics parameter.

        disaster_dynamics:
        0 = Static (no updates)
        1 = Slow evolution (5% chance per tick, smaller magnitude)
        2 = Medium evolution (10% chance per tick, medium magnitude)
        3 = Rapid evolution (20% chance per tick, larger magnitude)
        """
        # Store a copy of the current grid before updating
        if self.disaster_grid is not None:
            self.previous_grid = self.disaster_grid.copy()
        else:
            self.previous_grid = None

        # Apply disaster dynamics based on parameter
        if self.disaster_dynamics == 0:
            # Static disaster - no updates
            pass

        elif self.disaster_dynamics == 1:
            # Slow evolution: 5% chance, +1-2 magnitude
            if random.random() < 0.05:
                x, y = np.random.randint(0, self.disaster_grid.shape[0]), np.random.randint(0, self.disaster_grid.shape[1])
                magnitude = random.randint(1, 2)
                self.disaster_grid[x, y] = min(5, self.disaster_grid[x, y] + magnitude)

        elif self.disaster_dynamics == 2:
            # Medium evolution: 10% chance, +2-3 magnitude
            if random.random() < 0.1:
                x, y = np.random.randint(0, self.disaster_grid.shape[0]), np.random.randint(0, self.disaster_grid.shape[1])
                magnitude = random.randint(2, 3)
                self.disaster_grid[x, y] = min(5, self.disaster_grid[x, y] + magnitude)

        elif self.disaster_dynamics == 3:
            # Rapid evolution: 20% chance, +3-4 magnitude
            if random.random() < 0.2:
                x, y = np.random.randint(0, self.disaster_grid.shape[0]), np.random.randint(0, self.disaster_grid.shape[1])
                magnitude = random.randint(3, 4)
                self.disaster_grid[x, y] = min(5, self.disaster_grid[x, y] + magnitude)

        # Detect significant changes
        if self.previous_grid is not None:
            grid_change = np.abs(self.disaster_grid - self.previous_grid)
            max_change = np.max(grid_change)
            if max_change >= self.event_threshold:
                self.event_ticks.append(self.tick)
                # print(f"Tick {self.tick}: Significant disaster event detected (max change: {max_change})")

    def update_ai_knowledge_maps(self):
        """Update knowledge maps after AIs have sensed the environment."""
        if hasattr(self, 'ai_knowledge_maps'):
            for ai_id, ai_agent in self.ais.items():
                knowledge_map = np.zeros((self.width, self.height))
                for cell in ai_agent.sensed:
                    if 0 <= cell[0] < self.width and 0 <= cell[1] < self.height:
                        knowledge_map[cell[0], cell[1]] = 1
                self.ai_knowledge_maps[ai_id] = knowledge_map

    def step(self):
        self.tick += 1
        self.tokens_this_tick = {}

        # Clear all agents' source tracking from previous tick
        # This must happen BEFORE agents run, so they can populate it fresh
        # and BEFORE data collection at the end of the tick reads it
        for agent in self.agent_list:
            if isinstance(agent, HumanAgent):
                agent.tokens_this_tick = {}

        self.update_disaster()
        random.shuffle(self.agent_list)

        total_reward_exploit = 0
        total_reward_explor = 0
        for agent in self.agent_list:
            r = agent.step()  # Process rewards and return numeric reward.
            if isinstance(agent, HumanAgent):
                if r is None:
                    print(f"ERROR: Agent {agent.unique_id} returned None from process_reward")
                reward = r if r is not None else 0  # Fallback to 0 if None
                if agent.agent_type == "exploitative":
                    total_reward_exploit += reward
                else:
                    total_reward_explor += reward
        self.rewards_data.append((total_reward_exploit, total_reward_explor))

        #update AI knowledge
        self.update_ai_knowledge_maps()

        # Create a proper token array counting all tokens
        token_array = np.zeros((self.width, self.height), dtype=int)

        # Fill the token array
        for pos, count_dict in self.tokens_this_tick.items():
            x, y = pos
            # Check bounds before accessing array
            if 0 <= x < self.width and 0 <= y < self.height:
                # Get token counts by type
                exploit_tokens = count_dict.get('exploit', 0)
                explor_tokens = count_dict.get('explor', 0)
                # Assign the sum to the token array
                token_array[x, y] = exploit_tokens + explor_tokens

        # Identify high-need cells (L4+) that received no tokens
        need_mask = self.disaster_grid >= 4
        tokens_mask = token_array == 0
        unmet = np.sum(need_mask & tokens_mask)
        self.unmet_needs_evolution.append(unmet)

        if self.tick % 10 == 0:
            self.track_ai_usage_patterns()

        if self.tick % 10 == 0:
            try:
                track_component_seci_evolution(self, tick_interval=10)
            except Exception as e:
                if self.debug_mode:
                    print(f"Error in track_component_seci_evolution: {e}")

        # Initialize metric storage to track last calculated values
        if not hasattr(self, '_last_metrics'):
            self._last_metrics = {
                'seci': {'tick': 0, 'exploit': 0, 'explor': 0},
                'aeci': {'tick': 0, 'exploit': 0, 'explor': 0},
                'retain_aeci': {'tick': 0, 'exploit': 0, 'explor': 0},
                'retain_seci': {'tick': 0, 'exploit': 0, 'explor': 0},
                'belief_error': {'tick': 0, 'exploit': 0, 'explor': 0},
                'belief_variance': {'tick': 0, 'exploit': 0, 'explor': 0},
                'component_seci': {'tick': 0, 'value': 0},
                'aeci_variance': {'tick': 0, 'value': 0},
                'component_aeci': {'tick': 0, 'value': 0},
                'component_ai_trust_variance': {'tick': 0, 'value': 0},
                'trust_stats': {'tick': 0, 'ai_exp': 0, 'friend_exp': 0, 'nonfriend_exp': 0,
                              'ai_expl': 0, 'friend_expl': 0, 'nonfriend_expl': 0},
                'running_aeci': {'tick': 0, 'exploit': 0, 'explor': 0},
                'info_diversity': {'tick': 0, 'exploit': 0, 'explor': 0}
            }

        # Every 5 ticks, compute additional metrics.
        if self.tick % 5 == 0:
            # --- SECI Calculation ---
            all_belief_levels = []

            # Ensure we have beliefs to process
            for agent in self.humans.values():
                if agent.beliefs:  # Check if any beliefs exist
                    for cell, belief_info in agent.beliefs.items():
                        if isinstance(belief_info, dict):
                            level = belief_info.get('level', 0)
                            all_belief_levels.append(level)

            # Compute global variance with safety check
            if len(all_belief_levels) > 1:
                global_var = np.var(all_belief_levels)
            else:
                global_var = 1e-6  # Default to small value if insufficient data

            # Initialize metric lists
            seci_exp_list = []
            seci_expl_list = []

            # More robust friend belief collection and SECI calculation
            for agent in self.humans.values():
                # Collect friend beliefs with proper checks
                friend_belief_levels = []
                for fid in agent.friends:
                    friend = self.humans.get(fid)
                    if friend and friend.beliefs:
                        for cell, belief_info in friend.beliefs.items():
                            if isinstance(belief_info, dict):
                                level = belief_info.get('level', 0)
                                friend_belief_levels.append(level)

                # Calculate friend variance only if we have enough data
                if len(friend_belief_levels) > 1:
                    friend_var = np.var(friend_belief_levels)
                else:
                    friend_var = global_var  # Default to global var if insufficient friend data

                # Calculate SECI safely
                if global_var > 1e-9:  # Ensure non-zero denominator
                    var_diff = friend_var - global_var
                    
                    if var_diff < 0:  # Variance reduction (echo chamber)
                        seci_val = max(-1, var_diff / global_var)
                    else:  # Variance increase (diversification)
                        max_possible_var = 5.0  # Upper bound for normalization
                        seci_val = min(1, var_diff / (max_possible_var - global_var))
                else:
                    seci_val = 0  # Default if global variance is essentially zero                # Store by agent type
                
                if agent.agent_type == "exploitative":
                    seci_exp_list.append(seci_val)
                else:
                    seci_expl_list.append(seci_val)

            # Store results with proper checks
            if seci_exp_list or seci_expl_list:  # Only store if we have data
                seci_exploit_mean = np.mean(seci_exp_list) if seci_exp_list else 0
                seci_explor_mean = np.mean(seci_expl_list) if seci_expl_list else 0

                # Update the last metrics dictionary
                self._last_metrics['seci'] = {
                    'tick': self.tick,
                    'exploit': seci_exploit_mean,
                    'explor': seci_explor_mean
                }

                self.seci_data.append((self.tick, seci_exploit_mean, seci_explor_mean))
            else:
                # Add default values if no data
                self.seci_data.append((self.tick, 0, 0))

            # --- AECI-Variance (AI Echo Chamber Index) ---
            aeci_variance_result = self.calculate_aeci_variance()

            # --- Component-AECI ---
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

            component_aeci_mean = np.mean(component_aeci_list) if component_aeci_list else 0

            # Update last metrics
            self._last_metrics['component_aeci'] = {
                'tick': self.tick,
                'value': component_aeci_mean
            }

            self.component_aeci_data.append((self.tick, component_aeci_mean))

            # --- Component SECI ---
            component_seci_list = []
            
            for component_nodes in nx.connected_components(self.social_network):
                if len(component_nodes) <= 1:
                    continue  # Skip components with only 1 node
                    
                # Get all beliefs from this component
                component_belief_levels = []
                for node_id in component_nodes:
                    agent_id = f"H_{node_id}"
                    agent = self.humans.get(agent_id)
                    if agent and agent.beliefs:
                        for cell, belief_info in agent.beliefs.items():
                            if isinstance(belief_info, dict):
                                level = belief_info.get('level', 0)
                                component_belief_levels.append(level)
                
                # Only calculate SECI if we have beliefs
                if len(component_belief_levels) > 0:
                    # Calculate global variance (all agents)
                    all_belief_levels = []
                    for agent in self.humans.values():
                        for cell, belief_info in agent.beliefs.items():
                            if isinstance(belief_info, dict):
                                level = belief_info.get('level', 0)
                                all_belief_levels.append(level)
                                
                    # More robust variance calculation
                    global_var = np.var(all_belief_levels) if len(all_belief_levels) > 1 else 1e-6
                    component_var = np.var(component_belief_levels) if len(component_belief_levels) > 1 else global_var
                    
                    # Calculate component SECI more robustly
                    if global_var > 1e-9:  # Avoid division by zero with small threshold
                        component_seci_val = max(0, min(1, (global_var - component_var) / global_var))
                        component_seci_list.append(component_seci_val)
                    else:
                        component_seci_list.append(0)
                        
            # Calculate mean component SECI and store
            component_seci_mean = np.mean(component_seci_list) if component_seci_list else 0

            # Update the last metrics dictionary and store in model data
            self._last_metrics['component_seci'] = {
                'tick': self.tick,
                'value': component_seci_mean
            }

            self.component_seci_data.append((self.tick, component_seci_mean))
            
            # --- Component AI Trust Variance ---
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

            component_ai_trust_var_mean = np.mean(component_ai_trust_var_list) if component_ai_trust_var_list else 0

            # Update last metrics
            self._last_metrics['component_ai_trust_variance'] = {
                'tick': self.tick,
                'value': component_ai_trust_var_mean
            }

            self.component_ai_trust_variance_data.append((self.tick, component_ai_trust_var_mean))

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
                            belief_info = agent.beliefs.get(cell, {})
                            if isinstance(belief_info, dict):
                                belief = belief_info.get('level', -1)
                                if belief != -1:
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

            # Update last metrics
            self._last_metrics['belief_error'] = {
                'tick': self.tick,
                'exploit': avg_mae_exploit,
                'explor': avg_mae_explor
            }

            self.belief_error_data.append((self.tick, avg_mae_exploit, avg_mae_explor))

            # --- Within-Type Belief Variance ---
            exploit_beliefs_levels = []
            explor_beliefs_levels = []
            for agent in self.humans.values():
                target_list = exploit_beliefs_levels if agent.agent_type == "exploitative" else explor_beliefs_levels
                for belief_info in agent.beliefs.values():
                    if isinstance(belief_info, dict):
                        target_list.append(belief_info.get('level', 0))

            var_exploit = np.var(exploit_beliefs_levels) if exploit_beliefs_levels else 0
            var_explor = np.var(explor_beliefs_levels) if explor_beliefs_levels else 0

            # Update last metrics
            self._last_metrics['belief_variance'] = {
                'tick': self.tick,
                'exploit': var_exploit,
                'explor': var_explor
            }

            self.belief_variance_data.append((self.tick, var_exploit, var_explor))

            # --- AECI Calculation ---
            aeci_exp = []
            aeci_expl = []

            for agent in self.humans.values():
                # Make sure counters are valid
                if not hasattr(agent, 'accum_calls_ai') or not hasattr(agent, 'accum_calls_total'):
                    if self.debug_mode:
                        print(f"Warning: Agent {agent.unique_id} missing call counters")
                    continue

                # Ensure no negative values
                agent.accum_calls_ai = max(0, agent.accum_calls_ai)
                agent.accum_calls_total = max(0, agent.accum_calls_total)

                # Calculate AECI (AI Query Ratio) with robust error handling
                # NOTE: AECI measures proportion of QUERIES to AI, not acceptances
                # High AECI = agent frequently queries AI (regardless of whether they accept the info)
                if agent.accum_calls_total > 0:
                    # Ensure ratio is properly bounded between 0 and 1
                    ratio = max(0.0, min(1.0, agent.accum_calls_ai / agent.accum_calls_total))

                    if agent.agent_type == "exploitative":
                        aeci_exp.append(ratio)
                    else:  # exploratory
                        aeci_expl.append(ratio)

            # Calculate means with added safety
            avg_aeci_exp = np.mean(aeci_exp) if aeci_exp else 0.0
            avg_aeci_expl = np.mean(aeci_expl) if aeci_expl else 0.0

            # Ensure averages are also properly bounded
            avg_aeci_exp = max(0.0, min(1.0, avg_aeci_exp))
            avg_aeci_expl = max(0.0, min(1.0, avg_aeci_expl))

            # Update last metrics
            self._last_metrics['aeci'] = {
                'tick': self.tick,
                'exploit': avg_aeci_exp,
                'explor': avg_aeci_expl
            }

            self.aeci_data.append((self.tick, avg_aeci_exp, avg_aeci_expl))

            # Calculate running AECI (exponential moving average)
            if not hasattr(self, 'running_aeci_exp') or self.running_aeci_exp is None:
                # Initialize if not present
                self.running_aeci_exp = avg_aeci_exp
                self.running_aeci_expl = avg_aeci_expl
            else:
                # Apply exponential moving average with safety bounds
                self.running_aeci_exp = max(0.0, min(1.0, self.running_aeci_exp * 0.8 + avg_aeci_exp * 0.2))
                self.running_aeci_expl = max(0.0, min(1.0, self.running_aeci_expl * 0.8 + avg_aeci_expl * 0.2))

            # Update last metrics
            self._last_metrics['running_aeci'] = {
                'tick': self.tick,
                'exploit': self.running_aeci_exp,
                'explor': self.running_aeci_expl
            }

            self.running_aeci_data.append((self.tick, self.running_aeci_exp, self.running_aeci_expl))

            # --- Information Diversity (Shannon Entropy) ---
            info_diversity_result = self.calculate_info_diversity()
            self.info_diversity_data.append(info_diversity_result)

            # Update last metrics
            self._last_metrics['info_diversity'] = {
                'tick': info_diversity_result[0],
                'exploit': info_diversity_result[1],
                'explor': info_diversity_result[2]
            }

            # --- Retainment Metrics ---
            retain_aeci_exp_list = []
            retain_aeci_expl_list = []
            retain_seci_exp_list = []
            retain_seci_expl_list = []

            if self.tick % 10 == 0 and self.debug_mode:
                print(f"\n--- Retainment Diagnostics at Tick {self.tick} ---")
                for agent_id, agent in list(self.humans.items())[:3]:  # Sample first 3 agents
                    print(f"Agent {agent_id} ({agent.agent_type}) retainment stats:")
                    print(f"  accepted_human: {agent.accepted_human}, accepted_friend: {agent.accepted_friend}, accepted_ai: {agent.accepted_ai}")
                    total = agent.accepted_human + agent.accepted_ai
                    if total > 0:
                        print(f"  retain_aeci: {agent.accepted_ai/total:.2f}, retain_seci: {agent.accepted_friend/total:.2f}")
                    else:
                        print("  No acceptances recorded")

            for agent in self.humans.values():
                total_accepted = agent.accepted_human + agent.accepted_ai
                total_accepted = total_accepted if total_accepted > 0 else 1

                # CLARIFICATION: retain_aeci = AI ACCEPTANCE ratio (accepted from AI / total accepted)
                # This is different from AECI which measures QUERY ratio (queries to AI / total queries)
                # retain_aeci shows actual AI INFLUENCE on beliefs
                retain_aeci_val = agent.accepted_ai / total_accepted if total_accepted > 0 else 0
                retain_seci_val = agent.accepted_friend / total_accepted if total_accepted > 0 else 0


                if agent.agent_type == "exploitative":
                    retain_aeci_exp_list.append(retain_aeci_val)
                    retain_seci_exp_list.append(retain_seci_val)
                else:
                    retain_aeci_expl_list.append(retain_aeci_val)
                    retain_seci_expl_list.append(retain_seci_val)

            retain_aeci_exp_mean = np.mean(retain_aeci_exp_list) if retain_aeci_exp_list else 0
            retain_aeci_expl_mean = np.mean(retain_aeci_expl_list) if retain_aeci_expl_list else 0

            # Update last metrics
            self._last_metrics['retain_aeci'] = {
                'tick': self.tick,
                'exploit': retain_aeci_exp_mean,
                'explor': retain_aeci_expl_mean
            }

            self.retain_aeci_data.append((self.tick, retain_aeci_exp_mean, retain_aeci_expl_mean))

            retain_seci_exp_mean = np.mean(retain_seci_exp_list) if retain_seci_exp_list else 0
            retain_seci_expl_mean = np.mean(retain_seci_expl_list) if retain_seci_expl_list else 0

            # Update last metrics
            self._last_metrics['retain_seci'] = {
                'tick': self.tick,
                'exploit': retain_seci_exp_mean,
                'explor': retain_seci_expl_mean
            }

            self.retain_seci_data.append((self.tick, retain_seci_exp_mean, retain_seci_expl_mean))

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

            # Update last metrics
            self._last_metrics['trust_stats'] = {
                'tick': self.tick,
                'ai_exp': ai_exp_mean,
                'friend_exp': friend_exp_mean,
                'nonfriend_exp': nonfriend_exp_mean,
                'ai_expl': ai_expl_mean,
                'friend_expl': friend_expl_mean,
                'nonfriend_expl': nonfriend_expl_mean
            }

            self.trust_stats.append((self.tick, ai_exp_mean, friend_exp_mean, nonfriend_exp_mean,
                                    ai_expl_mean, friend_expl_mean, nonfriend_expl_mean))

            # DON'T reset call counters - they should accumulate over the full run
            # AI-reliant detection needs min_calls_threshold=10, so counters must accumulate
            # Only reset acceptance counters which are used for retainment metrics
            for agent in self.humans.values():
                # Keep accum_calls_* to accumulate (for AI-reliant detection)
                # Reset only acceptance counters (for per-period retainment metrics)
                agent.accepted_friend = 0
                agent.accepted_human = 0
                agent.accepted_ai = 0
        else:
            # For ticks between calculations, use the last calculated values

            # For SECI - use last calculated values
            self.seci_data.append((
                self.tick,
                self._last_metrics['seci']['exploit'],
                self._last_metrics['seci']['explor']
            ))

            # For AECI - use last calculated values
            self.aeci_data.append((
                self.tick,
                self._last_metrics['aeci']['exploit'],
                self._last_metrics['aeci']['explor']
            ))

            # For info_diversity - use last calculated values
            self.info_diversity_data.append((
                self.tick,
                self._last_metrics['info_diversity']['exploit'],
                self._last_metrics['info_diversity']['explor']
            ))

            # For component_seci - use last calculated value
            self.component_seci_data.append((
                self.tick,
                self._last_metrics['component_seci']['value']
            ))

            # For aeci_variance - use last calculated value
            self.aeci_variance_data.append((
                self.tick,
                self._last_metrics['aeci_variance']['value']
            ))

            # For component_aeci - use last calculated value
            self.component_aeci_data.append((
                self.tick,
                self._last_metrics['component_aeci']['value']
            ))

            # For component_ai_trust_variance - use last calculated value
            self.component_ai_trust_variance_data.append((
                self.tick,
                self._last_metrics['component_ai_trust_variance']['value']
            ))

            # For belief_error - use last calculated values
            self.belief_error_data.append((
                self.tick,
                self._last_metrics['belief_error']['exploit'],
                self._last_metrics['belief_error']['explor']
            ))

            # For belief_variance - use last calculated values
            self.belief_variance_data.append((
                self.tick,
                self._last_metrics['belief_variance']['exploit'],
                self._last_metrics['belief_variance']['explor']
            ))

            # For retain_aeci - use last calculated values
            self.retain_aeci_data.append((
                self.tick,
                self._last_metrics['retain_aeci']['exploit'],
                self._last_metrics['retain_aeci']['explor']
            ))

            # For retain_seci - use last calculated values
            self.retain_seci_data.append((
                self.tick,
                self._last_metrics['retain_seci']['exploit'],
                self._last_metrics['retain_seci']['explor']
            ))

            # For running_aeci - use last calculated values
            self.running_aeci_data.append((
                self.tick,
                self._last_metrics['running_aeci']['exploit'],
                self._last_metrics['running_aeci']['explor']
            ))

            # For trust_stats - use last calculated values
            self.trust_stats.append((
                self.tick,
                self._last_metrics['trust_stats']['ai_exp'],
                self._last_metrics['trust_stats']['friend_exp'],
                self._last_metrics['trust_stats']['nonfriend_exp'],
                self._last_metrics['trust_stats']['ai_expl'],
                self._last_metrics['trust_stats']['friend_expl'],
                self._last_metrics['trust_stats']['nonfriend_expl']
            ))


#########################################
# Simulation and Experiment Functions
#########################################
def validate_social_network(model, save_dir="analysis_plots"):
    """
    Validates the social network structure and visualizes it.

    Args:
        model: The DisasterModel instance
        save_dir: Directory to save plots
    """

    os.makedirs(save_dir, exist_ok=True)

    # Calculate how many exploitative and exploratory agents
    num_exploitative = int(model.num_humans * model.share_exploitative)

    # Get network components
    components = list(nx.connected_components(model.social_network))

    print(f"Social Network Analysis:")
    print(f"  {model.num_humans} total agents ({num_exploitative} exploit, {model.num_humans - num_exploitative} explor)")
    print(f"  {model.social_network.number_of_edges()} connections")
    print(f"  {len(components)} connected components")

    # Analyze each component
    for i, component in enumerate(components):
        component_nodes = list(component)
        component_size = len(component_nodes)

        # Count agent types
        exploit_count = sum(1 for n in component_nodes if n < num_exploitative)
        explor_count = component_size - exploit_count

        # Calculate homophily (ratio of same-type connections)
        same_type_edges = 0
        cross_type_edges = 0

        for u, v in model.social_network.edges(component_nodes):
            u_type = "exploit" if u < num_exploitative else "explor"
            v_type = "exploit" if v < num_exploitative else "explor"

            if u_type == v_type:
                same_type_edges += 1
            else:
                cross_type_edges += 1

        total_edges = same_type_edges + cross_type_edges
        homophily = same_type_edges / total_edges if total_edges > 0 else 0

        print(f"  Component {i}: {component_size} agents ({exploit_count} exploit, {explor_count} explor)")
        print(f"    Homophily: {homophily:.2f} ({same_type_edges} same-type edges, {cross_type_edges} cross-type edges)")

    # Draw the network
    plt.figure(figsize=(12, 10))

    # Positions using spring layout with seed for reproducibility
    pos = nx.spring_layout(model.social_network, seed=42)

    # Color nodes by agent type
    node_colors = []
    for node in model.social_network.nodes():
        if node < num_exploitative:
            node_colors.append('red')  # Exploitative
        else:
            node_colors.append('blue')  # Exploratory

    # Draw the graph
    nx.draw_networkx_nodes(model.social_network, pos,
                          node_color=node_colors,
                          node_size=100,
                          alpha=0.8)

    nx.draw_networkx_edges(model.social_network, pos,
                          width=0.5,
                          alpha=0.5)

    plt.title(f"Social Network with {len(components)} Components\n"
              f"Red: Exploitative, Blue: Exploratory")
    plt.axis('off')

    # Save the figure
    plt.savefig(os.path.join(save_dir, "social_network.png"), dpi=300, bbox_inches='tight')
    print(f"Network visualization saved to {os.path.join(save_dir, 'social_network.png')}")
    plt.close()

    # Create a second visualization showing the components more clearly
    plt.figure(figsize=(12, 10))

    # Use a different layout that separates components better
    pos = nx.spring_layout(model.social_network, seed=42)

    # Draw each component with a different color
    for i, component in enumerate(components):
        subgraph = model.social_network.subgraph(component)
        color = plt.cm.tab10(i % 10)  # Use tab10 colormap for up to 10 components

        nx.draw_networkx_nodes(subgraph, pos,
                              node_color=color,
                              node_size=100,
                              alpha=0.8)

        nx.draw_networkx_edges(subgraph, pos,
                              width=0.5,
                              alpha=0.5,
                              edge_color=color)

    plt.title(f"Social Network with {len(components)} Components\n"
              f"Each color represents a different component")
    plt.axis('off')

    # Save the second figure
    plt.savefig(os.path.join(save_dir, "social_network_components.png"), dpi=300, bbox_inches='tight')
    print(f"Component visualization saved to {os.path.join(save_dir, 'social_network_components.png')}")
    plt.close()

    # Check if model.humans exists before trying to analyze beliefs
    if not hasattr(model, 'humans') or not model.humans:
        print("\nSkipping belief analysis - humans not populated yet")
        # Return minimal results with default average component SECI
        return {
            'num_components': len(components),
            'component_sizes': [len(c) for c in components],
            'component_seci_values': {i: 0 for i in range(len(components))},
            'avg_component_seci': 0
        }

    # Analyze belief divergence by component
    print("\nBelief Divergence Analysis by Network Component:")

    # Get all beliefs across all agents
    all_beliefs = {}
    for agent_id, agent in model.humans.items():
        for cell, belief_info in agent.beliefs.items():
            if isinstance(belief_info, dict):
                level = belief_info.get('level', 0)
                if cell not in all_beliefs:
                    all_beliefs[cell] = []
                all_beliefs[cell].append(level)

    # Calculate global variance for all beliefs
    global_vars = {}
    for cell, levels in all_beliefs.items():
        if len(levels) > 1:
            global_vars[cell] = np.var(levels)

    # Calculate variance within each component
    component_vars = {}
    for i, component in enumerate(components):
        component_vars[i] = {}
        for cell, levels in all_beliefs.items():
            # Get beliefs for agents in this component
            component_beliefs = []
            for node in component:
                agent_id = f"H_{node}"
                if agent_id in model.humans:
                    agent = model.humans[agent_id]
                    if cell in agent.beliefs and isinstance(agent.beliefs[cell], dict):
                        level = agent.beliefs[cell].get('level', 0)
                        component_beliefs.append(level)

            # Calculate variance if we have enough beliefs
            if len(component_beliefs) > 1:
                component_vars[i][cell] = np.var(component_beliefs)

    # Calculate SECI values for each component
    # Within the validate_social_network function, update the SECI calculation section:

    # Calculate SECI values for each component
    component_seci_values = {}
    for i, component in enumerate(components):
        # Only calculate SECI for components with enough agents
        if len(component) <= 1:
            print(f"  Component {i} too small for SECI calculation")
            component_seci_values[i] = 0
            continue

        # Get all beliefs from agents in this component
        component_beliefs_by_cell = {}
        for node in component:
            agent_id = f"H_{node}"
            if agent_id in model.humans:
                agent = model.humans[agent_id]
                for cell, belief_info in agent.beliefs.items():
                    if isinstance(belief_info, dict):
                        level = belief_info.get('level', 0)
                        if cell not in component_beliefs_by_cell:
                            component_beliefs_by_cell[cell] = []
                        component_beliefs_by_cell[cell].append(level)

        # Calculate SECI for this component
        seci_values = []
        for cell, component_levels in component_beliefs_by_cell.items():
            # Only consider cells where we have beliefs from multiple agents
            if len(component_levels) > 1 and cell in global_vars and global_vars[cell] > 0:
                # Calculate component variance
                component_var = np.var(component_levels)
                global_var = global_vars[cell]

                # Debug output for this specific cell
                print(f"    Cell {cell}: global_var={global_var:.4f}, component_var={component_var:.4f}")

                # SECI formula: (global_var - component_var) / global_var
                seci = (global_var - component_var) / global_var
                seci = max(0, min(1, seci))  # Bound to [0,1]
                seci_values.append(seci)

                # Extra debug if SECI is 0
                if seci < 0.01:
                    print(f"      Low SECI ({seci:.4f}) - component levels: {component_levels}")
                    print(f"      Global levels in cell {cell}: {all_beliefs.get(cell, [])}")

        # Average SECI for this component
        if seci_values:
            component_seci_values[i] = sum(seci_values) / len(seci_values)
            print(f"  Component {i} SECI: {component_seci_values[i]:.4f} (from {len(seci_values)} cells)")
        else:
            print(f"  Component {i}: No variance data available for SECI calculation")
            component_seci_values[i] = 0

    # Calculate average SECI across all components
    if component_seci_values:
        avg_component_seci = sum(component_seci_values.values()) / len(component_seci_values)
        print(f"  Average component SECI: {avg_component_seci:.4f}")
    else:
        avg_component_seci = 0

    # Create a histogram of SECI values
    plt.figure(figsize=(10, 6))
    plt.hist(list(component_seci_values.values()), bins=10, range=(0, 1),
             color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel('SECI Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Component SECI Values')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(save_dir, "component_seci_histogram.png"), dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'num_components': len(components),
        'component_sizes': [len(c) for c in components],
        'component_seci_values': component_seci_values,
        'avg_component_seci': avg_component_seci
    }

### function to calculate SECI components separately
def calculate_component_seci(model, components):
    """Calculate SECI values for network components without validation."""
    if not components:
        return 0

    # Get all beliefs across all agents
    all_beliefs = {}
    for agent_id, agent in model.humans.items():
        for cell, belief_info in agent.beliefs.items():
            if isinstance(belief_info, dict):
                level = belief_info.get('level', 0)
                if cell not in all_beliefs:
                    all_beliefs[cell] = []
                all_beliefs[cell].append(level)

    # Calculate global variance for all beliefs
    global_vars = {}
    for cell, levels in all_beliefs.items():
        if len(levels) > 1:
            global_vars[cell] = np.var(levels)

    # Calculate SECI for each component
    component_seci_values = {}
    for i, component in enumerate(components):
        # Only calculate SECI for components with enough agents
        if len(component) <= 1:
            component_seci_values[i] = 0
            continue

        # Get all beliefs from agents in this component
        component_beliefs_by_cell = {}
        for node in component:
            agent_id = f"H_{node}"
            if agent_id in model.humans:
                agent = model.humans[agent_id]
                for cell, belief_info in agent.beliefs.items():
                    if isinstance(belief_info, dict):
                        level = belief_info.get('level', 0)
                        if cell not in component_beliefs_by_cell:
                            component_beliefs_by_cell[cell] = []
                        component_beliefs_by_cell[cell].append(level)

        # Calculate SECI for this component
        seci_values = []
        for cell, component_levels in component_beliefs_by_cell.items():
            if len(component_levels) > 1 and cell in global_vars and global_vars[cell] > 0:
                component_var = np.var(component_levels)
                global_var = global_vars[cell]
                seci = (global_var - component_var) / global_var
                seci = max(0, min(1, seci))  # Bound to [0,1]
                seci_values.append(seci)

        # Average SECI for this component
        if seci_values:
            component_seci_values[i] = sum(seci_values) / len(seci_values)
        else:
            component_seci_values[i] = 0

    # Calculate average SECI across all components
    if component_seci_values:
        avg_component_seci = sum(component_seci_values.values()) / len(component_seci_values)
    else:
        avg_component_seci = 0

    return avg_component_seci

# Function to monitor component SECI evolution over time
def track_component_seci_evolution(model, tick_interval=10, save_dir="analysis_plots"):
    """
    Tracks and visualizes the evolution of component SECI over time.
    Enhanced with proper component-level tracking and run aggregation.
    """
    if model.tick % tick_interval != 0:
        return  # Only run at intervals

    os.makedirs(save_dir, exist_ok=True)

    # Initialize storage for tracking - Create if not exists
    if not hasattr(model, 'component_seci_evolution'):
        model.component_seci_evolution = {
            'ticks': [],
            'num_components': [],
            'avg_component_seci': [],
            'component_data': {}  # Store data for each component over time
        }

    # Get connected components
    components = list(nx.connected_components(model.social_network))
    num_components = len(components)

    # Record tick
    model.component_seci_evolution['ticks'].append(model.tick)
    model.component_seci_evolution['num_components'].append(num_components)

    # Get all beliefs across all agents
    all_beliefs = {}
    for agent_id, agent in model.humans.items():
        for cell, belief_info in agent.beliefs.items():
            if isinstance(belief_info, dict):
                level = belief_info.get('level', 0)
                if cell not in all_beliefs:
                    all_beliefs[cell] = []
                all_beliefs[cell].append(level)

    # Calculate global variance for all beliefs
    global_vars = {}
    for cell, levels in all_beliefs.items():
        if len(levels) > 1:
            global_vars[cell] = np.var(levels)

    # Calculate SECI for each component
    component_seci_values = {}
    for i, component in enumerate(components):
        # Only calculate SECI for components with enough agents
        if len(component) <= 1:
            component_seci_values[i] = 0
            continue

        # Get all beliefs from agents in this component
        component_beliefs_by_cell = {}
        for node in component:
            agent_id = f"H_{node}"
            if agent_id in model.humans:
                agent = model.humans[agent_id]
                for cell, belief_info in agent.beliefs.items():
                    if isinstance(belief_info, dict):
                        level = belief_info.get('level', 0)
                        if cell not in component_beliefs_by_cell:
                            component_beliefs_by_cell[cell] = []
                        component_beliefs_by_cell[cell].append(level)

        # Calculate SECI for this component
        seci_values = []
        for cell, component_levels in component_beliefs_by_cell.items():
            if len(component_levels) > 1 and cell in global_vars and global_vars[cell] > 0:
                component_var = np.var(component_levels)
                global_var = global_vars[cell]
                seci = (global_var - component_var) / global_var
                seci = max(0, min(1, seci))  # Bound to [0,1]
                seci_values.append(seci)

        # Average SECI for this component
        if seci_values:
            component_seci_values[i] = sum(seci_values) / len(seci_values)
        else:
            component_seci_values[i] = 0

    # Calculate average component SECI
    if component_seci_values:
        avg_component_seci = sum(component_seci_values.values()) / len(component_seci_values)
    else:
        avg_component_seci = 0

    # Record average component SECI
    model.component_seci_evolution['avg_component_seci'].append(avg_component_seci)

    # Record individual component data with unique identifiers
    for comp_id, seci_value in component_seci_values.items():
        # Create a unique stable identifier for this component
        # Use the smallest node ID in the component as its identifier
        component = list(components[comp_id])
        comp_key = f"comp_{min(component)}"
        
        if comp_key not in model.component_seci_evolution['component_data']:
            model.component_seci_evolution['component_data'][comp_key] = []

        # Add None for any missing ticks
        while len(model.component_seci_evolution['component_data'][comp_key]) < len(model.component_seci_evolution['ticks']) - 1:
            model.component_seci_evolution['component_data'][comp_key].append(None)

        # Add current value
        model.component_seci_evolution['component_data'][comp_key].append(seci_value)

    # Store in the model's component_seci_data list format with consistent structure
    if not hasattr(model, 'component_seci_data'):
        model.component_seci_data = []
    
    # Store in a format that can be easily aggregated later
    model.component_seci_data.append({
        'tick': model.tick, 
        'avg_component_seci': avg_component_seci,
        'component_values': component_seci_values  # Store values for each component
    })

    # Plot evolution only if we have enough data
    if len(model.component_seci_evolution['ticks']) > 1:
        # Create a simplified plot showing just average component SECI
        plt.figure(figsize=(12, 8))

        # Plot average component SECI
        plt.plot(model.component_seci_evolution['ticks'],
                model.component_seci_evolution['avg_component_seci'],
                'k-', linewidth=2, label='Average Component SECI')

        # Plot individual components if available
        for comp_key, values in model.component_seci_evolution['component_data'].items():
            # Extend with None values if needed
            while len(values) < len(model.component_seci_evolution['ticks']):
                values.append(None)
                
            # Convert None values to NaN for plotting
            values_array = np.array([np.nan if v is None else v for v in values])
            plt.plot(model.component_seci_evolution['ticks'], values_array, 
                   '--', alpha=0.5, linewidth=1, label=comp_key)

        plt.xlabel('Tick')
        plt.ylabel('SECI Value')
        plt.title('Evolution of Component SECI')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

        # Save the figure
        plt.savefig(os.path.join(save_dir, f"component_seci_evolution_tick_{model.tick}.png"),
                  dpi=300, bbox_inches='tight')
        plt.close()

def plot_component_seci_distribution(results_dict, title_suffix=""):
    """Plots the distribution of SECI values across different components"""
    
    # Extract component SECI data
    component_seci = results_dict.get('component_seci')
    component_seci_data = results_dict.get('component_seci_data', [])
    
    # Check if we have component-level data
    has_component_level_data = False
    if component_seci_data:
        # Check for component_values key in any item
        for tick_data in component_seci_data:
            if isinstance(tick_data, dict) and 'component_values' in tick_data:
                has_component_level_data = True
                break
    
    # If no component-level data, fall back to average values
    if not has_component_level_data:
        print(f"No component-level SECI data available for {title_suffix}")
        # Extract all SECI values from the array data
        all_seci_values = []
        if component_seci is not None and isinstance(component_seci, np.ndarray):
            if component_seci.ndim == 3 and component_seci.shape[2] > 1:
                # Get values from column 1
                all_seci_values = component_seci[:, :, 1].flatten()
                all_seci_values = all_seci_values[~np.isnan(all_seci_values)]
    else:
        # Collect all component values from all ticks
        all_seci_values = []
        for tick_data in component_seci_data:
            if isinstance(tick_data, dict) and 'component_values' in tick_data:
                component_values = tick_data['component_values']
                if isinstance(component_values, dict):
                    all_seci_values.extend(component_values.values())
    
    # Filter out non-numeric values
    all_seci_values = [v for v in all_seci_values if isinstance(v, (int, float)) and not np.isnan(v)]
    
    # Create the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: SECI distribution histogram
    if all_seci_values:
        ax1.hist(all_seci_values, bins=30, range=(0, 1),
                 color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(np.mean(all_seci_values), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(all_seci_values):.3f}')
        ax1.axvline(np.median(all_seci_values), color='green', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(all_seci_values):.3f}')
        ax1.set_xlabel('Component SECI Value')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Component SECI Distribution {title_suffix}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No component SECI data', ha='center', va='center')
    
    # Plot 2: Evolution of average component SECI
    if component_seci is not None and isinstance(component_seci, np.ndarray):
        if component_seci.ndim == 3 and component_seci.shape[1] > 0:
            mean_seci = np.nanmean(component_seci[:, :, 1], axis=0)
            ticks = np.arange(len(mean_seci))
            
            ax2.plot(ticks, mean_seci, 'b-', linewidth=2, label='Mean')
            ax2.set_xlabel('Tick')
            ax2.set_ylabel('Average Component SECI')
            ax2.set_title(f'Component SECI Evolution {title_suffix}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Invalid component SECI shape', ha='center', va='center')
    else:
        ax2.text(0.5, 0.5, 'No component SECI evolution data', ha='center', va='center')
    
    plt.tight_layout()
    save_path = f"agent_model_results/component_seci_distribution_{title_suffix}.png"
    plt.savefig(save_path.replace('(','').replace(')','').replace('=','_'))
    plt.close()

def run_simulation(params):
    model = DisasterModel(**params)
    for _ in range(params.get("ticks", 150)):
        model.step()
    return model

def safe_convert_to_array(data_list):
    """Safely convert data to numpy array, handling mixed types."""
    if not data_list:
        print("WARNING: Empty data_list in safe_convert_to_array")
        return np.array([])
    
    # Print first few items in data_list for debugging
    print(f"DEBUG: First few items in data_list:")
    for i, item in enumerate(data_list[:3]):
        print(f"  Item {i}: type={type(item)}, value={item}")
    
    # Case 1: Check if all elements are tuple with length 2 (AECI variance format)
    if all(isinstance(item, tuple) and len(item) == 2 for item in data_list):
        print(f"DEBUG: Converting tuple list to structured array")
        # Convert to structured array with named fields
        result = np.array(data_list, dtype=[('tick', 'i4'), ('value', 'f4')])
        print(f"DEBUG: Result shape: {result.shape}, dtype: {result.dtype}")
        return result
    
    # Case 2: Convert dict data to tuples
    if all(isinstance(item, dict) for item in data_list):
        print(f"DEBUG: Converting dict list to array")
        converted = []
        for item in data_list:
            if 'tick' in item and 'value' in item:
                converted.append((item['tick'], item['value']))
            elif 'tick' in item and 'exploit' in item and 'explor' in item:
                converted.append((item['tick'], item['exploit'], item['explor']))
            else:
                # Handle unknown dict format
                converted.append((item.get('tick', 0), sum(v for k, v in item.items() if k != 'tick')))
        print(f"DEBUG: Converted dicts to {len(converted)} tuples")
        return np.array(converted)
    
    # Case 3: Mixed types or unknown format
    print(f"DEBUG: Using object array for mixed types")
    return np.array(data_list, dtype=object)

def simulation_generator(num_runs, base_params):
    """Generator function that runs simulations and yields results one at a time."""
    for seed in range(num_runs):
        try:
            # Set seeds for reproducibility
            random.seed(seed +42) #including offset
            np.random.seed(seed +101)

            # Run the simulation with the given parameters
            print(f"Starting simulation run {seed+1}/{num_runs}...")
            model = run_simulation(base_params)

            # Extract all relevant data from the model
            # Fix AECI variance data to ensure consistent format
            aeci_variance_data = model.aeci_variance_data if hasattr(model, 'aeci_variance_data') else []
            
            # Convert list of tuples to 2D array (ticks x 2) explicitly
            if aeci_variance_data:
                ticks = [t for t, _ in aeci_variance_data]
                values = [v for _, v in aeci_variance_data]
                aeci_variance_array = np.column_stack((ticks, values))
            else:
                aeci_variance_array = np.array([])

            result = {
                "trust_stats": np.array(model.trust_stats),
                "seci": np.array(model.seci_data),
                "aeci": np.array(model.aeci_data),
                "retain_aeci": np.array(model.retain_aeci_data),
                "retain_seci": np.array(model.retain_seci_data),
                "belief_error": np.array(model.belief_error_data),
                "belief_variance": np.array(model.belief_variance_data),
                "unmet_needs_evolution": model.unmet_needs_evolution,
                # Handle component data flexibly
                "component_seci": safe_convert_to_array(getattr(model, 'component_seci_data', [])),
                "aeci_variance": aeci_variance_array,  # Use the explicitly converted 2D array
                "component_aeci": safe_convert_to_array(getattr(model, 'component_aeci_data', [])),
                "component_ai_trust_variance": safe_convert_to_array(getattr(model, 'component_ai_trust_variance_data', [])),
                "info_diversity": np.array(getattr(model, 'info_diversity_data', [])),
                "event_ticks": list(getattr(model, 'event_ticks', []))
            }

            # Yield the result and model object
            yield result, model

            # Clean up to free memory
            print(f"Completed simulation run {seed+1}/{num_runs}")
            del model
            gc.collect()

        except Exception as e:
            print(f"Error in simulation run {seed+1}: {e}")
            import traceback
            traceback.print_exc()
            # Yield empty result in case of error to keep the loop going
            yield {}, None


# --- Aggregate results ---
def aggregate_simulation_results(num_runs, base_params):
    """
    Runs multiple simulations, aggregates results, and calculates summary statistics.
    Enhanced to properly handle AECI variance data.
    """
    trust_list, seci_list, aeci_list, retain_aeci_list, retain_seci_list = [], [], [], [], []
    unmet_needs_evolution_list, belief_error_list, belief_variance_list = [], [], []
    component_seci_data_list = []
    # Lists for new metrics
    component_seci_list = []
    aeci_variance_list = []
    component_aeci_list = []
    component_ai_trust_variance_list = []  # Fixed: Single list instead of tuple of lists
    info_diversity_list = []  # Information Diversity (Shannon Entropy)
    event_ticks_list = [] # Collect event ticks from each run
    max_aeci_variance_per_run = [] # max aeci var

    # --- Lists for Assistance Counts ---
    exploit_correct_per_run, exploit_incorrect_per_run = [], []
    explor_correct_per_run, explor_incorrect_per_run = [], []

    print(f"Starting aggregation for {num_runs} runs...")

    # Use a counter to track actual successful runs
    successful_runs = 0
    generator = simulation_generator(num_runs, base_params)

    # Iterate through the generator using a for loop
    for result, model in generator:
        successful_runs += 1
        print(f"  Processing results from run {successful_runs}/{num_runs}...")

        if not result:  # Skip if result is empty (error case)
            print(f"  Empty result for run {successful_runs}, skipping")
            continue

        # --- Append All Metrics (using .get for safety) ---
        trust_list.append(result.get("trust_stats", np.array([])))
        seci_list.append(result.get("seci", np.array([])))
        aeci_list.append(result.get("aeci", np.array([])))
        retain_aeci_list.append(result.get("retain_aeci", np.array([])))
        retain_seci_list.append(result.get("retain_seci", np.array([])))
        unmet_needs_evolution_list.append(result.get("unmet_needs_evolution", []))
        belief_error_list.append(result.get("belief_error", np.array([])))
        belief_variance_list.append(result.get("belief_variance", np.array([])))

        # Append new metrics with improved error handling
        component_seci_list.append(result.get("component_seci", np.array([])))
        info_diversity_list.append(result.get("info_diversity", np.array([])))

        # Special handling for AECI variance data
        aeci_variance_data = result.get("aeci_variance", np.array([]))
        aeci_variance_list.append(aeci_variance_data)
        
        # Extract max AECI variance if available
        if isinstance(aeci_variance_data, np.ndarray) and aeci_variance_data.size > 0:
            # Handle different array structures
            if aeci_variance_data.ndim == 3 and aeci_variance_data.shape[2] > 1:
                # Extract values column
                variance_values = aeci_variance_data[:, 1]  # Values column
                if variance_values.size > 0:
                    # Get maximum value
                    max_variance = np.nanmax(variance_values)
                    if not np.isnan(max_variance) and not np.isinf(max_variance):
                        max_aeci_variance_per_run.append(max_variance)
                        print(f"  Max AECI variance for run {successful_runs}: {max_variance:.4f}")
                    else:
                        print(f"  Invalid max AECI variance for run {successful_runs}")
            else:
                print(f"  Unexpected AECI variance data shape: {aeci_variance_data.shape}")
        else:
            print(f"  No valid AECI variance data for run {successful_runs}")
        
        component_aeci_list.append(result.get("component_aeci", np.array([])))
        component_ai_trust_variance_list.append(result.get("component_ai_trust_variance", np.array([])))
        event_ticks_list.append(result.get("event_ticks", []))
        if hasattr(model, 'component_seci_data'):
            component_seci_data_list.append(model.component_seci_data)
        else:
            component_seci_data_list.append([])

        # --- Aggregate Assistance Counts ---
        run_exploit_correct, run_exploit_incorrect = 0, 0
        run_explor_correct, run_explor_incorrect = 0, 0

        if model and hasattr(model, 'humans'):
            for agent in model.humans.values():
                if agent.agent_type == "exploitative":
                    run_exploit_correct += agent.correct_targets
                    run_exploit_incorrect += agent.incorrect_targets
                else:  # Exploratory
                    run_explor_correct += agent.correct_targets
                    run_explor_incorrect += agent.incorrect_targets

        exploit_correct_per_run.append(run_exploit_correct)
        exploit_incorrect_per_run.append(run_exploit_incorrect)
        explor_correct_per_run.append(run_explor_correct)
        explor_incorrect_per_run.append(run_explor_incorrect)

        print(f"  Finished processing run {successful_runs}/{num_runs}")

        # Make sure model is deleted to free memory
        if model is not None:
            del model
        gc.collect()

    print(f"Completed {successful_runs} successful runs out of {num_runs} attempts")
    print("Aggregating results...")

    # --- Stack Arrays with improved error handling ---
    trust_array = safe_stack(trust_list)
    seci_array = safe_stack(seci_list)
    aeci_array = safe_stack(aeci_list)
    retain_aeci_array = safe_stack(retain_aeci_list)
    retain_seci_array = safe_stack(retain_seci_list)
    belief_error_array = safe_stack(belief_error_list)
    belief_variance_array = safe_stack(belief_variance_list)

    # Stack new data arrays with special handling for aeci_variance
    component_seci_array = safe_stack(component_seci_list)
    aeci_variance_array = safe_stack(aeci_variance_list)
    component_aeci_array = safe_stack(component_aeci_list)
    component_ai_trust_variance_array = safe_stack(component_ai_trust_variance_list)
    info_diversity_array = safe_stack(info_diversity_list)

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
            "lower": 0, "upper": 0  # Percentiles complex for ratios
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
        "aeci": aeci_array,
        "retain_aeci": retain_aeci_array,
        "retain_seci": retain_seci_array,
        "belief_error": belief_error_array,
        "belief_variance": belief_variance_array,
        "unmet_needs_evol": unmet_needs_evolution_list,

        # Assistance metrics
        "assist": assist_stats,
        "assist_ratio": ratio_stats,
        "raw_assist_counts": {
            "exploit_correct": exploit_correct_per_run,
            "exploit_incorrect": exploit_incorrect_per_run,
            "explor_correct": explor_correct_per_run,
            "explor_incorrect": explor_incorrect_per_run
        },

        # New aggregated metrics
        "component_seci": component_seci_array,
        "component_seci_data": component_seci_data_list,
        "aeci_variance": aeci_variance_array,
        "max_aeci_variance": max_aeci_variance_per_run,
        "component_aeci": component_aeci_array,
        "component_ai_trust_variance": component_ai_trust_variance_array,
        "info_diversity": info_diversity_array,
        "event_ticks_list": event_ticks_list
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

#Debug helper function
def debug_aeci_variance_data(results_dict, title_suffix=""):
    """Inspects and prints detailed AECI variance data structure"""
    print(f"\n=== AECI Variance Data Inspection for {title_suffix} ===")
    
    # Get aeci_variance data
    aeci_var_data = results_dict.get("aeci_variance")
    
    # Basic data check
    if aeci_var_data is None:
        print("ERROR: aeci_variance data is None")
        return
    
    if not isinstance(aeci_var_data, np.ndarray):
        print(f"ERROR: aeci_variance isn't a numpy array (type: {type(aeci_var_data)})")
        if isinstance(aeci_var_data, list):
            print(f"  List length: {len(aeci_var_data)}")
            for i, item in enumerate(aeci_var_data[:3]):
                print(f"  Item {i}: {type(item)} - {item}")
        return
    
    # Array shape analysis
    print(f"AECI Variance array shape: {aeci_var_data.shape}")
    
    # Inspect dimensions
    if aeci_var_data.ndim >= 3:
        # Expected shape: (runs, ticks, 2) where 2nd dim is [tick, value]
        print(f"First dimension (runs): {aeci_var_data.shape[0]}")
        print(f"Second dimension (ticks): {aeci_var_data.shape[1]}")
        print(f"Third dimension (data): {aeci_var_data.shape[2]}")
        
        # Inspect first few values
        print("\nSample values:")
        for run in range(min(2, aeci_var_data.shape[0])):
            print(f"Run {run}:")
            tick_slice = slice(0, min(5, aeci_var_data.shape[1]))
            print(f"  First 5 ticks: {aeci_var_data[run, tick_slice, :]}")
            
            # Check if values are in expected range [-1,1]
            if aeci_var_data.shape[2] > 1:
                values = aeci_var_data[run, :, 1]
                min_val, max_val = np.nanmin(values), np.nanmax(values)
                print(f"  Value range: [{min_val:.4f}, {max_val:.4f}]")
                print(f"  Mean value: {np.nanmean(values):.4f}")
                print(f"  Contains NaN: {np.isnan(values).any()}")
                print(f"  Contains Inf: {np.isinf(values).any()}")
    else:
        print(f"WARNING: Expected 3D array, got {aeci_var_data.ndim}D")
        # Try to analyze based on actual dimensions
        if aeci_var_data.ndim == 2:
            print("Assuming array format is (runs, values):")
            for run in range(min(2, aeci_var_data.shape[0])):
                print(f"Run {run}: {aeci_var_data[run, :]}")
        elif aeci_var_data.ndim == 1:
            print("Assuming array is a flat list of values:")
            print(f"Values: {aeci_var_data[:min(10, len(aeci_var_data))]}")
    
    print("=== End AECI Variance Data Inspection ===\n")

# Helper function
def safe_stack(data_list):
    """Safely stacks a list of numpy arrays, handling empty lists/arrays and AECI variance data."""
    if not data_list:
        print("WARNING: Empty data_list in safe_stack")
        return np.array([])
    
    # Check first few arrays for debugging
    print(f"DEBUG: Examining first few arrays to stack:")
    for i, item in enumerate(data_list[:3]):
        if isinstance(item, np.ndarray):
            print(f"  Array {i}: shape={item.shape}, dtype={item.dtype}")
        else:
            print(f"  Item {i}: type={type(item)}")
    
    # Special handling for AECI variance data (arrays with special dtype)
    if any(isinstance(item, np.ndarray) and hasattr(item, 'dtype') and 
           item.dtype.names is not None and 'tick' in item.dtype.names and 
           'value' in item.dtype.names for item in data_list):
        print("DEBUG: Detected AECI variance format with named fields")
        # Convert structured arrays to 3D arrays with shape (runs, ticks, 2)
        max_ticks = max(arr.shape[0] for arr in data_list if isinstance(arr, np.ndarray) and arr.ndim >= 1)
        result = np.zeros((len(data_list), max_ticks, 2))
        
        for i, arr in enumerate(data_list):
            if isinstance(arr, np.ndarray) and hasattr(arr, 'dtype') and arr.dtype.names is not None:
                # Extract tick and value columns
                ticks = arr['tick']
                values = arr['value']
                # Fill the result array
                result[i, :len(ticks), 0] = ticks
                result[i, :len(values), 1] = values
            else:
                print(f"  WARNING: Item {i} is not a structured array with named fields")
        
        print(f"DEBUG: Converted AECI variance data to shape {result.shape}")
        return result
    
    # Handle regular arrays
    valid_arrays = [item for item in data_list if isinstance(item, np.ndarray) and item.size > 0]
    if not valid_arrays:
        print("WARNING: No valid arrays to stack")
        return np.array([])
    
    try:
        # Find expected shape from first valid array
        expected_ndim = valid_arrays[0].ndim
        expected_shape_after_tick_col = valid_arrays[0].shape[1:] # Shape excluding the tick dimension
        
        print(f"DEBUG: Expected shape for stacking: ndim={expected_ndim}, shape[1:]={expected_shape_after_tick_col}")
        
        processed_list = []
        for i, item in enumerate(valid_arrays):
            # Only include arrays matching expected dimensions
            if item.ndim == expected_ndim and item.shape[1:] == expected_shape_after_tick_col:
                processed_list.append(item)
            else:
                print(f"WARNING: Skipping array with shape {item.shape} during stacking (expected ndim={expected_ndim}, shape[1:]={expected_shape_after_tick_col})")
        
        if not processed_list:
            print("WARNING: No arrays with matching shapes to stack")
            return np.array([])
        
        result = np.stack(processed_list, axis=0)
        print(f"DEBUG: Successfully stacked arrays to shape {result.shape}")
        return result
    except ValueError as e:
        print(f"ERROR during stacking: {e}")
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


#########################################
# Plotting Functions
#########################################

# helper function
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

def safe_plot(ax, data_array, col_idx, label, color, linestyle='-', is_ratio=True, ticks=None):
    """
    Safely plots data with proper error handling and diagnostics.
    """
    # Check if data exists
    if data_array is None:
        print(f"Warning: No data array for {label}")
        return False

    # Check if it's a numpy array with the right dimensions
    if not isinstance(data_array, np.ndarray):
        print(f"Warning: Data for {label} is not a numpy array (type: {type(data_array)})")
        return False

    # Check array dimensions
    if data_array.ndim < 3:
        print(f"Warning: Data for {label} has incorrect dimensions: {data_array.ndim}")
        return False

    # Check if we have any runs
    if data_array.shape[0] == 0:
        print(f"Warning: No runs in data for {label}")
        return False

    # Check if we have any timepoints
    if data_array.shape[1] == 0:
        print(f"Warning: No timepoints in data for {label}")
        return False

    # Check column index validity
    if col_idx >= data_array.shape[2]:
        print(f"Warning: Invalid column index {col_idx} for {label} (max: {data_array.shape[2]-1})")
        return False

    # Create default ticks if not provided
    if ticks is None:
        ticks = np.arange(data_array.shape[1])

    # Check length compatibility with ticks
    if data_array.shape[1] != len(ticks):
        print(f"Warning: Data length mismatch for {label}: {data_array.shape[1]} vs {len(ticks)} ticks")
        # Use the shorter length
        min_len = min(data_array.shape[1], len(ticks))
        plot_ticks = ticks[:min_len]
        data_slice = slice(0, min_len)
    else:
        plot_ticks = ticks
        data_slice = slice(None)

    try:
        # Extract data for the specific column
        values = data_array[:, data_slice, col_idx]

        # Check for NaNs or infinities
        nan_mask = np.isnan(values) | np.isinf(values)
        if np.any(nan_mask):
            num_issues = np.sum(nan_mask)
            total_values = values.size
            print(f"Warning: {label} has {num_issues}/{total_values} NaN/Inf values - replacing with zeros")
            values = np.where(nan_mask, 0, values)

        # Clip ratio metrics to [0,1]
        if is_ratio:
            out_of_bounds = (values < 0) | (values > 1)
            if np.any(out_of_bounds):
                num_issues = np.sum(out_of_bounds)
                total_values = values.size
                print(f"Warning: {label} has {num_issues}/{total_values} out-of-bounds values - clipping")
                values = np.clip(values, 0, 1)

        # Calculate statistics with safety checks
        if values.size > 0:
            mean = np.nanmean(values, axis=0)
            # Use nanpercentile if available, otherwise handle NaNs ourselves
            try:
                p25 = np.nanpercentile(values, 25, axis=0)
                p75 = np.nanpercentile(values, 75, axis=0)
            except AttributeError:
                # Older numpy versions don't have nanpercentile
                p25 = np.percentile(np.where(np.isnan(values), np.nanmean(values), values), 25, axis=0)
                p75 = np.percentile(np.where(np.isnan(values), np.nanmean(values), values), 75, axis=0)

            # Replace any remaining NaNs
            mean = np.nan_to_num(mean)
            p25 = np.nan_to_num(p25)
            p75 = np.nan_to_num(p75)

            # Ensure percentiles don't cross the mean
            lower = np.minimum(mean, p25)
            upper = np.maximum(mean, p75)

            # Plot with MORE VISIBLE percentile bands
            ax.plot(plot_ticks, mean, label=label, color=color, linestyle=linestyle)
            ax.fill_between(plot_ticks, lower, upper, color=color, alpha=0.4)  # INCREASED from 0.2 to 0.4

            # Set y-limits for ratio metrics
            if is_ratio:
                ax.set_ylim(0, 1.05)

            return True
        else:
            print(f"Warning: No valid values to plot for {label}")
            return False

    except Exception as e:
        print(f"Error plotting {label}: {e}")
        import traceback
        traceback.print_exc()
        return False
# Fix for plot_component_seci_distribution
def plot_component_seci_distribution(results_dict, title_suffix=""):
    """Plots the distribution of SECI values across different components"""
    
    # Extract component SECI data
    component_seci = results_dict.get('component_seci')
    
    if component_seci is None or not isinstance(component_seci, np.ndarray) or component_seci.size == 0:
        print(f"No component SECI data available for {title_suffix}")
        return
    
    # Extract all SECI values across all ticks and runs
    all_seci_values = []
    
    # Handle different data structures
    if component_seci.ndim == 3:  # (runs, ticks, values)
        # Get all non-zero values from the data
        values = component_seci[:, :, 1]  # Assuming column 1 contains the values
        all_seci_values = values[values > 0].flatten()
    else:
        print(f"Warning: Unexpected component SECI data shape: {component_seci.shape}")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: SECI distribution histogram
    if len(all_seci_values) > 0:
        ax1.hist(all_seci_values, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(np.mean(all_seci_values), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(all_seci_values):.3f}')
        ax1.axvline(np.median(all_seci_values), color='green', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(all_seci_values):.3f}')
        ax1.set_xlabel('Component SECI Value')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Component SECI Distribution {title_suffix}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No component SECI data', ha='center', va='center')
    
    # Plot 2: Evolution of average component SECI
    if component_seci.ndim == 3 and component_seci.shape[1] > 0:
        mean_seci = np.mean(component_seci[:, :, 1], axis=0)
        ticks = component_seci[0, :, 0]  # Get ticks from first run
        
        ax2.plot(ticks, mean_seci, 'b-', linewidth=2, label='Mean')
        ax2.set_xlabel('Tick')
        ax2.set_ylabel('Average Component SECI')
        ax2.set_title(f'Component SECI Evolution {title_suffix}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No component SECI evolution data', ha='center', va='center')
    
    plt.tight_layout()
    save_path = f"agent_model_results/component_seci_distribution_{title_suffix}.png"
    plt.savefig(save_path.replace('(','').replace(')','').replace('=','_'))
    plt.close()

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

def plot_summary_echo_indices_vs_alignment(results_b, alignment_values, title_suffix="Exp B"):
    """Plots the mean values of echo chamber indices vs AI alignment as boxplots.
    This function aggregates data across the entire simulation run, not just final values."""
    num_params = len(alignment_values)
    boxplot_width = 0.15

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Echo Chamber Indices vs AI Alignment ({title_suffix})", fontsize=16)

    # For each metric, we'll gather data across all timestamps for each alignment level
    seci_exploit_data = []
    seci_explor_data = []
    aeci_var_data = []
    aeci_call_exploit_data = []
    aeci_call_explor_data = []
    comp_ai_trust_var_data = []

    # Collect data for each alignment level
    alignment_labels = []
    for align in alignment_values:
        alignment_labels.append(str(align))
        res = results_b.get(align)
        if not res:
            # Add empty data for this alignment level to keep positions consistent
            seci_exploit_data.append([])
            seci_explor_data.append([])
            aeci_var_data.append([])
            aeci_call_exploit_data.append([])
            aeci_call_explor_data.append([])
            comp_ai_trust_var_data.append([])
            continue

        # Extract time-series data and calculate means across all time steps
        def get_all_values(data_array, col_index):
            if data_array is None or not isinstance(data_array, np.ndarray) or data_array.size == 0:
                return []

            if data_array.ndim < 3 or data_array.shape[1] == 0 or col_index >= data_array.shape[2]:
                return []

            # Get all values for all runs and all time steps
            # Reshape to flatten across runs and time steps
            values = data_array[:, :, col_index].flatten()

            # Handle NaNs and infinities
            values = values[~np.isnan(values) & ~np.isinf(values)]

            # IMPORTANT: Don't clip SECI or AECI-Var! They range from -1 to +1
            # Negative values indicate echo chambers, which we need to see!
            # Only AI call ratios are true [0,1] proportions

            return values

        # Collect data for each metric
        seci_exploit_data.append(get_all_values(res.get("seci"), 1))
        seci_explor_data.append(get_all_values(res.get("seci"), 2))
        aeci_var_data.append(get_all_values(res.get("aeci_variance"), 1))
        aeci_call_exploit_data.append(get_all_values(res.get("aeci"), 1))
        aeci_call_explor_data.append(get_all_values(res.get("aeci"), 2))
        comp_ai_trust_var_data.append(get_all_values(res.get("component_ai_trust_variance"), 1))

    # Plot SECI boxplots
    ax = axes[0, 0]
    positions = np.arange(len(alignment_values))
    bplot_exploit = ax.boxplot(seci_exploit_data, positions=positions-boxplot_width/2,
                             widths=boxplot_width, patch_artist=True,
                             boxprops=dict(facecolor='maroon', alpha=0.7),
                             flierprops=dict(marker='o', markerfacecolor='maroon', markersize=3, alpha=0.7),
                             medianprops=dict(color='black'))
    bplot_explor = ax.boxplot(seci_explor_data, positions=positions+boxplot_width/2,
                             widths=boxplot_width, patch_artist=True,
                             boxprops=dict(facecolor='salmon', alpha=0.7),
                             flierprops=dict(marker='o', markerfacecolor='salmon', markersize=3, alpha=0.7),
                             medianprops=dict(color='black'))
    ax.set_ylabel("SECI Value")
    ax.set_title("Social Echo Chamber (SECI)\n(Negative = Echo Chamber, Positive = Diversification)")
    ax.legend([bplot_exploit["boxes"][0], bplot_explor["boxes"][0]], ['Exploit', 'Explor'], loc='best')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)  # Zero reference line
    ax.set_ylim(-1.05, 1.05)  # SECI ranges from -1 to +1, allow full range

    # Plot AECI Variance boxplots
    ax = axes[0, 1]
    bplot_aeci_var = ax.boxplot(aeci_var_data, positions=positions,
                              widths=boxplot_width*1.5, patch_artist=True,
                              boxprops=dict(facecolor='magenta', alpha=0.7),
                              flierprops=dict(marker='o', markerfacecolor='magenta', markersize=3, alpha=0.7),
                              medianprops=dict(color='black'))
    ax.set_ylabel("AI Belief Variance Reduction")
    ax.set_title("AI Echo Chamber (AECI-Var)\n(Negative = AI Echo Chamber, Positive = AI Diversifies)")
    ax.legend([bplot_aeci_var["boxes"][0]], ['AI Reliant Group'], loc='best')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)  # Zero reference line
    ax.set_ylim(-1.05, 1.05)  # AECI-Var ranges from -1 to +1, allow full range

    # Plot AI Call Ratio boxplots
    ax = axes[1, 0]
    bplot_call_exploit = ax.boxplot(aeci_call_exploit_data, positions=positions-boxplot_width/2,
                                  widths=boxplot_width, patch_artist=True,
                                  boxprops=dict(facecolor='darkblue', alpha=0.7),
                                  flierprops=dict(marker='o', markerfacecolor='darkblue', markersize=3, alpha=0.7),
                                  medianprops=dict(color='black'))
    bplot_call_explor = ax.boxplot(aeci_call_explor_data, positions=positions+boxplot_width/2,
                                 widths=boxplot_width, patch_artist=True,
                                 boxprops=dict(facecolor='skyblue', alpha=0.7),
                                 flierprops=dict(marker='o', markerfacecolor='skyblue', markersize=3, alpha=0.7),
                                 medianprops=dict(color='black'))
    ax.set_ylabel("AI Call Ratio")
    ax.set_title("AI Usage (AECI Call Ratio)")
    ax.legend([bplot_call_exploit["boxes"][0], bplot_call_explor["boxes"][0]], ['Exploit', 'Explor'], loc='best')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.set_ylim(0, 1)

    # Plot Component AI Trust Variance boxplots
    ax = axes[1, 1]
    bplot_ai_trust_var = ax.boxplot(comp_ai_trust_var_data, positions=positions,
                                   widths=boxplot_width*1.5, patch_artist=True,
                                   boxprops=dict(facecolor='cyan', alpha=0.7),
                                   flierprops=dict(marker='o', markerfacecolor='cyan', markersize=3, alpha=0.7),
                                   medianprops=dict(color='black'))
    ax.set_ylabel("Component AI Trust Variance")
    ax.set_title("AI Trust Clustering")
    ax.legend([bplot_ai_trust_var["boxes"][0]], ['Component AI Trust Var.'], loc='best')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.set_ylim(bottom=0)

    # Set common x-axis properties
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks(positions)
            ax.set_xticklabels(alignment_labels)
            ax.set_xlabel("AI Alignment Level")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = f"agent_model_results/boxplot_echo_{title_suffix}.png"
    plt.savefig(save_path.replace('(','').replace(')','').replace('=','_'))
    plt.close(fig)

    return fig

def plot_summary_performance_vs_alignment(results_b, alignment_values, title_suffix="Exp B"):
    """Plots summary performance metrics vs AI alignment across the entire simulation."""
    num_params = len(alignment_values)
    boxplot_width = 0.15

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
    fig.suptitle(f"Performance Metrics vs AI Alignment ({title_suffix})", fontsize=16)

    # For each metric, we'll gather data across all timestamps for each alignment level
    mae_exploit_data = []
    mae_explor_data = []
    unmet_needs_data = []
    exploit_correct_ratio_data = []
    explor_correct_ratio_data = []

    # Collect data for each alignment level
    alignment_labels = []
    for align in alignment_values:
        alignment_labels.append(str(align))
        res = results_b.get(align)
        if not res:
            # Add empty data for this alignment level to keep positions consistent
            mae_exploit_data.append([])
            mae_explor_data.append([])
            unmet_needs_data.append([])
            exploit_correct_ratio_data.append([])
            explor_correct_ratio_data.append([])
            continue

        # Helper for extracting MAE data
        def get_all_mae_values(data_array, col_index):
            if data_array is None or not isinstance(data_array, np.ndarray) or data_array.size == 0:
                return []

            if data_array.ndim < 3 or data_array.shape[1] == 0 or col_index >= data_array.shape[2]:
                return []

            # Get all values for all runs and all time steps
            values = data_array[:, :, col_index].flatten()

            # Handle NaNs and infinities
            values = values[~np.isnan(values) & ~np.isinf(values)]

            # MAE should be non-negative
            values = np.clip(values, 0.0, None)

            return values

        # Helper for extracting unmet needs data
        def get_all_unmet_values(unmet_list):
            if not unmet_list:
                return []

            # Flatten all unmet needs data across runs and time steps
            all_values = []
            for run_data in unmet_list:
                if run_data and len(run_data) > 0:
                    all_values.extend(run_data)

            # Handle NaNs and infinities
            all_values = np.array(all_values)
            all_values = all_values[~np.isnan(all_values) & ~np.isinf(all_values)]

            # Unmet needs should be non-negative
            all_values = np.clip(all_values, 0.0, None)

            return all_values

        # Helper for extracting correct token ratio
        def get_all_correct_ratio_values(raw_counts, correct_key, incorrect_key):
            correct_list = raw_counts.get(correct_key, [])
            incorrect_list = raw_counts.get(incorrect_key, [])

            if not correct_list or not incorrect_list:
                return []

            # Calculate ratios for each run
            ratios = []
            for i in range(min(len(correct_list), len(incorrect_list))):
                total = correct_list[i] + incorrect_list[i]
                if total > 0:
                    ratios.append(correct_list[i] / total)

            # Handle NaNs and infinities
            ratios = np.array(ratios)
            ratios = ratios[~np.isnan(ratios) & ~np.isinf(ratios)]

            # Clip ratios to [0,1]
            ratios = np.clip(ratios, 0.0, 1.0)

            return ratios

        # Collect data for each metric
        mae_exploit_data.append(get_all_mae_values(res.get("belief_error"), 1))
        mae_explor_data.append(get_all_mae_values(res.get("belief_error"), 2))
        unmet_needs_data.append(get_all_unmet_values(res.get("unmet_needs_evol")))

        raw_counts = res.get("raw_assist_counts", {})
        exploit_correct_ratio_data.append(get_all_correct_ratio_values(raw_counts, "exploit_correct", "exploit_incorrect"))
        explor_correct_ratio_data.append(get_all_correct_ratio_values(raw_counts, "explor_correct", "explor_incorrect"))

    # Plot Belief Error (MAE) boxplots
    ax = axes[0]
    positions = np.arange(len(alignment_values))
    bplot_mae_exploit = ax.boxplot(mae_exploit_data, positions=positions-boxplot_width/2,
                                 widths=boxplot_width, patch_artist=True,
                                 boxprops=dict(facecolor='red', alpha=0.7),
                                 flierprops=dict(marker='o', markerfacecolor='red', markersize=3, alpha=0.7),
                                 medianprops=dict(color='black'))
    bplot_mae_explor = ax.boxplot(mae_explor_data, positions=positions+boxplot_width/2,
                                widths=boxplot_width, patch_artist=True,
                                boxprops=dict(facecolor='blue', alpha=0.7),
                                flierprops=dict(marker='o', markerfacecolor='blue', markersize=3, alpha=0.7),
                                medianprops=dict(color='black'))
    ax.set_ylabel("Belief Error (MAE)")
    ax.set_title("Belief Accuracy")
    ax.legend([bplot_mae_exploit["boxes"][0], bplot_mae_explor["boxes"][0]], ['Exploit', 'Explor'], loc='best')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.set_ylim(bottom=0)

    # Plot Unmet Needs boxplots
    ax = axes[1]
    bplot_unmet = ax.boxplot(unmet_needs_data, positions=positions,
                           widths=boxplot_width*1.5, patch_artist=True,
                           boxprops=dict(facecolor='purple', alpha=0.7),
                           flierprops=dict(marker='o', markerfacecolor='purple', markersize=3, alpha=0.7),
                           medianprops=dict(color='black'))
    ax.set_ylabel("Unmet Needs Count")
    ax.set_title("Disaster Response Gap")
    ax.legend([bplot_unmet["boxes"][0]], ['Unmet Needs'], loc='best')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.set_ylim(bottom=0)

    # Plot Correct Token Ratio boxplots
    ax = axes[2]
    bplot_correct_exploit = ax.boxplot(exploit_correct_ratio_data, positions=positions-boxplot_width/2,
                                     widths=boxplot_width, patch_artist=True,
                                     boxprops=dict(facecolor='red', alpha=0.7),
                                     flierprops=dict(marker='o', markerfacecolor='red', markersize=3, alpha=0.7),
                                     medianprops=dict(color='black'))
    bplot_correct_explor = ax.boxplot(explor_correct_ratio_data, positions=positions+boxplot_width/2,
                                    widths=boxplot_width, patch_artist=True,
                                    boxprops=dict(facecolor='blue', alpha=0.7),
                                    flierprops=dict(marker='o', markerfacecolor='blue', markersize=3, alpha=0.7),
                                    medianprops=dict(color='black'))
    ax.set_ylabel("Correct Token Ratio")
    ax.set_title("Assistance Quality")
    ax.legend([bplot_correct_exploit["boxes"][0], bplot_correct_explor["boxes"][0]], ['Exploit', 'Explor'], loc='best')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.set_ylim(0, 1)

    # Set common x-axis properties
    for ax in axes:
        ax.set_xticks(positions)
        ax.set_xticklabels(alignment_labels)
        ax.set_xlabel("AI Alignment Level")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = f"agent_model_results/boxplot_perf_{title_suffix}.png"
    plt.savefig(save_path.replace('(','').replace(')','').replace('=','_'))
    plt.close(fig)

    return fig


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
            # Filter out NaN or infinite values
            valid_max_values = [v for v in max_values if not np.isnan(v) and not np.isinf(v)]
            
            if valid_max_values:
                # Calculate statistics for this alignment level
                mean_max = np.mean(valid_max_values)
                p25 = np.percentile(valid_max_values, 25)
                p75 = np.percentile(valid_max_values, 75)

                max_var_means.append(mean_max)
                max_var_errors[0].append(max(0, mean_max - p25))  # Lower error
                max_var_errors[1].append(max(0, p75 - mean_max))  # Upper error
            else:
                # If no valid values, append zeros
                max_var_means.append(0)
                max_var_errors[0].append(0)
                max_var_errors[1].append(0)
        else:
            # If no data, append zeros
            max_var_means.append(0)
            max_var_errors[0].append(0)
            max_var_errors[1].append(0)

    # Create plot with error handling
    plt.figure(figsize=(10, 6))
    try:
        x = np.arange(len(alignment_values))
        plt.bar(x, max_var_means, width=0.6, yerr=max_var_errors, capsize=5,
                color='magenta', alpha=0.7, label='Max AECI Variance')
        
        # Add value labels with error handling
        for i, v in enumerate(max_var_means):
            if not np.isnan(v) and not np.isinf(v):
                plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.xlabel('AI Alignment Level')
        plt.ylabel('Maximum AECI Variance Reduction')
        plt.title(f'Maximum AI Belief Variance Reduction by Alignment Level {title_suffix}')
        plt.xticks(x, alignment_values)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        
        save_path = f"agent_model_results/max_aeci_variance_{title_suffix.replace('(','').replace(')','').replace('=','_')}.png"
        plt.savefig(save_path)
    except Exception as e:
        print(f"Error in plot_max_aeci_variance_by_alignment: {e}")
    finally:
        plt.close()

#########################################
# Echo chamber plotting functions
def plot_summary_echo_indices_vs_alignment(results_b, alignment_values, title_suffix="Exp B"):
    """Plots the mean values of echo chamber indices vs AI alignment as boxplots."""
    num_params = len(alignment_values)
    boxplot_width = 0.15

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Echo Chamber Indices vs AI Alignment ({title_suffix})", fontsize=16)

    # For each metric, we'll gather data across all timestamps for each alignment level
    seci_exploit_data = []
    seci_explor_data = []
    aeci_var_data = []
    aeci_call_exploit_data = []
    aeci_call_explor_data = []
    comp_ai_trust_var_data = []

    # Collect data for each alignment level
    alignment_labels = []
    for align in alignment_values:
        alignment_labels.append(str(align))
        res = results_b.get(align)
        if not res:
            # Add empty data for this alignment level to keep positions consistent
            seci_exploit_data.append([])
            seci_explor_data.append([])
            aeci_var_data.append([])
            aeci_call_exploit_data.append([])
            aeci_call_explor_data.append([])
            comp_ai_trust_var_data.append([])
            continue

        # Extract time-series data and calculate means across all time steps
        def get_all_values(data_array, col_index):
            if data_array is None or not isinstance(data_array, np.ndarray) or data_array.size == 0:
                return []

            if data_array.ndim < 3 or data_array.shape[1] == 0 or col_index >= data_array.shape[2]:
                return []

            # Get all values for all runs and all time steps
            # Reshape to flatten across runs and time steps
            values = data_array[:, :, col_index].flatten()

            # Handle NaNs and infinities
            values = values[~np.isnan(values) & ~np.isinf(values)]

            # Clip ratio metrics to [0,1]
            if any(x in str(col_index) for x in ['aeci', 'seci']) or col_index in [1, 2]:
                values = np.clip(values, 0.0, 1.0)

            return values

        # Collect data for each metric
        seci_exploit_data.append(get_all_values(res.get("seci"), 1))
        seci_explor_data.append(get_all_values(res.get("seci"), 2))
        aeci_var_data.append(get_all_values(res.get("aeci_variance"), 1))
        aeci_call_exploit_data.append(get_all_values(res.get("aeci"), 1))
        aeci_call_explor_data.append(get_all_values(res.get("aeci"), 2))
        comp_ai_trust_var_data.append(get_all_values(res.get("component_ai_trust_variance"), 1))

    # Plot SECI boxplots
    ax = axes[0, 0]
    positions = np.arange(len(alignment_values))
    bplot_exploit = ax.boxplot(seci_exploit_data, positions=positions-boxplot_width/2,
                             widths=boxplot_width, patch_artist=True,
                             boxprops=dict(facecolor='maroon', alpha=0.7),
                             flierprops=dict(marker='o', markerfacecolor='maroon', markersize=3, alpha=0.7),
                             medianprops=dict(color='black'))
    bplot_explor = ax.boxplot(seci_explor_data, positions=positions+boxplot_width/2,
                             widths=boxplot_width, patch_artist=True,
                             boxprops=dict(facecolor='salmon', alpha=0.7),
                             flierprops=dict(marker='o', markerfacecolor='salmon', markersize=3, alpha=0.7),
                             medianprops=dict(color='black'))
    ax.set_ylabel("SECI Value")
    ax.set_title("Social Echo Chamber (SECI)\n(Negative = Echo Chamber, Positive = Diversification)")
    ax.legend([bplot_exploit["boxes"][0], bplot_explor["boxes"][0]], ['Exploit', 'Explor'], loc='best')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)  # Zero reference line
    ax.set_ylim(-1.05, 1.05)  # SECI ranges from -1 to +1, allow full range

    # Plot AECI Variance boxplots
    ax = axes[0, 1]
    bplot_aeci_var = ax.boxplot(aeci_var_data, positions=positions,
                              widths=boxplot_width*1.5, patch_artist=True,
                              boxprops=dict(facecolor='magenta', alpha=0.7),
                              flierprops=dict(marker='o', markerfacecolor='magenta', markersize=3, alpha=0.7),
                              medianprops=dict(color='black'))
    ax.set_ylabel("AI Belief Variance Reduction")
    ax.set_title("AI Echo Chamber (AECI-Var)\n(Negative = AI Echo Chamber, Positive = AI Diversifies)")
    ax.legend([bplot_aeci_var["boxes"][0]], ['AI Reliant Group'], loc='best')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)  # Zero reference line
    ax.set_ylim(-1.05, 1.05)  # AECI-Var ranges from -1 to +1, allow full range

    # Plot AI Call Ratio boxplots
    ax = axes[1, 0]
    bplot_call_exploit = ax.boxplot(aeci_call_exploit_data, positions=positions-boxplot_width/2,
                                  widths=boxplot_width, patch_artist=True,
                                  boxprops=dict(facecolor='darkblue', alpha=0.7),
                                  flierprops=dict(marker='o', markerfacecolor='darkblue', markersize=3, alpha=0.7),
                                  medianprops=dict(color='black'))
    bplot_call_explor = ax.boxplot(aeci_call_explor_data, positions=positions+boxplot_width/2,
                                 widths=boxplot_width, patch_artist=True,
                                 boxprops=dict(facecolor='skyblue', alpha=0.7),
                                 flierprops=dict(marker='o', markerfacecolor='skyblue', markersize=3, alpha=0.7),
                                 medianprops=dict(color='black'))
    ax.set_ylabel("AI Call Ratio")
    ax.set_title("AI Usage (AECI Call Ratio)")
    ax.legend([bplot_call_exploit["boxes"][0], bplot_call_explor["boxes"][0]], ['Exploit', 'Explor'], loc='best')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.set_ylim(0, 1)

    # Plot Component AI Trust Variance boxplots
    ax = axes[1, 1]
    bplot_ai_trust_var = ax.boxplot(comp_ai_trust_var_data, positions=positions,
                                   widths=boxplot_width*1.5, patch_artist=True,
                                   boxprops=dict(facecolor='cyan', alpha=0.7),
                                   flierprops=dict(marker='o', markerfacecolor='cyan', markersize=3, alpha=0.7),
                                   medianprops=dict(color='black'))
    ax.set_ylabel("Component AI Trust Variance")
    ax.set_title("AI Trust Clustering")
    ax.legend([bplot_ai_trust_var["boxes"][0]], ['Component AI Trust Var.'], loc='best')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.set_ylim(bottom=0)

    # Set common x-axis properties
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks(positions)
            ax.set_xticklabels(alignment_labels)
            ax.set_xlabel("AI Alignment Level")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = f"agent_model_results/boxplot_echo_{title_suffix}.png"
    plt.savefig(save_path.replace('(','').replace(')','').replace('=','_'))
    plt.close(fig)

    return fig

def plot_average_performance_vs_alignment(results_b, alignment_values, title_suffix="Exp B"):
    """Plots mean performance metrics vs AI alignment with percentile bands."""
    num_params = len(alignment_values)
    bar_width = 0.2
    index = np.arange(num_params)

    # Empty lists to store metrics - CONSISTENTLY NAMED
    mae_exploit_mean, mae_exploit_err = [], [[],[]]
    mae_explor_mean, mae_explor_err = [], [[],[]]
    unmet_mean, unmet_err = [], [[],[]]
    correct_ratio_exploit_mean, correct_ratio_exploit_err = [], [[],[]]
    correct_ratio_explor_mean, correct_ratio_explor_err = [], [[],[]]

    for align in alignment_values:
        res = results_b.get(align)
        if not res: continue

        # Helper to extract average stats with percentiles
        def get_avg_stats(data_array, col_index):
            if data_array is None or not isinstance(data_array, np.ndarray) or data_array.size == 0:
                return 0, [0, 0]

            if data_array.ndim < 3 or data_array.shape[1] == 0 or col_index >= data_array.shape[2]:
                return 0, [0, 0]

            # Get all values across all time points for all runs
            all_vals = data_array[:, :, col_index].flatten()

            # Calculate mean and percentiles
            mean = np.mean(all_vals)
            p25 = np.percentile(all_vals, 25)
            p75 = np.percentile(all_vals, 75)

            # Return mean and error lengths [lower, upper]
            return mean, [max(0, mean - p25), max(0, p75 - mean)]

        # Helper for unmet needs (list of lists)
        def get_avg_unmet_stats(unmet_list):
            if not unmet_list:
                return 0, [0, 0]

            # Flatten all values from all runs
            all_vals = []
            for run_data in unmet_list:
                if run_data and len(run_data) > 0:
                    all_vals.extend(run_data)

            if not all_vals:
                return 0, [0, 0]

            # Calculate mean and percentiles
            mean = np.mean(all_vals)
            p25 = np.percentile(all_vals, 25)
            p75 = np.percentile(all_vals, 75)

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
        mean, err = get_avg_stats(res.get("belief_error"), 1)
        mae_exploit_mean.append(mean)
        mae_exploit_err[0].append(err[0])
        mae_exploit_err[1].append(err[1])

        mean, err = get_avg_stats(res.get("belief_error"), 2)
        mae_explor_mean.append(mean)
        mae_explor_err[0].append(err[0])
        mae_explor_err[1].append(err[1])

        # Extract unmet needs stats
        mean, err = get_avg_unmet_stats(res.get("unmet_needs_evol"))
        unmet_mean.append(mean)
        unmet_err[0].append(err[0])
        unmet_err[1].append(err[1])

        # Extract correct token ratios with percentiles
        raw_counts = res.get("raw_assist_counts", {})
        mean, err = get_ratio_stats(raw_counts, "exploit_correct", "exploit_incorrect")
        correct_ratio_exploit_mean.append(mean)
        correct_ratio_exploit_err[0].append(err[0])
        correct_ratio_exploit_err[1].append(err[1])

        mean, err = get_ratio_stats(raw_counts, "explor_correct", "explor_incorrect")
        correct_ratio_explor_mean.append(mean)
        correct_ratio_explor_err[0].append(err[0])
        correct_ratio_explor_err[1].append(err[1])

    # Setup plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    fig.suptitle(f"Average Performance vs AI Alignment ({title_suffix})", fontsize=16)

    # Plot Average MAE with percentile error bars - CONSISTENT NAMING
    axes[0].bar(index - bar_width/2, mae_exploit_mean, bar_width,
                yerr=mae_exploit_err, capsize=4, label='Exploit', color='red')
    axes[0].bar(index + bar_width/2, mae_explor_mean, bar_width,
                yerr=mae_explor_err, capsize=4, label='Explor', color='blue')
    axes[0].set_ylabel("Average MAE")
    axes[0].set_title("Belief Error (MAE)")
    axes[0].legend()
    axes[0].grid(True, axis='y', linestyle='--', alpha=0.6)
    axes[0].set_ylim(bottom=0)  # MAE cannot be negative

    # Plot Average Unmet Needs with percentile error bars
    axes[1].bar(index, unmet_mean, bar_width*1.5,
                yerr=unmet_err, capsize=4, label='Overall', color='purple')
    axes[1].set_ylabel("Average Unmet Needs")
    axes[1].set_title("Unmet Needs")
    axes[1].set_xlabel("AI Alignment Level")
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.6)
    axes[1].set_ylim(bottom=0)  # Unmet needs cannot be negative

    # Plot Average Correct Token Share with percentile error bars
    axes[2].bar(index - bar_width/2, correct_ratio_exploit_mean, bar_width,
                yerr=correct_ratio_exploit_err, capsize=4, label='Exploit', color='red')
    axes[2].bar(index + bar_width/2, correct_ratio_explor_mean, bar_width,
                yerr=correct_ratio_explor_err, capsize=4, label='Explor', color='blue')
    axes[2].set_ylabel("Average Correct Token Share")
    axes[2].set_title("Assistance Quality (Correct Ratio)")
    axes[2].legend()
    axes[2].grid(True, axis='y', linestyle='--', alpha=0.6)
    axes[2].set_ylim(0, 1)  # Ratio must be between 0 and 1

    for ax in axes:
        ax.set_xticks(index)
        ax.set_xticklabels(alignment_values)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = f"agent_model_results/avg_perf_{title_suffix}.png"
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

    # Save
    save_path = f"agent_model_results/trust_evolution_{title_suffix}.png"
    plt.savefig(save_path.replace('(','').replace(')','').replace('=','_'))
    plt.close(fig)

    return fig

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

# ---  PLOT 1: SIMULATION INDICES  ---
def plot_simulation_overview(results_dict, title_suffix=""):
    """Plots key performance and belief metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # Removed sharex=True for better control
    fig.suptitle(f"Simulation Overview {title_suffix}", fontsize=16)

    # --- Data Extraction ---
    belief_error = results_dict.get("belief_error")
    belief_variance = results_dict.get("belief_variance")
    unmet_needs_list = results_dict.get("unmet_needs_evol")
    assist_stats = results_dict.get("assist", {})
    raw_counts = results_dict.get("raw_assist_counts", {})

    # Debug: Check data availability and structure
    print(f"\n=== Simulation Overview Data Diagnostics ===")
    print(f"belief_error: {belief_error.shape if belief_error is not None and isinstance(belief_error, np.ndarray) else 'None or invalid'}")
    print(f"belief_variance: {belief_variance.shape if belief_variance is not None and isinstance(belief_variance, np.ndarray) else 'None or invalid'}")
    print(f"unmet_needs_evol: {len(unmet_needs_list) if unmet_needs_list else 0} runs")

    # Check for non-zero values
    if belief_error is not None and isinstance(belief_error, np.ndarray) and belief_error.size > 0:
        print(f"  belief_error non-zero: {np.count_nonzero(belief_error)} / {belief_error.size}")
        print(f"  belief_error range: [{np.nanmin(belief_error):.4f}, {np.nanmax(belief_error):.4f}]")
    if belief_variance is not None and isinstance(belief_variance, np.ndarray) and belief_variance.size > 0:
        print(f"  belief_variance non-zero: {np.count_nonzero(belief_variance)} / {belief_variance.size}")
        print(f"  belief_variance range: [{np.nanmin(belief_variance):.4f}, {np.nanmax(belief_variance):.4f}]")

    # Determine Ticks - Use a sequence of approaches to find the actual tick count
    num_ticks = None

    # Method 1: Check if belief_error contains valid tick data in first column
    if belief_error is not None and isinstance(belief_error, np.ndarray) and belief_error.ndim >= 3:
        if belief_error.shape[1] > 0 and belief_error.shape[2] > 0:
            # Some models store ticks as the first column of metrics
            first_run_ticks = belief_error[0, :, 0]
            if np.all(np.diff(first_run_ticks) > 0):  # Check if monotonically increasing
                num_ticks = len(first_run_ticks)
                print(f"Detected {num_ticks} ticks from belief_error tick column")
            else:
                # Just use the array shape if not storing ticks in column 0
                num_ticks = belief_error.shape[1]
                print(f"Using {num_ticks} timepoints from belief_error shape")

    # Method 2: If not found from belief_error, check other data arrays
    if num_ticks is None:
        for data_key, data_array in [
            ("belief_variance", belief_variance),
            # Add other arrays here if needed
        ]:
            if data_array is not None and isinstance(data_array, np.ndarray) and data_array.ndim >= 3:
                if data_array.shape[1] > 0:
                    num_ticks = data_array.shape[1]
                    print(f"Using {num_ticks} timepoints from {data_key} shape")
                    break

    # Method 3: As a last resort, use unmet_needs data
    if num_ticks is None and unmet_needs_list and any(unmet_needs_list):
        max_len = 0
        for run_data in unmet_needs_list:
            if run_data is not None and len(run_data) > max_len:
                max_len = len(run_data)
        if max_len > 0:
            num_ticks = max_len
            print(f"Using {num_ticks} timepoints from max unmet_needs length")

    # Fallback: If still no tick count found, use a reasonable default
    if num_ticks is None or num_ticks <= 0:
        print("Warning: Could not detect tick count from data, using default")
        num_ticks = 150  # Reasonable default

    # Generate tick array
    ticks = np.arange(num_ticks)
    print(f"Using {len(ticks)} ticks for plotting")

    # Updated helper function for better bands
    def _enhanced_plot_mean_iqr(ax, data_array, col_idx, label, color, linestyle='-'):
        """Helper to plot mean and IQR band with enhanced visibility."""
        if data_array is None or not isinstance(data_array, np.ndarray) or data_array.size == 0:
            return

        if data_array.ndim < 3 or col_idx >= data_array.shape[2]:
            return

        try:
            # Extract values for all runs (limited to actual ticks)
            data_ticks = min(data_array.shape[1], len(ticks))
            plot_ticks = ticks[:data_ticks]
            values = data_array[:, :data_ticks, col_idx]

            # Handle NaNs and infinities
            values = np.where(np.isnan(values) | np.isinf(values), 0, values)

            # Calculate statistics
            mean = np.mean(values, axis=0)
            p25 = np.percentile(values, 25, axis=0)
            p75 = np.percentile(values, 75, axis=0)

            # Ensure bounds are properly ordered
            lower = np.minimum(mean, p25)
            upper = np.maximum(mean, p75)

            # Plot with enhanced visibility for bands
            ax.plot(plot_ticks, mean, label=label, color=color, linestyle=linestyle)
            ax.fill_between(plot_ticks, lower, upper, color=color, alpha=0.4)

            return True
        except Exception as e:
            print(f"Error plotting {label}: {e}")
            return False

    # --- Subplot 1: Belief Error (MAE) ---
    ax = axes[0, 0]
    _enhanced_plot_mean_iqr(ax, belief_error, 1, "MAE Exploit", "red")
    _enhanced_plot_mean_iqr(ax, belief_error, 2, "MAE Explor", "blue")
    ax.set_title("Avg. Belief MAE")
    ax.set_ylabel("MAE")
    ax.set_xlabel("Tick")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize='small')
    ax.set_ylim(bottom=0)

    # --- Subplot 2: Belief Variance ---
    ax = axes[0, 1]
    _enhanced_plot_mean_iqr(ax, belief_variance, 1, "Var Exploit", "red")
    _enhanced_plot_mean_iqr(ax, belief_variance, 2, "Var Explor", "blue")
    ax.set_title("Within-Type Belief Variance")
    ax.set_ylabel("Variance")
    ax.set_xlabel("Tick")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize='small')
    ax.set_ylim(bottom=0)

    # --- Subplot 3: Unmet Needs ---
    ax = axes[1, 0]
    if unmet_needs_list:
        try:
            # Find actual length of unmet needs data
            T_needs = max(len(run_data) for run_data in unmet_needs_list if run_data is not None and len(run_data)>0)
            T_needs = min(T_needs, num_ticks)  # Limit to the number of ticks

            if T_needs > 0:
                unmet_array = np.full((len(unmet_needs_list), T_needs), np.nan)
                for i, run_data in enumerate(unmet_needs_list):
                    if run_data is not None and len(run_data) > 0:
                        data_len = min(len(run_data), T_needs)
                        unmet_array[i, :data_len] = run_data[:data_len]

                mean = np.nanmean(unmet_array, axis=0)
                lower = np.nanpercentile(unmet_array, 25, axis=0)
                upper = np.nanpercentile(unmet_array, 75, axis=0)

                plot_ticks_needs = np.arange(T_needs)

                ax.plot(plot_ticks_needs, mean, label="Unmet Need", color="purple")
                ax.fill_between(plot_ticks_needs, lower, upper, color="purple", alpha=0.4)
                ax.set_xlim(0, T_needs-1)  # Set xlim based on actual unmet needs data length
            else:
                ax.text(0.5, 0.5, 'No unmet needs data', ha='center', va='center')
        except Exception as e:
            print(f"Warning: Could not plot unmet needs: {e}")
    ax.set_title("Unmet Need Count")
    ax.set_ylabel("Count")
    ax.set_xlabel("Tick")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize='small')
    ax.set_ylim(bottom=0)

    # Set x-axis limits for belief plots based on their data
    if belief_error is not None and isinstance(belief_error, np.ndarray) and belief_error.ndim >= 2:
        axes[0, 0].set_xlim(0, belief_error.shape[1]-1)
    if belief_variance is not None and isinstance(belief_variance, np.ndarray) and belief_variance.ndim >= 2:
        axes[0, 1].set_xlim(0, belief_variance.shape[1]-1)

    # --- Subplot 4: token assistance summary pie chart ---
    ax = axes[1, 1]
    if assist_stats and raw_counts:
        try:
            # Access assist data safely
            exploit_correct = assist_stats.get("exploit_correct", {}).get("mean", 0)
            exploit_incorrect = assist_stats.get("exploit_incorrect", {}).get("mean", 0)
            explor_correct = assist_stats.get("explor_correct", {}).get("mean", 0)
            explor_incorrect = assist_stats.get("explor_incorrect", {}).get("mean", 0)

            # Skip if all values are zero
            total = exploit_correct + exploit_incorrect + explor_correct + explor_incorrect
            if total > 0:
                sizes = [exploit_correct, exploit_incorrect, explor_correct, explor_incorrect]
                labels = ['Exploit Correct', 'Exploit Incorrect', 'Explor Correct', 'Explor Incorrect']
                colors = ['forestgreen', 'lightcoral', 'skyblue', 'salmon']

                # Filter out zero values
                non_zero_data = [(size, label, color) for size, label, color in zip(sizes, labels, colors) if size > 0]
                if non_zero_data:
                    actual_sizes, actual_labels, actual_colors = zip(*non_zero_data)
                    wedges, texts, autotexts = ax.pie(actual_sizes, labels=actual_labels, colors=actual_colors,
                                                    autopct=lambda pct: f'{pct:.1f}%' if pct > 0 else '',
                                                    pctdistance=0.85)

                    # Improve text visibility
                    plt.setp(autotexts, size=8, weight="bold")
                    plt.setp(texts, size=9)
                    ax.set_title("Token Assistance\nDistribution")
                else:
                    ax.text(0.5, 0.5, 'No assistance data', ha='center', va='center')
            else:
                ax.text(0.5, 0.5, 'No tokens sent', ha='center', va='center')
        except Exception as e:
            print(f"Warning: Could not create pie chart: {e}")
            ax.text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
    else:
        ax.text(0.5, 0.5, 'Assistance data\nnot available', ha='center', va='center')

    # Save the main overview plot
    save_path = f"agent_model_results/simulation_overview_{title_suffix}.png"
    plt.savefig(save_path.replace('(','').replace(')','').replace('=','_'))
    plt.close(fig)

    return fig


# ---  PLOT 2: ECHO CHAMBERS ---
def plot_echo_chamber_indices(results_dict, title_suffix=""):
    """Plots various echo chamber and information flow metrics."""

    print(f"\nDiagnostics for Information Dynamics Plot {title_suffix}")

    # Examine each data array and print statistics
    for key in ["seci", "aeci", "retain_seci", "retain_aeci", "component_seci",
               "aeci_variance", "component_aeci", "component_ai_trust_variance"]:
        data = results_dict.get(key)
        if data is not None and isinstance(data, np.ndarray) and data.ndim >= 2:
            print(f"{key}: shape={data.shape}")

            # For non-empty arrays, print sample statistics
            if data.shape[0] > 0 and data.shape[1] > 0 and data.ndim >= 3:
                # Print first and last timepoint stats for key columns
                for col in range(min(data.shape[2], 3)):
                    first_vals = data[:, 0, col]
                    last_vals = data[:, -1, col]
                    print(f"  Col {col}: First tick mean={np.nanmean(first_vals):.3f}, Last tick mean={np.nanmean(last_vals):.3f}")
                    print(f"  Col {col}: Value range: [{np.nanmin(data[:, :, col]):.3f}, {np.nanmax(data[:, :, col]):.3f}]")
        else:
            print(f"{key}: Missing or invalid data")

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
    def _robust_plot_mean_iqr(ax, ticks, data_array, data_index, label, color, linestyle='-', is_ratio_metric=False):
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
    safe_plot(ax, seci, 1, "SECI Exploit", "maroon", ticks=ticks)
    safe_plot(ax, seci, 2, "SECI Explor", "salmon", ticks=ticks)
    safe_plot(ax, aeci, 1, "AI Call Ratio Exploit", "darkblue", ticks=ticks)
    safe_plot(ax, aeci, 2, "AI Call Ratio Explor", "skyblue", linestyle='--', ticks=ticks)
    add_event_lines(ax, avg_event_ticks)
    ax.set_title("Social Homogeneity (SECI) & AI Call Ratio (AECI)")
    ax.set_ylabel("Index / Ratio")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize='small')
    ax.set_ylim(-1.05, 1.05)  # SECI ranges from -1 to +1 (negative = echo chamber)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)  # Zero reference line
    ax.set_xlim(left=0)  # Start x-axis at 0

    # --- Subplot 2: Retainment ---
    ax = axes [0, 1]
    # Check if retainment data exists and has proper shape before plotting
    if retain_seci is not None and isinstance(retain_seci, np.ndarray) and retain_seci.shape[0] > 0:
        retain_seci_shape = retain_seci.shape
        print(f"Retain SECI data shape: {retain_seci_shape}")
        if retain_seci.ndim >= 3 and retain_seci.shape[1] > 0 and retain_seci.shape[2] > 2:
            # Safe plot with explicit error handling
            plot_success = safe_plot(ax, retain_seci, 1, "Retain Friend (Exploit)", "green", ticks=ticks)
            if not plot_success:
                print("WARNING: Failed to plot Retain Friend (Exploit)")

            plot_success = safe_plot(ax, retain_seci, 2, "Retain Friend (Explor)", "lightgreen", linestyle='--', ticks=ticks)
            if not plot_success:
                print("WARNING: Failed to plot Retain Friend (Explor)")

            plot_success = safe_plot(ax, retain_aeci, 1, "Retain AI (Exploit)", "purple", ticks=ticks)
            if not plot_success:
                print("WARNING: Failed to plot Retain AI (Exploit)")

            plot_success = safe_plot(ax, retain_aeci, 2, "Retain AI (Explor)", "plum", linestyle='--', ticks=ticks)
            if not plot_success:
                print("WARNING: Failed to plot Retain AI (Explor)")
        else:
            print(f"WARNING: Retainment data has wrong dimensions/shape: {retain_seci_shape}")
            # Fallback text
            ax.text(0.5, 0.5, "Retainment data has invalid shape",
                   ha='center', va='center', transform=ax.transAxes)
    else:
        print("WARNING: Retainment data missing or invalid")
        # Add info text to empty subplot
        ax.text(0.5, 0.5, "No retainment data available",
               ha='center', va='center', transform=ax.transAxes)

    # Add event lines if available
    add_event_lines(ax, avg_event_ticks)
    ax.set_title("Information Retainment (Share Accepted)")
    ax.set_ylabel("Share of Accepted Info")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize='small')
    ax.set_ylim(bottom=0, top=1.05)
    ax.set_xlim(left=0)  # Start x-axis at 0

    # --- Subplot 3: Component & AI Variance Indices ---
    ax = axes[1, 0]
    safe_plot(ax, comp_seci, 1, "Component SECI", "black", ticks=ticks)
    safe_plot(ax, comp_aeci, 1, "Component AI Call Ratio", "grey", ticks=ticks)
    safe_plot(ax, aeci_var, 1, "AI Bel. Var. Reduction", "magenta", ticks=ticks)  # AI Echo Chamber
    add_event_lines(ax, avg_event_ticks)
    ax.set_title("Component & AI Echo Chamber Indices")
    ax.set_ylabel("Index / Variance Reduction")
    ax.set_xlabel("Tick")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize='small')
    ax.set_ylim(-1.05, 1.05)  # Allow negative values (echo chambers)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)  # Zero reference
    ax.set_xlim(left=0)  # Start x-axis at 0

    # --- Subplot 4: AI Trust Clustering ---
    ax = axes[1, 1]
    safe_plot(ax, comp_ai_trust_var, 1, "Component AI Trust Var.", "cyan", is_ratio=False, ticks=ticks)
    add_event_lines(ax, avg_event_ticks)
    ax.set_title("AI Trust Clustering (Variance within Components)")
    ax.set_ylabel("Variance")
    ax.set_xlabel("Tick")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize='small')
    ax.set_ylim(bottom=0)  # Only set bottom for variance
    ax.set_xlim(left=0)  # Start x-axis at 0

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

            # Don't clip SECI - it can range from -1 to +1 (negative = echo chamber)
            # Only clip if it's a ratio metric (handled separately)

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
    save_path = f"agent_model_results/seci_aeci_evolution_{title_suffix}.png"
    plt.savefig(save_path.replace('(','').replace(')','').replace('=','_'))
    plt.close(fig)

    return fig

def plot_ai_trust_vs_alignment(model, save_dir="analysis_plots"):
    """Plot AI trust by agent type with respect to AI alignment level."""
    os.makedirs(save_dir, exist_ok=True)

    # Check if model.humans exists
    if not hasattr(model, 'humans') or not model.humans:
        print(f"Warning: No humans in model for AI trust plotting (alignment={model.ai_alignment_level})")
        return

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
    """Plots assistance metrics with improved bar sizing and spacing."""
    labels = ['Exploitative', 'Exploratory']

    # Extract data safely
    mean_correct = [
        assist_stats.get("exploit_correct", {}).get("mean", 0),
        assist_stats.get("explor_correct", {}).get("mean", 0)
    ]
    mean_incorrect = [
        assist_stats.get("exploit_incorrect", {}).get("mean", 0),
        assist_stats.get("explor_incorrect", {}).get("mean", 0)
    ]

    # Calculate errors with proper error handling
    errors_correct = [[0, 0], [0, 0]]
    errors_incorrect = [[0, 0], [0, 0]]

    # Helper for error calculation
    def calculate_errors(data, idx):
        if not data or len(data) == 0:
            return [0, 0]
        try:
            mean_val = np.mean(data)
            p25 = np.percentile(data, 25)
            p75 = np.percentile(data, 75)
            return [max(0, mean_val - p25), max(0, p75 - mean_val)]
        except Exception:
            return [0, 0]

    # Calculate errors if data exists
    if raw_assist_counts:
        errors_correct[0] = calculate_errors(raw_assist_counts.get("exploit_correct", []), 0)
        errors_correct[1] = calculate_errors(raw_assist_counts.get("explor_correct", []), 0)
        errors_incorrect[0] = calculate_errors(raw_assist_counts.get("exploit_incorrect", []), 0)
        errors_incorrect[1] = calculate_errors(raw_assist_counts.get("explor_incorrect", []), 0)

    # Format error bars for matplotlib
    yerr_correct = np.array(errors_correct).T
    yerr_incorrect = np.array(errors_incorrect).T

    # Create figure with adequate spacing
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set positions for bars - increased spacing
    x = np.arange(len(labels))
    width = 0.35  # Increased bar width

    # Create bars with proper spacing
    rects1 = ax.bar(x - width/2, mean_correct, width, yerr=yerr_correct, capsize=5,
                   label='Mean Correct Tokens', color='forestgreen')
    rects2 = ax.bar(x + width/2, mean_incorrect, width, yerr=yerr_incorrect, capsize=5,
                   label='Mean Incorrect Tokens', color='firebrick')

    # Add labels and formatting
    ax.set_ylabel('Mean Cumulative Tokens per Run')
    ax.set_title(f'Token Assistance Summary {title_suffix}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Add value labels with better positioning
    for i, rect in enumerate(rects1):
        height = rect.get_height()
        ax.annotate(f'{mean_correct[i]:.1f}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    for i, rect in enumerate(rects2):
        height = rect.get_height()
        ax.annotate(f'{mean_incorrect[i]:.1f}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    # Add grid for readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Ensure adequate spacing
    plt.tight_layout()

    # Save figure
    save_path = f"agent_model_results/assistance_bars_{title_suffix}.png"
    save_path = save_path.replace('(','').replace(')','').replace('=','_')
    try:
        plt.savefig(save_path)
    except Exception as e:
        print(f"Error saving plot: {e}")
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
    save_path = f"agent_model_results/retainment_comparison_{title_suffix}.png"
    plt.savefig(save_path.replace('(','').replace(')','').replace('=','_'))
    plt.close(fig)

    return fig

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

    # Create the figure object explicitly
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot bars for Exploitative
    ax.bar(x_pos - width/2, mean_shares_exploit, yerr=error_bars_exploit, capsize=5, color='skyblue', label='Mean Exploit Share') # Grouped bar position
    # Plot bars for Exploratory
    ax.bar(x_pos + width/2, mean_shares_explor, yerr=error_bars_explor, capsize=5, color='springgreen', label='Mean Explor Share') # Grouped bar position

    # Add bars for Exploratory if calculated (use x_pos + width/2 etc. for grouping)

    ax.set_xlabel("Share Exploitative Agents")
    ax.set_ylabel("Share of Correctly Targeted Tokens")
    ax.set_title("Correct Token Share vs. Agent Mix (Mean & IQR)") # Combined title
    ax.set_xticks(x_pos)
    ax.set_xticklabels(share_values) # Set x-axis labels to be the share values
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Save instead of displaying
    save_path = f"agent_model_results/correct_token_shares.png"
    plt.savefig(save_path)
    plt.close(fig)

    return fig


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

    fig = plt.figure(figsize=(8, 5))
    save_path = f"agent_model_results/belief_error_evolution_{title_suffix}.png"
    plt.savefig(save_path.replace('(','').replace(')','').replace('=','_'))
    plt.close(fig)

    return fig

# ---  Plot Function 5 ---
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
    save_path = f"agent_model_results/belief_variance_evolution_{title_suffix}.png"
    plt.savefig(save_path.replace('(','').replace(')','').replace('=','_'))
    plt.close(fig)

    return fig

# ---  Plot Function 6 ---
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
    save_path = f"agent_model_results/unmet_need_evolution_{title_suffix}.png"
    plt.savefig(save_path.replace('(','').replace(')','').replace('=','_'))
    plt.close(fig)

    return fig

def plot_experiment_c_comprehensive(results_c, dynamics_values, shock_values):
    """Create comprehensive visualization for experiment C with PROPERLY ISOLATED data for each scenario"""
    
    print("Starting comprehensive plot creation...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(18, 15), constrained_layout=True)
    axes = axes.flatten()
    
    # Initialize all metrics matrices with NaN values
    matrices = {}
    for name in ['correct_ratio', 'mae', 'unmet_needs', 'seci', 'aeci', 'ai_trust', 'component_seci']:
        matrices[name] = np.full((len(dynamics_values), len(shock_values)), np.nan)
       
    # Debug print to verify all combinations
    print("\nScenarios to process:")
    for i, dd in enumerate(dynamics_values):
        for j, sm in enumerate(shock_values):
            print(f"  Dynamics={dd}, Shock={sm}")
    
    # Explicitly process each cell in the matrices
    print("\nExtracting metrics for each cell:")
    for i, dd in enumerate(dynamics_values):
        for j, sm in enumerate(shock_values):
            key = (dd, sm)
            print(f"\nParameters: Dynamics={dd}, Shock={sm}")
            
            # Skip if no data for this combination
            if key not in results_c:
                print(f"  No data available for this combination")
                continue
                
            res = results_c[key]
            
            # 1. Calculate correct token ratio
            if "raw_assist_counts" in res:
                try:
                    raw_counts = res["raw_assist_counts"]
                    exploit_correct = sum(raw_counts.get("exploit_correct", [0]))
                    exploit_incorrect = sum(raw_counts.get("exploit_incorrect", [0]))
                    explor_correct = sum(raw_counts.get("explor_correct", [0]))
                    explor_incorrect = sum(raw_counts.get("explor_incorrect", [0]))
                    
                    total_correct = exploit_correct + explor_correct
                    total_tokens = total_correct + exploit_incorrect + explor_incorrect
                    
                    if total_tokens > 0:
                        matrices['correct_ratio'][i, j] = total_correct / total_tokens
                        print(f"  Correct ratio: {matrices['correct_ratio'][i, j]:.4f}")
                    else:
                        print("  No tokens sent")
                except Exception as e:
                    print(f"  Error calculating correct ratio: {e}")
            else:
                print("  No raw_assist_counts data")
            
            # 2. Calculate average MAE
            if "belief_error" in res and isinstance(res["belief_error"], np.ndarray):
                try:
                    if res["belief_error"].ndim >= 3 and res["belief_error"].shape[1] > 0:
                        # Average MAE across all runs, at last tick, for both agent types
                        mae_data = res["belief_error"][:, -1, 1:3]  # Last tick, exploit & explor columns
                        matrices['mae'][i, j] = np.nanmean(mae_data)
                        print(f"  Average MAE: {matrices['mae'][i, j]:.4f}")
                    else:
                        print(f"  Invalid belief_error shape: {res['belief_error'].shape}")
                except Exception as e:
                    print(f"  Error calculating MAE: {e}")
            else:
                print("  No belief_error data")
            
            # 3. Calculate final unmet needs
            if "unmet_needs_evol" in res:
                try:
                    final_unmet = []
                    for run_data in res["unmet_needs_evol"]:
                        if run_data is not None and len(run_data) > 0:
                            final_unmet.append(run_data[-1])
                    if final_unmet:
                        matrices['unmet_needs'][i, j] = np.mean(final_unmet)
                        print(f"  Final unmet needs: {matrices['unmet_needs'][i, j]:.4f}")
                    else:
                        print("  No final unmet needs data")
                except Exception as e:
                    print(f"  Error calculating unmet needs: {e}")
            else:
                print("  No unmet_needs_evol data")
            
            # 4. Calculate final SECI
            if "seci" in res and isinstance(res["seci"], np.ndarray):
                try:
                    if res["seci"].ndim >= 3 and res["seci"].shape[1] > 0:
                        # Average SECI at last tick, for both agent types
                        seci_data = res["seci"][:, -1, 1:3]  # Last tick, exploit & explor columns
                        matrices['seci'][i, j] = np.nanmean(seci_data)
                        print(f"  Average SECI: {matrices['seci'][i, j]:.4f}")
                    else:
                        print(f"  Invalid seci shape: {res['seci'].shape}")
                except Exception as e:
                    print(f"  Error calculating SECI: {e}")
            else:
                print("  No seci data")
            
            # 5. Calculate final AECI (AI Call Ratio)
            if "aeci" in res and isinstance(res["aeci"], np.ndarray):
                try:
                    if res["aeci"].ndim >= 3 and res["aeci"].shape[1] > 0:
                        # Average AECI at last tick, for both agent types
                        aeci_data = res["aeci"][:, -1, 1:3]  # Last tick, exploit & explor columns
                        matrices['aeci'][i, j] = np.nanmean(aeci_data)
                        print(f"  Average AECI: {matrices['aeci'][i, j]:.4f}")
                    else:
                        print(f"  Invalid aeci shape: {res['aeci'].shape}")
                except Exception as e:
                    print(f"  Error calculating AECI: {e}")
            else:
                print("  No aeci data")
            
            # 6. Calculate final AI Trust
            if "trust_stats" in res and isinstance(res["trust_stats"], np.ndarray):
                try:
                    if res["trust_stats"].ndim >= 3 and res["trust_stats"].shape[1] > 0:
                        # Get AI trust for exploitative agents at last tick
                        ai_trust_data = res["trust_stats"][:, -1, 1]  # Last tick, AI trust column
                        matrices['ai_trust'][i, j] = np.nanmean(ai_trust_data)
                        print(f"  Average AI Trust: {matrices['ai_trust'][i, j]:.4f}")
                    else:
                        print(f"  Invalid trust_stats shape: {res['trust_stats'].shape}")
                except Exception as e:
                    print(f"  Error calculating AI Trust: {e}")
            else:
                print("  No trust_stats data")

            # 7. Calculate Component SECI
            if "component_seci" in res and isinstance(res["component_seci"], np.ndarray):
                try:
                    if res["component_seci"].ndim >= 3 and res["component_seci"].shape[1] > 0:
                        # Average Component SECI at last tick
                        comp_seci_data = res["component_seci"][:, -1, 1]  # Last tick, value column
                        matrices['component_seci'][i, j] = np.nanmean(comp_seci_data)
                        print(f"  Average Component SECI: {matrices['component_seci'][i, j]:.4f}")
                    else:
                        print(f"  Invalid component_seci shape: {res['component_seci'].shape}")
                except Exception as e:
                    print(f"  Error calculating Component SECI: {e}")
            else:
                # Alternative approach: extract from component_seci_data list if available
                if "component_seci_data" in res and res["component_seci_data"]:
                    try:
                        # Collect values from all components in last tick of each run
                        comp_values = []
                        for run_data in res["component_seci_data"]:
                            if run_data:
                                # Get the last tick's data
                                last_tick_data = run_data[-1]
                                if isinstance(last_tick_data, dict) and 'avg_component_seci' in last_tick_data:
                                    comp_values.append(last_tick_data['avg_component_seci'])
                        
                        if comp_values:
                            matrices['component_seci'][i, j] = np.mean(comp_values)
                            print(f"  Average Component SECI (from list): {matrices['component_seci'][i, j]:.4f}")
                    except Exception as e:
                        print(f"  Error calculating Component SECI from list: {e}")
                else:
                    print("  No component_seci data")
    
    # Debug: Print all matrices to verify different values
    for name, matrix in matrices.items():
        print(f"\nMatrix for {name}:")
        for row in matrix:
            print("  " + " ".join(f"{val:.2f}" if not np.isnan(val) else "N/A" for val in row))
    
    # Create heatmaps for each metric
    heatmap_configs = [
        {'name': 'correct_ratio', 'title': 'Correct Token Ratio', 'cmap': 'RdYlGn', 'idx': 0},
        {'name': 'mae', 'title': 'Belief Error (MAE)', 'cmap': 'RdYlGn_r', 'idx': 1},
        {'name': 'unmet_needs', 'title': 'Unmet Needs', 'cmap': 'RdYlGn_r', 'idx': 2},
        {'name': 'seci', 'title': 'Social Echo Chamber Index', 'cmap': 'RdYlBu', 'idx': 3},
        {'name': 'aeci', 'title': 'AI Usage (AECI)', 'cmap': 'RdYlBu', 'idx': 4},
        {'name': 'ai_trust', 'title': 'AI Trust', 'cmap': 'RdYlBu', 'idx': 5},
        {'name': 'component_seci', 'title': 'Component SECI', 'cmap': 'RdYlBu', 'idx': 7}
    ]
    
    print("\nCreating heatmap plots...")
    for config in heatmap_configs:
        ax = axes[config['idx']]
        matrix = matrices[config['name']]
        
        # Create masked array for NaN handling
        masked_matrix = np.ma.masked_where(np.isnan(matrix), matrix)
        
        # Get valid range for the colormap
        valid_values = matrix[~np.isnan(matrix)]
        if len(valid_values) > 0:
            vmin, vmax = np.min(valid_values), np.max(valid_values)
            # Ensure range isn't zero
            if abs(vmax - vmin) < 1e-6:  # Close to zero range
                center = (vmin + vmax) / 2
                vmin, vmax = center * 0.9, center * 1.1
                if abs(center) < 1e-6:  # Center close to zero
                    vmin, vmax = -0.1, 0.1
        else:
            vmin, vmax = 0, 1
        
        # Create heatmap
        im = ax.imshow(masked_matrix, cmap=config['cmap'], aspect='auto', vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(config['title'])
        
        # Add value annotations
        for i in range(len(dynamics_values)):
            for j in range(len(shock_values)):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f'{matrix[i, j]:.2f}', 
                           ha='center', va='center', color='black', 
                           fontsize=9, fontweight='bold')
        
        # Set labels and title
        ax.set_xticks(range(len(shock_values)))
        ax.set_yticks(range(len(dynamics_values)))
        ax.set_xticklabels(shock_values)
        ax.set_yticklabels(dynamics_values)
        ax.set_xlabel('Shock Magnitude')
        ax.set_ylabel('Disaster Dynamics')
        ax.set_title(config['title'])
    
    # Calculate performance score for performance bar chart
    print("\nCalculating performance scores...")
    performance_data = []
    labels = []
    
    # Normalization values for performance calculation
    max_mae = np.nanmax(matrices['mae']) if not np.all(np.isnan(matrices['mae'])) else 1.0
    max_unmet = np.nanmax(matrices['unmet_needs']) if not np.all(np.isnan(matrices['unmet_needs'])) else 1.0
    
    # Create a proper list of scenarios
    scenarios = []
    for i, dd in enumerate(dynamics_values):
        for j, sm in enumerate(shock_values):
            scenarios.append((i, j, dd, sm))
    
    # Calculate performance for each scenario
    for i, j, dd, sm in scenarios:
        # Skip if missing essential data
        if np.isnan(matrices['correct_ratio'][i, j]) or np.isnan(matrices['mae'][i, j]) or np.isnan(matrices['unmet_needs'][i, j]):
            performance_data.append(0)
            print(f"  Setting zero performance for D={dd}, S={sm} (missing data)")
        else:
            # Calculate weighted performance score
            performance = (
                matrices['correct_ratio'][i, j] * 0.4 +                  # 40% weight on correct tokens
                (1 - matrices['mae'][i, j] / max_mae) * 0.3 +            # 30% weight on accuracy (inverted)
                (1 - matrices['unmet_needs'][i, j] / max_unmet) * 0.3    # 30% weight on meeting needs (inverted)
            )
            performance_data.append(performance)
            print(f"  Performance for D={dd}, S={sm}: {performance:.4f}")
        
        labels.append(f'D={dd},S={sm}')
    
    # Debug print performance data
    print("\nPerformance data:")
    for label, perf in zip(labels, performance_data):
        print(f"  {label}: {perf:.4f}")
    
    # Create performance bar chart
    if len(performance_data) > 0 and len(axes) > 6:
        ax_bar = axes[6]
        
        # Use viridis colormap based on performance values
        norm = plt.Normalize(0, max(performance_data)) if max(performance_data) > 0 else plt.Normalize(0, 1)
        colors = plt.cm.viridis(norm(performance_data))
        
        # Create bars with varying heights
        bars = ax_bar.bar(range(len(performance_data)), performance_data, color=colors)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, performance_data)):
            height = bar.get_height()
            if height > 0:  # Only label non-zero bars
                ax_bar.text(i, height + 0.01, f'{value:.2f}', 
                           ha='center', va='bottom')
        
        ax_bar.set_xticks(range(len(labels)))
        ax_bar.set_xticklabels(labels, rotation=45)
        ax_bar.set_ylabel('Overall Performance Score')
        ax_bar.set_title('Combined Performance Score (40% Correct Ratio + 30% Accuracy + 30% Effectiveness)')
        ax_bar.grid(axis='y', alpha=0.3)
    
    # Hide any unused axes
    for i in range(7, len(axes)):
        axes[i].axis('off')
    
    plt.savefig("agent_model_results/experiment_c_comprehensive.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("Comprehensive plot saved successfully.")

def plot_experiment_c_evolution(results_c, dynamics_values, shock_values):
    """Plot time evolution for key scenarios in Experiment C with robust error handling"""
    
    print("Starting evolution plot creation...")
    
    # Select first and last scenarios to compare
    try:
        scenarios_to_plot = []
        if len(dynamics_values) > 0 and len(shock_values) > 0:
            # Select first and last dynamics with first shock value
            scenarios_to_plot = [(dynamics_values[0], shock_values[0])]
            if len(dynamics_values) > 1:
                scenarios_to_plot.append((dynamics_values[-1], shock_values[0]))
        
        if not scenarios_to_plot:
            print("Not enough data to create evolution plots")
            return
        
        # Create figure and subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()  # Flatten for easier indexing
        
        colors = ['blue', 'red', 'green', 'orange']
        
        # Define metrics to plot
        metrics_to_plot = [
            {'key': 'trust_stats', 'col_idx': [1], 'title': 'AI Trust Evolution', 
             'ylabel': 'AI Trust', 'style': '--'},
            {'key': 'seci', 'col_idx': [1], 'title': 'SECI Evolution', 
             'ylabel': 'SECI Value', 'style': '-'},
            {'key': 'aeci', 'col_idx': [1], 'title': 'AECI Evolution', 
             'ylabel': 'AECI Value', 'style': '-'},
            {'key': 'belief_error', 'col_idx': [1], 'title': 'Belief Error Evolution', 
             'ylabel': 'MAE', 'style': '-'}
        ]
        
        # Create plots
        for plot_idx, metric in enumerate(metrics_to_plot):
            if plot_idx >= len(axes):
                print(f"Warning: Not enough axes for metric {metric['key']}")
                continue
                
            ax = axes[plot_idx]
            plots_created = False
            
            # Get label for metric
            metric_label = metric.get('title', metric['key'])
            
            for scenario_idx, (dd, sm) in enumerate(scenarios_to_plot):
                res_key = (dd, sm)
                if res_key not in results_c:
                    print(f"Data missing for scenario {res_key}")
                    continue
                    
                res = results_c[res_key]
                data_key = metric['key']
                
                # Skip if data doesn't exist
                if data_key not in res or not isinstance(res[data_key], np.ndarray) or res[data_key].size == 0:
                    print(f"No valid {data_key} data for scenario {res_key}")
                    continue
                
                data_array = res[data_key]
                
                # Skip if array has wrong shape
                if data_array.ndim < 3 or data_array.shape[1] == 0:
                    print(f"Invalid array shape for {data_key}: {data_array.shape}")
                    continue
                
                # Get ticks from first column
                ticks = np.arange(data_array.shape[1])  # Just use tick indices
                
                # Plot each column specified
                for col_idx_pos, col_idx in enumerate(metric['col_idx']):
                    # Skip if column index is invalid
                    if col_idx >= data_array.shape[2]:
                        print(f"Column index {col_idx} out of bounds for {data_key}")
                        continue
                    
                    # Get mean values and handle NaN
                    mean_values = np.nanmean(data_array[:, :, col_idx], axis=0)
                    
                    # Only plot if we have valid data
                    if np.any(~np.isnan(mean_values)):
                        label = f'D={dd}, S={sm}, Type={(col_idx)}'
                        color_idx = (scenario_idx * len(metric['col_idx']) + col_idx_pos) % len(colors)
                        
                        ax.plot(ticks, mean_values, 
                               color=colors[color_idx],
                               linestyle=metric['style'],
                               linewidth=2,
                               label=label)
                        plots_created = True
            
            # Set title and labels
            ax.set_title(metric['title'])
            ax.set_xlabel('Tick')
            ax.set_ylabel(metric['ylabel'])
            
            # Only add legend if plots were created
            if plots_created:
                ax.legend()
            else:
                ax.text(0.5, 0.5, 'No data to display', 
                       ha='center', va='center', transform=ax.transAxes)
            
            ax.grid(True, alpha=0.3)
            
            # Set y-limits based on the metric type
            if metric['key'] in ['seci', 'aeci', 'trust_stats']:
                ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig("agent_model_results/experiment_c_evolution.png", dpi=300)
        plt.close()
    
    except Exception as e:
        print(f"Error in plot_experiment_c_evolution: {e}")
        import traceback
        traceback.print_exc()

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
# NEW: Advanced Bubble Mechanics Visualizations
#########################################

def plot_phase_diagram_bubbles(results_dict, param_values, param_name="AI Alignment"):
    """
    Phase diagram showing bubble regimes across parameter space.

    Creates a multi-panel heatmap showing:
    1. Social bubble strength (SECI)
    2. AI bubble strength (AECI-Var)
    3. Information diversity
    4. Dominant information source

    Args:
        results_dict: Dictionary mapping parameter values to aggregated results
        param_values: List of parameter values (e.g., alignment levels)
        param_name: Name of the parameter being varied
    """
    print(f"\n=== Generating Phase Diagram for {param_name} ===")

    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Filter Bubble Phase Diagram vs {param_name}", fontsize=16, fontweight='bold')

        # Prepare data matrices for heatmaps
        num_params = len(param_values)

        # Extract final values for each metric
        seci_exploit_final = []
        seci_explor_final = []
        aeci_var_final = []
        info_div_exploit_final = []
        info_div_explor_final = []
        ai_trust_final = []
        friend_trust_final = []

        for param_val in param_values:
            res = results_dict.get(param_val, {})

            # SECI (Social Echo Chamber Index)
            seci = res.get("seci", np.array([]))
            if seci.ndim >= 3 and seci.shape[1] > 0:
                seci_exploit_final.append(np.mean(seci[:, -1, 1]))  # Final tick, exploit column
                seci_explor_final.append(np.mean(seci[:, -1, 2]))   # Final tick, explor column
            else:
                seci_exploit_final.append(0)
                seci_explor_final.append(0)

            # AECI-Var (AI Echo Chamber - Variance based)
            aeci_var = res.get("aeci_variance", np.array([]))
            if isinstance(aeci_var, np.ndarray) and aeci_var.size > 0:
                try:
                    if aeci_var.ndim == 3:  # (runs, ticks, 2)
                        final_values = aeci_var[:, -1, 1]  # Last tick, value column
                        aeci_var_final.append(np.mean(final_values))
                    elif aeci_var.ndim == 2:  # (ticks, 2)
                        aeci_var_final.append(aeci_var[-1, 1])  # Last tick, value column
                    else:
                        aeci_var_final.append(0)
                except:
                    aeci_var_final.append(0)
            else:
                aeci_var_final.append(0)

            # Info Diversity (Shannon Entropy)
            info_div = res.get("info_diversity", np.array([]))
            if info_div.ndim >= 3 and info_div.shape[1] > 0:
                info_div_exploit_final.append(np.mean(info_div[:, -1, 1]))  # Final tick, exploit
                info_div_explor_final.append(np.mean(info_div[:, -1, 2]))   # Final tick, explor
            else:
                info_div_exploit_final.append(0)
                info_div_explor_final.append(0)

            # Trust levels
            trust = res.get("trust_stats", np.array([]))
            if trust.ndim >= 3 and trust.shape[1] > 0:
                # Avg AI trust across both agent types
                ai_trust_final.append(np.mean([trust[:, -1, 1], trust[:, -1, 4]]))  # AI exp, AI expl
                # Avg Friend trust across both agent types
                friend_trust_final.append(np.mean([trust[:, -1, 2], trust[:, -1, 5]]))  # Friend exp, Friend expl
            else:
                ai_trust_final.append(0)
                friend_trust_final.append(0)

        # Panel 1: Social Bubble Strength (SECI)
        ax = axes[0, 0]
        seci_combined = [(seci_exploit_final[i] + seci_explor_final[i]) / 2 for i in range(num_params)]

        # Debug output
        print(f"\nSocial Bubble Strength (SECI) values:")
        for i, (param, val) in enumerate(zip(param_values, seci_combined)):
            print(f"  {param_name}={param:.2f}: SECI={val:.4f}")

        bars = ax.barh(range(num_params), seci_combined, height=0.6,
                       color=['red' if v < -0.2 else 'yellow' if v < 0.1 else 'green' for v in seci_combined],
                       edgecolor='black', linewidth=0.5)

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, seci_combined)):
            ax.text(val, i, f' {val:.3f}', va='center', fontsize=9)

        ax.set_yticks(range(num_params))
        ax.set_yticklabels([f"{v:.2f}" for v in param_values])
        ax.set_xlabel("SECI (Social Echo Chamber Index)", fontsize=11)
        ax.set_ylabel(param_name, fontsize=11)
        ax.set_title("Social Bubble Strength\n(Red=Strong, Yellow=Moderate, Green=Weak)", fontsize=12)
        ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(True, axis='x', alpha=0.3)

        # Panel 2: AI Bubble Strength (AECI-Var)
        ax = axes[0, 1]

        # Debug output
        print(f"\nAI Bubble Strength (AECI-Var) values:")
        for i, (param, val) in enumerate(zip(param_values, aeci_var_final)):
            print(f"  {param_name}={param:.2f}: AECI-Var={val:.4f}")

        bars = ax.barh(range(num_params), aeci_var_final, height=0.6,
                      color=['blue' if v > 0.1 else 'white' if v > -0.1 else 'red' for v in aeci_var_final],
                      edgecolor='black', linewidth=0.5)

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, aeci_var_final)):
            ax.text(val, i, f' {val:.3f}', va='center', fontsize=9)

        ax.set_yticks(range(num_params))
        ax.set_yticklabels([f"{v:.2f}" for v in param_values])
        ax.set_xlabel("AECI-Var (AI Echo Chamber Index)", fontsize=11)
        ax.set_ylabel(param_name, fontsize=11)
        ax.set_title("AI Bubble Strength\n(Blue=Diversifies, Red=Echo Chamber)", fontsize=12)
        ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(True, axis='x', alpha=0.3)

        # Panel 3: Information Diversity
        ax = axes[1, 0]

        # Debug output
        print(f"\nInformation Diversity (Shannon Entropy) values:")
        for i, (param, exploit, explor) in enumerate(zip(param_values, info_div_exploit_final, info_div_explor_final)):
            print(f"  {param_name}={param:.2f}: Exploit={exploit:.4f}, Explor={explor:.4f}")

        x = np.arange(num_params)
        width = 0.35
        bars1 = ax.barh(x - width/2, info_div_exploit_final, width, label='Exploitative', alpha=0.8, color='coral', edgecolor='black', linewidth=0.5)
        bars2 = ax.barh(x + width/2, info_div_explor_final, width, label='Exploratory', alpha=0.8, color='skyblue', edgecolor='black', linewidth=0.5)

        # Add value labels on bars
        for i, val in enumerate(info_div_exploit_final):
            if val > 0.01:  # Only show label if value is significant
                ax.text(val, i - width/2, f' {val:.2f}', va='center', fontsize=8)
        for i, val in enumerate(info_div_explor_final):
            if val > 0.01:  # Only show label if value is significant
                ax.text(val, i + width/2, f' {val:.2f}', va='center', fontsize=8)

        ax.set_yticks(range(num_params))
        ax.set_yticklabels([f"{v:.2f}" for v in param_values])
        ax.set_xlabel("Shannon Entropy (bits)", fontsize=11)
        ax.set_ylabel(param_name, fontsize=11)
        ax.set_title("Information Source Diversity\n(Higher=More Diverse Sources)", fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, axis='x', alpha=0.3)

        # Panel 4: Dominant Source (AI vs Friends)
        ax = axes[1, 1]
        trust_ratio = [ai_trust_final[i] - friend_trust_final[i] for i in range(num_params)]

        # Debug output
        print(f"\nDominant Information Source (Trust Difference) values:")
        for i, (param, ai_tr, fr_tr, ratio) in enumerate(zip(param_values, ai_trust_final, friend_trust_final, trust_ratio)):
            print(f"  {param_name}={param:.2f}: AI={ai_tr:.4f}, Friend={fr_tr:.4f}, Diff={ratio:.4f}")

        bars = ax.barh(range(num_params), trust_ratio, height=0.6,
                      color=['orange' if v > 0.1 else 'gray' if abs(v) <= 0.1 else 'purple' for v in trust_ratio],
                      edgecolor='black', linewidth=0.5)

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, trust_ratio)):
            label_x = val if val > 0 else val  # Position label at end of bar
            ax.text(label_x, i, f' {val:.3f}', va='center', fontsize=9)

        ax.set_yticks(range(num_params))
        ax.set_yticklabels([f"{v:.2f}" for v in param_values])
        ax.set_xlabel("Trust Difference (AI - Friends)", fontsize=11)
        ax.set_ylabel(param_name, fontsize=11)
        ax.set_title("Dominant Information Source\n(Orange=AI, Purple=Friends, Gray=Mixed)", fontsize=12)
        ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(True, axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig("agent_model_results/phase_diagram_bubbles.png", dpi=300, bbox_inches='tight')
        print("Phase diagram saved successfully")
        plt.close()

    except Exception as e:
        print(f"Error in plot_phase_diagram_bubbles: {e}")
        import traceback
        traceback.print_exc()


def plot_tipping_point_waterfall(results_dict, param_values, param_name="AI Alignment"):
    """
    Waterfall diagram showing sequential behavioral transitions (tipping points).

    Identifies and visualizes when:
    1. AI trust exceeds friend trust (aggregate)
    2. Social bubble breaks (SECI crosses zero)
    3. AI bubble breaks (AECI crosses zero)
    4. Exploitative agents prefer AI over friends
    5. Information diversity increases significantly

    Args:
        results_dict: Dictionary mapping parameter values to aggregated results
        param_values: List of parameter values (sorted)
        param_name: Name of the parameter being varied
    """
    print(f"\n=== Generating Tipping Point Waterfall for {param_name} ===")

    try:
        fig, ax = plt.subplots(figsize=(14, 8))

        # Track transitions for each metric
        tipping_points = {
            'AI Trust > Friend Trust': None,
            'Social Bubble Breaks (SECI→0)': None,
            'AI Bubble Breaks (AECI→0)': None,
            'Exploiters Prefer AI': None,
            'Info Diversity Surge': None
        }

        # Baseline values for comparison
        prev_ai_trust = 0
        prev_friend_trust = 0
        prev_seci = 0
        prev_aeci_var = 0
        prev_exploit_ai_pref = 0
        prev_info_div = 0

        # Diagnostic output
        print(f"\n=== Tipping Point Detection Diagnostics ===")
        print(f"Looking for transitions across {len(param_values)} parameter values: {param_values}")

        for i, param_val in enumerate(sorted(param_values)):
            res = results_dict.get(param_val, {})

            # Calculate current metrics
            trust = res.get("trust_stats", np.array([]))
            if trust.ndim >= 3 and trust.shape[1] > 0:
                curr_ai_trust = np.mean([trust[:, -1, 1], trust[:, -1, 4]])
                curr_friend_trust = np.mean([trust[:, -1, 2], trust[:, -1, 5]])
            else:
                curr_ai_trust = prev_ai_trust
                curr_friend_trust = prev_friend_trust

            seci = res.get("seci", np.array([]))
            if seci.ndim >= 3 and seci.shape[1] > 0:
                curr_seci = np.mean([seci[:, -1, 1], seci[:, -1, 2]])
            else:
                curr_seci = prev_seci

            aeci_var = res.get("aeci_variance", np.array([]))
            if isinstance(aeci_var, np.ndarray) and aeci_var.size > 0:
                try:
                    if aeci_var.ndim == 3:
                        curr_aeci_var = np.mean(aeci_var[:, -1, 1])
                    elif aeci_var.ndim == 2:
                        curr_aeci_var = aeci_var[-1, 1]
                    else:
                        curr_aeci_var = prev_aeci_var
                except:
                    curr_aeci_var = prev_aeci_var
            else:
                curr_aeci_var = prev_aeci_var

            # Check for exploiter preference (from AECI data)
            aeci = res.get("aeci", np.array([]))
            if aeci.ndim >= 3 and aeci.shape[1] > 0:
                curr_exploit_ai_pref = np.mean(aeci[:, -1, 1])  # Exploitative AECI
            else:
                curr_exploit_ai_pref = prev_exploit_ai_pref

            # Info diversity
            info_div = res.get("info_diversity", np.array([]))
            if info_div.ndim >= 3 and info_div.shape[1] > 0:
                curr_info_div = np.mean([info_div[:, -1, 1], info_div[:, -1, 2]])
            else:
                curr_info_div = prev_info_div

            # Diagnostic output for each parameter value
            print(f"\n{param_name}={param_val:.2f}:")
            print(f"  AI trust: {curr_ai_trust:.4f}, Friend trust: {curr_friend_trust:.4f}, Diff: {curr_ai_trust - curr_friend_trust:.4f}")
            print(f"  SECI: {curr_seci:.4f}, AECI-Var: {curr_aeci_var:.4f}")
            print(f"  Exploit AI pref: {curr_exploit_ai_pref:.4f}, Info div: {curr_info_div:.4f}")

            # Detect tipping points (only if not already detected)
            if i > 0:
                print(f"  Checking transitions from previous parameter value:")
                print(f"    AI > Friend? prev: AI={prev_ai_trust:.4f} < Friend={prev_friend_trust:.4f} → curr: AI={curr_ai_trust:.4f} > Friend={curr_friend_trust:.4f}? {prev_ai_trust < prev_friend_trust and curr_ai_trust > curr_friend_trust}")
                print(f"    SECI→0? prev: {prev_seci:.4f} < -0.05 → curr: {curr_seci:.4f} > -0.05? {prev_seci < -0.05 and curr_seci > -0.05}")
                print(f"    AECI→0? prev: {prev_aeci_var:.4f} < -0.05 → curr: {curr_aeci_var:.4f} > -0.05? {prev_aeci_var < -0.05 and curr_aeci_var > -0.05}")
                print(f"    Exploit prefer AI? prev: {prev_exploit_ai_pref:.4f} < 0.5 → curr: {curr_exploit_ai_pref:.4f} >= 0.5? {prev_exploit_ai_pref < 0.5 and curr_exploit_ai_pref >= 0.5}")
                print(f"    Info surge? prev: {prev_info_div:.4f} → curr: {curr_info_div:.4f} (ratio: {curr_info_div/prev_info_div if prev_info_div > 0 else 0:.2f}x)? {prev_info_div > 0 and curr_info_div / prev_info_div > 1.5}")

                # AI trust overtakes friend trust
                if tipping_points['AI Trust > Friend Trust'] is None:
                    if prev_ai_trust < prev_friend_trust and curr_ai_trust > curr_friend_trust:
                        tipping_points['AI Trust > Friend Trust'] = (param_values[i-1] + param_val) / 2

                # SECI crosses zero (social bubble breaks)
                if tipping_points['Social Bubble Breaks (SECI→0)'] is None:
                    if prev_seci < -0.05 and curr_seci > -0.05:
                        tipping_points['Social Bubble Breaks (SECI→0)'] = (param_values[i-1] + param_val) / 2

                # AECI crosses zero (AI bubble breaks)
                if tipping_points['AI Bubble Breaks (AECI→0)'] is None:
                    if prev_aeci_var < -0.05 and curr_aeci_var > -0.05:
                        tipping_points['AI Bubble Breaks (AECI→0)'] = (param_values[i-1] + param_val) / 2

                # Exploiters prefer AI (AECI > 0.5)
                if tipping_points['Exploiters Prefer AI'] is None:
                    if prev_exploit_ai_pref < 0.5 and curr_exploit_ai_pref >= 0.5:
                        tipping_points['Exploiters Prefer AI'] = (param_values[i-1] + param_val) / 2

                # Info diversity surge (50% increase)
                if tipping_points['Info Diversity Surge'] is None:
                    if prev_info_div > 0 and curr_info_div / prev_info_div > 1.5:
                        tipping_points['Info Diversity Surge'] = (param_values[i-1] + param_val) / 2
            else:
                print(f"  (First parameter value - establishing baseline)")

            # Update previous values
            prev_ai_trust = curr_ai_trust
            prev_friend_trust = curr_friend_trust
            prev_seci = curr_seci
            prev_aeci_var = curr_aeci_var
            prev_exploit_ai_pref = curr_exploit_ai_pref
            prev_info_div = curr_info_div

        # Plot tipping points as vertical lines
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        y_positions = [0.8, 0.6, 0.4, 0.2, 0.1]

        for (label, tp_value), color, y_pos in zip(tipping_points.items(), colors, y_positions):
            if tp_value is not None:
                ax.axvline(tp_value, color=color, linestyle='--', linewidth=2.5, alpha=0.8, label=label)
                ax.text(tp_value, y_pos, f'{tp_value:.3f}',
                       rotation=90, verticalalignment='bottom', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

        # Formatting
        ax.set_xlim(min(param_values), max(param_values))
        ax.set_ylim(0, 1)
        ax.set_xlabel(param_name, fontsize=13, fontweight='bold')
        ax.set_ylabel("Tipping Point Cascade", fontsize=13, fontweight='bold')
        ax.set_title(f"Sequential Behavioral Transitions vs {param_name}\n(Critical Points Where System Dynamics Shift)",
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
        ax.grid(True, axis='x', alpha=0.3, linestyle=':')
        ax.set_yticks([])

        # Add shaded regions for qualitative regimes
        if len([v for v in tipping_points.values() if v is not None]) > 0:
            first_tp = min([v for v in tipping_points.values() if v is not None])
            last_tp = max([v for v in tipping_points.values() if v is not None])

            ax.axvspan(min(param_values), first_tp, alpha=0.1, color='red', label='Pre-transition')
            ax.axvspan(first_tp, last_tp, alpha=0.1, color='yellow', label='Transition zone')
            ax.axvspan(last_tp, max(param_values), alpha=0.1, color='green', label='Post-transition')

        plt.tight_layout()
        plt.savefig("agent_model_results/tipping_point_waterfall.png", dpi=300, bbox_inches='tight')
        print("Tipping point waterfall saved successfully")
        print(f"\nDetected tipping points:")
        for label, value in tipping_points.items():
            if value is not None:
                print(f"  {label}: {param_name} = {value:.3f}")
            else:
                print(f"  {label}: Not detected")
        plt.close()

    except Exception as e:
        print(f"Error in plot_tipping_point_waterfall: {e}")
        import traceback
        traceback.print_exc()


def debug_aeci_variance_data(results_dict, title_suffix=""):
    """Inspects and prints detailed AECI variance data structure"""
    print(f"\n=== AECI Variance Data Inspection for {title_suffix} ===")
    
    # Get aeci_variance data
    aeci_var_data = results_dict.get("aeci_variance")
    
    # Basic data check
    if aeci_var_data is None:
        print("ERROR: aeci_variance data is None")
        return
    
    if not isinstance(aeci_var_data, np.ndarray):
        print(f"ERROR: aeci_variance isn't a numpy array (type: {type(aeci_var_data)})")
        if isinstance(aeci_var_data, list):
            print(f"  List length: {len(aeci_var_data)}")
            for i, item in enumerate(aeci_var_data[:3]):
                print(f"  Item {i}: {type(item)} - {item}")
        return
    
    # Array shape analysis
    print(f"AECI Variance array shape: {aeci_var_data.shape}")
    
    # Inspect dimensions
    if aeci_var_data.ndim >= 3:
        # Expected shape: (runs, ticks, 2) where 2nd dim is [tick, value]
        print(f"First dimension (runs): {aeci_var_data.shape[0]}")
        print(f"Second dimension (ticks): {aeci_var_data.shape[1]}")
        print(f"Third dimension (data): {aeci_var_data.shape[2]}")
        
        # Inspect first few values
        print("\nSample values:")
        for run in range(min(2, aeci_var_data.shape[0])):
            print(f"Run {run}:")
            tick_slice = slice(0, min(5, aeci_var_data.shape[1]))
            print(f"  First 5 ticks: {aeci_var_data[run, tick_slice, :]}")
            
            # Check if values are in expected range [0,1]
            if aeci_var_data.shape[2] > 1:
                values = aeci_var_data[run, :, 1]
                min_val, max_val = np.nanmin(values), np.nanmax(values)
                print(f"  Value range: [{min_val:.4f}, {max_val:.4f}]")
                print(f"  Mean value: {np.nanmean(values):.4f}")
                print(f"  Contains NaN: {np.isnan(values).any()}")
                print(f"  Contains Inf: {np.isinf(values).any()}")
    else:
        print(f"WARNING: Expected 3D array, got {aeci_var_data.ndim}D")
        # Try to analyze based on actual dimensions
        if aeci_var_data.ndim == 2:
            print("Assuming array format is (runs, values):")
            for run in range(min(2, aeci_var_data.shape[0])):
                print(f"Run {run}: {aeci_var_data[run, :]}")
        elif aeci_var_data.ndim == 1:
            print("Assuming array is a flat list of values:")
            print(f"Values: {aeci_var_data[:min(10, len(aeci_var_data))]}")
    
    print("=== End AECI Variance Data Inspection ===\n")

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
    num_runs = 10
    save_dir = "agent_model_results"
    os.makedirs(save_dir, exist_ok=True)

    ##############################################
    # Experiment A: Vary share_exploitative
    ##############################################
    share_values = [0.2, 0,5, 0.8]
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
        results_dict = results_a.get(share, {})
        title_suffix = f"({param_name_a}={share})"

        if results_dict:
            # Call consolidated plot functions
            plot_simulation_overview(results_dict, title_suffix)
            plot_echo_chamber_indices(results_dict, title_suffix)
            plot_trust_evolution(results_dict["trust_stats"], title_suffix)
        else:
            print(f"  Skipping plots for {param_name_a}={share} (missing data)")

    # --- Plot SUMMARY Comparisons Across Parameters (AFTER LOOP) ---
    print("\n--- Plotting Summary Comparisons for Experiment A ---")
    if results_a:
        plot_correct_token_shares_bars(results_a, share_values)

        # NEW: Advanced bubble mechanics visualizations
        print("\n--- Plotting Advanced Bubble Mechanics for Experiment A ---")
        plot_phase_diagram_bubbles(results_a, share_values, param_name="Share Exploitative")
        plot_tipping_point_waterfall(results_a, share_values, param_name="Share Exploitative")

    ##############################################
    # Experiment B: Vary AI Alignment Level
    ##############################################
    alignment_values = [0.0, 0.25, 0.5, 0.75, 1.0]  # Sweep
    param_name_b = "AI Alignment Tipping Point"
    file_b_pkl = os.path.join(save_dir, f"results_{param_name_b.replace(' ','_')}.pkl")

    print(f"\nRunning {param_name_b} Experiment...")
    results_b = experiment_alignment_tipping_point(base_params, alignment_values, num_runs=10)

    # Save results
    with open(file_b_pkl, "wb") as f:
        pickle.dump(results_b, f)

    # Get all alignment values (including fine-grained ones added by tipping point detection)
    all_alignment_values = sorted(list(results_b.keys()))

    # --- Plot SUMMARY Comparisons (only once, after getting all results) ---
    print(f"\n--- Plotting Summary Comparisons for {param_name_b} ---")
    if results_b:
        # Use all alignment values found in results
        plot_final_echo_indices_vs_alignment(results_b, all_alignment_values, title_suffix="Tipping Points")
        plot_average_performance_vs_alignment(results_b, all_alignment_values, title_suffix="Tipping Points")

        # Boxplot summaries
        print(f"\n--- Plotting Boxplot Summaries for {param_name_b} ---")
        plot_summary_echo_indices_vs_alignment(results_b, all_alignment_values, title_suffix="Tipping Points")
        plot_summary_performance_vs_alignment(results_b, all_alignment_values, title_suffix="Tipping Points")

        # NEW: Advanced bubble mechanics visualizations
        print(f"\n--- Plotting Advanced Bubble Mechanics for {param_name_b} ---")
        plot_phase_diagram_bubbles(results_b, all_alignment_values, param_name="AI Alignment")
        plot_tipping_point_waterfall(results_b, all_alignment_values, param_name="AI Alignment")

    ##############################################
    # Experiment C: Vary Disaster Dynamics and Shock Magnitude
    ##############################################
    # COMMENTED OUT - Focus on Experiments A and B
    # print("\n=== STARTING EXPERIMENT C ===")
    #
    # try:
    #     dynamics_values = [1, 2, 3]
    #     shock_values = [1, 2, 3]
    #
    #     print(f"Running experiment with {len(dynamics_values)}x{len(shock_values)} parameter combinations...")
    #     results_c = experiment_disaster_dynamics(base_params, dynamics_values, shock_values, num_runs)
    #
    #     print(f"Got results for {len(results_c)} parameter combinations")
    #
    #     # Debug the structure of results_c
    #     print("Parameter combinations in results_c:")
    #     for key in sorted(results_c.keys()):
    #         print(f"  {key}: {type(results_c[key])}")
    #
    #     # Generate visualizations with robust error handling
    #     print("\n--- Creating Visualizations ---")
    #
    #     try:
    #         print("Generating comprehensive analysis...")
    #         plot_experiment_c_comprehensive(results_c, dynamics_values, shock_values)
    #         print("Comprehensive analysis complete")
    #     except Exception as e:
    #         print(f"Error in comprehensive analysis: {e}")
    #         import traceback
    #         traceback.print_exc()
    #
    #     try:
    #         print("Generating evolution plots...")
    #         plot_experiment_c_evolution(results_c, dynamics_values, shock_values)
    #         print("Evolution plots complete")
    #     except Exception as e:
    #         print(f"Error in evolution plots: {e}")
    #         import traceback
    #         traceback.print_exc()
    #
    #     # Debug one specific parameter combination
    #     if (1, 1) in results_c:
    #         print("\nExamining data for dynamics=1, shock=1:")
    #         sample_result = results_c[(1, 1)]
    #         for key in sorted(sample_result.keys()):
    #             if isinstance(sample_result[key], np.ndarray):
    #                 print(f"  {key}: ndarray with shape {sample_result[key].shape}")
    #             elif isinstance(sample_result[key], list):
    #                 print(f"  {key}: list with {len(sample_result[key])} items")
    #             else:
    #                 print(f"  {key}: {type(sample_result[key])}")
    #
    # except Exception as e:
    #     print(f"Experiment C failed: {e}")
    #     import traceback
    #     traceback.print_exc()
    #
    # print("=== EXPERIMENT C COMPLETED ===")

    ##############################################
    # Experiment D: Vary Learning Rate and Epsilon
    ##############################################
    # COMMENTED OUT - Focus on Experiments A and B
    # learning_rate_values = [0.03, 0.05, 0.07]
    # epsilon_values = [0.2, 0.3]
    # results_d = experiment_learning_trust(base_params, learning_rate_values, epsilon_values, num_runs)
    #
    # # --- Plot 1: Final SECI vs LR/Epsilon (Bar Chart) ---
    # fig_d_seci, ax_d_seci = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    # fig_d_seci.suptitle("Experiment D: Final SECI vs Learning Rate / Epsilon (Mean & IQR)")
    # bar_width = 0.35
    #
    # for idx, eps in enumerate(epsilon_values):
    #     means_exploit = []; errors_exploit = [[],[]]
    #     means_explor = []; errors_explor = [[],[]]
    #
    #     for lr in learning_rate_values:
    #         res_key = (lr, eps)
    #         if res_key not in results_d: continue
    #         res = results_d[res_key]
    #
    #         if res["seci"].ndim >= 3 and res["seci"].shape[1] > 0:
    #             seci_exploit_final = res["seci"][:, -1, 1]
    #             seci_explor_final = res["seci"][:, -1, 2]
    #
    #             mean_exp = np.mean(seci_exploit_final); p25_exp = np.percentile(seci_exploit_final, 25); p75_exp = np.percentile(seci_exploit_final, 75)
    #             mean_er = np.mean(seci_explor_final); p25_er = np.percentile(seci_explor_final, 25); p75_er = np.percentile(seci_explor_final, 75)
    #
    #             means_exploit.append(mean_exp); errors_exploit[0].append(mean_exp-p25_exp); errors_exploit[1].append(p75_exp-mean_exp)
    #             means_explor.append(mean_er); errors_explor[0].append(mean_er-p25_er); errors_explor[1].append(p75_er-mean_er)
    #         else:
    #             means_exploit.append(0); errors_exploit[0].append(0); errors_exploit[1].append(0)
    #             means_explor.append(0); errors_explor[0].append(0); errors_explor[1].append(0)
    #
    #     x_pos = np.arange(len(learning_rate_values))
    #     ax = ax_d_seci[idx]
    #     rects1 = ax.bar(x_pos - bar_width/2, means_exploit, bar_width, yerr=errors_exploit, capsize=4, label='Exploitative', color='tab:blue', error_kw=dict(alpha=0.5))
    #     rects2 = ax.bar(x_pos + bar_width/2, means_explor, bar_width, yerr=errors_explor, capsize=4, label='Exploratory', color='tab:orange', error_kw=dict(alpha=0.5))
    #
    #     ax.set_xlabel("Learning Rate")
    #     ax.set_ylabel("Mean Final SECI")
    #     ax.set_title(f"Epsilon = {eps}")
    #     ax.set_xticks(x_pos)
    #     ax.set_xticklabels(learning_rate_values)
    #     ax.legend()
    #     ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    #     ax.set_ylim(bottom=0)
    #
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.savefig("agent_model_results/experiment_d_seci.png")
    # plt.close(fig_d_seci)
    # gc.collect()

# Note: Google Drive is mounted at the top of the file for Colab compatibility
