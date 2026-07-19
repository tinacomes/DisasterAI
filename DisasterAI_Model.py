
import os
import random
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from mesa import Agent, Model
from mesa.space import MultiGrid

save_dir = "agent_model_results"
os.makedirs(save_dir, exist_ok=True)

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
                 trust_learning_rate=0.05,
                 exploit_trust_lr=0.03,
                 explor_trust_lr=0.06,
                 d_exploit=2.0, delta_exploit=3.5,
                 d_explor=4.0, delta_explor=1.2):

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

        self.exploit_trust_lr = exploit_trust_lr


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
        # Lifetime source-mode choice tally, split by whether the choice came from
        # the epsilon exploration branch (uniform random over modes) or the
        # Q-value exploitation branch (argmax). Instrumentation for diagnosing the
        # exploration floor — see DisasterModel.get_mode_choice_summary().
        self.mode_choice_counts = {
            'exploration':  {'self_action': 0, 'human': 0, 'ai': 0},
            'exploitation': {'self_action': 0, 'human': 0, 'ai': 0},
        }
                     
        # --- Q-Table for Source Values ---
        # Use high-level modes: self_action, human, ai (not individual AIs)
        # This allows agents to learn about source CATEGORIES, not just individuals
        self.q_table = {}
        self.q_table["self_action"] = 0.0
        self.q_table["human"] = 0.0  # Generic value of querying humans as a category
        # Explorers start with a positive AI affinity to bootstrap the intended
        # mechanism: open agents learn early that AI can be a useful source,
        # while exploiters remain neutral until Q-learning evidence accumulates.
        self.q_table["ai"] = 0.1 if agent_type == "exploratory" else 0.0

        # Track individual sources separately for selection within each mode
        # These are updated alongside mode Q-values for granular tracking
        for k in range(model.num_ai):
            self.q_table[f"A_{k}"] = 0.0

        # --- Belief Update Parameters ---
        # These control how beliefs change when info is ACCEPTED (separate from Q-learning)
        self.D = d_exploit if agent_type == "exploitative" else d_explor       # Acceptance threshold parameter
        self.delta = delta_exploit if agent_type == "exploitative" else delta_explor  # Acceptance sensitivity parameter
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
        self.accepted_ai = 0          # per-period counter (reset every 5 ticks for retain metrics)
        self.cum_accepted_ai = 0      # cumulative counter (never reset) used for AECI classification

        # Performance Counters
        self.correct_targets = 0
        self.incorrect_targets = 0
        self.tokens_sent_total = 0    # tokens counted at placement time (no evaluation delay)


    def initialize_beliefs(self, assigned_rumor=None):
        """
        Initializes agent beliefs with default values, applies assigned rumor if provided,
        and senses the local environment. Ensures all cells have valid belief dictionaries.
        """
        # disaster_grid is allocated np.zeros((width, height)) — axis 0 is x/width
        width, height = self.model.disaster_grid.shape
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
                sense_radius = 0  # Own cell only — 100 agents × r=2 covers 94% of grid per tick, eliminating information scarcity

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

    def find_highest_uncertainty_area(self):
        """
        Explorer seeks areas with highest combined uncertainty.
        Combined metric: 60% low confidence + 40% spatial variance (neighborhood disagreement).
        Returns a cell from top-5 uncertain areas for exploration diversity.

        IMPORTANT: Excludes cells within sensing radius - agents should query about
        cells they CANNOT directly sense, otherwise they can trivially verify info.
        """
        scored_cells = []

        for cell, belief in self.beliefs.items():
            if not isinstance(belief, dict):
                continue

            # Skip cells within sensing radius - querying about these is useless
            if self.is_within_sensing_range(cell):
                continue

            # Component 1: Low confidence (60% weight)
            conf_uncertainty = 1.0 - belief.get('confidence', 0.1)

            # Component 2: Spatial variance - disagreement among neighbors (40% weight)
            neighbors = self.model.grid.get_neighborhood(cell, moore=True, radius=1, include_center=False)
            neighbor_levels = []
            for n in neighbors:
                if n in self.beliefs and isinstance(self.beliefs[n], dict):
                    neighbor_levels.append(self.beliefs[n].get('level', 0))

            if len(neighbor_levels) > 1:
                # Calculate variance of neighbor beliefs
                mean_level = sum(neighbor_levels) / len(neighbor_levels)
                spatial_var = sum((lvl - mean_level) ** 2 for lvl in neighbor_levels) / len(neighbor_levels)
                # Normalize variance (max theoretical variance for levels 0-5 is 6.25)
                normalized_var = min(1.0, spatial_var / 6.25)
            else:
                # No neighbors with beliefs = high uncertainty
                normalized_var = 0.5

            # Combined uncertainty score
            uncertainty = 0.6 * conf_uncertainty + 0.4 * normalized_var
            scored_cells.append((cell, uncertainty))

        if not scored_cells:
            # Fallback: find ANY cell outside sensing range
            for cell in self.beliefs.keys():
                if not self.is_within_sensing_range(cell):
                    return cell
            # Last resort: random cell far from agent
            return (
                (self.pos[0] + 5) % self.model.width,
                (self.pos[1] + 5) % self.model.height
            )

        # Sort by uncertainty descending
        scored_cells.sort(key=lambda x: -x[1])

        # Return from top-5 uncertain cells (adds exploration diversity)
        top_k = scored_cells[:5]
        return random.choice(top_k)[0]

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
                    radius = 1  # Reference scale for decay (sensing_radius=0, so any distance is "far")
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

    def reward_belief_accuracy(self, cell, actual_level):
        """
        Accumulate belief accuracy reward for a cell (separate from source quality).
        This evaluates the agent's OWN assessment accuracy.
        Rewards are accumulated and applied as a single self_action Q-update
        at the end of sense_environment() to avoid inflating self_action Q
        with ~25 updates per tick while human/ai get 0-1 updates.
        """
        prior_belief = self.beliefs.get(cell, {})
        if not isinstance(prior_belief, dict):
            return  # No prior belief to evaluate

        prior_level = prior_belief.get('level', 0)
        prior_confidence = prior_belief.get('confidence', 0.1)

        # Only evaluate if agent had meaningful confidence in prior belief
        if prior_confidence < 0.3:
            return  # Too uncertain to count as a real assessment

        # Calculate belief accuracy reward
        belief_error = abs(prior_level - actual_level)
        if belief_error == 0:
            belief_reward = 0.4   # Perfect belief
        elif belief_error == 1:
            belief_reward = 0.1   # Close
        elif belief_error == 2:
            belief_reward = -0.1  # Moderate error
        else:
            belief_reward = -0.3  # Large error

        # Scale reward by confidence (more confident = higher stakes)
        belief_reward *= prior_confidence

        # Accumulate for batch update (applied in sense_environment or flush_belief_rewards)
        if not hasattr(self, '_belief_accuracy_rewards'):
            self._belief_accuracy_rewards = []
        self._belief_accuracy_rewards.append(belief_reward)

    def flush_belief_rewards(self):
        """Apply accumulated belief accuracy rewards to self_action Q-value.
        Called automatically at end of sense_environment, but can also be called
        directly for testing purposes."""
        if hasattr(self, '_belief_accuracy_rewards') and self._belief_accuracy_rewards:
            avg_reward = sum(self._belief_accuracy_rewards) / len(self._belief_accuracy_rewards)
            learning_rate = 0.1 if self.agent_type == "exploratory" else 0.08
            old_q = self.q_table.get("self_action", 0.0)
            self.q_table["self_action"] = old_q + learning_rate * (avg_reward - old_q)
            self._belief_accuracy_rewards = []

    def sense_environment(self):
        pos = self.pos
        radius = 0  # Own cell only — 100 agents × r=2 covers 94% of grid per tick, eliminating information scarcity
        cells = self.model.grid.get_neighborhood(pos, moore=True, radius=radius, include_center=True)
        for cell in cells:
            if 0 <= cell[0] < self.model.width and 0 <= cell[1] < self.model.height:
                actual = self.model.disaster_grid[cell[0], cell[1]]

                # Issue 2 FIX: Reward agent's own belief accuracy BEFORE updating belief
                # This is separate from source quality - evaluates agent's internal model
                self.reward_belief_accuracy(cell, actual)

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

        # Batch update self_action Q-value: ONE update per tick with average reward
        # This prevents self_action from being inflated by ~25 per-cell updates
        # while human/ai Q-values only get 0-1 updates per tick
        self.flush_belief_rewards()

    def evaluate_information_quality(self, cell, actual_level):
        """
        Evaluate pending information about a cell against a reference level.
        Called in two contexts:
        1. Direct sensing: actual_level is ground truth from the environment
        2. Cross-referencing: actual_level is from a high-confidence belief (>=0.6),
           allowing agents to verify remote queries without physically visiting the cell.
        Provides fast feedback (3-15 tick window) on information accuracy.
        """
        current_tick = self.model.tick
        evaluated = []

        for item in self.pending_info_evaluations:
            # Support 4-tuple, 6-tuple, and new 7-tuple (with was_accepted flag)
            if len(item) >= 6:
                tick_received, source_id, eval_cell, reported_level = item[0], item[1], item[2], item[3]
            else:
                tick_received, source_id, eval_cell, reported_level = item

            if eval_cell != cell:
                continue

            # Check if within evaluation window (3-15 ticks) - wider for better coverage
            ticks_elapsed = current_tick - tick_received
            if ticks_elapsed > 15:
                # Too old, remove from pending
                evaluated.append(item)
                continue

            if ticks_elapsed < 3:
                # Too soon, wait a bit for potential disaster evolution
                continue

            # Within window: evaluate information quality
            # Determine source type for differentiated evaluation
            is_ai_source = source_id.startswith("A_")
            is_human_source = source_id.startswith("H_")

            # Calculate ACCURACY score: how close was reported level to actual ground truth
            level_error = abs(reported_level - actual_level)
            if level_error == 0:
                accuracy_score = 1.0   # Perfect accuracy
            elif level_error == 1:
                accuracy_score = 0.5   # Close
            elif level_error == 2:
                accuracy_score = -0.2  # Moderate error
            else:
                accuracy_score = -0.6  # Large error

            # Calculate CONFIRMATION score: how close was reported level to agent's PRIOR belief
            # Use stored prior from tuple if available (uncontaminated by query update)
            if len(item) >= 6:
                prior_level = item[4]  # stored_prior_level
            else:
                prior_belief = self.beliefs.get(cell, {})
                prior_level = prior_belief.get('level', 0) if isinstance(prior_belief, dict) else 0
            prior_error = abs(reported_level - prior_level)
            if prior_error == 0:
                confirmation_score = 1.0   # Perfect confirmation of prior
            elif prior_error == 1:
                confirmation_score = 0.5   # Close to prior
            elif prior_error == 2:
                confirmation_score = -0.2  # Contradicts prior moderately
            else:
                confirmation_score = -0.6  # Strongly contradicts prior

            # SOURCE KNOWLEDGE CONFIDENCE: How likely did the source KNOW the truth?
            # This scales the learning rate (signal strength), not the reward itself.
            # - AI: Broad sensing radius → high confidence they knew truth
            # - Human on remote cell: Limited radius → may not have known (weaker signal)
            # - Human on nearby cell: Could have sensed it → stronger signal
            # - Friends: For exploiters, friend info is more trusted (authentic shared beliefs)
            #
            # The Q-learning then naturally learns to trust sources based on actual outcomes,
            # but with appropriate signal strength based on source knowledge.

            # Determine if this cell was likely within the source's sensing range
            # For humans, sensing radius is 2. For AI, effectively unlimited.
            # We use the querying agent's distance as a proxy (if remote for us, likely remote for human too)
            cell_was_remote = not self.is_within_sensing_range(cell)

            # Calculate source knowledge confidence
            if is_ai_source:
                # AI has broad knowledge - high confidence they knew the truth
                source_knowledge_conf = 1.0
            elif is_human_source:
                if cell_was_remote:
                    # Human likely didn't directly sense this cell - weaker signal
                    source_knowledge_conf = 0.5
                else:
                    # Human could have sensed this cell - stronger signal
                    source_knowledge_conf = 0.9
                # Friends get slight boost for exploiters (authentic shared beliefs)
                if self.agent_type == "exploitative" and source_id in self.friends:
                    source_knowledge_conf = min(1.0, source_knowledge_conf * 1.2)
            else:
                source_knowledge_conf = 0.7  # Unknown source type

            # Reward is TYPE-SPECIFIC:
            #   Explorers  → accuracy_score: "was the source truthful?"
            #                This drives them toward sources that report ground truth,
            #                so at low α (truthful AI) they learn to prefer AI.
            #   Exploiters → confirmation_score: "did the source agree with my belief?"
            #                Confirmation bias: they reward sources that confirm their
            #                prior, not sources that are objectively correct.
            #                At low α, AI contradicts their beliefs → Q["ai"] falls.
            #                At high α, AI confirms their beliefs → Q["ai"] rises.
            # D/δ acceptance still governs whether the info updates the belief;
            # this reward governs whether the *mode* gets called again.
            combined_reward = accuracy_score if self.agent_type == "exploratory" else confirmation_score

            # Scale to [-0.7, +0.5] range to match existing Q-update expectations
            accuracy_reward = combined_reward * 0.6 - 0.1

            # Determine mode from source_id (map to high-level modes)
            if is_human_source:
                mode = "human"
            elif is_ai_source:
                mode = "ai"  # Generic AI mode (not individual AI)
            else:
                mode = None

            # Update mode Q-value (what's used in action selection)
            # SCALE BY SOURCE KNOWLEDGE: stronger signal when source likely knew the truth
            if mode and mode in self.q_table:
                old_mode_q = self.q_table[mode]
                # Base learning rate by agent type
                base_lr = 0.25 if self.agent_type == "exploratory" else 0.12
                # Scale by source knowledge confidence
                info_learning_rate = base_lr * source_knowledge_conf
                # Use standard Q-learning update: Q += lr * (reward - Q)
                new_mode_q = old_mode_q + info_learning_rate * (accuracy_reward - old_mode_q)
                self.q_table[mode] = new_mode_q

            # Also update specific source Q-value (for tracking individuals)
            if source_id in self.q_table:
                old_q = self.q_table[source_id]
                base_lr = 0.25 if self.agent_type == "exploratory" else 0.12
                info_learning_rate = base_lr * source_knowledge_conf
                # Use standard Q-learning update: Q += lr * (reward - Q)
                new_q = old_q + info_learning_rate * (accuracy_reward - old_q)
                self.q_table[source_id] = new_q

            # Update trust with ASYMMETRIC learning: penalize bad info faster
            # SCALE BY SOURCE KNOWLEDGE: weaker updates when source may not have known
            if source_id in self.trust:
                old_trust = self.trust[source_id]
                # More aggressive trust target: bad info → low trust, good info → high trust
                # accuracy_reward range: [-0.7, +0.5] → trust_target range: [0.0, 0.75]
                if accuracy_reward < 0:
                    # Bad info: aggressive penalty, target drops to 0 for worst case
                    trust_target = max(0.0, 0.5 + accuracy_reward)  # -0.7→0, 0→0.5
                    base_trust_lr = 0.25 if self.agent_type == "exploratory" else 0.15
                else:
                    # Good info: moderate reward
                    trust_target = min(1.0, 0.5 + 0.5 * accuracy_reward)  # 0→0.5, +0.5→0.75
                    base_trust_lr = 0.12 if self.agent_type == "exploratory" else 0.06
                # Scale trust update by source knowledge confidence
                trust_lr = base_trust_lr * source_knowledge_conf
                new_trust = max(0.0, min(1.0, old_trust + trust_lr * (trust_target - old_trust)))
                self.trust[source_id] = new_trust

            # Mark as evaluated
            evaluated.append(item)

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

    def get_network_consensus(self, cell, min_trust=0.4):
        """
        Compute the confidence-weighted mean belief of trusted friends about a cell.

        Used by exploiters as a reference in evaluate_pending_info instead of their own
        (potentially contaminated) belief. The network may share the same wrong beliefs —
        the filter bubble is preserved — but the reference is:
          - Built from multiple independent data points (less noise)
          - Not contaminated by this agent's own interaction with the source being evaluated
          - Architecturally consistent: exploiters trust their network, so the network IS
            their effective ground truth

        Returns (consensus_level, consensus_conf) or (None, 0.0) if < 2 friends have data.
        """
        weighted_levels = []
        weights = []
        for fid in self.friends:
            friend = self.model.humans.get(fid)
            if not friend:
                continue
            trust_val = self.trust.get(fid, 0.1)
            if trust_val < min_trust:
                continue
            belief = friend.beliefs.get(cell, {})
            if not isinstance(belief, dict):
                continue
            level = belief.get('level', None)
            conf = belief.get('confidence', 0.0)
            if level is not None and conf >= 0.2:
                weighted_levels.append(level * trust_val)
                weights.append(trust_val)
        if len(weights) < 2:
            return None, 0.0
        consensus_level = int(round(sum(weighted_levels) / sum(weights)))
        # Shrink confidence toward uncertainty to avoid over-weighting small networks
        consensus_conf = min(0.75, sum(weights) / (len(weights) + 2))
        return consensus_level, consensus_conf

    def evaluate_pending_info(self):
        """
        Dedicated pass to evaluate ALL pending info evaluations using current beliefs.
        Called each tick after both sense_environment() and seek_information() have run,
        so beliefs are up-to-date from both direct sensing and queried reports.

        This decouples info feedback from sensing: agents don't need to physically
        visit a cell to evaluate whether the info they received about it was good.
        They cross-reference against their current belief (which may come from
        sensing, other queries, or accumulated knowledge).
        """
        if not self.pending_info_evaluations:
            return

        current_tick = self.model.tick
        evaluated = []

        for item in self.pending_info_evaluations:
            # Support 4-tuple, 6-tuple, and new 7-tuple (adds was_accepted flag at index 6)
            if len(item) >= 6:
                tick_received, source_id, cell, reported_level, stored_prior_level, stored_prior_conf = \
                    item[0], item[1], item[2], item[3], item[4], item[5]
                was_accepted = item[6] if len(item) >= 7 else True  # default True for old tuples
            else:
                tick_received, source_id, cell, reported_level = item
                stored_prior_level = None
                stored_prior_conf = 0.0
                was_accepted = True

            ticks_elapsed = current_tick - tick_received

            # Too soon — wait for beliefs to stabilize
            if ticks_elapsed < 3:
                continue

            # Check if this is a remote cell (outside sensing range)
            is_remote_cell = not self.is_within_sensing_range(cell)

            # CRITICAL FIX: Remote cells for EXPLORERS get extended expiry window (30 ticks)
            # They need time to move and sense the cell for ground truth verification.
            # Non-remote cells and exploiters use normal 15-tick window.
            if self.agent_type == "exploratory" and is_remote_cell:
                expiry_window = 30  # Extended window for explorers to reach and sense remote cells
            else:
                expiry_window = 15  # Normal window

            # Too old — expire without evaluation
            if ticks_elapsed > expiry_window:
                evaluated.append(item)
                continue

            # Within evaluation window: use current belief as reference
            belief = self.beliefs.get(cell, {})
            if not isinstance(belief, dict):
                continue

            belief_conf = belief.get('confidence', 0.0)
            reference_level = belief.get('level', 0)

            # Fix circular eval: Detect self-contaminated references.
            # The previous check only caught FULL contamination (belief == reported).
            # PARTIAL contamination occurs when the belief shifted TOWARD the reported
            # value (closer than prior was) without independent confirmation.
            # In that case, the belief is a partially circular reference and we should
            # use the stored prior instead to avoid rewarding sources for being consistent
            # with the belief they themselves caused.
            use_prior_as_reference = False
            if stored_prior_level is not None:
                # Detect partial contamination: belief moved toward reported level
                # without enough independent confirmation (confidence stayed low).
                belief_moved_toward_reported = (
                    abs(reference_level - reported_level) <
                    abs(stored_prior_level - reported_level)
                )
                no_independent_confirmation = belief_conf < stored_prior_conf + 0.3
                if belief_moved_toward_reported and no_independent_confirmation:
                    # Belief is (partially) self-contaminated. Use stored prior as
                    # reference to avoid circular evaluation.
                    use_prior_as_reference = True
                    reference_level = stored_prior_level
                    belief_conf = stored_prior_conf

            # Fix 3: For exploiters, replace own belief with network consensus when available.
            # The network may share the same wrong beliefs (filter bubble preserved), but:
            #   - Multiple friends' data → less noise than single-agent belief
            #   - Independent from this agent's own interaction with the source being evaluated
            # Fall back to own belief if fewer than 2 friends have data for this cell.
            used_network_consensus = False
            if self.agent_type == "exploitative":
                net_level, net_conf = self.get_network_consensus(cell)
                if net_conf >= 0.3:
                    reference_level = net_level
                    belief_conf = net_conf
                    used_network_consensus = True

            # For cells outside sensing range, agents cannot see ground truth directly.
            # CRITICAL: Explorers should NOT penalize sources for disagreeing with their
            # uncertain beliefs about remote cells. This would cause them to distrust
            # truthful sources that report correct info that differs from wrong beliefs.
            #
            # EXPLOITERS: Use network/own-belief confirmation (echo chamber by design)
            # EXPLORERS: For accepted remote reports, wait for an external verification
            #            ("situation report") — see block below. Rejected remote reports
            #            are scored against the stored prior (belief unchanged).

            # --- Situation-report verification (explorers, accepted remote reports) ---
            # Accuracy-seeking explorers actively verify accepted remote reports against
            # external information (official assessments, media, situation reports).
            # Each evaluation attempt, verification arrives with probability
            # model.verification_probability; when it arrives it carries the same noise
            # model as direct human sensing (±1 with prob 0.2). Until it arrives the item
            # stays pending (retried next tick); if it never arrives before the 30-tick
            # expiry the item lapses unevaluated. Explicitly NO fallback to scoring
            # against the agent's own prior — that would silently turn the explorers'
            # accuracy channel into a confirmation channel.
            # Replaces the previous instant/perfect ground-truth reference: verification
            # is now delayed (geometric arrival on top of the 3-tick minimum), noisy, and
            # evaluated against the CURRENT disaster state, so reports about cells that
            # have since evolved are naturally penalised less consistently.
            # Truthful AI (α≈0): reported≈truth → small error → positive reward →
            # explorers learn AI is useful. Confirming AI (α≈1): reported≈prior (often
            # wrong) → large error → negative reward. This preserves the Q-learning
            # asymmetry, but through a defensible information channel.
            verified_reference = False
            if is_remote_cell and self.agent_type == "exploratory" and was_accepted:
                if not (0 <= cell[0] < self.model.width and 0 <= cell[1] < self.model.height):
                    continue  # out-of-bounds cell: skip safely
                if random.random() < self.model.verification_probability:
                    verified_level = int(self.model.disaster_grid[cell[0], cell[1]])
                    if random.random() < 0.2:  # same noise model as human sensing
                        verified_level = max(0, min(5, verified_level + random.choice([-1, 1])))
                    reference_level = verified_level
                    belief_conf = 1.0  # external report: full-strength reference
                    verified_reference = True
                else:
                    continue  # not yet verified — stays pending, retried next tick

            if belief_conf < 0.15:
                continue  # Skip only if we have almost no information

            # Scale learning rate by confidence - uncertain references lead to weaker updates
            # (verified situation reports have belief_conf=1.0 → full strength)
            confidence_scaling = min(1.0, belief_conf / 0.5)  # Full strength at 0.5+ confidence

            # --- Salience weighting (C12 counterfactual, model.salience_weight) ---
            # The uniform per-cell verification reward is base-rate dominated: ~90% of
            # explorer-queried cells are truly empty, so a fully confirming AI passes
            # verification on ~93% of cells and keeps explorer trust regardless of α,
            # failing only on the rare high-severity cells (finding C12).
            # With salience_weight s ∈ [0,1], the learning-rate scaling of verified
            # evaluations is multiplied by (1−s) + s·(max(truth, reported)+1)/6:
            #   s=0 (default): uniform evaluation — current baseline behaviour.
            #   s=1: full salience — an error about a disaster cell (miss OR false
            #        alarm) carries up to 6× the weight of a confirmed empty cell,
            #        modelling negativity/salience bias ("being wrong about a disaster
            #        is more memorable than being right about nothing").
            if verified_reference:
                s = getattr(self.model, 'salience_weight', 0.0)
                if s > 0.0:
                    salience = (max(reference_level, reported_level) + 1) / 6.0
                    confidence_scaling *= (1.0 - s) + s * salience

            # --- Accuracy score: reported vs current reference ---
            # Verified items use the (noisy) situation-report level; rejected remote
            # queries (was_accepted=False) and local cells use reference_level
            # (the stored prior for rejected items, as the belief was not updated).
            level_error = abs(reported_level - reference_level)
            if level_error == 0:
                accuracy_score = 1.0
            elif level_error == 1:
                accuracy_score = 0.5
            elif level_error == 2:
                accuracy_score = -0.2
            else:
                accuracy_score = -0.6

            # --- Confirmation score: reported vs STORED prior (uncontaminated) ---
            prior_level = stored_prior_level if stored_prior_level is not None else reference_level
            prior_error = abs(reported_level - prior_level)
            if prior_error == 0:
                confirmation_score = 1.0
            elif prior_error == 1:
                confirmation_score = 0.5
            elif prior_error == 2:
                confirmation_score = -0.4  # Exploiters penalize more for moderate disagreement
            else:
                confirmation_score = -0.8  # Exploiters strongly penalize large disagreement

            # Determine source type from source_id
            is_ai_source = source_id.startswith("A_")
            is_human_source = source_id.startswith("H_")

            # SOURCE KNOWLEDGE CONFIDENCE: How likely did the source KNOW the truth?
            # This scales the learning rate - stronger signal when source likely knew.
            # - AI: Broad sensing → high confidence they knew truth
            # - Human on remote cell: Limited radius → may not have known (weaker signal)
            # - Friends: For exploiters, friend info is more meaningful (authentic shared beliefs)
            if is_ai_source:
                source_knowledge_conf = 1.0  # AI has broad knowledge
            elif is_human_source:
                if is_remote_cell:
                    source_knowledge_conf = 0.5  # Human likely didn't sense this cell
                else:
                    source_knowledge_conf = 0.9  # Human could have sensed it
                # Friends are more meaningful for exploiters (authentic shared beliefs)
                if self.agent_type == "exploitative" and source_id in self.friends:
                    source_knowledge_conf = min(1.0, source_knowledge_conf * 1.2)
            else:
                source_knowledge_conf = 0.7  # Unknown source type

            # Type-specific reward:
            #   Explorers  → accuracy_score  (truth-seeking; rejected queries use stored_prior)
            #   Exploiters → confirmation_score  (confirmation bias)
            combined_reward = accuracy_score if self.agent_type == "exploratory" else confirmation_score

            accuracy_reward = combined_reward * 0.7 - 0.1  # Slightly larger scale

            # Determine mode from source_id
            if is_human_source:
                mode = "human"
            elif is_ai_source:
                mode = "ai"
            else:
                mode = None

            # Update mode Q-value (scaled by confidence AND source knowledge)
            if mode and mode in self.q_table:
                old_mode_q = self.q_table[mode]
                base_lr = 0.25 if self.agent_type == "exploratory" else 0.12
                # Scale by BOTH reference confidence and source knowledge
                info_lr = base_lr * confidence_scaling * source_knowledge_conf
                self.q_table[mode] = old_mode_q + info_lr * (accuracy_reward - old_mode_q)

            # Update specific source Q-value (scaled by confidence AND source knowledge)
            if source_id in self.q_table:
                old_q = self.q_table[source_id]
                base_lr = 0.25 if self.agent_type == "exploratory" else 0.12
                info_lr = base_lr * confidence_scaling * source_knowledge_conf
                self.q_table[source_id] = old_q + info_lr * (accuracy_reward - old_q)

            # Fix trust learning rate: align with evaluate_information_quality logic.
            # Previously inverted: exploiters penalized bad info faster (0.20) than
            # explorers (0.10), but evaluate_information_quality correctly has explorers
            # penalizing faster (0.25 vs 0.15) since they care about accuracy.
            # Consistent design: explorers penalize inaccuracy faster, exploiters
            # penalize disagreement-with-belief more slowly (they are stubborn).
            # confidence_scaling already reduces effect for uncertain belief references.
            if source_id in self.trust:
                old_trust = self.trust[source_id]
                if accuracy_reward < 0:
                    trust_target = max(0.0, 0.5 + accuracy_reward)
                    # EXPLORERS: Faster penalty for inaccurate sources (accuracy-focused)
                    # EXPLOITERS: Slower penalty (stubborn; high-confidence beliefs resist change)
                    base_trust_lr = 0.25 if self.agent_type == "exploratory" else 0.15
                else:
                    trust_target = min(1.0, 0.5 + 0.5 * accuracy_reward)
                    # EXPLORERS: Moderate reward for accurate sources
                    # EXPLOITERS: Slow reward (suspicious of new trust), but Fix 4: raise
                    # the positive LR when the network consensus was used as reference —
                    # network corroboration is a stronger signal than own-belief confirmation.
                    if self.agent_type == "exploitative":
                        base_trust_lr = 0.09 if used_network_consensus else 0.06
                    else:
                        base_trust_lr = 0.12
                # Scale by BOTH reference confidence and source knowledge
                trust_lr = base_trust_lr * confidence_scaling * source_knowledge_conf
                new_trust = max(0.0, min(1.0, old_trust + trust_lr * (trust_target - old_trust)))
                self.trust[source_id] = new_trust

            evaluated.append(item)

            if self.model.debug_mode and hasattr(self, 'id_num') and (self.id_num < 2 or (50 <= self.id_num < 52)):
                print(f"[DEBUG] Agent {self.unique_id} ({self.agent_type}) PENDING INFO EVAL: source={source_id}, mode={mode}, error={level_error}, acc_rew={accuracy_reward:.3f}, trust={new_trust:.2f}, net_consensus={used_network_consensus}, verified={verified_reference}")

        # Remove evaluated/expired items
        self.pending_info_evaluations = [
            item for item in self.pending_info_evaluations
            if item not in evaluated
        ]

        # Triangulation: multi-source agreement/disagreement on same cell → trust signal
        self.triangulate_sources()

    def triangulate_sources(self):
        """
        Compare reports from multiple independent sources about the same cell
        within a 5-tick window. Consensus is a weak trust signal for explorers
        when ground truth hasn't arrived yet via process_reward; each report
        contributes to at most ONE triangulation event (one-shot), because
        pending items persist for many ticks and re-scoring the same pair every
        tick compounds into unbounded drift.

        Mode Q-values are deliberately NOT updated here: consensus is highest
        inside echo chambers, so rewarding the query mode for agreement would
        structurally favour bubble sources and contaminate the very
        echo-chamber metrics this model measures. Q-values learn only from
        information-quality feedback (evaluate_pending_info /
        evaluate_information_quality) and action-outcome feedback
        (process_reward). Exploiters are excluded entirely — they already
        receive a confirmation reward from the same pending items in
        evaluate_pending_info.
        """
        if self.agent_type != "exploratory":
            return

        current_tick = self.model.tick
        recent_window = 5

        if not hasattr(self, '_triangulated_items'):
            self._triangulated_items = set()

        # Group recent, not-yet-triangulated evaluations by cell
        cell_reports = {}  # cell -> [(item, source_id, reported_level), ...]
        for item in self.pending_info_evaluations:
            tick_received, source_id, cell = item[0], item[1], item[2]
            reported_level = item[3]
            if current_tick - tick_received > recent_window:
                continue
            if item in self._triangulated_items:
                continue
            if cell not in cell_reports:
                cell_reports[cell] = []
            cell_reports[cell].append((item, source_id, reported_level))

        for cell, reports in cell_reports.items():
            # Need at least 2 distinct sources
            source_ids = [r[1] for r in reports]
            if len(set(source_ids)) < 2:
                continue

            levels = [r[2] for r in reports]
            max_disagreement = max(levels) - min(levels)

            if max_disagreement <= 1:
                trust_delta = 0.03   # Consensus → weak trust boost
            elif max_disagreement >= 3:
                trust_delta = -0.03  # Strong conflict → weak trust penalty
            else:
                continue  # Ambiguous — no signal; items stay eligible for later reports

            for item, source_id, _ in reports:
                if source_id in self.trust:
                    old_t = self.trust[source_id]
                    self.trust[source_id] = max(0.0, min(1.0, old_t + trust_delta))
                # One-shot: this report can never be triangulated again
                self._triangulated_items.add(item)

        # Prune the marker set so it doesn't outlive the pending list
        if self._triangulated_items:
            self._triangulated_items &= set(self.pending_info_evaluations)

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

        return report

    def is_within_sensing_range(self, cell):
        """Check if a cell is within the agent's sensing radius (Moore neighborhood)."""
        if not self.pos or not cell:
            return False
        sensing_radius = 0  # Own cell only
        return cell[0] == self.pos[0] and cell[1] == self.pos[1]

    def find_believed_epicenter(self):
        """
        Finds the cell with the highest believed disaster level.
        IMPORTANT: Excludes cells within sensing radius - agents should query about
        cells they CANNOT directly sense, otherwise they can trivially verify info.
        """
        max_level = -1
        best_cells = []
        # Check own beliefs, excluding cells within sensing range
        for cell, belief_info in self.beliefs.items():
            if isinstance(belief_info, dict):
                # Skip cells within sensing radius - querying about these is useless
                if self.is_within_sensing_range(cell):
                    continue
                level = belief_info.get('level', -1)
                if level > max_level:
                    max_level = level
                    best_cells = [cell]
                elif level == max_level:
                    best_cells.append(cell)

        # If no beliefs > 0 outside sensing range, return None
        if best_cells and max_level > 0:  # Only consider if found something >= L1
            self.believed_epicenter = random.choice(best_cells)
        else:
            self.believed_epicenter = None  # No valid epicenter outside sensing range


    def apply_trust_decay(self):
        """Applies decay to all trust relationships toward neutral points.
        EXPLOITERS: Maintain friend/non-friend distinction - friends decay toward 0.6,
                    non-friends and AI decay toward 0.35. This preserves social network loyalty.
        EXPLORERS: All sources decay toward 0.5 - they're open to re-evaluating anyone.
        """
        if self.agent_type == "exploitative":
            decay_rate = 0.008  # Slow decay - exploiters are stubborn
            for source_id in list(self.trust.keys()):
                old_trust = self.trust[source_id]
                # Friends have higher neutral point (maintain loyalty)
                if source_id in self.friends:
                    neutral = 0.60
                else:
                    # Non-friends and AI have lower neutral point (maintain suspicion)
                    neutral = 0.35
                if old_trust > neutral:
                    self.trust[source_id] = max(neutral, old_trust - decay_rate)
                elif old_trust < neutral:
                    self.trust[source_id] = min(neutral, old_trust + decay_rate)
        else:  # exploratory
            decay_rate = 0.012  # Moderate decay - responsive to change
            neutral_trust = 0.5
            for source_id in list(self.trust.keys()):
                old_trust = self.trust[source_id]
                if old_trust > neutral_trust:
                    self.trust[source_id] = max(neutral_trust, old_trust - decay_rate)
                elif old_trust < neutral_trust:
                    self.trust[source_id] = min(neutral_trust, old_trust + decay_rate)

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

            # D/DELTA ACCEPTANCE MECHANISM (from social judgment theory):
            # P(accept) = D^δ / (d^δ + D^δ)  where d = distance between report and prior
            # Exploitative: D=2.0, δ=3.5 → sharp narrow acceptance window
            # Exploratory:  D=4.0, δ=1.2 → gradual wide acceptance window
            # Friends get a trust-widened threshold (more open to friend info)
            is_friend = source_id in self.friends if source_id else False
            d = abs(reported_level - prior_level)
            if d == 0:
                accept_prob = 1.0
            else:
                D_eff = self.D * (1.0 + 0.5 * self.trust.get(source_id, 0.5)) if is_friend else self.D
                accept_prob = (D_eff ** self.delta) / (d ** self.delta + D_eff ** self.delta)

            if random.random() > accept_prob:
                # REJECT - still track for feedback but don't update belief
                if source_id:
                    self.pending_info_evaluations.append((
                        self.model.tick, source_id, cell,
                        int(reported_level), int(prior_level), float(prior_confidence),
                        False  # was_accepted = False (rejected by D/delta)
                    ))
                return (False, False)  # (significant_change, was_accepted)

            # Convert confidence to precision with agent-specific scaling
            # Exploiters: stronger prior (1.5x) → more resistant to change
            # Explorers: weaker prior (0.8x) → more open to change
            if self.agent_type == "exploitative":
                prior_precision = 1.5 * prior_confidence / (1 - prior_confidence + 1e-6)
            else:
                prior_precision = 0.8 * prior_confidence / (1 - prior_confidence + 1e-6)

            # Source precision — trust-based, moderate scaling
            if self.agent_type == "exploitative":
                source_precision_base = 2.0 * source_trust / (1 - source_trust + 1e-6)
            else:
                source_precision_base = 1.5 * source_trust / (1 - source_trust + 1e-6)

            source_precision = source_precision_base

            if source_trust < 0.3:
                trust_factor = (source_trust / 0.3) ** 0.5
                source_precision = source_precision_base * trust_factor

            if source_trust < 0.03:
                source_precision *= 0.05

            source_precision = min(source_precision, 8.0)

            # Combine using precision-weighted (Bayesian) averaging
            posterior_precision = prior_precision + source_precision
            posterior_level = (prior_precision * prior_level + source_precision * reported_level) / posterior_precision
            posterior_confidence = posterior_precision / (1 + posterior_precision)

            # Constrain to valid ranges
            posterior_level = max(0, min(5, round(posterior_level)))
            posterior_confidence = max(0.1, min(0.98, posterior_confidence))

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
                    int(reported_level),  # The level reported by the source (NOT posterior)
                    int(prior_level),     # Prior belief BEFORE this update (for uncontaminated cross-ref)
                    float(prior_confidence),  # Prior confidence BEFORE this update
                    True  # was_accepted = True (passed D/delta threshold)
                ))
                if self.model.debug_mode and hasattr(self, 'id_num') and (self.id_num < 2 or (50 <= self.id_num < 52)):
                    print(f"[DEBUG] Agent {self.unique_id} ({self.agent_type}) added pending info eval: tick={self.model.tick}, source={source_id}, cell={cell}, reported={reported_level}")

            # Track AI source information for later trust updates
            is_ai_source = hasattr(self, 'ai_info_sources') and cell in self.ai_info_sources
            if is_ai_source and significant_change:
                if not hasattr(self, 'ai_acceptances'):
                    self.ai_acceptances = {}
                self.ai_acceptances[cell] = self.model.tick

            # Return (significant_change, was_accepted=True) — passed D/delta threshold
            return (significant_change, True)

        except Exception as e:
            print(f"ERROR in Agent {self.unique_id} update_belief_bayesian: {e}")
            return False

    def select_human_source(self):
        """Pick the specific human to query when the 'human' mode was chosen.

        query_scope='global' (baseline): trust-weighted draw over ALL humans —
        friends weighted by trust, non-friends at a 0.05 baseline so Q-learning
        can discover them. Note this means topology does not gate access: with
        ~85 non-friends the baseline weights sum to ~4.3 vs ~3 for friends.

        query_scope='network' (design proposal §3): access follows edges.
        Default draw is friends-only, trust-weighted exactly like the friend
        branch above. With p = 0.1 the query instead goes to a uniform draw
        from the 2-hop neighbourhood (friends-of-friends, excluding self and
        direct friends) — exploration without teleportation: reaching a
        stranger requires an intermediary, which is what makes weak-tie
        bridges valuable (dormant-tie activation). Agents with no reachable
        neighbourhood at all fall back to a uniform draw over the population
        (rare isolates only).

        Returns a human agent id ('H_j') or None if no candidate exists.
        """
        if getattr(self.model, 'query_scope', 'global') == 'network':
            # Sorted for deterministic iteration under seeded runs (set order of
            # string ids varies with PYTHONHASHSEED).
            friends_avail = sorted(f for f in self.friends
                                   if f in self.model.humans and f != self.unique_id)
            if friends_avail and random.random() < 0.1:
                # Friends-of-friends: 2-hop reach, uniform
                two_hop = set()
                for f_id in friends_avail:
                    two_hop.update(self.model.humans[f_id].friends)
                two_hop.discard(self.unique_id)
                two_hop.difference_update(friends_avail)
                two_hop = sorted(h for h in two_hop if h in self.model.humans)
                if two_hop:
                    return random.choice(two_hop)
                # No friends-of-friends (tight clique with no bridge): fall
                # through to the friends draw below.
            if friends_avail:
                candidates = friends_avail
                weights = [max(0.05, self.trust.get(h_id, 0.1)) for h_id in candidates]
            else:
                # Isolate with no friends: uniform over the population so the
                # 'human' mode stays explorable for Q-learning.
                candidates = sorted(h for h in self.model.humans if h != self.unique_id)
                if not candidates:
                    return None
                weights = [1.0] * len(candidates)
        else:
            # Global scope (baseline): every human is reachable.
            candidates = [h for h in self.model.humans if h != self.unique_id]
            if not candidates:
                return None
            weights = []
            for h_id in candidates:
                trust_val = self.trust.get(h_id, 0.1)
                if h_id in self.friends:
                    weights.append(max(0.05, trust_val))  # Friends: weight by trust
                else:
                    weights.append(0.05)  # Non-friends: small baseline for exploration

        total_w = sum(weights)
        r = random.random() * total_w
        cumulative = 0.0
        source_id = candidates[-1]  # fallback
        for cid, w in zip(candidates, weights):
            cumulative += w
            if r <= cumulative:
                source_id = cid
                break
        return source_id

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

                    # Second try: Use highest confidence cell OUTSIDE sensing range
                    if not interest_point or self.beliefs.get(interest_point, {}).get('level', 0) <= 0:
                        max_conf = -1
                        highest_conf_cells = []

                        # Search for cells outside sensing range
                        if len(self.beliefs) > 0:
                            for cell, belief_info in self.beliefs.items():
                                if isinstance(belief_info, dict):
                                    # Skip cells within sensing radius
                                    if self.is_within_sensing_range(cell):
                                        continue
                                    conf = belief_info.get('confidence', 0.0)
                                    if conf > max_conf:
                                        max_conf = conf
                                        highest_conf_cells = [cell]
                                    elif conf == max_conf:
                                        highest_conf_cells.append(cell)

                        if highest_conf_cells:
                            interest_point = random.choice(highest_conf_cells)
                        else:
                            # Absolute fallback: pick a random cell OUTSIDE sensing range
                            # Offset by at least sensing_radius + 1 to ensure outside
                            offset = 1  # sensing_radius (0) + 1
                            interest_point = (
                                (self.pos[0] + offset + random.randrange(self.model.width - 2*offset)) % self.model.width,
                                (self.pos[1] + offset + random.randrange(self.model.height - 2*offset)) % self.model.height
                            )
                            if self.model.debug_mode:
                                print(f"Agent {self.unique_id}: Using random fallback interest point {interest_point} (outside sensing range)")

            else:  # Exploratory
                # Explorers seek HIGH UNCERTAINTY areas - not just their current position
                # This enables them to gather information about poorly understood regions
                interest_point = self.find_highest_uncertainty_area()
                query_radius = 2  # Standard query radius

                # Fallback if uncertainty search fails - pick cell OUTSIDE sensing range
                if not interest_point:
                    offset = 1  # sensing_radius (0) + 1
                    interest_point = (
                        (self.pos[0] + offset + random.randrange(max(1, self.model.width - 2*offset))) % self.model.width,
                        (self.pos[1] + offset + random.randrange(max(1, self.model.height - 2*offset))) % self.model.height
                    )
                    if self.model.debug_mode:
                        print(f"Agent {self.unique_id}: Uncertainty search failed, using random cell outside sensing range")

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

            if not interest_point:
                print(f"Agent {self.unique_id}: No valid interest point, skipping seek_information.")
                return

            # Source selection (epsilon-greedy with type-specific biases)
            # Use 3-mode structure: self_action, human, ai
            # Then select specific source within chosen mode
            possible_modes = ["self_action", "human", "ai"]

            # Store Q-values
            for mode in possible_modes:
                decision_factors['q_values'][mode] = self.q_table.get(mode, 0.0)

            # Epsilon greedy strategy - not to be confused with the agent types :)
            # Effective exploration rate: constant by default (epsilon_decay == 1.0,
            # exact baseline behaviour), or exponentially annealed toward epsilon_min
            # so that late in the run agents act on their learned Q-values instead of
            # picking a source uniformly at random.
            eps_eff = self.epsilon
            _decay = getattr(self.model, 'epsilon_decay', 1.0)
            if _decay < 1.0:
                eps_eff = max(getattr(self.model, 'epsilon_min', 0.05),
                              self.epsilon * (_decay ** self.model.tick))
            decision_factors['epsilon_eff'] = eps_eff

            # Exploration case - record randomly chosen mode
            if random.random() < eps_eff: #epsilon parameter for randomness
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

                # No hardcoded biases — Q-values emerge purely from feedback
                # Agent-type differences arise from D/delta acceptance and network structure
                decision_factors['biases'] = {}

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

            # Instrumentation: tally the mode choice by branch (exploration vs
            # exploitation) so the epsilon floor is directly observable.
            _sel = decision_factors['selection_type']
            if chosen_mode in self.mode_choice_counts[_sel]:
                self.mode_choice_counts[_sel][chosen_mode] += 1

            self.tokens_this_tick = {chosen_mode: 1}
            self.last_queried_source_ids = []

            # Query source based on chosen mode

            if chosen_mode == "self_action":
                reports = self.report_beliefs(interest_point, query_radius)
                source_id = None  # No external source used

            elif chosen_mode == "human":
                source_id = self.select_human_source()
                if source_id is None:
                    return

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
                    reports = source_agent.report_beliefs(interest_point, query_radius, self, self.trust.get(source_id, 0.1))

                    # track AI source
                    self.last_queried_source_ids = [source_id]

                    if not hasattr(self, 'ai_info_sources'):
                        self.ai_info_sources = {}

                    # Track which cells got info from which AI
                    for cell, reported_value in reports.items():
                        self.ai_info_sources[cell] = source_id

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

                # Use the Bayesian update function — returns (significant_change, was_accepted)
                result = self.update_belief_bayesian(cell, reported_level, source_trust, source_id)
                significant_update, was_accepted = result if isinstance(result, tuple) else (result, False)

                # cum_accepted_ai: count ALL D/delta acceptances (not just significant changes).
                # This correctly captures confirming AI that increases confidence without changing
                # the level — late-stage echo chamber reinforcement that significant_update misses.
                if was_accepted and source_id and source_id.startswith("A_"):
                    self.cum_accepted_ai += 1  # cumulative; drives AECI classification

                # Track if this was a significant belief update (level or confidence change)
                if significant_update:
                    belief_updates += 1

                    # Track source acceptance (for per-period retainment metrics)
                    if source_id:
                        if source_id.startswith("H_"):
                            self.accepted_human += 1
                            if source_id in self.friends:
                                self.accepted_friend += 1
                        elif source_id.startswith("A_"):
                            self.accepted_ai += 1  # per-period counter (reset every 5 ticks)

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
                    # Linear confidence prevents confirming AI from gaining 5× volume advantage
                    # over truthful AI through confidence amplification alone (confidence^1.5 → 5.2× ratio)
                    score = (level / 5.0) * confidence if self.agent_type == "exploitative" else (
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
                        self.tokens_sent_total += 1

                # NOTE: correct_targets / incorrect_targets are updated in process_reward()
                # with delayed ground truth (15-25 ticks). DO NOT also update them here —
                # that would double-count every targeted cell and inflate precision metrics.

                # Snapshot pending info evaluations per targeted cell so process_reward can
                # retroactively evaluate source accuracy against real ground truth (Fix 5).
                # Entries may expire from pending_info_evaluations before process_reward fires,
                # so we keep a copy here.
                cell_eval_snapshots = {}
                for cell in targeted_cells:
                    cell_evals = [item for item in self.pending_info_evaluations if item[2] == cell]
                    if cell_evals:
                        cell_eval_snapshots[cell] = list(cell_evals)

                # Relief outcome delay: 15-25 ticks (realistic logistics delay)
                # Random variation models variable communication/logistics times
                relief_delay = random.randint(15, 25)
                self.pending_rewards.append((
                    self.model.tick + relief_delay,
                    responsible_mode,
                    reward_cells,
                    self.last_queried_source_ids,
                    cell_eval_snapshots
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
            for reward_data in expired:
                reward_tick, mode, cells_and_beliefs, source_ids = reward_data[0], reward_data[1], reward_data[2], reward_data[3]
                cell_eval_snapshots = reward_data[4] if len(reward_data) > 4 else {}
                if not cells_and_beliefs:
                    continue

                batch_reward = 0
                correct_in_batch = 0
                incorrect_in_batch = 0
                cell_rewards = []

                for cell, belief_level in cells_and_beliefs:
                    if not (0 <= cell[0] < self.model.width and 0 <= cell[1] < self.model.height):
                        continue

                    # Get the ACTUAL disaster level for this cell
                    actual_level = self.model.disaster_grid[cell[0], cell[1]]

                    # Define correctness criteria
                    is_correct = actual_level >= 3  # Keep this threshold for "high need"

                    if is_correct:
                        correct_in_batch += 1
                    else:
                        incorrect_in_batch += 1

                    # Calculate granular reward based on actual level
                    if actual_level == 5:
                        cell_reward = 5.0  # Perfect targeting
                    elif actual_level == 4:
                        cell_reward = 3.0  # Very good targeting
                    elif actual_level == 3:
                        cell_reward = 1.5  # Good targeting
                    elif actual_level == 2:
                        cell_reward = 0.0  # Neutral - borderline need
                    elif actual_level == 1:
                        cell_reward = -1.0  # Poor targeting - low need
                    else:  # Level 0
                        cell_reward = -2.0  # Very poor targeting - no need

                    cell_rewards.append(cell_reward)

                    # Fix 1: Retroactive correctness-based feedback for explorers.
                    # Explorer remote-cell queries get deferred in evaluate_pending_info
                    # (no reliable belief reference), but process_reward provides real
                    # ground truth for every targeted cell. Evaluate snapshotted pending
                    # info entries against actual_level now.
                    if self.agent_type == "exploratory" and cell in cell_eval_snapshots:
                        for eval_item in cell_eval_snapshots[cell]:
                            eval_source_id = eval_item[1]
                            eval_reported_level = eval_item[3]
                            level_err = abs(eval_reported_level - actual_level)
                            if level_err == 0:
                                acc_score = 1.0
                            elif level_err == 1:
                                acc_score = 0.5
                            elif level_err == 2:
                                acc_score = -0.2
                            else:
                                acc_score = -0.6
                            acc_reward = acc_score * 0.6 - 0.1
                            is_ai_src = eval_source_id.startswith("A_")
                            is_human_src = eval_source_id.startswith("H_")
                            eval_mode = "ai" if is_ai_src else ("human" if is_human_src else None)
                            # AI always knows truth; human on remote cell less certain
                            src_conf = 1.0 if is_ai_src else 0.5
                            if eval_mode and eval_mode in self.q_table:
                                lr = 0.25 * src_conf
                                self.q_table[eval_mode] += lr * (acc_reward - self.q_table[eval_mode])
                            if eval_source_id in self.q_table:
                                lr = 0.25 * src_conf
                                self.q_table[eval_source_id] += lr * (acc_reward - self.q_table[eval_source_id])
                            if eval_source_id in self.trust:
                                old_t = self.trust[eval_source_id]
                                if acc_reward < 0:
                                    t_target = max(0.0, 0.5 + acc_reward)
                                    t_lr = 0.25 * src_conf
                                else:
                                    t_target = min(1.0, 0.5 + 0.5 * acc_reward)
                                    t_lr = 0.12 * src_conf
                                self.trust[eval_source_id] = max(0.0, min(1.0, old_t + t_lr * (t_target - old_t)))

                    # Update beliefs based on ground truth
                    if cell in self.beliefs and isinstance(self.beliefs[cell], dict):
                        old_belief = self.beliefs[cell].copy()

                        # Direct update with ground truth (with some noise)
                        noise = random.choice([-1, 0, 0, 0, 1]) if random.random() < 0.2 else 0
                        corrected_level = max(0, min(5, actual_level + noise))

                        # Blend with existing belief
                        update_weight = 0.7  # Strong update toward reality
                        blended_level = int(round(update_weight * corrected_level +
                                                (1 - update_weight) * old_belief.get('level', 0)))

                        # Update confidence based on accuracy - KEY CHANGE: Make this agent-type dependent
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

                # Calculate batch reward based on correctness ratio and actual reward
                if cell_rewards:
                    # Blend components for final reward - KEY CHANGE: Make this agent-type dependent
                    # Uniform accuracy-based reward for both agent types.
                    # Agent-type differences in targeting emerge from belief accuracy
                    # differences (shaped by D/delta), not from reward shaping.
                    avg_actual_reward = sum(cell_rewards) / len(cell_rewards)
                    batch_reward = avg_actual_reward

                    # Cap the reward range
                    batch_reward = max(-3.0, min(5.0, batch_reward))
                else:
                    batch_reward = -1.0  # Penalty for targeting nothing

                total_reward += batch_reward

                self.correct_targets += correct_in_batch
                self.incorrect_targets += incorrect_in_batch

                # Normalize reward to [-1, 1] for Q-learning
                scaled_reward = max(-1.0, min(1.0, batch_reward / 5.0))
                target_trust = (scaled_reward + 1.0) / 2.0  # Map to [0,1] for trust

                # Uniform Q-update: agent-type differences emerge from D/delta, not learning rate
                if mode == "self_action":
                    old_q = self.q_table.get("self_action", 0.0)
                    new_q = old_q + self.learning_rate * (scaled_reward - old_q)
                    self.q_table["self_action"] = new_q

                elif source_ids:
                    # Update generic mode Q-value (fixes mode vs source ID mismatch)
                    # mode is what's used in action selection (e.g., "human", "A_0")
                    if mode in self.q_table:
                        old_mode_q = self.q_table[mode]
                        new_mode_q = old_mode_q + self.learning_rate * (scaled_reward - old_mode_q)
                        self.q_table[mode] = new_mode_q

                    # Also update specific source Q-values (for tracking individual sources)
                    for source_id in source_ids:
                        if source_id in self.q_table:
                            old_q = self.q_table[source_id]
                            new_q = old_q + self.learning_rate * (scaled_reward - old_q)
                            self.q_table[source_id] = new_q

                        if source_id in self.trust:
                            old_trust = self.trust[source_id]

                            # ASYMMETRIC trust update: penalize bad outcomes faster
                            if target_trust < old_trust:
                                # Bad outcome: faster learning rate to drop trust
                                effective_trust_lr = self.trust_learning_rate * 2.5
                            else:
                                # Good outcome: normal learning rate
                                effective_trust_lr = self.trust_learning_rate
                            trust_change = effective_trust_lr * (target_trust - old_trust)

                            new_trust = max(0.0, min(1.0, old_trust + trust_change))
                            self.trust[source_id] = new_trust

        except Exception as e:
            print(f"ERROR in Agent {self.unique_id} process_reward at tick {current_tick}: {e}")
            import traceback
            traceback.print_exc()

        return total_reward

    # =========================================================================
    # AGENT STEP: Three-Phase Decision Cycle
    # =========================================================================
    # Phase 1 (OBSERVE): sense_environment() - Agent observes local area
    # Phase 2 (REQUEST): seek_information() - Agent requests info to confirm (exploit) or explore (explore)
    # Phase 3 (DECIDE):  send_relief() - Agent decides where to send relief based on beliefs
    # =========================================================================

    def phase_observe(self):
        """Phase 1: OBSERVE - Sense local environment and update beliefs from direct perception."""
        self.sense_environment()

    def phase_request(self):
        """Phase 2: REQUEST - Seek additional information from humans or AI.
        - Exploiters query around believed epicenter (confirmation-seeking)
        - Explorers query high-uncertainty areas (information-seeking)
        """
        self.seek_information()

    def phase_decide(self):
        """Phase 3: DECIDE - Send relief based on current beliefs."""
        self.send_relief()

    def move(self):
        """Home-anchored mobility (design proposal §1). One grid step per tick.

        Exploitative agents are RETURNERS (Pappalardo et al. 2015): random walk
        near home_pos; outside r_home = 3 a return bias (p = 0.3) pulls each
        step back toward home. Coverage ~25-30 cells around home over a run.

        Exploratory agents are EXPLORERS: with p = 0.2 per tick they step toward
        their current highest-uncertainty target; beyond r_explore = 8 from home
        the same return bias kicks in first. Otherwise they random walk.
        Coverage ~100-200 cells, but still home-anchored.

        Movement IS the sensing mechanism (radius stays 0): a far-spawned
        returner genuinely cannot observe the disaster, which is what makes
        spatial periphery real.
        """
        home = getattr(self, 'home_pos', None) or self.initial_pos
        x, y = self.pos
        dist_home = math.sqrt((x - home[0]) ** 2 + (y - home[1]) ** 2)
        r_limit = 3 if self.agent_type == "exploitative" else 8

        target = None
        if dist_home > r_limit and random.random() < 0.3:
            target = home                       # return bias
        elif (self.agent_type == "exploratory" and dist_home <= r_limit
              and random.random() < 0.2):
            # Excursion step — only while inside the r_explore cap, so targets
            # beyond it cannot drag the agent arbitrarily far from home.
            target = self.find_highest_uncertainty_area()

        if target and tuple(target) != (x, y):
            # One step toward the target (8-directional), clipped to the grid
            nx_ = x + (1 if target[0] > x else -1 if target[0] < x else 0)
            ny_ = y + (1 if target[1] > y else -1 if target[1] < y else 0)
            new_pos = (int(np.clip(nx_, 0, self.model.width - 1)),
                       int(np.clip(ny_, 0, self.model.height - 1)))
        else:
            neighbors = self.model.grid.get_neighborhood(
                self.pos, moore=True, include_center=False)
            if not neighbors:
                return
            if self.agent_type == "exploitative":
                # Returners random-walk WITHIN the home disk: steps that would
                # leave radius r_home are not taken (unless already outside,
                # where the return bias above pulls the walk back).
                inside = [n for n in neighbors
                          if (n[0] - home[0]) ** 2 + (n[1] - home[1]) ** 2 <= r_limit ** 2]
                if inside:
                    neighbors = inside
            new_pos = random.choice(neighbors)

        if new_pos != self.pos:
            self.model.grid.move_agent(self, new_pos)
            self.pos = new_pos
        if not hasattr(self, 'cells_visited'):
            self.cells_visited = {self.pos}
        else:
            self.cells_visited.add(self.pos)

    def step(self):
        """Execute the decision cycle: Move → Sense → Query → Evaluate → Decide
        Movement (phase 0, only when model.mobility=1) relocates the agent one
        step; sensing radius stays 0, so movement is what determines coverage.
        Sensing is supplementary (updates beliefs from local environment).
        Querying is MANDATORY (agents must seek info each tick).
        Order matters: sensing first gives agents ground-truth beliefs that
        improve query target selection and info quality evaluation.
        """
        # Phase 0: MOVE - home-anchored mobility (off by default)
        if getattr(self.model, 'mobility', 0):
            self.move()

        # Phase 1: OBSERVE - sense local environment (supplementary)
        self.phase_observe()

        # Phase 2: REQUEST - seek information (MANDATORY, every tick)
        self.phase_request()

        # Phase 3: Evaluate pending info quality against current beliefs
        # (runs after both sense and query so beliefs are up-to-date)
        self.evaluate_pending_info()

        # Phase 4: DECIDE - send relief based on beliefs
        self.phase_decide()

        # Process delayed feedback from previous relief decisions
        reward = self.process_reward()

        # Maintenance: update trust and apply decay
        self.update_trust_for_accuracy()
        self.apply_trust_decay()
        self.apply_confidence_decay()
        return reward

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

                    # Apply direct trust update based on accuracy (SYMMETRIC: both boost AND penalty)
                    if ai_source in self.trust:
                        old_trust = self.trust[ai_source]
                        if accuracy > 0.8:  # High accuracy: small boost
                            trust_change = 0.03
                        elif accuracy < 0.4:  # Low accuracy: penalty (LARGER than boost)
                            trust_change = -0.08
                        else:
                            trust_change = 0.0  # Neutral range
                        new_trust = max(0.0, min(1.0, old_trust + trust_change))
                        self.trust[ai_source] = new_trust


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
        # disaster_grid is allocated np.zeros((width, height)) — axis 0 is x/width
        width, height = self.model.disaster_grid.shape
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

                # Constant sensing noise, independent of alignment (C8 fix).
                # The old `0.1 * ai_alignment_level` coupling made high-α AI
                # simultaneously more confirming AND noisier, so α no longer
                # isolated the confirmation effect in the alignment sweep.
                noise_prob = 0.1

                if random.random() < noise_prob:
                    value = max(0, min(5, value + random.choice([-1, 1])))

                self.sensed[cell] = value
                self.memory[(current_tick, cell)] = value

        # Update knowledge map in the model - FIXED INDENTATION
        if hasattr(self.model, 'ai_knowledge_maps'):
            knowledge_map = np.zeros((self.model.width, self.model.height))
            for cell in self.sensed:
                if 0 <= cell[0] < self.model.width and 0 <= cell[1] < self.model.height:
                    knowledge_map[cell[0], cell[1]] = 1
            self.model.ai_knowledge_maps[self.unique_id] = knowledge_map

    def report_beliefs(self, interest_point, query_radius, caller, caller_trust_in_ai):
        """
        Reports AI's beliefs about cells within query_radius of interest_point,
        applying alignment based on caller's trust and beliefs.

        `caller` is the querying HumanAgent. Its beliefs and agent_type are used
        in the alignment formula. For exploitative callers the AI confirms the
        community's network-consensus belief (not the individual prior) so that
        confirming AI amplifies social echo chambers rather than locking in
        idiosyncratic individual beliefs.
        """
        report = {}
        caller_beliefs = caller.beliefs if hasattr(caller, 'beliefs') else {}

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
                            # 3. No nearby sensed cells: use ONLY position-based estimate.
                            # DO NOT use caller_beliefs here — that creates implicit confirmation
                            # bias at ALL alpha levels (even α=0 "truthful" AI would echo the
                            # caller's belief for ~85% of unvisited cells, collapsing the α sweep).
                            # The alignment formula later applies α AFTER this base estimate, so
                            # the base must be belief-independent for α to have its intended effect.
                            dist_to_interest = math.sqrt(
                                (cell[0] - interest_point[0])**2 +
                                (cell[1] - interest_point[1])**2
                            )
                            scaled_dist = min(1.0, dist_to_interest / (query_radius + 1))
                            # Position-based prior: cells closer to queried epicenter more likely high
                            if scaled_dist < 0.3:
                                value_to_use = random.choice([2, 3, 3, 4, 4])
                            elif scaled_dist < 0.6:
                                value_to_use = random.choice([1, 2, 2, 3, 3])
                            else:
                                value_to_use = random.choice([0, 0, 1, 1, 2])

                        is_guessed = True

                # If we have a value to use (either sensed or guessed), process it
                if value_to_use is not None:
                    valid_cells_in_query.append(cell)
                    sensed_vals_list.append(int(value_to_use))

                    # Get the belief level that the confirming AI should target.
                    # For exploitative callers: use the network-consensus belief so that
                    # confirming AI amplifies the community's shared narrative (echo chamber).
                    # Without this, confirming individual priors disrupts social convergence
                    # and makes SECI less negative than the no-AI control — the opposite of H1.
                    # For exploratory callers: use individual belief (unchanged behaviour).
                    caller_belief_info = caller_beliefs.get(cell, {'level': 0, 'confidence': 0.1})
                    human_level = caller_belief_info.get('level', 0)
                    human_confidence = caller_belief_info.get('confidence', 0.1)
                    if (hasattr(caller, 'agent_type') and caller.agent_type == "exploitative"
                            and hasattr(caller, 'get_network_consensus')):
                        net_level, net_conf = caller.get_network_consensus(cell)
                        if net_level is not None and net_conf >= 0.2:
                            human_level = int(round(net_level))   # confirm community belief
                    human_vals_list.append(int(human_level))
                    human_confidence_list.append(human_confidence)

        # If no cells to report on, return empty
        if not valid_cells_in_query:
            if hasattr(self.model, 'debug_mode') and self.model.debug_mode and random.random() < 0.1:
                print(f"DEBUG: AI {self.unique_id} has no data to report!")
            return {}

        # Convert to numpy arrays for alignment logic
        sensed_vals = np.array(sensed_vals_list)
        human_vals = np.array(human_vals_list)
        human_conf = np.array(human_confidence_list)

        # --- Alignment Logic ---
        # α = 0  → AI reports pure ground truth (fully informative)
        # α = 1  → AI reports exactly what the agent already believes (pure confirmation)
        # α = 0.5 → AI reports the midpoint between truth and belief
        #
        # Simple linear interpolation: reported = truth + α * (belief - truth)
        #   = (1-α)*truth + α*belief
        #
        # Previous formula used α*(1 + conf*2 + trust_term) which amplified the
        # adjustment factor well above 1, causing AI to report PAST the agent's belief
        # (e.g. belief=4, truth=0, α=0.5 → factor≈1.5 → reported=6→clipped to 5).
        # That collapsed α=0.5, 0.8, 1.0 into the same clipped report, making the
        # alignment sweep meaningless and all intermediate α levels look like α=1.
        alignment_strength = self.model.ai_alignment_level

        belief_differences = human_vals - sensed_vals
        corrected = np.round(sensed_vals + alignment_strength * belief_differences)
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
                 d_exploit=2.0,          # Gap-scalar: acceptance threshold for exploitative agents
                 delta_exploit=3.5,      # Gap-scalar: acceptance sensitivity for exploitative agents
                 d_explor=4.0,           # Gap-scalar: acceptance threshold for exploratory agents
                 delta_explor=1.2,       # Gap-scalar: acceptance sensitivity for exploratory agents
                 epicenter=None,         # Optional fixed epicenter [x, y]; random if None
                 verification_probability=0.3,  # Per-attempt arrival prob of external "situation report"
                                                # verification for explorers' accepted remote reports
                 salience_weight=0.0,    # 0 = uniform verification rewards (baseline);
                                         # 1 = full severity-weighted (salience) evaluation — see C12
                 epsilon_decay=1.0,      # Per-tick multiplicative decay of the exploration rate.
                                         # 1.0 = constant epsilon (baseline, no decay). <1.0 anneals
                                         # epsilon toward epsilon_min so agents increasingly act on
                                         # learned Q-values instead of choosing a source at random.
                 epsilon_min=0.05,       # Floor for the annealed exploration rate (ignored when
                                         # epsilon_decay == 1.0).
                 # --- Spatial network / mobility switches (design proposal) -----------
                 # All three default to the pre-proposal behaviour so existing sweeps
                 # are unaffected; turn on together for the periphery/brokerage design.
                 mobility=0,              # 0 = immobile (baseline); 1 = home-anchored movement:
                                          # exploitative agents are "returners" (random walk within
                                          # r_home of their spawn cell), exploratory agents are
                                          # "explorers" (excursions toward high-uncertainty areas).
                 network_type='components',  # 'components' = disconnected type-homogeneous
                                          # communities (baseline). 'spatial_bridged' = spatially
                                          # embedded communities (Gaussian spawn around community
                                          # centroids) plus distance-decayed weak-tie bridges —
                                          # brokers exist by construction.
                 query_scope='global',    # 'global' = human queries may reach any human, with
                                          # non-friends at a small baseline weight (baseline).
                                          # 'network' = friends-only trust-weighted pool; with
                                          # p = 0.1 the query goes to the 2-hop neighbourhood
                                          # (friends-of-friends) instead, so reaching a stranger
                                          # requires an intermediary and access follows edges.
                 p_within=0.5,            # spatial_bridged: within-community edge probability
                 p_bridge=0.15,           # spatial_bridged: probability an agent carries one bridge
                 bridge_decay=2.0,        # spatial_bridged: bridge endpoint prob ∝ (1+d)^-decay
                 n_communities_per_type=4,  # spatial_bridged: community count per agent type
                 spawn_sigma=2.5          # spatial_bridged: Gaussian spawn spread around centroid
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
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.ticks = ticks
        self.low_trust_amplification_factor = low_trust_amplification_factor
        self.verification_probability = verification_probability
        self.salience_weight = salience_weight

        # Learning rates and biases
        self.exploit_trust_lr = exploit_trust_lr
        self.explor_trust_lr = explor_trust_lr
        self.d_exploit = d_exploit
        self.delta_exploit = delta_exploit
        self.d_explor = d_explor
        self.delta_explor = delta_explor

        # Spatial network / mobility switches (design proposal)
        self.mobility = mobility
        self.network_type = network_type
        self.query_scope = query_scope
        self.p_within = p_within
        self.p_bridge = p_bridge
        self.bridge_decay = bridge_decay
        self.n_communities_per_type = n_communities_per_type
        self.spawn_sigma = spawn_sigma
        # Populated by initialize_social_network():
        self.communities = []           # list of (set[node_id], type_label) — the unit for
                                        # SECI, rumor seeding, and component-level metrics
        self.agent_spawn_positions = None  # {node_id: (x, y)} when spawn is community-coupled
        self.bridge_agents = set()      # node_ids that DRAW a bridge (spatial_bridged only)
        self.bridge_endpoints = set()   # node_ids on either end of a bridge (broker flag)
        self.bridge_edges = []          # [(u, v), ...] weak-tie bridge edges

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

        # NOTE: in-simulation tipping-point tracking (tp_* attributes) was removed.
        # It used the old variance-based AECI sign convention (negative = echo chamber),
        # which is inverted relative to the current AECI formula (positive = echo
        # chamber), and its outputs were never consumed downstream. Transition timing
        # is measured post-hoc in test_filter_bubbles.py (_first_sustained_break /
        # _first_sustained_cross) instead.

        # Initialize disaster grid with Gaussian decay around an epicenter.
        self.disaster_grid = np.zeros((width, height), dtype=int)
        self.baseline_grid = np.zeros((width, height), dtype=int)
        if epicenter is not None:
            self.epicenter = tuple(epicenter)
        else:
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

        if self.network_type == 'spatial_bridged':
            # Bridges are SUPPOSED to connect the graph — communities live in
            # self.communities, not in the component structure.
            print(f"Successfully created spatial bridged network "
                  f"({len(self.communities)} communities, {num_components} component(s))")
        elif num_components < 2:
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
                             learning_rate=self.learning_rate,
                             epsilon=self.epsilon,
                             trust_learning_rate=current_trust_lr,
                             exploit_trust_lr=self.exploit_trust_lr,
                             explor_trust_lr=self.explor_trust_lr,
                             d_exploit=self.d_exploit,
                             delta_exploit=self.delta_exploit,
                             d_explor=self.d_explor,
                             delta_explor=self.delta_explor,
                             )
            self.humans[f"H_{i}"] = agent
            self.agent_list.append(agent)
            if self.agent_spawn_positions is not None:
                # spatial_bridged: spawn is coupled to the community centroid
                pos = self.agent_spawn_positions.get(
                    i, (random.randrange(width), random.randrange(height)))
            else:
                pos = (random.randrange(width), random.randrange(height))
            self.grid.place_agent(agent, pos)
            agent.pos = pos
            agent.initial_pos = pos  # spawn location for spatial periphery classification
            agent.home_pos = pos     # mobility anchor: agents return here (returners) or
                                     # base excursions from here (explorers)
            agent.cells_visited = {pos}  # coverage tracking (grows only when mobility=1)

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

        # Rumors are seeded per stored community (not per connected component):
        # with network_type='spatial_bridged' the graph is one component, but a
        # rumor is still a shared narrative of one social community.
        for i, (component_nodes, _community_type) in enumerate(self.communities):
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
            # EXPLOITERS: High trust in friends (same beliefs), low trust in non-friends
            # EXPLORERS: More uniform trust, willing to hear from anyone
            for other_id in all_human_ids:
                if agent.unique_id == other_id: continue # Skip self

                if agent.agent_type == "exploitative":
                    if other_id in agent.friends:
                        # Exploiters HIGHLY trust friends (they share beliefs)
                        initial_t = random.uniform(0.65, 0.80)
                    else:
                        # Exploiters are skeptical of non-friends
                        initial_t = random.uniform(0.15, 0.30)
                else:  # exploratory
                    # Explorers have more uniform trust - willing to hear from anyone
                    initial_t = random.uniform(base_human_trust - 0.05, base_human_trust + 0.05)
                    if other_id in agent.friends:
                        initial_t = min(1.0, initial_t + 0.05)  # Small friend boost
                agent.trust[other_id] = initial_t
                agent.q_table[other_id] = default_q_value # Initialize Q for this specific human

            # Initialize trust/Q for AI agents
            # EXPLOITERS: Start skeptical of AI (it's not in their social network)
            # EXPLORERS: More open to AI as information source
            for ai_id in all_ai_ids:
                if agent.agent_type == "exploitative":
                    # Exploiters start very skeptical of AI
                    initial_ai_t = random.uniform(0.10, 0.25)
                else:
                    # Explorers more open to AI
                    initial_ai_t = random.uniform(base_ai_trust_val, base_ai_trust_val + 0.15)
                agent.trust[ai_id] = max(0.0, min(1.0, initial_ai_t))
                agent.q_table[ai_id] = default_q_value # Initialize Q for this specific AI



    def debug_log(self, message, force=False):
        """Log debug messages if debug mode is enabled or forced."""

    def calculate_aeci_variance(self):
        """AECI-Var: AI Echo Chamber Index, variance-based, on a [-1, +1] scale.

        Compares the belief variance of AI-reliant agents against the global
        belief variance. NEGATIVE = AI-reliant agents more homogeneous than the
        population (AI echo chamber); positive = more diverse. Same sign
        convention as SECI and AECI-Err.

        Classification: WITHIN-TYPE median split by cum_accepted_ai (cumulative
        accepted AI belief updates = actual AI influence on beliefs), the same
        basis and split as AECI-Err. A population-level split let the type
        composition of the "AI-reliant" group vary with alpha (25% exploiters at
        alpha=0 vs 45-75% at alpha=1 in seeded probes), so the index partly
        measured which agent type self-selects into AI use rather than what AI
        does to beliefs (finding C11). Per-type values are averaged for the
        headline series; per-type values are stored in _last_metrics for
        diagnostics.

        Belief pool: L1+ beliefs only, exactly like SECI. Including the
        no-impact majority (~85% of cells at level 0) made the index dominated
        by how many cells agents believe are affected at all, and made
        total_bubble = |SECI| + |AECI-Var| a sum over two different belief
        populations, contradicting the METHODS claim that the two indices are
        structurally identical variance ratios.
        """
        # Global belief variance (all agents pooled, L1+ — same pool as SECI)
        all_beliefs = []
        for agent in self.humans.values():
            for belief_info in agent.beliefs.values():
                if isinstance(belief_info, dict):
                    level = belief_info.get('level', 0)
                    if not np.isnan(level) and level >= 1:
                        all_beliefs.append(level)
        global_var = np.var(all_beliefs) if len(all_beliefs) > 1 else 0.0

        def _aeci_var_for(type_label):
            """AECI-Var for one agent type; None if no valid group/signal."""
            type_agents = [a for a in self.humans.values()
                           if a.agent_type == type_label and hasattr(a, 'cum_accepted_ai')]
            if len(type_agents) < 4 or global_var <= 0:
                return None
            sorted_agents = sorted(type_agents, key=lambda a: a.cum_accepted_ai)
            top_half = sorted_agents[len(sorted_agents) // 2:]
            # Require actual AI influence — with zero acceptances everywhere
            # there is no "AI-reliant" group and no signal.
            if not top_half or top_half[-1].cum_accepted_ai <= 0:
                return None
            reliant_beliefs = []
            for agent in top_half:
                for belief_info in agent.beliefs.values():
                    if isinstance(belief_info, dict):
                        level = belief_info.get('level', 0)
                        if not np.isnan(level) and level >= 1:
                            reliant_beliefs.append(level)
            if len(reliant_beliefs) < 2:
                return None
            var_diff = np.var(reliant_beliefs) - global_var
            if var_diff < 0:  # Variance reduction → AI echo chamber (negative)
                return max(-1.0, var_diff / global_var)
            max_possible_var = 5.0  # Max variance for belief levels 0-5
            denom = max_possible_var - global_var
            return min(1.0, var_diff / denom) if denom > 1e-9 else 0.0

        per_type = {t: _aeci_var_for(t) for t in ("exploitative", "exploratory")}
        valid = [v for v in per_type.values() if v is not None]
        aeci_variance = float(np.mean(valid)) if valid else 0.0

        # Create a CORRECTLY formatted tuple
        aeci_variance_tuple = (self.tick, aeci_variance)

        # Update metrics dictionary with consistent format (+ per-type diagnostics)
        self._last_metrics['aeci_variance'] = {
            'tick': self.tick,
            'value': aeci_variance,
            'exploit': per_type["exploitative"],
            'explor': per_type["exploratory"],
        }

        # Store in the array with consistent format
        self.aeci_variance_data.append(aeci_variance_tuple)

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

    def get_mode_choice_summary(self):
        """Aggregate lifetime source-mode choices across agents, split by agent
        type and by whether the choice came from the epsilon exploration branch
        (uniform random over the 3 modes) or the Q-value exploitation branch
        (argmax over the Q-table).

        Diagnoses the exploration floor: with a constant epsilon the exploration
        branch alone forces each mode to be chosen ~epsilon/3 of the time
        regardless of what Q-learning prefers, so the AI-query share can never
        fall below ~epsilon/3 nor rise above ~1-2*epsilon/3. Compare the
        'exploration' sub-dict (should be ~1/3 each — pure noise) against
        'exploitation' (reflects learned Q-values).
        """
        modes = ('self_action', 'human', 'ai')
        summary = {}
        for label, atype in (('exploit', 'exploitative'), ('explor', 'exploratory')):
            agents = [a for a in self.humans.values() if a.agent_type == atype]
            expl = {m: 0 for m in modes}   # exploration branch (random)
            expt = {m: 0 for m in modes}   # exploitation branch (argmax)
            for a in agents:
                mc = getattr(a, 'mode_choice_counts', None)
                if not mc:
                    continue
                for m in modes:
                    expl[m] += mc['exploration'][m]
                    expt[m] += mc['exploitation'][m]
            n_expl = sum(expl.values())
            n_expt = sum(expt.values())
            total = n_expl + n_expt

            def _frac(d, n):
                return {m: (d[m] / n if n else 0.0) for m in modes}

            summary[label] = {
                'overall':          _frac({m: expl[m] + expt[m] for m in modes}, total),
                'exploration':      _frac(expl, n_expl),
                'exploitation':     _frac(expt, n_expt),
                'exploration_rate': (n_expl / total if total else 0.0),
                'n_decisions':      total,
            }
        return summary

    def initialize_social_network(self):
        """Initialize social network with multiple components and meaningful homophily."""
        # Calculate how many exploitative and exploratory agents we have
        num_exploitative = int(self.num_humans * self.share_exploitative)
        num_exploratory = self.num_humans - num_exploitative

        if self.network_type == 'spatial_bridged':
            return self.initialize_spatial_bridged_network(num_exploitative, num_exploratory)

        print(f"Initializing social network with {num_exploitative} exploitative and {num_exploratory} exploratory agents")

        # Create an empty graph
        self.social_network = nx.Graph()

        # Add all nodes
        for i in range(self.num_humans):
            self.social_network.add_node(i)

        # Type-homogeneous community assignment: exploiters cluster with exploiters,
        # explorers cluster with explorers.  This is the necessary condition for
        # the D/δ acceptance differences to produce DISTINCT echo-chamber dynamics
        # (SECI) per type.  Mixed communities caused both types to converge on the
        # same community consensus, making SECI identical across types.
        exploitative_agents = list(range(num_exploitative))
        exploratory_agents  = list(range(num_exploitative, self.num_humans))
        random.shuffle(exploitative_agents)
        random.shuffle(exploratory_agents)

        def _split_into_communities(agents, n_comms):
            """Divide agents into n_comms roughly equal, non-empty lists."""
            n_comms = max(1, min(n_comms, len(agents)))
            size = max(1, len(agents) // n_comms)
            comms = [agents[i * size:(i + 1) * size] for i in range(n_comms - 1)]
            comms.append(agents[(n_comms - 1) * size:])
            return [c for c in comms if c]

        # Target ~25 agents per community within each type
        n_exploit_comms = max(2, num_exploitative  // 25)
        n_explor_comms  = max(2, num_exploratory   // 25)

        communities = (
            _split_into_communities(exploitative_agents, n_exploit_comms) +
            _split_into_communities(exploratory_agents,  n_explor_comms)
        )
        print(f"Creating {len(communities)} type-homogeneous communities "
              f"({n_exploit_comms} exploitative, {n_explor_comms} exploratory)")

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

        # Store explicit community membership (design-proposal metric refactor).
        # SECI, rumor seeding, and the component-level metrics iterate these stored
        # communities instead of nx.connected_components, so they stay correct when
        # bridge edges (network_type='spatial_bridged') connect the whole graph into
        # one component. Here (no bridges) each component IS a community, and the
        # list preserves connected_components order so seeded runs are unchanged.
        self.communities = [
            (set(comp),
             "exploitative" if sum(1 for a in comp if a < num_exploitative) * 2 >= len(comp)
             else "exploratory")
            for comp in components
        ]

        return components

    def initialize_spatial_bridged_network(self, num_exploitative, num_exploratory):
        """Spatial caveman graph with distance-decayed weak-tie bridges (design proposal §2).

        1. Per agent type, place n_communities_per_type centroids on the grid with a
           minimum-separation constraint; agents spawn Gaussian(spawn_sigma) around
           their community centroid — network membership is coupled to geography.
        2. Erdős–Rényi edges with p_within inside each (type-homogeneous) community.
        3. Each agent independently becomes a bridge agent with p_bridge and draws ONE
           extra edge to an agent in another community, endpoint probability
           ∝ (1 + d)^(-bridge_decay) with d the grid distance between spawn cells
           (Kleinberg decay). Bridges are type-agnostic: the only routes between
           communities, so their endpoints are structural-hole brokers by construction.

        Sets: self.communities, self.agent_spawn_positions, self.bridge_agents,
        self.bridge_endpoints, self.bridge_edges. Returns connected components
        (for the caller's diagnostics print only — metrics use self.communities).
        """
        print(f"Initializing SPATIAL BRIDGED network with {num_exploitative} exploitative "
              f"and {num_exploratory} exploratory agents "
              f"(p_within={self.p_within}, p_bridge={self.p_bridge}, "
              f"bridge_decay={self.bridge_decay}, spawn_sigma={self.spawn_sigma})")

        self.social_network = nx.Graph()
        for i in range(self.num_humans):
            self.social_network.add_node(i)

        # --- 1a. Community centroids with minimum separation (rejection sampling) ---
        n_per_type = max(1, self.n_communities_per_type)
        n_centroids = 2 * n_per_type
        min_sep = max(3.0, min(self.width, self.height) / 4.0)
        centroids = []
        for _ in range(n_centroids):
            best_candidate, best_min_dist = None, -1.0
            for _attempt in range(200):
                cand = (random.uniform(0, self.width - 1), random.uniform(0, self.height - 1))
                d_min = min((math.dist(cand, c) for c in centroids), default=float('inf'))
                if d_min >= min_sep:
                    best_candidate = cand
                    break
                if d_min > best_min_dist:
                    best_min_dist, best_candidate = d_min, cand
            centroids.append(best_candidate)

        # --- 1b. Assign agents to type-homogeneous communities, one centroid each ---
        exploitative_agents = list(range(num_exploitative))
        exploratory_agents  = list(range(num_exploitative, self.num_humans))
        random.shuffle(exploitative_agents)
        random.shuffle(exploratory_agents)

        def _split(agents, n_comms):
            n_comms = max(1, min(n_comms, len(agents)))
            size = max(1, len(agents) // n_comms)
            comms = [agents[i * size:(i + 1) * size] for i in range(n_comms - 1)]
            comms.append(agents[(n_comms - 1) * size:])
            return [c for c in comms if c]

        member_lists = _split(exploitative_agents, n_per_type) + \
                       _split(exploratory_agents, n_per_type)
        self.communities = [
            (set(members),
             "exploitative" if members[0] < num_exploitative else "exploratory")
            for members in member_lists
        ]

        # --- 1c. Spawn positions: Gaussian around the community centroid ---
        self.agent_spawn_positions = {}
        for comm_idx, members in enumerate(member_lists):
            cx, cy = centroids[comm_idx % len(centroids)]
            for node in members:
                px = int(round(np.clip(random.gauss(cx, self.spawn_sigma), 0, self.width - 1)))
                py = int(round(np.clip(random.gauss(cy, self.spawn_sigma), 0, self.height - 1)))
                self.agent_spawn_positions[node] = (px, py)

        # --- 2. Within-community Erdős–Rényi edges ---
        for members in member_lists:
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    if random.random() < self.p_within:
                        self.social_network.add_edge(members[i], members[j])

        # --- 3. Weak-tie bridges with Kleinberg distance decay ---
        node_community = {}
        for comm_idx, members in enumerate(member_lists):
            for node in members:
                node_community[node] = comm_idx

        self.bridge_agents = set()
        self.bridge_endpoints = set()
        self.bridge_edges = []
        for node in range(self.num_humans):
            if random.random() >= self.p_bridge:
                continue
            src_pos = self.agent_spawn_positions[node]
            candidates, weights = [], []
            for other in range(self.num_humans):
                if node_community[other] == node_community[node]:
                    continue
                d = math.dist(src_pos, self.agent_spawn_positions[other])
                candidates.append(other)
                weights.append((1.0 + d) ** (-self.bridge_decay))
            if not candidates:
                continue
            total_w = sum(weights)
            r = random.random() * total_w
            cumulative, target = 0.0, candidates[-1]
            for cand, w in zip(candidates, weights):
                cumulative += w
                if r <= cumulative:
                    target = cand
                    break
            self.social_network.add_edge(node, target)
            self.bridge_agents.add(node)
            self.bridge_endpoints.update((node, target))
            self.bridge_edges.append((node, target))

        n_edges = self.social_network.number_of_edges()
        print(f"Spatial bridged network: {len(member_lists)} communities, {n_edges} edges, "
              f"{len(self.bridge_edges)} bridges from {len(self.bridge_agents)} bridge agents "
              f"({len(self.bridge_endpoints)} broker endpoints)")

        components = list(nx.connected_components(self.social_network))
        print(f"Connectivity: {len(components)} connected component(s)")
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
        Update disaster grid: stochastic drift toward baseline + random patch shocks.

        disaster_dynamics scales the PACE of change:
        0 = Static (no updates — null condition)
        1 = Slow   (0.5x drift and shock rates)
        2 = Medium (1.0x — default)
        3 = Rapid  (2.0x)

        Mechanics:
        - Drift: each cell independently moves 1 level toward its baseline
          (initial Gaussian) value with per-tick probability 0.05 * pace.
          This erodes shock damage over time and anchors the disaster field,
          so the environment is non-stationary but does not random-walk away.
        - Shock: with per-tick probability shock_probability * pace, a shock
          hits a random patch (Moore radius 2): the centre shifts by
          +/- shock_magnitude, neighbours by a distance-attenuated amount.
          Positive shocks create new hotspots agents must discover;
          negative shocks remove need where agents may still expect it.

        At the defaults (shock_probability=0.1, shock_magnitude=2,
        disaster_dynamics=2) this matches the METHODS description: per-tick
        drift toward baseline plus random shocks (p=0.10, magnitude +/-2).
        """
        # Store a copy of the current grid before updating
        if self.disaster_grid is not None:
            self.previous_grid = self.disaster_grid.copy()
        else:
            self.previous_grid = None

        if self.disaster_dynamics == 0:
            # Static disaster - no updates
            return

        pace = {1: 0.5, 2: 1.0, 3: 2.0}.get(
            self.disaster_dynamics, self.disaster_dynamics / 2.0)

        # 1. Stochastic drift toward baseline (cell-independent, 1 level per event)
        drift_mask = np.random.random(self.disaster_grid.shape) < (0.05 * pace)
        toward_baseline = np.sign(self.baseline_grid - self.disaster_grid)
        self.disaster_grid = np.clip(
            self.disaster_grid + drift_mask * toward_baseline, 0, 5).astype(int)

        # 2. Random patch shock (distance-attenuated, either sign)
        if random.random() < self.shock_probability * pace:
            cx = random.randrange(self.width)
            cy = random.randrange(self.height)
            sign = random.choice([-1, 1])
            radius = 2
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    x, y = cx + dx, cy + dy
                    if 0 <= x < self.width and 0 <= y < self.height:
                        dist = max(abs(dx), abs(dy))  # Chebyshev rings of the Moore patch
                        delta = sign * max(0, self.shock_magnitude - dist)
                        if delta != 0:
                            self.disaster_grid[x, y] = min(5, max(0, self.disaster_grid[x, y] + delta))

        # Detect significant changes
        if self.previous_grid is not None:
            grid_change = np.abs(self.disaster_grid - self.previous_grid)
            max_change = np.max(grid_change)
            if max_change >= self.event_threshold:
                self.event_ticks.append(self.tick)

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

        # Identify high-need cells (L3+, matching send_relief targeting threshold) that received no tokens
        need_mask = self.disaster_grid >= 3
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
            # Collect belief levels split by type so each type's SECI is
            # normalised against its own global variance.  Using a pooled
            # global_var (all agents combined) diluted the signal: because
            # explorers maintain high belief diversity, the pooled baseline
            # was always pulled upward, making both types look equally
            # un-echo-chambered.  Type-specific normalisation lets exploiters
            # (converging on the false epicenter) show a strongly negative
            # SECI while explorers (spread across uncertain cells) show a
            # weaker one.  The pooled global_var is retained for AECI, which
            # compares AI-reliant agents against the full population.
            # --- SECI: community-level variance vs global variance, split by type ---
            # Uses actual social-network components (not raw type pools) so the metric
            # reflects network-mediated convergence rather than the structural D/δ
            # difference between types (exploiters always converge more than explorers
            # just due to their narrow acceptance window, confounding the type-pool measure).
            #
            # For each connected component (community), pool L1+ beliefs of members.
            # Compare that community's internal variance to the global variance across
            # all L1+ beliefs.  Low community variance → agents share the same (possibly
            # false) focal belief → echo chamber signal (SECI < 0).
            # Split by community type (network is type-homogeneous) to retain separate
            # exploit / explor signals.
            #
            # Aligned with Cinelli et al. (2021 PNAS) which measures echo chambers as
            # within-community information homogeneity relative to the global pool.

            # Collect global L1+ beliefs (all types) for baseline variance
            all_belief_levels = []
            for agent in self.humans.values():
                if agent.beliefs:
                    for b in agent.beliefs.values():
                        if isinstance(b, dict) and b.get('level', 0) >= 1:
                            all_belief_levels.append(b.get('level', 0))

            global_var = np.var(all_belief_levels) if len(all_belief_levels) > 1 else 1e-6

            max_possible_var = 5.0
            def _seci_val(community_var, gvar):
                if gvar < 1e-9:
                    return 0.0
                var_diff = community_var - gvar
                if var_diff < 0:  # community more homogeneous than global → echo chamber
                    return max(-1.0, var_diff / gvar)
                else:             # community more diverse than global → anti-bubble
                    denom = max_possible_var - gvar
                    return min(1.0, var_diff / denom) if denom > 1e-9 else 0.0

            exploit_community_vars = []
            explor_community_vars  = []

            for component_nodes, community_type in self.communities:
                if len(component_nodes) <= 1:
                    continue
                # Stored communities are type-homogeneous by construction; the stored
                # label stays correct even when bridges merge the graph's components.
                comp_agents = [self.humans.get(f"H_{n}") for n in component_nodes
                               if self.humans.get(f"H_{n}") is not None]
                if not comp_agents:
                    continue
                comp_levels = [
                    b.get('level', 0)
                    for a in comp_agents if a.beliefs
                    for b in a.beliefs.values()
                    if isinstance(b, dict) and b.get('level', 0) >= 1
                ]
                if len(comp_levels) > 1:
                    cvar = np.var(comp_levels)
                    if community_type == "exploitative":
                        exploit_community_vars.append(cvar)
                    else:
                        explor_community_vars.append(cvar)

            mean_exploit_community_var = np.mean(exploit_community_vars) if exploit_community_vars else 1e-6
            mean_explor_community_var  = np.mean(explor_community_vars)  if explor_community_vars  else 1e-6

            seci_exploit_mean = _seci_val(mean_exploit_community_var, global_var)
            seci_explor_mean  = _seci_val(mean_explor_community_var,  global_var)

            # Store results with proper checks
            if True:

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

            # --- Component-AECI (per stored community) ---
            component_aeci_list = []
            for component_nodes, _community_type in self.communities:
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

            # --- Component SECI (per stored community) ---
            component_seci_list = []

            for component_nodes, _community_type in self.communities:
                if len(component_nodes) <= 1:
                    continue  # Skip components with only 1 node
                    
                # Get L1+ beliefs from this component (filter out L0 noise)
                component_belief_levels = []
                for node_id in component_nodes:
                    agent_id = f"H_{node_id}"
                    agent = self.humans.get(agent_id)
                    if agent and agent.beliefs:
                        for cell, belief_info in agent.beliefs.items():
                            if isinstance(belief_info, dict):
                                level = belief_info.get('level', 0)
                                if level >= 1:
                                    component_belief_levels.append(level)

                # Only calculate SECI if we have beliefs
                if len(component_belief_levels) > 0:
                    # Calculate global variance (all agents, L1+ only — consistent baseline)
                    all_belief_levels = []
                    for agent in self.humans.values():
                        for cell, belief_info in agent.beliefs.items():
                            if isinstance(belief_info, dict):
                                level = belief_info.get('level', 0)
                                if level >= 1:
                                    all_belief_levels.append(level)
                                
                    # More robust variance calculation
                    global_var = np.var(all_belief_levels) if len(all_belief_levels) > 1 else 1e-6
                    component_var = np.var(component_belief_levels) if len(component_belief_levels) > 1 else global_var
                    
                    # Calculate component SECI with same [-1,1] convention as agent-level SECI:
                    # negative = component beliefs more homogeneous than global (echo chamber)
                    # positive = component beliefs more diverse than global (healthy diversity)
                    if global_var > 1e-9:
                        var_diff = component_var - global_var
                        if var_diff < 0:  # Echo chamber: component more homogeneous
                            component_seci_val = max(-1.0, var_diff / global_var)
                        else:  # Diversification: component more varied
                            max_possible_var = 5.0
                            component_seci_val = min(1.0, var_diff / max(1e-9, max_possible_var - global_var))
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
            
            # --- Component AI Trust Variance (per stored community) ---
            component_ai_trust_var_list = []
            for component_nodes, _community_type in self.communities:
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

            # --- AECI-Err: Confidence-Weighted Belief Error split ---
            #
            # One of the three AECI constructs (unified naming, all with the
            # SECI sign convention: NEGATIVE = echo chamber):
            #   AECI-Acc: acceptance share accepted_AI/(accepted_AI+accepted_human)
            #             → stored as retain_aeci (0..1 reliance measure, unsigned)
            #   AECI-Err: THIS metric (confidence-weighted error split, [-1,+1])
            #   AECI-Var: variance ratio of AI-reliant vs global beliefs
            #             → calculate_aeci_variance() ([-1,+1])
            #
            # AECI-Err formula: for each agent compute the mean of
            #   confidence × |believed_level − true_level|
            # over all L1+ cells (high enough belief to matter).  This "confident
            # error" is the actual echo-chamber signal: agents locked into
            # confirming AI accumulate high-confidence FALSE beliefs.
            #
            # Within each type, median-split by cum_accepted_ai (accepted AI belief
            # updates — actual AI influence; same classification basis as AECI-Var):
            #   heavy_err > light_err → AI-heavy agents have more confident false
            #   beliefs → algorithmic echo chamber → AECI-Err < 0
            #   heavy_err < light_err → AI corrects beliefs → AECI-Err > 0
            #   AECI-Err ≈ 0 → AI usage has no directional effect on belief accuracy
            #
            # Normalised to [-1, +1] (SIGN CONVENTION CHANGED 2026-07: previously
            # positive = echo chamber; now negative = echo chamber to match SECI
            # and AECI-Var. JSONs written before this change carry the old sign —
            # the test_filter_bubbles collect step auto-converts unmarked files.)
            aeci_exp = []
            aeci_expl = []

            def _weighted_error(agents):
                """Mean confidence × |believed_level − truth| over L1+ beliefs.
                AI echo chamber = high confidence in WRONG beliefs."""
                errors = []
                for a in agents:
                    for cell, binfo in a.beliefs.items():
                        if not isinstance(binfo, dict):
                            continue
                        conf = binfo.get('confidence', 0)
                        if conf <= 0.1:
                            continue
                        lv = binfo.get('level', 0)
                        if lv < 1:
                            continue
                        if not (0 <= cell[0] < self.width and 0 <= cell[1] < self.height):
                            continue
                        truth = float(self.disaster_grid[cell[0], cell[1]])
                        errors.append(conf * abs(float(lv) - truth))
                return float(np.mean(errors)) if errors else 0.0

            for agent_type_label, aeci_list in [("exploitative", aeci_exp),
                                                  ("exploratory",  aeci_expl)]:
                type_agents = [a for a in self.humans.values()
                               if a.agent_type == agent_type_label
                               and hasattr(a, 'cum_accepted_ai')]
                if len(type_agents) < 4:
                    continue  # need at least 2 per half

                # Median split by cum_accepted_ai within this type: accepted AI
                # belief updates measure actual AI INFLUENCE on beliefs (queries
                # inflated by low-trust probing do not shape beliefs). This is the
                # single classification basis shared with AECI-Var. At high α,
                # acceptance rates converge across agents, so acceptances remain
                # proportional to calls and the split keeps its variation.
                sorted_agents = sorted(type_agents, key=lambda a: a.cum_accepted_ai)
                mid = len(sorted_agents) // 2
                ai_light = sorted_agents[:mid]
                ai_heavy = sorted_agents[mid:]

                light_err = _weighted_error(ai_light)
                heavy_err = _weighted_error(ai_heavy)

                err_diff = heavy_err - light_err
                # SIGN: negative = echo chamber (matches SECI / AECI-Var convention)
                if err_diff >= 0:   # AI-heavy more confident-wrong → AI echo chamber
                    aeci_val = -min(1.0, err_diff / max(heavy_err, 1e-6))
                else:               # AI-heavy more accurate → AI breaks bubbles
                    aeci_val = min(1.0, -err_diff / max(light_err, 1e-6))

                aeci_list.append(aeci_val)

            avg_aeci_exp  = float(np.mean(aeci_exp))  if aeci_exp  else 0.0
            avg_aeci_expl = float(np.mean(aeci_expl)) if aeci_expl else 0.0

            avg_aeci_exp = max(-1.0, min(1.0, avg_aeci_exp))
            avg_aeci_expl = max(-1.0, min(1.0, avg_aeci_expl))

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
                self.running_aeci_exp = max(-1.0, min(1.0, self.running_aeci_exp * 0.8 + avg_aeci_exp * 0.2))
                self.running_aeci_expl = max(-1.0, min(1.0, self.running_aeci_expl * 0.8 + avg_aeci_expl * 0.2))

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
# Metric Tracking Helpers
#########################################
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

    # Iterate stored communities when available (bridge edges may merge the
    # graph into one connected component); fall back to components otherwise.
    stored = getattr(model, 'communities', None)
    if stored:
        components = [members for members, _type in stored]
    else:
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

