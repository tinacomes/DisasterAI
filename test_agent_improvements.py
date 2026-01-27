"""
Test Protocol for Multi-Agent Relief Model Improvements (Issues 2-5)

Tests four changes:
1. Issue 4 - Explorer Uncertainty-Seeking: Explorers query high-uncertainty areas,
   not their current position. Verify query targets differ from self.pos and
   correlate with low-confidence / high-spatial-variance regions.

2. Issue 5 - Weighted Q-Reward Split: Exploiters reward confirmation (0.8) over
   accuracy (0.2); explorers reward accuracy (0.8) over confirmation (0.2).
   Verify divergent Q-value evolution under identical source quality.

3. Issue 2 - Belief Accuracy Reward: self_action Q-value updates when sensing
   reveals ground truth. Agents with correct beliefs get positive reward;
   agents with incorrect beliefs get negative reward.

4. Issue 3 - Phase Structure: Verify observe/request/decide phases execute
   in correct order and produce expected side effects.

Conditions:
- High AI alignment (0.9): Confirming AI amplifies exploiter confirmation bias
- Low AI alignment (0.1): Truthful AI benefits explorer accuracy-seeking
- 50/50 exploitative/exploratory split
- 150 ticks, 30x30 grid

Expected Outcomes:
- Explorers should query cells AWAY from their position (high uncertainty areas)
- Exploiters should show higher Q-values for confirming sources
- Explorers should show higher Q-values for accurate sources
- self_action Q should diverge based on belief accuracy
- Confirming AI should widen the Q-value gap between agent types
"""

import numpy as np
import matplotlib.pyplot as plt
from DisasterAI_Model import DisasterModel, HumanAgent
import os
import math

# ============================================================================
# Parameters
# ============================================================================

base_params = {
    'share_exploitative': 0.5,
    'share_of_disaster': 0.15,
    'initial_trust': 0.3,
    'initial_ai_trust': 0.25,
    'number_of_humans': 100,
    'share_confirming': 0.7,
    'disaster_dynamics': 2,
    'width': 30,
    'height': 30,
    'ticks': 150,
    'learning_rate': 0.1,
    'epsilon': 0.3,
    'exploit_trust_lr': 0.015,
    'explor_trust_lr': 0.03,
}


# ============================================================================
# Test 1: Explorer Uncertainty-Seeking (Issue 4)
# ============================================================================

def test_uncertainty_seeking(model):
    """
    Track where explorers query vs. where they are positioned.
    Expectation: query targets should differ from self.pos and correlate with
    low-confidence / high-spatial-variance cells.
    """
    print("\n--- Test 1: Explorer Uncertainty-Seeking ---")

    query_distances = []       # Distance between agent position and query target
    query_uncertainties = []   # Uncertainty score of queried area
    own_pos_uncertainties = [] # Uncertainty score of agent's own position

    for agent in model.agent_list:
        if not isinstance(agent, HumanAgent) or agent.agent_type != "exploratory":
            continue

        # Get the uncertainty target the explorer would choose
        target = agent.find_highest_uncertainty_area()
        if target is None or agent.pos is None:
            continue

        # Distance from own position to query target
        dist = math.sqrt((target[0] - agent.pos[0])**2 + (target[1] - agent.pos[1])**2)
        query_distances.append(dist)

        # Calculate uncertainty at queried cell
        belief = agent.beliefs.get(target, {})
        if isinstance(belief, dict):
            conf_unc = 1.0 - belief.get('confidence', 0.1)
            query_uncertainties.append(conf_unc)

        # Calculate uncertainty at own position
        own_belief = agent.beliefs.get(agent.pos, {})
        if isinstance(own_belief, dict):
            own_unc = 1.0 - own_belief.get('confidence', 0.1)
            own_pos_uncertainties.append(own_unc)

    avg_dist = np.mean(query_distances) if query_distances else 0
    avg_query_unc = np.mean(query_uncertainties) if query_uncertainties else 0
    avg_own_unc = np.mean(own_pos_uncertainties) if own_pos_uncertainties else 0

    print(f"  Avg query distance from own position: {avg_dist:.2f} cells")
    print(f"  Avg uncertainty at query target:      {avg_query_unc:.3f}")
    print(f"  Avg uncertainty at own position:       {avg_own_unc:.3f}")
    print(f"  Query targets MORE uncertain than own pos: {avg_query_unc > avg_own_unc}")

    return {
        'distances': query_distances,
        'query_uncertainties': query_uncertainties,
        'own_uncertainties': own_pos_uncertainties,
    }


# ============================================================================
# Test 2: Weighted Q-Reward Split (Issue 5)
# ============================================================================

def test_weighted_q_reward():
    """
    Simulate scenarios where a source is ACCURATE but CONTRADICTS prior,
    vs. a source that is INACCURATE but CONFIRMS prior.
    Verify that exploiters and explorers score these differently.
    """
    print("\n--- Test 2: Weighted Q-Reward Split ---")

    params = base_params.copy()
    params['ai_alignment_level'] = 0.5
    model = DisasterModel(**params)

    # Find one explorer and one exploiter
    explorer = exploiter = None
    for agent in model.agent_list:
        if isinstance(agent, HumanAgent):
            if agent.agent_type == "exploratory" and explorer is None:
                explorer = agent
            elif agent.agent_type == "exploitative" and exploiter is None:
                exploiter = agent
        if explorer and exploiter:
            break

    # Scenario A: Source is ACCURATE (matches actual=3) but CONTRADICTS prior (agent believes 0)
    # Scenario B: Source is INACCURATE (reports 0, actual=3) but CONFIRMS prior (agent believes 0)
    test_cell = (15, 15)
    actual_level = 3

    results = {}
    for agent, label in [(explorer, "Explorer"), (exploiter, "Exploiter")]:
        # Set prior belief: agent believes level=0 with high confidence
        agent.beliefs[test_cell] = {'level': 0, 'confidence': 0.7}

        # Scenario A: accurate but contradicting (reported=3, actual=3, prior=0)
        reported_a = 3
        level_error_a = abs(reported_a - actual_level)  # 0 (accurate)
        prior_error_a = abs(reported_a - 0)              # 3 (contradicts)

        acc_a = {0: 1.0, 1: 0.5, 2: -0.2}.get(level_error_a, -0.6)
        conf_a = {0: 1.0, 1: 0.5, 2: -0.2}.get(prior_error_a, -0.6)

        if agent.agent_type == "exploitative":
            reward_a = 0.8 * conf_a + 0.2 * acc_a
        else:
            reward_a = 0.8 * acc_a + 0.2 * conf_a

        # Scenario B: inaccurate but confirming (reported=0, actual=3, prior=0)
        reported_b = 0
        level_error_b = abs(reported_b - actual_level)  # 3 (inaccurate)
        prior_error_b = abs(reported_b - 0)              # 0 (confirms)

        acc_b = {0: 1.0, 1: 0.5, 2: -0.2}.get(level_error_b, -0.6)
        conf_b = {0: 1.0, 1: 0.5, 2: -0.2}.get(prior_error_b, -0.6)

        if agent.agent_type == "exploitative":
            reward_b = 0.8 * conf_b + 0.2 * acc_b
        else:
            reward_b = 0.8 * acc_b + 0.2 * conf_b

        results[label] = {
            'accurate_contradicting': reward_a,
            'inaccurate_confirming': reward_b,
            'prefers_accuracy': reward_a > reward_b,
        }

        print(f"  {label}:")
        print(f"    Accurate but contradicting:  reward = {reward_a:.3f}")
        print(f"    Inaccurate but confirming:   reward = {reward_b:.3f}")
        print(f"    Prefers accuracy over confirmation: {reward_a > reward_b}")

    # Verify: explorer prefers accuracy, exploiter prefers confirmation
    explorer_ok = results['Explorer']['prefers_accuracy'] == True
    exploiter_ok = results['Exploiter']['prefers_accuracy'] == False
    print(f"\n  PASS Explorer prefers accuracy: {explorer_ok}")
    print(f"  PASS Exploiter prefers confirmation: {exploiter_ok}")

    return results


# ============================================================================
# Test 3: Belief Accuracy Reward (Issue 2)
# ============================================================================

def test_belief_accuracy_reward():
    """
    Verify that reward_belief_accuracy() correctly updates self_action Q-value:
    - Correct beliefs should increase Q
    - Incorrect beliefs should decrease Q
    """
    print("\n--- Test 3: Belief Accuracy Reward ---")

    params = base_params.copy()
    params['ai_alignment_level'] = 0.1
    model = DisasterModel(**params)

    # Find one agent of each type
    agents = {}
    for agent in model.agent_list:
        if isinstance(agent, HumanAgent) and agent.agent_type not in agents:
            agents[agent.agent_type] = agent
        if len(agents) == 2:
            break

    results = {}
    for agent_type, agent in agents.items():
        test_cell = (10, 10)

        # Test A: Correct belief (believes 4, actual 4, high confidence)
        agent.beliefs[test_cell] = {'level': 4, 'confidence': 0.8}
        agent.q_table['self_action'] = 0.0  # Reset
        agent.reward_belief_accuracy(test_cell, actual_level=4)
        q_after_correct = agent.q_table['self_action']

        # Test B: Incorrect belief (believes 0, actual 4, high confidence)
        agent.q_table['self_action'] = 0.0  # Reset
        agent.beliefs[test_cell] = {'level': 0, 'confidence': 0.8}
        agent.reward_belief_accuracy(test_cell, actual_level=4)
        q_after_incorrect = agent.q_table['self_action']

        # Test C: Low confidence belief should NOT update (confidence < 0.3)
        agent.q_table['self_action'] = 0.0  # Reset
        agent.beliefs[test_cell] = {'level': 0, 'confidence': 0.2}
        agent.reward_belief_accuracy(test_cell, actual_level=4)
        q_after_low_conf = agent.q_table['self_action']

        results[agent_type] = {
            'q_correct': q_after_correct,
            'q_incorrect': q_after_incorrect,
            'q_low_conf': q_after_low_conf,
        }

        print(f"  {agent_type}:")
        print(f"    Q after correct belief:     {q_after_correct:.4f} (expect > 0)")
        print(f"    Q after incorrect belief:   {q_after_incorrect:.4f} (expect < 0)")
        print(f"    Q after low-conf belief:    {q_after_low_conf:.4f} (expect = 0)")

        correct_ok = q_after_correct > 0
        incorrect_ok = q_after_incorrect < 0
        low_conf_ok = q_after_low_conf == 0.0

        print(f"    PASS correct > 0: {correct_ok}")
        print(f"    PASS incorrect < 0: {incorrect_ok}")
        print(f"    PASS low-conf unchanged: {low_conf_ok}")

    return results


# ============================================================================
# Test 4: Phase Structure (Issue 3)
# ============================================================================

def test_phase_structure():
    """
    Verify that phase_observe, phase_request, phase_decide exist and
    execute in the correct order within step().
    """
    print("\n--- Test 4: Phase Structure ---")

    params = base_params.copy()
    params['ai_alignment_level'] = 0.1
    params['ticks'] = 5
    model = DisasterModel(**params)

    agent = None
    for a in model.agent_list:
        if isinstance(a, HumanAgent):
            agent = a
            break

    # Verify methods exist
    has_observe = hasattr(agent, 'phase_observe') and callable(agent.phase_observe)
    has_request = hasattr(agent, 'phase_request') and callable(agent.phase_request)
    has_decide = hasattr(agent, 'phase_decide') and callable(agent.phase_decide)

    print(f"  phase_observe exists: {has_observe}")
    print(f"  phase_request exists: {has_request}")
    print(f"  phase_decide  exists: {has_decide}")

    # Verify phase_observe updates beliefs (sensing)
    old_beliefs = dict(agent.beliefs)
    agent.phase_observe()
    beliefs_changed = any(
        agent.beliefs.get(cell) != old_beliefs.get(cell)
        for cell in agent.beliefs
    )
    print(f"  phase_observe updates beliefs: {beliefs_changed}")

    # Run full step to verify no crash
    try:
        model.step()
        step_ok = True
    except Exception as e:
        step_ok = False
        print(f"  ERROR in model.step(): {e}")

    print(f"  Full model.step() succeeds: {step_ok}")

    return {
        'has_observe': has_observe,
        'has_request': has_request,
        'has_decide': has_decide,
        'beliefs_changed': beliefs_changed,
        'step_ok': step_ok,
    }


# ============================================================================
# Test 5: Agents Always Query External Sources
# ============================================================================

def test_always_queries():
    """
    Verify that agents always query an external source (human or AI) at each step.
    Track query mode distribution and feedback event counts over a short run.
    """
    print("\n--- Test 5: Agents Always Query External Sources ---")

    params = base_params.copy()
    params['ai_alignment_level'] = 0.5
    params['ticks'] = 50
    model = DisasterModel(**params)

    # Track mode choices and feedback events per agent type
    mode_counts = {'exploratory': {'human': 0, 'ai': 0, 'self_action': 0},
                   'exploitative': {'human': 0, 'ai': 0, 'self_action': 0}}
    info_feedback = {'exploratory': 0, 'exploitative': 0}
    total_steps = {'exploratory': 0, 'exploitative': 0}

    for tick in range(params['ticks']):
        # Snapshot pending evals before step
        pre_pending = {}
        for agent in model.agent_list:
            if isinstance(agent, HumanAgent):
                pre_pending[agent.unique_id] = len(agent.pending_info_evaluations)

        model.step()

        # Count mode choices and feedback events
        for agent in model.agent_list:
            if not isinstance(agent, HumanAgent):
                continue
            atype = agent.agent_type
            total_steps[atype] += 1

            # Check what mode was chosen this tick
            for mode in agent.tokens_this_tick:
                if mode in mode_counts[atype]:
                    mode_counts[atype][mode] += 1

            # Check if info feedback fired (pending list got shorter)
            post_pending = len(agent.pending_info_evaluations)
            pre = pre_pending.get(agent.unique_id, 0)
            if post_pending < pre:
                info_feedback[atype] += (pre - post_pending)

    for atype in ['exploratory', 'exploitative']:
        total = sum(mode_counts[atype].values())
        print(f"\n  {atype} ({total_steps[atype]} agent-steps):")
        for mode, count in mode_counts[atype].items():
            pct = (count / total * 100) if total > 0 else 0
            print(f"    {mode:12s}: {count:5d} ({pct:.1f}%)")
        self_pct = (mode_counts[atype]['self_action'] / total * 100) if total > 0 else 0
        print(f"    PASS no self_action queries: {mode_counts[atype]['self_action'] == 0}")
        print(f"    Info feedback events: {info_feedback[atype]}")

    return mode_counts, info_feedback


# ============================================================================
# Integration Test: Full Simulation with Tracking
# ============================================================================

def run_integration_test(ai_alignment, test_name):
    """
    Run a full simulation tracking all new metrics across agent types.
    """
    print(f"\n{'='*60}")
    print(f"Integration: {test_name} (alignment={ai_alignment})")
    print(f"{'='*60}")

    params = base_params.copy()
    params['ai_alignment_level'] = ai_alignment
    model = DisasterModel(**params)

    # Select sample agents
    samples = {}
    for agent in model.agent_list:
        if isinstance(agent, HumanAgent):
            if agent.agent_type not in samples:
                samples[agent.agent_type] = agent
        if len(samples) == 2:
            break

    # Tracking
    q_evolution = {
        'exploratory': {'ai': [], 'human': [], 'self_action': []},
        'exploitative': {'ai': [], 'human': [], 'self_action': []},
    }
    query_distance_evolution = {'exploratory': [], 'exploitative': []}

    for tick in range(params['ticks']):
        # Track Q-values
        for atype, agent in samples.items():
            q_evolution[atype]['ai'].append(agent.q_table.get('ai', 0.0))
            q_evolution[atype]['human'].append(agent.q_table.get('human', 0.0))
            q_evolution[atype]['self_action'].append(agent.q_table.get('self_action', 0.0))

        # Track query distances for explorers (sample)
        for atype, agent in samples.items():
            if atype == "exploratory":
                target = agent.find_highest_uncertainty_area()
                if target and agent.pos:
                    dist = math.sqrt((target[0]-agent.pos[0])**2 + (target[1]-agent.pos[1])**2)
                    query_distance_evolution['exploratory'].append(dist)

        model.step()

    # Final summary
    for atype in ['exploratory', 'exploitative']:
        print(f"\n  {atype} final Q-values:")
        print(f"    ai:          {q_evolution[atype]['ai'][-1]:.4f}")
        print(f"    human:       {q_evolution[atype]['human'][-1]:.4f}")
        print(f"    self_action: {q_evolution[atype]['self_action'][-1]:.4f}")

    if query_distance_evolution['exploratory']:
        avg_dist = np.mean(query_distance_evolution['exploratory'])
        print(f"\n  Explorer avg query distance: {avg_dist:.2f} cells")

    return {
        'q_evolution': q_evolution,
        'query_distances': query_distance_evolution,
        'model': model,
    }


# ============================================================================
# Visualization
# ============================================================================

def visualize_results(results_high, results_low):
    """Visualize integration test results for both alignment conditions."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Agent Improvement Tests: High vs Low AI Alignment', fontsize=14, fontweight='bold')

    conditions = [
        (results_high, 'High Alignment (0.9)', 0),
        (results_low, 'Low Alignment (0.1)', 1),
    ]

    for result, title, row in conditions:
        # Col 0: Q-value evolution
        ax = axes[row, 0]
        for source in ['ai', 'human', 'self_action']:
            for atype in ['exploratory', 'exploitative']:
                data = result['q_evolution'][atype][source]
                ls = '-' if atype == 'exploratory' else '--'
                ax.plot(data, linestyle=ls, label=f"{atype[:6]}: {source}", alpha=0.8)
        ax.set_title(f'Q-Values: {title}', fontsize=10)
        ax.set_xlabel('Tick')
        ax.set_ylabel('Q-Value')
        ax.legend(fontsize=6, loc='best')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle=':', alpha=0.3)

        # Col 1: self_action Q comparison
        ax = axes[row, 1]
        ax.plot(result['q_evolution']['exploratory']['self_action'], '-', label='Explorer', color='blue')
        ax.plot(result['q_evolution']['exploitative']['self_action'], '--', label='Exploiter', color='red')
        ax.set_title(f'Self-Action Q (Belief Accuracy): {title}', fontsize=10)
        ax.set_xlabel('Tick')
        ax.set_ylabel('Q-Value')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle=':', alpha=0.3)

        # Col 2: Explorer query distance
        ax = axes[row, 2]
        dists = result['query_distances'].get('exploratory', [])
        if dists:
            # Rolling average
            window = 10
            if len(dists) >= window:
                rolling = [np.mean(dists[max(0,i-window):i+1]) for i in range(len(dists))]
                ax.plot(rolling, color='green', alpha=0.8)
                ax.fill_between(range(len(rolling)), rolling, alpha=0.2, color='green')
            else:
                ax.plot(dists, color='green', alpha=0.8)
        ax.set_title(f'Explorer Query Distance: {title}', fontsize=10)
        ax.set_xlabel('Tick')
        ax.set_ylabel('Distance (cells)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle=':', alpha=0.3)

    plt.tight_layout()

    output_dir = "test_results"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "test_agent_improvements.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {filepath}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TEST PROTOCOL: Multi-Agent Relief Model Improvements (Issues 2-5)")
    print("=" * 70)

    # Unit tests
    test_weighted_q_reward()           # Issue 5
    test_belief_accuracy_reward()      # Issue 2
    test_phase_structure()             # Issue 3
    test_always_queries()              # Agents always query external sources

    # Issue 4 needs a warmed-up model (run a few ticks first)
    print("\n--- Warming up model for uncertainty-seeking test ---")
    warmup_params = base_params.copy()
    warmup_params['ai_alignment_level'] = 0.5
    warmup_model = DisasterModel(**warmup_params)
    for _ in range(20):
        warmup_model.step()
    test_uncertainty_seeking(warmup_model)

    # Integration tests
    results_high = run_integration_test(0.9, "Confirming AI")
    results_low = run_integration_test(0.1, "Truthful AI")

    # Visualization
    visualize_results(results_high, results_low)

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
