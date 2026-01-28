"""
Diagnostic test for checking agent mechanics:
1. Exploiters prefer confirming info
2. Explorers prefer accurate info
3. Exploiters prefer social network cells
4. AI alignment effects on filter bubbles
"""

import random
import numpy as np
from DisasterAI_Model import DisasterModel, HumanAgent

def run_short_simulation(ai_alignment, ticks=50):
    """Run a short simulation and collect key metrics."""
    params = {
        'share_exploitative': 0.5,
        'share_of_disaster': 0.15,
        'initial_trust': 0.3,
        'initial_ai_trust': 0.25,
        'number_of_humans': 50,
        'share_confirming': 0.7,
        'disaster_dynamics': 2,
        'width': 20,
        'height': 20,
        'ticks': ticks,
        'learning_rate': 0.1,
        'epsilon': 0.3,
        'ai_alignment_level': ai_alignment,
    }

    model = DisasterModel(**params)

    # Run simulation
    for _ in range(ticks):
        model.step()

    return model

def analyze_agent_behavior(model):
    """Analyze agent behavior patterns."""
    exploiters = [a for a in model.agent_list if isinstance(a, HumanAgent) and a.agent_type == "exploitative"]
    explorers = [a for a in model.agent_list if isinstance(a, HumanAgent) and a.agent_type == "exploratory"]

    results = {
        'exploit': {'q_self': [], 'q_human': [], 'q_ai': [], 'ai_calls': [], 'human_calls': [], 'trust_ai': [], 'trust_human_friends': [], 'belief_accuracy': []},
        'explor': {'q_self': [], 'q_human': [], 'q_ai': [], 'ai_calls': [], 'human_calls': [], 'trust_ai': [], 'trust_human_friends': [], 'belief_accuracy': []}
    }

    for agent_type, agents in [('exploit', exploiters), ('explor', explorers)]:
        for agent in agents:
            # Q-values
            results[agent_type]['q_self'].append(agent.q_table.get('self_action', 0))
            results[agent_type]['q_human'].append(agent.q_table.get('human', 0))
            results[agent_type]['q_ai'].append(agent.q_table.get('ai', 0))

            # Call counts
            results[agent_type]['ai_calls'].append(agent.accum_calls_ai)
            results[agent_type]['human_calls'].append(agent.accum_calls_human)

            # Trust values
            ai_trusts = [agent.trust.get(f"A_{i}", 0) for i in range(model.num_ai)]
            results[agent_type]['trust_ai'].append(np.mean(ai_trusts) if ai_trusts else 0)

            friend_trusts = [agent.trust.get(f, 0) for f in agent.friends if f in agent.trust]
            results[agent_type]['trust_human_friends'].append(np.mean(friend_trusts) if friend_trusts else 0)

            # Belief accuracy (MAE)
            mae = 0
            count = 0
            for cell, belief in agent.beliefs.items():
                if isinstance(belief, dict) and 0 <= cell[0] < model.width and 0 <= cell[1] < model.height:
                    mae += abs(belief.get('level', 0) - model.disaster_grid[cell])
                    count += 1
            results[agent_type]['belief_accuracy'].append(mae / count if count > 0 else 0)

    return results

def check_mechanic_1_confirmation_preference():
    """Check: Do exploiters prefer confirming info and explorers prefer accurate info?"""
    print("\n" + "="*60)
    print("MECHANIC 1: Information Preference")
    print("="*60)
    print("Expected: Exploiters prefer confirming info, Explorers prefer accurate info")
    print("-"*60)

    # Run with mixed AI (some truthful, some confirming behavior via alignment)
    model = run_short_simulation(ai_alignment=0.5, ticks=50)
    results = analyze_agent_behavior(model)

    print(f"\nExploiter Q-values:")
    print(f"  self_action: {np.mean(results['exploit']['q_self']):.3f}")
    print(f"  human:       {np.mean(results['exploit']['q_human']):.3f}")
    print(f"  ai:          {np.mean(results['exploit']['q_ai']):.3f}")

    print(f"\nExplorer Q-values:")
    print(f"  self_action: {np.mean(results['explor']['q_self']):.3f}")
    print(f"  human:       {np.mean(results['explor']['q_human']):.3f}")
    print(f"  ai:          {np.mean(results['explor']['q_ai']):.3f}")

    print(f"\nBelief Accuracy (lower = better):")
    print(f"  Exploiters: {np.mean(results['exploit']['belief_accuracy']):.3f}")
    print(f"  Explorers:  {np.mean(results['explor']['belief_accuracy']):.3f}")

    # Diagnostic: Check if feedback mechanism is firing
    print(f"\nFeedback Check:")
    print(f"  Exploiter calls: AI={np.mean(results['exploit']['ai_calls']):.1f}, Human={np.mean(results['exploit']['human_calls']):.1f}")
    print(f"  Explorer calls:  AI={np.mean(results['explor']['ai_calls']):.1f}, Human={np.mean(results['explor']['human_calls']):.1f}")

def check_mechanic_2_social_network_preference():
    """Check: Do exploiters prefer cells connected to social network?"""
    print("\n" + "="*60)
    print("MECHANIC 2: Social Network Preference")
    print("="*60)
    print("Expected: Exploiters should prefer querying friends and/or targeting cells friends know about")
    print("-"*60)

    model = run_short_simulation(ai_alignment=0.5, ticks=50)
    results = analyze_agent_behavior(model)

    print(f"\nTrust in Friends (higher = stronger preference):")
    print(f"  Exploiters: {np.mean(results['exploit']['trust_human_friends']):.3f}")
    print(f"  Explorers:  {np.mean(results['explor']['trust_human_friends']):.3f}")

    print(f"\nHuman Query Rate:")
    exploit_human_rate = np.mean([h/(h+a+0.001) for h, a in zip(results['exploit']['human_calls'], results['exploit']['ai_calls'])])
    explor_human_rate = np.mean([h/(h+a+0.001) for h, a in zip(results['explor']['human_calls'], results['explor']['ai_calls'])])
    print(f"  Exploiters: {exploit_human_rate:.2%}")
    print(f"  Explorers:  {explor_human_rate:.2%}")

    # Check if exploiters are querying about cells their friends know
    # This is tricky - let me check the actual targeting logic
    print("\n  NOTE: Current implementation - exploiters target their OWN believed epicenter,")
    print("        not cells their friends know about. This may need review.")

def check_mechanic_3_ai_alignment_effect():
    """Check: Does AI alignment create expected filter bubble patterns?"""
    print("\n" + "="*60)
    print("MECHANIC 3: AI Alignment Effect on Filter Bubbles")
    print("="*60)
    print("Expected: High alignment breaks social bubbles but creates AI bubbles")
    print("          Low alignment: explorers use AI, exploiter bubbles remain")
    print("-"*60)

    # Run with low alignment (truthful AI)
    print("\n--- Low Alignment (0.1 - Truthful AI) ---")
    model_low = run_short_simulation(ai_alignment=0.1, ticks=50)
    results_low = analyze_agent_behavior(model_low)

    print(f"  Exploiter AI trust: {np.mean(results_low['exploit']['trust_ai']):.3f}")
    print(f"  Explorer AI trust:  {np.mean(results_low['explor']['trust_ai']):.3f}")
    print(f"  Exploiter Q(ai):    {np.mean(results_low['exploit']['q_ai']):.3f}")
    print(f"  Explorer Q(ai):     {np.mean(results_low['explor']['q_ai']):.3f}")

    # Run with high alignment (confirming AI)
    print("\n--- High Alignment (0.9 - Confirming AI) ---")
    model_high = run_short_simulation(ai_alignment=0.9, ticks=50)
    results_high = analyze_agent_behavior(model_high)

    print(f"  Exploiter AI trust: {np.mean(results_high['exploit']['trust_ai']):.3f}")
    print(f"  Explorer AI trust:  {np.mean(results_high['explor']['trust_ai']):.3f}")
    print(f"  Exploiter Q(ai):    {np.mean(results_high['exploit']['q_ai']):.3f}")
    print(f"  Explorer Q(ai):     {np.mean(results_high['explor']['q_ai']):.3f}")

    print("\n--- Comparison ---")
    print(f"  Exploiter AI trust change (high-low): {np.mean(results_high['exploit']['trust_ai']) - np.mean(results_low['exploit']['trust_ai']):.3f}")
    print(f"  Explorer AI trust change (high-low):  {np.mean(results_high['explor']['trust_ai']) - np.mean(results_low['explor']['trust_ai']):.3f}")

    # Expected: High alignment should make exploiters trust AI MORE (because it confirms)
    #           High alignment should make explorers trust AI LESS (because it's inaccurate)

def check_mechanic_4_delayed_feedback():
    """Check: Is delayed feedback mechanism working?"""
    print("\n" + "="*60)
    print("MECHANIC 4: Delayed Feedback Mechanism")
    print("="*60)
    print("Expected: Relief feedback delayed 15-25 ticks, info feedback 3-15 ticks")
    print("-"*60)

    model = run_short_simulation(ai_alignment=0.5, ticks=50)

    # Check pending rewards and info evaluations
    pending_rewards_count = sum(len(a.pending_rewards) for a in model.agent_list if isinstance(a, HumanAgent))
    pending_info_count = sum(len(a.pending_info_evaluations) for a in model.agent_list if isinstance(a, HumanAgent))

    print(f"\n  Total pending rewards: {pending_rewards_count}")
    print(f"  Total pending info evaluations: {pending_info_count}")

    # Check if Q-values are being updated
    q_values = {'self': [], 'human': [], 'ai': []}
    for a in model.agent_list:
        if isinstance(a, HumanAgent):
            q_values['self'].append(a.q_table.get('self_action', 0))
            q_values['human'].append(a.q_table.get('human', 0))
            q_values['ai'].append(a.q_table.get('ai', 0))

    print(f"\n  Q-value ranges (should vary if feedback working):")
    print(f"    self_action: min={min(q_values['self']):.3f}, max={max(q_values['self']):.3f}, std={np.std(q_values['self']):.3f}")
    print(f"    human:       min={min(q_values['human']):.3f}, max={max(q_values['human']):.3f}, std={np.std(q_values['human']):.3f}")
    print(f"    ai:          min={min(q_values['ai']):.3f}, max={max(q_values['ai']):.3f}, std={np.std(q_values['ai']):.3f}")

def main():
    print("="*60)
    print("DISASTER AI MECHANICS DIAGNOSTIC")
    print("="*60)

    check_mechanic_1_confirmation_preference()
    check_mechanic_2_social_network_preference()
    check_mechanic_3_ai_alignment_effect()
    check_mechanic_4_delayed_feedback()

    print("\n" + "="*60)
    print("SUMMARY OF POTENTIAL ISSUES")
    print("="*60)
    print("""
1. EXPLOITERS SOCIAL NETWORK CELL PREFERENCE:
   - Current: Exploiters target their OWN believed epicenter
   - Expected: Should prefer cells their social network knows about
   - FIX NEEDED: Modify seek_information() to factor in friend beliefs

2. AI ALIGNMENT MECHANISM:
   - Current: High alignment = AI confirms caller beliefs
   - Issue: If AI confirms wrong beliefs, it REINFORCES filter bubbles
   - Expected: High alignment should somehow BREAK social bubbles
   - Question: Should high alignment mean AI provides alternative info?

3. CONFIRMATION BIAS IN FEEDBACK:
   - Exploiters rate sources 80% on confirmation, 20% on accuracy
   - This means they'll trust inaccurate sources that confirm beliefs
   - May need to check if this creates the expected dynamics

4. QUERY TARGETING:
   - Explorers seek uncertainty areas (correct)
   - Exploiters seek their believed epicenter (may not match intent)
""")

if __name__ == "__main__":
    main()
