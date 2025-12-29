"""
Diagnostic Script: Analyze Agent Information Acceptance Patterns

This script helps diagnose whether agents are properly accepting information
or if they've become "locked in" to specific sources.

Usage: Add this code to your experiment analysis after running a simulation.
"""

def diagnose_acceptance_patterns(model):
    """
    Analyzes agent acceptance patterns to identify filter bubbles and lock-in.

    Returns detailed statistics about:
    - Which agents accept information and from whom
    - Whether agents are mixed (use multiple sources) or specialized (one source)
    - Agent type differences
    """

    print("\n" + "="*70)
    print("DIAGNOSTIC: Agent Information Acceptance Patterns")
    print("="*70)

    # Categorize agents by acceptance patterns
    no_acceptance = []  # Agents who accepted nothing
    ai_only = []        # Only accepted from AI
    friend_only = []    # Only accepted from friends/humans
    mixed = []          # Accepted from both AI and friends

    # Track by agent type
    exploit_patterns = {"no_accept": 0, "ai_only": 0, "friend_only": 0, "mixed": 0}
    explor_patterns = {"no_accept": 0, "ai_only": 0, "friend_only": 0, "mixed": 0}

    for agent_id, agent in model.humans.items():
        # Get acceptance counts
        ai_accept = agent.accepted_ai
        human_accept = agent.accepted_human
        total_accept = ai_accept + human_accept

        # Categorize this agent
        if total_accept == 0:
            no_acceptance.append(agent_id)
            category = "no_accept"
        elif ai_accept > 0 and human_accept > 0:
            mixed.append(agent_id)
            category = "mixed"
        elif ai_accept > 0:
            ai_only.append(agent_id)
            category = "ai_only"
        else:
            friend_only.append(agent_id)
            category = "friend_only"

        # Track by agent type
        if agent.agent_type == "exploitative":
            exploit_patterns[category] += 1
        else:
            explor_patterns[category] += 1

    total_agents = len(model.humans)

    # Print summary
    print(f"\nTotal Agents: {total_agents}")
    print(f"  - Exploitative: {sum(exploit_patterns.values())}")
    print(f"  - Exploratory: {sum(explor_patterns.values())}")

    print("\n--- ACCEPTANCE PATTERNS ---")
    print(f"No Acceptance:  {len(no_acceptance):3d} ({100*len(no_acceptance)/total_agents:5.1f}%)")
    print(f"AI Only:        {len(ai_only):3d} ({100*len(ai_only)/total_agents:5.1f}%)")
    print(f"Friends Only:   {len(friend_only):3d} ({100*len(friend_only)/total_agents:5.1f}%)")
    print(f"Mixed Sources:  {len(mixed):3d} ({100*len(mixed)/total_agents:5.1f}%)")

    print("\n--- BY AGENT TYPE ---")
    print("Exploitative Agents:")
    for cat, count in exploit_patterns.items():
        total_exploit = sum(exploit_patterns.values())
        pct = 100 * count / total_exploit if total_exploit > 0 else 0
        print(f"  {cat:15s}: {count:3d} ({pct:5.1f}%)")

    print("\nExploratory Agents:")
    for cat, count in explor_patterns.items():
        total_explor = sum(explor_patterns.values())
        pct = 100 * count / total_explor if total_explor > 0 else 0
        print(f"  {cat:15s}: {count:3d} ({pct:5.1f}%)")

    # Calculate specialization index
    specialization_rate = (len(ai_only) + len(friend_only)) / total_agents

    print("\n--- FILTER BUBBLE INDICATORS ---")
    print(f"Specialization Rate: {100*specialization_rate:.1f}%")
    print(f"  (% of agents relying on single source type)")

    if specialization_rate > 0.7:
        print("  ⚠️  HIGH specialization - strong filter bubbles forming")
    elif specialization_rate > 0.4:
        print("  ℹ️  MODERATE specialization - filter bubbles present")
    else:
        print("  ✓ LOW specialization - diverse information seeking")

    # Detailed breakdown of sample agents
    print("\n--- SAMPLE AGENT DETAILS ---")

    # Pick one agent from each category for detailed analysis
    sample_agents = []
    if no_acceptance:
        sample_agents.append(("No Accept", model.humans[no_acceptance[0]]))
    if ai_only:
        sample_agents.append(("AI Only", model.humans[ai_only[0]]))
    if friend_only:
        sample_agents.append(("Friend Only", model.humans[friend_only[0]]))
    if mixed:
        sample_agents.append(("Mixed", model.humans[mixed[0]]))

    for category, agent in sample_agents:
        print(f"\n{category} - Agent {agent.unique_id} ({agent.agent_type}):")
        print(f"  Total calls: {agent.accum_calls_total}")
        print(f"  AI calls: {agent.accum_calls_ai} ({100*agent.accum_calls_ai/max(1,agent.accum_calls_total):.0f}%)")
        print(f"  Human calls: {agent.accum_calls_human} ({100*agent.accum_calls_human/max(1,agent.accum_calls_total):.0f}%)")
        print(f"  Acceptances:")
        print(f"    - AI: {agent.accepted_ai}")
        print(f"    - Human: {agent.accepted_human}")
        print(f"    - Friend: {agent.accepted_friend}")

        # Show Q-values to understand why they prefer certain sources
        print(f"  Q-values:")
        print(f"    - self_action: {agent.q_table.get('self_action', 0):.3f}")
        print(f"    - human: {agent.q_table.get('human', 0):.3f}")
        for k in range(model.num_ai):
            ai_id = f"A_{k}"
            print(f"    - {ai_id}: {agent.q_table.get(ai_id, 0):.3f}")

        # Show trust values
        print(f"  Sample trust values:")
        ai_trusts = [agent.trust.get(f"A_{k}", 0) for k in range(model.num_ai)]
        avg_ai_trust = sum(ai_trusts) / len(ai_trusts) if ai_trusts else 0
        print(f"    - Avg AI trust: {avg_ai_trust:.3f}")
        if agent.friends:
            friend_trusts = [agent.trust.get(f, 0) for f in list(agent.friends)[:3]]
            avg_friend_trust = sum(friend_trusts) / len(friend_trusts) if friend_trusts else 0
            print(f"    - Avg friend trust: {avg_friend_trust:.3f}")

    print("\n" + "="*70)

    # Return summary stats for further analysis
    return {
        "no_acceptance": len(no_acceptance),
        "ai_only": len(ai_only),
        "friend_only": len(friend_only),
        "mixed": len(mixed),
        "specialization_rate": specialization_rate,
        "exploit_patterns": exploit_patterns,
        "explor_patterns": explor_patterns
    }


def check_if_concerning(stats, model):
    """
    Determines if the acceptance patterns are concerning or expected.
    """
    print("\n" + "="*70)
    print("ASSESSMENT: Are These Patterns Concerning?")
    print("="*70)

    concerns = []
    expected_behaviors = []

    # Check 1: Too many agents with no acceptance
    if stats["no_acceptance"] / len(model.humans) > 0.3:
        concerns.append(
            "⚠️  >30% of agents accept NO information - may indicate:\n"
            "   - Confidence levels too high (beliefs locked in)\n"
            "   - Trust levels too low (no source is trusted)\n"
            "   - Learning rate too low (beliefs not updating)"
        )
    else:
        expected_behaviors.append(
            "✓ Acceptance rate is healthy (most agents updating beliefs)"
        )

    # Check 2: Specialization rate
    if stats["specialization_rate"] > 0.8:
        expected_behaviors.append(
            "✓ HIGH specialization (>80%) - EXPECTED for filter bubble research!\n"
            "  This shows agents segregating into distinct information ecosystems:\n"
            f"  - {stats['ai_only']} agents rely only on AI\n"
            f"  - {stats['friend_only']} agents rely only on friends\n"
            "  This is likely what your model is designed to show!"
        )
    elif stats["specialization_rate"] < 0.2:
        concerns.append(
            "⚠️  Very LOW specialization (<20%) - filter bubbles may not be forming\n"
            "   - Check if Q-learning is converging\n"
            "   - Consider increasing agent biases"
        )
    else:
        expected_behaviors.append(
            f"✓ MODERATE specialization ({100*stats['specialization_rate']:.0f}%) - reasonable for dynamic environment"
        )

    # Check 3: Agent type differences
    exploit_spec = (stats['exploit_patterns']['ai_only'] + stats['exploit_patterns']['friend_only']) / \
                   max(1, sum(stats['exploit_patterns'].values()))
    explor_spec = (stats['explor_patterns']['ai_only'] + stats['explor_patterns']['friend_only']) / \
                  max(1, sum(stats['explor_patterns'].values()))

    if abs(exploit_spec - explor_spec) < 0.1:
        concerns.append(
            "⚠️  Agent types behave similarly - differentiation may be too weak\n"
            "   - Exploitative and exploratory agents should show different patterns\n"
            "   - Check bias parameters (exploit_friend_bias, etc.)"
        )
    else:
        expected_behaviors.append(
            "✓ Agent types show different behaviors:\n"
            f"  - Exploitative specialization: {100*exploit_spec:.0f}%\n"
            f"  - Exploratory specialization: {100*explor_spec:.0f}%"
        )

    # Print assessment
    if expected_behaviors:
        print("\nEXPECTED BEHAVIORS (Not concerning):")
        for behavior in expected_behaviors:
            print(f"\n{behavior}")

    if concerns:
        print("\n\nPOTENTIAL CONCERNS:")
        for concern in concerns:
            print(f"\n{concern}")
    else:
        print("\n\n✅ No major concerns detected - patterns appear normal for this model!")

    print("\n" + "="*70)

    return len(concerns) == 0  # Returns True if no concerns


# Example usage in your experiment:
"""
# After running a simulation:
model = run_simulation(base_params)

# Run diagnostics
stats = diagnose_acceptance_patterns(model)
is_ok = check_if_concerning(stats, model)
"""
