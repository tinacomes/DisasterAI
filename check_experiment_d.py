"""
Quick check for Experiment D results

Add this code right after Experiment D completes in your main file.
"""

# After Experiment D runs, add this:
print("\n" + "="*80)
print("CHECKING EXPERIMENT D: Information Acceptance Patterns")
print("="*80)

# Import the diagnostic
from diagnostic_acceptance_patterns import diagnose_acceptance_patterns, check_if_concerning

# Check one parameter combination from Experiment D
# Let's use the baseline learning rate and epsilon
test_params = base_params.copy()
test_params["learning_rate"] = 0.05
test_params["epsilon"] = 0.3

print(f"\nTesting with LR={test_params['learning_rate']}, epsilon={test_params['epsilon']}")
print(f"AI alignment level: {test_params['ai_alignment_level']}")

# Run one simulation
test_model = run_simulation(test_params)

# Diagnose
stats = diagnose_acceptance_patterns(test_model)
is_ok = check_if_concerning(stats, test_model)

# Additional context for Experiment D
print("\n" + "="*80)
print("CONTEXT FOR EXPERIMENT D (Learning Rate & Epsilon)")
print("="*80)
print("""
What to expect in Experiment D:

1. HIGHER learning rates (0.07) → Faster Q-value convergence
   - Agents should specialize MORE (pick AI or friends faster)
   - Less mixed acceptance expected

2. LOWER learning rates (0.03) → Slower Q-value convergence
   - Agents may remain more mixed
   - More exploration of different sources

3. HIGHER epsilon (0.3) → More exploration
   - Should see more mixed acceptance patterns
   - Less specialization

4. LOWER epsilon (0.2) → More exploitation
   - Should see MORE specialization
   - Agents lock into AI-only or friend-only patterns

EXPECTED PATTERN:
- High LR + Low epsilon → HIGHEST specialization (agents quickly find and stick to best source)
- Low LR + High epsilon → LOWEST specialization (agents keep exploring)
""")

if stats["specialization_rate"] > 0.6:
    print("\n✓ Your observation is EXPECTED!")
    print("  High specialization indicates Q-learning is working:")
    print("  - Agents learned which sources give good rewards")
    print("  - They're now exploiting those sources")
    print("  - This creates the filter bubbles you're studying!")
else:
    print("\n⚠️  Low specialization might indicate:")
    print("  - Learning rate too low (agents not converging)")
    print("  - Epsilon too high (too much random exploration)")
    print("  - Or dynamic environment prevents convergence")

print("\n" + "="*80)
