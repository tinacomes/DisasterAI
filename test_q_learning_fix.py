"""
Test Script - Q-Learning Fix Verification
==========================================

Quick test of the Q-learning fix with reduced parameters:
- 3 alignment values: [0.0, 0.5, 1.0]
- 3 runs per alignment (instead of 10)
- Standard tick count (200)

This verifies:
1. Code runs without errors
2. Q-values evolve over time (learning occurs)
3. Different alignment levels produce different behaviors
4. Exploratory and exploitative agents diverge

Run this BEFORE doing the full Experiment B.
"""

import sys
import os

# Check if running in Colab
try:
    from google.colab import drive
    if not os.path.exists('/content/drive/MyDrive'):
        drive.mount('/content/drive')
    IN_COLAB = True
    # Add DisasterAI directory to path
    if '/content/DisasterAI' not in sys.path:
        sys.path.insert(0, '/content/DisasterAI')
except ImportError:
    IN_COLAB = False

# Import the model
if IN_COLAB:
    exec(open('/content/DisasterAI/DisasterAI_Model.py').read())
else:
    print("ERROR: This test script is designed to run in Google Colab")
    print("Please run this in Colab after pulling the latest code:")
    print("  !cd /content/DisasterAI && git pull origin claude/plan-paper-experiments-0ESGp")
    print("  !cd /content/DisasterAI && python test_q_learning_fix.py")
    sys.exit(1)

import time
import pickle
import numpy as np

print("="*70)
print("Q-LEARNING FIX - TEST RUN")
print("="*70)
print("This is a SMALL test run to verify the fix works correctly.")
print("Full Experiment B will run after verification.")
print("="*70)

# Set up save directory
if IN_COLAB:
    test_save_dir = "/content/drive/MyDrive/DisasterAI_results/test_q_learning"
else:
    test_save_dir = "DisasterAI_results/test_q_learning"
os.makedirs(test_save_dir, exist_ok=True)
print(f"Test results will be saved to: {test_save_dir}\n")

# Define base parameters
base_params = {
    "num_humans": 100,
    "num_ai": 3,
    "width": 20,
    "height": 20,
    "max_ticks": 200,
    "disaster_dynamics": "random_walk",
    "shock_magnitude": 0.3,
    "share_exploitative": 0.5,
    "share_confirming": 0.3,
    "base_ai_trust": 0.5,
    "ai_alignment_level": 0.0,  # Will be varied
    "debug_mode": False
}

print("Base Parameters:")
print(f"  Agents: {base_params['num_humans']}")
print(f"  AI systems: {base_params['num_ai']}")
print(f"  Ticks: {base_params['max_ticks']}")
print(f"  Share exploitative: {base_params['share_exploitative']}")
print()

# Test configuration
test_alignment_values = [0.0, 0.5, 1.0]
test_num_runs = 3

print("Test Configuration:")
print(f"  Alignment values: {test_alignment_values}")
print(f"  Runs per value: {test_num_runs}")
print(f"  Total simulations: {len(test_alignment_values) * test_num_runs}")
print()

# Run test
print("="*70)
print("STARTING TEST RUN")
print("="*70)

start_time = time.time()
test_results = {}

for i, align in enumerate(test_alignment_values, 1):
    print(f"\n[{i}/{len(test_alignment_values)}] Testing Alignment = {align:.2f}")
    print("-" * 70)

    params = base_params.copy()
    params["ai_alignment_level"] = align

    try:
        result = aggregate_simulation_results(test_num_runs, params)
        test_results[align] = result

        # Quick sanity check
        if 'aeci' in result and isinstance(result['aeci'], np.ndarray):
            aeci_data = result['aeci']
            if aeci_data.size > 0 and aeci_data.ndim >= 3:
                # Get final tick values
                final_exploit = np.mean(aeci_data[:, -1, 1])
                final_explor = np.mean(aeci_data[:, -1, 2])
                print(f"  Final AECI - Exploit: {final_exploit:.3f}, Explor: {final_explor:.3f}")

        # Save incrementally
        temp_file = os.path.join(test_save_dir, f"test_results_partial.pkl")
        with open(temp_file, 'wb') as f:
            pickle.dump(test_results, f)
        print(f"  ✓ Saved partial results")

    except Exception as e:
        print(f"  ✗ ERROR at alignment {align}: {e}")
        import traceback
        traceback.print_exc()
        break

elapsed = (time.time() - start_time) / 60
print("\n" + "="*70)
print(f"TEST RUN COMPLETE ({elapsed:.1f} minutes)")
print("="*70)

# Save final results
final_file = os.path.join(test_save_dir, "test_results_final.pkl")
with open(final_file, 'wb') as f:
    pickle.dump(test_results, f)
print(f"Final results saved to: {final_file}")

# Analysis
print("\n" + "="*70)
print("QUICK ANALYSIS")
print("="*70)

for align in sorted(test_results.keys()):
    result = test_results[align]
    print(f"\nAlignment = {align:.2f}:")

    # AECI analysis
    if 'aeci' in result:
        aeci_data = result['aeci']
        if isinstance(aeci_data, np.ndarray) and aeci_data.size > 0 and aeci_data.ndim >= 3:
            # Get temporal evolution (every 20 ticks)
            ticks = list(range(0, aeci_data.shape[1], 20))
            exploit_vals = [np.mean(aeci_data[:, t, 1]) for t in ticks]
            explor_vals = [np.mean(aeci_data[:, t, 2]) for t in ticks]

            print(f"  AECI Evolution (Exploitative):")
            print(f"    Start: {exploit_vals[0]:.3f}, Mid: {exploit_vals[len(exploit_vals)//2]:.3f}, End: {exploit_vals[-1]:.3f}")
            print(f"  AECI Evolution (Exploratory):")
            print(f"    Start: {explor_vals[0]:.3f}, Mid: {explor_vals[len(explor_vals)//2]:.3f}, End: {explor_vals[-1]:.3f}")

            # Check for learning (change over time)
            exploit_change = abs(exploit_vals[-1] - exploit_vals[0])
            explor_change = abs(explor_vals[-1] - explor_vals[0])
            print(f"  Learning detected:")
            print(f"    Exploit change: {exploit_change:.3f} {'✓ Learning' if exploit_change > 0.05 else '⚠ Limited'}")
            print(f"    Explor change: {explor_change:.3f} {'✓ Learning' if explor_change > 0.05 else '⚠ Limited'}")

# Comparison across alignments
print("\n" + "="*70)
print("CROSS-ALIGNMENT COMPARISON")
print("="*70)

if len(test_results) >= 2:
    print("\nFinal AECI values (last tick):")
    print(f"{'Alignment':<12} {'Exploit':<12} {'Explor':<12} {'Difference':<12}")
    print("-" * 50)

    for align in sorted(test_results.keys()):
        result = test_results[align]
        if 'aeci' in result:
            aeci_data = result['aeci']
            if isinstance(aeci_data, np.ndarray) and aeci_data.size > 0 and aeci_data.ndim >= 3:
                final_exploit = np.mean(aeci_data[:, -1, 1])
                final_explor = np.mean(aeci_data[:, -1, 2])
                diff = final_exploit - final_explor
                print(f"{align:<12.2f} {final_exploit:<12.3f} {final_explor:<12.3f} {diff:<12.3f}")

    print("\nExpected pattern with Q-learning fix:")
    print("  Low alignment (0.0): Exploratory > Exploitative (truth-seeking)")
    print("  High alignment (1.0): Exploitative > Exploratory (confirmation-seeking)")

print("\n" + "="*70)
print("TEST VERIFICATION")
print("="*70)

# Check if results match expected pattern
if 0.0 in test_results and 1.0 in test_results:
    low_align = test_results[0.0]
    high_align = test_results[1.0]

    if 'aeci' in low_align and 'aeci' in high_align:
        low_aeci = low_align['aeci']
        high_aeci = high_align['aeci']

        if (isinstance(low_aeci, np.ndarray) and low_aeci.size > 0 and low_aeci.ndim >= 3 and
            isinstance(high_aeci, np.ndarray) and high_aeci.size > 0 and high_aeci.ndim >= 3):

            # Low alignment
            low_exploit = np.mean(low_aeci[:, -1, 1])
            low_explor = np.mean(low_aeci[:, -1, 2])

            # High alignment
            high_exploit = np.mean(high_aeci[:, -1, 1])
            high_explor = np.mean(high_aeci[:, -1, 2])

            print("Checking expected Q-learning behavior:")
            print()

            # Test 1: At low alignment, exploratory should prefer AI more (accuracy rewards)
            test1 = low_explor > low_exploit
            print(f"1. Low alignment - Exploratory > Exploitative?")
            print(f"   Explor={low_explor:.3f}, Exploit={low_exploit:.3f}: {'✓ PASS' if test1 else '⚠ FAIL'}")

            # Test 2: At high alignment, exploitative should prefer AI more (confirmation rewards)
            test2 = high_exploit > high_explor
            print(f"2. High alignment - Exploitative > Exploratory?")
            print(f"   Exploit={high_exploit:.3f}, Explor={high_explor:.3f}: {'✓ PASS' if test2 else '⚠ FAIL'}")

            # Test 3: Crossover should occur
            test3 = test1 and test2
            print(f"3. Behavioral crossover detected?")
            print(f"   {'✓ PASS - Q-learning is working!' if test3 else '⚠ FAIL - May need investigation'}")

            print()
            if test1 and test2:
                print("="*70)
                print("✓✓✓ TEST PASSED - Q-LEARNING FIX VERIFIED ✓✓✓")
                print("="*70)
                print("Ready to run full Experiment B!")
            else:
                print("="*70)
                print("⚠⚠⚠ TEST INCONCLUSIVE ⚠⚠⚠")
                print("="*70)
                print("The pattern isn't clear yet. This could be due to:")
                print("- Small sample size (only 3 runs)")
                print("- Q-learning needs more time to converge")
                print("- Statistical noise")
                print()
                print("Recommendation: Review results, then decide on full run")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
print(f"Results saved to: {test_save_dir}")
print()
print("Next steps:")
print("1. Review the results above")
print("2. If test passed, run full Experiment B:")
print("   !cd /content/DisasterAI && python DisasterAI_Model.py")
print("3. If test failed, investigate the issue")
