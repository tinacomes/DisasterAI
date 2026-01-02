"""
Diagnostic Script - Verify Experiment Results
==============================================

Run this after experiments complete to verify:
1. All expected files were created
2. Data integrity (no corrupted pickles)
3. Quick summary statistics
4. Data structure validation

Run in Colab after experiments finish.
"""

import os
import pickle
import numpy as np
from datetime import datetime

# Configuration
try:
    from google.colab import drive
    if not os.path.exists('/content/drive/MyDrive'):
        drive.mount('/content/drive')
    RESULTS_DIR = "/content/drive/MyDrive/DisasterAI_results"
    IN_COLAB = True
except ImportError:
    RESULTS_DIR = "DisasterAI_results"
    IN_COLAB = False

print("="*70)
print("EXPERIMENT RESULTS DIAGNOSTIC CHECK")
print("="*70)
print(f"Checking directory: {RESULTS_DIR}\n")

#########################################
# Check 1: File Existence
#########################################

print("="*70)
print("CHECK 1: File Existence")
print("="*70)

expected_files = {
    'Experiment A': 'results_experiment_A.pkl',
    'Experiment B': 'results_experiment_B.pkl',
    'Experiment D': 'results_experiment_D.pkl'
}

files_found = {}
for exp_name, filename in expected_files.items():
    filepath = os.path.join(RESULTS_DIR, filename)
    exists = os.path.exists(filepath)
    files_found[exp_name] = exists

    status = "✓" if exists else "✗"
    print(f"{status} {exp_name}: {filename}")

    if exists:
        size_mb = os.path.getsize(filepath) / (1024*1024)
        mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
        print(f"   Size: {size_mb:.2f} MB | Modified: {mod_time.strftime('%Y-%m-%d %H:%M')}")

all_found = all(files_found.values())
print(f"\n{'✓ All files found!' if all_found else '⚠ Some files missing!'}")

#########################################
# Check 2: Data Integrity
#########################################

print("\n" + "="*70)
print("CHECK 2: Data Integrity (Can files be loaded?)")
print("="*70)

loaded_results = {}
for exp_name, filename in expected_files.items():
    if not files_found[exp_name]:
        print(f"⊗ {exp_name}: Skipped (file not found)")
        continue

    filepath = os.path.join(RESULTS_DIR, filename)
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        loaded_results[exp_name] = data
        print(f"✓ {exp_name}: Loaded successfully")
        print(f"   Type: {type(data)}")
        if isinstance(data, dict):
            print(f"   Parameter values: {sorted(list(data.keys()))}")
    except Exception as e:
        print(f"✗ {exp_name}: FAILED to load")
        print(f"   Error: {e}")

#########################################
# Check 3: Data Structure Validation
#########################################

print("\n" + "="*70)
print("CHECK 3: Data Structure Validation")
print("="*70)

def validate_experiment_data(exp_name, data):
    """Validate structure of experiment results."""
    if not isinstance(data, dict):
        print(f"✗ {exp_name}: Not a dictionary!")
        return False

    if len(data) == 0:
        print(f"✗ {exp_name}: Empty results!")
        return False

    # Check first parameter
    first_key = sorted(list(data.keys()))[0]
    first_result = data[first_key]

    print(f"✓ {exp_name}: Structure OK")
    print(f"   Parameters: {len(data)} values")
    print(f"   Sample parameter: {first_key}")

    if isinstance(first_result, dict):
        metrics = list(first_result.keys())
        print(f"   Metrics available: {len(metrics)}")

        # Check for key metrics
        key_metrics = ['seci', 'aeci_variance', 'trust_stats']
        found_metrics = [m for m in key_metrics if m in metrics]
        missing_metrics = [m for m in key_metrics if m not in metrics]

        if found_metrics:
            print(f"   ✓ Found: {', '.join(found_metrics)}")
        if missing_metrics:
            print(f"   ⚠ Missing: {', '.join(missing_metrics)}")

        # Check data shapes
        if 'seci' in first_result:
            seci_data = first_result['seci']
            if isinstance(seci_data, np.ndarray):
                print(f"   SECI shape: {seci_data.shape}")
            else:
                print(f"   SECI type: {type(seci_data)}")

    return True

for exp_name, data in loaded_results.items():
    validate_experiment_data(exp_name, data)
    print()

#########################################
# Check 4: Quick Summary Statistics
#########################################

print("="*70)
print("CHECK 4: Summary Statistics")
print("="*70)

def get_final_metrics(results_dict):
    """Extract final values from results."""
    summary = {}

    for param_val, res in results_dict.items():
        if not isinstance(res, dict):
            continue

        param_summary = {'param': param_val}

        # Extract final SECI
        if 'seci' in res and isinstance(res['seci'], np.ndarray):
            seci_data = res['seci']
            if seci_data.ndim >= 3 and seci_data.shape[1] > 0:
                # Average across runs, get final tick, both agent types
                final_seci_exploit = np.mean(seci_data[:, -1, 1])
                final_seci_explor = np.mean(seci_data[:, -1, 2])
                param_summary['seci_exploit'] = final_seci_exploit
                param_summary['seci_explor'] = final_seci_explor

        # Extract final AECI
        if 'aeci_variance' in res:
            aeci_data = res['aeci_variance']
            if isinstance(aeci_data, np.ndarray) and aeci_data.size > 0:
                try:
                    if aeci_data.ndim >= 2:
                        final_aeci = np.mean(aeci_data[:, -1, 1])
                    else:
                        final_aeci = np.mean(aeci_data[-1])
                    param_summary['aeci_var'] = final_aeci
                except:
                    pass

        summary[param_val] = param_summary

    return summary

for exp_name, data in loaded_results.items():
    print(f"\n{exp_name}:")
    print("-" * 60)

    summary = get_final_metrics(data)

    if summary:
        # Print as table
        print(f"{'Param':<10} {'SECI (Exploit)':<15} {'SECI (Explor)':<15} {'AECI-Var':<10}")
        print("-" * 60)

        for param, metrics in sorted(summary.items()):
            seci_exp = metrics.get('seci_exploit', float('nan'))
            seci_expl = metrics.get('seci_explor', float('nan'))
            aeci = metrics.get('aeci_var', float('nan'))

            # Handle both float and tuple params (Exp D uses tuples)
            if isinstance(param, tuple):
                param_str = f"LR={param[0]:.2f},ε={param[1]:.1f}"
            else:
                param_str = f"{param:.2f}"

            print(f"{param_str:<20} {seci_exp:<15.3f} {seci_expl:<15.3f} {aeci:<10.3f}")
    else:
        print("  ⚠ Could not extract summary statistics")

#########################################
# Check 5: Data Completeness
#########################################

print("\n" + "="*70)
print("CHECK 5: Data Completeness (Runs per parameter)")
print("="*70)

for exp_name, data in loaded_results.items():
    print(f"\n{exp_name}:")

    for param_val, res in sorted(data.items()):
        if isinstance(res, dict) and 'seci' in res:
            seci_data = res['seci']
            if isinstance(seci_data, np.ndarray) and seci_data.ndim >= 3:
                num_runs = seci_data.shape[0]
                num_ticks = seci_data.shape[1]

                # Handle both float and tuple params
                if isinstance(param_val, tuple):
                    param_str = f"LR={param_val[0]:.2f},ε={param_val[1]:.1f}"
                else:
                    param_str = f"{param_val:.2f}"

                print(f"  Param {param_str}: {num_runs} runs × {num_ticks} ticks")

#########################################
# Final Summary
#########################################

print("\n" + "="*70)
print("DIAGNOSTIC SUMMARY")
print("="*70)

all_passed = all(files_found.values()) and len(loaded_results) == len(expected_files)

if all_passed:
    print("✓✓✓ ALL CHECKS PASSED ✓✓✓")
    print("\nResults are ready for analysis!")
    print("\nNext step: Run analyze_existing_results.py")
else:
    print("⚠⚠⚠ SOME CHECKS FAILED ⚠⚠⚠")
    print("\nIssues detected:")

    if not all(files_found.values()):
        missing = [name for name, found in files_found.items() if not found]
        print(f"  - Missing files: {', '.join(missing)}")

    if len(loaded_results) < len(expected_files):
        print(f"  - Could not load {len(expected_files) - len(loaded_results)} file(s)")

print("="*70)
