"""
Master Analysis Script - Run All Analyses
==========================================

Runs all analysis scripts in correct order to generate:
- Verification checks
- Temporal evolution plots
- Chamber dynamics analysis
- Experiment-specific plots
- Cross-experiment comparisons
- Publication tables
- Key findings summary
"""

import subprocess
import sys
import os

def run_script(script_name, description):
    """Run a Python script and report status."""
    print("\n" + "="*70)
    print(f"▶ {description}")
    print("="*70)

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,
            text=True,
            check=True
        )
        print(f"✓ {script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {script_name} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"✗ {script_name} failed: {e}")
        return False

print("="*70)
print("COMPREHENSIVE ANALYSIS PIPELINE")
print("="*70)
print("This will generate all plots, tables, and summaries for your paper.")
print("="*70)

# Track successes and failures
results = {}

# 1. Verification
results['verify'] = run_script(
    'verify_results.py',
    'Step 1: Verify Data Integrity'
)

# 2. Experiment B Temporal Analysis
results['temporal_b'] = run_script(
    'plot_experiment_b_temporal.py',
    'Step 2: Experiment B Temporal Evolution Plots'
)

# 3. Chamber Dynamics Analysis (KEY!)
results['dynamics'] = run_script(
    'plot_chamber_dynamics.py',
    'Step 3: Chamber Dynamics Analysis (Amplification vs Disruption)'
)

# 4. Experiment D Robustness Analysis
results['experiment_d'] = run_script(
    'plot_experiment_d.py',
    'Step 4: Experiment D Robustness Analysis'
)

# 5. Enhanced Analysis (cross-experiment)
results['enhanced'] = run_script(
    'analyze_existing_results.py',
    'Step 5: Enhanced Cross-Experiment Analysis'
)

# 6. Publication Tables
results['tables'] = run_script(
    'create_paper_tables.py',
    'Step 6: Generate Publication Tables (CSV, Markdown, LaTeX)'
)

# 7. Key Findings Summary
results['summary'] = run_script(
    'create_key_findings_figure.py',
    'Step 7: Generate Key Findings Summary'
)

# Summary Report
print("\n" + "="*70)
print("ANALYSIS PIPELINE COMPLETE")
print("="*70)

successes = sum(1 for v in results.values() if v)
total = len(results)

print(f"Completed: {successes}/{total} scripts")
print()

for name, success in results.items():
    status = "✓" if success else "✗"
    print(f"  {status} {name}")

if successes == total:
    print("\n" + "="*70)
    print("✓✓✓ ALL ANALYSES SUCCESSFUL ✓✓✓")
    print("="*70)
    print("\nGenerated outputs:")
    print("  - temporal_plots/: Experiment B temporal evolution")
    print("  - chamber_dynamics/: Amplification vs disruption analysis")
    print("  - experiment_d_plots/: Robustness heatmaps")
    print("  - enhanced_analysis/: Cross-experiment plots")
    print("  - paper_tables/: Publication-ready tables")
    print("  - KEY_FINDINGS_SUMMARY.png & .txt: Summary figure and text")
    print("="*70)
else:
    print("\n" + "="*70)
    print("⚠ SOME ANALYSES FAILED - CHECK OUTPUT ABOVE")
    print("="*70)
    sys.exit(1)
