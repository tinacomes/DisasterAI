"""
Paper Tables Generator
======================

Creates publication-ready summary tables from experiment results.
Outputs:
- CSV files for easy import to Word/Excel
- LaTeX tables for direct paper inclusion
- Markdown tables for quick review

Run after experiments complete.
"""

import os
import pickle
import numpy as np
import pandas as pd

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

OUTPUT_DIR = os.path.join(RESULTS_DIR, "paper_tables")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"✓ Loading results from: {RESULTS_DIR}")
print(f"✓ Saving tables to: {OUTPUT_DIR}")

#########################################
# Load Results
#########################################

def load_experiment(exp_name):
    """Load experiment results."""
    filepath = os.path.join(RESULTS_DIR, f"results_experiment_{exp_name}.pkl")
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None

results_a = load_experiment('A')
results_b = load_experiment('B')
results_d = load_experiment('D')

#########################################
# Table 1: Main Effects Summary
#########################################

def create_main_effects_table():
    """Create summary table of main experimental effects."""

    rows = []

    # Experiment A
    if results_a:
        for share_val, res in sorted(results_a.items()):
            if isinstance(res, dict):
                # Extract final metrics
                seci_data = res.get('seci', np.array([]))
                aeci_data = res.get('aeci_variance', np.array([]))

                row = {'Experiment': 'A', 'Parameter': f'Share Exploit={share_val:.1f}'}

                if seci_data.size > 0 and seci_data.ndim >= 3:
                    row['SECI (Exploit)'] = f"{np.mean(seci_data[:, -1, 1]):.3f} ± {np.std(seci_data[:, -1, 1]):.3f}"
                    row['SECI (Explor)'] = f"{np.mean(seci_data[:, -1, 2]):.3f} ± {np.std(seci_data[:, -1, 2]):.3f}"

                if isinstance(aeci_data, np.ndarray) and aeci_data.size > 0:
                    try:
                        final_aeci = np.mean(aeci_data[:, -1, 1])
                        std_aeci = np.std(aeci_data[:, -1, 1])
                        row['AECI-Var'] = f"{final_aeci:.3f} ± {std_aeci:.3f}"
                    except:
                        pass

                # Performance
                correct = res.get('correct_tokens', {})
                incorrect = res.get('incorrect_tokens', {})
                if 'exploitative' in correct:
                    total = correct['exploitative'] + incorrect.get('exploitative', 0)
                    if total > 0:
                        row['Accuracy'] = f"{correct['exploitative']/total:.3f}"

                rows.append(row)

    # Experiment B
    if results_b:
        for align_val, res in sorted(results_b.items()):
            if isinstance(res, dict):
                seci_data = res.get('seci', np.array([]))
                aeci_data = res.get('aeci_variance', np.array([]))

                row = {'Experiment': 'B', 'Parameter': f'AI Align={align_val:.2f}'}

                if seci_data.size > 0 and seci_data.ndim >= 3:
                    row['SECI (Exploit)'] = f"{np.mean(seci_data[:, -1, 1]):.3f} ± {np.std(seci_data[:, -1, 1]):.3f}"
                    row['SECI (Explor)'] = f"{np.mean(seci_data[:, -1, 2]):.3f} ± {np.std(seci_data[:, -1, 2]):.3f}"

                if isinstance(aeci_data, np.ndarray) and aeci_data.size > 0:
                    try:
                        final_aeci = np.mean(aeci_data[:, -1, 1])
                        std_aeci = np.std(aeci_data[:, -1, 1])
                        row['AECI-Var'] = f"{final_aeci:.3f} ± {std_aeci:.3f}"
                    except:
                        pass

                rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Save in multiple formats
    df.to_csv(os.path.join(OUTPUT_DIR, "table1_main_effects.csv"), index=False)
    df.to_markdown(os.path.join(OUTPUT_DIR, "table1_main_effects.md"), index=False)
    df.to_latex(os.path.join(OUTPUT_DIR, "table1_main_effects.tex"), index=False)

    print("✓ Table 1: Main Effects")
    print(df.to_string())
    return df

#########################################
# Table 2: Robustness Check (Exp D)
#########################################

def create_robustness_table():
    """Create robustness analysis table for Experiment D."""

    if not results_d:
        print("⚠ Experiment D results not found")
        return None

    rows = []

    for (lr, eps), res in sorted(results_d.items()):
        if isinstance(res, dict):
            seci_data = res.get('seci', np.array([]))
            aeci_data = res.get('aeci_variance', np.array([]))

            row = {
                'Learning Rate': lr,
                'Epsilon': eps
            }

            if seci_data.size > 0 and seci_data.ndim >= 3:
                row['SECI (Final)'] = f"{np.mean(seci_data[:, -1, 1]):.3f}"
                row['SECI (Std)'] = f"{np.std(seci_data[:, -1, 1]):.3f}"

            if isinstance(aeci_data, np.ndarray) and aeci_data.size > 0:
                try:
                    row['AECI (Final)'] = f"{np.mean(aeci_data[:, -1, 1]):.3f}"
                    row['AECI (Std)'] = f"{np.std(aeci_data[:, -1, 1]):.3f}"
                except:
                    pass

            rows.append(row)

    df = pd.DataFrame(rows)

    # Calculate variance across parameters
    if len(df) > 0:
        print("\n" + "="*60)
        print("ROBUSTNESS SUMMARY")
        print("="*60)
        print(f"SECI variance across parameters: {df['SECI (Final)'].astype(float).var():.4f}")
        print(f"AECI variance across parameters: {df['AECI (Final)'].astype(float).var():.4f}")
        print("Low variance = Robust to parameter changes")

    # Save
    df.to_csv(os.path.join(OUTPUT_DIR, "table2_robustness.csv"), index=False)
    df.to_markdown(os.path.join(OUTPUT_DIR, "table2_robustness.md"), index=False)
    df.to_latex(os.path.join(OUTPUT_DIR, "table2_robustness.tex"), index=False)

    print("\n✓ Table 2: Robustness Analysis")
    print(df.to_string())
    return df

#########################################
# Table 3: Agent Type Comparison
#########################################

def create_agent_comparison_table():
    """Compare exploitative vs exploratory agents across experiments."""

    rows = []

    # From Experiment A
    if results_a:
        for share_val, res in sorted(results_a.items()):
            if isinstance(res, dict):
                seci_data = res.get('seci', np.array([]))

                if seci_data.size > 0 and seci_data.ndim >= 3:
                    row = {
                        'Condition': f'Exp A (Share={share_val:.1f})',
                        'SECI Exploit': f"{np.mean(seci_data[:, -1, 1]):.3f}",
                        'SECI Explor': f"{np.mean(seci_data[:, -1, 2]):.3f}",
                        'Difference': f"{np.mean(seci_data[:, -1, 1]) - np.mean(seci_data[:, -1, 2]):.3f}"
                    }
                    rows.append(row)

    # From Experiment B (select key values)
    if results_b:
        key_alignments = [0.0, 0.5, 1.0]
        for align_val in key_alignments:
            if align_val in results_b:
                res = results_b[align_val]
                seci_data = res.get('seci', np.array([]))

                if seci_data.size > 0 and seci_data.ndim >= 3:
                    row = {
                        'Condition': f'Exp B (Align={align_val:.1f})',
                        'SECI Exploit': f"{np.mean(seci_data[:, -1, 1]):.3f}",
                        'SECI Explor': f"{np.mean(seci_data[:, -1, 2]):.3f}",
                        'Difference': f"{np.mean(seci_data[:, -1, 1]) - np.mean(seci_data[:, -1, 2]):.3f}"
                    }
                    rows.append(row)

    df = pd.DataFrame(rows)

    # Save
    df.to_csv(os.path.join(OUTPUT_DIR, "table3_agent_comparison.csv"), index=False)
    df.to_markdown(os.path.join(OUTPUT_DIR, "table3_agent_comparison.md"), index=False)
    df.to_latex(os.path.join(OUTPUT_DIR, "table3_agent_comparison.tex"), index=False)

    print("\n✓ Table 3: Agent Type Comparison (RQ4)")
    print(df.to_string())
    return df

#########################################
# Main Execution
#########################################

if __name__ == "__main__":
    print("\n" + "="*60)
    print("GENERATING PAPER TABLES")
    print("="*60)

    # Check what results we have
    print("\nAvailable results:")
    print(f"  Experiment A: {'✓' if results_a else '✗'}")
    print(f"  Experiment B: {'✓' if results_b else '✗'}")
    print(f"  Experiment D: {'✓' if results_d else '✗'}")
    print()

    # Generate tables
    table1 = create_main_effects_table()
    print("\n" + "="*60 + "\n")

    table2 = create_robustness_table()
    print("\n" + "="*60 + "\n")

    table3 = create_agent_comparison_table()

    print("\n" + "="*60)
    print("✓ ALL TABLES GENERATED")
    print("="*60)
    print(f"Tables saved to: {OUTPUT_DIR}")
    print("\nFiles created:")
    print("  - table1_main_effects.csv/.md/.tex")
    print("  - table2_robustness.csv/.md/.tex")
    print("  - table3_agent_comparison.csv/.md/.tex")
    print("\nReady to copy into your paper!")
    print("="*60)
