#!/usr/bin/env python3
"""
Diagnostic script to understand SECI behavior and sensitivity to alignment.
"""

import numpy as np
import pickle
import os

def diagnose_seci_sensitivity():
    """Examine why SECI shows little variation despite large AECI changes."""

    # Load results
    results_file = None
    for search_dir in ["agent_model_results", "/content/drive/MyDrive/DisasterAI_results"]:
        test_path = os.path.join(search_dir, "results_experiment_B.pkl")
        if os.path.exists(test_path):
            results_file = test_path
            break

    if not results_file:
        print("❌ Could not find results file")
        return

    print(f"📂 Loading: {results_file}\n")

    with open(results_file, 'rb') as f:
        results_b = pickle.load(f)

    alignment_values = sorted(results_b.keys())

    print("="*70)
    print("SECI vs AECI SENSITIVITY ANALYSIS")
    print("="*70)

    for alignment in alignment_values:
        res = results_b[alignment]

        seci = res.get("seci", np.array([]))
        aeci = res.get("aeci", np.array([]))

        if seci.ndim != 3 or aeci.ndim != 3:
            print(f"\nAlignment {alignment}: Invalid data shape")
            continue

        print(f"\n{'='*70}")
        print(f"ALIGNMENT = {alignment}")
        print(f"{'='*70}")

        # Extract time series
        seci_exploit = seci[:, :, 1]  # (runs, ticks)
        seci_explor = seci[:, :, 2]
        aeci_exploit = aeci[:, :, 1]
        aeci_explor = aeci[:, :, 2]

        # Calculate statistics across full simulation
        print("\n📊 FULL SIMULATION STATISTICS:")
        print("\nSECI (Social Echo Chamber - based on friend network):")
        print(f"  Exploit agents: mean={np.mean(seci_exploit):.4f}, std={np.std(seci_exploit):.4f}")
        print(f"                  min={np.min(seci_exploit):.4f}, max={np.max(seci_exploit):.4f}")
        print(f"  Explor agents:  mean={np.mean(seci_explor):.4f}, std={np.std(seci_explor):.4f}")
        print(f"                  min={np.min(seci_explor):.4f}, max={np.max(seci_explor):.4f}")

        print("\nAECI (AI Query Ratio - source preference):")
        print(f"  Exploit agents: mean={np.mean(aeci_exploit):.4f}, std={np.std(aeci_exploit):.4f}")
        print(f"                  min={np.min(aeci_exploit):.4f}, max={np.max(aeci_exploit):.4f}")
        print(f"  Explor agents:  mean={np.mean(aeci_explor):.4f}, std={np.std(aeci_explor):.4f}")
        print(f"                  min={np.min(aeci_explor):.4f}, max={np.max(aeci_explor):.4f}")

        # Peak analysis (when echo chambers are strongest)
        print("\n📈 PEAK ECHO CHAMBER STRENGTH:")
        peak_seci_exploit = np.min(seci_exploit, axis=1)  # Most negative per run
        peak_seci_explor = np.min(seci_explor, axis=1)

        print(f"  Exploit peak SECI: mean={np.mean(peak_seci_exploit):.4f} ± {np.std(peak_seci_exploit):.4f}")
        print(f"  Explor peak SECI:  mean={np.mean(peak_seci_explor):.4f} ± {np.std(peak_seci_explor):.4f}")

        # Time evolution
        mean_seci_over_time = np.mean(seci_exploit, axis=0)
        mean_aeci_over_time = np.mean(aeci_exploit, axis=0)

        print("\n⏱️  TEMPORAL EVOLUTION (Exploit agents):")
        print(f"  Tick 0-30:    SECI={np.mean(mean_seci_over_time[:30]):.4f}, AECI={np.mean(mean_aeci_over_time[:30]):.4f}")
        print(f"  Tick 30-60:   SECI={np.mean(mean_seci_over_time[30:60]):.4f}, AECI={np.mean(mean_aeci_over_time[30:60]):.4f}")
        print(f"  Tick 60-90:   SECI={np.mean(mean_seci_over_time[60:90]):.4f}, AECI={np.mean(mean_aeci_over_time[60:90]):.4f}")
        print(f"  Tick 90-120:  SECI={np.mean(mean_seci_over_time[90:120]):.4f}, AECI={np.mean(mean_aeci_over_time[90:120]):.4f}")
        print(f"  Tick 120-150: SECI={np.mean(mean_seci_over_time[120:]):.4f}, AECI={np.mean(mean_aeci_over_time[120:]):.4f}")

    # Cross-alignment comparison
    print("\n" + "="*70)
    print("CROSS-ALIGNMENT COMPARISON")
    print("="*70)

    seci_by_alignment = []
    aeci_by_alignment = []

    for alignment in alignment_values:
        res = results_b[alignment]
        seci = res.get("seci", np.array([]))
        aeci = res.get("aeci", np.array([]))

        if seci.ndim == 3:
            # Peak values (most negative)
            peak_seci = np.min(seci[:, :, 1])  # Exploit agents
            mean_aeci = np.mean(aeci[:, :, 1])  # Exploit agents

            seci_by_alignment.append(peak_seci)
            aeci_by_alignment.append(mean_aeci)

    print("\n📊 Peak SECI vs Mean AECI (Exploit agents):")
    print("Alignment | Peak SECI | Mean AECI | SECI Range")
    print("-" * 60)
    for i, alignment in enumerate(alignment_values):
        if i < len(seci_by_alignment):
            print(f"  {alignment:5.2f}   | {seci_by_alignment[i]:9.4f} | {aeci_by_alignment[i]:9.4f} |")

    # Calculate variation
    seci_variation = np.max(seci_by_alignment) - np.min(seci_by_alignment)
    aeci_variation = np.max(aeci_by_alignment) - np.min(aeci_by_alignment)

    print(f"\n📉 VARIATION ACROSS ALIGNMENTS:")
    print(f"  SECI range: {seci_variation:.4f}  (peak values)")
    print(f"  AECI range: {aeci_variation:.4f}  (mean values)")
    print(f"  AECI/SECI variation ratio: {aeci_variation / max(0.0001, abs(seci_variation)):.2f}x")

    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    print("""
1. SECI measures SOCIAL NETWORK HOMOPHILY
   - Based on friend network structure (who you're connected to)
   - Compares belief variance among friends vs global population
   - Negative = friends have more similar beliefs (echo chamber)

2. AECI measures INFORMATION SOURCE PREFERENCE
   - Based on querying behavior (who you ask)
   - Ratio of AI queries to total queries
   - High = prefer AI over friends for information

3. WHY MIGHT SECI NOT VARY WITH ALIGNMENT?
   - Social network is FIXED at initialization (doesn't rewire based on AI use)
   - Alignment affects AI quality → changes AECI (who you query)
   - But doesn't change friend network structure → SECI stays similar
   - You can query AI a lot but still have diverse friends!

4. THRESHOLD QUESTION:
   - The -0.1 SECI threshold means "10% variance reduction vs global"
   - This is somewhat arbitrary
   - Better approach: Use data-driven thresholds (e.g., 1 std dev from zero)
   - Or focus on MAGNITUDE of peak rather than binary threshold
    """)

if __name__ == "__main__":
    diagnose_seci_sensitivity()
