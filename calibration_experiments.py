"""
Calibration Experiments for DisasterAI Memory-Based Belief System

This script runs systematic parameter sweeps to calibrate:
1. D/δ acceptance parameters
2. Memory size
3. Learning rates
4. Trust learning rates

Usage:
    python calibration_experiments.py --quick    # Fast calibration (~30 min)
    python calibration_experiments.py --full     # Full calibration (~2-4 hours)
    python calibration_experiments.py --phase 2  # Run specific phase only
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DisasterAI_Model import DisasterModel, HumanAgent
import os
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

BASELINE_PARAMS = {
    'share_exploitative': 0.5,
    'share_of_disaster': 0.15,
    'initial_trust': 0.3,
    'initial_ai_trust': 0.25,
    'number_of_humans': 50,
    'share_confirming': 0.7,
    'disaster_dynamics': 2,
    'width': 20,
    'height': 20,
    'learning_rate': 0.15,
    'epsilon': 0.3,
    'exploit_trust_lr': 0.03,
    'explor_trust_lr': 0.05,
    'ai_alignment_level': 0.5,
    'ticks': 100,
}

OUTPUT_DIR = 'calibration_results'

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/plots', exist_ok=True)

def calc_acceptance_prob(d, D, delta):
    """Calculate theoretical acceptance probability."""
    if d == 0:
        return 1.0
    d_norm = d / 5.0
    D_norm = D / 5.0
    try:
        if d_norm ** delta > 1e10:  # Prevent overflow
            return 0.0 if d_norm > D_norm else 1.0
        p = (D_norm ** delta) / (d_norm ** delta + D_norm ** delta)
    except (OverflowError, ZeroDivisionError):
        p = 1.0 if d_norm < D_norm else 0.0
    return p

def run_simulation(params, seed=None):
    """Run a single simulation and collect metrics."""
    if seed is not None:
        np.random.seed(seed)
        import random
        random.seed(seed)

    model = DisasterModel(**params)

    # Run simulation
    for _ in range(params['ticks']):
        model.step()

    # Collect metrics
    metrics = {}

    # Separate by agent type
    exploit_agents = [a for a in model.humans.values() if a.agent_type == 'exploitative']
    explore_agents = [a for a in model.humans.values() if a.agent_type == 'exploratory']

    # 1. Memory utilization
    def avg_memory_fullness(agents):
        if not agents:
            return 0
        fullness = []
        for a in agents:
            if hasattr(a, 'belief_memory') and hasattr(a, 'memory_size'):
                cells_with_memory = len(a.belief_memory)
                total_items = sum(len(m) for m in a.belief_memory.values())
                max_items = cells_with_memory * a.memory_size
                if max_items > 0:
                    fullness.append(total_items / max_items)
        return np.mean(fullness) if fullness else 0

    metrics['memory_fullness_exploit'] = avg_memory_fullness(exploit_agents)
    metrics['memory_fullness_explore'] = avg_memory_fullness(explore_agents)

    # 2. Final SECI (last value)
    if model.seci_data:
        metrics['final_seci_exploit'] = model.seci_data[-1][1]
        metrics['final_seci_explore'] = model.seci_data[-1][2]
    else:
        metrics['final_seci_exploit'] = 0
        metrics['final_seci_explore'] = 0

    # 3. Final MAE (belief error)
    if model.belief_error_data:
        metrics['final_mae_exploit'] = model.belief_error_data[-1][1]
        metrics['final_mae_explore'] = model.belief_error_data[-1][2]
    else:
        metrics['final_mae_exploit'] = 0
        metrics['final_mae_explore'] = 0

    # 4. Trust in AI
    def avg_ai_trust(agents):
        if not agents:
            return 0
        trusts = []
        for a in agents:
            ai_trusts = [a.trust.get(f'A_{i}', 0) for i in range(5)]
            trusts.append(np.mean(ai_trusts))
        return np.mean(trusts)

    metrics['ai_trust_exploit'] = avg_ai_trust(exploit_agents)
    metrics['ai_trust_explore'] = avg_ai_trust(explore_agents)

    # 5. Q-value stability (CV in last 50 ticks)
    # This would require tracking Q-values over time, skip for now
    metrics['q_stability'] = 1.0  # Placeholder

    # 6. Acceptance rate approximation (from memory turnover)
    def estimate_acceptance_rate(agents):
        if not agents:
            return 0
        rates = []
        for a in agents:
            if hasattr(a, 'belief_memory'):
                # Count items from external sources (not self-sensing)
                external_items = 0
                for cell_memory in a.belief_memory.values():
                    for item in cell_memory:
                        if item.get('source_id') and item['source_id'] != a.unique_id:
                            external_items += 1
                # Rough estimate based on ticks
                rates.append(external_items / max(1, params['ticks']))
        return np.mean(rates)

    metrics['acceptance_rate_exploit'] = estimate_acceptance_rate(exploit_agents)
    metrics['acceptance_rate_explore'] = estimate_acceptance_rate(explore_agents)

    return metrics

def run_sweep(param_name, param_values, n_reps=3, base_params=None):
    """Run parameter sweep and collect results."""
    if base_params is None:
        base_params = BASELINE_PARAMS.copy()

    results = []
    total_runs = len(param_values) * n_reps
    run_count = 0

    for val in param_values:
        for rep in range(n_reps):
            run_count += 1
            print(f"  [{run_count}/{total_runs}] {param_name}={val}, rep={rep+1}")

            params = base_params.copy()
            params[param_name] = val

            try:
                metrics = run_simulation(params, seed=rep*1000 + hash(str(val)) % 1000)
                metrics['param_name'] = param_name
                metrics['param_value'] = val
                metrics['replication'] = rep
                results.append(metrics)
            except Exception as e:
                print(f"    ERROR: {e}")

    return pd.DataFrame(results)

def run_d_delta_sweep(d_values, delta_values, agent_type, n_reps=3):
    """
    Sweep D and delta for a specific agent type.
    This modifies the HumanAgent class parameters directly.
    """
    results = []
    total_runs = len(d_values) * len(delta_values) * n_reps
    run_count = 0

    for D in d_values:
        for delta in delta_values:
            for rep in range(n_reps):
                run_count += 1
                print(f"  [{run_count}/{total_runs}] D={D}, delta={delta}, rep={rep+1}")

                params = BASELINE_PARAMS.copy()

                try:
                    # Create model
                    np.random.seed(rep * 1000)
                    import random
                    random.seed(rep * 1000)

                    model = DisasterModel(**params)

                    # Override D/delta for target agent type
                    for agent in model.humans.values():
                        if agent.agent_type == agent_type:
                            agent.D = D
                            agent.delta = delta

                    # Run
                    for _ in range(params['ticks']):
                        model.step()

                    # Collect metrics
                    target_agents = [a for a in model.humans.values() if a.agent_type == agent_type]

                    # Calculate acceptance rate from memory
                    external_items = 0
                    total_cells = 0
                    for a in target_agents:
                        if hasattr(a, 'belief_memory'):
                            for cell_memory in a.belief_memory.values():
                                total_cells += 1
                                for item in cell_memory:
                                    if item.get('source_id') and item['source_id'] != a.unique_id:
                                        external_items += 1

                    acceptance_rate = external_items / max(1, total_cells)

                    # Get SECI and MAE
                    idx = 1 if agent_type == 'exploitative' else 2
                    final_seci = model.seci_data[-1][idx] if model.seci_data else 0
                    final_mae = model.belief_error_data[-1][idx] if model.belief_error_data else 0

                    results.append({
                        'D': D,
                        'delta': delta,
                        'agent_type': agent_type,
                        'replication': rep,
                        'acceptance_rate': acceptance_rate,
                        'final_seci': final_seci,
                        'final_mae': final_mae,
                        'theoretical_p_at_2': calc_acceptance_prob(2, D, delta),
                        'theoretical_p_at_3': calc_acceptance_prob(3, D, delta),
                    })

                except Exception as e:
                    print(f"    ERROR: {e}")

    return pd.DataFrame(results)

# ============================================================================
# CALIBRATION PHASES
# ============================================================================

def phase1_baseline_validation():
    """Phase 1: Validate baseline behavior."""
    print("\n" + "="*60)
    print("PHASE 1: BASELINE VALIDATION")
    print("="*60)

    # Test 1: Acceptance formula
    print("\n1. Testing acceptance probability formula...")
    print("   Exploiter (D=1.5, δ=20):")
    for diff in [0, 1, 2, 3]:
        p = calc_acceptance_prob(diff, 1.5, 20)
        print(f"      diff={diff}: P(accept)={p:.4f}")

    print("   Explorer (D=3.0, δ=8):")
    for diff in [0, 1, 2, 3]:
        p = calc_acceptance_prob(diff, 3.0, 8)
        print(f"      diff={diff}: P(accept)={p:.4f}")

    # Test 2: Run extreme conditions
    print("\n2. Testing extreme conditions...")

    # High alignment + all exploiters
    print("   Test A: High alignment (0.9) + 80% exploiters")
    params = BASELINE_PARAMS.copy()
    params['ai_alignment_level'] = 0.9
    params['share_exploitative'] = 0.8
    params['ticks'] = 50
    metrics_a = run_simulation(params, seed=42)
    print(f"      SECI exploit: {metrics_a['final_seci_exploit']:.4f}")
    print(f"      MAE exploit: {metrics_a['final_mae_exploit']:.4f}")

    # Low alignment + all explorers
    print("   Test B: Low alignment (0.1) + 80% explorers")
    params['ai_alignment_level'] = 0.1
    params['share_exploitative'] = 0.2
    metrics_b = run_simulation(params, seed=42)
    print(f"      SECI explore: {metrics_b['final_seci_explore']:.4f}")
    print(f"      MAE explore: {metrics_b['final_mae_explore']:.4f}")

    print("\n   VALIDATION COMPLETE")
    return True

def phase2_d_delta_calibration(quick=False):
    """Phase 2: Calibrate D and delta parameters."""
    print("\n" + "="*60)
    print("PHASE 2: D/δ CALIBRATION")
    print("="*60)

    if quick:
        d_exploit = [1.0, 1.5, 2.0]
        delta_exploit = [15, 20, 25]
        d_explore = [2.5, 3.0, 3.5]
        delta_explore = [6, 8, 10]
        n_reps = 2
    else:
        d_exploit = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
        delta_exploit = [12, 16, 20, 24, 28]
        d_explore = [2.0, 2.5, 3.0, 3.5, 4.0]
        delta_explore = [4, 6, 8, 10, 12]
        n_reps = 3

    # Sweep exploiter D/delta
    print("\n2a. Sweeping exploiter D/δ...")
    exploit_results = run_d_delta_sweep(d_exploit, delta_exploit, 'exploitative', n_reps)

    # Sweep explorer D/delta
    print("\n2b. Sweeping explorer D/δ...")
    explore_results = run_d_delta_sweep(d_explore, delta_explore, 'exploratory', n_reps)

    # Combine and save
    all_results = pd.concat([exploit_results, explore_results], ignore_index=True)
    all_results.to_csv(f'{OUTPUT_DIR}/phase2_d_delta.csv', index=False)

    # Find best values
    print("\n2c. Finding optimal values...")

    # For exploiters: want low acceptance at diff>=2
    exploit_agg = exploit_results.groupby(['D', 'delta']).agg({
        'theoretical_p_at_2': 'mean',
        'final_mae': 'mean',
    }).reset_index()

    # Target: P(accept|diff=2) < 0.05
    exploit_agg['score'] = (0.05 - exploit_agg['theoretical_p_at_2']).clip(lower=0) - exploit_agg['final_mae'] * 0.1
    best_exploit = exploit_agg.loc[exploit_agg['score'].idxmax()]
    print(f"   Best exploiter: D={best_exploit['D']}, δ={best_exploit['delta']}")
    print(f"      P(diff=2)={best_exploit['theoretical_p_at_2']:.4f}, MAE={best_exploit['final_mae']:.4f}")

    # For explorers: want high acceptance at diff=2, moderate at diff=3
    explore_agg = explore_results.groupby(['D', 'delta']).agg({
        'theoretical_p_at_2': 'mean',
        'theoretical_p_at_3': 'mean',
        'final_mae': 'mean',
    }).reset_index()

    # Target: P(accept|diff=2) > 0.8, P(accept|diff=3) ~ 0.5
    explore_agg['score'] = (explore_agg['theoretical_p_at_2'] - 0.8).clip(lower=0) + \
                          (1 - abs(explore_agg['theoretical_p_at_3'] - 0.5)) - \
                          explore_agg['final_mae'] * 0.1
    best_explore = explore_agg.loc[explore_agg['score'].idxmax()]
    print(f"   Best explorer: D={best_explore['D']}, δ={best_explore['delta']}")
    print(f"      P(diff=2)={best_explore['theoretical_p_at_2']:.4f}, P(diff=3)={best_explore['theoretical_p_at_3']:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Exploiter heatmap
    pivot_exploit = exploit_agg.pivot(index='D', columns='delta', values='theoretical_p_at_2')
    im1 = axes[0].imshow(pivot_exploit.values, cmap='RdYlGn_r', aspect='auto')
    axes[0].set_xticks(range(len(pivot_exploit.columns)))
    axes[0].set_xticklabels(pivot_exploit.columns)
    axes[0].set_yticks(range(len(pivot_exploit.index)))
    axes[0].set_yticklabels([f'{x:.2f}' for x in pivot_exploit.index])
    axes[0].set_xlabel('δ')
    axes[0].set_ylabel('D')
    axes[0].set_title('Exploiter P(accept|diff=2)\nLower is better')
    plt.colorbar(im1, ax=axes[0])

    # Explorer heatmap
    pivot_explore = explore_agg.pivot(index='D', columns='delta', values='theoretical_p_at_2')
    im2 = axes[1].imshow(pivot_explore.values, cmap='RdYlGn', aspect='auto')
    axes[1].set_xticks(range(len(pivot_explore.columns)))
    axes[1].set_xticklabels(pivot_explore.columns)
    axes[1].set_yticks(range(len(pivot_explore.index)))
    axes[1].set_yticklabels([f'{x:.2f}' for x in pivot_explore.index])
    axes[1].set_xlabel('δ')
    axes[1].set_ylabel('D')
    axes[1].set_title('Explorer P(accept|diff=2)\nHigher is better')
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/plots/phase2_d_delta_heatmap.png', dpi=150)
    plt.close()

    return {
        'exploit_D': best_exploit['D'],
        'exploit_delta': best_exploit['delta'],
        'explore_D': best_explore['D'],
        'explore_delta': best_explore['delta'],
    }

def phase3_memory_calibration(quick=False):
    """Phase 3: Calibrate memory size."""
    print("\n" + "="*60)
    print("PHASE 3: MEMORY SIZE CALIBRATION")
    print("="*60)

    if quick:
        memory_exploit = [2, 3, 4]
        memory_explore = [5, 7, 9]
        n_reps = 2
    else:
        memory_exploit = [2, 3, 4, 5]
        memory_explore = [5, 6, 7, 8, 9, 10]
        n_reps = 3

    results = []

    # Test different memory sizes
    for mem_exp in memory_exploit:
        for mem_expl in memory_explore:
            print(f"  Testing memory_exploit={mem_exp}, memory_explore={mem_expl}")

            for rep in range(n_reps):
                params = BASELINE_PARAMS.copy()

                np.random.seed(rep * 1000)
                import random
                random.seed(rep * 1000)

                model = DisasterModel(**params)

                # Override memory sizes
                for agent in model.humans.values():
                    if agent.agent_type == 'exploitative':
                        agent.memory_size = mem_exp
                    else:
                        agent.memory_size = mem_expl

                # Run
                for _ in range(params['ticks']):
                    model.step()

                # Collect metrics
                idx_exp = 1
                idx_expl = 2

                results.append({
                    'memory_exploit': mem_exp,
                    'memory_explore': mem_expl,
                    'replication': rep,
                    'final_seci_exploit': model.seci_data[-1][idx_exp] if model.seci_data else 0,
                    'final_seci_explore': model.seci_data[-1][idx_expl] if model.seci_data else 0,
                    'final_mae_exploit': model.belief_error_data[-1][idx_exp] if model.belief_error_data else 0,
                    'final_mae_explore': model.belief_error_data[-1][idx_expl] if model.belief_error_data else 0,
                })

    df = pd.DataFrame(results)
    df.to_csv(f'{OUTPUT_DIR}/phase3_memory.csv', index=False)

    # Find best combination
    agg = df.groupby(['memory_exploit', 'memory_explore']).agg({
        'final_mae_exploit': 'mean',
        'final_mae_explore': 'mean',
        'final_seci_exploit': 'mean',
        'final_seci_explore': 'mean',
    }).reset_index()

    # Score: low MAE, distinct SECI between types
    agg['score'] = -(agg['final_mae_exploit'] + agg['final_mae_explore']) + \
                   abs(agg['final_seci_exploit'] - agg['final_seci_explore'])

    best = agg.loc[agg['score'].idxmax()]
    print(f"\n   Best memory sizes: exploit={int(best['memory_exploit'])}, explore={int(best['memory_explore'])}")

    return {
        'memory_exploit': int(best['memory_exploit']),
        'memory_explore': int(best['memory_explore']),
    }

def phase4_learning_rate_calibration(quick=False):
    """Phase 4: Calibrate learning rates."""
    print("\n" + "="*60)
    print("PHASE 4: LEARNING RATE CALIBRATION")
    print("="*60)

    if quick:
        learning_rates = [0.08, 0.15, 0.22]
        epsilons = [0.2, 0.3, 0.4]
        n_reps = 2
    else:
        learning_rates = [0.05, 0.10, 0.15, 0.20, 0.25]
        epsilons = [0.15, 0.25, 0.35, 0.45]
        n_reps = 3

    results = []

    for lr in learning_rates:
        for eps in epsilons:
            print(f"  Testing learning_rate={lr}, epsilon={eps}")

            for rep in range(n_reps):
                params = BASELINE_PARAMS.copy()
                params['learning_rate'] = lr
                params['epsilon'] = eps

                try:
                    metrics = run_simulation(params, seed=rep*1000)
                    metrics['learning_rate'] = lr
                    metrics['epsilon'] = eps
                    metrics['replication'] = rep
                    results.append(metrics)
                except Exception as e:
                    print(f"    ERROR: {e}")

    df = pd.DataFrame(results)
    df.to_csv(f'{OUTPUT_DIR}/phase4_learning.csv', index=False)

    # Find best
    agg = df.groupby(['learning_rate', 'epsilon']).agg({
        'final_mae_exploit': 'mean',
        'final_mae_explore': 'mean',
    }).reset_index()

    agg['total_mae'] = agg['final_mae_exploit'] + agg['final_mae_explore']
    best = agg.loc[agg['total_mae'].idxmin()]

    print(f"\n   Best: learning_rate={best['learning_rate']}, epsilon={best['epsilon']}")

    return {
        'learning_rate': best['learning_rate'],
        'epsilon': best['epsilon'],
    }

def phase5_trust_calibration(quick=False):
    """Phase 5: Calibrate trust learning rates."""
    print("\n" + "="*60)
    print("PHASE 5: TRUST LEARNING RATE CALIBRATION")
    print("="*60)

    if quick:
        exploit_lrs = [0.02, 0.03, 0.05]
        explore_lrs = [0.04, 0.06, 0.08]
        n_reps = 2
    else:
        exploit_lrs = [0.01, 0.02, 0.03, 0.04, 0.05]
        explore_lrs = [0.03, 0.05, 0.07, 0.09, 0.11]
        n_reps = 3

    results = []

    for exp_lr in exploit_lrs:
        for expl_lr in explore_lrs:
            # Constraint: explorer should be faster
            if expl_lr < exp_lr * 1.3:
                continue

            print(f"  Testing exploit_lr={exp_lr}, explore_lr={expl_lr}")

            for rep in range(n_reps):
                params = BASELINE_PARAMS.copy()
                params['exploit_trust_lr'] = exp_lr
                params['explor_trust_lr'] = expl_lr

                try:
                    metrics = run_simulation(params, seed=rep*1000)
                    metrics['exploit_trust_lr'] = exp_lr
                    metrics['explor_trust_lr'] = expl_lr
                    metrics['replication'] = rep
                    results.append(metrics)
                except Exception as e:
                    print(f"    ERROR: {e}")

    df = pd.DataFrame(results)
    df.to_csv(f'{OUTPUT_DIR}/phase5_trust.csv', index=False)

    # Find best (maximize trust differentiation while minimizing MAE)
    agg = df.groupby(['exploit_trust_lr', 'explor_trust_lr']).agg({
        'ai_trust_exploit': 'mean',
        'ai_trust_explore': 'mean',
        'final_mae_exploit': 'mean',
        'final_mae_explore': 'mean',
    }).reset_index()

    agg['trust_diff'] = abs(agg['ai_trust_exploit'] - agg['ai_trust_explore'])
    agg['total_mae'] = agg['final_mae_exploit'] + agg['final_mae_explore']
    agg['score'] = agg['trust_diff'] - agg['total_mae'] * 0.5

    best = agg.loc[agg['score'].idxmax()]

    print(f"\n   Best: exploit_trust_lr={best['exploit_trust_lr']}, explor_trust_lr={best['explor_trust_lr']}")

    return {
        'exploit_trust_lr': best['exploit_trust_lr'],
        'explor_trust_lr': best['explor_trust_lr'],
    }

def phase6_validation(calibrated_params, n_reps=5):
    """Phase 6: Validate calibrated parameters."""
    print("\n" + "="*60)
    print("PHASE 6: VALIDATION")
    print("="*60)

    # Full factorial on research variables
    alignments = [0.1, 0.5, 0.9]
    shares = [0.3, 0.5, 0.7]

    results = []

    for align in alignments:
        for share in shares:
            print(f"  Testing alignment={align}, share_exploit={share}")

            for rep in range(n_reps):
                params = BASELINE_PARAMS.copy()
                params.update(calibrated_params)
                params['ai_alignment_level'] = align
                params['share_exploitative'] = share

                try:
                    metrics = run_simulation(params, seed=rep*1000)
                    metrics['ai_alignment_level'] = align
                    metrics['share_exploitative'] = share
                    metrics['replication'] = rep
                    results.append(metrics)
                except Exception as e:
                    print(f"    ERROR: {e}")

    df = pd.DataFrame(results)
    df.to_csv(f'{OUTPUT_DIR}/phase6_validation.csv', index=False)

    # Summarize
    print("\n   VALIDATION SUMMARY:")
    for align in alignments:
        subset = df[df['ai_alignment_level'] == align]
        print(f"\n   Alignment = {align}:")
        print(f"      SECI exploit: {subset['final_seci_exploit'].mean():.4f} ± {subset['final_seci_exploit'].std():.4f}")
        print(f"      SECI explore: {subset['final_seci_explore'].mean():.4f} ± {subset['final_seci_explore'].std():.4f}")
        print(f"      MAE exploit:  {subset['final_mae_exploit'].mean():.4f} ± {subset['final_mae_exploit'].std():.4f}")
        print(f"      MAE explore:  {subset['final_mae_explore'].mean():.4f} ± {subset['final_mae_explore'].std():.4f}")

    return df

# ============================================================================
# MAIN
# ============================================================================

def write_summary(calibrated_params):
    """Write calibration summary to file."""
    with open(f'{OUTPUT_DIR}/calibration_summary.md', 'w') as f:
        f.write("# Calibration Results Summary\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Calibrated Parameters\n\n")
        f.write("| Parameter | Value |\n")
        f.write("|-----------|-------|\n")
        for key, val in calibrated_params.items():
            f.write(f"| {key} | {val} |\n")
        f.write("\n## Usage\n\n")
        f.write("```python\n")
        f.write("params = {\n")
        for key, val in calibrated_params.items():
            f.write(f"    '{key}': {val},\n")
        f.write("}\n")
        f.write("```\n")

def main():
    parser = argparse.ArgumentParser(description='DisasterAI Calibration')
    parser.add_argument('--quick', action='store_true', help='Quick calibration mode')
    parser.add_argument('--full', action='store_true', help='Full calibration mode')
    parser.add_argument('--phase', type=int, help='Run specific phase only (1-6)')
    args = parser.parse_args()

    quick = args.quick or not args.full

    ensure_output_dir()

    print("="*60)
    print("DisasterAI CALIBRATION EXPERIMENTS")
    print(f"Mode: {'QUICK' if quick else 'FULL'}")
    print("="*60)

    calibrated = {}

    if args.phase is None or args.phase == 1:
        phase1_baseline_validation()

    if args.phase is None or args.phase == 2:
        result = phase2_d_delta_calibration(quick)
        calibrated.update(result)

    if args.phase is None or args.phase == 3:
        result = phase3_memory_calibration(quick)
        calibrated.update(result)

    if args.phase is None or args.phase == 4:
        result = phase4_learning_rate_calibration(quick)
        calibrated.update(result)

    if args.phase is None or args.phase == 5:
        result = phase5_trust_calibration(quick)
        calibrated.update(result)

    if args.phase is None or args.phase == 6:
        phase6_validation(calibrated, n_reps=3 if quick else 5)

    # Write summary
    if calibrated:
        write_summary(calibrated)
        print("\n" + "="*60)
        print("CALIBRATION COMPLETE")
        print("="*60)
        print("\nCalibrated parameters:")
        for key, val in calibrated.items():
            print(f"  {key}: {val}")
        print(f"\nResults saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
