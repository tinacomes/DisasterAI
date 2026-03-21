"""
Goldilocks AI Alignment Experiment: Social vs. AI Filter Bubble Interplay

Research Question:
What is the optimal AI alignment level α* that breaks social echo chambers
without creating AI-enforced filter bubbles?

Background — Triple Social Bubbles:
1. Ego-network bubble: agents share/accept info primarily from friends
2. Agent-type cluster: exploiters cluster with exploiters, explorers with explorers
3. Shared-observation bubble: agents near same cells converge via direct sensing

AI Alignment Interplay:
- High alignment (AI confirms user beliefs): AI penetrates networks but amplifies all three bubbles
- Low alignment (AI reports truth): AI may be rejected (D/delta mechanism rejects divergent info)
- Goldilocks α*: AI is trusted and used, but divergent enough to disrupt bubble consensus

Metrics:
- SECI (Social Echo Chamber Index): -1 to +1
  * -1 = Strong echo chamber (friends very similar)
  *  0 = No echo chamber effect
  * +1 = Anti-echo chamber (friends more diverse)
- AECI (AI Echo Chamber Index): -1 to +1  (same variance formula as SECI)
  * -1 = AI-reliant agents have much lower belief variance than global (AI bubble)
  *  0 = No AI echo chamber effect
  * +1 = AI-reliant agents are more belief-diverse than global
- total_bubble = |SECI| + |AECI|  (minimise both)
- Belief MAE: accuracy cost at each alignment level
- Unmet needs: high-need cells (≥L4) that received zero relief tokens per tick
- Targeting precision: fraction of relief sent to genuinely high-need cells

Goldilocks detection: argmin of total_bubble across alignment sweep

Run locally (3 replications, ~15 min):
    python3 test_filter_bubbles.py

For large-N CI runs use simulate.py + plot_results.py instead.
"""

import numpy as np
import matplotlib.pyplot as plt
from DisasterAI_Model import DisasterModel, HumanAgent
import os

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

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
    'ticks': 100,
    'learning_rate': 0.1,
    'epsilon': 0.3,
    'exploit_trust_lr': 0.015,
    'explor_trust_lr': 0.03,
}

ALIGNMENT_SWEEP    = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
STEADY_STATE_WINDOW = 15

N_RUNS        = 3    # replications for primary alignment sweep
N_FACTOR_RUNS = 2    # replications for factor sweeps

FACTOR_ALPHA       = 0.5
RUMOR_SWEEP        = [0.0, 0.5, 1.0]
DISASTER_SWEEP     = [0, 2, 3]
EXPLOITATIVE_SWEEP = [0.2, 0.5, 0.8]


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_one_sim(params):
    """Run a single simulation and return per-tick metrics dict."""
    model = DisasterModel(**params)

    seci_exploit, seci_explor = [], []
    aeci_exploit, aeci_explor = [], []
    mae_exploit,  mae_explor  = [], []
    prec_exploit, prec_explor = [], []
    metric_ticks = []

    for tick in range(params['ticks']):
        model.step()

        if model.seci_data:
            s = model.seci_data[-1]
            seci_exploit.append(float(s[1]))
            seci_explor.append(float(s[2]))

        if model.aeci_data:
            a = model.aeci_data[-1]
            aeci_exploit.append(float(a[1]))
            aeci_explor.append(float(a[2]))

        if tick % 5 == 0:
            ex_errors, er_errors = [], []
            ex_correct = ex_total = er_correct = er_total = 0
            for agent in model.agent_list:
                if not isinstance(agent, HumanAgent):
                    continue
                err = float(np.mean([
                    abs(b.get('level', 0) - model.disaster_grid[c])
                    for c, b in agent.beliefs.items()
                    if isinstance(b, dict)
                ])) if agent.beliefs else 0.0
                total = agent.correct_targets + agent.incorrect_targets
                if agent.agent_type == 'exploitative':
                    ex_errors.append(err)
                    ex_correct += agent.correct_targets
                    ex_total   += total
                else:
                    er_errors.append(err)
                    er_correct += agent.correct_targets
                    er_total   += total
            mae_exploit.append(float(np.mean(ex_errors)) if ex_errors else 0.0)
            mae_explor.append( float(np.mean(er_errors)) if er_errors else 0.0)
            prec_exploit.append(ex_correct / ex_total if ex_total > 0 else float('nan'))
            prec_explor.append( er_correct / er_total if er_total > 0 else float('nan'))
            metric_ticks.append(tick)

    return {
        'seci_exploit': seci_exploit,
        'seci_explor':  seci_explor,
        'aeci_exploit': aeci_exploit,
        'aeci_explor':  aeci_explor,
        'mae_exploit':  mae_exploit,
        'mae_explor':   mae_explor,
        'prec_exploit': prec_exploit,
        'prec_explor':  prec_explor,
        'unmet_needs':  [float(v) for v in model.unmet_needs_evolution],
        'metric_ticks': metric_ticks,
    }


def _aggregate(runs):
    """Compute mean and std across replications for all metrics."""
    keys = ['seci_exploit', 'seci_explor', 'aeci_exploit', 'aeci_explor',
            'mae_exploit',  'mae_explor',  'prec_exploit', 'prec_explor',
            'unmet_needs']
    result = {
        'metric_ticks': runs[0]['metric_ticks'],
        'n_ticks': len(runs[0]['seci_exploit']),
    }
    for key in keys:
        arrays = []
        for run in runs:
            arrays.append([
                float('nan') if (v is None or (isinstance(v, float) and np.isnan(v))) else v
                for v in run[key]
            ])
        min_len = min(len(a) for a in arrays)
        mat = np.array([a[:min_len] for a in arrays], dtype=float)
        result[f'{key}_mean'] = np.nanmean(mat, axis=0).tolist()
        result[f'{key}_std']  = np.nanstd( mat, axis=0).tolist()
    return result


def run_replicated(params, n_runs, label=''):
    """Run n_runs independent simulations and return aggregated mean/std dict."""
    print(f"\n{'='*60}")
    print(f"Running: {label}  ({n_runs} replication{'s' if n_runs > 1 else ''})")
    print(f"{'='*60}")
    runs = []
    for i in range(n_runs):
        print(f"  Replicate {i+1}/{n_runs}...")
        runs.append(run_one_sim(params))
    return _aggregate(runs)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def ss(series, window=STEADY_STATE_WINDOW):
    """Steady-state mean: mean of last `window` values, NaN-safe."""
    if not series:
        return float('nan')
    return float(np.nanmean(series[-window:]))


def compute_goldilocks_metrics(all_results):
    """Compute steady-state scalar metrics (mean ± std) from replicated results."""
    metrics = {}
    for alpha, res in zip(ALIGNMENT_SWEEP, all_results):
        def ms(key_e, key_r=None):
            key_r = key_r or key_e.replace('exploit', 'explor')
            m = (ss(res[f'{key_e}_mean']) + ss(res[f'{key_r}_mean'])) / 2
            s = (ss(res[f'{key_e}_std'])  + ss(res[f'{key_r}_std']))  / 2
            return m, s

        seci_m, seci_s = ms('seci_exploit', 'seci_explor')
        aeci_m, aeci_s = ms('aeci_exploit', 'aeci_explor')
        mae_m,  mae_s  = ms('mae_exploit',  'mae_explor')
        prec_m, prec_s = ms('prec_exploit', 'prec_explor')
        unmet_m = ss(res['unmet_needs_mean'])
        unmet_s = ss(res['unmet_needs_std'])

        metrics[alpha] = {
            'seci': seci_m, 'seci_std': seci_s,
            'aeci': aeci_m, 'aeci_std': aeci_s,
            'mae':  mae_m,  'mae_std':  mae_s,
            'prec': prec_m, 'prec_std': prec_s,
            'unmet': unmet_m, 'unmet_std': unmet_s,
            'total_bubble': abs(seci_m) + abs(aeci_m),
        }
    return metrics


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_goldilocks(metrics, all_results, save_dir):
    """2×3 Goldilocks summary with mean ± std error bars."""
    alphas     = ALIGNMENT_SWEEP
    total_vals = [metrics[a]['total_bubble'] for a in alphas]
    best_alpha = alphas[int(np.argmin(total_vals))]
    print(f"\nGoldilocks α* = {best_alpha}  (min total_bubble = {min(total_vals):.3f})")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f'Goldilocks AI Alignment (α*={best_alpha}) — mean ± std, N={N_RUNS} replications',
        fontsize=13, fontweight='bold'
    )

    def eb(ax, key, color, ylabel, title, ylim=None):
        means = [metrics[a][key] for a in alphas]
        stds  = [metrics[a][f'{key}_std'] for a in alphas]
        ax.errorbar(alphas, means, yerr=stds, fmt='-o', color=color, linewidth=2,
                    capsize=5, capthick=1.5)
        ax.axvline(best_alpha, color='gold', linestyle='--', linewidth=2, label=f'α*={best_alpha}')
        ax.set_xlabel('AI Alignment Level (α)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    eb(axes[0, 0], 'seci', 'b', 'SECI (-1 to +1)',
       'Social Echo Chamber\n(negative = stronger bubble)', (-1.1, 1.1))
    axes[0, 0].axhline(0, color='k', linestyle=':', alpha=0.5)

    eb(axes[0, 1], 'aeci', 'r', 'AECI (-1 to +1)',
       'AI-Induced Bubble\n(negative = stronger AI bubble)', (-1.1, 1.1))
    axes[0, 1].axhline(0, color='k', linestyle=':', alpha=0.5)

    ax = axes[0, 2]
    ax.plot(alphas, total_vals, 'k-o', linewidth=2.5, label='|SECI|+|AECI|')
    ax.plot(best_alpha, min(total_vals), 'g*', markersize=18, zorder=5, label=f'α*={best_alpha}')
    ax.fill_between(alphas, total_vals, alpha=0.15, color='purple')
    ax.axvline(best_alpha, color='gold', linestyle='--', linewidth=2)
    ax.set_xlabel('α')
    ax.set_ylabel('|SECI| + |AECI|')
    ax.set_title('Total Bubble Intensity\n(minimise to find Goldilocks zone)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    eb(axes[1, 0], 'mae', 'm', 'Mean Absolute Error',
       'Belief Accuracy\n(lower = beliefs closer to ground truth)')

    eb(axes[1, 1], 'unmet', 'darkorange', 'Unmet high-need cells',
       'Unmet Needs (level ≥4, 0 tokens)\n(lower = better disaster response)')

    eb(axes[1, 2], 'prec', 'teal', 'Correct / Total targets',
       'Relief Targeting Precision\n(higher = relief on high-need cells)', (0, 1.05))

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'goldilocks_alignment_sweep.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Goldilocks figure saved: {path}")

    _plot_timeseries(all_results, save_dir, best_alpha)
    return best_alpha


def _plot_timeseries(all_results, save_dir, best_alpha=None):
    """2×3 timeseries with ± std shading per alignment level."""
    colors = plt.cm.viridis(np.linspace(0, 1, len(ALIGNMENT_SWEEP)))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f'Filter Bubble & Delivery Metrics Over Time  (mean ± std, N={N_RUNS})',
        fontsize=13, fontweight='bold'
    )
    ax_seci, ax_aeci, ax_mae   = axes[0]
    ax_prec, ax_unmet, ax_spare = axes[1]
    ax_spare.axis('off')

    for color, (res, alpha) in zip(colors, zip(all_results, ALIGNMENT_SWEEP)):
        tf    = list(range(res['n_ticks']))
        ts    = res['metric_ticks']
        label = f'α={alpha}' + (' ★' if alpha == best_alpha else '')

        # SECI / AECI — combined exploit+explor, per tick
        for mk, ax_d in [('seci', ax_seci), ('aeci', ax_aeci)]:
            m = (np.array(res[f'{mk}_exploit_mean']) + np.array(res[f'{mk}_explor_mean'])) / 2
            s = (np.array(res[f'{mk}_exploit_std'])  + np.array(res[f'{mk}_explor_std']))  / 2
            x = tf[:len(m)]
            ax_d.plot(x, m, color=color, linewidth=1.8, label=label)
            ax_d.fill_between(x, m - s, m + s, color=color, alpha=0.2)

        # MAE — combined, sampled every 5 ticks
        mae_m = (np.array(res['mae_exploit_mean']) + np.array(res['mae_explor_mean'])) / 2
        mae_s = (np.array(res['mae_exploit_std'])  + np.array(res['mae_explor_std']))  / 2
        x = ts[:len(mae_m)]
        ax_mae.plot(x, mae_m, color=color, linewidth=1.8, label=label)
        ax_mae.fill_between(x, mae_m - mae_s, mae_m + mae_s, color=color, alpha=0.2)

        # Precision — exploit (dashed) and explor (solid), sampled, skip NaN
        for mk, ls, sfx in [('prec_exploit', '--', 'exploit'), ('prec_explor', '-', 'explor')]:
            pm = np.array(res[f'{mk}_mean'])
            ps = np.array(res[f'{mk}_std'])
            valid = ~np.isnan(pm)
            if np.any(valid):
                tv = np.array(ts[:len(pm)])[valid]
                ax_prec.plot(tv, pm[valid], color=color, linewidth=1.5,
                             linestyle=ls, label=f'α={alpha} {sfx}')
                ax_prec.fill_between(tv, (pm - ps)[valid], (pm + ps)[valid],
                                     color=color, alpha=0.15)

        # Unmet needs — per tick
        un_m = np.array(res['unmet_needs_mean'])
        un_s = np.array(res['unmet_needs_std'])
        x = tf[:len(un_m)]
        ax_unmet.plot(x, un_m, color=color, linewidth=1.8, label=label)
        ax_unmet.fill_between(x, un_m - un_s, un_m + un_s, color=color, alpha=0.2)

    for ax, title, ylabel, ylim, hl in [
        (ax_seci,  'SECI Over Time',
         'SECI (-1 to +1)', (-1.1, 1.1), 0),
        (ax_aeci,  'AECI Over Time',
         'AECI (-1 to +1)', (-1.1, 1.1), 0),
        (ax_mae,   'Belief MAE Over Time',
         'Mean Absolute Error', (0, None), None),
        (ax_prec,  'Relief Targeting Precision\n(solid=exploratory, dashed=exploitative)',
         'Correct / Total', (0, 1.05), 0.6),
        (ax_unmet, 'Unmet High-Need Cells (level ≥4, 0 tokens)',
         'Count', (0, None), None),
    ]:
        ax.set_xlabel('Tick')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        lo, hi = ylim
        if lo is not None:
            ax.set_ylim(bottom=lo)
        if hi is not None:
            ax.set_ylim(top=hi)
        if hl is not None:
            ax.axhline(hl, color='k', linestyle=':', alpha=0.4)

    ax_seci.legend(fontsize=8)
    ax_aeci.legend(fontsize=8)
    ax_mae.legend(fontsize=8)
    ax_unmet.legend(fontsize=8)
    # Precision legend: deduplicate (keep one entry per α)
    hs, ls = ax_prec.get_legend_handles_labels()
    ax_prec.legend(hs[::2], [l.replace(' exploit', '') for l in ls[::2]], fontsize=8)

    plt.tight_layout()
    path = os.path.join(save_dir, 'bubble_timeseries.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Time-series figure saved: {path}")


def plot_factor_comparison(rumor_res, disaster_res, mix_res, save_dir):
    """3×3 bar chart comparing factor effects on bubble & response metrics."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle(
        f'Factor Effects at α={FACTOR_ALPHA}  (mean ± std, N={N_FACTOR_RUNS} replications)\n'
        'Each column: one factor swept while others held at base values',
        fontsize=12, fontweight='bold'
    )

    factor_cols = [
        ('Rumour probability\n(0=no rumours, 1=all communities)',
         RUMOR_SWEEP, rumor_res),
        ('Disaster tempo\n(0=static, 2=medium, 3=rapid)',
         DISASTER_SWEEP, disaster_res),
        ('Exploitative share\n(fraction of exploitative agents)',
         EXPLOITATIVE_SWEEP, mix_res),
    ]
    row_metrics = [
        ('Steady-state SECI\n(negative = social bubble)', 'seci_exploit', (-1.1, 1.1)),
        ('Steady-state MAE\n(lower = accurate beliefs)',  'mae_exploit',  (0, None)),
        ('Final unmet needs\n(lower = better response)',  'unmet_needs',  (0, None)),
    ]
    bar_colors = ['#2196F3', '#FF9800', '#4CAF50']

    for col, (factor_label, factor_levels, res_dict) in enumerate(factor_cols):
        for row, (metric_label, metric_key, ylim) in enumerate(row_metrics):
            ax = axes[row, col]
            means, stds = [], []
            for lv in factor_levels:
                res = res_dict[lv]
                m = ss(res[f'{metric_key}_mean'])
                s = ss(res[f'{metric_key}_std'])
                means.append(m)
                stds.append(s if not np.isnan(s) else 0.0)

            x = np.arange(len(factor_levels))
            ax.bar(x, means, yerr=stds,
                   color=bar_colors[:len(factor_levels)],
                   alpha=0.8, capsize=6, edgecolor='white', linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels([str(v) for v in factor_levels], fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')

            if row == 0:
                ax.set_title(factor_label, fontsize=10, fontweight='bold')
            if col == 0:
                ax.set_ylabel(metric_label, fontsize=9)

            lo, hi = ylim
            if lo is not None:
                ax.set_ylim(bottom=lo)
            if hi is not None:
                ax.set_ylim(top=hi)
            if metric_key == 'seci_exploit':
                ax.axhline(0, color='k', linestyle=':', alpha=0.5)

    plt.tight_layout()
    path = os.path.join(save_dir, 'factor_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Factor comparison figure saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('=' * 70)
    print('GOLDILOCKS ALIGNMENT EXPERIMENT: SOCIAL vs. AI FILTER BUBBLE INTERPLAY')
    print('=' * 70)
    print(f'Sweeping alignment levels: {ALIGNMENT_SWEEP}')
    print(f'Ticks per run: {base_params["ticks"]}')
    print(f'Replications (primary sweep): {N_RUNS}')
    print(f'Replications (factor sweeps): {N_FACTOR_RUNS}')
    print(f'Steady-state window: last {STEADY_STATE_WINDOW} ticks\n')

    save_dir = 'test_results'

    # 1. Primary alignment sweep
    all_results = []
    for alpha in ALIGNMENT_SWEEP:
        params = {**base_params, 'ai_alignment_level': alpha}
        all_results.append(run_replicated(params, N_RUNS, f'Alignment α={alpha:.1f}'))

    metrics = compute_goldilocks_metrics(all_results)

    print('\n' + '=' * 70)
    print('STEADY-STATE METRICS SUMMARY')
    print('=' * 70)
    print(f"{'α':>6}  {'SECI':>8}  {'AECI':>8}  {'|S|+|A|':>8}  {'MAE':>7}  {'Unmet':>7}")
    print('-' * 55)
    min_bubble = min(v['total_bubble'] for v in metrics.values())
    for alpha in ALIGNMENT_SWEEP:
        m = metrics[alpha]
        tag = '  ← α*' if abs(m['total_bubble'] - min_bubble) < 1e-9 else ''
        print(f"{alpha:>6.1f}  {m['seci']:>8.3f}  {m['aeci']:>8.3f}  "
              f"{m['total_bubble']:>8.3f}  {m['mae']:>7.3f}  {m['unmet']:>7.1f}{tag}")

    plot_goldilocks(metrics, all_results, save_dir)

    # 2. Factor sweeps (at fixed α = FACTOR_ALPHA)
    print('\n' + '=' * 70)
    print(f'FACTOR SWEEPS  (all at α={FACTOR_ALPHA})')
    print('=' * 70)

    rumor_results = {}
    for rp in RUMOR_SWEEP:
        params = {**base_params, 'ai_alignment_level': FACTOR_ALPHA, 'rumor_probability': rp}
        rumor_results[rp] = run_replicated(params, N_FACTOR_RUNS, f'Rumour p={rp}')

    disaster_results = {}
    for dd in DISASTER_SWEEP:
        params = {**base_params, 'ai_alignment_level': FACTOR_ALPHA, 'disaster_dynamics': dd}
        disaster_results[dd] = run_replicated(params, N_FACTOR_RUNS, f'Disaster dynamics={dd}')

    mix_results = {}
    for se in EXPLOITATIVE_SWEEP:
        params = {**base_params, 'ai_alignment_level': FACTOR_ALPHA, 'share_exploitative': se}
        mix_results[se] = run_replicated(params, N_FACTOR_RUNS, f'Exploitative share={se}')

    plot_factor_comparison(rumor_results, disaster_results, mix_results, save_dir)

    print('\n' + '=' * 70)
    print('EXPERIMENT COMPLETE')
    print('=' * 70)
    print('\nInterpretation guide:')
    print('  SECI < 0    : social echo chamber (friends converge more than random)')
    print('  AECI < 0    : AI-induced bubble (AI users more homogeneous than global)')
    print('  total_bubble: |SECI| + |AECI| — minimise to find Goldilocks α*')
    print('  unmet_needs : cells at disaster L4+ with zero relief — measures response failure')
    print('  precision   : fraction of relief correctly sent to high-need cells')
