"""
Information Landscape Tracking - Query vs Acceptance Divergence
===============================================================

Tracks how agents' information-seeking behavior (queries) differs from
their information-acceptance behavior (trust/belief formation) over time.

This reveals:
- Which sources agents explore vs which they trust
- How Q-learning shapes query patterns
- How acceptance criteria filter information
- Divergence between exploration and exploitation
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def track_information_landscape(model):
    """Track query vs acceptance patterns for exploitative and exploratory agents.

    Call this every 5 ticks to build up temporal data.
    """
    if not hasattr(model, 'info_landscape_data'):
        model.info_landscape_data = {
            'ticks': [],
            'exploit_query_ai': [], 'exploit_query_human': [], 'exploit_query_self': [],
            'exploit_accept_ai': [], 'exploit_accept_friend': [],
            'explor_query_ai': [], 'explor_query_human': [], 'explor_query_self': [],
            'explor_accept_ai': [], 'explor_accept_friend': []
        }

    data = model.info_landscape_data
    data['ticks'].append(model.tick)

    # Collect data by agent type
    for agent_type_name in ['exploitative', 'exploratory']:
        agents = [a for a in model.humans.values() if a.agent_type == agent_type_name]

        if not agents:
            continue

        # Query ratios
        query_ai_ratios = []
        query_human_ratios = []
        query_self_ratios = []

        # Acceptance ratios
        accept_ai_ratios = []
        accept_friend_ratios = []

        for agent in agents:
            # QUERIES
            total_calls = agent.accum_calls_total
            if total_calls > 0:
                query_ai_ratios.append(agent.accum_calls_ai / total_calls)
                query_human_ratios.append(agent.accum_calls_human / total_calls)
                # self_action = total - ai - human
                query_self_ratios.append(max(0, (total_calls - agent.accum_calls_ai - agent.accum_calls_human) / total_calls))

            # ACCEPTANCES
            total_accepts = agent.accepted_ai + agent.accepted_friend
            if total_accepts > 0:
                accept_ai_ratios.append(agent.accepted_ai / total_accepts)
                accept_friend_ratios.append(agent.accepted_friend / total_accepts)

        # Store means
        prefix = 'exploit' if agent_type_name == 'exploitative' else 'explor'

        data[f'{prefix}_query_ai'].append(np.mean(query_ai_ratios) if query_ai_ratios else 0)
        data[f'{prefix}_query_human'].append(np.mean(query_human_ratios) if query_human_ratios else 0)
        data[f'{prefix}_query_self'].append(np.mean(query_self_ratios) if query_self_ratios else 0)

        data[f'{prefix}_accept_ai'].append(np.mean(accept_ai_ratios) if accept_ai_ratios else 0)
        data[f'{prefix}_accept_friend'].append(np.mean(accept_friend_ratios) if accept_friend_ratios else 0)


def plot_information_landscape(model, save_dir="analysis_plots"):
    """Create comprehensive visualization of information landscape evolution."""
    if not hasattr(model, 'info_landscape_data'):
        print("No information landscape data to plot")
        return

    os.makedirs(save_dir, exist_ok=True)
    data = model.info_landscape_data

    if len(data['ticks']) == 0:
        print("No data points collected yet")
        return

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

    ticks = data['ticks']

    # === EXPLOITATIVE AGENTS ===

    # Panel 1: Exploitative - Queries
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ticks, data['exploit_query_ai'], 'o-', label='AI', color='#3498DB', linewidth=2, markersize=4)
    ax1.plot(ticks, data['exploit_query_human'], 's-', label='Human', color='#E74C3C', linewidth=2, markersize=4)
    ax1.plot(ticks, data['exploit_query_self'], '^-', label='Self', color='#95A5A6', linewidth=2, markersize=4)
    ax1.set_ylabel('Query Ratio', fontsize=11, fontweight='bold')
    ax1.set_title('Exploitative: Information SEEKING (Queries)', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)

    # Panel 2: Exploitative - Acceptances
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(ticks, data['exploit_accept_ai'], 'o-', label='AI', color='#3498DB', linewidth=2, markersize=4)
    ax2.plot(ticks, data['exploit_accept_friend'], 's-', label='Friends', color='#E74C3C', linewidth=2, markersize=4)
    ax2.set_ylabel('Acceptance Ratio', fontsize=11, fontweight='bold')
    ax2.set_title('Exploitative: BELIEF FORMATION (Acceptances)', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)

    # Panel 3: Exploitative - Divergence
    ax3 = fig.add_subplot(gs[2, 0])
    ai_divergence = np.array(data['exploit_query_ai']) - np.array(data['exploit_accept_ai'])
    friend_divergence = np.array(data['exploit_query_human']) - np.array(data['exploit_accept_friend'])
    ax3.plot(ticks, ai_divergence, 'o-', label='AI Divergence', color='#3498DB', linewidth=2, markersize=4)
    ax3.plot(ticks, friend_divergence, 's-', label='Friend Divergence', color='#E74C3C', linewidth=2, markersize=4)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_ylabel('Query - Accept', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Tick', fontsize=11, fontweight='bold')
    ax3.set_title('Exploitative: QUERY-ACCEPTANCE DIVERGENCE', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', framealpha=0.9)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Exploitative - Interpretation
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.axis('off')
    interpretation = """
    INTERPRETATION (Exploitative):

    • Positive divergence: Query more than accept
      → Exploring source but not trusting it

    • Negative divergence: Accept more than query
      → Highly selective but trust what queries

    • Near zero: Balanced query-acceptance
      → Consistent trust relationship
    """
    ax4.text(0.1, 0.5, interpretation, fontsize=10, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # === EXPLORATORY AGENTS ===

    # Panel 5: Exploratory - Queries
    ax5 = fig.add_subplot(gs[0, 1])
    ax5.plot(ticks, data['explor_query_ai'], 'o-', label='AI', color='#3498DB', linewidth=2, markersize=4, linestyle='--')
    ax5.plot(ticks, data['explor_query_human'], 's-', label='Human', color='#E74C3C', linewidth=2, markersize=4, linestyle='--')
    ax5.plot(ticks, data['explor_query_self'], '^-', label='Self', color='#95A5A6', linewidth=2, markersize=4, linestyle='--')
    ax5.set_ylabel('Query Ratio', fontsize=11, fontweight='bold')
    ax5.set_title('Exploratory: Information SEEKING (Queries)', fontsize=12, fontweight='bold')
    ax5.legend(loc='best', framealpha=0.9)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(-0.05, 1.05)

    # Panel 6: Exploratory - Acceptances
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.plot(ticks, data['explor_accept_ai'], 'o-', label='AI', color='#3498DB', linewidth=2, markersize=4, linestyle='--')
    ax6.plot(ticks, data['explor_accept_friend'], 's-', label='Friends', color='#E74C3C', linewidth=2, markersize=4, linestyle='--')
    ax6.set_ylabel('Acceptance Ratio', fontsize=11, fontweight='bold')
    ax6.set_title('Exploratory: BELIEF FORMATION (Acceptances)', fontsize=12, fontweight='bold')
    ax6.legend(loc='best', framealpha=0.9)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(-0.05, 1.05)

    # Panel 7: Exploratory - Divergence
    ax7 = fig.add_subplot(gs[2, 1])
    ai_divergence = np.array(data['explor_query_ai']) - np.array(data['explor_accept_ai'])
    friend_divergence = np.array(data['explor_query_human']) - np.array(data['explor_accept_friend'])
    ax7.plot(ticks, ai_divergence, 'o-', label='AI Divergence', color='#3498DB', linewidth=2, markersize=4, linestyle='--')
    ax7.plot(ticks, friend_divergence, 's-', label='Friend Divergence', color='#E74C3C', linewidth=2, markersize=4, linestyle='--')
    ax7.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax7.set_ylabel('Query - Accept', fontsize=11, fontweight='bold')
    ax7.set_xlabel('Tick', fontsize=11, fontweight='bold')
    ax7.set_title('Exploratory: QUERY-ACCEPTANCE DIVERGENCE', fontsize=12, fontweight='bold')
    ax7.legend(loc='best', framealpha=0.9)
    ax7.grid(True, alpha=0.3)

    # Panel 8: Exploratory - Interpretation
    ax8 = fig.add_subplot(gs[3, 1])
    ax8.axis('off')
    interpretation = """
    INTERPRETATION (Exploratory):

    • Q-learning shapes queries
      → Agents explore different sources

    • Acceptance filters by accuracy
      → Only trust accurate sources

    • Divergence shows learning
      → Testing sources, keeping good ones
    """
    ax8.text(0.1, 0.5, interpretation, fontsize=10, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.suptitle(f'Information Landscape Evolution (Alignment={model.ai_alignment_level:.2f})',
                 fontsize=14, fontweight='bold', y=0.995)

    filename = f"info_landscape_align_{model.ai_alignment_level:.2f}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved information landscape visualization: {filename}")
