"""
Automated Diagram Generator for DisasterAI Model

This script creates publication-ready diagrams of the model architecture.

Requirements:
    pip install matplotlib networkx graphviz

Usage:
    python create_model_diagrams.py

Output:
    - Figure1_ModelArchitecture.png/pdf
    - Figure2_DecisionCycle.png/pdf
    - Figure3_SocialNetwork.png/pdf
    - Figure4_QLearning.png/pdf
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import networkx as nx
import numpy as np

# Color scheme (colorblind-friendly)
COLORS = {
    'exploitative': '#D55E00',  # Vermillion
    'exploratory': '#009E73',   # Bluish green
    'ai': '#0072B2',            # Blue
    'environment': '#999999',   # Gray
    'process': '#F0E442',       # Yellow
}

# Output directory
OUTPUT_DIR = "model_diagrams"
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_figure1_architecture():
    """
    Figure 1: Model Architecture Overview
    Shows environment, agents, and relationships
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.5, 'DisasterAI Model Architecture',
            ha='center', va='top', fontsize=16, fontweight='bold')

    # Environment box (top)
    env_box = FancyBboxPatch((0.5, 7), 9, 1.8,
                             boxstyle="round,pad=0.1",
                             edgecolor='black', facecolor=COLORS['environment'],
                             linewidth=2, alpha=0.3)
    ax.add_patch(env_box)
    ax.text(5, 8.5, 'Environment', ha='center', va='top', fontsize=12, fontweight='bold')
    ax.text(2, 8.0, '• Disaster Grid (30×30)\n  Levels: L0-L5',
            ha='left', va='top', fontsize=9)
    ax.text(5, 8.0, '• Social Network\n  2-3 components',
            ha='left', va='top', fontsize=9)
    ax.text(7.5, 8.0, '• Rumors\n  30% probability',
            ha='left', va='top', fontsize=9)

    # Exploitative agents box (left)
    exp_box = FancyBboxPatch((0.5, 3.5), 4, 2.5,
                             boxstyle="round,pad=0.1",
                             edgecolor='black', facecolor=COLORS['exploitative'],
                             linewidth=2, alpha=0.3)
    ax.add_patch(exp_box)
    ax.text(2.5, 5.7, 'Exploitative Agents (N=15)',
            ha='center', va='top', fontsize=11, fontweight='bold')
    ax.text(2.5, 5.2,
            'Goal: Confirm beliefs\n'
            'Strategy: Trust friends\n'
            'Sensing radius: 2 cells\n'
            'Trust LR: 0.03\n'
            'Biases: +0.1 friends, +0.1 self',
            ha='center', va='top', fontsize=8)

    # Exploratory agents box (middle)
    exr_box = FancyBboxPatch((5, 3.5), 4, 2.5,
                             boxstyle="round,pad=0.1",
                             edgecolor='black', facecolor=COLORS['exploratory'],
                             linewidth=2, alpha=0.3)
    ax.add_patch(exr_box)
    ax.text(7, 5.7, 'Exploratory Agents (N=15)',
            ha='center', va='top', fontsize=11, fontweight='bold')
    ax.text(7, 5.2,
            'Goal: Maximize accuracy\n'
            'Strategy: Seek truth\n'
            'Sensing radius: 3 cells\n'
            'Trust LR: 0.08\n'
            'Biases: -0.05 self, +α AI',
            ha='center', va='top', fontsize=8)

    # AI agents box (bottom)
    ai_box = FancyBboxPatch((2, 0.5), 6, 2,
                            boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor=COLORS['ai'],
                            linewidth=2, alpha=0.3)
    ax.add_patch(ai_box)
    ax.text(5, 2.2, 'AI Agents (N=5)',
            ha='center', va='top', fontsize=11, fontweight='bold')
    ax.text(5, 1.7,
            'Knowledge coverage: 15% per tick\n'
            'Memory retention: 80%\n'
            'Guessing probability: 75%\n'
            'Alignment parameter α ∈ [0,1]',
            ha='center', va='top', fontsize=8)

    # Arrows
    # Environment to agents
    arrow1 = FancyArrowPatch((2.5, 7.0), (2.5, 6.0),
                            arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='black')
    ax.add_patch(arrow1)
    ax.text(2.8, 6.5, 'Sense', fontsize=8, rotation=-90)

    arrow2 = FancyArrowPatch((7, 7.0), (7, 6.0),
                            arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='black')
    ax.add_patch(arrow2)
    ax.text(7.3, 6.5, 'Sense', fontsize=8, rotation=-90)

    # Agents to AI (bidirectional)
    arrow3 = FancyArrowPatch((2.5, 3.5), (4, 2.5),
                            arrowstyle='<->', mutation_scale=20,
                            linewidth=2, color='black', linestyle='dashed')
    ax.add_patch(arrow3)
    ax.text(2.8, 3, 'Query/Report', fontsize=7, rotation=-45)

    arrow4 = FancyArrowPatch((7, 3.5), (6, 2.5),
                            arrowstyle='<->', mutation_scale=20,
                            linewidth=2, color='black', linestyle='dashed')
    ax.add_patch(arrow4)
    ax.text(6.5, 3, 'Query/Report', fontsize=7, rotation=45)

    # Agents to agents (social network)
    arrow5 = FancyArrowPatch((4.5, 4.5), (5.0, 4.5),
                            arrowstyle='<->', mutation_scale=20,
                            linewidth=2, color='black', linestyle='dotted')
    ax.add_patch(arrow5)
    ax.text(4.75, 4.7, 'Social\nNetwork', fontsize=7, ha='center')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Figure1_ModelArchitecture.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/Figure1_ModelArchitecture.pdf', bbox_inches='tight')
    print("✓ Created Figure 1: Model Architecture")
    plt.close()


def create_figure2_decision_cycle():
    """
    Figure 2: Agent Decision Cycle
    Flowchart showing 6-step process
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 11))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Title
    ax.text(4, 11.5, 'Agent Decision Cycle (per time step)',
            ha='center', va='top', fontsize=14, fontweight='bold')

    # Define steps with colors
    steps = [
        (4, 10.5, 'Start', 'white', 'ellipse'),
        (4, 9.5, '1. Sense Environment\nRadius: 2-3 cells\nNoise: 8%', '#e1f5ff', 'box'),
        (4, 8.0, '2. Select Source\nε-greedy (ε=0.3)\nQ-learning + Biases', '#fff4e1', 'box'),
        (4, 6.5, '3. Query Source\n(Self/Friend/AI)\nGet information', '#ffe1f5', 'box'),
        (4, 5.0, '4. Update Beliefs\nBayesian updating\nPrecision-weighted', '#e1ffe1', 'box'),
        (4, 3.5, '5. Send Relief\nTop 5 cells (L≥3)\nQueue rewards', '#f5e1ff', 'box'),
        (4, 2.0, '6. Process Rewards\n(2-tick delay)\nUpdate Q & Trust', '#ffe1e1', 'box'),
        (4, 0.7, '7. Apply Decay\nConfidence: -0.0003-0.0005\nTrust: -0.0005-0.003', '#f0f0f0', 'box'),
        (4, -0.3, 'End', 'white', 'ellipse'),
    ]

    for x, y, text, color, shape in steps:
        if shape == 'ellipse':
            ellipse = mpatches.Ellipse((x, y), 1.5, 0.6,
                                       edgecolor='black', facecolor=color, linewidth=2)
            ax.add_patch(ellipse)
        else:
            box = FancyBboxPatch((x-1.5, y-0.4), 3, 0.8,
                                boxstyle="round,pad=0.05",
                                edgecolor='black', facecolor=color, linewidth=1.5)
            ax.add_patch(box)

        ax.text(x, y, text, ha='center', va='center', fontsize=8, multialignment='center')

    # Arrows between steps
    arrow_positions = [
        (4, 10.2, 4, 9.9),
        (4, 9.1, 4, 8.4),
        (4, 7.6, 4, 6.9),
        (4, 6.1, 4, 5.4),
        (4, 4.6, 4, 3.9),
        (4, 3.1, 4, 2.4),
        (4, 1.6, 4, 1.1),
        (4, 0.4, 4, 0.0),
    ]

    for x1, y1, x2, y2 in arrow_positions:
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', mutation_scale=15,
                               linewidth=2, color='black')
        ax.add_patch(arrow)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Figure2_DecisionCycle.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/Figure2_DecisionCycle.pdf', bbox_inches='tight')
    print("✓ Created Figure 2: Decision Cycle")
    plt.close()


def create_figure3_social_network():
    """
    Figure 3: Example Social Network
    Shows network with multiple components
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Create a sample network with 3 components
    # Component 1: 12 nodes
    G1 = nx.watts_strogatz_graph(12, 4, 0.3, seed=42)
    # Component 2: 10 nodes
    G2 = nx.watts_strogatz_graph(10, 4, 0.3, seed=43)
    # Component 3: 8 nodes
    G3 = nx.watts_strogatz_graph(8, 3, 0.3, seed=44)

    # Combine into one graph (disjoint)
    G = nx.Graph()
    G.add_nodes_from([(i, {'component': 1}) for i in range(12)])
    G.add_nodes_from([(i+12, {'component': 2}) for i in range(10)])
    G.add_nodes_from([(i+22, {'component': 3}) for i in range(8)])

    # Add edges
    for u, v in G1.edges():
        G.add_edge(u, v)
    for u, v in G2.edges():
        G.add_edge(u+12, v+12)
    for u, v in G3.edges():
        G.add_edge(u+22, v+22)

    # Layout
    pos = {}
    pos.update(nx.spring_layout(G.subgraph(range(12)), center=(0, 0), k=0.5, seed=42))
    pos.update(nx.spring_layout(G.subgraph(range(12, 22)), center=(3, 0), k=0.5, seed=43))
    pos.update(nx.spring_layout(G.subgraph(range(22, 30)), center=(1.5, -2), k=0.5, seed=44))

    # Assign agent types (50-50 split)
    agent_types = {}
    for i in range(30):
        agent_types[i] = 'exploitative' if i < 15 else 'exploratory'

    # Node colors
    node_colors = [COLORS[agent_types[node]] for node in G.nodes()]

    # Draw
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                          node_size=300, alpha=0.7, ax=ax)
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.5, ax=ax)
    nx.draw_networkx_labels(G, pos,
                           labels={i: f'H{i}' for i in range(30)},
                           font_size=7, ax=ax)

    # Add component labels
    ax.text(0, 1.2, 'Component 1\n(N=12)', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(3, 1.2, 'Component 2\n(N=10)', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(1.5, -3, 'Component 3\n(N=8)', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['exploitative'], alpha=0.7,
                      edgecolor='black', label='Exploitative Agents'),
        mpatches.Patch(facecolor=COLORS['exploratory'], alpha=0.7,
                      edgecolor='black', label='Exploratory Agents')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    ax.set_title('Example Social Network Structure (N=30)', fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Figure3_SocialNetwork.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/Figure3_SocialNetwork.pdf', bbox_inches='tight')
    print("✓ Created Figure 3: Social Network")
    plt.close()


def create_figure4_qlearning():
    """
    Figure 4: Q-Learning Mechanism
    Shows source selection and reward loop
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(6, 7.5, 'Q-Learning Mechanism for Source Selection',
            ha='center', va='top', fontsize=14, fontweight='bold')

    # Q-Table
    qtable = FancyBboxPatch((0.5, 5), 3, 1.8,
                           boxstyle="round,pad=0.1",
                           edgecolor='black', facecolor='#e1f5ff', linewidth=2)
    ax.add_patch(qtable)
    ax.text(2, 6.5, 'Q-Table', ha='center', fontsize=11, fontweight='bold')
    ax.text(2, 6.0,
            'Self: 0.0\n'
            'Human: 0.05\n'
            'AI_0-4: 0.0',
            ha='center', va='top', fontsize=8)

    # Epsilon-greedy decision
    decision = FancyBboxPatch((4.5, 5), 3, 1.8,
                             boxstyle="round,pad=0.1",
                             edgecolor='black', facecolor='#fff4e1', linewidth=2)
    ax.add_patch(decision)
    ax.text(6, 6.5, 'ε-Greedy Selection', ha='center', fontsize=11, fontweight='bold')
    ax.text(6, 6.0,
            'if random() < 0.3:\n'
            '  Explore (random)\n'
            'else:\n'
            '  Exploit (max Q+bias)',
            ha='center', va='top', fontsize=7, family='monospace')

    # Chosen source
    source = FancyBboxPatch((8.5, 5), 3, 1.8,
                           boxstyle="round,pad=0.1",
                           edgecolor='black', facecolor='#e1ffe1', linewidth=2)
    ax.add_patch(source)
    ax.text(10, 6.5, 'Chosen Source', ha='center', fontsize=11, fontweight='bold')
    ax.text(10, 6.0,
            'Query:\n'
            '• Self beliefs\n'
            '• Friend\n'
            '• AI agent',
            ha='center', va='top', fontsize=8)

    # Action & Outcome
    action = FancyBboxPatch((8.5, 2.5), 3, 1.8,
                           boxstyle="round,pad=0.1",
                           edgecolor='black', facecolor='#ffe1f5', linewidth=2)
    ax.add_patch(action)
    ax.text(10, 4.0, 'Send Relief', ha='center', fontsize=11, fontweight='bold')
    ax.text(10, 3.5,
            'Target cells\n'
            'based on beliefs\n'
            '(top 5, L≥3)',
            ha='center', va='top', fontsize=8)

    # Reward calculation
    reward = FancyBboxPatch((4.5, 0.5), 3, 1.5,
                           boxstyle="round,pad=0.1",
                           edgecolor='black', facecolor='#ffe1e1', linewidth=2)
    ax.add_patch(reward)
    ax.text(6, 1.7, 'Reward (2 ticks later)', ha='center', fontsize=11, fontweight='bold')
    ax.text(6, 1.2,
            'L5: +5, L4: +3, L3: +1.5\n'
            'L2: 0, L1: -1, L0: -2',
            ha='center', va='top', fontsize=7)

    # Q-update
    update = FancyBboxPatch((0.5, 0.5), 3, 1.5,
                           boxstyle="round,pad=0.1",
                           edgecolor='black', facecolor='#f5e1ff', linewidth=2)
    ax.add_patch(update)
    ax.text(2, 1.7, 'Update Q-Value', ha='center', fontsize=11, fontweight='bold')
    ax.text(2, 1.2,
            'Q ← Q + α(R - Q)\n'
            'α = 0.15',
            ha='center', va='top', fontsize=8)

    # Arrows
    arrows = [
        ((3.5, 6), (4.5, 6), 'black'),           # Q-table to decision
        ((7.5, 6), (8.5, 6), 'black'),           # Decision to source
        ((10, 5), (10, 4.3), 'black'),           # Source to action
        ((8.5, 3.5), (7.5, 1.5), 'black'),      # Action to reward
        ((4.5, 1.2), (3.5, 1.2), 'black'),      # Reward to update
        ((2, 2), (2, 5), 'black'),              # Update back to Q-table (feedback)
    ]

    for (x1, y1), (x2, y2), color in arrows:
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', mutation_scale=15,
                               linewidth=2, color=color)
        ax.add_patch(arrow)

    # Add labels on arrows
    ax.text(4, 6.3, 'Select', fontsize=8)
    ax.text(8, 6.3, 'Query', fontsize=8)
    ax.text(10.5, 4.7, 'Act', fontsize=8)
    ax.text(7.8, 2.5, 'Evaluate', fontsize=8, rotation=-30)
    ax.text(4, 0.9, 'Calculate', fontsize=8)
    ax.text(1.5, 3.5, 'Update', fontsize=8, rotation=90)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Figure4_QLearning.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/Figure4_QLearning.pdf', bbox_inches='tight')
    print("✓ Created Figure 4: Q-Learning Mechanism")
    plt.close()


def create_all_figures():
    """Generate all figures"""
    print("\n" + "="*60)
    print("DisasterAI Model Diagram Generator")
    print("="*60)
    print(f"\nGenerating diagrams in '{OUTPUT_DIR}/' directory...\n")

    create_figure1_architecture()
    create_figure2_decision_cycle()
    create_figure3_social_network()
    create_figure4_qlearning()

    print("\n" + "="*60)
    print("All diagrams created successfully!")
    print("="*60)
    print(f"\nFiles created in '{OUTPUT_DIR}/':")
    print("  - Figure1_ModelArchitecture.png/pdf")
    print("  - Figure2_DecisionCycle.png/pdf")
    print("  - Figure3_SocialNetwork.png/pdf")
    print("  - Figure4_QLearning.png/pdf")
    print("\nReady for publication! (300 DPI, PDF format)")
    print("="*60 + "\n")


if __name__ == "__main__":
    create_all_figures()
