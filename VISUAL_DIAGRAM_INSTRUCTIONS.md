# Instructions for Creating Model Visuals

This document provides three methods for creating publication-quality diagrams of the DisasterAI model.

---

## Method 1: Automated Python Diagram (Recommended)

### Installation
```bash
pip install graphviz matplotlib networkx
```

### Run the diagram generator
See `create_model_diagrams.py` (created below)

---

## Method 2: Mermaid Diagrams (Web-based, Easy)

### Tool: Mermaid Live Editor
URL: https://mermaid.live/

### Diagram 1: Model Architecture

Copy this code into Mermaid Live Editor:

```mermaid
graph TB
    subgraph "Environment"
        A[Disaster Grid<br/>30×30 cells<br/>L0-L5 levels]
        B[Social Network<br/>2-3 components<br/>Small-world topology]
        C[Rumor System<br/>30% probability<br/>Component-specific]
    end

    subgraph "Human Agents N=30"
        D[Exploitative 50%<br/>Goal: Confirmation<br/>Sensing: 2 cells<br/>Trust LR: 0.03]
        E[Exploratory 50%<br/>Goal: Accuracy<br/>Sensing: 3 cells<br/>Trust LR: 0.08]
    end

    subgraph "AI Agents N=5"
        F[Knowledge: 15% coverage<br/>Memory: 80% retention<br/>Guessing: 75% probability]
        G[Alignment Parameter α<br/>0 = Truthful<br/>1 = Fully Aligned]
    end

    A --> D
    A --> E
    B --> D
    B --> E
    D <-->|Query/Report| E
    D <-->|Query/Report| F
    E <-->|Query/Report| F
    C -.->|Misinformation| D
    C -.->|Misinformation| E
    G --> F

    style D fill:#ffcccc
    style E fill:#ccffcc
    style F fill:#ccccff
    style G fill:#ffffcc
```

### Diagram 2: Agent Decision Cycle

```mermaid
flowchart TD
    Start([Start Time Step t]) --> Sense[1. Sense Environment<br/>Radius: 2-3 cells<br/>Noise: 8%]

    Sense --> Interest{Agent Type?}
    Interest -->|Exploitative| Epi[Find Believed<br/>Epicenter]
    Interest -->|Exploratory| Uncert[Find Uncertain<br/>High-Level Areas]

    Epi --> Select
    Uncert --> Select

    Select[2. Select Information Source<br/>ε-greedy Q-learning<br/>ε=0.3]

    Select --> Source{Source?}
    Source -->|Self| Self[Use Own Beliefs]
    Source -->|Human| Human[Query Friend<br/>Highest Trust]
    Source -->|AI| AI[Query AI Agent<br/>Best Coverage]

    Self --> Bayesian
    Human --> Bayesian
    AI --> Bayesian

    Bayesian[3. Bayesian Belief Update<br/>Precision-weighted<br/>Trust-modulated]

    Bayesian --> Relief[4. Send Relief<br/>Top 5 cells L≥3<br/>Queue rewards]

    Relief --> Process[5. Process Rewards<br/>2-tick delay<br/>Update Q & Trust]

    Process --> Decay[6. Apply Decay<br/>Confidence: -0.0003-0.0005<br/>Trust: -0.0005-0.003]

    Decay --> End([End Time Step t])

    style Sense fill:#e1f5ff
    style Select fill:#fff4e1
    style Bayesian fill:#ffe1f5
    style Relief fill:#e1ffe1
    style Process fill:#f5e1ff
    style Decay fill:#ffe1e1
```

### Diagram 3: Q-Learning Mechanism

```mermaid
flowchart LR
    subgraph "Source Selection"
        A[Q-Table<br/>Self: 0.0<br/>Human: 0.05<br/>AI_0-4: 0.0]
        B{Random < ε?}
        B -->|Yes 30%| C[Explore:<br/>Random Source]
        B -->|No 70%| D[Exploit:<br/>Max Q + Bias]
    end

    subgraph "Biases"
        E[Exploitative:<br/>+0.1 Human<br/>+0.1 Self]
        F[Exploratory:<br/>-0.05 Self<br/>+α·AI]
    end

    D --> E
    D --> F
    E --> G[Choose Source]
    F --> G
    C --> G

    G --> H[Query & Update<br/>Beliefs]
    H --> I[Send Relief<br/>to High-L Cells]
    I --> J[Wait 2 Ticks]
    J --> K{Reward<br/>Calculation}

    K -->|L5| L[R = +5.0]
    K -->|L4| M[R = +3.0]
    K -->|L3| N[R = +1.5]
    K -->|L0| O[R = -2.0]

    L --> P[Update Q-Table<br/>Q ← Q + α·R - Q]
    M --> P
    N --> P
    O --> P

    P --> A

    style A fill:#e1f5ff
    style B fill:#fff4e1
    style K fill:#ffe1e1
    style P fill:#e1ffe1
```

### Diagram 4: AI Alignment Mechanism

```mermaid
flowchart TD
    A[AI Receives Query<br/>from Human Agent] --> B{Has Direct<br/>Knowledge?}

    B -->|Yes| C[Use Sensed Value<br/>L_sensed]
    B -->|No| D{Guess?<br/>75% chance}

    D -->|Yes| E[Interpolate from<br/>Nearby Known Cells]
    D -->|No| F[No Report<br/>for this cell]

    E --> C

    C --> G[Calculate Alignment Factor<br/>A = α·1+2C_caller + α·λ·1-T_caller]

    G --> H[Adjust Report<br/>L_report = L_sensed + A·L_caller - L_sensed]

    H --> I{Alignment<br/>Level α?}

    I -->|α = 0.0| J[Pure Truth<br/>L_report = L_sensed]
    I -->|α = 0.3| K[Moderate Shift<br/>Toward Caller Belief]
    I -->|α = 1.0| L[Full Alignment<br/>L_report ≈ L_caller]

    J --> M[Send Report<br/>to Human]
    K --> M
    L --> M

    M --> N[Human Updates<br/>via Bayesian]

    style A fill:#e1f5ff
    style G fill:#fff4e1
    style H fill:#ffe1f5
    style I fill:#ffe1e1
    style M fill:#e1ffe1
```

---

## Method 3: Manual Creation (PowerPoint/Draw.io)

### Tool Options
1. **Microsoft PowerPoint** (simple, widely available)
2. **Draw.io** (free, web-based: https://app.diagrams.net/)
3. **Lucidchart** (professional, requires account)
4. **Adobe Illustrator** (professional, complex)

### Step-by-Step for PowerPoint

#### Diagram A: Model Components Overview

1. **Create 4 boxes** (Insert → Shapes → Rectangle):
   ```
   Box 1: "Environment"
   - Disaster Grid (30×30)
   - Social Network (2-3 components)
   - Rumors (30% probability)

   Box 2: "Exploitative Agents (N=15)"
   - Sensing radius: 2
   - Trust LR: 0.03
   - Prefer: Friends, Self

   Box 3: "Exploratory Agents (N=15)"
   - Sensing radius: 3
   - Trust LR: 0.08
   - Prefer: Truthful AI

   Box 4: "AI Agents (N=5)"
   - Coverage: 15%
   - Alignment: 0-1
   - Guessing: 75%
   ```

2. **Add arrows** between boxes:
   - Environment → Both agent types (solid arrows)
   - Agents ↔ Agents (double arrows, label "Query/Report")
   - Agents ↔ AI (double arrows, label "Query/Report")

3. **Color coding**:
   - Exploitative: Light Red (#ffcccc)
   - Exploratory: Light Green (#ccffcc)
   - AI: Light Blue (#ccccff)
   - Environment: Light Gray (#eeeeee)

4. **Export**: File → Save As → PDF (for publication)

#### Diagram B: Agent Decision Cycle (Flowchart)

1. **Create oval shape**: "Start" (Insert → Shapes → Oval)

2. **Create rectangles** for each step:
   ```
   1. Sense Environment (radius 2-3)
   2. Select Source (ε-greedy)
   3. Update Beliefs (Bayesian)
   4. Send Relief (top 5 cells)
   5. Process Rewards (after 2 ticks)
   6. Apply Decay (confidence & trust)
   ```

3. **Create diamond** for decision point:
   ```
   "Agent Type?"
   → Exploitative: "Find Epicenter"
   → Exploratory: "Find Uncertainty"
   ```

4. **Connect with arrows** (Insert → Shapes → Arrow)

5. **Add colors** to distinguish phases:
   - Sense: Blue
   - Select: Yellow
   - Update: Purple
   - Relief: Green
   - Reward: Orange
   - Decay: Red

6. **Export as PDF**

### Layout Recommendations

**For Journal Articles**:
- Size: 6-8 inches wide
- Font: Arial or Helvetica, 10-12pt
- Line width: 1-2pt
- Resolution: 300 DPI minimum

**For Presentations**:
- Size: Full slide (10×7.5 inches)
- Font: Arial or Helvetica, 18-24pt
- Line width: 2-3pt
- High contrast colors

**For Posters**:
- Size: Scale up 2x
- Font: Arial or Helvetica, 24-36pt
- Line width: 3-4pt
- Bold outlines

---

## Method 4: Python Code to Generate Diagrams

See the accompanying file `create_model_diagrams.py` for automated generation.

This script creates:
1. **Figure 1**: Model architecture (boxes and arrows)
2. **Figure 2**: Agent decision cycle (flowchart)
3. **Figure 3**: Sample social network (graph)
4. **Figure 4**: Q-learning mechanism (state diagram)

All figures are publication-ready (300 DPI, PDF format).

---

## Recommended Figure Set for Publication

### Main Text Figures

**Figure 1: Model Architecture**
- Shows: Environment, agents, AI, relationships
- Format: Box diagram with arrows
- Caption: "Schematic overview of the DisasterAI model showing environment components, agent types, and information flows."

**Figure 2: Agent Decision Cycle**
- Shows: 6-step process per time step
- Format: Flowchart
- Caption: "Agent decision-making cycle showing sensing, source selection, belief updating, relief allocation, reward processing, and decay mechanisms."

### Supplementary Figures

**Figure S1: Q-Learning Mechanism**
- Shows: Source selection and reward feedback loop
- Format: State diagram
- Caption: "Q-learning mechanism for information source selection with ε-greedy exploration and delayed reward processing."

**Figure S2: AI Alignment Mechanism**
- Shows: How alignment parameter affects reporting
- Format: Flowchart with equations
- Caption: "AI information manipulation mechanism showing how alignment level α modulates reported values based on caller beliefs and trust."

**Figure S3: Social Network Example**
- Shows: Sample network with components
- Format: Network graph
- Caption: "Example social network structure with 30 agents organized into 3 connected components (communities) showing small-world topology."

---

## Color Scheme Recommendations

For colorblind-friendly diagrams:

```
Exploitative agents:  #D55E00 (vermillion)
Exploratory agents:   #009E73 (bluish green)
AI agents:            #0072B2 (blue)
Environment:          #999999 (gray)
Positive outcomes:    #009E73 (green)
Negative outcomes:    #D55E00 (red)
Neutral/Process:      #F0E442 (yellow)
```

Source: Wong (2011) "Points of view: Color blindness" Nature Methods

---

## File Format Recommendations

**For Submission**:
- Vector: PDF (preferred) or EPS
- Raster: TIFF or PNG (300 DPI minimum)
- Avoid: JPEG (lossy compression)

**File Naming**:
```
Figure1_ModelArchitecture.pdf
Figure2_DecisionCycle.pdf
FigureS1_QLearning.pdf
FigureS2_AIAlignment.pdf
FigureS3_SocialNetwork.pdf
```

---

## Next Steps

1. Choose your preferred method (Mermaid is fastest, Python is most customizable)
2. Generate initial diagrams
3. Review with co-authors
4. Iterate based on feedback
5. Export in journal-required format

For questions or custom diagrams, refer to `create_model_diagrams.py` or the Mermaid templates above.
