# Experiment B-bis: Heterogeneous AI Biases

## Research Question

**Do echo chambers persist when agents can self-select into different AI information sources with different systematic biases?**

## Hypothesis

With diverse AI biases and rumor-initialized belief diversity:
1. **Exploitative agents** will preferentially trust AIs that confirm their rumor-based beliefs
2. Different **social network components** will cluster around different AIs
3. **SECI will show strong variation** as friend networks maintain belief homogeneity while global population stays diverse
4. **Echo chambers will persist** throughout the simulation (not dissolve like in Experiment B)

## Current Problem (from Experiment B)

- All AIs have the same `alignment` parameter → uniform information environment
- Initial diversity (from rumors) erodes as all agents query identical AIs
- SECI shows minimal variation across alignment values
- Temporal dynamics show formation then dissolution

## Proposed Design

### AI Bias Structure

Create 5 AIs with **systematic directional biases**:

```python
AI_0: bias = -2.0  # Systematic underestimation (minimizer)
AI_1: bias = -1.0  # Mild underestimation
AI_2: bias =  0.0  # Unbiased/truthful
AI_3: bias = +1.0  # Mild overestimation
AI_4: bias = +2.0  # Systematic overestimation (alarmist)
```

**Bias implementation**:
```python
# In AIAgent.query_ai():
ground_truth = sensed_vals  # What AI actually senses
biased_response = ground_truth + self.bias  # Apply systematic bias
biased_response = np.clip(biased_response, 0, 5)  # Keep in valid range
```

**Key difference from alignment parameter**:
- `alignment`: Adjusts toward human beliefs (different for each query)
- `bias`: Systematic offset (same direction for all reports)

### Parameter Sweep

**Independent variable**: AI bias diversity level

**Option A - Bias Spread** (Recommended):
```python
bias_spread_values = [0.0, 0.5, 1.0, 1.5, 2.0]

# bias_spread = 0.0: All AIs unbiased [0, 0, 0, 0, 0]
# bias_spread = 1.0: Moderate diversity [-1, -0.5, 0, +0.5, +1]
# bias_spread = 2.0: High diversity [-2, -1, 0, +1, +2]
```

**Option B - Bias Polarization**:
```python
polarization_values = [0.0, 0.25, 0.5, 0.75, 1.0]

# polarization = 0.0: All neutral [0, 0, 0, 0, 0]
# polarization = 0.5: Moderate poles [-1, 0, 0, 0, +1]
# polarization = 1.0: Strong poles [-2, -1, 0, +1, +2]
```

### Control Variables

Keep constant to isolate bias effects:
- `ai_alignment_level = 0.0` (no confirmation bias, only systematic bias)
- `rumor_probability = 0.7` (high initial diversity)
- `share_exploitative = 0.6` (standard mix)
- All other params from base_params

### Metrics to Track

**Primary (existing)**:
- SECI (exploit and explor separately)
- AECI (overall AI usage)
- AECI-Var (belief variance among AI users)

**New metrics needed**:
- **AI preference distribution**: Which AI does each agent trust most?
- **AI clustering coefficient**: Do friends trust similar AIs?
- **Belief-AI correlation**: Do agents with high beliefs trust high-bias AIs?
- **Component-AI alignment**: Does each network component cluster around one AI?

### Expected Results

#### Prediction 1: AI Self-Selection
- Agents with high rumor beliefs → trust high-bias AIs (AI_4)
- Agents with low/no rumor → trust low-bias AIs (AI_0, AI_2)
- Exploitative agents show stronger clustering than exploratory

#### Prediction 2: Persistent Echo Chambers
- SECI remains negative throughout simulation (unlike Experiment B)
- Higher bias spread → stronger SECI (more persistent chambers)
- Friend networks maintain belief homogeneity via shared AI source

#### Prediction 3: Component Clustering
- Each social network component gravitates toward one AI
- Components with same rumor type cluster around same AI
- This creates measurable between-component belief divergence

#### Prediction 4: AECI Patterns
- AECI still high (agents query AI frequently)
- But WHICH AI they query differs by component
- AECI-Var shows larger variance with higher bias spread

## Implementation Steps

### 1. Modify AIAgent Class

Add individual bias parameter:
```python
class AIAgent(Agent):
    def __init__(self, unique_id, model, bias=0.0):
        super().__init__(unique_id, model)
        self.bias = bias  # Systematic bias for this AI

    def query_ai(self, cells_to_report_on, caller_id, caller_trust_in_ai):
        # ... existing sensing logic ...

        # Apply systematic bias INSTEAD of alignment-based adjustment
        biased_vals = sensed_vals + self.bias
        corrected = np.clip(biased_vals, 0, 5)

        # Return biased report
        return {cell: corrected[i] for i, cell in enumerate(cells_to_report_on)}
```

### 2. Modify DisasterModel Initialization

Assign biases during AI creation:
```python
def __init__(self, ..., ai_bias_spread=0.0, ...):
    self.ai_bias_spread = ai_bias_spread

    # Create AI agents with diverse biases
    self.ais = {}
    for k in range(self.num_ai):
        # Calculate bias for this AI
        bias = self._calculate_ai_bias(k, ai_bias_spread)
        ai_agent = AIAgent(unique_id=f"A_{k}", model=self, bias=bias)
        self.ais[f"A_{k}"] = ai_agent

def _calculate_ai_bias(self, ai_index, bias_spread):
    """Calculate systematic bias for AI based on index and spread."""
    # Distribute biases evenly from -bias_spread to +bias_spread
    # For 5 AIs: [-spread, -spread/2, 0, +spread/2, +spread]
    num_ais = self.num_ai
    position = ai_index / (num_ais - 1)  # 0.0 to 1.0
    bias = (position - 0.5) * 2 * bias_spread  # -spread to +spread
    return bias
```

### 3. Add New Metrics Collection

Track AI preference per agent:
```python
def collect_ai_preferences(self):
    """Track which AI each agent trusts most."""
    agent_ai_preferences = {}

    for agent_id, agent in self.humans.items():
        if not hasattr(agent, 'trust'):
            continue

        # Find AI with highest trust
        ai_trusts = {k: v for k, v in agent.trust.items() if k.startswith("A_")}
        if ai_trusts:
            preferred_ai = max(ai_trusts.items(), key=lambda x: x[1])[0]
            agent_ai_preferences[agent_id] = {
                'preferred_ai': preferred_ai,
                'trust_level': ai_trusts[preferred_ai],
                'belief_mean': np.mean([b['level'] for b in agent.beliefs.values()])
            }

    return agent_ai_preferences
```

### 4. Create Experiment Function

```python
def experiment_heterogeneous_ai_bias(base_params, bias_spread_values, num_runs=20):
    """
    Test echo chamber formation with heterogeneous AI biases.

    Args:
        base_params: Base simulation parameters
        bias_spread_values: List of bias spread values to test (e.g., [0.0, 0.5, 1.0, 2.0])
        num_runs: Number of runs per bias spread value

    Returns:
        dict: Results keyed by bias_spread value
    """
    results = {}

    for bias_spread in bias_spread_values:
        params = base_params.copy()
        params["ai_bias_spread"] = bias_spread
        params["ai_alignment_level"] = 0.0  # No confirmation bias, only systematic bias

        print(f"\n{'='*60}")
        print(f"Running ai_bias_spread = {bias_spread}")
        print(f"  AI biases: {[f'{(i/(4)-0.5)*2*bias_spread:.2f}' for i in range(5)]}")
        print(f"{'='*60}")

        result = aggregate_simulation_results(num_runs, params)
        results[bias_spread] = result

    return results
```

### 5. Create Visualizations

**Plot 1: AI Preference Heatmap**
- Rows: Agents (sorted by network component)
- Columns: Time ticks
- Color: Which AI agent trusts most
- Shows clustering patterns over time

**Plot 2: Belief-AI Correlation**
- X-axis: Agent's mean belief level
- Y-axis: Preferred AI's bias
- Color: Agent type (exploit vs explor)
- Shows if high-belief agents cluster to high-bias AIs

**Plot 3: SECI vs Bias Spread**
- Compare with Experiment B results
- Show that SECI increases with bias diversity
- Temporal plots show persistence vs dissolution

**Plot 4: Component-AI Clustering**
- Network diagram colored by preferred AI
- Shows if components cluster around specific AIs

## Testing Plan

### Phase 1: Small-Scale Validation
```python
test_params = base_params.copy()
test_params["ticks"] = 50  # Shorter run
test_params["ai_bias_spread"] = 2.0  # Maximum diversity

# Run 3 test simulations
results = aggregate_simulation_results(3, test_params)

# Verify:
# 1. Different AIs have different biases
# 2. Agents show preference patterns
# 3. Metrics are collected correctly
```

### Phase 2: Full Experiment
```python
bias_spread_values = [0.0, 0.5, 1.0, 1.5, 2.0]
num_runs = 20  # Full statistical power

results_b_bis = experiment_heterogeneous_ai_bias(
    base_params,
    bias_spread_values,
    num_runs
)

# Save results
save_path = "agent_model_results/results_experiment_B_bis.pkl"
with open(save_path, 'wb') as f:
    pickle.dump(results_b_bis, f)
```

## Expected Outcomes

### If Hypothesis Confirmed:

1. **SECI shows strong variation** across bias_spread values
2. **Echo chambers persist** (not dissolve) when bias_spread > 0
3. **Component clustering** visible in AI preference patterns
4. **Exploitative agents** show stronger clustering than exploratory

### If Hypothesis Rejected:

Possible reasons:
- Trust learning too slow to create stable preferences
- Exploratory behavior overrides confirmation bias
- Initial diversity still insufficient
- Need to combine with dynamic network rewiring

## Comparison with Experiment B

| Aspect | Experiment B | Experiment B-bis |
|--------|-------------|------------------|
| **IV** | AI alignment (0→1) | AI bias spread (0→2) |
| **AI diversity** | None (all identical) | High (5 different biases) |
| **Expected SECI** | Minimal variation | Strong increase with spread |
| **Echo persistence** | Form then dissolve | Persist if spread > 0 |
| **Mechanism** | Uniform convergence | Self-selection clustering |

## Next Steps After Results

### If successful:
1. Test interaction: bias_spread × alignment
2. Add dynamic network rewiring
3. Explore 3-AI polarization (left/center/right)
4. Test different bias distributions (normal vs uniform)

### If unsuccessful:
1. Increase rumor intensity for stronger initial diversity
2. Increase exploitative agent bias parameters
3. Reduce exploratory exploration rate
4. Test with static AI assignments (no learning)

## Files to Create/Modify

**New files**:
- `experiment_heterogeneous_ai.py` - Experiment runner
- `plot_ai_preferences.py` - Visualization functions

**Modified files**:
- `DisasterAI_Model.py`:
  - AIAgent: Add bias parameter
  - DisasterModel: Add ai_bias_spread parameter
  - collect_data: Add AI preference tracking
- `DisasterAI_Model.py` (experiments section):
  - Add experiment_heterogeneous_ai_bias function

**Analysis files**:
- Update `replot_results.py` to handle B-bis results
- Add to `diagnose_seci.py` for bias-specific diagnostics

## Computational Cost

- Similar to Experiment B (same number of ticks)
- 5 bias_spread values × 20 runs = 100 simulations
- Estimated time: ~2-3 hours on Colab (with GPU acceleration)
- Can reduce to 10 runs if needed

## Success Criteria

Experiment considered successful if:
1. ✓ SECI at bias_spread=2.0 is significantly more negative than bias_spread=0.0
2. ✓ Echo chambers persist past tick 100 when bias_spread > 1.0
3. ✓ AI preference clustering correlation > 0.5 (agents with similar beliefs prefer similar AIs)
4. ✓ Component-level AI preference shows > 60% consensus within components

This would demonstrate that **heterogeneous information environments** create persistent echo chambers through self-selection, even without network rewiring!
