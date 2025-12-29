# Recommended Experiments for DisasterAI Model

Based on your research questions about filter bubbles and AI influence, here are the most important experiments to run.

---

## Priority 1: Core Experiments (MUST RUN)

### ✅ Experiment D: Learning Parameters (Already Running)
**Status**: Currently executing
**Purpose**: Understand how Q-learning speed affects filter bubble formation

**Parameters**:
```python
learning_rate_values = [0.03, 0.05, 0.07]
epsilon_values = [0.2, 0.3]
```

**Expected Insights**:
- Do faster learners (high LR) specialize more quickly?
- Does exploration rate (epsilon) prevent filter bubbles?

---

### 🔥 Experiment B: AI Alignment Tipping Points (HIGHLY RECOMMENDED)
**Status**: Currently commented out (lines 6876-6881)
**Purpose**: **This is your MAIN research question!**

**Why it's critical**:
- Explores when AI breaks vs. creates filter bubbles
- Identifies tipping points where agent behavior shifts
- Direct answer to "how does AI alignment affect filter bubbles?"

**How to activate**:
Uncomment lines 6876-6927 in DisasterAI_Model.py:

```python
# Around line 6876 - UNCOMMENT THIS SECTION:
alignment_values = [0.0, 0.25, 0.5, 0.75, 0.95]
param_name_b = "AI Alignment Tipping Point"
file_b_pkl = os.path.join(save_dir, f"results_{param_name_b.replace(' ','_')}.pkl")

print(f"\nRunning {param_name_b} Experiment...")
results_b = experiment_alignment_tipping_point(base_params, alignment_values, num_runs=10)

# Save results
with open(file_b_pkl, "wb") as f:
    pickle.dump(results_b, f)
```

**Parameters**:
```python
alignment_values = [0.0, 0.25, 0.5, 0.75, 0.95]
# Will automatically add fine-grained search if tipping point detected
```

**Expected Insights**:
- At what alignment level do exploratory agents stop trusting AI?
- When do exploitative agents prefer AI over friends?
- Is there a sharp tipping point or gradual transition?

**Visualization**: Will create plots showing:
- SECI vs alignment
- AECI vs alignment
- Trust evolution vs alignment
- Belief accuracy vs alignment

---

### 📊 Experiment A: Agent Mix Ratios (RECOMMENDED)
**Status**: Currently commented out (lines 6820-6870)
**Purpose**: Understand how population composition affects bubbles

**How to activate**:
Uncomment lines 6820-6870:

```python
# Around line 6820 - UNCOMMENT THIS SECTION:
share_values = [0.3, 0.5, 0.7]  # More variety
param_name_a = "Share Exploitative"
print("Running Experiment A...")
results_a = experiment_share_exploitative(base_params, share_values, num_runs)
```

**Parameters**:
```python
share_exploitative = [0.3, 0.5, 0.7]
# 30% = mostly exploratory (accuracy-seeking)
# 50% = balanced
# 70% = mostly exploitative (confirmation-seeking)
```

**Expected Insights**:
- Do exploitative majorities create stronger social bubbles?
- Can exploratory minorities break filter bubbles?
- Is there a critical mass for either strategy?

---

## Priority 2: Extended Experiments (OPTIONAL)

### ⚙️ Experiment C: Disaster Dynamics (Currently Active)
**Status**: Running (lines 6930-6987)
**Purpose**: Test robustness to environmental volatility

**Already configured**:
```python
dynamics_values = [1, 2, 3]  # Disaster evolution speed
shock_values = [1, 2, 3]     # Hotspot magnitude
```

**Expected Insights**:
- Do volatile disasters prevent filter bubble formation?
- Does uncertainty force agents to diversify sources?

**Note**: This is active but less critical than Experiment B for your research question.

---

## Priority 3: Novel Experiments (HIGHLY RECOMMENDED TO ADD)

### 🆕 Experiment E: Trust Initialization (NEW - ADD THIS!)
**Purpose**: Explore how initial AI trust affects long-term dynamics

**Add this code after Experiment D**:

```python
##############################################
# Experiment E: Initial AI Trust Levels
##############################################
def experiment_initial_trust(base_params, ai_trust_values, num_runs=20):
    results = {}
    for ai_trust in ai_trust_values:
        params = base_params.copy()
        params["initial_ai_trust"] = ai_trust
        print(f"Running initial_ai_trust = {ai_trust}")
        results[ai_trust] = aggregate_simulation_results(num_runs, params)
    return results

ai_trust_values = [0.2, 0.5, 0.8]  # Low, medium, high initial trust
results_e = experiment_initial_trust(base_params, ai_trust_values, num_runs)

# Save results
with open(os.path.join(save_dir, "results_experiment_E.pkl"), "wb") as f:
    pickle.dump(results_e, f)
```

**Why it matters**:
- Does starting with high AI trust create dependency?
- Can low initial trust be overcome by accurate AI?
- Interaction with alignment level

**Expected Insights**:
- Path dependency in AI adoption
- Trust recovery dynamics
- Initial conditions vs. long-term equilibrium

---

### 🆕 Experiment F: Rumor Intensity (NEW - ADD THIS!)
**Purpose**: Explore misinformation's role in filter bubble formation

**Add this code**:

```python
##############################################
# Experiment F: Rumor Probability
##############################################
def experiment_rumor_intensity(base_params, rumor_prob_values, num_runs=20):
    results = {}
    for rumor_prob in rumor_prob_values:
        params = base_params.copy()
        params["rumor_probability"] = rumor_prob
        print(f"Running rumor_probability = {rumor_prob}")
        results[rumor_prob] = aggregate_simulation_results(num_runs, params)
    return results

rumor_prob_values = [0.0, 0.3, 0.6]  # No rumors, moderate, high
results_f = experiment_rumor_intensity(base_params, rumor_prob_values, num_runs)

# Save results
with open(os.path.join(save_dir, "results_experiment_F.pkl"), "wb") as f:
    pickle.dump(results_f, f)
```

**Expected Insights**:
- Do rumors amplify social echo chambers?
- Can AI break rumor-induced bubbles?
- Interaction between rumors and AI alignment

---

### 🆕 Experiment G: AI-Human Ratio (NEW - ADVANCED)
**Purpose**: Explore how AI agent density affects influence

**Requires code modification**:

Currently, `num_ai = 5` is hardcoded. You'd need to make it configurable:

```python
# In DisasterModel.__init__, change:
self.num_ai = 5  # Fixed
# To:
self.num_ai = number_of_ai  # Parameter

# Then add parameter:
def __init__(self, ..., number_of_ai=5, ...):
```

**Then run experiment**:

```python
##############################################
# Experiment G: Number of AI Agents
##############################################
def experiment_ai_density(base_params, num_ai_values, num_runs=20):
    results = {}
    for num_ai in num_ai_values:
        params = base_params.copy()
        params["number_of_ai"] = num_ai
        print(f"Running number_of_ai = {num_ai}")
        results[num_ai] = aggregate_simulation_results(num_runs, params)
    return results

num_ai_values = [1, 3, 5, 10]  # Sparse to dense AI
results_g = experiment_ai_density(base_params, num_ai_values, num_runs)
```

**Expected Insights**:
- Does AI density create stronger echo chambers?
- Is there a saturation point?
- Competition vs. consensus among AI agents

---

## Recommended Execution Order

### Phase 1: Core Experiments (Week 1)
1. ✅ **Experiment D** (Learning Parameters) - Currently running
2. 🔥 **Experiment B** (AI Alignment Tipping Points) - **Run this next!**
3. 📊 **Experiment A** (Agent Mix Ratios)

### Phase 2: Validation (Week 2)
4. ⚙️ **Experiment C** (Disaster Dynamics) - Already active
5. 🆕 **Experiment E** (Initial Trust)

### Phase 3: Extensions (Week 3)
6. 🆕 **Experiment F** (Rumor Intensity)
7. 🆕 **Experiment G** (AI Density) - Requires code changes

---

## Quick Setup: Activate Experiment B Now

**Step 1**: Edit DisasterAI_Model.py

Find line 6876 and uncomment the block:

```python
# FROM:
    #alignment_values = [0.0, 0.25, 0.5, 0.75, 0.95]
    #param_name_b = "AI Alignment Tipping Point"
    # ... (rest commented)

# TO:
    alignment_values = [0.0, 0.25, 0.5, 0.75, 0.95]
    param_name_b = "AI Alignment Tipping Point"
    file_b_pkl = os.path.join(save_dir, f"results_{param_name_b.replace(' ','_')}.pkl")

    print(f"\nRunning {param_name_b} Experiment...")
    results_b = experiment_alignment_tipping_point(base_params, alignment_values, num_runs=10)

    with open(file_b_pkl, "wb") as f:
        pickle.dump(results_b, f)
```

**Step 2**: Set num_runs

For quick test:
```python
num_runs = 2  # Fast, line 6814
```

For publication:
```python
num_runs = 20  # Robust statistics
```

**Step 3**: Run in Colab

```python
# Just run the whole file
# It will execute Experiments C, D, and B (if uncommented)
```

---

## Expected Runtime

| Experiment | Configurations | Runs/Config | Total Runs | Time (estimated) |
|------------|---------------|-------------|------------|------------------|
| A (Share) | 3 | 20 | 60 | ~2 hours |
| B (Alignment) | 5-15* | 10 | 50-150 | ~2-6 hours |
| C (Dynamics) | 9 | 2 | 18 | ~30 min |
| D (Learning) | 6 | 2 | 12 | ~20 min |
| E (Trust) | 3 | 20 | 60 | ~2 hours |
| F (Rumor) | 3 | 20 | 60 | ~2 hours |
| G (AI Density) | 4 | 20 | 80 | ~3 hours |

*Experiment B adds fine-grained search if tipping point detected

**Parallelization**: If you have access to multiple cores or machines, run different parameter combinations in parallel.

---

## Publication Priority

For a strong publication, you MUST have:

1. **Experiment B** (AI Alignment) - This is your main story!
2. **Experiment D** (Learning Parameters) - Robustness check
3. **Experiment A** (Agent Mix) - Population dynamics

Nice to have:
4. **Experiment E** (Initial Trust) - Path dependency
5. **Experiment F** (Rumor) - Misinformation angle

Advanced/Future work:
6. **Experiment C** (Dynamics) - Environmental robustness
7. **Experiment G** (AI Density) - Scaling effects

---

## Data Analysis Checklist

After running experiments, analyze:

- [ ] **Tipping points**: Where do agent behaviors shift?
- [ ] **Agent type differences**: Do exploitative/exploratory differ?
- [ ] **SECI vs AECI**: Social vs AI bubbles - which dominates?
- [ ] **Specialization rates**: % of agents in AI-only vs friend-only
- [ ] **Belief accuracy**: Does AI improve or harm accuracy?
- [ ] **Trust evolution**: How does trust in AI vs friends change?
- [ ] **Acceptance patterns**: Use diagnostic_acceptance_patterns.py

---

## Output for Each Experiment

You should create:

1. **Bar charts**: Final metrics by parameter
2. **Time series**: Evolution of SECI, AECI, trust over time
3. **Heatmaps**: 2D parameter space visualization
4. **Scatter plots**: Agent-level heterogeneity
5. **Statistical tests**: ANOVA/t-tests for significance

Use the fix_experiment_d_plots.py as a template for other experiments.

---

## Questions to Answer

Your experiments should address:

1. **Main RQ**: Does AI break or create filter bubbles? (Experiment B)
2. **When**: At what alignment level does this happen? (Experiment B)
3. **Who**: Do exploratory agents respond differently? (All experiments)
4. **Why**: What mechanism drives this? (Q-learning, trust, belief updating)
5. **Robustness**: Does this hold across populations and environments? (Experiments A, C, E)

---

## Next Steps

1. **Fix Experiment D plots**: Run `python fix_experiment_d_plots.py`
2. **Activate Experiment B**: Uncomment lines 6876-6927
3. **Run Experiment B**: This is your MAIN contribution!
4. **Add Experiment E**: Copy code above for initial trust
5. **Analyze results**: Use diagnostic and plotting tools

Good luck! 🚀
