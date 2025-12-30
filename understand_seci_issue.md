# SECI Shows Little Variation: Root Cause Analysis

## The Problem
You correctly observed: **AECI shifts tremendously with alignment (0.3 → 0.95+), but SECI barely changes.**

## Root Cause: Static Social Network

Looking at `DisasterAI_Model.py:2058-2156`, the social network is:

1. **Initialized ONCE** at the start via `initialize_social_network()`
2. **NEVER modified** during simulation (no rewiring, no edge changes)
3. **Independent of alignment parameter**

```python
def initialize_social_network(self):
    # Creates fixed communities with 70% connection probability
    # Network structure NEVER changes after this!
    for i in range(len(community)):
        for j in range(i+1, len(community)):
            if random.random() < p_within:
                self.social_network.add_edge(community[i], community[j])
    # NO dynamic rewiring based on beliefs, AI use, or anything else
```

## Why SECI Doesn't Vary with Alignment

**SECI Formula** (line 2464-2473):
```python
var_diff = friend_var - global_var
if var_diff < 0:  # Echo chamber
    seci_val = max(-1, var_diff / global_var)
```

**SECI measures**: Do your **friends** (network neighbors) have similar beliefs to you?

**The disconnect**:
- **Alignment affects**: AI quality → querying behavior → **AECI** ↑
- **Alignment DOESN'T affect**: Friend network structure → **SECI** ~constant
- You can query AI 95% of the time (high AECI) but **still have the same diverse friend network** from initialization!

## About the -0.1 Threshold

**Current usage** (echo_chamber_evolution.py:127-129):
```python
# Chamber duration: ticks where SECI < -0.1
seci_exploit_chamber_ticks = np.sum(seci_exploit_mean < -0.1)
```

**What -0.1 means**: Friend network has 10% lower belief variance than global population.

**Why it's problematic**:
1. **Arbitrary**: No theoretical justification for exactly 10%
2. **Ignores scale**: Same threshold used regardless of observed SECI range
3. **Binary**: Treats SECI of -0.09 and -0.01 identically (both "not chamber")

## Better Approaches

### Option 1: Data-Driven Thresholds
```python
# Use standard deviation from baseline
seci_baseline = np.mean(seci_values_at_tick_0)
seci_threshold = seci_baseline - np.std(seci_values_at_tick_0)

# Or use percentiles
seci_threshold = np.percentile(all_seci_values, 25)  # Bottom quartile = chamber
```

### Option 2: Continuous Metrics (No Threshold)
Instead of counting "chamber duration," track:
- **Peak absolute SECI**: Maximum echo chamber strength
- **Area under curve**: Total "echo chamber exposure"
- **Relative change**: How much SECI changes from baseline

### Option 3: Fix the Model (Add Dynamic Rewiring)
Make social network RESPONSIVE to behavior:
```python
def rewire_network_based_on_beliefs(self):
    """Agents preferentially connect to similar others."""
    for agent in self.humans.values():
        # Probability of severing edge with dissimilar friend
        for friend_id in agent.friends:
            belief_distance = calculate_belief_distance(agent, friend_id)
            if belief_distance > threshold and random.random() < p_rewire:
                # Drop dissimilar friend
                self.social_network.remove_edge(agent.node_id, friend_id)
                # Add similar friend
                similar_agent = find_similar_agent(agent)
                self.social_network.add_edge(agent.node_id, similar_agent.node_id)
```

This would make SECI **responsive to alignment** because:
- High alignment → agents trust AI → similar beliefs via AI → rewire to similar others → SECI ↓

## Recommended Next Steps

1. **Run diagnostic** to quantify actual SECI variation:
   ```bash
   python3 diagnose_seci.py
   ```

2. **Decision point**: Is the static network intentional?
   - If YES: Accept that SECI won't vary much, focus on other metrics
   - If NO: Implement dynamic rewiring to make network behavior-responsive

3. **Visualization fix**: Replace binary threshold with continuous metrics
   - Show peak SECI magnitude
   - Show SECI trajectory (already in evolution plots)
   - Remove arbitrary -0.1 threshold line

## Bottom Line

**The -0.1 threshold is arbitrary**, but more importantly:
**SECI can't be sensitive to alignment in the current model because the social network is static.**

The model separates:
- Information seeking (who you ASK) ← affected by alignment
- Social structure (who you're FRIENDS with) ← fixed at initialization

This may or may not be realistic depending on your research question!
