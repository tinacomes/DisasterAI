# Belief Update Analysis: Paper vs Current Implementation

## The Reference Paper

**Geschke, Lorenz, & Holtz (2019). "The triple-filter bubble: Using agent-based modelling to test a meta-theoretical framework for the emergence of filter bubbles and echo chambers." British Journal of Social Psychology.**

- DOI: [10.1111/bjso.12286](https://doi.org/10.1111/bjso.12286)
- PMC: [PMC6585863](https://pmc.ncbi.nlm.nih.gov/articles/PMC6585863/)
- Code: [github.com/janlorenz/TripleFilterBubble](https://github.com/janlorenz/TripleFilterBubble)

---

## Paper's Model

### 1. Acceptance Decision (Integration Probability)

The paper uses this formula:

```
P(d) = D^δ / (d^δ + D^δ)
```

Where:
- **D** = latitude of acceptance (default: 0.3 on a [-1,1] scale)
- **δ** = sharpness parameter (default: 20)
- **d** = distance between agent's opinion and the info-bit
- **P(d)** = probability of integrating (accepting) the information

**Key properties:**
- When d = D: P(d) = 0.5 (50% acceptance)
- When d < D: P(d) > 0.5 (likely accept)
- When d > D: P(d) < 0.5 (likely reject)
- When δ → ∞: Becomes deterministic (accept if d < D, reject if d > D)

### 2. Opinion Update (After Acceptance)

The paper uses a **MEMORY-BASED AVERAGING** model:

```
Agent opinion = MEAN(all info-bits in memory)
```

- Agents have LIMITED MEMORY (can hold N info-bits)
- When memory is full, FORGET oldest info-bit, ADD new one
- Opinion = simple arithmetic mean of all stored info-bits
- **NO weighted averaging based on trust**
- **NO Bayesian updating**

### 3. Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| D | 0.3 | Latitude of acceptance (opinion scale -1 to +1) |
| δ | 20 | Sharpness (higher = more binary) |
| Memory size | varies | Number of info-bits agent can hold |

---

## Current Implementation

### 1. Acceptance Decision (Lines 1177-1203)

```python
level_diff = abs(reported_level - prior_level)
if self.agent_type == "exploitative" and prior_confidence > 0.4:
    if level_diff >= 3:
        rejection_prob = 0.9 * prior_confidence
    elif level_diff >= 2:
        rejection_prob = 0.7 * prior_confidence
    elif level_diff >= 1:
        rejection_prob = 0.3 * prior_confidence
```

**Problems:**
- D and delta are DEFINED but NOT USED
- Hardcoded thresholds (3, 2, 1) instead of D
- Hardcoded probabilities (0.9, 0.7, 0.3) instead of formula
- Only applies to exploiters (explorers accept everything)

### 2. Opinion Update (Lines 1246-1283)

```python
posterior_precision = prior_precision + source_precision
posterior_level = (prior_precision * prior_level + source_precision * reported_level) / posterior_precision
posterior_confidence = posterior_precision / (1 + posterior_precision)
```

**This is Bayesian precision-weighted averaging:**
- prior_precision = f(confidence, agent_type)
- source_precision = f(trust, agent_type)
- Update is weighted by relative precisions
- NO memory mechanism - single belief per cell, continuously updated

### 3. Parameters (Defined but NOT USED)

```python
self.D = 2.0 if agent_type == "exploitative" else 4  # UNUSED!
self.delta = 3.5 if agent_type == "exploitative" else 1.2  # UNUSED!
```

---

## Key Differences

| Aspect | Paper | Current Code |
|--------|-------|--------------|
| Acceptance formula | P(d) = D^δ / (d^δ + D^δ) | Hardcoded rejection_prob |
| D parameter | Used in formula | Defined but NOT used |
| δ parameter | Sharpness in formula | Defined but NOT used |
| Opinion update | Simple mean of memory | Bayesian precision-weighted |
| Memory | Limited (N info-bits) | Single belief per cell |
| Trust weighting | None | Yes (in precision) |
| Agent-type difference | Same formula, different D | Different mechanisms |

---

## Proposed Fix: Integrate D and δ Properly

### Option A: Follow Paper Exactly (Memory-Based)

```python
def decide_acceptance(self, reported_level, prior_level):
    """Paper's integration probability formula."""
    d = abs(reported_level - prior_level)

    # Normalize d to [0, 1] scale (disaster levels are 0-5)
    d_normalized = d / 5.0

    # Paper formula: P(d) = D^δ / (d^δ + D^δ)
    if d_normalized == 0:
        return True  # Always accept identical info

    D = self.D / 5.0  # Normalize D to [0, 1] scale
    delta = self.delta

    p_accept = (D ** delta) / (d_normalized ** delta + D ** delta)
    return random.random() < p_accept

def update_belief_memory_based(self, cell, reported_level):
    """Paper's memory-based averaging."""
    if not hasattr(self, 'belief_memory'):
        self.belief_memory = {}

    if cell not in self.belief_memory:
        self.belief_memory[cell] = []

    # Add new info to memory
    self.belief_memory[cell].append(reported_level)

    # Enforce memory limit
    memory_size = 10  # Configurable
    if len(self.belief_memory[cell]) > memory_size:
        self.belief_memory[cell].pop(0)  # Forget oldest

    # Update belief to mean of memory
    new_level = int(round(np.mean(self.belief_memory[cell])))
    self.beliefs[cell] = {'level': new_level, 'confidence': len(self.belief_memory[cell]) / memory_size}
```

### Option B: Hybrid (D/δ for Acceptance, Bayesian for Update)

This keeps the current Bayesian update (which is reasonable) but fixes acceptance:

```python
def decide_acceptance(self, reported_level, prior_level, source_trust, source_id=None):
    """
    Integration probability using D and δ parameters.
    Incorporates trust and friend status as modifiers.
    """
    d = abs(reported_level - prior_level)

    # Normalize distance to [0, 1] scale
    d_normalized = d / 5.0

    if d_normalized == 0:
        return True  # Always accept identical info

    # Normalize D to [0, 1] scale
    # D=2 means accept when level_diff < 2 (on 0-5 scale), so D_norm = 2/5 = 0.4
    D_normalized = self.D / 5.0
    delta = self.delta

    # Paper formula
    p_accept = (D_normalized ** delta) / (d_normalized ** delta + D_normalized ** delta)

    # Modifiers (extensions beyond paper)
    # 1. Trust modifier: Higher trust increases acceptance probability
    trust_modifier = 0.5 + 0.5 * source_trust  # Range [0.5, 1.0]
    p_accept = p_accept * trust_modifier

    # 2. Friend modifier (for exploiters): More likely to accept from friends
    is_friend = source_id in self.friends if source_id else False
    if self.agent_type == "exploitative" and is_friend:
        p_accept = min(1.0, p_accept * 1.3)  # 30% boost for friends

    # 3. Confidence modifier: High confidence reduces acceptance of conflicting info
    if self.agent_type == "exploitative":
        prior_confidence = self.beliefs.get(cell, {}).get('confidence', 0.1)
        confidence_penalty = 1.0 - 0.3 * prior_confidence  # Range [0.7, 1.0]
        p_accept = p_accept * confidence_penalty

    return random.random() < p_accept
```

---

## Recommended Implementation

**Use Option B (Hybrid)** because:

1. The paper's acceptance formula (D/δ) is well-grounded theoretically
2. The current Bayesian update is reasonable for this domain (continuous belief about disaster levels)
3. The memory-based approach would require significant refactoring
4. Trust should influence updates (not in paper, but makes sense for AI alignment research)

### Recommended D and δ Values

Based on the paper's defaults (D=0.3 on [-1,1] scale, δ=20):

| Agent Type | D (0-5 scale) | δ | Interpretation |
|------------|---------------|---|----------------|
| Exploitative | 1.5 | 20 | Accept 50% when diff=1.5, steep rejection beyond |
| Exploratory | 3.0 | 8 | Accept 50% when diff=3.0, gradual rejection |

**Sharpness (δ) interpretation:**
- δ=20: Nearly binary (sharp cutoff at D)
- δ=8: More gradual transition
- δ=4: Very gradual (large acceptance region)

### Visualization of Acceptance Probability

```
D=1.5, δ=20 (Exploiter):        D=3.0, δ=8 (Explorer):
P(d)                             P(d)
1.0|****                         1.0|********
   |    *                           |        **
0.5|     *                       0.5|          **
   |      *                         |            **
0.0|       ******                0.0|              ****
   +------------------d             +------------------d
   0  1  2  3  4  5                 0  1  2  3  4  5
```

---

## Full Implementation Code

```python
def update_belief_bayesian(self, cell, reported_level, source_trust, source_id=None):
    """Update agent's belief about a cell using D/δ acceptance + Bayesian update."""
    try:
        if cell not in self.beliefs:
            self.beliefs[cell] = {'level': 0, 'confidence': 0.1}

        current_belief = self.beliefs[cell]
        prior_level = current_belief.get('level', 0)
        prior_confidence = max(0.1, current_belief.get('confidence', 0.1))

        # ============================================
        # STEP 1: ACCEPTANCE DECISION using D and δ
        # ============================================
        d = abs(reported_level - prior_level)
        d_normalized = d / 5.0  # Normalize to [0, 1]

        if d_normalized > 0:  # If info differs from prior
            D_normalized = self.D / 5.0
            delta = self.delta

            # Paper formula: P(d) = D^δ / (d^δ + D^δ)
            p_accept = (D_normalized ** delta) / (d_normalized ** delta + D_normalized ** delta)

            # Trust modifier
            trust_modifier = 0.5 + 0.5 * source_trust
            p_accept *= trust_modifier

            # Friend modifier (exploiters only)
            is_friend = source_id in self.friends if source_id else False
            if self.agent_type == "exploitative" and is_friend:
                p_accept = min(1.0, p_accept * 1.3)

            # Confidence modifier (exploiters only)
            if self.agent_type == "exploitative":
                confidence_penalty = 1.0 - 0.3 * prior_confidence
                p_accept *= confidence_penalty

            # Make acceptance decision
            if random.random() >= p_accept:
                # REJECT - track for feedback but don't update
                if source_id:
                    self.pending_info_evaluations.append((
                        self.model.tick, source_id, cell,
                        int(reported_level), int(prior_level), float(prior_confidence)
                    ))
                return False  # No belief change

        # ============================================
        # STEP 2: BELIEF UPDATE (Bayesian, if accepted)
        # ============================================
        # Convert confidence to precision
        if self.agent_type == "exploitative":
            prior_precision = 2.5 * prior_confidence / (1 - prior_confidence + 1e-6)
        else:
            prior_precision = 0.6 * prior_confidence / (1 - prior_confidence + 1e-6)

        # Source precision from trust
        if self.agent_type == "exploitative":
            source_precision = 4.0 * source_trust / (1 - source_trust + 1e-6)
        else:
            source_precision = 2.5 * source_trust / (1 - source_trust + 1e-6)

        # Cap source precision
        max_precision = 8.0 if self.agent_type == "exploitative" else 12.0
        source_precision = min(source_precision, max_precision)

        # Bayesian update
        posterior_precision = prior_precision + source_precision
        posterior_level = (prior_precision * prior_level + source_precision * reported_level) / posterior_precision
        posterior_confidence = posterior_precision / (1 + posterior_precision)

        # Constrain to valid ranges
        posterior_level = max(0, min(5, round(posterior_level)))
        posterior_confidence = max(0.1, min(0.98, posterior_confidence))

        # Update the belief
        self.beliefs[cell] = {
            'level': int(posterior_level),
            'confidence': posterior_confidence
        }

        # Track for feedback
        if source_id:
            self.pending_info_evaluations.append((
                self.model.tick, source_id, cell,
                int(reported_level), int(prior_level), float(prior_confidence)
            ))

        return abs(posterior_level - prior_level) >= 1 or abs(posterior_confidence - prior_confidence) >= 0.1

    except Exception as e:
        print(f"ERROR in update_belief_bayesian: {e}")
        return False
```

---

## Summary

| Component | Paper | Recommended |
|-----------|-------|-------------|
| **Acceptance** | P(d) = D^δ / (d^δ + D^δ) | Same + trust/friend modifiers |
| **D (exploiter)** | 0.3 (on 0-1 scale) | 1.5 (on 0-5 scale) = 0.3 normalized |
| **D (explorer)** | 0.3 or higher | 3.0 (on 0-5 scale) = 0.6 normalized |
| **δ (exploiter)** | 20 | 20 (sharp cutoff) |
| **δ (explorer)** | 20 | 8 (gradual) |
| **Update** | Memory mean | Bayesian precision-weighted |
| **Trust role** | None | Modifies acceptance prob & precision |
