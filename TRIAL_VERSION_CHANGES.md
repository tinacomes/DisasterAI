# Trial Version - Correct Fix: Remove Alignment Biases

## Critical Error Identified

**Problem:** The code was hardcoding alignment effects as biases, which:
- Tells agents a priori whether AI is truthful/biased
- Completely undermines Q-learning mechanism
- Prevents agents from discovering optimal strategies through experience

## Correct Solution: Let Q-Learning Work

### What Q-Learning Should Discover

**Exploratory Agents (80% accuracy rewards, 20% confirmation):**
- Low alignment AI → accurate info → high rewards → **LEARN** to prefer AI
- High alignment AI → biased info → low rewards → **LEARN** to avoid AI

**Exploitative Agents (20% accuracy rewards, 80% confirmation):**
- High alignment AI → confirms beliefs → high rewards → **LEARN** to prefer AI
- Low alignment AI → may not confirm → lower rewards → **LEARN** relative preferences

### What Was Wrong

#### BEFORE (Hardcoded Alignment Effects):

**Exploitative:**
```python
# Lines 929-936 - WRONG!
ai_alignment_factor = alignment * 0.3
scores[ai_id] += ai_alignment_factor
# Telling exploitative agents: "High alignment AI is good for you!"
```

**Exploratory:**
```python
# Lines 943-962 - WRONG!
inverse_alignment_factor = (1 - alignment) * 0.5 + 0.1  # baseline distortion
scores[ai_id] += inverse_alignment_factor
human_bias = alignment * 0.15
# Telling exploratory agents: "Low alignment AI is good, high alignment AI is bad!"
```

### AFTER (Let Q-Learning Discover):

**Exploitative:**
```python
# Intrinsic preferences only
scores["human"] += self.exploit_friend_bias      # Prefer friends (confirmation)
scores["self_action"] += self.exploit_self_bias  # Prefer self (confirmation)
# NO AI bias - Q-learning discovers if AI confirms through rewards
```

**Exploratory:**
```python
# Intrinsic preferences only
scores["self_action"] -= 0.05  # Avoid self-confirmation
# NO AI bias - Q-learning discovers accuracy through rewards
# NO human bias - Q-learning discovers best sources
```

## How This Should Work

### Learning Process

**Tick 1-50:** Agents explore randomly (epsilon-greedy), get rewards, update Q-values
- Exploratory at low alignment: AI accurate → positive rewards → Q(AI) increases
- Exploitative at high alignment: AI confirms → positive rewards → Q(AI) increases

**Tick 50-100:** Q-values stabilize, agents increasingly exploit learned preferences
- Behavior emerges from LEARNING, not from hardcoded biases

**Tick 100+:** Stable behavior reflecting learned optimal strategies

### Expected Emergent Behavior

**At Low Alignment (truthful AI):**
- Exploratory: High Q(AI) from accuracy rewards → prefer AI
- Exploitative: Moderate Q(AI) - accurate but may not confirm → mixed behavior

**At High Alignment (biased AI):**
- Exploratory: Low Q(AI) from inaccuracy penalties → avoid AI
- Exploitative: High Q(AI) from confirmation rewards → prefer AI

**Clear differentiation emerges from learning, not from a priori biases!**

## Why This Matters

### Scientific Validity
- Q-learning experiments must let agents LEARN from environment
- Hardcoded biases = researcher imposing results, not discovering them
- Emergent behavior from learning = valid scientific finding

### Interpretability
- "Agents learned to prefer X" >> "Agents were biased toward X"
- Shows adaptation to environment
- Demonstrates intelligence/learning capability

### Experimental Design
- Manipulate alignment → observe learning → measure behavioral outcomes
- NOT: Manipulate alignment → tell agents about it → observe hardcoded behavior

## Impact on Results

### Hypothesis Testing
Now we can properly test: *"Do agents learn to adapt their information-seeking based on AI characteristics?"*

**Testable predictions:**
1. Q-values for AI should increase with experience (learning occurs)
2. Q(AI) trajectories should differ by alignment level (environment matters)
3. Q(AI) differences should vary by agent type (reward structures matter)

### Metrics to Track
- **Q-value evolution:** Does Q(AI) change over time?
- **Behavioral crossover:** When do exploratory agents switch preferences?
- **Learning speed:** How fast do Q-values stabilize?

## Files Modified
- `DisasterAI_Model.py` (lines 921-941): Removed ALL alignment-based biases
- Both exploitative AND exploratory agent types corrected
