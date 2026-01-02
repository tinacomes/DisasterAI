# Essential Experiments for Paper Completion

## 🎯 What You Need to Run

### **Must Have (Priority 1):**
1. ✅ **Experiment A** - Share Exploitative [0.2, 0.5, 0.8]
2. ✅ **Experiment B** - AI Alignment [0.0, 0.25, 0.5, 0.75, 1.0]
3. ✅ **Experiment D** - Q-Learning Sensitivity (robustness validation)

### **Nice to Have (Priority 2):**
4. ⭐ **Experiment E** - Noise & Rumors (for RQ5)

---

## 📋 Quick Setup Instructions

### Step 1: Open `DisasterAI_Model.py` in Colab

Upload the file to Colab or clone from GitHub:
```python
!git clone https://github.com/tinacomes/DisasterAI.git
%cd DisasterAI
!git checkout claude/plan-paper-experiments-0ESGp
```

### Step 2: Activate Experiment D

Find line ~7446 in `DisasterAI_Model.py` and **uncomment** Experiment D:

**Change this:**
```python
# COMMENTED OUT - Focus on Experiments A and B
# learning_rate_values = [0.03, 0.05, 0.07]
# epsilon_values = [0.2, 0.3]
```

**To this:**
```python
# ACTIVATED - Robustness validation
learning_rate_values = [0.05, 0.1, 0.15]
epsilon_values = [0.2, 0.3, 0.4]
results_d = experiment_learning_trust(base_params, learning_rate_values, epsilon_values, num_runs)

# Save results
file_d_pkl = os.path.join(save_dir, "results_experiment_D.pkl")
with open(file_d_pkl, "wb") as f:
    pickle.dump(results_d, f)
print(f"✓ Experiment D saved to: {file_d_pkl}")
```

### Step 3: Fix the typo in Experiment A parameters

Find line ~7314 and fix:

**Change this:**
```python
share_values = [0.2, 0,5, 0.8]  # Has comma instead of period!
```

**To this:**
```python
share_values = [0.2, 0.5, 0.8]  # Fixed!
```

### Step 4: Update save directory

Find line ~30 and ensure it points to the right place:

```python
if IN_COLAB:
    save_dir = "/content/drive/MyDrive/DisasterAI_results"  # lowercase 'results'!
else:
    save_dir = "DisasterAI_results"
```

### Step 5: Run in Colab

```python
# Just run the whole file!
%run DisasterAI_Model.py
```

---

## ⏱️ Expected Runtime

| Experiment | Parameters | Runs | Time |
|------------|-----------|------|------|
| A | 3 values | 10 | ~30 min |
| B | 5 values | 10 | ~90 min |
| D | 9 combos | 10 | ~60 min |
| **Total** | | | **~3 hours** |

💡 **Tip:** Start it before bed and let it run overnight!

---

## 📊 What You'll Get

After completion, you'll have in `/content/drive/MyDrive/DisasterAI_results/`:

```
results_experiment_A.pkl  # Share exploitative results
results_experiment_A.csv
results_experiment_B.pkl  # AI alignment results
results_experiment_D.pkl  # Q-learning sensitivity
```

Plus all the visualization PNGs!

---

## 🔬 Then Run Enhanced Analysis

Once experiments finish, run the analysis script:

```python
# Run analyze_existing_results.py to get:
# - SECI + AECI overlay plots
# - Net chamber strength metrics
# - Chamber dominance analysis
```

---

## ✅ Paper Readiness Checklist

After running these experiments, you'll have:

- [x] RQ1: Does AI Break Social Filter Bubbles? → **Experiment B**
- [x] RQ2: Does AI Create New Filter Bubbles? → **Experiment B**
- [x] RQ3: Tipping Points? → **Experiment B**
- [x] RQ4: Agent Type Differences? → **Experiment A**
- [ ] RQ5: Environmental Factors? → **Need Experiment E** (optional)
- [x] Robustness Validation → **Experiment D**

---

## 🚀 Ready to Submit!

With A + B + D, you have:
- Core findings
- Agent composition effects
- Robustness validation
- Strong academic paper!

**Experiment E (Noise/Rumors) can be added later if reviewers ask for it.**
