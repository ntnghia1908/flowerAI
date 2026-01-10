# 🐛 Bug Report - Implementation Issues in FedAvg & FedAvgM

**Date:** 2026-01-10
**Severity:** CRITICAL for FedAvgM, HIGH for FedAvg
**Status:** Identified, needs fixing

---

## 🔴 CRITICAL BUG #1: Missing `server_learning_rate` in FedAvgM

### **Issue:**
FedAvgM strategy is **NOT receiving** the `server_learning_rate` parameter, only `server_momentum`.

### **Evidence:**

**Config file (FedAvgM_homo_npy.toml):**
```toml
server-momentum = 0.9
server-learning-rate = 0.5  # ← This is defined
```

**Server reads it (server_app_experiment.py:148-149):**
```python
if strategy_name == "FedAvgM":
    strategy_params["server_momentum"] = context.run_config.get("server-momentum", 0.9)
    strategy_params["server_learning_rate"] = context.run_config.get("server-learning-rate", 0.5)
```

**BUT strategies.py only passes `server_momentum` (line 61-68):**
```python
elif strategy_name == "FedAvgM":
    # FedAvg with server-side momentum
    server_momentum = kwargs.get("server_momentum", 0.9)
    return FedAvgMWithMetricsAggregation(
        evaluate_metrics_aggregation_fn=evaluate_metrics_agg_fn,
        **common_params,
        server_momentum=server_momentum  # ❌ Missing server_learning_rate!
    )
```

**FedAvgM signature requires both:**
```python
FedAvgM.__init__(
    ...,
    server_learning_rate: float = 1.0,  # ← Default is 1.0!
    server_momentum: float = 0.0
)
```

### **Impact:**

**🔴 CATASTROPHIC:** FedAvgM is using `server_learning_rate=1.0` (default) instead of `0.5` (configured)!

This explains the model collapse:
- **Actual:** `server_lr = 1.0`, `momentum = 0.9`
- **Intended:** `server_lr = 0.5`, `momentum = 0.9`
- **Effect:** 2x learning rate → massive overshoot → divergence → collapse

### **Reproduction:**

All 9 FedAvgM experiments used:
```
server_learning_rate = 1.0  (default, BUG)
server_momentum = 0.9       (correct)
```

This causes:
1. Momentum accumulates updates
2. Server LR=1.0 amplifies accumulated momentum 2x
3. Overshooting → gradient explosion → model collapse

### **Fix:**

```python
# In pytorchexample/strategies.py line 61-68
elif strategy_name == "FedAvgM":
    # FedAvg with server-side momentum
    server_momentum = kwargs.get("server_momentum", 0.9)
    server_learning_rate = kwargs.get("server_learning_rate", 0.5)  # ← ADD THIS
    return FedAvgMWithMetricsAggregation(
        evaluate_metrics_aggregation_fn=evaluate_metrics_agg_fn,
        **common_params,
        server_momentum=server_momentum,
        server_learning_rate=server_learning_rate  # ← ADD THIS
    )
```

---

## ⚠️ HIGH PRIORITY ISSUE #2: No Learning Rate Decay

### **Issue:**
Neither FedAvg nor FedAvgM implement learning rate decay/scheduling.

### **Evidence:**

**Client-side training (task.py:169-216):**
```python
def train(net, trainloader, epochs, lr, device, proximal_mu=0.0, global_params=None):
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    # ← No LR scheduler!

    for _ in range(epochs):  # ← Fixed LR for all 500 rounds
        for batch in trainloader:
            # training loop
```

**Server-side (server_app_experiment.py:187-193):**
```python
result = strategy.start(
    grid=grid,
    initial_arrays=arrays,
    train_config=ConfigRecord({"lr": lr}),  # ← Same LR every round
    num_rounds=num_rounds,
    evaluate_fn=evaluate_fn_wrapper,
)
```

### **Impact:**

**⚠️ HIGH:** This explains post-peak degradation in FedAvg:

**Observed pattern:**
```
FedAvg_homo:
Round 60:  63.75% accuracy, loss=1.189  (peak)
Round 500: 59.10% accuracy, loss=7.079  (degraded)
```

**Root cause:**
1. Early rounds (R1-60): Large LR (0.01) → fast convergence
2. Peak reached at R60
3. Later rounds (R61-500): LR still 0.01 → overshooting near optimum
4. Result: Loss explosion (1.19 → 7.08), accuracy drop (-4.65%)

### **Common in FL:**
- FedAvg paper recommends LR decay
- Typical schedule: decay by 0.5-0.9 every 100-200 rounds
- Or use cosine annealing

### **Fix Options:**

**Option 1: Step decay (simple)**
```python
# In client_app_experiment.py train() function
current_lr = lr
if server_round > 100:
    current_lr = lr * 0.5
if server_round > 300:
    current_lr = lr * 0.25
```

**Option 2: Exponential decay**
```python
current_lr = lr * (0.995 ** server_round)
```

**Option 3: Cosine annealing**
```python
import math
current_lr = lr * (1 + math.cos(math.pi * server_round / num_rounds)) / 2
```

---

## ⚠️ MEDIUM ISSUE #3: No Early Stopping

### **Issue:**
Experiments continue for full 500 rounds even after peak performance.

### **Evidence:**

**Many experiments peak early:**
```
FedAvg_homo:    Peak R60  → 440 wasted rounds
FedAvg_C5:      Peak R16  → 484 wasted rounds
FedAvgM_homo:   Peak R39  → 461 rounds of degradation
FedAvgM_C3:     Peak R51  → 449 rounds of collapse
```

### **Impact:**

1. **Wasted compute:** 80-95% of compute spent after peak
2. **Degradation:** Post-peak training hurts performance
3. **Time:** Could save 4-5 days of experiment time

### **Fix:**

**Add early stopping in server_app_experiment.py:**

```python
# Add to global variables
best_accuracy = 0.0
rounds_without_improvement = 0
patience = 50  # Stop after 50 rounds without improvement

def global_evaluate(...):
    global best_accuracy, rounds_without_improvement

    # ... existing evaluation code ...

    current_accuracy = metrics['accuracy']

    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy
        rounds_without_improvement = 0
    else:
        rounds_without_improvement += 1

    # Signal to stop early
    if rounds_without_improvement >= patience:
        print(f"\n⚠️ Early stopping at round {server_round}")
        print(f"Best accuracy: {best_accuracy:.4f}")
        # Set flag to stop training
        # (Requires modification to strategy.start() loop)
```

---

## ⚠️ MEDIUM ISSUE #4: Client-side Momentum Not Configurable

### **Issue:**
Client-side SGD uses hardcoded `momentum=0.9`.

### **Evidence:**

**In task.py:183:**
```python
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)  # ← Hardcoded
```

### **Impact:**

- Cannot test different client momentum values
- Client momentum (0.9) + Server momentum (0.9 in FedAvgM) = double momentum
- May contribute to FedAvgM instability

### **Recommendation:**

Make client momentum configurable:
```python
# In config files
client-momentum = 0.9

# In task.py
def train(net, trainloader, epochs, lr, device, client_momentum=0.9, ...):
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=client_momentum)
```

---

## ℹ️ LOW ISSUE #5: Loss Function Not Configurable

### **Issue:**
CrossEntropyLoss is hardcoded, cannot experiment with label smoothing or other losses.

### **Current:**
```python
criterion = torch.nn.CrossEntropyLoss().to(device)
```

### **Better:**
```python
# For better generalization, especially with non-IID
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
```

### **Impact:** Minor, but label smoothing could help with non-IID data.

---

## 🔧 Priority Fix List

### **Must Fix (Before Re-running):**

1. **🔴 FedAvgM `server_learning_rate` bug** (5 min fix)
   - Add parameter passing in strategies.py
   - Re-run all 9 FedAvgM experiments

2. **⚠️ Learning rate decay** (30 min implementation)
   - Add LR schedule to client training
   - Test with step decay or cosine annealing
   - Re-run FedAvg to verify improvement

### **Should Fix (For Better Results):**

3. **⚠️ Early stopping** (1 hour implementation)
   - Save compute time (4-5 days)
   - Prevent post-peak degradation
   - Implement patience-based stopping

### **Nice to Have:**

4. **⚠️ Configurable client momentum** (15 min)
5. **ℹ️ Label smoothing** (5 min)

---

## 📊 Expected Impact After Fixes

### **FedAvgM with Correct LR:**

**Before (LR=1.0, buggy):**
```
homo:  57.74% → 40.72%  (collapse)
C4:    60.36% → 12.29%  (catastrophic)
Average: 25.69% final
```

**After fix (LR=0.5, correct):**
```
Expected: 55-60% final accuracy
No collapse expected
Comparable to FedAvg or slightly better
```

### **FedAvg with LR Decay:**

**Before (fixed LR=0.01):**
```
homo:  Peak 63.75% → Final 59.10%  (-4.65%)
C4:    Peak 76.57% → Final 75.32%  (-1.25%)
```

**After LR decay:**
```
Expected: Maintain peak accuracy
homo:  ~63-64% final
C4:    ~76-77% final
No loss explosion
```

### **With Early Stopping:**

**Time savings:**
```
Current: 72 experiments × 2 hours = 144 hours (6 days)
With early stopping (avg 200 rounds): ~60 hours (2.5 days)
Savings: 84 hours (3.5 days)
```

---

## 🎯 Verification Plan

### **Step 1: Fix FedAvgM Bug**
```bash
# Edit strategies.py (add server_learning_rate)
# Test with 3-round config
flwr run configs/FedAvgM_homo_npy_test.toml

# Verify in logs:
# - server_learning_rate should be 0.5
# - No collapse expected
```

### **Step 2: Add LR Decay**
```python
# Implement in client_app_experiment.py or task.py
# Test with FedAvg_homo (3 rounds)
```

### **Step 3: Re-run FedAvgM**
```bash
# After fix, re-run all 9 FedAvgM experiments
python run_all_experiments.py
# Filter for FedAvgM only
```

### **Step 4: Compare Results**
```bash
python verify_results.py
# Compare:
# - Old FedAvgM: 25.69% average
# - New FedAvgM: Expected 55-60% average
```

---

## 📚 References

**FedAvg Paper (McMahan et al., 2017):**
- Recommends learning rate decay
- Suggests momentum on both client and server

**FedAvgM (Original):**
- `server_lr` typically 0.1-1.0
- `server_momentum` typically 0.9
- **Key:** Must tune both together

**Common FL Best Practices:**
- LR decay after convergence plateau
- Early stopping with patience=50-100
- Gradient clipping for stability

---

## ✅ Action Items

- [ ] Fix `server_learning_rate` bug in strategies.py
- [ ] Test fix with FedAvgM_homo_npy_test (3 rounds)
- [ ] Implement learning rate decay (choose schedule)
- [ ] Test LR decay with FedAvg_homo (3 rounds)
- [ ] Implement early stopping (optional but recommended)
- [ ] Re-run all FedAvgM experiments (9 experiments)
- [ ] Compare old vs new FedAvgM results
- [ ] Update analysis report with new findings
- [ ] Consider re-running FedAvg with LR decay

---

**Report End**

*Next: Implement fixes and verify improvements*
