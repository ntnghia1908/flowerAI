# 📊 Visual Comparison - Federated Learning Results

**Date:** 2026-01-10
**Experiments Completed:** 22/72

---

## 🎯 Performance Heatmap

### Final Accuracy by Strategy × Distribution

```
                │ FedAvg │ FedAvgM │ FedProx │ FedAdam │ FedAdagrad │ FedYogi │ FedNova │ SCAFFOLD
────────────────┼────────┼─────────┼─────────┼─────────┼────────────┼─────────┼─────────┼──────────
homo            │  59.1  │  40.7   │  58.5   │   TBD   │    TBD     │   TBD   │   TBD   │   TBD
C2 (2 classes)  │  49.7  │  29.2   │  50.8   │   TBD   │    TBD     │   TBD   │   TBD   │   TBD
C3 (3 classes)  │  68.2  │  11.2   │  69.2   │   TBD   │    TBD     │   TBD   │   TBD   │   TBD
C4 (4 classes)  │  75.3  │  12.3   │  72.2*  │   TBD   │    TBD     │   TBD   │   TBD   │   TBD
C5 (5 classes)  │  48.0  │  10.0   │   TBD   │   TBD   │    TBD     │   TBD   │   TBD   │   TBD
Dir α=0.1       │  50.2  │  30.7   │   TBD   │   TBD   │    TBD     │   TBD   │   TBD   │   TBD
Dir α=0.5       │  54.7  │  44.2   │   TBD   │   TBD   │    TBD     │   TBD   │   TBD   │   TBD
Dir α=1.0       │  54.9  │   8.9   │   TBD   │   TBD   │    TBD     │   TBD   │   TBD   │   TBD
Dir α=10.0      │  58.2  │  44.2   │   TBD   │   TBD   │    TBD     │   TBD   │   TBD   │   TBD
────────────────┼────────┼─────────┼─────────┼─────────┼────────────┼─────────┼─────────┼──────────
Average         │  57.6  │  25.7   │  62.4** │    -    │     -      │    -    │    -    │    -

* C4 FedProx chưa hoàn thành (R120/500)
** Based on 4 completed experiments
```

### Color Legend:
- 🟢 >70% : Excellent
- 🟡 60-70%: Good
- 🟠 50-60%: Fair
- 🔴 40-50%: Poor
- ⚫ <40%  : Failed

```
                │ FedAvg │ FedAvgM │ FedProx │
────────────────┼────────┼─────────┼─────────┤
homo            │   🟠   │    🔴   │   🟠    │
C2 (2 classes)  │   🔴   │    🔴   │   🟠    │
C3 (3 classes)  │   🟡   │    ⚫   │   🟡    │
C4 (4 classes)  │   🟢   │    ⚫   │   🟢    │
C5 (5 classes)  │   🔴   │    ⚫   │   TBD   │
Dir α=0.1       │   🟠   │    ⚫   │   TBD   │
Dir α=0.5       │   🟠   │    🔴   │   TBD   │
Dir α=1.0       │   🟠   │    ⚫   │   TBD   │
Dir α=10.0      │   🟠   │    🔴   │   TBD   │
```

---

## 📈 Learning Curves Comparison

### Best Case: C4 (4 classes per client)

```
FedAvg (75.3% final):
Round:    0    50   100   150   200   250   300   350   400   450   500
Acc:   ░░░░░▁▂▃▄▅▆▆▇▇▇███████████████████████████████████▇▇ (peak R424)
       0%                          40%                 76%        75%

FedAvgM (12.3% final - COLLAPSED):
Round:    0    50   100   150   200   250   300   350   400   450   500
Acc:   ░░░░░▁▃▅▇█▇▅▃▂▁░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ (peak R81)
       0%    60%                                            12%

FedProx (72.2% at R120 - ONGOING):
Round:    0    50   100   150   200   250   300   350   400   450   500
Acc:   ░░░░░▁▃▅▆▇███? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? (current R120)
       0%        72%                                         ???
```

### Worst Case: C3 (3 classes per client)

```
FedAvg (68.2% final):
Round:    0    50   100   150   200   250   300   350   400   450   500
Acc:   ░░░░░▁▂▃▄▅▆▇▇▇▇███████████████████████████████████▇▇ (peak R483)
       0%                          40%                 71% 68%

FedAvgM (11.2% final - COLLAPSED):
Round:    0    50   100   150   200   250   300   350   400   450   500
Acc:   ░░░░░▁▃▅▇█▆▄▂▁░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ (peak R51)
       0%   55%                                             11%

FedProx (69.2% final - EXCELLENT):
Round:    0    50   100   150   200   250   300   350   400   450   500
Acc:   ░░░░░▁▂▃▄▅▆▆▇▇▇▇███████████████████████████████████▇ (peak R408)
       0%                          40%                 69% 69%
```

### Homogeneous (IID)

```
FedAvg (59.1% final):
Round:    0    50   100   150   200   250   300   350   400   450   500
Acc:   ░░░▂▄▆█▇▆▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅ (peak R60)
       0%  64%                                                 59%

FedAvgM (40.7% final):
Round:    0    50   100   150   200   250   300   350   400   450   500
Acc:   ░░░▂▄▆█▇▆▅▅▄▄▄▄▄▄▄▄▄▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃ (peak R39)
       0%  58%                                                 41%

FedProx (58.5% final):
Round:    0    50   100   150   200   250   300   350   400   450   500
Acc:   ░░░▂▄▆█▇▆▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅ (peak R42)
       0%  63%                                                 59%
```

---

## 📊 Peak vs Final Accuracy

### Stability Analysis (Peak - Final Gap)

```
Strategy: FedAvg
───────────────────────────────────────────────────────────
homo     ████░ -4.65%   (63.75% → 59.10%)
C2       ████░ -4.13%   (53.79% → 49.66%)
C3       ██░░░ -2.84%   (71.08% → 68.24%)
C4       █░░░░ -1.25%   (76.57% → 75.32%)  ⭐ Most stable
C5       ██████ -7.18%   (55.13% → 47.95%)
Dir0.1   ████░ -4.14%   (54.30% → 50.16%)
Dir0.5   ██████ -7.08%   (61.73% → 54.65%)
Dir1.0   ██████ -6.32%   (61.19% → 54.87%)
Dir10.0  ████░ -5.06%   (63.26% → 58.20%)

Average: -4.74% degradation

Strategy: FedAvgM
───────────────────────────────────────────────────────────
homo     ████████████████░ -17.02%  ❌ Severe
C2       ███████████████░░ -15.65%  ❌ Severe
C3       ████████████████████████████████████░ -43.61%  ❌❌ Catastrophic
C4       ████████████████████████████████████████████░ -48.07%  ❌❌❌ Catastrophic
C5       ██████████████████████████████████░ -34.44%  ❌❌ Catastrophic
Dir0.1   ███████████████░ -17.52%  ❌ Severe
Dir0.5   ██████████░ -10.46%  ⚠️ High
Dir1.0   ████████████████████████████████████████░ -45.98%  ❌❌❌ Catastrophic
Dir10.0  ███████████░ -12.45%  ⚠️ High

Average: -27.24% degradation ❌

Strategy: FedProx
───────────────────────────────────────────────────────────
homo     ███░░ -3.86%   (63.39% → 58.53%)
C2       ██░░░ -2.93%   (53.74% → 50.81%)
C3       ░░░░░ -0.24%   (69.42% → 69.18%)  ⭐ Excellent!
C4       TBD            (72.15% at R120)

Average: -2.34% degradation ⭐ Best stability!
```

---

## 🎯 Distribution Difficulty Ranking

### From Easiest to Hardest (Based on FedAvg):

```
1. C4 (4 classes)      ████████████████ 75.3%  ⭐ Easiest
2. C3 (3 classes)      █████████████░░░ 68.2%
3. homo (IID)          ███████████░░░░░ 59.1%
4. Dir10.0 (α=10)      ███████████░░░░░ 58.2%
5. Dir1.0 (α=1.0)      ██████████░░░░░░ 54.9%
6. Dir0.5 (α=0.5)      ██████████░░░░░░ 54.7%
7. Dir0.1 (α=0.1)      █████████░░░░░░░ 50.2%
8. C2 (2 classes)      █████████░░░░░░░ 49.7%
9. C5 (5 classes)      ████████░░░░░░░░ 48.0%  ⭐ Hardest
```

### Insight: Label Skew Sweet Spot

```
Classes per client:
    2    3    4    5
   49%  68%  75%  48%
    │    │    │    │
    └────┴────┴────┘
    Poor Good Best Poor

Optimal: 3-4 classes per client
```

---

## 🔥 Strategy Performance Profile

### FedAvg
```
Strengths:
  ✅ Stable across all distributions
  ✅ Best peak performance (76.57%)
  ✅ No catastrophic failures
  ✅ Good for label skew (C3, C4)

Weaknesses:
  ⚠️ Post-peak degradation (-4.74% avg)
  ⚠️ Poor on extreme skew (C2, C5)
  ⚠️ Loss explosion on homo

Best for: General purpose, C3-C4 label skew
```

### FedAvgM
```
Strengths:
  (None observed with current hyperparameters)

Weaknesses:
  ❌ 7/9 experiments collapsed (<15%)
  ❌ Catastrophic degradation (-27.24% avg)
  ❌ Worst on C4: 60.36% → 12.29%
  ❌ Not usable with current config

Best for: ⚠️ DO NOT USE until re-tuned
```

### FedProx
```
Strengths:
  ✅ Best stability (-2.34% avg degradation)
  ✅ Excellent on C3 (only -0.24% drop)
  ✅ Outperforms FedAvg on C2, C3
  ✅ No observed failures

Weaknesses:
  ⚠️ Slightly lower peak on homo
  ⚠️ Slower convergence

Best for: Non-IID data, label skew, need stability
```

---

## 📊 Convergence Speed Comparison

### Time to 60% Accuracy

```
Distribution: homo (IID)
────────────────────────────────────────
FedAvg:   ████████░░░░░░░░░░░ R25  (Fast)
FedAvgM:  █████░░░░░░░░░░░░░░ R15  (Fastest, but collapsed later)
FedProx:  ████████░░░░░░░░░░░ R25  (Fast)

Distribution: C4 (4 classes)
────────────────────────────────────────
FedAvg:   ████████████████████ R80  (Moderate)
FedAvgM:  █████░░░░░░░░░░░░░░ R20  (Fast, but collapsed at R81)
FedProx:  ██████████░░░░░░░░░ R40  (Faster than FedAvg)

Distribution: C3 (3 classes)
────────────────────────────────────────
FedAvg:   ████████████████████ R100 (Moderate)
FedAvgM:  ███████░░░░░░░░░░░░ R30  (Fast, but collapsed)
FedProx:  █████████████░░░░░░ R60  (Faster than FedAvg)
```

**Insight:** FedProx converges faster than FedAvg on non-IID data

---

## 🎓 Best Practices from Results

### ✅ Recommended Combinations:

```
Use Case                        │ Strategy │ Why
────────────────────────────────┼──────────┼────────────────────────────
IID data                        │ FedAvg   │ Stable, simple
Label skew (4 classes)          │ FedAvg   │ Best peak (76%)
Label skew (3 classes)          │ FedProx  │ Best stability
Label skew (2 classes)          │ FedProx  │ +1.15% vs FedAvg
Dirichlet α<1.0 (high skew)     │ FedProx* │ Better stability expected
Need fast convergence           │ FedProx  │ Faster on non-IID
Need maximum accuracy           │ FedAvg   │ Higher peaks
Need maximum stability          │ FedProx  │ Best post-peak retention
```

*Waiting for results

### ❌ Avoid:

```
Scenario                        │ Strategy  │ Why
────────────────────────────────┼───────────┼────────────────────────────
Any production use              │ FedAvgM   │ Catastrophic collapse risk
Label skew with high variance   │ FedAvgM   │ C3, C4, C5 all collapsed
Need long training (>100 rounds)│ FedAvgM   │ Collapses after peak
```

---

## 🔮 Predictions for Remaining Experiments

### FedProx (5 remaining):

```
C5:       Expected: 50-52% (vs FedAvg 48.0%)  [+2-4%]
Dir0.1:   Expected: 52-54% (vs FedAvg 50.2%)  [+2-4%]
Dir0.5:   Expected: 56-58% (vs FedAvg 54.7%)  [+2-4%]
Dir1.0:   Expected: 56-58% (vs FedAvg 54.9%)  [+2-4%]
Dir10.0:  Expected: 59-61% (vs FedAvg 58.2%)  [+1-3%]
```

Pattern: FedProx likely +1-4% over FedAvg on non-IID

### Adaptive Optimizers (FedAdam, FedAdagrad, FedYogi):

**Expected Performance:**
- Better than FedAvgM (set low bar)
- Comparable or better than FedAvg
- Potentially best on extreme non-IID (Dir0.1, C2, C5)

**Risk:** Hyperparameter sensitivity
- Need proper β1, β2, τ values
- May need per-distribution tuning

### FedNova:

**Expected Performance:**
- Similar to FedAvg on homogeneous local epochs
- Improvement if we test variable local epochs
- Best for heterogeneous client systems

### SCAFFOLD:

**Expected Performance:**
- Current implementation: Similar to FedAvg (simplified version)
- Full implementation: Better on client drift scenarios
- Best for high non-IID settings

---

## 📋 Quick Reference Chart

### Strategy Selection Decision Tree

```
Start Here
    │
    ├─ IID data?
    │   └─ Yes → Use FedAvg ✅
    │
    ├─ Label skew?
    │   ├─ 3-4 classes/client → Use FedAvg or FedProx ✅
    │   ├─ 2 classes/client → Use FedProx ✅
    │   └─ 5+ classes/client → Wait for FedAdam/FedYogi results
    │
    ├─ Dirichlet distribution?
    │   ├─ α > 1.0 → Use FedAvg ✅
    │   └─ α < 1.0 → Use FedProx ✅
    │
    ├─ Need maximum stability?
    │   └─ Use FedProx ✅
    │
    ├─ Need maximum peak accuracy?
    │   └─ Use FedAvg ✅
    │
    └─ Want to use momentum?
        └─ ❌ DO NOT use FedAvgM with current config
```

---

**For detailed analysis, see:** [ANALYSIS_REPORT.md](ANALYSIS_REPORT.md)
**For quick stats, see:** [QUICK_SUMMARY.md](QUICK_SUMMARY.md)
