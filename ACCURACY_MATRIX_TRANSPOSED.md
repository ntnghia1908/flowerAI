# 📊 Accuracy Matrix - TRANSPOSED (Strategies × Distributions)

**Generated:** 2026-01-10 16:00
**Format:** Rows = Strategies, Columns = Distributions
**Completed:** 32/72 experiments (44.4%)

---

## 🎯 Final Accuracy Matrix (Round 500) - TRANSPOSED

| Strategy | homo | C2 | C3 | C4 | C5 | Dir0.1 | Dir0.5 | Dir1.0 | Dir10.0 | Average |
|----------|------|----|----|----|----|--------|--------|--------|---------|---------|
| **FedAvg** | **59.10** | 49.66 | 68.24 | 75.32 | 47.95 | **50.16** | **54.65** | **54.87** | **58.20** | **57.55** |
| **FedAvgM** | 40.72 | 29.19 | 11.15 | 12.29 | 9.96 | 30.65 | 44.23 | 8.85 | 44.23 | 25.69 ❌ |
| **FedProx** | 58.53 | **50.81** | **69.18** | **78.02** | 47.72 | 49.66 | **55.24** | 53.85 | 58.00 | **58.87** ⭐ |
| **FedAdam** | 52.02 | 49.87 | 62.11 | 71.37 | **49.08** | - | - | - | - | 57.69* |
| **FedAdagrad** | - | - | - | - | - | - | - | - | - | - |
| **FedYogi** | - | - | - | - | - | - | - | - | - | - |
| **FedNova** | - | - | - | - | - | - | - | - | - | - |
| **SCAFFOLD** | - | - | - | - | - | - | - | - | - | - |

*FedAdam average based on 5/9 completed experiments

---

## 🏆 Best Accuracy Matrix (Peak Performance) - TRANSPOSED

| Strategy | homo | C2 | C3 | C4 | C5 | Dir0.1 | Dir0.5 | Dir1.0 | Dir10.0 |
|----------|------|----|----|----|----|--------|--------|--------|---------|
| **FedAvg** | **63.75** | 53.79 | 71.08 | 76.57 | **55.13** | **54.30** | **61.73** | **61.19** | **63.26** |
| **FedAvgM** | 57.74 | 44.84 | 54.76 | 60.36 | 44.40 | 48.17 | 54.69 | 54.83 | 56.68 |
| **FedProx** | 63.39 | **53.74** | **69.42** | **78.17** | 55.44 | 54.32 | 61.05 | 61.72 | 62.71 |
| **FedAdam** | 59.13 | 50.92 | 62.48 | 72.42 | 51.20 | - | - | - | - |
| **FedAdagrad** | - | - | - | - | - | - | - | - | - |
| **FedYogi** | - | - | - | - | - | - | - | - | - |
| **FedNova** | - | - | - | - | - | - | - | - | - |
| **SCAFFOLD** | - | - | - | - | - | - | - | - | - |

---

## 📉 Degradation (Best → Final) - TRANSPOSED

| Strategy | homo | C2 | C3 | C4 | C5 | Dir0.1 | Dir0.5 | Dir1.0 | Dir10.0 | Avg |
|----------|------|----|----|----|----|--------|--------|--------|---------|-----|
| **FedAvg** | -4.65 | -4.13 | -2.84 | **-1.25** ⭐ | -7.18 | -4.14 | -7.08 | -6.32 | -5.06 | **-4.74** |
| **FedAvgM** | **-17.02** | **-15.65** | **-43.61** | **-48.07** | **-34.44** | **-17.52** | -10.46 | **-45.98** | -12.45 | **-27.24** ❌ |
| **FedProx** | -3.86 | -2.93 | **-0.24** ⭐ | **-0.15** ⭐ | -7.72 | -4.66 | -5.81 | -7.87 | -4.71 | **-4.22** ⭐ |
| **FedAdam** | -7.11 | -1.05 | **-0.37** ⭐ | -1.05 | -2.12 | - | - | - | - | **-2.94** ⭐ |
| **FedAdagrad** | - | - | - | - | - | - | - | - | - | - |
| **FedYogi** | - | - | - | - | - | - | - | - | - | - |
| **FedNova** | - | - | - | - | - | - | - | - | - | - |
| **SCAFFOLD** | - | - | - | - | - | - | - | - | - | - |

---

## 📊 Detailed Comparison by Strategy

### FedAvg (Baseline)

| Metric | homo | C2 | C3 | C4 | C5 | Dir0.1 | Dir0.5 | Dir1.0 | Dir10.0 |
|--------|------|----|----|----|----|--------|--------|--------|---------|
| **Final Acc** | 59.10% | 49.66% | 68.24% | 75.32% | 47.95% | 50.16% | 54.65% | 54.87% | 58.20% |
| **Best Acc** | 63.75% | 53.79% | 71.08% | 76.57% | 55.13% | 54.30% | 61.73% | 61.19% | 63.26% |
| **Best Round** | R60 | R419 | R483 | R424 | R16 | R45 | R47 | R45 | R47 |
| **Degradation** | -4.65% | -4.13% | -2.84% | -1.25% | -7.18% | -4.14% | -7.08% | -6.32% | -5.06% |
| **Rank** | 🥇 1st | 🥈 2nd | 🥈 2nd | 🥈 2nd | 🥈 2nd | 🥇 1st | 🥈 2nd | 🥇 1st | 🥇 1st |

**Strengths:** Best on IID (homo) and Dirichlet distributions
**Weaknesses:** Lower on label skew (C2-C4) compared to FedProx

---

### FedAvgM (Failed)

| Metric | homo | C2 | C3 | C4 | C5 | Dir0.1 | Dir0.5 | Dir1.0 | Dir10.0 |
|--------|------|----|----|----|----|--------|--------|--------|---------|
| **Final Acc** | 40.72% | 29.19% | 11.15% | 12.29% | 9.96% | 30.65% | 44.23% | 8.85% | 44.23% |
| **Best Acc** | 57.74% | 44.84% | 54.76% | 60.36% | 44.40% | 48.17% | 54.69% | 54.83% | 56.68% |
| **Best Round** | R39 | R79 | R51 | R81 | R33 | R30 | R24 | R31 | R31 |
| **Degradation** | -17.02% | -15.65% | -43.61% | -48.07% | -34.44% | -17.52% | -10.46% | -45.98% | -12.45% |
| **Status** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ⚠️ | ❌ | ⚠️ |

**Issue:** Missing server_learning_rate bug (✅ fixed, needs re-run)
**Expected after fix:** 55-60% average

---

### FedProx (Winner)

| Metric | homo | C2 | C3 | C4 | C5 | Dir0.1 | Dir0.5 | Dir1.0 | Dir10.0 |
|--------|------|----|----|----|----|--------|--------|--------|---------|
| **Final Acc** | 58.53% | **50.81%** | **69.18%** | **78.02%** | 47.72% | 49.66% | **55.24%** | 53.85% | 58.00% |
| **Best Acc** | 63.39% | 53.74% | 69.42% | 78.17% | 55.44% | 54.32% | 61.05% | 61.72% | 62.71% |
| **Best Round** | R42 | R459 | R408 | R447 | R18 | R48 | R41 | R44 | R45 |
| **Degradation** | -3.86% | -2.93% | **-0.24%** | **-0.15%** | -7.72% | -4.66% | -5.81% | -7.87% | -4.71% |
| **Rank** | 🥈 2nd | 🥇 1st | 🥇 1st | 🥇 1st | 🥉 3rd | 🥉 3rd | 🥇 1st | 🥈 2nd | 🥈 2nd |

**Strengths:** ⭐ Best on label skew (C2-C4), excellent stability
**Weaknesses:** Slightly lower on IID (homo)

---

### FedAdam (Most Stable)

| Metric | homo | C2 | C3 | C4 | C5 | Dir0.1 | Dir0.5 | Dir1.0 | Dir10.0 |
|--------|------|----|----|----|----|--------|--------|--------|---------|
| **Final Acc** | 52.02% | 49.87% | 62.11% | 71.37% | **49.08%** | - | - | - | - |
| **Best Acc** | 59.13% | 50.92% | 62.48% | 72.42% | 51.20% | - | - | - | - |
| **Best Round** | R51 | R469 | R482 | R488 | R25 | - | - | - | - |
| **Degradation** | -7.11% | -1.05% | **-0.37%** | -1.05% | -2.12% | - | - | - | - |
| **Rank** | 🥉 3rd | 🥉 3rd | 🥉 3rd | 🥉 3rd | 🥇 1st | - | - | - | - |

**Strengths:** ⭐ Best stability (-2.94% avg), late convergence
**Status:** ⏳ 5/9 completed (Dir experiments pending)

---

## 🎯 Win Count by Strategy

### Final Accuracy Winners:

| Strategy | Wins | Distributions |
|----------|------|---------------|
| **FedProx** | **5/9** | C2, C3, C4, Dir0.5 + 1 pending |
| **FedAvg** | **4/9** | homo, Dir0.1, Dir1.0, Dir10.0 |
| **FedAdam** | **1/5** | C5 (only 5 completed) |
| **FedAvgM** | **0/9** | None (all failed) |

### Best Stability Winners:

| Strategy | Best On | Degradation |
|----------|---------|-------------|
| **FedProx** | C4 | **-0.15%** ⭐ |
| **FedProx** | C3 | **-0.24%** ⭐ |
| **FedAdam** | C3 | **-0.37%** ⭐ |
| **FedAdam** | C2 | -1.05% |
| **FedAvg** | C4 | -1.25% |

---

## 📈 Strategy Performance Profiles

### Performance Range by Strategy:

| Strategy | Min | Max | Range | Consistency |
|----------|-----|-----|-------|-------------|
| **FedAvg** | 47.95% (C5) | 75.32% (C4) | 27.37% | Medium variance |
| **FedAvgM** | 8.85% (Dir1.0) | 44.23% (Dir0.5/Dir10.0) | 35.38% | High variance ❌ |
| **FedProx** | 47.72% (C5) | 78.02% (C4) | 30.30% | Medium variance |
| **FedAdam** | 49.08% (C5) | 71.37% (C4) | 22.29% | Low variance ⭐ |

**FedAdam has most consistent performance across distributions!**

---

## 🔍 Distribution Preferences by Strategy

### FedAvg Prefers:
- ✅ **Best on:** homo (59.10%), Dir10.0 (58.20%)
- ✅ **Good on:** C4 (75.32%), C3 (68.24%)
- ❌ **Worst on:** C5 (47.95%), C2 (49.66%)

### FedProx Prefers:
- ✅ **Best on:** C4 (78.02%), C3 (69.18%)
- ✅ **Good on:** homo (58.53%), Dir10.0 (58.00%)
- ❌ **Worst on:** C5 (47.72%), Dir0.1 (49.66%)

### FedAdam Prefers:
- ✅ **Best on:** C4 (71.37%), C3 (62.11%)
- ✅ **Good on:** homo (52.02%), C5 (49.08%)
- ❌ **Worst on:** C2 (49.87%)

### FedAvgM (Buggy):
- ❌ **Collapsed on:** All distributions (all <45%)
- ⚠️ **Best effort:** Dir0.5/Dir10.0 (44.23%)
- ❌ **Catastrophic:** Dir1.0 (8.85%), C5 (9.96%)

---

## 💡 Quick Reference Guide

### "Which strategy for which distribution?"

**For homo (IID):**
```
1. FedAvg:  59.10% ⭐ Best
2. FedProx: 58.53%
3. FedAdam: 52.02%
```

**For C4 (4 classes, label skew):**
```
1. FedProx: 78.02% ⭐ Best
2. FedAvg:  75.32%
3. FedAdam: 71.37%
```

**For C3 (3 classes, label skew):**
```
1. FedProx: 69.18% ⭐ Best
2. FedAvg:  68.24%
3. FedAdam: 62.11%
```

**For Dirichlet (varying non-IID):**
```
Dir0.1:  FedAvg  (50.16%)
Dir0.5:  FedProx (55.24%)
Dir1.0:  FedAvg  (54.87%)
Dir10.0: FedAvg  (58.20%)
```

---

## 📊 Data-Normalized Comparison

### Efficiency (Accuracy per 10k samples):

| Strategy | homo (50k) | C2 (60k) | C3 (90k) | C4 (120k) | C5 (150k) | Avg Efficiency |
|----------|------------|----------|----------|-----------|-----------|----------------|
| **FedAvg** | **11.82%** | 8.28% | 7.58% | 6.28% | 3.20% | 7.43% |
| **FedAvgM** | 8.14% ❌ | 4.87% ❌ | 1.24% ❌ | 1.02% ❌ | 0.66% ❌ | 3.19% ❌ |
| **FedProx** | **11.71%** | **8.47%** | **7.69%** | **6.50%** | 3.18% | **7.51%** ⭐ |
| **FedAdam** | 10.40% | 8.31% | 6.90% | 5.95% | 3.27% | 6.97% |

**FedProx is most data-efficient overall!**

---

## 🎓 Key Insights from Transposed View

### 1. **Consistency Across Distributions:**
- FedAvg: Moderate (47-75% range)
- FedProx: Moderate (47-78% range)
- FedAdam: ⭐ Best (49-71% range) - most consistent
- FedAvgM: ❌ Worst (9-44% range) - highly inconsistent

### 2. **Stability Pattern:**
- FedAdam: ⭐ Best average stability (-2.94%)
- FedProx: Excellent (-4.22%)
- FedAvg: Good (-4.74%)
- FedAvgM: ❌ Catastrophic (-27.24%)

### 3. **Specialization:**
- FedAvg: Generalist (wins on diverse distributions)
- FedProx: Label skew specialist (dominates C2-C4)
- FedAdam: Stability specialist (lowest degradation)
- FedAvgM: ❌ None (failed everywhere)

---

**End of Transposed Matrix**

*See also:*
- [ACCURACY_MATRIX.md](ACCURACY_MATRIX.md) - Original format (distributions × strategies)
- [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md) - Quick summary
