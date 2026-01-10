# 📊 Complete Data Size Comparison - All Distributions

**Date:** 2026-01-10
**Critical Finding:** Label skew uses replicated data, Dirichlet does not

---

## 🎯 Summary Table

| Distribution | Total Samples | vs Homo | Samples/Client | Replication | FedAvg Acc | Normalized Acc* |
|-------------|---------------|---------|----------------|-------------|------------|----------------|
| **homo (IID)** | 50,000 | 1.0x | 8,333 | No | 59.1% | **11.82%** |
| **C2** | 60,000 | **1.2x** | 10,000 | Yes | 49.7% | 8.28% |
| **C3** | 90,000 | **1.8x** | 15,000 | Yes | 68.2% | 7.58% |
| **C4** | 120,000 | **2.4x** | 20,000 | Yes | 75.3% | 6.28% |
| **C5** | 150,000 | **3.0x** | 25,000 | Yes | 48.0% | 3.20% |
| **Dir0.1** | 50,000 | 1.0x | Variable | No | 50.2% | 10.04% |
| **Dir0.5** | ~50,000 | ~1.0x | Variable | No | 54.7% | ~10.94% |
| **Dir1.0** | ~50,000 | ~1.0x | Variable | No | 54.9% | ~10.98% |
| **Dir10.0** | ~50,000 | ~1.0x | Variable | No | 58.2% | ~11.64% |

*Normalized Acc = (Accuracy / Total Samples) × 100,000

---

## 🔍 Detailed Breakdown

### Label Skew (C2-C5): WITH REPLICATION

**C2 (2 classes per client):**
```
Client 0: [0,1]   → 8,000 samples
Client 1: [2,3]   → 8,000 samples
Client 2: [4,5]   → 8,000 samples
Client 3: [6,7]   → 8,000 samples
Client 4: [8,9]   → 8,000 samples
Client 5: [0,1]   → 8,000 samples (DUPLICATE of client 0!)

Total: 60,000 (1.2x replication)
Unique: 50,000 (each class used in 1.2 clients on average)
```

**C3 (3 classes per client):**
```
Client 0: [0,1,2]   → 12,000 samples
Client 1: [3,4,5]   → 12,000 samples
Client 2: [6,7,8]   → 12,000 samples
Client 3: [0,1,9]   → 12,000 samples (0,1 repeated)
Client 4: [2,3,4]   → 12,000 samples (2,3,4 repeated)
Client 5: [5,6,7]   → 12,000 samples (5,6,7 repeated)

Total: 90,000 (1.8x replication)
Each class appears in ~1.8 clients
```

**C4 (4 classes per client):**
```
Client 0: [0,1,2,3] → 16,000 samples
Client 1: [4,5,6,7] → 16,000 samples
Client 2: [0,1,8,9] → 16,000 samples
Client 3: [2,3,4,5] → 16,000 samples
Client 4: [6,7,8,9] → 16,000 samples
Client 5: [0,1,2,3] → 16,000 samples (EXACT DUPLICATE of client 0!)

Total: 120,000 (2.4x replication)
Each class appears in ~2.4 clients
```

**C5 (5 classes per client):**
```
Client 0: [0,1,2,3,4] → 20,000 samples
Client 1: [5,6,7,8,9] → 20,000 samples
Client 2: [0,1,2,3,4] → 20,000 samples (EXACT DUPLICATE!)
Client 3: [5,6,7,8,9] → 20,000 samples (EXACT DUPLICATE!)
Client 4: [0,1,2,3,4] → 20,000 samples (EXACT DUPLICATE!)
Client 5: [5,6,7,8,9] → 20,000 samples (EXACT DUPLICATE!)

Total: 150,000 (3.0x replication)
Each class appears in 3 clients exactly
```

### Dirichlet (Dir α): NO REPLICATION

**Dir0.1 (highly non-IID):**
```
Client 0: 8,290 samples (all 10 classes, skewed)
Client 1: 6,808 samples
Client 2: 7,061 samples
Client 3: 4,511 samples
Client 4: 8,456 samples
Client 5: 4,872 samples

Total: 50,000 (NO replication)
Unique: 50,000 (each sample used once)
```

**Dir0.5, Dir1.0, Dir10.0:**
- All use ~50,000 total samples
- NO replication
- Variable samples per client based on Dirichlet distribution

---

## 📈 Performance Re-Analysis

### Absolute Performance (Raw Accuracy):

**Ranking:**
```
1. C4:    75.3%  (120,000 samples)  ⭐ Highest accuracy
2. C3:    68.2%  (90,000 samples)
3. homo:  59.1%  (50,000 samples)
4. Dir10: 58.2%  (50,000 samples)
...
```

### Data-Normalized Performance (Accuracy per 10k samples):

**Ranking:**
```
1. homo:   11.82%/10k  ⭐ Most efficient
2. Dir10:  11.64%/10k
3. Dir1.0: 10.98%/10k
4. Dir0.5: 10.94%/10k
5. Dir0.1: 10.04%/10k
6. C2:      8.28%/10k
7. C3:      7.58%/10k
8. C4:      6.28%/10k  ⭐ Least efficient
9. C5:      3.20%/10k  ⭐ Worst efficiency
```

### Key Insights:

**1. Homo (IID) is MOST data-efficient:**
- 59.1% with 50k samples
- If given 120k samples (like C4), might reach **71%+**
- Best learning per sample

**2. C4 reaches highest accuracy BUT is inefficient:**
- 75.3% with 120k samples (2.4x more data)
- If limited to 50k (like homo), might only reach **31-37%**
- Needs more data to perform well

**3. C5 has catastrophic inefficiency:**
- 48.0% with 150k samples (3.0x more data!)
- Only 3.20% accuracy per 10k samples
- Huge data replication doesn't help

**4. Dirichlet maintains efficiency:**
- Dir10.0: 58.2% with 50k (11.64%/10k) - close to homo
- Dir0.1: 50.2% with 50k (10.04%/10k) - reasonable
- NO replication needed

---

## 💡 Hypothesis: Why Label Skew Uses Replication?

### Possible Reasons:

**1. Designed to simulate "popular classes":**
- In real world, some classes more common
- E.g., "cat" photos might appear in many users' datasets
- Replication simulates this

**2. Testing data overlap handling:**
- How does FL handle when multiple clients have same data?
- Does algorithm avoid overfitting to replicated data?

**3. Ensuring sufficient data per client:**
- With only 50k images / 10 classes = 5k per class
- C4 with 4 classes = 20k samples/client without replication
- But partitioner might replicate to ensure each client has enough data

**4. Accidental design:**
- Partitioner might not check for duplicates
- Easy to assign overlapping class sets

---

## ⚠️ Impact on Conclusions

### BEFORE (Naive Interpretation):

> "C4 (4 classes/client) is optimal for label skew, achieving 75.3% accuracy.
> Homo (IID) only reaches 59.1%, showing label skew helps performance."

### AFTER (Data-Aware Interpretation):

> "C4 reaches 75.3% accuracy BUT uses 2.4x more training data (120k vs 50k).
> When normalized for data amount, homo is actually MOST efficient (11.82%/10k vs C4's 6.28%/10k).
> C4's advantage comes from data quantity, not algorithmic superiority.
> True comparison requires equal data amounts."

---

## 🎯 Recommendations

### For Fair Comparison:

**Option 1: Normalize homo to match C4 (Recommended)**
```python
# Replicate homo data 2.4x to match C4
# Each client trains on same data multiple times
# Or use data augmentation to reach 120k effective samples
```

**Option 2: Subsample C4 to match homo**
```python
# Randomly sample 50k from C4's 120k
# Each client gets ~8,333 samples
# Compare with homo's 8,333 samples
```

**Option 3: Report both metrics**
```
- Absolute accuracy (current)
- Data-normalized accuracy (new)
- Clearly note sample count differences
```

### For Analysis:

**Always include these columns in result tables:**
```
| Strategy | Distribution | Accuracy | Total Samples | Acc/10k | Samples/Client |
```

**Example:**
```
| FedAvg | C4   | 75.3% | 120,000 | 6.28%/10k | 20,000 |
| FedAvg | homo | 59.1% |  50,000 | 11.82%/10k |  8,333 |
```

---

## 📊 Updated Performance Table

### FedAvg Results (Data-Aware):

| Rank | Distribution | Accuracy | Samples | Acc/10k | Data Efficiency |
|------|-------------|----------|---------|---------|-----------------|
| 1 | **homo** | 59.1% | 50k | **11.82%** | ⭐⭐⭐⭐⭐ Best |
| 2 | Dir10.0 | 58.2% | 50k | 11.64% | ⭐⭐⭐⭐⭐ |
| 3 | Dir1.0 | 54.9% | 50k | 10.98% | ⭐⭐⭐⭐ |
| 4 | Dir0.5 | 54.7% | 50k | 10.94% | ⭐⭐⭐⭐ |
| 5 | Dir0.1 | 50.2% | 50k | 10.04% | ⭐⭐⭐ |
| 6 | C2 | 49.7% | 60k | 8.28% | ⭐⭐ |
| 7 | **C3** | 68.2% | 90k | 7.58% | ⭐⭐ |
| 8 | **C4** | **75.3%** | 120k | 6.28% | ⭐ |
| 9 | C5 | 48.0% | 150k | 3.20% | ⚠️ Worst |

### Interpretation:

**Absolute Accuracy Leaders:**
- C4: 75.3% (but uses 2.4x data)
- C3: 68.2% (but uses 1.8x data)

**Efficiency Leaders:**
- homo: 11.82%/10k ⭐ Best per-sample learning
- Dir10.0: 11.64%/10k ⭐ Close second

**Worst Performers:**
- C5: 3.20%/10k (despite 3x data!)
- Shows more classes + replication hurts efficiency

---

## 🔬 Suggested Experiment

### Test: "Does homo improve with more data?"

**Hypothesis:**
If homo (IID) is given 120k samples (like C4), it will reach 70-75% accuracy.

**Method:**
```python
# Replicate homo dataset 2.4x
# Each client gets 20,000 samples (instead of 8,333)
# Run FedAvg with same hyperparameters
# Compare: homo_120k vs C4_120k
```

**Expected Result:**
```
homo_50k:   59.1% (baseline)
homo_120k:  ~71% (predicted, 2.4x data)
C4_120k:    75.3% (current)

Difference: C4 only 4-5% better when data is equal
```

**This would show:**
- True effect of label skew is small (~4-5%)
- Most of C4's advantage is from more data
- IID is actually very efficient

---

## ✅ Action Items

1. **✅ DONE:** Document data size differences
2. **✅ DONE:** Calculate data-normalized metrics
3. **⏳ TODO:** Update all analysis reports with data awareness
4. **⏳ TODO:** Add "samples/client" column to result tables
5. **⏳ TODO:** Consider running fair comparison experiments
6. **⏳ TODO:** Update conclusions to reflect data quantity confound

---

**Conclusion:**

The initial conclusion that "C4 is best for label skew" is **partially misleading**.

**Reality:**
- C4 uses 2.4x more data than homo
- When normalized, homo is actually most efficient
- Label skew's "advantage" is confounded with data quantity
- Fair comparison requires equal data amounts

**This is a critical finding that changes interpretation of all results!**

---

**End of Report**
