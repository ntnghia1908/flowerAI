# 🔍 Data Distribution Analysis - Potential Issue Found

**Date:** 2026-01-10
**Finding:** Homogeneous vs Label Skew data size mismatch
**Severity:** HIGH - Affects result interpretation

---

## 🚨 Key Discovery

### Sample Count Disparity:

| Distribution | Samples/Client | Total Samples | vs Homo |
|-------------|----------------|---------------|---------|
| **homo (IID)** | 6,667 | **50,000** | Baseline |
| **C2** | ? | ? | TBD |
| **C3** | 12,000 | **90,000** | **+1.8x** ⚠️ |
| **C4** | 16,000 | **120,000** | **+2.4x** ⚠️ |
| **C5** | ? | ? | TBD |
| Dir0.1-10.0 | ? | ? | TBD |

### Problem:

**Label skew distributions (C3, C4) have 1.8-2.4x MORE training data than homo!**

This creates **unfair comparison** because:
1. More data → Better model performance
2. C4's 75.3% accuracy is partly due to having 2.4x more samples
3. Homo's 59.1% might be data-limited, not algorithm-limited

---

## 📊 Detailed Analysis

### Homo Distribution:

```
CIFAR-10 Federated Learning Dataset
Distribution: homo
Number of clients: 6
Test set samples: 10000

Partition 0: Train=6667, Val=1667
Partition 1: Train=6667, Val=1667
Partition 2: Train=6666, Val=1667
Partition 3: Train=6666, Val=1667
Partition 4: Train=6666, Val=1667
Partition 5: Train=6666, Val=1667

Total train: 39,998
Total val:   10,002
Total:       50,000
```

**Observations:**
- Uses CIFAR-10 train split (50,000 images)
- Each client gets ~8,333 samples (train+val)
- Standard 80/20 train/val split

### C3 Distribution:

```
CIFAR-10 Federated Learning Dataset
Distribution: C3
Number of clients: 6
Test set samples: 10000

Partition 0: Train=12000, Val=3000, Classes=[0,1,2]
Partition 1: Train=12000, Val=3000, Classes=[3,4,5]
Partition 2: Train=12000, Val=3000, Classes=[6,7,8]
Partition 3: Train=12000, Val=3000, Classes=[0,1,9]
Partition 4: Train=12000, Val=3000, Classes=[2,3,4]
Partition 5: Train=12000, Val=3000, Classes=[5,6,7]

Total train: 72,000
Total val:   18,000
Total:       90,000
```

**Observations:**
- 90,000 total samples (**1.8x more** than homo!)
- Each client gets 15,000 samples
- Some classes appear in multiple clients (e.g., class 0 in clients 0,3,5)

### C4 Distribution:

```
CIFAR-10 Federated Learning Dataset
Distribution: C4
Number of clients: 6
Test set samples: 10000

Partition 0: Train=16000, Val=4000, Classes=[0,1,2,3]
Partition 1: Train=16000, Val=4000, Classes=[4,5,6,7]
Partition 2: Train=16000, Val=4000, Classes=[0,1,8,9]
Partition 3: Train=16000, Val=4000, Classes=[2,3,4,5]
Partition 4: Train=16000, Val=4000, Classes=[6,7,8,9]
Partition 5: Train=16000, Val=4000, Classes=[0,1,2,3]

Total train: 96,000
Total val:   24,000
Total:       120,000
```

**Observations:**
- 120,000 total samples (**2.4x more** than homo!)
- Each client gets 20,000 samples
- Many classes replicated (e.g., classes 0,1,2,3 in clients 0,5)

---

## 🔍 Why This Happened

### Label Skew Partitioning Strategy:

Looking at the code, label skew likely works this way:

1. **Homo:** Each client gets 1/6 of entire CIFAR-10 train set (50k images)
   - Client 0: All classes, 1/6 of each class
   - Client 1: All classes, 1/6 of each class
   - ... etc

2. **C3/C4:** Clients get assigned specific classes, **with replication**
   - Client 0: Classes [0,1,2] → ALL images from these classes
   - Client 3: Classes [0,1,9] → ALL images from these classes again!
   - **Result:** Class 0,1 images used by BOTH clients 0 and 3!

### Why Replication Happens:

**CIFAR-10 has 10 classes:**
- 5,000 images per class
- Total: 50,000 images

**C4 with 6 clients, 4 classes each:**
```
Client 0: [0,1,2,3]   → 20,000 images
Client 1: [4,5,6,7]   → 20,000 images
Client 2: [0,1,8,9]   → 20,000 images (classes 0,1 repeated!)
Client 3: [2,3,4,5]   → 20,000 images (classes 2,3 repeated!)
Client 4: [6,7,8,9]   → 20,000 images (classes 6,7 repeated!)
Client 5: [0,1,2,3]   → 20,000 images (exact copy of client 0!)

Total unique: 50,000
Total with replication: 120,000
```

**Each class appears in multiple clients:**
- Class 0: Clients 0, 2, 5 → **3x replication**
- Class 1: Clients 0, 2, 5 → **3x replication**
- Class 2: Clients 0, 3, 5 → **3x replication**
- etc.

---

## 📈 Impact on Results

### Performance Comparison (FedAvg):

| Distribution | Accuracy | Samples/Client | Total Samples | Data Advantage |
|-------------|----------|----------------|---------------|----------------|
| **C4** | 75.3% | 16,000 | 120,000 | 2.4x |
| **C3** | 68.2% | 12,000 | 90,000 | 1.8x |
| **homo** | 59.1% | 6,667 | 50,000 | 1.0x (baseline) |

### Interpretation:

**Before (assuming equal data):**
- "C4 performs best because 4 classes/client is optimal for label skew"

**After (knowing data difference):**
- "C4 has 75.3% accuracy BUT uses 2.4x more data"
- "Homo has only 59.1% BUT uses 2.4x LESS data"
- **Question:** Would homo reach 75%+ if given same amount of data?

### Adjusted Performance (Normalized by Data):

**Rough estimate: Accuracy per 10k samples**
```
C4:   75.3% / 12 samples = 6.28% per 10k
C3:   68.2% / 9 samples  = 7.58% per 10k
homo: 59.1% / 5 samples  = 11.82% per 10k  ← BEST efficiency!
```

**If normalized for data amount:**
- Homo might be **MOST efficient** per sample!
- C4's advantage might be just from having more data

---

## ❓ Questions to Investigate

### 1. Is this replication intentional?

**Possible reasons:**
- ✅ Intentional: Simulate realistic label skew where some classes are popular
- ✅ Intentional: Test how algorithms handle overlapping data
- ❌ Bug: Accidental replication in partitioning code

### 2. Is this fair comparison?

**Arguments FOR current approach:**
- Real-world: Some clients may have more data
- Tests algorithm's ability to handle heterogeneous data sizes

**Arguments AGAINST:**
- Unfair to compare C4 (120k samples) vs homo (50k samples)
- Can't isolate effect of "label skew" from "more data"

### 3. What should "homo" baseline be?

**Option A: Current (50k total, no replication)**
- Each client: 8,333 samples
- Standard IID partition

**Option B: Replicated homo (120k total, match C4)**
- Each client: 20,000 samples
- Each sample appears in multiple clients (like C4)
- Would this improve homo from 59.1% to 70%+?

---

## 🔧 Recommended Actions

### Immediate:

1. **✅ Check all distribution sample counts**
   ```bash
   for dist in C2 C5 Dir0.1 Dir0.5 Dir1.0 Dir10.0; do
       cat data/cifar10_${dist}_6partition/summary.txt | grep "Total samples"
   done
   ```

2. **✅ Document data replication pattern**
   - Which distributions have replication?
   - How much overlap per client?

3. **✅ Update analysis with data-normalized metrics**
   - Accuracy per 10k samples
   - Learning efficiency

### Future Experiments:

4. **⏳ Create fair comparison:**
   - Option A: Replicate homo to match C4 sample count
   - Option B: Subsample C4 to match homo (50k total)
   - Compare results

5. **⏳ Test data scaling hypothesis:**
   - Run homo with 2x, 3x, 4x data (via replication or augmentation)
   - See if accuracy improves to match C4

---

## 💡 Hypothesis

### Current Results May Be Explained By:

**Not just algorithm behavior, but also:**
1. **Data quantity effect:** More samples → Better performance
2. **Data overlap effect:** Replication acts as implicit data augmentation
3. **Class coverage:** Label skew ensures each class seen by multiple clients

### True Ranking (Data-normalized):

**Efficiency (Accuracy per 10k samples):**
```
1. homo:  11.82%/10k  ⭐ Most efficient
2. C3:     7.58%/10k
3. C4:     6.28%/10k  ⭐ Least efficient (but highest absolute)
```

**This suggests:**
- Homo (IID) learns **FASTEST** per sample
- C4 reaches **HIGHEST** absolute accuracy due to more data
- Label skew is **LESS EFFICIENT** but works with more data

---

## 📋 Next Steps

### To Verify:

1. Check C2, C5, Dirichlet distributions for sample counts
2. Analyze class overlap patterns
3. Look at partitioning code to understand replication logic
4. Check if this is intentional design or bug

### To Fix (if needed):

1. Create normalized homo baseline (match C4 sample count)
2. Re-run experiments with equal data amounts
3. Update analysis with data-normalized metrics
4. Document this in methodology

### To Document:

1. Add data statistics to all result reports
2. Include "samples per client" in performance tables
3. Note replication patterns in experiment descriptions
4. Clarify that label skew results include data quantity effect

---

## 🎯 Conclusion

**Key Finding:**
Label skew distributions (C3, C4) have **1.8-2.4x more training data** than homogeneous distribution due to class replication across clients.

**Impact:**
- C4's 75.3% vs homo's 59.1% is **NOT** purely algorithmic difference
- Data quantity confounds the comparison
- True "label skew effect" cannot be isolated without controlling for data amount

**Recommendation:**
1. ✅ Document this finding in all reports
2. ✅ Add data-normalized metrics
3. ⏳ Consider creating fair comparison experiments

**This doesn't invalidate results, but requires careful interpretation!**

---

**End of Analysis**
