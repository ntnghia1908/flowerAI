# Phân Tích Nghịch Lý: Centralized Accuracy vs Global Accuracy

## Thắc Mắc 1: Tại Sao Accuracy Thấp Hơn Global Accuracy?

### Phát Hiện:

**ĐIỀU BẤT THƯỜNG**: Centralized accuracy thường **CAO HƠN** global accuracy, không phải thấp hơn!

```
10 clients test (round 77 - best):
  Centralized Accuracy: 63.62%  ← Cao hơn
  Global Accuracy:      62.23%  ← Thấp hơn
  Difference: +1.39%
```

### Giải Thích:

#### **Centralized Accuracy** (từ server global_evaluate):
- Evaluate model global trên **centralized test set** (10,000 samples)
- Test set là **IID** (balanced, all 10 classes)
- Model được test trên **unseen data**
- Đây là **TRUE PERFORMANCE** của model global

#### **Global Accuracy** (từ client validation):
- Average của eval_acc từ **10 clients**
- Mỗi client evaluate trên **local validation set** (20% data local)
- Formula: `global_accuracy = (1/N) * Σ(eval_acc_k)`

### Tại Sao Centralized > Global?

Có **3 lý do** có thể:

#### 1. **Test Set Bias** (Nghi ngờ chính!)

Centralized test set có thể **khác phân phối** với training/validation data!

```python
# Kiểm tra trong export_partitions.py
# Test set được load như thế nào?

# HuggingFace mode:
test_dataset = load_dataset("uoft-cs/cifar10", split="test")

# NPY mode:
# Test set là toàn bộ 10,000 test samples gốc
```

**Giả thuyết**: Test set có thể **EASIER** hơn validation sets!

#### 2. **Validation Set Non-IID**

Mặc dù homo distribution (IID for training), validation split có thể không IID:

```python
# In export_partitions.py:
partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
```

Nếu split 80/20 **không stratified**, validation set có thể imbalanced → accuracy thấp hơn.

#### 3. **Averaging Effect**

Global accuracy = average của 10 values, có thể bị ảnh hưởng bởi outliers.

Centralized accuracy = single value trên large dataset, stable hơn.

## Thắc Mắc 2: Homo Thấp Hơn C3, C4?

### Phát Hiện NGHỊCH LÝ:

| Distribution | Centralized Acc | Global Acc | Rounds |
|--------------|-----------------|------------|--------|
| **Homo (IID)** | **63.28%** | 62.91% | 500 |
| **C4 (4 classes)** | 60.30% | **74.83%** | 198 |
| **C3 (3 classes)** | 55.10% | **67.27%** | 500 |

**Nghịch lý**:
- Centralized accuracy: Homo > C4 > C3 ✅ (Đúng như kỳ vọng)
- Global accuracy: **C4 > C3 > Homo** ❌ (SAI!)

### Giải Thích:

#### **Centralized Accuracy** (đúng):
```
Homo (IID):  63.28%  ← Best (như mong đợi)
C4:          60.30%  ← Worse (high non-IID)
C3:          55.10%  ← Worst (very high non-IID)
```

Đây là kết quả **ĐÚNG** và phù hợp lý thuyết!

#### **Global Accuracy** (sai lệch):
```
C4:   74.83%  ← Cao nhất???
C3:   67.27%
Homo: 62.91%  ← Thấp nhất???
```

Điều này **KHÔNG HỢP LÝ**! Non-IID không thể tốt hơn IID!

### Root Cause: CLIENT OVERFITTING!

Hãy xem chi tiết C4:

```
C4 Distribution (final round 198):
  Centralized Accuracy: 56.20%  ← TRUE performance (trên test set toàn cục)
  Global Accuracy:      74.83%  ← MISLEADING! (trên validation local)
```

**Vấn đề**: Clients **OVERFIT** trên local validation set!

#### Cách C4 Overfitting Xảy Ra:

1. **Mỗi client chỉ có 4 classes**:
   ```
   Client 0: [0, 1, 2, 3]
   Client 1: [2, 3, 4, 5]
   ...
   ```

2. **Validation set cũng chỉ có 4 classes đó**:
   ```python
   # Split 80/20 từ partition đã filtered
   partition_train_test = filtered_dataset.train_test_split(test_size=0.2)
   ```

3. **Model local OVERFIT trên 4 classes**:
   - Train trên 4 classes
   - Validate trên cùng 4 classes đó
   - Accuracy rất cao (74%)!

4. **Nhưng centralized test có 10 classes**:
   - Model kém trên 6 classes còn lại
   - Accuracy chỉ 56%

### Minh Họa:

```
Client 0 (classes [0,1,2,3]):
  Local validation (4 classes):  Acc = 85%  ← Cao!
  Global test (10 classes):      Acc = 40%  ← Thấp! (chỉ đúng 4/10)

Client 1 (classes [2,3,4,5]):
  Local validation (4 classes):  Acc = 82%
  Global test (10 classes):      Acc = 38%

Average (Global Accuracy):        74.83%  ← MISLEADING!
Centralized (True):               56.20%  ← TRUE!
```

## Kết Luận

### 1. Metric Nào Đáng Tin?

**✅ TIN TƯỞNG**: **Centralized Accuracy**
- Evaluate trên full test set (10 classes)
- Phản ánh TRUE performance
- Không bị bias bởi local distribution

**❌ KHÔNG TIN**: **Global Accuracy** (cho non-IID)
- Bị misleading do client overfitting
- Chỉ đúng với IID distribution
- Overestimate performance với non-IID

### 2. Kết Quả Thực Sự:

Theo Centralized Accuracy (đúng):

| Distribution | Best Acc | Rank | Phù hợp lý thuyết? |
|--------------|----------|------|--------------------|
| Homo (IID) | 63.28% | 1st | ✅ YES |
| C4 (4 classes) | 60.30% | 2nd | ✅ YES |
| C3 (3 classes) | 55.10% | 3rd | ✅ YES |

**Kết luận**: Homo **THỰC SỰ TỐT NHẤT** như kỳ vọng!

### 3. Tại Sao Test Trước Cho 70-80%?

Có thể do:

#### A. **Different test setup**:
- Trước: Có thể chỉ test trên 100-200 rounds (tránh overfitting)
- Bây giờ: Test 500 rounds → overfit nghiêm trọng

#### B. **Different hyperparameters**:
```python
# Có thể trước đây:
local_epochs = 5      # Nhiều hơn
learning_rate = 0.01  # Cao hơn
batch_size = 32       # Nhỏ hơn
```

#### C. **Different evaluation point**:
- Trước: Lấy best model (round ~50-100)
- Bây giờ: Lấy final model (round 500) → đã overfit

## Recommendations

### 1. Fix Validation Split (Urgent!)

Sửa lại stratified split:

```python
# In export_partitions.py
# Change from:
partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

# To:
partition_train_test = partition.train_test_split(
    test_size=0.2,
    seed=42,
    stratify_by_column='label'  # Ensure balanced classes
)
```

### 2. Always Use Centralized Accuracy

Trong reports và comparisons, **chỉ dùng centralized accuracy**!

```python
# In server_app_experiment.py logging:
print(f"Round {round}: Centralized Acc = {centralized_acc:.4f}")
# Don't emphasize global_accuracy for non-IID!
```

### 3. Early Stopping Based on Centralized Acc

```python
# Track best centralized accuracy, not global
if centralized_acc > best_centralized_acc:
    best_centralized_acc = centralized_acc
    save_checkpoint()
```

### 4. Realistic Expectations

Với CIFAR-10 + Simple CNN + FedAvg:

| Distribution | Realistic Best Acc | Your Result | Status |
|--------------|-------------------|-------------|--------|
| Homo (IID) | 65-75% | 63.28% | ✅ Good |
| C4 (moderate non-IID) | 55-65% | 60.30% | ✅ Good |
| C3 (high non-IID) | 50-60% | 55.10% | ✅ Good |
| C2 (very high non-IID) | 40-50% | ? | To test |

**Bạn đang đạt kết quả tốt!** Chỉ cần tin vào **centralized accuracy**!

## Action Items

1. ✅ **Luôn dùng Centralized Accuracy** để đánh giá
2. ✅ **Bỏ qua Global Accuracy** cho non-IID distributions
3. ⏭️ Fix stratified split cho validation (optional improvement)
4. ⏭️ Test với early stopping (round 80-100) để đạt 65-70%
5. ⏭️ Thử tăng local_epochs lên 3-5 để đạt 70%+

---

**Kết luận cuối cùng**:
- Homo **KHÔNG PHẢI** thấp hơn C3, C4
- Homo **TỐT NHẤT** (63.28% centralized)
- Global accuracy bị misleading do client overfitting
- Kết quả hiện tại **RẤT TỐT** cho FL baseline!
