# Label-Skew Federated Learning Experiments

## Tổng Quan

Codebase này đã được cập nhật để hỗ trợ **label-skew** (phân chia không đồng nhất theo lớp), trong đó mỗi client chỉ có dữ liệu từ **k lớp** cố định.

### Thay đổi chính:

- **6 clients cố định**: Tất cả thí nghiệm sử dụng đúng 6 clients
- **C = k**: Ý nghĩa mới của `C` là **số lớp mỗi client** (không phải số clients mỗi round)
- **100% participation**: Tất cả 6 clients tham gia mỗi round (training và evaluation)

---

## Ý Nghĩa của C (Classes per Client)

| Giá trị C | Ý nghĩa | Ví dụ phân chia (CIFAR-10) |
|-----------|---------|----------------------------|
| **C=1** | Mỗi client chỉ có 1 lớp | Client 0: chỉ có "xe hơi"<br>Client 1: chỉ có "máy bay"<br>... |
| **C=2** | Mỗi client chỉ có 2 lớp | Client 0: "xe hơi", "máy bay"<br>Client 1: "chim", "mèo"<br>... |
| **C=3** | Mỗi client chỉ có 3 lớp | Client 0: lớp 0,1,2<br>Client 1: lớp 3,4,5<br>... |
| **C=4** | Mỗi client chỉ có 4 lớp | Client 0: lớp 0,1,2,3<br>Client 1: lớp 4,5,6,7<br>... |
| **C=5** | Mỗi client chỉ có 5 lớp | Client 0: lớp 0,1,2,3,4<br>Client 1: lớp 5,6,7,8,9<br>... |

### Cơ chế phân chia:

```python
# Ví dụ với C=2 (6 clients, 10 lớp CIFAR-10)
Client 0: lớp [0, 1]
Client 1: lớp [2, 3]
Client 2: lớp [4, 5]
Client 3: lớp [6, 7]
Client 4: lớp [8, 9]
Client 5: lớp [0, 1]  # Lặp lại để đảm bảo đủ 6 clients
```

**Lưu ý**: Các lớp được phân chia luân phiên (rotation) để đảm bảo mỗi lớp được sử dụng một cách cân bằng.

---

## Cấu Hình Mặc Định

File: `pyproject.toml`

```toml
[tool.flwr.app.config]
num-clients = 6               # Cố định 6 clients
min-train-nodes = 6           # Tất cả 6 clients train mỗi round
min-evaluate-nodes = 6        # Tất cả 6 clients evaluate mỗi round
fraction-train = 1.0          # 100% participation
fraction-evaluate = 1.0       # 100% participation

distribution = "C(2)"         # Mỗi client có 2 lớp
experiment-name = "FedAvg_C2"
num-server-rounds = 500
```

---

## Cách Chạy Thí Nghiệm

### 1. Test Nhanh (10 rounds)

```bash
python scripts/test_labelskew.py
```

Kết quả:
- Test với C=2 (mỗi client có 2 lớp)
- Chạy 10 rounds
- Kiểm tra xem implementation có hoạt động không

### 2. Chạy Batch Experiments

#### Quick mode (C=2, 10 rounds):
```bash
python scripts/run_labelskew_experiments.py --quick
```

#### Medium mode (C=1,2,3, 100 rounds):
```bash
python scripts/run_labelskew_experiments.py --medium
```

#### Full mode (C=1,2,3,4,5, 500 rounds):
```bash
python scripts/run_labelskew_experiments.py --full
```

#### Custom mode:
```bash
# Chạy C=2,3 với 200 rounds
python scripts/run_labelskew_experiments.py --classes 2 3 --rounds 200

# Chạy C=1 với FedProx strategy
python scripts/run_labelskew_experiments.py --classes 1 --strategy FedProx
```

### 3. Chạy Thủ Công

```bash
# Ví dụ: C=3 với FedAvg
flwr run . local-simulation --run-config \
  strategy=FedAvg \
  distribution=C(3) \
  num-server-rounds=500 \
  experiment-name=FedAvg_C3
```

---

## Cấu Trúc Files Đã Thay Đổi

### Files Chính:

1. **`pytorchexample/partitioner.py`** ✨ MỚI
   - Thêm class `LabelSkewPartitioner`
   - Hỗ trợ format `C(k)` để chỉ định số lớp mỗi client
   - Phân chia lớp luân phiên cho 6 clients

2. **`pytorchexample/task.py`** 🔧 CẬP NHẬT
   - Xử lý `LabelSkewPartitioner` riêng biệt
   - Filter dataset theo các lớp được assign

3. **`pytorchexample/server_app_experiment.py`** 🔧 CẬP NHẬT
   - Mặc định: 6 clients (thay vì 10)
   - 100% participation (thay vì 50%)

4. **`pyproject.toml`** 🔧 CẬP NHẬT
   - `num-clients = 6`
   - `distribution = "C(2)"`
   - `min-train-nodes = 6`

### Scripts Mới:

5. **`scripts/run_labelskew_experiments.py`** ✨ MỚI
   - Automation script cho C=1,2,3,4,5
   - Hỗ trợ quick/medium/full modes

6. **`scripts/test_labelskew.py`** ✨ MỚI
   - Quick test script (10 rounds, C=2)

---

## Kết Quả Output

### Naming Convention Mới:

```
Tên model: {Strategy}_C{k}_final_model.pt

Ví dụ:
- FedAvg_C1_final_model.pt   # Mỗi client có 1 lớp
- FedAvg_C2_final_model.pt   # Mỗi client có 2 lớp
- FedAvg_C3_final_model.pt   # Mỗi client có 3 lớp
- FedAvg_C5_final_model.pt   # Mỗi client có 5 lớp
```

### CSV Files:

```
results/
├── FedAvg_C1_global_<timestamp>.csv       # Global metrics
├── FedAvg_C1_weight_<timestamp>.csv       # Weight changes
├── FedAvg_C1_config_<timestamp>.txt       # Configuration
├── FedAvg_C2_global_<timestamp>.csv
├── FedAvg_C3_global_<timestamp>.csv
...
```

---

## So Sánh: Cũ vs Mới

| Aspect | Cũ (Sai) | Mới (Đúng) |
|--------|----------|------------|
| **Số clients** | 10 clients | 6 clients (cố định) |
| **Ý nghĩa C** | Số clients mỗi round | **Số lớp mỗi client** |
| **Participation** | 50% (5/10 clients) | 100% (6/6 clients) |
| **Distribution** | IID, Dirichlet | **Label-skew C(k)** |
| **Tên model** | FedAvg_homo_C5.pt | **FedAvg_C2.pt** |

### Ví dụ Cụ Thể:

**Cũ**: `FedAvg_homo_C5_final_model.pt`
- Nghĩa: FedAvg, IID, 5 clients mỗi round

**Mới**: `FedAvg_C2_final_model.pt`
- Nghĩa: FedAvg, **mỗi client có 2 lớp**, 6 clients (100% participation)

---

## Phân Tích Kết Quả

Sau khi chạy xong:

```bash
# Phân tích tất cả label-skew experiments
python scripts/analyze_results.py --pattern "FedAvg_C*"

# So sánh các C values
python scripts/analyze_results.py --pattern "FedAvg_C1*" "FedAvg_C2*" "FedAvg_C5*"
```

---

## Kiểm Tra Data Distribution

Để verify data distribution hoạt động đúng:

```python
from pytorchexample.partitioner import LabelSkewPartitioner

# Tạo partitioner với C=2
partitioner = LabelSkewPartitioner(num_partitions=6, classes_per_client=2)

# Kiểm tra lớp của mỗi client
for client_id in range(6):
    classes = partitioner.get_partition_classes(client_id)
    print(f"Client {client_id}: {classes}")

# Output:
# Client 0: [0, 1]
# Client 1: [2, 3]
# Client 2: [4, 5]
# Client 3: [6, 7]
# Client 4: [8, 9]
# Client 5: [0, 1]
```

---

## Troubleshooting

### Lỗi: "Invalid distribution type"

**Nguyên nhân**: Dùng format sai cho distribution

**Giải pháp**: Sử dụng `C(k)` format
```bash
# Sai
distribution="label-skew"

# Đúng
distribution="C(2)"
```

### Lỗi: "num-supernodes mismatch"

**Nguyên nhân**: Config file cũ có 10 clients

**Giải pháp**: Sử dụng config mới hoặc override:
```bash
flwr run . local-simulation --run-config num-clients=6
```

### Performance Issues

Nếu chạy chậm với 6 clients:
- Giảm `batch-size` xuống 16 hoặc 8
- Sử dụng GPU nếu có: `flwr run . local-simulation-gpu`

---

## Next Steps

1. **Test Implementation**:
   ```bash
   python scripts/test_labelskew.py
   ```

2. **Run Quick Experiments**:
   ```bash
   python scripts/run_labelskew_experiments.py --quick
   ```

3. **Run Full Experiments** (khi đã verify OK):
   ```bash
   python scripts/run_labelskew_experiments.py --full
   ```

4. **Analyze Results**:
   ```bash
   python scripts/analyze_results.py --pattern "FedAvg_C*"
   ```

---

## Tham Khảo

- **Main config**: `pyproject.toml`
- **Partitioner implementation**: `pytorchexample/partitioner.py`
- **Data loading**: `pytorchexample/task.py`
- **Experiment scripts**: `scripts/run_labelskew_experiments.py`
