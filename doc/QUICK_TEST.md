# Quick Test Guide - 10 Rounds

## 🚀 Chạy ngay (Copy & Paste)

### Chạy TẤT CẢ tests (khuyến nghị):
```bash
# Windows Batch
run_all_tests.bat

# Hoặc Python
python run_all_tests.py
```

### Chạy TỪNG test:
```bash
# Homogeneous (IID) - Baseline
python run_single_test.py homo

# Label Skew - So sánh mức độ non-IID
python run_single_test.py C1     # Extreme (1 class/client)
python run_single_test.py C2     # Severe (2 classes/client)
python run_single_test.py C3     # Moderate (3 classes/client)
python run_single_test.py C4     # Mild (4 classes/client)
python run_single_test.py C5     # Light (5 classes/client)

# Dirichlet - So sánh alpha values
python run_single_test.py Dir0.1   # Very non-IID (α=0.1)
python run_single_test.py Dir0.5   # Moderate non-IID (α=0.5)
python run_single_test.py Dir1.0   # Mild non-IID (α=1.0)
python run_single_test.py Dir10.0  # Nearly IID (α=10.0)
```

### Chạy manual:
```bash
conda run -n flwr flwr run . --run-config configs/test_homo.toml
conda run -n flwr flwr run . --run-config configs/test_C2.toml
conda run -n flwr flwr run . --run-config configs/test_Dir0p5.toml
```

## 📊 Các Test Cases

| Test Case | Distribution | Non-IID Level | Time (~) |
|-----------|-------------|---------------|----------|
| homo | IID (Equal) | None | 3-4 min |
| C1 | 1 class/client | ⭐⭐⭐⭐⭐ Extreme | 3-4 min |
| C2 | 2 classes/client | ⭐⭐⭐⭐ Severe | 3-4 min |
| C3 | 3 classes/client | ⭐⭐⭐ Moderate | 3-4 min |
| C4 | 4 classes/client | ⭐⭐ Mild | 3-4 min |
| C5 | 5 classes/client | ⭐ Light | 3-4 min |
| Dir0.1 | Dirichlet(0.1) | ⭐⭐⭐⭐⭐ Very High | 3-4 min |
| Dir0.5 | Dirichlet(0.5) | ⭐⭐⭐ Moderate | 3-4 min |
| Dir1.0 | Dirichlet(1.0) | ⭐⭐ Mild | 3-4 min |
| Dir10.0 | Dirichlet(10.0) | ⭐ Low | 3-4 min |

**Total time for all 10 tests**: ~35-40 minutes

## 📁 Kết quả

```
results/
├── test_homo_global_20251226_174512.csv      ← Global metrics
├── test_homo_client_20251226_174512.csv      ← Per-client metrics
├── test_homo_weight_20251226_174512.csv      ← Weight changes
├── test_homo_config_20251226_174512.txt      ← Config backup
├── test_C1_global_...csv
├── test_C2_global_...csv
└── ... (24 files total cho 8 tests)
```

## 🔍 Xem kết quả nhanh

```bash
# List all results
dir results\test_*.csv

# View latest global results
type results\test_homo_global_*.csv | head
type results\test_C2_global_*.csv | head
```

## 💡 Tips

1. **Chạy 1 test trước** để đảm bảo mọi thứ hoạt động:
   ```bash
   python run_single_test.py homo
   ```

2. **So sánh nhanh**: Chạy homo (baseline) và C2 (severe non-IID):
   ```bash
   python run_single_test.py homo
   python run_single_test.py C2
   ```

3. **Kiểm tra convergence**: So sánh accuracy cuối cùng:
   - Homo (IID): Thường đạt ~70-80% sau 10 rounds
   - C2 (2 classes): Thường ~10-20% (khó converge)
   - Dir(0.5): Thường ~30-50% (moderate)

## ❓ Troubleshooting

**Lỗi không tìm thấy flwr**:
```bash
conda activate flwr
flwr run . --run-config configs/test_homo.toml
```

**RAM không đủ**:
Edit config file, giảm batch-size từ 32 xuống 16

**NaN trong loss**:
Bình thường với extreme non-IID (C1), có thể ignore

## 📈 Phân tích nhanh với Python

```python
import pandas as pd
import glob

# Load all global results
files = glob.glob('results/test_*_global_*.csv')
for f in files:
    df = pd.read_csv(f)
    test_name = f.split('_global_')[0].split('\\')[-1]
    final_acc = df['accuracy'].iloc[-1]
    print(f"{test_name:15s}: {final_acc:.4f}")
```

## ✅ Expected Results

Sau khi chạy tất cả tests, bạn sẽ có:
- ✅ 8 experiments completed
- ✅ 24 CSV files (3 per test)
- ✅ 8 config backups
- ✅ Có thể so sánh convergence giữa các distributions
