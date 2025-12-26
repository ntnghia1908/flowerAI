# 🚀 BẮT ĐẦU NGAY - HƯỚNG DẪN NHANH

## ✅ Đã chuẩn bị sẵn

Tất cả code đã được sửa và sẵn sàng chạy:
- ✅ Server app với custom logging
- ✅ Client app với comprehensive metrics
- ✅ 8 test configs (10 rounds each)
- ✅ Scripts để chạy tự động
- ✅ CSV logging hoạt động 100%

## 🎯 CHẠY NGAY (3 BƯỚC)

### Bước 1: Test nhanh (3-4 phút)
Chạy 1 test để verify mọi thứ OK:

```bash
python run_single_test.py homo
```

Kết quả sẽ xuất hiện trong `results/`:
- `test_homo_global_*.csv` - Global metrics
- `test_homo_client_*.csv` - Client metrics
- `test_homo_weight_*.csv` - Weight changes

### Bước 2: Chạy tất cả tests (30-35 phút)
Sau khi bước 1 thành công, chạy tất cả:

```bash
python run_all_tests.py
```

Hoặc Windows:
```bash
run_all_tests.bat
```

### Bước 3: Phân tích kết quả
```bash
python analyze_results.py
```

Sẽ tạo ra:
- Summary statistics
- Convergence analysis
- Comparison charts (PNG files)

## 📊 Các Test Cases

| Command | Distribution | Non-IID Level | Time |
|---------|-------------|---------------|------|
| `python run_single_test.py homo` | IID (Equal) | None | 3-4 min |
| `python run_single_test.py C1` | 1 class/client | ⭐⭐⭐⭐⭐ Extreme | 3-4 min |
| `python run_single_test.py C2` | 2 classes/client | ⭐⭐⭐⭐ Severe | 3-4 min |
| `python run_single_test.py C3` | 3 classes/client | ⭐⭐⭐ Moderate | 3-4 min |
| `python run_single_test.py C4` | 4 classes/client | ⭐⭐ Mild | 3-4 min |
| `python run_single_test.py C5` | 5 classes/client | ⭐ Light | 3-4 min |
| `python run_single_test.py Dir0.1` | Dirichlet(0.1) | ⭐⭐⭐⭐⭐ Very High | 3-4 min |
| `python run_single_test.py Dir0.5` | Dirichlet(0.5) | ⭐⭐⭐ Moderate | 3-4 min |
| `python run_single_test.py Dir1.0` | Dirichlet(1.0) | ⭐⭐ Mild | 3-4 min |
| `python run_single_test.py Dir10.0` | Dirichlet(10.0) | ⭐ Low | 3-4 min |

## 📁 File Structure

```
d:\Desktop\flowerAI\
│
├── 📂 configs/                    # Test configurations (10 files)
│   ├── test_homo.toml            # Homogeneous (IID)
│   ├── test_C1.toml              # Label skew C1
│   ├── test_C2.toml              # Label skew C2
│   ├── test_C3.toml              # Label skew C3
│   ├── test_C4.toml              # Label skew C4 ✨NEW
│   ├── test_C5.toml              # Label skew C5 ✨NEW
│   ├── test_Dir0p1.toml          # Dirichlet(0.1)
│   ├── test_Dir0p5.toml          # Dirichlet(0.5)
│   ├── test_Dir1p0.toml          # Dirichlet(1.0)
│   └── test_Dir10p0.toml         # Dirichlet(10.0)
│
├── 📂 pytorchexample/            # Main code
│   ├── server_app_experiment.py  # ✅ Custom server with logging
│   ├── client_app_experiment.py  # ✅ Client with metrics
│   ├── task.py                   # Model & training
│   ├── partitioner.py            # Data partitioning
│   ├── metrics.py                # ✅ Fixed NaN issues
│   └── logger.py                 # CSV logging
│
├── 📂 results/                   # Output directory (auto-created)
│   └── (test results will be here)
│
├── 🔧 run_all_tests.py           # Run all 8 tests
├── 🔧 run_single_test.py         # Run single test
├── 🔧 run_all_tests.bat          # Windows batch version
├── 📊 analyze_results.py         # Analyze & visualize
│
├── 📖 START_HERE.md              # ← You are here!
├── 📖 QUICK_TEST.md              # Quick reference
├── 📖 TEST_GUIDE.md              # Detailed guide
└── 📖 README_TESTS.md            # Full documentation
```

## 🎓 Hiểu kết quả

### CSV Files:

**1. Global CSV** - Metrics của global model:
```csv
round,loss,accuracy,precision,recall,f1
0,2.3046,0.1000,0.0100,0.1000,0.0182
1,2.4075,0.1000,0.0100,0.1000,0.0182
...
```

**2. Client CSV** - Metrics từng client:
```csv
round,client_id,phase,loss,accuracy,precision,recall,f1,num_examples
1,0,evaluate,2.547,0.0,0.0,0.0,0.0,2000
1,1,evaluate,1.602,0.504,0.252,0.5,0.335,2000
...
```

**3. Weight CSV** - Thay đổi weights:
```csv
round,weight_norm,weight_change,weight_relative_change
0,8.899,0.0,0.0
1,16.536,14.575,1.638
...
```

## 💡 Tips

1. **Chạy test nhỏ trước**:
   ```bash
   python run_single_test.py homo
   ```
   Verify xem có file CSV được tạo trong `results/`

2. **So sánh 2 extremes**:
   ```bash
   python run_single_test.py homo  # IID
   python run_single_test.py C2    # Severe non-IID
   ```

3. **Check convergence**:
   - Homo (IID): Accuracy nên lên ~70-80% sau 10 rounds
   - C2 (2 classes): Accuracy thường stuck ở ~10-30%

## 🔧 Customization

Để thay đổi parameters, edit file config:

```toml
# configs/test_C2.toml
num-server-rounds = 20        # Tăng lên 20 rounds
learning-rate = 0.05          # Giảm learning rate
batch-size = 64               # Tăng batch size
```

## ⚠️ Troubleshooting

**"command not found: flwr"**
```bash
conda activate flwr
```

**RAM không đủ**
Edit config, giảm `batch-size = 16`

**NaN trong results**
Bình thường với extreme non-IID (C1), có thể ignore

## 📈 Production Experiments

Sau khi test OK với 10 rounds, chạy full experiment (500 rounds):

```bash
# Edit pyproject.toml, change num-server-rounds to 500
conda run -n flwr flwr run .
```

## 🎯 Quick Commands

```bash
# Single test
python run_single_test.py homo

# All tests
python run_all_tests.py

# Analyze results
python analyze_results.py

# View results
dir results\test_*.csv
type results\test_homo_global_*.csv
```

## 📚 Đọc thêm

- [QUICK_TEST.md](QUICK_TEST.md) - Commands reference
- [TEST_GUIDE.md](TEST_GUIDE.md) - Detailed guide & analysis
- [README_TESTS.md](README_TESTS.md) - Full documentation

## ✅ Checklist

- [ ] Đã chạy test `homo` thành công
- [ ] Kiểm tra 3 file CSV được tạo trong `results/`
- [ ] Chạy tất cả 8 tests
- [ ] Phân tích kết quả với `analyze_results.py`
- [ ] So sánh convergence giữa các distributions

---

## 🚀 BẮT ĐẦU NGAY!

```bash
python run_single_test.py homo
```

Good luck! 🎉
