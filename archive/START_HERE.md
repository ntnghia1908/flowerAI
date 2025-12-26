# Federated Learning Experiments - START HERE

## ✅ Status: TESTED & WORKING

Framework hoàn chỉnh để chạy thí nghiệm Federated Learning với nhiều thuật toán, phân phối dữ liệu, và cấu hình clients.

**Đã test thành công trên Windows ngày 22/12/2024**

---

## 🚀 Quick Start (Copy & Paste)

```bash
# 1. Cài đặt
pip install -e .

# 2. Switch sang config cho experiments
cp pyproject_experiment.toml pyproject.toml

# 3. Test nhanh (10 rounds, 2 phút)
python run_experiments.py --strategies FedAvg --distributions homo --client-configs C5 --num-rounds 10 --yes

# 4. Xem kết quả
ls -lh results/
python analyze_results.py
```

**Nếu thành công**, bạn sẽ thấy:
- `[PASS] Experiment ... completed successfully!`
- CSV files trong `results/` directory
- Metrics: loss, accuracy, precision, recall, F1

---

## 📚 Documentation (Đọc theo thứ tự)

| File | Mục đích | Đọc khi nào |
|------|----------|-------------|
| **[FINAL_SETUP.md](FINAL_SETUP.md)** | ⭐ **Bắt đầu tại đây** | Setup & chạy experiments đầu tiên |
| [QUICK_START.md](QUICK_START.md) | Hướng dẫn nhanh | Khi muốn chạy experiments cụ thể |
| [EXPERIMENTS_README.md](EXPERIMENTS_README.md) | Tài liệu chi tiết đầy đủ | Khi cần hiểu sâu về framework |
| [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) | Chi tiết implementation | Khi muốn custom/extend framework |
| [CHECKLIST.md](CHECKLIST.md) | Checklist verify | Khi cần check tất cả tính năng |

---

## 🎯 Main Features

### FL Strategies (6 algorithms)
✅ FedAvg | FedAvgM | FedProx | FedAdam | FedAdagrad | FedYogi

### Data Distributions (6 types)
✅ homo (IID) | Dir(10.0) | Dir(1.0) | Dir(0.5) | Dir(0.1) | Dir(0.01)

### Client Configurations (5 configs)
✅ C1 (1 client) | C2 (2) | C3 (3) | C4 (4) | C5 (5)

### Metrics Collected
✅ **Global**: loss, accuracy, precision, recall, F1
✅ **Client**: loss, accuracy, precision, recall, F1
✅ **Weight**: norm, change, relative_change

### Automation
✅ Batch experiment runner
✅ CSV logging tự động
✅ Analysis & visualization tools
✅ Configuration management

---

## 📊 Sample Results (Verified)

```csv
round,loss,accuracy,precision,recall,f1
0,2.304,0.1000,0.0100,0.1000,0.0182
10,1.978,0.2759,0.3332,0.2759,0.2462
```

**Files generated:**
```
results/
├── FedAvg_homo_C5_global_*.csv
├── FedAvg_homo_C5_client_*.csv
└── FedAvg_homo_C5_weight_*.csv
```

---

## 🔥 Common Commands

### Test Commands
```bash
# Quick test (10 rounds)
python run_experiments.py --quick --yes

# Medium test (100 rounds)
python run_experiments.py --medium --yes
```

### Production Commands
```bash
# Single distribution (30 experiments)
python run_experiments.py \
  --strategies FedAvg FedAvgM FedProx FedAdam FedAdagrad FedYogi \
  --distributions homo \
  --client-configs C1 C2 C3 C4 C5 \
  --num-rounds 500 --yes

# Full experiment set (180 experiments)
python run_experiments.py --all --num-rounds 500 --yes
```

### Analysis Commands
```bash
# View all results
python analyze_results.py

# Compare experiments
python analyze_results.py --pattern "FedAvg*" --compare --metric accuracy
```

---

## ⏱️ Time Estimates

| Experiment Type | Time |
|----------------|------|
| Quick test (10 rounds) | ~2 minutes |
| Single experiment (500 rounds) | ~30-60 minutes |
| 30 experiments (1 distribution) | ~15-30 hours |
| **180 experiments (full)** | **~90-180 hours (3-7 days)** |

---

## ⚠️ Important Notes

### Before Running Experiments

1. **Switch config:**
   ```bash
   cp pyproject_experiment.toml pyproject.toml
   ```

2. **Verify setup:**
   ```bash
   python test_experiment.py
   ```

3. **Check disk space:**
   - CSV files: ~1-10MB per experiment
   - 180 experiments: ~500MB-2GB total

### After Experiments

1. **Backup results:**
   ```bash
   cp -r results results_backup_$(date +%Y%m%d)
   ```

2. **Restore config:**
   ```bash
   cp pyproject.toml.original pyproject.toml
   ```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Import errors | `pip install -e .` |
| No CSV files | Use `pyproject_experiment.toml` |
| "Key not found" error | Check all keys defined in config |
| Unicode errors | Already fixed with ASCII replacements |
| Out of memory | Reduce batch size or clients |
| Slow execution | Use `--gpu` or reduce rounds |

---

## 📦 Files Overview

### Core Files (Don't modify)
- `pytorchexample/*.py` - Framework implementation
- `pyproject_experiment.toml` - Experiment configuration
- `requirements.txt` - Dependencies

### Scripts (Use these)
- `run_experiments.py` - Main experiment runner ⭐
- `test_experiment.py` - Quick test
- `analyze_results.py` - Results analysis

### Results (Auto-generated)
- `results/` - All CSV files and configs

### Documentation (Read these)
- `FINAL_SETUP.md` - Main setup guide ⭐
- `QUICK_START.md` - Quick reference
- `EXPERIMENTS_README.md` - Detailed docs

---

## ✅ Verification

Run this to verify everything is working:

```bash
# 1. Test framework
python test_experiment.py

# Expected output:
# [PASS] TEST PASSED!

# 2. Check results
ls results/test_*.csv

# Expected: 3 CSV files (global, client, weight)

# 3. Quick experiment
python run_experiments.py --quick --yes

# Expected: [PASS] Experiment completed successfully!
```

If all tests pass, you're ready! 🎉

---

## 🎯 Workflow Recommendation

### Day 1: Setup & Test
```bash
pip install -e .
python test_experiment.py
python run_experiments.py --quick --yes
python analyze_results.py
```

### Day 2-3: Medium Tests
```bash
python run_experiments.py --medium --yes
python analyze_results.py
```

### Week 1-2: Full Experiments
```bash
# Run overnight or over weekend
python run_experiments.py --all --num-rounds 500 --yes
```

### Final: Analysis & Paper
```bash
python analyze_results.py
# Create tables and figures
# Write paper
```

---

## 🎓 Learning Path

1. **Beginner**: Read [FINAL_SETUP.md](FINAL_SETUP.md) → Run quick test
2. **Intermediate**: Read [QUICK_START.md](QUICK_START.md) → Run medium test
3. **Advanced**: Read [EXPERIMENTS_README.md](EXPERIMENTS_README.md) → Customize experiments
4. **Expert**: Read [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) → Extend framework

---

## 📞 Help & Support

### Documentation
- [FINAL_SETUP.md](FINAL_SETUP.md) - Complete setup guide
- [EXPERIMENTS_README.md](EXPERIMENTS_README.md) - Full documentation
- [Flower Docs](https://flower.ai/docs/) - Official Flower documentation

### Issues
If you encounter problems:
1. Check [FINAL_SETUP.md](FINAL_SETUP.md) Troubleshooting section
2. Verify config with `python test_experiment.py`
3. Check `pyproject_experiment.toml` is being used

---

## 🎉 Success Criteria

✅ Framework tạo thành công
✅ Test pass với 10 rounds
✅ CSV files được generate
✅ Metrics chính xác (loss, acc, precision, recall, F1)
✅ Weight metrics hoạt động
✅ Batch runner hoạt động
✅ Analysis tools hoạt động

**→ Sẵn sàng cho 180 experiments!**

---

**🚀 Start with: [FINAL_SETUP.md](FINAL_SETUP.md)**
