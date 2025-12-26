# Setup & Run Experiments: 6 Clients, 500 Rounds

Hướng dẫn chạy thí nghiệm với **6 clients** (tất cả tham gia mỗi round) và **500 rounds**.

---

## 📋 Configuration

### Số lượng Clients
- **Total clients**: 6
- **Clients per round**: 6 (100% participation)
- **min-train-nodes**: 6
- **min-evaluate-nodes**: 6
- **fraction-train**: 1.0
- **fraction-evaluate**: 1.0

### Training Parameters
- **Rounds**: 500
- **Local epochs**: 1
- **Batch size**: 32
- **Learning rate**: 0.1

### Experiments
- **Strategies**: 6 (FedAvg, FedAvgM, FedProx, FedAdam, FedAdagrad, FedYogi)
- **Distributions**: 6 (homo, Dir(10.0), Dir(1.0), Dir(0.5), Dir(0.1), Dir(0.01))
- **Total**: **36 experiments** (6 × 6)

---

## 🚀 Quick Start

### Step 1: Setup Config
```bash
# Switch to 6-clients config
cp pyproject_6clients_500rounds.toml pyproject.toml
```

### Step 2: Test nhanh
```bash
# Test với 10 rounds (2-3 phút)
python run_6clients_experiments.py --quick --yes
```

### Step 3: Chạy full experiments
```bash
# Test với 100 rounds trước
python run_6clients_experiments.py --test --yes

# Full 500 rounds, all 36 experiments
python run_6clients_experiments.py --all --yes
```

---

## ⏱️ Time Estimates (6 Clients)

| Rounds | Time per Experiment | Total Time (36 experiments) |
|--------|--------------------|-----------------------------|
| 10     | ~2 minutes         | ~72 minutes (1.2 hours)     |
| 100    | ~8-16 minutes      | ~288-576 minutes (5-10 hours) |
| 500    | ~40-80 minutes     | ~1440-2880 minutes (24-48 hours) |

**Note**: Với 6 clients (thay vì 10), mỗi round nhanh hơn khoảng 40%.

---

## 📊 Experiment Matrix

### Tổ hợp experiments (36 total)

```
Strategies (6):
├── FedAvg
├── FedAvgM
├── FedProx
├── FedAdam
├── FedAdagrad
└── FedYogi

×

Distributions (6):
├── homo (IID)
├── Dir(10.0) - Mild non-IID
├── Dir(1.0) - Moderate non-IID
├── Dir(0.5) - Strong non-IID
├── Dir(0.1) - Very strong non-IID
└── Dir(0.01) - Extreme non-IID

=

36 Experiments
```

---

## 💻 Commands

### Quick Test (10 rounds)
```bash
python run_6clients_experiments.py --quick --yes
```
→ 1 experiment (FedAvg + homo), ~2 minutes

### Test Mode (100 rounds)
```bash
python run_6clients_experiments.py --test --yes
```
→ 4 experiments (2 strategies × 2 distributions), ~30-60 minutes

### Full Run - By Distribution

#### homo (IID) - 6 experiments
```bash
python run_6clients_experiments.py \
  --strategies FedAvg FedAvgM FedProx FedAdam FedAdagrad FedYogi \
  --distributions homo \
  --num-rounds 500 --yes
```
→ ~4-8 hours

#### Dir(10.0) - Mild non-IID - 6 experiments
```bash
python run_6clients_experiments.py \
  --strategies FedAvg FedAvgM FedProx FedAdam FedAdagrad FedYogi \
  --distributions "Dir(10.0)" \
  --num-rounds 500 --yes
```

#### Dir(1.0) - Moderate non-IID - 6 experiments
```bash
python run_6clients_experiments.py \
  --strategies FedAvg FedAvgM FedProx FedAdam FedAdagrad FedYogi \
  --distributions "Dir(1.0)" \
  --num-rounds 500 --yes
```

#### Dir(0.5) - Strong non-IID - 6 experiments
```bash
python run_6clients_experiments.py \
  --strategies FedAvg FedAvgM FedProx FedAdam FedAdagrad FedYogi \
  --distributions "Dir(0.5)" \
  --num-rounds 500 --yes
```

#### Dir(0.1) - Very strong non-IID - 6 experiments
```bash
python run_6clients_experiments.py \
  --strategies FedAvg FedAvgM FedProx FedAdam FedAdagrad FedYogi \
  --distributions "Dir(0.1)" \
  --num-rounds 500 --yes
```

#### Dir(0.01) - Extreme non-IID - 6 experiments
```bash
python run_6clients_experiments.py \
  --strategies FedAvg FedAvgM FedProx FedAdam FedAdagrad FedYogi \
  --distributions "Dir(0.01)" \
  --num-rounds 500 --yes
```

### Full Run - All at Once (36 experiments)
```bash
python run_6clients_experiments.py --all --num-rounds 500 --yes
```
→ ~24-48 hours

---

## 📁 Results Location

```
results/
├── FedAvg_homo_6clients_global_*.csv
├── FedAvg_homo_6clients_client_*.csv
├── FedAvg_homo_6clients_weight_*.csv
├── FedAvg_Dir10p0_6clients_global_*.csv
├── ...
└── (36 experiments × 3 files = 108 CSV files)
```

### CSV Format

**Global Metrics:**
```csv
round,loss,accuracy,precision,recall,f1
0,2.303,0.1000,0.0100,0.1000,0.0182
...
500,0.523,0.8250,0.8190,0.8210,0.8200
```

**Client Metrics:**
```csv
round,client_id,phase,loss,accuracy,precision,recall,f1,num_examples
1,0,train,2.1,0.25,0.24,0.23,0.235,4167
1,0,evaluate,2.2,0.23,0.22,0.21,0.215,1042
...
```

**Weight Metrics:**
```csv
round,weight_norm,weight_change,weight_relative_change
1,8.92,0.0,0.0
2,10.26,5.12,0.57
...
```

---

## 🔧 Configuration Files

### Main Config: `pyproject_6clients_500rounds.toml`

```toml
[tool.flwr.app.config]
num-server-rounds = 500
num-clients = 6
fraction-train = 1.0              # All 6 clients train
fraction-evaluate = 1.0           # All 6 clients evaluate
min-train-nodes = 6
min-evaluate-nodes = 6

[tool.flwr.federations.local-simulation]
options.num-supernodes = 6        # 6 clients total
```

### Key Differences from 10-Client Setup

| Parameter | 10 Clients | 6 Clients |
|-----------|-----------|-----------|
| `num-clients` | 10 | 6 |
| `num-supernodes` | 10 | 6 |
| `min-train-nodes` | 5 (C5) | 6 (all) |
| `min-evaluate-nodes` | 5 (C5) | 6 (all) |
| `fraction-train` | 0.5 (50%) | 1.0 (100%) |
| `fraction-evaluate` | 0.5 (50%) | 1.0 (100%) |

---

## 📈 Expected Results

### Performance (6 Clients vs 10 Clients)

**Advantages:**
- ✅ Faster per-round (fewer clients to coordinate)
- ✅ Less memory usage
- ✅ Easier to debug

**Trade-offs:**
- ⚠️ Less data per client (if same dataset)
- ⚠️ May have different convergence characteristics
- ⚠️ Non-IID effects may be stronger

### Typical Accuracy (CIFAR-10)

| Distribution | Expected Accuracy (500 rounds) |
|-------------|-------------------------------|
| homo (IID) | 75-85% |
| Dir(10.0) | 70-80% |
| Dir(1.0) | 65-75% |
| Dir(0.5) | 60-70% |
| Dir(0.1) | 50-65% |
| Dir(0.01) | 40-55% |

---

## 🎯 Workflow Recommendation

### Incremental Testing Approach

#### Day 1: Quick Test
```bash
cp pyproject_6clients_500rounds.toml pyproject.toml
python run_6clients_experiments.py --quick --yes
python analyze_results.py
```
→ Verify everything works (~5 minutes)

#### Day 2: Test Mode
```bash
python run_6clients_experiments.py --test --yes
python analyze_results.py
```
→ Test 100 rounds (~1 hour)

#### Week 1: Run by Distribution
```bash
# Run one distribution per day
python run_6clients_experiments.py \
  --strategies FedAvg FedAvgM FedProx FedAdam FedAdagrad FedYogi \
  --distributions homo \
  --num-rounds 500 --yes
```
→ 6 experiments per day (~4-8 hours)

#### Week 2: Complete Remaining
```bash
# Continue with other distributions
# Total: 6 days for all 36 experiments
```

---

## 📊 Analysis

### After Experiments Complete

```bash
# View all results
python analyze_results.py

# Compare strategies on one distribution
python analyze_results.py --pattern "*_homo_6clients*" --compare --metric accuracy

# Compare distributions for one strategy
python analyze_results.py --pattern "FedAvg_*_6clients*" --compare --metric accuracy

# View specific experiment
python analyze_results.py --pattern "FedAvg_homo_6clients*"
```

---

## 🔄 Switching Between Configs

### To 6 Clients
```bash
cp pyproject_6clients_500rounds.toml pyproject.toml
```

### To 10 Clients (original)
```bash
cp pyproject_experiment.toml pyproject.toml
```

### Back to Default
```bash
cp pyproject.toml.original pyproject.toml
```

---

## ⚠️ Important Notes

### Memory Requirements
- **6 clients**: ~3-6GB RAM
- **10 clients**: ~5-10GB RAM
- Lower memory footprint with 6 clients

### GPU Usage
```bash
# Each client gets 1/3 GPU (3 clients per GPU)
python run_6clients_experiments.py --all --gpu --yes
```

### Disk Space
- 36 experiments × 3 CSV files × ~5MB = ~540MB
- Keep ~1GB free for safety

---

## ✅ Verification

### Check Config
```bash
grep "num-clients" pyproject.toml
# Should show: num-clients = 6

grep "num-supernodes" pyproject.toml
# Should show: options.num-supernodes = 6
```

### Test Run
```bash
python run_6clients_experiments.py --quick --yes
```

Expected output:
```
Clients: 6 (all participate each round)
Rounds: 10
[PASS] Experiment FedAvg_homo_6clients completed successfully!
```

---

## 📝 Summary

**Setup:** 6 clients, 500 rounds, 36 experiments
**Time:** ~24-48 hours for all experiments
**Results:** 108 CSV files with full metrics
**Commands:**
- Test: `python run_6clients_experiments.py --quick --yes`
- Full: `python run_6clients_experiments.py --all --yes`

**Ready to run! 🚀**
