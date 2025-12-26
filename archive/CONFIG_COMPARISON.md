# Configuration Comparison: 6 vs 10 Clients

So sánh chi tiết giữa 2 setup configurations.

---

## 📊 Configuration Overview

| Parameter | **6 Clients** | **10 Clients** |
|-----------|---------------|----------------|
| **Total Clients** | 6 | 10 |
| **Clients/Round** | 6 (100%) | 5 (50%) hoặc 1-5 |
| **Rounds** | 500 | 500 |
| **Experiments** | 36 (6×6) | 30 (6×6×1) or 180 (6×6×5) |
| **Config File** | `pyproject_6clients_500rounds.toml` | `pyproject_experiment.toml` |
| **Runner Script** | `run_6clients_experiments.py` | `run_experiments.py` |

---

## 🎯 Use Cases

### 6 Clients Setup
**Best for:**
- ✅ Faster experimentation
- ✅ Lower resource usage
- ✅ Simpler debugging
- ✅ When you need all clients to participate each round
- ✅ Quick turnaround for paper deadlines

**Challenges:**
- ⚠️ Less diversity (fewer clients)
- ⚠️ Cannot test different participation rates
- ⚠️ May not scale well to real-world scenarios

### 10 Clients Setup
**Best for:**
- ✅ More realistic federated scenarios
- ✅ Testing different participation rates (C1-C5)
- ✅ Better statistical significance
- ✅ Closer to real-world deployments
- ✅ More comprehensive experiments

**Challenges:**
- ⚠️ Slower per-round execution
- ⚠️ Higher resource requirements
- ⚠️ Longer total experiment time

---

## ⚙️ Technical Comparison

### Configuration Parameters

#### 6 Clients Config
```toml
[tool.flwr.app.config]
num-clients = 6
min-train-nodes = 6
min-evaluate-nodes = 6
fraction-train = 1.0
fraction-evaluate = 1.0

[tool.flwr.federations.local-simulation]
options.num-supernodes = 6
```

#### 10 Clients Config
```toml
[tool.flwr.app.config]
num-clients = 10
min-train-nodes = 5        # For C5 config
min-evaluate-nodes = 5
fraction-train = 0.5
fraction-evaluate = 0.5

[tool.flwr.federations.local-simulation]
options.num-supernodes = 10
```

---

## ⏱️ Time Comparison

### Per Experiment (500 rounds)

| Setup | Time per Experiment |
|-------|---------------------|
| 6 Clients | 40-80 minutes |
| 10 Clients (C5) | 50-100 minutes |
| 10 Clients (C1) | 30-60 minutes |

**Note**: 6 clients is ~20-30% faster than 10 clients (C5) per experiment.

### Total Time for All Experiments

| Setup | Experiments | Total Time |
|-------|-------------|------------|
| **6 Clients** | **36** | **24-48 hours** |
| 10 Clients (C5 only) | 36 | 30-60 hours |
| 10 Clients (all C1-C5) | 180 | 150-300 hours |

---

## 💾 Resource Comparison

### Memory Usage

| Setup | RAM Required | GPU VRAM (if used) |
|-------|-------------|-------------------|
| 6 Clients | 3-6 GB | 2-3 GB |
| 10 Clients | 5-10 GB | 3-5 GB |

### Disk Space

| Setup | CSV Files | Total Size |
|-------|-----------|------------|
| 6 Clients | 108 (36×3) | ~500 MB |
| 10 Clients (C5) | 108 (36×3) | ~500 MB |
| 10 Clients (all) | 540 (180×3) | ~2-3 GB |

---

## 📈 Performance Characteristics

### Data Distribution per Client

**Dataset**: CIFAR-10 (50,000 training images)

| Setup | Images per Client | IID | Non-IID Impact |
|-------|------------------|-----|----------------|
| 6 Clients | ~8,333 | More data/client | Stronger non-IID |
| 10 Clients | ~5,000 | Less data/client | Moderate non-IID |

**Impact:**
- 6 clients: Each client has ~67% more data
- 10 clients: Better simulation of many-client scenarios

### Convergence Characteristics

**Expected Accuracy** (CIFAR-10, 500 rounds, FedAvg, homo):

| Setup | Expected Accuracy |
|-------|------------------|
| 6 Clients | 78-85% |
| 10 Clients | 75-82% |

**Note**: More clients often means slightly slower convergence but better generalization.

---

## 🎯 Experiment Matrix Comparison

### 6 Clients: 36 Experiments
```
6 Strategies × 6 Distributions × 1 Client Config = 36
```

| Strategy | homo | Dir(10.0) | Dir(1.0) | Dir(0.5) | Dir(0.1) | Dir(0.01) |
|----------|------|-----------|----------|----------|----------|-----------|
| FedAvg   | ✓    | ✓         | ✓        | ✓        | ✓        | ✓         |
| FedAvgM  | ✓    | ✓         | ✓        | ✓        | ✓        | ✓         |
| FedProx  | ✓    | ✓         | ✓        | ✓        | ✓        | ✓         |
| FedAdam  | ✓    | ✓         | ✓        | ✓        | ✓        | ✓         |
| FedAdagrad | ✓  | ✓         | ✓        | ✓        | ✓        | ✓         |
| FedYogi  | ✓    | ✓         | ✓        | ✓        | ✓        | ✓         |

### 10 Clients (Full): 180 Experiments
```
6 Strategies × 6 Distributions × 5 Client Configs = 180
```

Each of the 36 combinations above × 5 client configs (C1, C2, C3, C4, C5)

---

## 🚀 Quick Start Commands

### 6 Clients
```bash
# Setup
cp pyproject_6clients_500rounds.toml pyproject.toml

# Test
python run_6clients_experiments.py --quick --yes

# Full
python run_6clients_experiments.py --all --yes
```

### 10 Clients
```bash
# Setup
cp pyproject_experiment.toml pyproject.toml

# Test
python run_experiments.py --quick --yes

# Full (C5 only - 36 experiments)
python run_experiments.py \
  --strategies FedAvg FedAvgM FedProx FedAdam FedAdagrad FedYogi \
  --distributions homo Dir(10.0) Dir(1.0) Dir(0.5) Dir(0.1) Dir(0.01) \
  --client-configs C5 \
  --yes

# Full (all C1-C5 - 180 experiments)
python run_experiments.py --all --yes
```

---

## 📝 Decision Guide

### Choose **6 Clients** if:
- ✅ You want faster results
- ✅ You have limited computational resources
- ✅ You want to test all clients participating
- ✅ You're doing initial exploration
- ✅ Paper deadline is tight

### Choose **10 Clients** if:
- ✅ You want more realistic scenarios
- ✅ You want to test different participation rates
- ✅ You need comprehensive results
- ✅ You have time and resources
- ✅ You want to match common FL benchmarks

### Choose **10 Clients (C5 only)** if:
- ✅ You want balance between 6 and full 10-client setup
- ✅ Same number of experiments as 6-client (36)
- ✅ More clients for better simulation
- ✅ Don't need to test participation rates

---

## 🔄 Switching Between Configs

```bash
# View current config
head -30 pyproject.toml

# Switch to 6 clients
cp pyproject_6clients_500rounds.toml pyproject.toml

# Switch to 10 clients
cp pyproject_experiment.toml pyproject.toml

# Verify
grep "num-clients\|num-supernodes" pyproject.toml
```

---

## 📊 Results Comparison

### File Naming

**6 Clients:**
```
FedAvg_homo_6clients_global_20251222_120000.csv
FedAvg_Dir0p5_6clients_global_20251222_130000.csv
```

**10 Clients:**
```
FedAvg_homo_C5_global_20251222_120000.csv
FedAvg_Dir0p5_C3_global_20251222_130000.csv
```

### CSV Format (Same for Both)
```csv
round,loss,accuracy,precision,recall,f1
0,2.303,0.1000,0.0100,0.1000,0.0182
500,0.523,0.8250,0.8190,0.8210,0.8200
```

---

## 💡 Recommendations

### For Your Research

**If you want to match your table:**
- Table shows different #C values (1, 2, 3, 4, 5)
- Use **10 Clients setup** with all C1-C5 configs
- Total: 180 experiments
- Time: 150-300 hours (~1-2 weeks)

**If you want faster results first:**
- Use **6 Clients setup** for initial results
- 36 experiments in 24-48 hours
- Then extend to 10 clients if needed

**Hybrid Approach:**
1. Week 1: Run **6 clients** (36 experiments) → Get initial results
2. Week 2-3: Run **10 clients C5** (36 experiments) → Compare
3. Week 4-5: If needed, run remaining **10 clients C1-C4** (144 experiments)

---

## ✅ Summary Table

| Aspect | 6 Clients | 10 Clients (C5) | 10 Clients (All) |
|--------|-----------|-----------------|------------------|
| **Experiments** | 36 | 36 | 180 |
| **Time** | 24-48h | 30-60h | 150-300h |
| **RAM** | 3-6 GB | 5-10 GB | 5-10 GB |
| **Disk** | ~500 MB | ~500 MB | ~2-3 GB |
| **Complexity** | Simple | Medium | High |
| **Realism** | Low | Medium | High |
| **Speed** | Fast | Medium | Slow |
| **Recommended for** | Quick tests | Balanced | Comprehensive |

---

## 🎯 Final Recommendation

**Start with 6 clients, then expand if needed:**

```bash
# Phase 1: Quick validation (6 clients)
cp pyproject_6clients_500rounds.toml pyproject.toml
python run_6clients_experiments.py --all --yes
# → 24-48 hours, 36 experiments

# Phase 2: If results look good, expand to 10 clients C5
cp pyproject_experiment.toml pyproject.toml
python run_experiments.py --client-configs C5 --all --yes
# → 30-60 hours, 36 more experiments

# Phase 3: If needed for paper, complete all C1-C4
python run_experiments.py --all --yes
# → 120-240 hours, 144 more experiments
```

**Total: 216 experiments across all phases if you do everything! 🚀**
