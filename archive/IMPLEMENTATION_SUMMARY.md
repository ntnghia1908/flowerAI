# Implementation Summary - FL Experiments Framework

## Tổng quan

Framework hoàn chỉnh để chạy thí nghiệm Federated Learning với:
- ✅ 6 thuật toán FL
- ✅ 6 phân phối dữ liệu (IID + 5 Dirichlet)
- ✅ 5 cấu hình số clients
- ✅ Metrics đầy đủ: loss, accuracy, precision, recall, F1
- ✅ Logging tự động ra CSV
- ✅ Visualization và analysis tools

## Files đã tạo

### 1. Core Modules (pytorchexample/)

| File | Mô tả | Chức năng |
|------|-------|-----------|
| **metrics.py** | Tính toán metrics | Tính loss, accuracy, precision, recall, F1, weight changes |
| **logger.py** | CSV logging | Log metrics ra 3 file CSV (global, client, weight) |
| **partitioner.py** | Data partitioning | IID và Dirichlet partitioning với các alpha khác nhau |
| **strategies.py** | FL strategies | FedAvg, FedAvgM, FedProx, FedAdam, FedAdagrad, FedYogi |
| **server_app_experiment.py** | Experimental server | Server với comprehensive logging |
| **client_app_experiment.py** | Experimental client | Client với comprehensive logging |
| **task.py** | (Đã update) | Thêm support cho custom partitioner |

### 2. Scripts

| File | Mô tả | Cách dùng |
|------|-------|-----------|
| **run_experiments.py** | Automated experiment runner | Chạy batch experiments tự động |
| **test_experiment.py** | Quick test script | Test nhanh framework (10 rounds) |
| **analyze_results.py** | Results analysis | Phân tích và visualize kết quả |

### 3. Configuration

| File | Mô tả |
|------|-------|
| **pyproject_experiment.toml** | Config cho experiments |
| **requirements.txt** | Python dependencies |

### 4. Documentation

| File | Mô tả |
|------|-------|
| **QUICK_START.md** | Hướng dẫn nhanh |
| **EXPERIMENTS_README.md** | Tài liệu chi tiết đầy đủ |
| **IMPLEMENTATION_SUMMARY.md** | File này - tổng quan implementation |

## Kiến trúc

```
┌─────────────────────────────────────────────────────────┐
│                   run_experiments.py                     │
│          (Orchestrates multiple experiments)             │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ├──> Strategy Selection (strategies.py)
                  ├──> Data Partitioning (partitioner.py)
                  └──> Flower Run Command
                        │
        ┌───────────────┴──────────────┐
        │                              │
┌───────▼────────┐            ┌───────▼────────┐
│  ServerApp     │            │   ClientApp    │
│  (experiment)  │◄──────────►│  (experiment)  │
└───────┬────────┘            └───────┬────────┘
        │                              │
        │                              │
┌───────▼────────┐            ┌───────▼────────┐
│   Metrics      │            │   Metrics      │
│  Calculation   │            │  Calculation   │
└───────┬────────┘            └───────┬────────┘
        │                              │
        └──────────┬───────────────────┘
                   │
           ┌───────▼────────┐
           │  CSV Logger    │
           └───────┬────────┘
                   │
           ┌───────▼────────┐
           │  results/      │
           │  - global.csv  │
           │  - client.csv  │
           │  - weight.csv  │
           └────────────────┘
```

## Workflow

### 1. Single Experiment

```
1. Configure experiment parameters
   ├─> Strategy: FedAvg, FedAvgM, etc.
   ├─> Distribution: homo, Dir(α)
   ├─> Clients config: C1-C5
   └─> Rounds: 500

2. Initialize components
   ├─> Create partitioner
   ├─> Create strategy
   ├─> Create logger
   └─> Load global model

3. Run FL rounds
   For each round:
     ├─> Server sends model to clients
     ├─> Clients train locally
     │   ├─> Calculate training metrics
     │   └─> Log client metrics
     ├─> Server aggregates updates
     ├─> Server evaluates global model
     │   ├─> Calculate global metrics
     │   ├─> Calculate weight changes
     │   └─> Log to CSV
     └─> Print progress

4. Save results
   ├─> Save final model
   ├─> Close CSV files
   └─> Print summary
```

### 2. Batch Experiments

```
1. Parse command line arguments
   └─> Select strategies, distributions, configs

2. Generate experiment combinations
   └─> Cartesian product of all parameters

3. For each experiment:
   ├─> Build flwr run command
   ├─> Execute experiment
   ├─> Track success/failure
   └─> Move to next

4. Print batch summary
   └─> Total, successful, failed experiments
```

## Metrics Collected

### Global Metrics (Server-side evaluation)
```csv
round,loss,accuracy,precision,recall,f1
1,2.3026,0.1000,0.1000,0.1000,0.1000
2,2.1234,0.2500,0.2400,0.2450,0.2425
...
```

### Client Metrics (Client-side evaluation)
```csv
round,client_id,phase,loss,accuracy,precision,recall,f1,num_examples
1,0,train,2.1,0.25,0.24,0.23,0.235,4000
1,0,evaluate,2.2,0.23,0.22,0.21,0.215,1000
...
```

### Weight Metrics (Model changes)
```csv
round,weight_norm,weight_change,weight_relative_change
1,125.456,0.0,0.0
2,126.789,5.234,0.0417
...
```

## Key Features

### 1. Metrics Calculation
- **Precision**: TP / (TP + FP) - macro average
- **Recall**: TP / (TP + FN) - macro average
- **F1 Score**: 2 * (precision * recall) / (precision + recall)
- **Weight metrics**: L2 norm, absolute change, relative change

### 2. Data Partitioning
- **IID (homo)**: Uniform random distribution
- **Dirichlet(α)**: Non-IID with varying degrees
  - α = 10.0: Mild non-IID
  - α = 1.0: Moderate non-IID
  - α = 0.5: Strong non-IID
  - α = 0.1: Very strong non-IID
  - α = 0.01: Extreme non-IID

### 3. FL Strategies
- **FedAvg**: Vanilla federated averaging
- **FedAvgM**: FedAvg + server momentum
- **FedProx**: FedAvg + proximal term (µ)
- **FedAdam**: FedAvg + Adam optimizer
- **FedAdagrad**: FedAvg + Adagrad optimizer
- **FedYogi**: FedAvg + Yogi optimizer

### 4. Automatic Logging
- Timestamped filenames
- Separate files for different metric types
- Config file with experiment parameters
- No manual intervention needed

## Usage Examples

### Quick Test (2-3 minutes)
```bash
python test_experiment.py
```

### Single Experiment
```bash
flwr run . local-simulation --run-config \
  strategy=FedAvg \
  distribution=homo \
  num-server-rounds=500 \
  min-fit-clients=5
```

### Batch Experiments - Medium Test
```bash
python run_experiments.py --medium
```

### Full Table Reproduction
```bash
python run_experiments.py \
  --strategies FedAvg FedAvgM FedProx FedAdam FedAdagrad FedYogi \
  --distributions homo Dir(10.0) Dir(1.0) Dir(0.5) Dir(0.1) Dir(0.01) \
  --client-configs C1 C2 C3 C4 C5 \
  --num-rounds 500
```
Total: **180 experiments**

### Analyze Results
```bash
# Summary of all experiments
python analyze_results.py

# Compare specific pattern
python analyze_results.py --pattern "FedAvg*" --compare --metric accuracy

# Analyze single experiment
python analyze_results.py --pattern "FedAvg_homo_C5"
```

## Performance Considerations

### Time Estimates
- 1 round (~10 clients): ~3-6 seconds
- 500 rounds: ~30-60 minutes
- 180 experiments: ~90-180 hours (3-7 days)

### Optimization Tips
1. **Use GPU**: Add `--gpu` flag
2. **Reduce rounds**: Test with 100 rounds first
3. **Parallel execution**: Run multiple scripts in parallel
4. **Batch size**: Adjust based on memory
5. **Fewer clients**: Use `--total-clients 5` for faster testing

### Memory Management
- Each client: ~500MB-1GB RAM
- 10 clients: ~5-10GB RAM total
- GPU: ~2-4GB VRAM per experiment

## Extending the Framework

### Add New Strategy
```python
# In strategies.py
from flwr.serverapp.strategy import NewStrategy

def get_strategy(strategy_name, ...):
    ...
    elif strategy_name == "NewStrategy":
        return NewStrategy(**common_params)
```

### Add New Metric
```python
# In metrics.py
def calculate_metrics(...):
    ...
    new_metric = custom_calculation(predictions, labels)
    return {
        ...,
        'new_metric': new_metric
    }
```

### Add New Partitioner
```python
# In partitioner.py
def get_partitioner(distribution_type, num_partitions):
    ...
    elif distribution_type == "custom":
        return CustomPartitioner(num_partitions)
```

## Troubleshooting

### Common Issues

1. **Import errors**: Run `pip install -e .`
2. **No CSV files**: Check `results/` directory exists
3. **Slow execution**: Reduce rounds or use GPU
4. **Out of memory**: Reduce batch size or clients
5. **Wrong config**: Make sure using `pyproject_experiment.toml`

### Debug Mode
```bash
# Run single quick experiment
python run_experiments.py --quick

# Check a specific experiment
python test_experiment.py
```

## Next Steps

1. ✅ Test framework: `python test_experiment.py`
2. ✅ Run medium test: `python run_experiments.py --medium`
3. ✅ Analyze results: `python analyze_results.py`
4. ✅ Run full experiments: `python run_experiments.py --all`
5. ✅ Create publication-ready plots and tables

## References

- [Flower Framework](https://flower.ai/)
- [FedAvg Paper](https://arxiv.org/abs/1602.05629)
- [FedProx Paper](https://arxiv.org/abs/1812.06127)
- [Adaptive Federated Optimization](https://arxiv.org/abs/2003.00295)
- [Non-IID Data in Federated Learning](https://arxiv.org/abs/1806.00582)

---

**Author**: Claude Code
**Date**: 2025-12-21
**Version**: 1.0.0
