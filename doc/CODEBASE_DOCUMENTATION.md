# Tài Liệu Codebase - Federated Learning với Flower và PyTorch

## Tổng Quan Dự Án

Đây là một dự án thí nghiệm **Federated Learning (FL)** sử dụng framework **Flower** và **PyTorch**, được thiết kế để huấn luyện mô hình CNN trên tập dữ liệu CIFAR-10 với nhiều chiến lược phân phối dữ liệu khác nhau (IID và Non-IID).

### Mục Đích
- Nghiên cứu hiệu suất của Federated Learning dưới các phân phối dữ liệu khác nhau
- So sánh các kịch bản Non-IID: Label Skew và Dirichlet Distribution
- Đo lường và ghi log các chỉ số chi tiết (accuracy, precision, recall, F1-score)

---

## Cấu Trúc Thư Mục

```
flowerAI/
├── pytorchexample/                 # Package chính chứa logic FL
│   ├── client_app_experiment.py   # Client-side application
│   ├── server_app_experiment.py   # Server-side application
│   ├── task.py                     # Model definition và data loading (HuggingFace)
│   ├── task_npy.py                 # Data loading từ .npy files (pre-partitioned)
│   ├── partitioner.py              # Các chiến lược phân vùng dữ liệu
│   ├── metrics.py                  # Tính toán metrics (accuracy, F1, etc.)
│   ├── logger.py                   # Ghi log vào CSV files
│   └── strategies.py               # Các FL strategies (FedAvg, etc.)
│
├── configs/                        # TOML configuration files
│   ├── test_homo.toml             # IID distribution
│   ├── test_C2.toml, test_C3.toml # Label skew (2-5 classes/client)
│   └── test_Dir0.1_npy.toml       # Dirichlet distributions
│
├── data/                           # Pre-partitioned .npy datasets
├── results/                        # Experiment results (CSV files)
├── models/                         # Saved model checkpoints
│
├── run_all_tests.py               # Script chạy tất cả experiments
├── run_all_tests_npy.py           # Script cho .npy datasets
├── export_partitions.py           # Export partitioned data to .npy
└── pyproject.toml                 # Project configuration

```

---

## Kiến Trúc Hệ Thống

### 1. Flower Framework Architecture

Hệ thống sử dụng kiến trúc Federated Learning của Flower:

```
┌─────────────────────────────────────────────────────────────┐
│                     Flower Server                            │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  server_app_experiment.py                             │  │
│  │  - Khởi tạo global model                              │  │
│  │  - Thực thi FedAvg strategy                           │  │
│  │  - Evaluate trên centralized test set                 │  │
│  │  - Log global metrics                                 │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ Aggregation (FedAvg)
                            ▼
        ┌───────────────────────────────────────────┐
        │          Flower Simulation Grid            │
        │     (Local simulation hoặc distributed)    │
        └───────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   Client 0    │   │   Client 1    │   │   Client N    │
│               │   │               │   │               │
│ - Load data   │   │ - Load data   │   │ - Load data   │
│ - Train       │   │ - Train       │   │ - Train       │
│ - Evaluate    │   │ - Evaluate    │   │ - Evaluate    │
└───────────────┘   └───────────────┘   └───────────────┘
```

### 2. Data Flow

```
1. Server khởi tạo global model
2. Server gửi model weights đến clients
3. Mỗi client:
   - Load local data partition (IID hoặc Non-IID)
   - Train model trên local data
   - Gửi updated weights về server
4. Server aggregate weights (FedAvg)
5. Server evaluate trên centralized test set
6. Lặp lại cho N rounds
```

---

## Chi Tiết Các Module

### 1. `task.py` - Model và Data Loading (HuggingFace Mode)

**Model: Simple CNN**
```python
class Net(nn.Module):
    - Conv1: 3 channels → 6 channels (5x5 kernel)
    - MaxPool: 2x2
    - Conv2: 6 channels → 16 channels (5x5 kernel)
    - MaxPool: 2x2
    - FC1: 16*5*5 → 120
    - FC2: 120 → 84
    - FC3: 84 → 10 (CIFAR-10 classes)
```

**Chức năng chính:**
- `load_data()`: Load partitioned CIFAR-10 data cho mỗi client
  - Hỗ trợ 2 modes: `huggingface` (partition on-the-fly) và `npy` (pre-partitioned)
  - Chia data: 80% train, 20% test cho mỗi client
- `load_centralized_dataset()`: Load toàn bộ test set cho server evaluation
- `train()`: Train model với SGD optimizer (momentum=0.9)
- `test()`: Evaluate model

### 2. `client_app_experiment.py` - Client Application

**Decorator-based Flower Client:**
```python
@app.train()
def train(msg: Message, context: Context):
    # 1. Load model từ server
    # 2. Load local data partition
    # 3. Train local model
    # 4. Calculate metrics (accuracy, precision, recall, F1)
    # 5. Return updated weights + metrics

@app.evaluate()
def evaluate(msg: Message, context: Context):
    # 1. Load model từ server
    # 2. Evaluate trên local test data
    # 3. Return evaluation metrics
```

**Metrics được gửi về server:**
- Training: `train_loss`, `train_accuracy`, `train_precision`, `train_recall`, `train_f1`
- Evaluation: `eval_loss`, `eval_acc`, `eval_precision`, `eval_recall`, `eval_f1`

### 3. `server_app_experiment.py` - Server Application

**Custom FedAvg Strategy:**
```python
class CustomFedAvg(FedAvg):
    def aggregate_fit():
        # Aggregate training results từ clients
        # (Currently commented out client training logging)

    def aggregate_evaluate():
        # Aggregate evaluation results từ clients
        # Calculate:
        #   - Global Accuracy: (1/N) * Σ(Accuracy_k)
        #   - Weighted Accuracy: Σ(n_k * Accuracy_k) / Σ(n_k)
```

**Global Evaluation:**
```python
def global_evaluate():
    # 1. Load centralized test set
    # 2. Evaluate global model
    # 3. Calculate comprehensive metrics
    # 4. Merge với client aggregate metrics
    # 5. Log tất cả vào CSV
```

**Metrics được log:**
- Centralized metrics: `loss`, `accuracy`, `precision`, `recall`, `f1`
- Client aggregate metrics: `global_accuracy`, `weighted_accuracy`

### 4. `partitioner.py` - Data Partitioning Strategies

**3 loại phân vùng dữ liệu:**

#### a. IID (Homogeneous)
```python
IidPartitioner(num_partitions=N)
```
- Phân chia đồng đều tất cả classes cho mọi clients
- Mỗi client có distribution tương tự dataset gốc

#### b. Label Skew (Non-IID Extreme)
```python
LabelSkewPartitioner(num_partitions=N, classes_per_client=k)
```
- Mỗi client chỉ có `k` classes (ví dụ: C2 = 2 classes/client)
- Tạo extreme label skew
- Ví dụ với 10 clients, 2 classes/client:
  - Client 0: classes [0, 1]
  - Client 1: classes [2, 3]
  - Client 2: classes [4, 5]
  - ...

#### c. Dirichlet Distribution (Non-IID Moderate)
```python
DirichletPartitioner(alpha=α)
```
- Sử dụng Dirichlet distribution để phân chia
- `α` càng nhỏ → càng Non-IID:
  - `α = 0.1`: Very non-IID
  - `α = 0.5`: Moderate non-IID
  - `α = 1.0`: Mild non-IID
  - `α = 10.0`: Nearly IID

### 5. `metrics.py` - Metrics Calculation

**Comprehensive Metrics:**
```python
calculate_metrics(net, dataloader, device):
    Returns:
    - loss: Cross-entropy loss
    - accuracy: Correct predictions / Total samples
    - precision: Macro-average precision
    - recall: Macro-average recall
    - f1: Macro-average F1-score
```

**Weight Change Metrics (currently disabled):**
```python
calculate_weight_metrics(current_weights, previous_weights):
    Returns:
    - weight_norm: L2 norm của weights
    - weight_change: L2 norm của weight difference
    - weight_relative_change: Relative change
```

### 6. `logger.py` - Experiment Logger

**3 CSV files được tạo cho mỗi experiment:**

#### a. Global Metrics CSV
```
round, loss, accuracy, precision, recall, f1, global_accuracy, weighted_accuracy
```

#### b. Client Metrics CSV
```
round, client_id, phase, loss, accuracy, precision, recall, f1, num_examples
```

#### c. Weight Metrics CSV (currently not used)
```
round, weight_norm, weight_change, weight_relative_change
```

**Naming convention:**
```
{experiment_name}_{type}_{timestamp}.csv
Ví dụ: test_homo_global_20260103_150129.csv
```

---

## Configuration Files (TOML)

### Cấu trúc cơ bản:

```toml
[tool.flwr.app.config]
num-server-rounds = 500           # Số rounds FL
fraction-train = 1.0              # % clients tham gia training
fraction-evaluate = 1.0           # % clients tham gia evaluation
min-train-nodes = 10              # Minimum clients cho training
min-evaluate-nodes = 10           # Minimum clients cho evaluation

local-epochs = 1                  # Số epochs train local
learning-rate = 0.01              # Learning rate
batch-size = 32                   # Batch size

strategy = "FedAvg"               # FL strategy
distribution = "homo"             # Data distribution type
experiment-name = "test_homo"     # Experiment name
num-clients = 10                  # Số lượng clients
data-source = "huggingface"       # "huggingface" hoặc "npy"
```

### Các loại distributions được hỗ trợ:

1. **Homogeneous (IID)**
   ```toml
   distribution = "homo"
   ```

2. **Label Skew**
   ```toml
   distribution = "C2"  # 2 classes per client
   distribution = "C3"  # 3 classes per client
   distribution = "C4"  # 4 classes per client
   ```

3. **Dirichlet**
   ```toml
   distribution = "Dir0.1"   # Very non-IID
   distribution = "Dir0.5"   # Moderate non-IID
   distribution = "Dir1.0"   # Mild non-IID
   distribution = "Dir10.0"  # Nearly IID
   ```

---

## Running Experiments

### 1. Single Experiment

```bash
flwr run . --run-config configs/test_homo.toml
```

### 2. All Experiments (Sequential)

```bash
python run_all_tests.py
```

Hoặc với .npy datasets:
```bash
python run_all_tests_npy.py
```

### 3. Export Data to .npy Format

```bash
python export_partitions.py
```

---

## Experiment Workflow

### Bước 1: Configuration
```python
# pyproject.toml hoặc TOML config file
- Định nghĩa số clients, rounds, learning rate
- Chọn distribution type
- Set data source (huggingface hoặc npy)
```

### Bước 2: Data Partitioning
```python
# Tự động trong load_data()
if data_source == "huggingface":
    partitioner = get_partitioner(distribution, num_partitions)
    # Partition on-the-fly
else:  # npy
    # Load pre-partitioned data từ .npy files
```

### Bước 3: Federated Training
```python
for round in range(num_rounds):
    # 1. Server gửi global model
    # 2. Clients train locally
    # 3. Server aggregates với FedAvg
    # 4. Server evaluates
    # 5. Log metrics
```

### Bước 4: Results
```
results/
├── test_homo_global_20260103_150129.csv      # Global metrics
├── test_homo_client_20260103_150129.csv      # Per-client metrics
├── test_homo_weight_20260103_150129.csv      # Weight changes
└── test_homo_config_20260103_150129.txt      # Config snapshot
```

---

## Key Metrics Explained

### 1. Centralized Metrics (từ server evaluation)
- **Loss**: Cross-entropy loss trên centralized test set
- **Accuracy**: Overall accuracy trên toàn bộ test set
- **Precision/Recall/F1**: Macro-averaged metrics

### 2. Client Aggregate Metrics
- **Global Accuracy**: `(1/N) * Σ(Accuracy_k)`
  - Average của accuracies từ tất cả clients
  - Treat mọi clients equally

- **Weighted Accuracy**: `Σ(n_k * Accuracy_k) / Σ(n_k)`
  - Weighted average theo số samples của mỗi client
  - Clients với nhiều data hơn có weight cao hơn

### 3. Sự khác biệt giữa Accuracy vs Global/Weighted Accuracy
- **Accuracy**: Đánh giá global model trên centralized test set
- **Global/Weighted Accuracy**: Aggregate từ client evaluations trên local test sets
- Có thể khác nhau do:
  - Distribution shift giữa local và global test sets
  - Non-IID data distributions

---

## Data Sources

### 1. HuggingFace Mode (Default)
```python
data_source = "huggingface"
```
- Load CIFAR-10 từ HuggingFace Datasets
- Partition on-the-fly với specified strategy
- Flexible nhưng chậm hơn
- Offline mode: `HF_DATASETS_OFFLINE=1`

### 2. NPY Mode (Pre-partitioned)
```python
data_source = "npy"
```
- Load từ pre-partitioned `.npy` files
- Nhanh hơn, consistent hơn
- Cần export trước bằng `export_partitions.py`
- Structure:
  ```
  data/
  ├── homo_10clients/
  │   ├── client_0_train_images.npy
  │   ├── client_0_train_labels.npy
  │   ├── client_0_test_images.npy
  │   └── ...
  └── C2_10clients/
      └── ...
  ```

---

## Dependencies

### Core Frameworks
- **Flower**: `flwr[simulation]>=1.24.0` - FL framework
- **PyTorch**: `torch==2.8.0` - Deep learning
- **torchvision**: `0.23.0` - Vision utilities

### Data & ML
- **flwr-datasets**: Data partitioning
- **scikit-learn**: Metrics calculation
- **datasets**: HuggingFace datasets

---

## Experiment Analysis Files

Có các file phân tích đã được tạo:
- `NPY_COMPLETE_SUMMARY.md`: Tổng kết experiments với .npy data
- `COMPARISON_6_vs_10_CLIENTS.md`: So sánh 6 vs 10 clients
- `ANALYSIS_PARADOX.md`: Phân tích các hiện tượng bất thường
- `PROBLEM_ANALYSIS.md`: Phân tích các vấn đề
- `ROOT_CAUSE_ANALYSIS.md`: Phân tích nguyên nhân gốc rễ

---

## Key Design Decisions

### 1. Tại sao có 2 data modes?
- **HuggingFace**: Flexible, dễ modify partitioning strategies
- **NPY**: Fast, reproducible, consistent partitions across runs

### 2. Tại sao log cả client metrics và global metrics?
- Client metrics: Hiểu performance của từng client
- Global metrics: Hiểu overall system performance
- So sánh để phát hiện distribution issues

### 3. Tại sao weight metrics bị disable?
```python
# In server_app_experiment.py (lines 166, 237-240, 250)
# previous_weights = deepcopy(global_model.state_dict())  # Commented
# weight_metrics = calculate_weight_metrics(...)          # Commented
```
- Memory overhead với large models
- Có thể enable nếu cần analyze convergence

### 4. FedAvg Strategy
- Default aggregation: Weighted average theo số samples
- `fraction_train/evaluate = 1.0`: All clients participate
- Synchronous: Chờ tất cả clients mỗi round

---

## Common Issues & Solutions

### Issue 1: Ray environment variables
```python
# run_all_tests.py line 10
os.environ['RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO'] = '0'
```
- Fix cho Windows compatibility với Ray backend

### Issue 2: HuggingFace offline mode
```python
# task.py lines 81, 162
os.environ['HF_DATASETS_OFFLINE'] = '1'
```
- Tránh re-download datasets mỗi lần chạy

### Issue 3: NaN/Inf in metrics
```python
# metrics.py lines 40-42, 117-123
if torch.isnan(loss) or torch.isinf(loss):
    print(f"Warning: NaN/Inf detected")
    continue
```
- Safety checks để tránh crash

---

## Extension Points

### 1. Add New FL Strategy
```python
# strategies.py
def get_strategy(strategy_name, **kwargs):
    if strategy_name == "FedProx":
        return FedProx(...)
    elif strategy_name == "FedAvgM":
        return FedAvgM(...)
```

### 2. Add New Partitioning Strategy
```python
# partitioner.py
class CustomPartitioner:
    def __init__(self, ...):
        pass

    def load_partition(self, partition_id):
        # Custom logic
        pass
```

### 3. Add New Metrics
```python
# metrics.py
def calculate_custom_metrics(net, dataloader, device):
    # Add confusion matrix, per-class accuracy, etc.
    pass
```

### 4. Change Model Architecture
```python
# task.py - modify Net class
class Net(nn.Module):
    def __init__(self):
        # Different architecture (ResNet, VGG, etc.)
        pass
```

---

## Testing và Validation

### Quick Test (10 rounds)
```toml
num-server-rounds = 10
```

### Full Experiment (500 rounds)
```toml
num-server-rounds = 500
```

### Validation Checklist
1. Check CSV files được tạo trong `results/`
2. Verify final model saved: `{experiment_name}_final_model.pt`
3. Check metrics convergence
4. Compare global_accuracy vs weighted_accuracy
5. Analyze per-client performance

---

## Performance Considerations

### Memory
- Each client loads own partition vào memory
- Server holds global model
- Với 10 clients, simulation cần ~4-8GB RAM

### Speed
- NPY mode nhanh hơn HuggingFace mode ~2-3x
- GPU acceleration: Set `num-gpus` in federation config
- Parallel clients: Automatically handled by Flower

### Disk Space
- NPY datasets: ~500MB per distribution
- Results CSVs: ~10-50MB per experiment
- Model checkpoints: ~1MB per model

---

## Summary

Đây là một **well-structured Federated Learning research codebase** với:
- ✅ Modular design: Tách biệt client, server, data, metrics
- ✅ Flexible data partitioning: IID và multiple Non-IID strategies
- ✅ Comprehensive logging: CSV files cho analysis
- ✅ Dual data modes: HuggingFace và NPY
- ✅ Production-ready: Error handling, safety checks
- ✅ Experiment automation: Scripts để chạy multiple tests

**Use Cases:**
1. Research on Non-IID data in FL
2. Compare different aggregation strategies
3. Study convergence under different data distributions
4. Benchmark FL performance

**Next Steps:**
1. Run experiments với different configurations
2. Analyze results từ CSV files
3. Visualize metrics (accuracy trends, per-client performance)
4. Extend với custom strategies hoặc models
