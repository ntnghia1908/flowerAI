# Code Overview - Tổng quan toàn bộ codebase

Giải thích cấu trúc và vai trò của từng file trong project.

---

## 📁 Cấu trúc Project

```
quickstart-pytorch/
├── pytorchexample/           # Core FL implementation
│   ├── __init__.py
│   ├── task.py              # Model & training logic
│   ├── client_app.py        # Client-side FL
│   ├── server_app.py        # Server-side FL (basic)
│   ├── client_app_experiment.py  # Client với metrics
│   ├── server_app_experiment.py  # Server với logging
│   ├── metrics.py           # Tính precision, recall, F1
│   ├── logger.py            # CSV logging
│   ├── partitioner.py       # Data partitioning
│   └── strategies.py        # FL algorithms
│
├── results/                  # Auto-generated
│   └── *.csv
│
├── pyproject.toml           # Config chính
└── requirements.txt         # Dependencies
```

---

## 🎯 Core Files Explanation

### 1. `task.py` - Model & Training Logic

**Vai trò**: Định nghĩa model architecture và training/testing logic

**Components chính**:

#### A. Model (Class `Net`)
```python
class Net(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(3, 6, 5)      # Conv layer 1
        self.conv2 = nn.Conv2d(6, 16, 5)     # Conv layer 2
        self.fc1 = nn.Linear(16*5*5, 120)   # FC layer 1
        self.fc2 = nn.Linear(120, 84)        # FC layer 2
        self.fc3 = nn.Linear(84, 10)         # Output (10 classes)
```

**Architecture**:
```
Input (3x32x32 RGB image)
    ↓
Conv1 (6 filters, 5x5) + ReLU + MaxPool
    ↓
Conv2 (16 filters, 5x5) + ReLU + MaxPool
    ↓
Flatten
    ↓
FC1 (120 units) + ReLU
    ↓
FC2 (84 units) + ReLU
    ↓
FC3 (10 units - CIFAR-10 classes)
    ↓
Output (logits)
```

#### B. Data Loading (`load_data`)
```python
def load_data(partition_id, num_partitions, batch_size):
    # 1. Tạo partitioner (IID hoặc Dirichlet)
    # 2. Load CIFAR-10 partition cho client
    # 3. Split 80/20 train/val
    # 4. Return DataLoaders
```

**Flow**:
```
CIFAR-10 (50,000 images)
    ↓
Partition by IidPartitioner hoặc DirichletPartitioner
    ↓
Client 0: partition 0 (5,000 images)
Client 1: partition 1 (5,000 images)
...
    ↓
Each partition split:
  - Train: 4,000 (80%)
  - Val: 1,000 (20%)
```

#### C. Training (`train`)
```python
def train(net, trainloader, epochs, lr, device):
    # 1. Set model to train mode
    # 2. Create optimizer (SGD) và loss (CrossEntropy)
    # 3. For each epoch:
    #      - For each batch:
    #          - Forward pass
    #          - Compute loss
    #          - Backward pass
    #          - Update weights
    # 4. Return average loss
```

#### D. Testing (`test`)
```python
def test(net, testloader, device):
    # 1. Set model to eval mode
    # 2. Disable gradient computation
    # 3. For each batch:
    #      - Forward pass
    #      - Compute loss
    #      - Count correct predictions
    # 4. Return (loss, accuracy)
```

---

### 2. `client_app.py` - Client Side Logic

**Vai trò**: Xử lý training và evaluation ở phía client

**Chi tiết**: Xem [CODE_EXPLAINED.md](CODE_EXPLAINED.md) - Giải thích đầy đủ

**Tóm tắt**:
- **`@app.train()`**: Nhận global weights → Train local → Gửi updated weights
- **`@app.evaluate()`**: Nhận weights → Evaluate → Gửi metrics

**Message Flow**:
```
Server ──(global weights)──> Client
Client ──(train local)────> Updated weights
Client ──(updated weights)──> Server

Server ──(aggregated weights)──> Client
Client ──(evaluate)──────────> Metrics only
Client ──(metrics)──────────> Server
```

---

### 3. `server_app.py` - Server Side Logic (Basic)

**Vai trò**: Điều phối FL process, aggregate weights

**Components**:

#### A. Main Function
```python
@app.main()
def main(grid: Grid, context: Context):
    # 1. Read config (rounds, lr, fraction_evaluate)
    # 2. Load global model (random init)
    # 3. Create strategy (FedAvg)
    # 4. Start FL process
    # 5. Save final model
```

#### B. Global Evaluation
```python
def global_evaluate(server_round, arrays):
    # 1. Load model với aggregated weights
    # 2. Load centralized test set
    # 3. Evaluate on test set
    # 4. Return metrics (loss, accuracy)
```

**Flow**:
```
Round 1:
  Server: Send global weights to clients
  Clients: Train local → Send updated weights
  Server: Aggregate weights (FedAvg)
  Server: Evaluate global model

Round 2:
  Server: Send aggregated weights...
  ...

Round N:
  Final model saved
```

---

### 4. `server_app_experiment.py` - Server với Full Logging

**Vai trò**: Như `server_app.py` nhưng thêm comprehensive logging

**Additions**:

#### A. Comprehensive Metrics
```python
from pytorchexample.metrics import calculate_metrics

metrics = calculate_metrics(model, dataloader, device)
# Returns: {loss, accuracy, precision, recall, f1}
```

#### B. CSV Logging
```python
from pytorchexample.logger import ExperimentLogger

logger = ExperimentLogger("FedAvg_homo_C5")
logger.log_global_metrics(round, metrics)
logger.log_weight_metrics(round, weight_metrics)
```

#### C. Weight Tracking
```python
from pytorchexample.metrics import calculate_weight_metrics

weight_metrics = calculate_weight_metrics(current_weights, previous_weights)
# Returns: {weight_norm, weight_change, weight_relative_change}
```

**Enhanced Output**:
```
Round   1 | Loss: 2.3026 | Acc: 0.1000 | F1: 0.0182 | Weight Change: 0.5736
Round   2 | Loss: 2.1904 | Acc: 0.1367 | F1: 0.0477 | Weight Change: 0.5026
...
```

---

### 5. `client_app_experiment.py` - Client với Full Metrics

**Vai trò**: Như `client_app.py` nhưng tính thêm precision, recall, F1

**Enhancements**:

```python
from pytorchexample.metrics import calculate_metrics

# Training metrics
train_metrics = calculate_metrics(model, trainloader, device)
# Returns: {loss, accuracy, precision, recall, f1, num_examples}

# Evaluation metrics
eval_metrics = calculate_metrics(model, valloader, device)
# Returns: {loss, accuracy, precision, recall, f1, num_examples}
```

**Richer Metrics**:
```python
# Before (basic)
metrics = {"train_loss": 1.234, "num-examples": 4000}

# After (comprehensive)
metrics = {
    "train_loss": 1.234,
    "train_accuracy": 0.567,
    "train_precision": 0.543,
    "train_recall": 0.556,
    "train_f1": 0.549,
    "num-examples": 4000
}
```

---

### 6. `metrics.py` - Advanced Metrics Calculation

**Vai trò**: Tính toán metrics nâng cao (precision, recall, F1)

**Functions**:

#### A. `calculate_metrics()`
```python
def calculate_metrics(net, dataloader, device, num_classes=10):
    # 1. Run inference on all data
    # 2. Collect predictions và labels
    # 3. Calculate:
    #    - Accuracy: correct / total
    #    - Precision: TP / (TP + FP) - macro average
    #    - Recall: TP / (TP + FN) - macro average
    #    - F1: 2 * (precision * recall) / (precision + recall)
    # 4. Return dict với all metrics
```

**Ví dụ**:
```python
metrics = calculate_metrics(model, dataloader, device)
# {
#   'loss': 1.234,
#   'accuracy': 0.567,
#   'precision': 0.543,  # Macro average across 10 classes
#   'recall': 0.556,     # Macro average
#   'f1': 0.549          # Macro average
# }
```

#### B. `calculate_weight_metrics()`
```python
def calculate_weight_metrics(current_weights, previous_weights):
    # Track how much model changed
    # 1. Calculate L2 norm of weights
    # 2. Calculate change from previous round
    # 3. Calculate relative change
```

**Use case**: Monitor model convergence
```python
# Round 1: weight_change = 5.12 (large)
# Round 10: weight_change = 1.23 (medium)
# Round 100: weight_change = 0.15 (small, converging)
```

---

### 7. `logger.py` - CSV Logging System

**Vai trò**: Tự động log metrics ra CSV files

**Class `ExperimentLogger`**:

```python
logger = ExperimentLogger("FedAvg_homo_C5")
```

**Creates 3 CSV files**:
1. `FedAvg_homo_C5_global_{timestamp}.csv`
2. `FedAvg_homo_C5_client_{timestamp}.csv`
3. `FedAvg_homo_C5_weight_{timestamp}.csv`

**Methods**:

#### A. `log_global_metrics()`
```python
logger.log_global_metrics(round_num=1, metrics={
    'loss': 2.303,
    'accuracy': 0.100,
    'precision': 0.010,
    'recall': 0.100,
    'f1': 0.018
})
```

**Output** (`*_global_*.csv`):
```csv
round,loss,accuracy,precision,recall,f1
1,2.303,0.100,0.010,0.100,0.018
```

#### B. `log_client_metrics()`
```python
logger.log_client_metrics(
    round_num=1,
    client_id=0,
    phase='train',
    metrics={...}
)
```

**Output** (`*_client_*.csv`):
```csv
round,client_id,phase,loss,accuracy,precision,recall,f1,num_examples
1,0,train,2.1,0.25,0.24,0.23,0.235,4000
1,0,evaluate,2.2,0.23,0.22,0.21,0.215,1000
```

#### C. `log_weight_metrics()`
```python
logger.log_weight_metrics(round_num=1, metrics={
    'weight_norm': 8.919,
    'weight_change': 5.116,
    'weight_relative_change': 0.574
})
```

**Output** (`*_weight_*.csv`):
```csv
round,weight_norm,weight_change,weight_relative_change
1,8.919,5.116,0.574
```

---

### 8. `partitioner.py` - Data Partitioning

**Vai trò**: Chia dữ liệu cho clients (IID hoặc non-IID)

**Function `get_partitioner()`**:

#### A. IID Partitioning
```python
partitioner = get_partitioner("homo", num_partitions=10)
```

**Result**:
```
Client 0: Uniform random 10% (cân bằng các classes)
Client 1: Uniform random 10%
...
```

#### B. Dirichlet Partitioning
```python
partitioner = get_partitioner("Dir(0.5)", num_partitions=10)
```

**Effect**:
- `α = 10.0`: Gần IID (mỗi client có nhiều classes)
- `α = 0.5`: Strong non-IID (mỗi client có ít classes)
- `α = 0.01`: Extreme non-IID (mỗi client chủ yếu 1-2 classes)

**Ví dụ Dir(0.1)**:
```
Client 0: 90% class 0, 10% others
Client 1: 80% class 3, 20% others
Client 2: 85% class 7, 15% others
...
```

---

### 9. `strategies.py` - FL Algorithms

**Vai trò**: Các thuật toán aggregation khác nhau

**Function `get_strategy()`**:

#### Supported Strategies:

**A. FedAvg** (Baseline)
```python
strategy = get_strategy("FedAvg")
# Simple averaging: new_weights = mean(client_weights)
```

**B. FedAvgM** (Momentum)
```python
strategy = get_strategy("FedAvgM", server_momentum=0.9)
# Adds momentum: velocity = β*velocity + (1-β)*update
```

**C. FedProx** (Proximal)
```python
strategy = get_strategy("FedProx", proximal_mu=0.01)
# Adds penalty: keep local weights close to global
```

**D. FedAdam** (Adaptive)
```python
strategy = get_strategy("FedAdam", eta=0.01)
# Uses Adam optimizer on server side
```

**E. FedAdagrad** & **FedYogi**
```python
strategy = get_strategy("FedAdagrad", eta=0.01)
strategy = get_strategy("FedYogi", eta=0.01)
# Other adaptive optimizers
```

---

## 🔄 Complete FL Round Flow

### Initialization
```
1. Server creates global model (random weights)
2. Server selects clients for round 1
3. Server sends global weights to selected clients
```

### Round Execution

#### Phase 1: Training
```
For each selected client:
  1. Client receives global weights
  2. @app.train() is called
  3. Client loads its data partition
  4. Client trains for N epochs
  5. Client sends updated weights back
```

#### Phase 2: Aggregation
```
Server:
  1. Receives all client weights
  2. Applies aggregation strategy (e.g., FedAvg)
  3. new_global_weights = aggregate(client_weights)
```

#### Phase 3: Evaluation
```
For each selected client:
  1. Client receives aggregated weights
  2. @app.evaluate() is called
  3. Client evaluates on validation set
  4. Client sends metrics back

Server:
  1. Receives all client metrics
  2. Computes aggregated metrics
  3. Evaluates global model on test set
  4. Logs metrics (if using experimental server)
```

#### Phase 4: Next Round
```
Server:
  1. Check if max_rounds reached
  2. If not, select clients for next round
  3. Repeat from Phase 1
  4. If yes, save final model
```

---

## 📊 Data Flow Summary

```
┌──────────────────────────────────────────────────┐
│                   DATASET                        │
│   CIFAR-10: 50,000 train, 10,000 test          │
└─────────────────┬────────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        │   Partitioner     │
        │  (IID or Dirichlet)│
        └─────────┬─────────┘
                  │
     ┌────────────┼────────────┐
     │            │            │
┌────▼────┐  ┌───▼────┐  ┌───▼────┐
│Client 0 │  │Client 1│  │Client 9│
│5000 img │  │5000 img│  │5000 img│
└────┬────┘  └───┬────┘  └───┬────┘
     │           │            │
  80/20       80/20        80/20
  Split       Split        Split
     │           │            │
┌────▼────┐  ┌───▼────┐  ┌───▼────┐
│Train:   │  │Train:  │  │Train:  │
│4000 img │  │4000 img│  │4000 img│
└────┬────┘  └───┬────┘  └───┬────┘
     │           │            │
┌────▼────┐  ┌───▼────┐  ┌───▼────┐
│Val:     │  │Val:    │  │Val:    │
│1000 img │  │1000 img│  │1000 img│
└─────────┘  └────────┘  └────────┘
```

---

## 💡 Key Takeaways

### 1. Separation of Concerns
- **task.py**: Pure ML (model, train, test)
- **client_app.py**: FL client logic
- **server_app.py**: FL server logic
- **metrics.py**: Advanced metrics
- **logger.py**: Persistence

### 2. Two Modes
- **Basic** (`client_app.py` + `server_app.py`): Simple, no logging
- **Experimental** (`*_experiment.py`): Full metrics, CSV logging

### 3. Flexibility
- Multiple strategies (FedAvg, FedProx, etc.)
- Multiple distributions (IID, Dirichlet)
- Configurable via `pyproject.toml`

### 4. Privacy Preserved
- Raw data never leaves client
- Only model weights are shared
- Each client has separate partition

---

## 📚 Related Documentation

- **[CODE_EXPLAINED.md](CODE_EXPLAINED.md)** - Chi tiết `client_app.py`
- **[REFACTOR_SUMMARY.md](REFACTOR_SUMMARY.md)** - Cấu trúc project
- **[README_MAIN.md](README_MAIN.md)** - Quick start guide

---

**Hiểu code = Hiểu FL! 🎓**
