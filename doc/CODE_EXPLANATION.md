# Giải Thích Chi Tiết Code - Federated Learning Framework

## 📋 Mục Lục
1. [Tổng Quan Hệ Thống](#tổng-quan-hệ-thống)
2. [Cấu Trúc Thư Mục](#cấu-trúc-thư-mục)
3. [Giải Thích Chi Tiết Từng File](#giải-thích-chi-tiết-từng-file)
4. [Luồng Hoạt Động](#luồng-hoạt-động)
5. [Các Thuật Toán Federated Learning](#các-thuật-toán-federated-learning)
6. [Metrics & Logging](#metrics--logging)

---

## 🎯 Tổng Quan Hệ Thống

### Mục Đích
Dự án này implement một framework hoàn chỉnh để so sánh **6 thuật toán Federated Learning** trên **9 phân phối dữ liệu khác nhau** (tổng 54 experiments).

### Công Nghệ Sử Dụng
- **Framework**: Flower v1.24.0+ (Federated Learning)
- **Deep Learning**: PyTorch 2.8.0
- **Dataset**: CIFAR-10 (10 classes, 60,000 images)
- **Hardware Monitoring**: psutil, pynvml (GPU)

### Các Thành Phần Chính
```
Input (Data) → Client Training → Server Aggregation → Evaluation → Logging
                     ↓                    ↓                ↓            ↓
               Local Models      Global Model      Metrics    CSV/Models
```

---

## 📁 Cấu Trúc Thư Mục

```
flowerAI/
├── pytorchexample/              # Core FL implementation
│   ├── task.py                  # Model definition & data loading
│   ├── client_app_experiment.py # Client-side training logic
│   ├── server_app_experiment.py # Server-side aggregation logic
│   ├── strategies.py            # 6 FL strategy implementations
│   ├── metrics.py               # Metrics calculation
│   └── logger.py                # Experiment logging system
├── configs/                     # 54 experiment configs (.toml)
├── results/                     # Experiment results organized by strategy
│   ├── FedAvg/
│   ├── FedAvgM/
│   └── ...
├── models/                      # Saved models organized by strategy
│   ├── FedAvg/
│   └── ...
├── data/                        # CIFAR-10 data partitions (.npy files)
├── generate_configs.py          # Generate all 54 configs
├── run_all_experiments.py       # Run all experiments sequentially
├── test_strategies.py           # Test individual strategies
├── check_progress.py            # Monitor experiment progress
└── verify_results.py            # Verify experiment results
```

---

## 📝 Giải Thích Chi Tiết Từng File

### 1. `pytorchexample/task.py` - Model & Data Loading

#### **Chức năng chính:**
- Define CNN model cho CIFAR-10
- Load và partition dữ liệu cho các clients

#### **Class Net - CNN Model:**
```python
class Net(nn.Module):
    def __init__(self):
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # Input: 3 channels (RGB)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # Output: 64 feature maps
        self.pool = nn.MaxPool2d(2, 2)               # Reduce spatial dimensions

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 128)        # Flatten to dense
        self.fc2 = nn.Linear(128, 10)                # 10 classes output
```

**Kiến trúc:**
```
Input (32x32x3)
  → Conv1(32 filters) → ReLU → Pool(16x16x32)
  → Conv2(64 filters) → ReLU → Pool(8x8x64)
  → Flatten(4096) → FC1(128) → ReLU → Dropout(0.5)
  → FC2(10) → Output
```

#### **Data Partitioning:**

**1. IID Distribution (homo):**
```python
# Mỗi client có phân phối dữ liệu giống nhau
# Ví dụ: 6 clients, mỗi client có ~8,333 samples
# Tất cả có đủ 10 classes
```

**2. Non-IID by Class (C2, C3, C4, C5):**
```python
# C2: Mỗi client chỉ có 2 classes
# C3: Mỗi client chỉ có 3 classes
# C4: Mỗi client chỉ có 4 classes
# C5: Mỗi client chỉ có 5 classes
# → Heterogeneous data distribution
```

**3. Dirichlet Distribution (Dir0.1, Dir0.5, Dir1.0, Dir10.0):**
```python
# Sử dụng Dirichlet distribution với parameter α
# α = 0.1: Rất non-IID (mỗi client thiên về ít classes)
# α = 10.0: Gần như IID (phân phối đồng đều)
# → Realistic data distribution
```

---

### 2. `pytorchexample/client_app_experiment.py` - Client Training

#### **Chức năng:**
- Training model trên local data của client
- Evaluate model trên local test data
- Gửi updates (gradients/weights) về server

#### **Training Function:**
```python
def train(net, trainloader, epochs, learning_rate, device):
    """
    Local training tại client.

    Args:
        net: Neural network model
        trainloader: DataLoader với data của client này
        epochs: Số epochs train (thường = 1 trong FL)
        learning_rate: Learning rate (0.01)
        device: CPU hoặc GPU

    Returns:
        metrics: {loss, accuracy, precision, recall, f1, num_examples}
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)

    net.train()
    for epoch in range(epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

    # Calculate comprehensive metrics
    return calculate_metrics(net, trainloader, device)
```

**Luồng xử lý:**
```
1. Nhận global model từ server
2. Load local data partition
3. Train local model (1 epoch)
4. Calculate local metrics (loss, acc, precision, recall, f1)
5. Gửi updated weights về server
```

---

### 3. `pytorchexample/server_app_experiment.py` - Server Aggregation

#### **Chức năng chính:**
- Khởi tạo global model
- Chọn strategy (FedAvg, FedAvgM, etc.)
- Aggregate client updates
- Evaluate global model
- Save models và logging

#### **Main Function Flow:**
```python
@app.main()
def main(grid: Grid, context: Context) -> None:
    # 1. Đọc config từ .toml file
    num_rounds = context.run_config.get("num-server-rounds", 500)
    strategy_name = context.run_config.get("strategy", "FedAvg")
    batch_size = context.run_config.get("batch-size", 64)

    # 2. Khởi tạo logger
    experiment_logger = ExperimentLogger(experiment_name)

    # 3. Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # 4. Chọn strategy và parameters
    strategy_params = {}
    if strategy_name == "FedAvgM":
        strategy_params["server_momentum"] = 0.9
        strategy_params["server_learning_rate"] = 0.5
    # ... các strategies khác

    # 5. Tạo strategy instance
    strategy = get_strategy(
        strategy_name=strategy_name,
        evaluate_metrics_aggregation_fn=eval_metrics_agg_fn,
        **strategy_params
    )

    # 6. Chạy FL training
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate
    )

    # 7. Save models
    save_final_model(result, experiment_name)
    save_best_model(result, experiment_logger.best_round)
```

#### **Global Evaluation:**
```python
def global_evaluate(server_round, arrays):
    """
    Đánh giá global model trên centralized test set.

    Flow:
    1. Load model với weights mới nhất
    2. Evaluate trên toàn bộ test set
    3. Calculate metrics (loss, acc, precision, recall, f1)
    4. Log metrics (global + hardware)
    5. Track best model
    """
    # Load model
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())

    # Evaluate
    metrics = calculate_metrics(model, test_dataloader, device)

    # Log metrics + hardware
    experiment_logger.log_global_metrics(server_round, metrics)
    experiment_logger.log_hardware_metrics(server_round)

    return MetricRecord(metrics)
```

---

### 4. `pytorchexample/strategies.py` - FL Strategies

#### **Strategy Pattern:**
Tất cả 6 strategies đều extend từ base strategy và override `aggregate_evaluate()`:

```python
class FedAvgWithMetricsAggregation(FedAvg):
    """
    FedAvg với metrics aggregation callback.

    Aggregation formula:
    w_global = Σ(n_i / N) * w_i

    Trong đó:
    - w_i: weights từ client i
    - n_i: số samples của client i
    - N: tổng số samples
    """
    def aggregate_evaluate(self, server_round, results, failures):
        # 1. Call parent aggregation
        aggregated_result = super().aggregate_evaluate(...)

        # 2. Extract client metrics
        for reply in results:
            metrics = reply.content.metrics_records['metrics']
            client_metrics.append(metrics)

        # 3. Call custom metrics aggregation callback
        if self.evaluate_metrics_aggregation_fn:
            self.evaluate_metrics_aggregation_fn(results)

        return aggregated_result
```

#### **6 Strategies Implemented:**

**1. FedAvg (Federated Averaging):**
```python
# Công thức: w_t+1 = Σ(n_k/N) * w_k^t
# - Simple weighted average
# - No momentum, no adaptive learning rate
```

**2. FedAvgM (FedAvg with Momentum):**
```python
# Công thức:
# m_t+1 = β*m_t + Δw_t
# w_t+1 = w_t - η*m_t+1
#
# Parameters:
# - server_momentum (β): 0.9
# - server_learning_rate (η): 0.5
```

**3. FedProx (Federated Proximal):**
```python
# Thêm proximal term vào local objective:
# L_i = F_i(w) + (μ/2)||w - w_global||²
#
# Parameters:
# - proximal_mu (μ): 0.01
# - Giúp clients không đi quá xa global model
```

**4. FedAdam (Federated Adam):**
```python
# Adaptive learning rate với momentum:
# m_t = β1*m_t-1 + (1-β1)*Δw_t
# v_t = β2*v_t-1 + (1-β2)*Δw_t²
# w_t+1 = w_t - η * m_t / (√v_t + τ)
#
# Parameters:
# - eta (η): 0.01 (server learning rate)
# - beta_1 (β1): 0.9 (first moment)
# - beta_2 (β2): 0.99 (second moment)
# - tau (τ): 1e-9 (numerical stability)
```

**5. FedAdagrad (Federated Adagrad):**
```python
# Adaptive learning rate without momentum:
# v_t = v_t-1 + Δw_t²
# w_t+1 = w_t - η * Δw_t / (√v_t + τ)
#
# Parameters:
# - eta (η): 0.01
# - tau (τ): 1e-9
```

**6. FedYogi (Federated Yogi):**
```python
# Similar to FedAdam but with different v update:
# v_t = v_t-1 - (1-β2) * Δw_t² * sign(v_t-1 - Δw_t²)
#
# Parameters: Same as FedAdam
# - Better convergence for non-IID data
```

---

### 5. `pytorchexample/logger.py` - Experiment Logging

#### **ExperimentLogger Class:**

**Chức năng:**
- Log global metrics (loss, accuracy, precision, recall, f1)
- Log client metrics cho từng client
- Track hardware metrics (CPU, RAM, GPU)
- Track best model
- Organize results by strategy

**Cấu trúc files:**
```
results/
  FedAvg/
    FedAvg_homo_npy_global_20260109_231146.csv
    FedAvg_homo_npy_client_20260109_231146.csv
    FedAvg_homo_npy_hardware_20260109_231146.csv
    FedAvg_homo_npy_config_20260109_231146.txt
```

#### **Hardware Metrics Collection:**
```python
def _get_hardware_metrics(self):
    """
    Thu thập hardware metrics.

    Metrics:
    - cpu_percent: CPU usage của process (%)
    - ram_gb: RAM usage của process (GB)
    - gpu_temp: GPU temperature (°C) - nếu có NVIDIA GPU
    - gpu_util: GPU utilization (%) - nếu có NVIDIA GPU
    - vram_gb: VRAM usage (GB) - nếu có NVIDIA GPU
    """
    # CPU & RAM (process-specific)
    ram_gb = self.process.memory_info().rss / (1024**3)
    cpu_percent = self.process.cpu_percent()

    # GPU metrics (system-wide, NVIDIA only)
    if self.has_gpu_mon:
        handle = self.pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        gpu_temp = pynvml.nvmlDeviceGetTemperature(handle)
        vram_gb = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024**3)

    return {cpu_percent, ram_gb, gpu_temp, gpu_util, vram_gb}
```

#### **Best Model Tracking:**
```python
def log_global_metrics(self, round_num, metrics):
    """
    Log metrics và track best model.
    """
    # Log to CSV
    self._write_to_global_csv(round_num, metrics)

    # Track best model based on global_accuracy
    current_acc = metrics['global_accuracy']
    if current_acc > self.best_accuracy:
        self.best_accuracy = current_acc
        self.best_round = round_num
```

---

### 6. `pytorchexample/metrics.py` - Metrics Calculation

#### **Calculate Comprehensive Metrics:**
```python
def calculate_metrics(model, dataloader, device):
    """
    Tính toán metrics chi tiết.

    Returns:
        {
            'loss': float,           # Cross-entropy loss
            'accuracy': float,       # Overall accuracy
            'precision': float,      # Macro-averaged precision
            'recall': float,         # Macro-averaged recall
            'f1': float,             # Macro-averaged F1-score
            'num_examples': int      # Total number of examples
        }
    """
    criterion = nn.CrossEntropyLoss()
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images.to(device))
            loss = criterion(outputs, labels.to(device))
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics using sklearn
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )

    return {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'num_examples': len(all_labels)
    }
```

---

### 7. `generate_configs.py` - Config Generation

#### **Chức năng:**
Tạo 54 config files (6 strategies × 9 distributions)

```python
STRATEGIES = {
    "FedAvg": {},
    "FedAvgM": {"server-momentum": 0.9, "server-learning-rate": 0.5},
    "FedProx": {"proximal-mu": 0.01},
    "FedAdam": {"eta": 0.01, "beta-1": 0.9, "beta-2": 0.99, "tau": 1e-9},
    "FedAdagrad": {"eta": 0.01, "tau": 1e-9},
    "FedYogi": {"eta": 0.01, "beta-1": 0.9, "beta-2": 0.99, "tau": 1e-9},
}

DISTRIBUTIONS = ["homo", "C2", "C3", "C4", "C5",
                 "Dir0.1", "Dir0.5", "Dir1.0", "Dir10.0"]

BASE_CONFIG = """
num-server-rounds = 500
batch-size = 64
learning-rate = 0.01
local-epochs = 1
num-clients = 6
...
"""
```

**Config file example:**
```toml
# FedAvg_homo_npy.toml
num-server-rounds = 500
batch-size = 64
learning-rate = 0.01
strategy = "FedAvg"
distribution = "homo"
experiment-name = "FedAvg_homo_npy"
```

---

### 8. `run_all_experiments.py` - Experiment Runner

#### **Chức năng:**
Chạy tuần tự 54 experiments

```python
def main():
    total_experiments = len(STRATEGIES) * len(DISTRIBUTIONS)  # 54

    for i, strategy in enumerate(STRATEGIES):
        for j, distribution in enumerate(DISTRIBUTIONS):
            config = f"configs/{strategy}_{distribution}_npy.toml"

            print(f"\n[{current}/{total_experiments}] "
                  f"Running: {strategy}_{distribution}")

            # Run experiment
            subprocess.run([
                "flower-simulation",
                "--app", ".",
                "--num-supernodes", "6",
                "--run-config", config
            ], check=True)
```

**Thời gian ước tính:**
```
Mỗi experiment (500 rounds): ~30-40 phút
54 experiments: ~27-36 giờ
```

---

### 9. `verify_results.py` - Results Verification

#### **Chức năng:**
Verify tất cả experiment results và show best metrics

```python
def verify_csv_file(csv_path, expected_type):
    """
    Verify CSV structure và data.

    For client CSV:
    - Check ~3000 rows (6 clients × 500 rounds)
    - Check columns: round, client_id, phase, loss, accuracy, ...
    - Check non-zero metrics

    For global CSV:
    - Check ~501 rows (rounds 0-500)
    - Check global_accuracy và weighted_accuracy non-zero
    """
    df = pd.read_csv(csv_path)

    if expected_type == "global":
        # Find best round
        best_idx = df[df['round'] > 0]['global_accuracy'].idxmax()
        best_round = df.loc[best_idx]

        print(f"Best (R{best_round['round']}): "
              f"G_Acc={best_round['global_accuracy']:.4f}")
```

---

## 🔄 Luồng Hoạt Động

### Một FL Round Complete Flow:

```
Round t:

1. SERVER: Select clients (all 6 clients)
   └─> Send global model weights w_global^t

2. CLIENTS (parallel):
   Client 1: Load local data → Train 1 epoch → Calculate metrics
   Client 2: Load local data → Train 1 epoch → Calculate metrics
   ...
   Client 6: Load local data → Train 1 epoch → Calculate metrics
   └─> Send updated weights {w_1^t, w_2^t, ..., w_6^t} + metrics

3. SERVER: Aggregate updates
   Strategy-specific aggregation:
   - FedAvg: w_global^(t+1) = Σ(n_i/N) * w_i^t
   - FedAvgM: Use momentum
   - FedProx: Apply proximal term
   - FedAdam/Adagrad/Yogi: Adaptive learning rate

4. SERVER: Global evaluation
   └─> Evaluate w_global^(t+1) on centralized test set
   └─> Calculate metrics (loss, acc, precision, recall, f1)
   └─> Log to CSV + Track hardware + Check if best model

5. SERVER: Log metrics
   ├─> Global metrics CSV
   ├─> Client metrics CSV (for all 6 clients)
   └─> Hardware metrics CSV

Repeat for 500 rounds
```

### Complete Experiment Flow:

```
1. Generate configs (54 files)
   └─> generate_configs.py

2. Run experiments
   └─> run_all_experiments.py
       ├─> Experiment 1/54: FedAvg_homo
       │   └─> 500 rounds × 6 clients
       ├─> Experiment 2/54: FedAvg_C2
       │   └─> 500 rounds × 6 clients
       ...
       └─> Experiment 54/54: FedYogi_Dir10.0
           └─> 500 rounds × 6 clients

3. Save results
   ├─> results/{strategy}/{experiment}_global_*.csv
   ├─> results/{strategy}/{experiment}_client_*.csv
   ├─> results/{strategy}/{experiment}_hardware_*.csv
   ├─> models/{strategy}/{experiment}_final_model.pt
   └─> models/{strategy}/{experiment}_best_round{N}_model.pt

4. Verify results
   └─> verify_results.py
       └─> Check all CSV files + Show best metrics
```

---

## 📊 Metrics Explained

### Global Metrics (Centralized Test Set):
```
loss: Cross-entropy loss trên toàn bộ test set
accuracy: Tỉ lệ predictions đúng
precision: Macro-averaged precision (avg across 10 classes)
recall: Macro-averaged recall
f1: Macro-averaged F1-score
```

### Client Aggregation Metrics:
```
global_accuracy:
  = (acc_1 + acc_2 + ... + acc_6) / 6
  = Unweighted average của client accuracies

weighted_accuracy:
  = (n_1*acc_1 + n_2*acc_2 + ... + n_6*acc_6) / N
  = Weighted average dựa trên số samples
  Trong đó n_i là số samples của client i, N là tổng samples
```

### Hardware Metrics:
```
cpu_percent: CPU usage của process (%)
ram_gb: RAM usage của process (GB)
gpu_temp: GPU temperature nếu có (°C)
gpu_util: GPU utilization nếu có (%)
vram_gb: VRAM usage nếu có (GB)
```

---

## 🎯 Key Design Decisions

### 1. **Strategy Pattern cho FL Algorithms**
```
Tại sao: Dễ dàng thêm/sửa strategies mà không ảnh hưởng code khác
Cách: Extend base Flower strategies + override aggregate_evaluate()
```

### 2. **Custom Metrics Aggregation Callback**
```
Vấn đề: Flower v1.24.0+ không support evaluate_metrics_aggregation_fn
Giải pháp: Override aggregate_evaluate() trong custom strategy wrappers
```

### 3. **Organized Results by Strategy**
```
results/
  FedAvg/     # Tất cả results của FedAvg ở đây
  FedAvgM/    # Tất cả results của FedAvgM ở đây
  ...

models/
  FedAvg/     # Tất cả models của FedAvg ở đây
  ...

Lợi ích: Dễ quản lý và so sánh results
```

### 4. **Best Model Tracking**
```
Track best model dựa trên global_accuracy
Save cả final model VÀ best model
Lưu best_round để biết model tốt nhất ở round nào
```

### 5. **Hardware Metrics Logging**
```
Log hardware metrics ở mỗi round để:
- Analyze resource usage
- Identify bottlenecks
- Compare efficiency của các strategies
```

---

## 🚀 Cách Sử Dụng

### 1. Setup & Configuration:
```bash
# Generate configs cho 54 experiments
python generate_configs.py

# Config sẽ có:
# - num-server-rounds = 500
# - batch-size = 64
# - learning-rate = 0.01
```

### 2. Run Experiments:
```bash
# Chạy tất cả 54 experiments
python run_all_experiments.py

# Hoặc test từng strategy (với 3 rounds):
python test_strategies.py
```

### 3. Monitor Progress:
```bash
# Theo dõi tiến trình mỗi 10 phút
python check_progress.py
```

### 4. Verify Results:
```bash
# Verify và show best metrics
python verify_results.py
```

---

## 🔍 Debugging & Troubleshooting

### 1. Check Logs:
```bash
# Xem logs của experiment
tail -f experiments.log

# Check hardware metrics
cat results/FedAvg/FedAvg_homo_npy_hardware_*.csv
```

### 2. Verify Metrics:
```bash
# Client metrics should have ~3000 rows (6 clients × 500 rounds)
wc -l results/FedAvg/FedAvg_homo_npy_client_*.csv

# Global metrics should have ~501 rows (rounds 0-500)
wc -l results/FedAvg/FedAvg_homo_npy_global_*.csv
```

### 3. Check Models:
```bash
# List all saved models
ls -lh models/FedAvg/

# Should see:
# *_final_model.pt (~252KB)
# *_best_round{N}_model.pt (~252KB)
```

---

## 📈 Expected Results

### Convergence Patterns:

**IID (homo):**
```
- Fastest convergence
- Highest accuracy (~70-80%)
- All strategies perform similarly
```

**Non-IID (C2, C3, Dir0.1):**
```
- Slower convergence
- Lower accuracy (~50-60%)
- FedProx, FedAdam, FedYogi outperform FedAvg
```

**Moderate Non-IID (C4, C5, Dir1.0):**
```
- Medium convergence
- Medium accuracy (~60-70%)
- All adaptive strategies help
```

---

## 📚 References

### Papers:
- **FedAvg**: [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
- **FedProx**: [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127)
- **FedAdam/Adagrad/Yogi**: [Adaptive Federated Optimization](https://arxiv.org/abs/2003.00295)

### Framework:
- **Flower Documentation**: https://flower.ai/docs/

---

## 📝 Summary

Dự án này implement một **complete FL framework** với:
- ✅ **6 state-of-the-art FL algorithms**
- ✅ **9 data distributions** (IID + non-IID)
- ✅ **Comprehensive metrics** (accuracy, precision, recall, f1)
- ✅ **Hardware monitoring** (CPU, RAM, GPU)
- ✅ **Best model tracking**
- ✅ **Organized logging system**
- ✅ **Automated verification**

**Total**: 54 experiments để so sánh hiệu quả của các FL algorithms trên các data distributions khác nhau.
