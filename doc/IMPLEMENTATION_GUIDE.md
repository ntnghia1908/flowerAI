# Hướng Dẫn Implement Thuật Toán Federated Learning Mới

## Tổng Quan

Tài liệu này hướng dẫn cách implement các thuật toán Federated Learning mới vào hệ thống một cách đúng đắn, tránh các lỗi về logging metrics và aggregation.

## Vấn Đề Đã Gặp Phải

Khi implement các strategy FL từ Flower framework, chúng ta gặp phải vấn đề:

1. **Client metrics không được log**: File CSV của client rỗng hoặc toàn giá trị 0
2. **Global accuracy và weighted accuracy luôn = 0**: Không tính được metrics aggregate từ clients
3. **Callback không được gọi**: Strategy wrapper pattern không hoạt động với Flower's internal workflow

### Nguyên Nhân

Flower v1.24.0+ không hỗ trợ truyền `evaluate_metrics_aggregation_fn` trực tiếp vào constructor của các strategy. Phương thức `aggregate_evaluate()` được gọi internally và cần được override để inject custom logic.

## Giải Pháp: Custom Strategy Wrappers

### Bước 1: Tạo Custom Strategy Wrapper

Mỗi strategy cần một wrapper class kế thừa từ strategy gốc và override `aggregate_evaluate()`.

**File**: `pytorchexample/custom_strategies.py`

```python
"""Custom federated learning strategies with metrics aggregation support."""

from typing import Callable, Optional
from flwr.serverapp.strategy import (
    FedAvg, FedAvgM, FedProx, FedAdam, FedAdagrad, FedYogi,
)


class FedAvgWithMetricsAggregation(FedAvg):
    """FedAvg with support for evaluate_metrics_aggregation_fn callback."""

    def __init__(
        self,
        evaluate_metrics_aggregation_fn: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

    def aggregate_evaluate(self, server_round: int, replies):
        """Aggregate evaluation metrics and call the aggregation callback."""
        # Call the callback if provided
        if self.evaluate_metrics_aggregation_fn is not None:
            self.evaluate_metrics_aggregation_fn(replies)

        # Call parent's aggregate_evaluate
        return super().aggregate_evaluate(server_round, replies)
```

**Lưu ý quan trọng**:
- Method signature: `aggregate_evaluate(self, server_round: int, replies)` (KHÔNG có `failures` parameter)
- Callback phải được gọi TRƯỚC `super().aggregate_evaluate()`
- Kiểm tra callback không None trước khi gọi

### Bước 2: Update Strategy Factory

Modify `pytorchexample/strategies.py` để sử dụng custom wrappers:

```python
from pytorchexample.custom_strategies import (
    FedAvgWithMetricsAggregation,
    FedAvgMWithMetricsAggregation,
    # ... other wrappers
)

def get_strategy(
    strategy_name: str,
    fraction_train: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_train_nodes: int = 2,
    min_evaluate_nodes: int = 2,
    min_available_nodes: int = 2,
    **kwargs
):
    # Extract callback từ kwargs
    evaluate_metrics_agg_fn = kwargs.pop("evaluate_metrics_aggregation_fn", None)

    common_params = {
        "fraction_train": fraction_train,
        "fraction_evaluate": fraction_evaluate,
        "min_train_nodes": min_train_nodes,
        "min_evaluate_nodes": min_evaluate_nodes,
        "min_available_nodes": min_available_nodes,
    }

    if strategy_name == "YourNewStrategy":
        # Strategy-specific parameters
        param1 = kwargs.get("param1", default_value)

        return YourNewStrategyWithMetricsAggregation(
            evaluate_metrics_aggregation_fn=evaluate_metrics_agg_fn,
            **common_params,
            param1=param1,
            # ... other params
        )
```

### Bước 3: Tạo Callback Function Đúng Cách

**File**: `pytorchexample/server_app_experiment.py`

```python
def create_evaluate_metrics_aggregation_fn(logger):
    """Create callback function for aggregating evaluation metrics."""

    def evaluate_metrics_aggregation_fn(replies):
        """Aggregate evaluation metrics from clients and log them."""
        global client_aggregate_metrics, current_round

        client_accuracies = []
        client_num_examples = []

        # Convert to list để iterate
        replies_list = list(replies)

        # Replies là iterable của Message objects
        for idx, reply in enumerate(replies_list):
            # Extract metrics từ reply.content (RecordDict)
            if hasattr(reply, 'content') and reply.content is not None:
                record_dict = reply.content

                # QUAN TRỌNG: Metrics nằm trong record_dict.metrics_records['metrics']
                if hasattr(record_dict, 'metrics_records') and 'metrics' in record_dict.metrics_records:
                    metrics_dict = record_dict.metrics_records['metrics']

                    # num_examples có key là 'num-examples' (có dấu gạch ngang)
                    num_examples = metrics_dict.get('num-examples', 0)

                    if logger is not None:
                        # Log individual client metrics
                        client_metrics = {
                            'loss': metrics_dict.get('eval_loss', 0.0),
                            'accuracy': metrics_dict.get('eval_acc', 0.0),
                            'precision': metrics_dict.get('eval_precision', 0.0),
                            'recall': metrics_dict.get('eval_recall', 0.0),
                            'f1': metrics_dict.get('eval_f1', 0.0),
                            'num_examples': num_examples
                        }
                        logger.log_client_metrics(current_round, idx, 'evaluate', client_metrics)

                        # Collect for aggregate calculation
                        client_accuracies.append(metrics_dict.get('eval_acc', 0.0))
                        client_num_examples.append(num_examples)

        # Calculate aggregate metrics
        N = len(client_accuracies)
        if N > 0:
            global_accuracy = sum(client_accuracies) / N
            total_examples = sum(client_num_examples)
            weighted_accuracy = sum(acc * n for acc, n in zip(client_accuracies, client_num_examples)) / total_examples if total_examples > 0 else 0.0
        else:
            global_accuracy = 0.0
            weighted_accuracy = 0.0

        # Store for global_evaluate to use
        client_aggregate_metrics = {
            'global_accuracy': global_accuracy,
            'weighted_accuracy': weighted_accuracy
        }

        return {}

    return evaluate_metrics_aggregation_fn
```

### Bước 4: Data Structure Reference

**Cấu trúc của `replies` parameter**:

```
replies: Iterable[Message]
  └─> Message
      ├─> content: RecordDict
      │   └─> metrics_records: dict
      │       └─> 'metrics': dict
      │           ├─> 'eval_loss': float
      │           ├─> 'eval_acc': float
      │           ├─> 'eval_precision': float
      │           ├─> 'eval_recall': float
      │           ├─> 'eval_f1': float
      │           └─> 'num-examples': int  # CHÚ Ý: có dấu gạch ngang
      └─> metadata: Metadata (có thể không chứa num_examples)
```

**Lỗi thường gặp**:
- ❌ Tìm `metrics_dict` attribute (không tồn tại)
- ❌ Lấy `num_examples` từ `reply.metadata` (thường = 0)
- ❌ Dùng key `'num_examples'` thay vì `'num-examples'`
- ❌ Không convert replies thành list trước khi iterate

## Template Implement Strategy Mới

### 1. Thêm vào `custom_strategies.py`:

```python
class YourNewStrategyWithMetricsAggregation(YourNewStrategy):
    """YourNewStrategy with support for evaluate_metrics_aggregation_fn callback."""

    def __init__(
        self,
        evaluate_metrics_aggregation_fn: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

    def aggregate_evaluate(self, server_round: int, replies):
        if self.evaluate_metrics_aggregation_fn is not None:
            self.evaluate_metrics_aggregation_fn(replies)
        return super().aggregate_evaluate(server_round, replies)
```

### 2. Thêm vào `strategies.py`:

```python
# Import
from pytorchexample.custom_strategies import (
    # ... existing imports
    YourNewStrategyWithMetricsAggregation,
)

# Trong get_strategy():
elif strategy_name == "YourNewStrategy":
    # Extract strategy-specific parameters
    param1 = kwargs.get("param1", default_value)
    param2 = kwargs.get("param2", default_value)

    return YourNewStrategyWithMetricsAggregation(
        evaluate_metrics_aggregation_fn=evaluate_metrics_agg_fn,
        **common_params,
        param1=param1,
        param2=param2,
    )
```

### 3. Tạo config file mới:

**File**: `configs/YourNewStrategy_homo_npy.toml`

```toml
# YourNewStrategy on homo distribution
num-server-rounds = 3
fraction-train = 1.0
fraction-evaluate = 1.0
min-train-nodes = 6
min-evaluate-nodes = 6
local-epochs = 1
learning-rate = 0.01
batch-size = 32
strategy = "YourNewStrategy"
distribution = "homo"
experiment-name = "YourNewStrategy_homo_npy"
num-clients = 6
data-source = "npy"

# Strategy-specific parameters
param1 = value1
param2 = value2
```

### 4. Update `generate_configs.py`:

```python
STRATEGIES = {
    "FedAvg": {},
    "FedAvgM": {"server-momentum": 0.9},
    "FedProx": {"proximal-mu": 0.01},
    # ... existing strategies
    "YourNewStrategy": {"param1": value1, "param2": value2},
}
```

## Testing Checklist

Sau khi implement strategy mới, kiểm tra:

- [ ] **Strategy wrapper được tạo đúng**: Class kế thừa từ base strategy và override `aggregate_evaluate()`
- [ ] **Import đúng**: Custom strategy được import vào `strategies.py`
- [ ] **Factory function**: Strategy được thêm vào `get_strategy()` với đầy đủ parameters
- [ ] **Config files**: Tạo config files cho tất cả distributions (9 files)
- [ ] **Run test**: `flwr run . --run-config configs/YourNewStrategy_homo_npy.toml`
- [ ] **Check logs**: Xem console output có global_accuracy và weighted_accuracy khác 0
- [ ] **Verify CSV files**:
  - `results/YourNewStrategy_homo_npy_client_*.csv` có data rows với metrics thực
  - `results/YourNewStrategy_homo_npy_global_*.csv` có global_accuracy và weighted_accuracy khác 0

## Common Issues và Solutions

### Issue 1: Client CSV rỗng hoặc toàn 0

**Nguyên nhân**: Callback không được gọi hoặc metrics extraction sai

**Solutions**:
1. Kiểm tra wrapper class có override `aggregate_evaluate()` đúng
2. Kiểm tra callback được pass vào strategy constructor
3. Thêm debug print trong callback để verify nó được gọi

### Issue 2: TypeError về method signature

**Error**: `TypeError: aggregate_evaluate() missing 1 required positional argument: 'failures'`

**Solution**: Method signature phải là `aggregate_evaluate(self, server_round: int, replies)` (KHÔNG có `failures`)

### Issue 3: Global accuracy = 0

**Nguyên nhân**: Metrics extraction từ RecordDict sai

**Solution**:
```python
# ĐÚNG
metrics_dict = record_dict.metrics_records['metrics']
num_examples = metrics_dict.get('num-examples', 0)  # hyphenated key

# SAI
metrics_dict = record_dict.metrics_dict  # không tồn tại
num_examples = reply.metadata.num_examples  # thường = 0
```

### Issue 4: Strategy không nhận parameters

**Nguyên nhân**: Parameters không được extract từ config hoặc không được pass vào strategy

**Solution**: Kiểm tra parameter mapping trong `strategies.py`:
```python
# Extract from kwargs
param_value = kwargs.get("param-name", default)  # key dùng hyphen trong config

# Pass to strategy
return StrategyWithMetrics(
    **common_params,
    param_name=param_value,  # parameter name dùng underscore
)
```

## Best Practices

1. **Naming Convention**:
   - Config files: `{Strategy}_{Distribution}_npy.toml`
   - Wrapper class: `{Strategy}WithMetricsAggregation`
   - Config parameter keys: dùng hyphens (`proximal-mu`)
   - Python parameter names: dùng underscores (`proximal_mu`)

2. **Strategy Parameters**:
   - Luôn cung cấp default values
   - Document rõ ràng ý nghĩa của từng parameter
   - Test với nhiều giá trị khác nhau

3. **Error Handling**:
   - Kiểm tra callback không None trước khi gọi
   - Kiểm tra dictionary keys tồn tại trước khi access
   - Use `.get()` method với default values

4. **Testing**:
   - Test với 1 distribution trước (thường là `homo`)
   - Verify metrics trong console output
   - Check CSV files có data đúng
   - Mới generate full 54 config files và run all experiments

## Examples

### Ví dụ 1: Implement FedOpt (FedAdagrad/FedAdam/FedYogi)

Các strategy này cần parameters: `eta`, `eta_l`, `tau`, `beta_1`, `beta_2`

**custom_strategies.py**:
```python
class FedAdamWithMetricsAggregation(FedAdam):
    def __init__(self, evaluate_metrics_aggregation_fn=None, **kwargs):
        super().__init__(**kwargs)
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

    def aggregate_evaluate(self, server_round: int, replies):
        if self.evaluate_metrics_aggregation_fn is not None:
            self.evaluate_metrics_aggregation_fn(replies)
        return super().aggregate_evaluate(server_round, replies)
```

**strategies.py**:
```python
elif strategy_name == "FedAdam":
    eta = kwargs.get("eta", 1e-2)
    eta_l = kwargs.get("eta_l", 1e-1)
    beta_1 = kwargs.get("beta_1", 0.9)
    beta_2 = kwargs.get("beta_2", 0.99)
    tau = kwargs.get("tau", 1e-9)

    return FedAdamWithMetricsAggregation(
        evaluate_metrics_aggregation_fn=evaluate_metrics_agg_fn,
        **common_params,
        eta=eta,
        eta_l=eta_l,
        beta_1=beta_1,
        beta_2=beta_2,
        tau=tau
    )
```

**Config (FedAdam_homo_npy.toml)**:
```toml
strategy = "FedAdam"
eta = 0.01
eta-l = 0.1
beta-1 = 0.9
beta-2 = 0.99
tau = 1e-9
```

### Ví dụ 2: Implement FedProx

Strategy này cần parameter: `proximal_mu` và modify training loop

**custom_strategies.py**:
```python
class FedProxWithMetricsAggregation(FedProx):
    def __init__(self, evaluate_metrics_aggregation_fn=None, **kwargs):
        super().__init__(**kwargs)
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

    def aggregate_evaluate(self, server_round: int, replies):
        if self.evaluate_metrics_aggregation_fn is not None:
            self.evaluate_metrics_aggregation_fn(replies)
        return super().aggregate_evaluate(server_round, replies)
```

**strategies.py**:
```python
elif strategy_name == "FedProx":
    proximal_mu = kwargs.get("proximal_mu", 0.01)

    return FedProxWithMetricsAggregation(
        evaluate_metrics_aggregation_fn=evaluate_metrics_agg_fn,
        **common_params,
        proximal_mu=proximal_mu
    )
```

**task.py** (training loop modification):
```python
def train(net, trainloader, epochs, lr, device, proximal_mu=0.0, global_params=None):
    """Train with optional FedProx proximal term."""
    # ... setup ...

    # Save global parameters for proximal term
    global_params_dict = None
    if proximal_mu > 0.0 and global_params is not None:
        global_params_dict = {k: v.clone().detach().to(device)
                              for k, v in global_params.items()}

    for _ in range(epochs):
        for batch in trainloader:
            # ... forward pass ...
            loss = criterion(outputs, labels)

            # Add proximal term: μ/2 ||w - w_global||²
            if proximal_mu > 0.0 and global_params_dict is not None:
                proximal_term = 0.0
                for name, param in net.named_parameters():
                    if name in global_params_dict:
                        proximal_term += ((param - global_params_dict[name]) ** 2).sum()
                loss += (proximal_mu / 2) * proximal_term

            # ... backward pass ...
```

## Tóm Tắt Workflow

1. **Tạo wrapper class** trong `custom_strategies.py`
2. **Update factory function** trong `strategies.py`
3. **Tạo config files** cho tất cả distributions
4. **Test với 1 config** (`homo` distribution)
5. **Verify outputs**: console logs + CSV files
6. **Run full experiments** với `python run_all_experiments.py`

## References

- [Flower Documentation](https://flower.ai/docs/)
- [Custom Strategies Guide](https://flower.ai/docs/framework/how-to-implement-strategies.html)
- Project files:
  - [pytorchexample/custom_strategies.py](pytorchexample/custom_strategies.py)
  - [pytorchexample/strategies.py](pytorchexample/strategies.py)
  - [pytorchexample/server_app_experiment.py](pytorchexample/server_app_experiment.py)

---

**Lưu ý**: Document này được tạo dựa trên kinh nghiệm debug và fix các issue thực tế khi implement FedAvg, FedAvgM, FedProx, FedAdam, FedAdagrad, và FedYogi trong Flower framework v1.24.0+.
