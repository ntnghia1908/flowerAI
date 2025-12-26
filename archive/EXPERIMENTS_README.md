# Federated Learning Experiments Guide

Hướng dẫn chạy thí nghiệm Federated Learning với nhiều thuật toán và phân phối dữ liệu khác nhau.

## Tổng quan

Framework thí nghiệm này cho phép bạn:

- ✅ Chạy nhiều thuật toán FL: **FedAvg, FedAvgM, FedProx, FedAdam, FedAdagrad, FedYogi**
- ✅ Thử nghiệm với các phân phối dữ liệu: **homo, Dir(10.0), Dir(1.0), Dir(0.5), Dir(0.1), Dir(0.01)**
- ✅ Điều chỉnh số lượng clients tham gia: **1, 2, 3, 4, 5 clients/round**
- ✅ Thu thập metrics đầy đủ: **loss, accuracy, precision, recall, F1**
- ✅ Lưu kết quả ra CSV tự động

## Cài đặt

```bash
# Cài đặt dependencies (bao gồm scikit-learn cho metrics)
pip install -e .
```

## Cấu trúc Files

```
pytorchexample/
├── metrics.py                    # Tính toán metrics (precision, recall, F1)
├── logger.py                     # Logger CSV cho experiments
├── partitioner.py                # Data partitioning (IID, Dirichlet)
├── strategies.py                 # FL strategies (FedAvg, FedProx, etc.)
├── server_app_experiment.py      # Server với comprehensive logging
├── client_app_experiment.py      # Client với comprehensive logging
└── task.py                       # Model và training (đã cập nhật)

run_experiments.py                # Script tự động chạy experiments
```

## Cách sử dụng

### 1. Chạy thí nghiệm đơn lẻ

```bash
# Chạy FedAvg với IID data, 5 clients, 500 rounds
flwr run . local-simulation \
  --run-config \
  strategy=FedAvg \
  distribution=homo \
  num-server-rounds=500 \
  min-fit-clients=5 \
  min-evaluate-clients=5
```

```bash
# Chạy FedProx với Dirichlet(0.5), 3 clients, 500 rounds
flwr run . local-simulation \
  --run-config \
  strategy=FedProx \
  distribution=Dir(0.5) \
  num-server-rounds=500 \
  min-fit-clients=3 \
  min-evaluate-clients=3
```

### 2. Chạy batch experiments tự động

#### Quick test (10 rounds, để debug)
```bash
python run_experiments.py --quick
```

#### Medium test (100 rounds, để kiểm tra)
```bash
python run_experiments.py --medium
```

#### Chạy thí nghiệm cụ thể
```bash
# Chạy FedAvg và FedAvgM với 2 distributions
python run_experiments.py \
  --strategies FedAvg FedAvgM \
  --distributions homo Dir(0.5) \
  --client-configs C3 C5 \
  --num-rounds 500
```

#### Chạy TẤT CẢ thí nghiệm (rất lâu!)
```bash
python run_experiments.py --all --num-rounds 500
```

### 3. Sử dụng GPU

```bash
# Single experiment với GPU
flwr run . local-simulation-gpu --run-config strategy=FedAvg

# Batch experiments với GPU
python run_experiments.py --gpu --medium
```

## Các thuật toán FL hỗ trợ

| Thuật toán | Mô tả | Tham số đặc biệt |
|-----------|-------|------------------|
| **FedAvg** | Federated Averaging (baseline) | - |
| **FedAvgM** | FedAvg + server momentum | server_momentum=0.9 |
| **FedProx** | FedAvg + proximal term | proximal_mu=0.01 |
| **FedAdam** | FedAvg + Adam optimizer | eta, beta_1, beta_2 |
| **FedAdagrad** | FedAvg + Adagrad optimizer | eta, tau |
| **FedYogi** | FedAvg + Yogi optimizer | eta, beta_1, beta_2 |

## Phân phối dữ liệu

| Distribution | Mô tả | Non-IID Level |
|-------------|-------|---------------|
| **homo** | IID - Dữ liệu phân bố đồng nhất | ✅ IID |
| **Dir(10.0)** | Dirichlet α=10.0 - Gần IID | ⚠️ Mild non-IID |
| **Dir(1.0)** | Dirichlet α=1.0 - Non-IID vừa | ⚠️⚠️ Moderate non-IID |
| **Dir(0.5)** | Dirichlet α=0.5 - Non-IID mạnh | ⚠️⚠️⚠️ Strong non-IID |
| **Dir(0.1)** | Dirichlet α=0.1 - Non-IID rất mạnh | ⚠️⚠️⚠️⚠️ Very strong non-IID |
| **Dir(0.01)** | Dirichlet α=0.01 - Extreme non-IID | ⚠️⚠️⚠️⚠️⚠️ Extreme non-IID |

**Dirichlet Distribution**: Mỗi client nhận dữ liệu theo phân phối Dirichlet, dẫn đến label distribution không đồng nhất. α nhỏ hơn = non-IID mạnh hơn.

## Số lượng clients (#C)

| Config | Clients/Round | Fraction |
|--------|---------------|----------|
| **C1** | 1/10 | 10% |
| **C2** | 2/10 | 20% |
| **C3** | 3/10 | 30% |
| **C4** | 4/10 | 40% |
| **C5** | 5/10 | 50% |

## Kết quả Experiments

Kết quả được lưu tự động trong thư mục `results/`:

```
results/
├── FedAvg_homo_C5_global_20231221_143022.csv      # Global metrics
├── FedAvg_homo_C5_client_20231221_143022.csv      # Client metrics
├── FedAvg_homo_C5_weight_20231221_143022.csv      # Weight metrics
├── FedAvg_homo_C5_config_20231221_143022.txt      # Experiment config
└── ...
```

### CSV Formats

**Global Metrics** (`*_global_*.csv`):
```csv
round,loss,accuracy,precision,recall,f1
1,2.3026,0.1000,0.1000,0.1000,0.1000
2,2.1234,0.2500,0.2400,0.2450,0.2425
...
```

**Client Metrics** (`*_client_*.csv`):
```csv
round,client_id,phase,loss,accuracy,precision,recall,f1,num_examples
1,0,train,2.1,0.25,0.24,0.23,0.235,4000
1,0,evaluate,2.2,0.23,0.22,0.21,0.215,1000
...
```

**Weight Metrics** (`*_weight_*.csv`):
```csv
round,weight_norm,weight_change,weight_relative_change
1,125.456,0.0,0.0
2,126.789,5.234,0.0417
...
```

## Ví dụ thí nghiệm như trong bảng

Để tái tạo thí nghiệm trong bảng của bạn (500 rounds + 10 clients):

```bash
# Chạy TẤT CẢ các strategies với TẤT CẢ distributions và TẤT CẢ client configs
python run_experiments.py \
  --strategies FedAvg FedAvgM FedProx FedAdam FedAdagrad FedYogi \
  --distributions homo Dir(10.0) Dir(1.0) Dir(0.5) Dir(0.1) Dir(0.01) \
  --client-configs C1 C2 C3 C4 C5 \
  --num-rounds 500 \
  --total-clients 10
```

**Tổng số experiments**: 6 strategies × 6 distributions × 5 client configs = **180 experiments**

⚠️ **Lưu ý**: Mỗi experiment 500 rounds mất ~30-60 phút, tổng thời gian có thể lên đến vài ngày!

## Phân tích kết quả

Sau khi chạy xong, bạn có thể phân tích kết quả bằng Python:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load global metrics
df = pd.read_csv('results/FedAvg_homo_C5_global_20231221_143022.csv')

# Plot accuracy over rounds
plt.plot(df['round'], df['accuracy'], label='Accuracy')
plt.plot(df['round'], df['f1'], label='F1 Score')
plt.xlabel('Round')
plt.ylabel('Metric')
plt.legend()
plt.title('FedAvg Performance (homo, C=5)')
plt.show()
```

## Tối ưu hóa

### Giảm thời gian experiments

1. **Giảm số rounds**: Test với 100 rounds trước
2. **Giảm số client configs**: Chỉ test C5
3. **Giảm số distributions**: Bắt đầu với homo và Dir(0.5)
4. **Sử dụng GPU**: `--gpu` flag
5. **Chạy parallel**: Chạy nhiều scripts song song (cẩn thận với RAM!)

### Debug mode

```bash
# Test nhanh với 10 rounds
python run_experiments.py \
  --strategies FedAvg \
  --distributions homo \
  --client-configs C5 \
  --num-rounds 10
```

## Troubleshooting

### Lỗi "Out of Memory"
- Giảm batch size: `--batch-size 16`
- Giảm số clients: `--total-clients 5`
- Tắt GPU simulation nếu không cần

### Experiments chạy quá chậm
- Sử dụng `--quick` hoặc `--medium` mode
- Giảm số rounds
- Sử dụng GPU nếu có

### Kết quả không như mong đợi
- Kiểm tra distribution có đúng không
- Kiểm tra learning rate (có thể cần điều chỉnh)
- Chạy lại với seed cố định

## Tham khảo

- [Flower Documentation](https://flower.ai/docs/)
- [FedAvg Paper](https://arxiv.org/abs/1602.05629)
- [FedProx Paper](https://arxiv.org/abs/1812.06127)
- [Adaptive Federated Optimization](https://arxiv.org/abs/2003.00295)
