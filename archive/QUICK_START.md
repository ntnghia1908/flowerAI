# Quick Start Guide - FL Experiments

Hướng dẫn nhanh để chạy thí nghiệm Federated Learning.

## Bước 1: Cài đặt

```bash
# Cài đặt dependencies
pip install -e .
```

## Bước 2: Test thử (2-3 phút)

```bash
# Test nhanh với 10 rounds
python test_experiment.py
```

Nếu thành công, bạn sẽ thấy:
- ✓ TEST PASSED!
- Files CSV trong thư mục `results/`

## Bước 3: Chạy thí nghiệm thực

### Option A: Chạy thí nghiệm đơn lẻ

```bash
# Backup config gốc
cp pyproject.toml pyproject.toml.original

# Sử dụng config cho experiments
cp pyproject_experiment.toml pyproject.toml

# Chạy thí nghiệm
flwr run . local-simulation --run-config \
  strategy=FedAvg \
  distribution=homo \
  num-server-rounds=500 \
  min-fit-clients=5

# Restore config gốc
cp pyproject.toml.original pyproject.toml
```

### Option B: Chạy batch experiments tự động

```bash
# Test nhỏ (100 rounds, 2 strategies, 2 distributions)
python run_experiments.py --medium

# Thí nghiệm đầy đủ theo bảng
python run_experiments.py \
  --strategies FedAvg FedAvgM FedProx FedAdam FedAdagrad FedYogi \
  --distributions homo Dir(10.0) Dir(1.0) Dir(0.5) Dir(0.1) Dir(0.01) \
  --client-configs C1 C2 C3 C4 C5 \
  --num-rounds 500
```

## Bước 4: Phân tích kết quả

```bash
# Xem summary của tất cả experiments
python analyze_results.py

# Xem kết quả của 1 experiment cụ thể
python analyze_results.py --pattern "FedAvg_homo*"

# So sánh nhiều experiments
python analyze_results.py --pattern "FedAvg*" --compare --metric accuracy

# Chỉ xem summary, không plot
python analyze_results.py --no-plot
```

## Cấu trúc kết quả

```
results/
├── FedAvg_homo_C5_global_20231221_143022.csv      # Metrics global
├── FedAvg_homo_C5_client_20231221_143022.csv      # Metrics từng client
├── FedAvg_homo_C5_weight_20231221_143022.csv      # Weight changes
├── FedAvg_homo_C5_config_20231221_143022.txt      # Config
└── FedAvg_homo_C5_global_20231221_143022.png      # Plot tự động
```

## Ví dụ cụ thể theo bảng của bạn

### 1. Test với homo (IID)

```bash
python run_experiments.py \
  --strategies FedAvg FedAvgM FedProx FedAdam FedAdagrad FedYogi \
  --distributions homo \
  --client-configs C1 C2 C3 C4 C5 \
  --num-rounds 500
```

Kết quả: 6 strategies × 1 distribution × 5 configs = **30 experiments**

### 2. Test với Dir(10.0) - Mild non-IID

```bash
python run_experiments.py \
  --strategies FedAvg FedAvgM FedProx FedAdam FedAdagrad FedYogi \
  --distributions "Dir(10.0)" \
  --client-configs C1 C2 C3 C4 C5 \
  --num-rounds 500
```

### 3. Test với Dir(1.0) - Moderate non-IID

```bash
python run_experiments.py \
  --strategies FedAvg FedAvgM FedProx FedAdam FedAdagrad FedYogi \
  --distributions "Dir(1.0)" \
  --client-configs C1 C2 C3 C4 C5 \
  --num-rounds 500
```

### 4. Test với Dir(0.5) - Strong non-IID

```bash
python run_experiments.py \
  --strategies FedAvg FedAvgM FedProx FedAdam FedAdagrad FedYogi \
  --distributions "Dir(0.5)" \
  --client-configs C1 C2 C3 C4 C5 \
  --num-rounds 500
```

### 5. Test với Dir(0.1) - Very strong non-IID

```bash
python run_experiments.py \
  --strategies FedAvg FedAvgM FedProx FedAdam FedAdagrad FedYogi \
  --distributions "Dir(0.1)" \
  --client-configs C1 C2 C3 C4 C5 \
  --num-rounds 500
```

### 6. Test với Dir(0.01) - Extreme non-IID

```bash
python run_experiments.py \
  --strategies FedAvg FedAvgM FedProx FedAdam FedAdagrad FedYogi \
  --distributions "Dir(0.01)" \
  --client-configs C1 C2 C3 C4 C5 \
  --num-rounds 500
```

## Chạy song song (để nhanh hơn)

Bạn có thể mở nhiều terminal và chạy song song:

**Terminal 1:**
```bash
python run_experiments.py --strategies FedAvg FedAvgM --distributions homo --num-rounds 500
```

**Terminal 2:**
```bash
python run_experiments.py --strategies FedProx FedAdam --distributions homo --num-rounds 500
```

**Terminal 3:**
```bash
python run_experiments.py --strategies FedAdagrad FedYogi --distributions homo --num-rounds 500
```

## Ước tính thời gian

- 1 experiment (500 rounds, 10 clients): ~30-60 phút
- 30 experiments (1 distribution, all strategies, all configs): ~15-30 giờ
- 180 experiments (toàn bộ): ~90-180 giờ (3-7 ngày)

**Khuyến nghị:**
1. Bắt đầu với `--medium` (100 rounds) để test
2. Chạy lần lượt từng distribution
3. Sử dụng `--gpu` nếu có GPU
4. Chạy overnight hoặc khi không dùng máy

## Phân tích sau khi chạy xong

```python
import pandas as pd
import glob

# Load tất cả global metrics
all_files = glob.glob("results/*_global_*.csv")
results = {}

for file in all_files:
    exp_name = file.split('/')[-1].split('_global')[0]
    df = pd.read_csv(file)
    final_acc = df.iloc[-1]['accuracy']
    final_f1 = df.iloc[-1]['f1']
    results[exp_name] = {'accuracy': final_acc, 'f1': final_f1}

# Tạo bảng kết quả
results_df = pd.DataFrame(results).T
print(results_df.sort_values('accuracy', ascending=False))
```

## Troubleshooting

**Lỗi: "No module named 'sklearn'"**
```bash
pip install scikit-learn
```

**Lỗi: "flwr: command not found"**
```bash
pip install flwr[simulation]>=1.24.0
```

**Experiments chạy quá chậm**
- Giảm rounds: `--num-rounds 100`
- Sử dụng GPU: `--gpu`
- Test với ít configs hơn: `--client-configs C5`

**Out of memory**
- Giảm batch size trong pyproject_experiment.toml
- Giảm số clients: `--total-clients 5`

## Đọc thêm

- [EXPERIMENTS_README.md](EXPERIMENTS_README.md) - Tài liệu chi tiết đầy đủ
- [Flower Docs](https://flower.ai/docs/) - Documentation chính thức
