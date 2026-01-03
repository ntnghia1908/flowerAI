# Using Pre-partitioned .npy Data for Faster Training

## Overview

This guide explains how to use pre-partitioned CIFAR-10 data stored as `.npy` files instead of partitioning on-the-fly. This approach significantly speeds up data loading during federated learning experiments.

## Benefits

- **Faster loading**: Data is pre-partitioned and stored in numpy format
- **No HuggingFace timeout**: No network calls during training
- **Reproducible partitions**: Same partition every time
- **Easy to share**: Can copy partition folders between machines

## Step 1: Export Partitions

### Export a single distribution:

```bash
# Export homo (IID) distribution with 6 clients
python export_partitions.py --distribution homo --num-clients 6 --output-dir ./data

# Export C2 (label skew, 2 classes per client)
python export_partitions.py --distribution C2 --num-clients 6 --output-dir ./data

# Export Dirichlet distribution (alpha=0.5)
python export_partitions.py --distribution Dir0.5 --num-clients 6 --output-dir ./data
```

### Export all distributions at once:

```bash
python export_all_partitions.py
```

This will export all 9 test configurations:
- `homo` (IID)
- `C2`, `C3`, `C4`, `C5` (Label Skew)
- `Dir0.1`, `Dir0.5`, `Dir1.0`, `Dir10.0` (Dirichlet)

## Step 2: Folder Structure

After export, you'll have:

```
data/
├── cifar10_homo_6partition/
│   ├── partition_0/
│   │   ├── train/
│   │   │   ├── images.npy  # (N, 32, 32, 3) numpy array
│   │   │   └── labels.npy  # (N,) numpy array
│   │   └── val/
│   │       ├── images.npy
│   │       └── labels.npy
│   ├── partition_1/
│   ├── ...
│   ├── partition_5/
│   ├── test/
│   │   ├── images.npy  # Centralized test set (10000 samples)
│   │   └── labels.npy
│   └── summary.txt  # Statistics
├── cifar10_C2_6partition/
├── cifar10_C3_6partition/
└── ...
```

## Step 3: Run Training with .npy Data

### Method 1: Use config file

```bash
flwr run . --run-config configs/test_homo_npy.toml
```

### Method 2: Use modified pyproject.toml

Edit `pyproject.toml` to use npy apps:

```toml
[tool.flwr.app.components]
serverapp = "pytorchexample.server_app_npy:app"
clientapp = "pytorchexample.client_app_npy:app"

[tool.flwr.app.config]
distribution = "homo"  # Must match exported data
data-dir = "./data"    # Base directory for .npy partitions
```

Then run:

```bash
flwr run .
```

## Key Files

### Export Scripts
- **export_partitions.py**: Export single distribution
- **export_all_partitions.py**: Export all 9 distributions

### Data Loading
- **pytorchexample/task_npy.py**: Load .npy data utilities
  - `load_npy_partition()`: Load client partition
  - `load_npy_centralized_test()`: Load centralized test set
  - `NumpyDataset`: Custom dataset for .npy files

### Flower Apps (NPY version)
- **pytorchexample/client_app_npy.py**: Client app using .npy data
- **pytorchexample/server_app_npy.py**: Server app using .npy data

### Configurations
- **configs/test_homo_npy.toml**: Example config for .npy mode
- **pyproject_npy.toml**: Example pyproject for .npy mode

## Configuration Parameters

When using .npy data, set these in your config file:

```toml
distribution = "homo"     # Must match exported folder name
num-clients = 6           # Number of partitions
data-dir = "./data"       # Base directory containing partitions
```

## Example: Full Workflow

```bash
# 1. Export homo distribution
python export_partitions.py --distribution homo --num-clients 6

# 2. Run quick test (10 rounds)
flwr run . --run-config configs/test_homo_npy.toml

# 3. Check results
cat data/cifar10_homo_6partition/summary.txt
ls results/
```

## Performance Comparison

| Mode | Data Loading Time per Round | HuggingFace Calls |
|------|------------------------------|-------------------|
| On-the-fly | ~5-10 seconds | Every client init |
| Pre-partitioned .npy | ~0.5-1 second | None |

**Speedup**: ~5-10x faster data loading!

## Troubleshooting

### "FileNotFoundError: [Errno 2] No such file or directory"

Make sure you've exported the distribution first:

```bash
python export_partitions.py --distribution homo --num-clients 6
```

### "Distribution mismatch"

Make sure the `distribution` parameter in your config matches the exported folder name:
- Config: `distribution = "homo"`
- Folder: `data/cifar10_homo_6partition/`

### "ValueError: Cannot find data directory"

Check that `data-dir` points to the correct location:

```toml
data-dir = "./data"  # Should contain cifar10_*_6partition folders
```

## Notes

- Export time: ~2-3 minutes per distribution
- Disk space: ~200-300 MB per distribution
- The exported data is already normalized and ready for training
- You can delete the original HuggingFace cache after export if needed
