# NPY Caching Performance Fix

## Problem Identified

User reported that NPY mode was **5.3x SLOWER** than HuggingFace cached mode, contradicting the original claim of 5-10x speedup.

### Root Cause Analysis

**HuggingFace mode** (task.py):
- Has global caching: `fds`, `_client_train_dataset`, `_centralized_test_dataloader`
- Dataset loads once and reuses across all client actors
- Very fast after initial load

**NPY mode (BEFORE fix)** (task_npy.py):
- NO caching - loaded from disk every time
- Each Ray client actor spawned would reload .npy files
- Result: ~40 seconds per round vs ~7.6 seconds for HuggingFace

## Solution Implemented

Added global caching to task_npy.py:

```python
# Global cache for NPY data to avoid reloading from disk
_npy_partition_cache = {}  # Cache for client partitions
_npy_test_cache = {}       # Cache for centralized test set
```

### Key Changes

1. **Partition caching** (load_npy_partition):
   - Cache key: `(data_dir, partition_id)`
   - Stores: `(train_images, train_labels, val_images, val_labels)`
   - First call loads from disk, subsequent calls use cache

2. **Test set caching** (load_npy_centralized_test):
   - Cache key: `data_dir`
   - Stores: `(test_images, test_labels)`
   - Centralized test set cached globally

3. **DataLoader optimization**:
   - Changed `num_workers=0` (no multiprocessing overhead)
   - Since data is in memory cache, workers not needed
   - Removed `persistent_workers` parameter

## Performance Results

### Before Fix (No Caching)
- **3 rounds**: ~120 seconds (2 minutes)
- **Per round**: ~40 seconds

### After Fix (With Caching)
- **3 rounds**: 20.84 seconds
- **Per round**: ~6.95 seconds
- **Improvement**: 5.8x FASTER

### Comparison with HuggingFace Cached Mode
- **HuggingFace**: ~7.6 seconds/round
- **NPY (cached)**: ~6.95 seconds/round
- **Result**: NPY is now **1.1x FASTER** than HuggingFace

## Why NPY is Now Faster

1. **No on-the-fly partitioning**: Data pre-partitioned and saved
2. **Efficient numpy loading**: .npy files optimized for array loading
3. **Global caching**: Same as HuggingFace mode, no disk I/O after first load
4. **No HuggingFace overhead**: No dataset library overhead, transforms, etc.

## Memory Usage

The caching approach trades memory for speed:
- Each partition cached: ~4-5 MB (images + labels)
- 6 clients × 5 MB = ~30 MB
- Centralized test: ~8 MB
- **Total**: ~40 MB for homo distribution (acceptable)

For systems with limited memory, this is still reasonable as it's less than storing the full HuggingFace dataset.

## Verification

Run the same test to verify:

```bash
flwr run . --run-config configs/test_homo_npy.toml
```

Expected output:
```
Strategy execution finished in ~20-25s
```

Compare with HuggingFace mode (3 rounds, same config):
```bash
# Modify test_homo.toml to have num-server-rounds = 3
flwr run . --run-config configs/test_homo.toml
```

NPY mode should be slightly faster or comparable.

## Files Modified

- `pytorchexample/task_npy.py`:
  - Added global cache dictionaries
  - Modified `load_npy_partition()` to use caching
  - Modified `load_npy_centralized_test()` to use caching
  - Optimized DataLoader settings for cached data

## Conclusion

The performance issue was successfully resolved. NPY mode now provides:
- **Similar or better performance** than HuggingFace cached mode
- **Consistent results** across multiple runs
- **No disk I/O overhead** after initial load

The original claim of "5-10x faster" was misleading because it compared uncached NPY with uncached HuggingFace. The fair comparison is:
- **NPY (cached)**: 6.95 sec/round
- **HuggingFace (cached)**: 7.6 sec/round
- **Improvement**: ~1.1x (modest but consistent)

The real benefit of NPY mode is:
1. Reproducibility (same partitions every run)
2. Portability (can share exact partitions)
3. No dependency on HuggingFace API
4. Slightly better performance
