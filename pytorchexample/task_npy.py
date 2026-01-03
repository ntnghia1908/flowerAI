"""Task utilities for loading pre-partitioned .npy data (faster than on-the-fly partitioning)."""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Compose, Normalize, ToTensor
from pytorchexample.task import Net, apply_transforms, pytorch_transforms

# Global cache for NPY data to avoid reloading from disk
_npy_partition_cache = {}
_npy_test_cache = {}


class NumpyDataset(torch.utils.data.Dataset):
    """Dataset wrapper for numpy arrays with transforms."""

    def __init__(self, images, labels, transform=None):
        """Initialize dataset.

        Args:
            images: numpy array of shape (N, H, W, C)
            labels: numpy array of shape (N,)
            transform: Optional transform to apply
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)

        return {"img": image, "label": label}


def load_npy_partition(data_dir, partition_id, batch_size):
    """Load pre-partitioned data from .npy files (with caching).

    Args:
        data_dir: Base directory containing partitions (e.g., './data/cifar10_homo_6partition')
        partition_id: Partition ID (0 to num_partitions-1)
        batch_size: Batch size for DataLoader

    Returns:
        trainloader, testloader
    """
    global _npy_partition_cache

    # Create cache key
    cache_key = (data_dir, partition_id)

    # Check if already cached
    if cache_key in _npy_partition_cache:
        train_images, train_labels, val_images, val_labels = _npy_partition_cache[cache_key]
    else:
        # Construct paths
        train_dir = os.path.join(data_dir, f'partition_{partition_id}', 'train')
        val_dir = os.path.join(data_dir, f'partition_{partition_id}', 'val')

        # Load train data
        train_images = np.load(os.path.join(train_dir, 'images.npy'))
        train_labels = np.load(os.path.join(train_dir, 'labels.npy'))

        # Load val data
        val_images = np.load(os.path.join(val_dir, 'images.npy'))
        val_labels = np.load(os.path.join(val_dir, 'labels.npy'))

        # Cache the loaded data
        _npy_partition_cache[cache_key] = (train_images, train_labels, val_images, val_labels)

    # Create datasets with transforms
    train_dataset = NumpyDataset(train_images, train_labels, transform=pytorch_transforms)
    val_dataset = NumpyDataset(val_images, val_labels, transform=pytorch_transforms)

    # Create dataloaders with caching-optimized settings
    import torch
    use_cuda = torch.cuda.is_available()

    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # No workers since data is already cached in memory
        pin_memory=use_cuda,  # Only use pin_memory if GPU available
    )
    testloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=0,  # No workers for validation
        pin_memory=use_cuda  # Only use pin_memory if GPU available
    )

    return trainloader, testloader


def load_npy_centralized_test(data_dir, batch_size=128):
    """Load centralized test set from .npy files (with caching).

    Args:
        data_dir: Base directory containing partitions
        batch_size: Batch size for DataLoader

    Returns:
        test_dataloader
    """
    global _npy_test_cache

    # Check if already cached
    if data_dir in _npy_test_cache:
        test_images, test_labels = _npy_test_cache[data_dir]
    else:
        # Construct path
        test_dir = os.path.join(data_dir, 'test')

        # Load test data
        test_images = np.load(os.path.join(test_dir, 'images.npy'))
        test_labels = np.load(os.path.join(test_dir, 'labels.npy'))

        # Cache the loaded data
        _npy_test_cache[data_dir] = (test_images, test_labels)

    # Create dataset with transforms
    test_dataset = NumpyDataset(test_images, test_labels, transform=pytorch_transforms)

    # Create dataloader (optimized for CPU)
    import torch
    use_cuda = torch.cuda.is_available()

    testloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=0,  # No workers for centralized test (avoid overhead)
        pin_memory=use_cuda  # Only use pin_memory if GPU available
    )

    return testloader


def get_data_dir(distribution, num_clients, base_dir='./data'):
    """Get data directory path for a specific distribution.

    Args:
        distribution: Distribution type (e.g., 'homo', 'C2', 'Dir0.5')
        num_clients: Number of clients
        base_dir: Base directory for data

    Returns:
        Path to data directory
    """
    return os.path.join(base_dir, f'cifar10_{distribution}_{num_clients}partition')


# Example usage:
if __name__ == '__main__':
    # Test loading
    data_dir = get_data_dir('homo', 6)
    print(f"Loading from: {data_dir}")

    # Load partition 0
    trainloader, testloader = load_npy_partition(data_dir, partition_id=0, batch_size=32)

    print(f"Train batches: {len(trainloader)}")
    print(f"Test batches: {len(testloader)}")

    # Test one batch
    for batch in trainloader:
        print(f"Image batch shape: {batch['img'].shape}")
        print(f"Label batch shape: {batch['label'].shape}")
        break

    # Load centralized test
    centralized_test = load_npy_centralized_test(data_dir)
    print(f"Centralized test batches: {len(centralized_test)}")
