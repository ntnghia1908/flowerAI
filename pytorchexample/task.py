"""pytorchexample: A Flower / PyTorch app."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from pytorchexample.partitioner import LabelSkewPartitioner

# Global cache for centralized test dataset
_centralized_test_dataloader = None

# Global cache for client train dataset (used by LabelSkewPartitioner)
_client_train_dataset = None


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


fds = None  # Cache FederatedDataset

pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def load_data(partition_id: int, num_partitions: int, batch_size: int, partitioner=None,
              data_source="huggingface", distribution="homo"):
    """Load partition CIFAR10 data.

    Args:
        partition_id: Client partition ID
        num_partitions: Total number of partitions
        batch_size: Batch size for DataLoader
        partitioner: Partitioner object (for HuggingFace mode)
        data_source: "huggingface" (default) or "npy"
        distribution: Distribution name (for NPY mode), e.g. "homo", "C2", "Dir0.5"

    Returns:
        Tuple of (trainloader, testloader)
    """
    global fds, _client_train_dataset

    # NPY mode - use pre-partitioned .npy files
    if data_source == "npy":
        from pytorchexample.task_npy import load_npy_partition, get_data_dir
        data_dir = get_data_dir(distribution, num_partitions, "./data")
        return load_npy_partition(data_dir, partition_id, batch_size)

    # HuggingFace mode - partition on-the-fly
    # Handle LabelSkewPartitioner differently (custom implementation)
    if isinstance(partitioner, LabelSkewPartitioner):
        # Load full CIFAR-10 train dataset (cached)
        if _client_train_dataset is None:
            import os
            os.environ['HF_DATASETS_OFFLINE'] = '1'
            _client_train_dataset = load_dataset("uoft-cs/cifar10", split="train")
        train_dataset = _client_train_dataset

        # Get assigned classes for this partition
        assigned_classes = partitioner.get_partition_classes(partition_id)

        # Filter dataset to only include assigned classes
        def filter_by_classes(example):
            return example['label'] in assigned_classes

        filtered_dataset = train_dataset.filter(filter_by_classes)

        # Divide data: 80% train, 20% test
        partition_train_test = filtered_dataset.train_test_split(test_size=0.2, seed=42)

        # Apply transforms
        partition_train_test = partition_train_test.with_transform(apply_transforms)

        # Create dataloaders
        trainloader = DataLoader(
            partition_train_test["train"], batch_size=batch_size, shuffle=True
        )
        testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)

        return trainloader, testloader

    # Standard FederatedDataset approach for IID and Dirichlet
    # Only initialize `FederatedDataset` once
    if fds is None:
        if partitioner is None:
            partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    # Construct dataloaders
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader


def reset_federated_dataset():
    """Reset the global FederatedDataset cache."""
    global fds, _client_train_dataset
    fds = None
    _client_train_dataset = None


def load_centralized_dataset(data_source="huggingface", distribution="homo", num_clients=6):
    """Load test set and return dataloader (cached).

    Args:
        data_source: "huggingface" (default) or "npy"
        distribution: Distribution name (for NPY mode)
        num_clients: Number of clients (for NPY mode)

    Returns:
        DataLoader for centralized test set
    """
    global _centralized_test_dataloader

    # NPY mode - use pre-partitioned .npy test set
    if data_source == "npy":
        from pytorchexample.task_npy import load_npy_centralized_test, get_data_dir
        data_dir = get_data_dir(distribution, num_clients, "./data")
        return load_npy_centralized_test(data_dir, batch_size=128)

    # HuggingFace mode - load from HuggingFace datasets
    # Return cached dataloader if already loaded
    if _centralized_test_dataloader is not None:
        return _centralized_test_dataloader

    # Load entire test set (first time only)
    import os
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    test_dataset = load_dataset("uoft-cs/cifar10", split="test")
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    _centralized_test_dataloader = DataLoader(dataset, batch_size=128)
    return _centralized_test_dataloader


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
