"""Data partitioning utilities for federated learning experiments."""

from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from typing import Optional
import numpy as np


class LabelSkewPartitioner:
    """Partitioner that assigns exactly k classes to each client (extreme label skew).

    This creates non-IID data distribution where each client only has access to
    a fixed number of classes, simulating extreme label skew scenarios.

    Args:
        num_partitions: Number of clients
        classes_per_client: Number of classes each client should have (k)
        num_classes: Total number of classes in dataset (default: 10 for CIFAR-10)
    """

    def __init__(self, num_partitions: int, classes_per_client: int, num_classes: int = 10):
        self.num_partitions = num_partitions
        self.classes_per_client = classes_per_client
        self.num_classes = num_classes
        self._partition_id_to_classes = {}

        # Assign classes to each client
        self._assign_classes_to_clients()

    def _assign_classes_to_clients(self):
        """Assign k classes to each client in a balanced way."""
        # Create a list of class assignments
        all_classes = list(range(self.num_classes))

        for partition_id in range(self.num_partitions):
            # Rotate through classes to ensure balanced distribution
            start_idx = (partition_id * self.classes_per_client) % self.num_classes
            selected_classes = []

            for i in range(self.classes_per_client):
                class_idx = (start_idx + i) % self.num_classes
                selected_classes.append(all_classes[class_idx])

            self._partition_id_to_classes[partition_id] = sorted(selected_classes)

    def load_partition(self, partition_id: int):
        """Load a specific partition containing only assigned classes.

        Args:
            partition_id: ID of the partition to load

        Returns:
            Filtered dataset containing only samples from assigned classes
        """
        if partition_id not in self._partition_id_to_classes:
            raise ValueError(f"Invalid partition_id: {partition_id}")

        assigned_classes = self._partition_id_to_classes[partition_id]

        # This will be called by the dataset wrapper
        # We return a filter function that will be applied to the dataset
        return lambda dataset: self._filter_by_classes(dataset, assigned_classes)

    def _filter_by_classes(self, dataset, assigned_classes):
        """Filter dataset to only include samples from assigned classes."""
        # This is a placeholder - actual implementation depends on dataset structure
        # The real filtering will happen in load_data function in task.py
        return dataset

    def get_partition_classes(self, partition_id: int):
        """Get the classes assigned to a specific partition."""
        return self._partition_id_to_classes.get(partition_id, [])


def get_partitioner(distribution_type: str, num_partitions: int, classes_per_client: Optional[int] = None):
    """Get data partitioner based on distribution type.

    Args:
        distribution_type: Type of data distribution
            - "homo" or "iid": IID partitioning
            - "Dir(10.0)", "Dir(1.0)", "Dir(0.5)", "Dir(0.1)", "Dir(0.01)": Dirichlet partitioning
            - "label-skew" or "C(k)": Label skew with k classes per client
        num_partitions: Number of partitions (clients)
        classes_per_client: Number of classes per client (only for label-skew)

    Returns:
        Partitioner object from flwr_datasets or custom LabelSkewPartitioner

    Examples:
        >>> partitioner = get_partitioner("homo", 10)
        >>> partitioner = get_partitioner("Dir(0.5)", 10)
        >>> partitioner = get_partitioner("label-skew", 6, classes_per_client=2)
        >>> partitioner = get_partitioner("C(3)", 6)  # 3 classes per client
    """
    # Handle IID/homogeneous case
    if distribution_type.lower() in ["homo", "iid"]:
        return IidPartitioner(num_partitions=num_partitions)

    # Handle label skew with fixed classes per client
    if (distribution_type.lower() == "label-skew" or
        distribution_type.startswith("C") or
        distribution_type.lower().startswith("labelskew")):
        # Extract k from different formats
        if distribution_type.startswith("C(") and distribution_type.endswith(")"):
            try:
                k_str = distribution_type[2:-1]  # Remove "C(" and ")"
                k = int(k_str)
                return LabelSkewPartitioner(num_partitions, k, num_classes=10)
            except ValueError:
                raise ValueError(f"Invalid classes-per-client value: {k_str}")
        elif distribution_type.startswith("C") and len(distribution_type) > 1:
            # Format: C1, C2, C3, etc.
            try:
                k_str = distribution_type[1:]
                k = int(k_str)
                return LabelSkewPartitioner(num_partitions, k, num_classes=10)
            except ValueError:
                raise ValueError(f"Invalid C format: {distribution_type}. Use Ck where k is number of classes")
        elif distribution_type.lower().startswith("labelskew"):
            # Format: labelskew1, labelskew2, etc.
            try:
                k_str = distribution_type.lower().replace("labelskew", "")
                k = int(k_str)
                return LabelSkewPartitioner(num_partitions, k, num_classes=10)
            except ValueError:
                raise ValueError(f"Invalid labelskew format: {distribution_type}. Use labelskewk where k is number of classes")
        elif classes_per_client is not None:
            return LabelSkewPartitioner(num_partitions, classes_per_client, num_classes=10)
        else:
            raise ValueError("Label-skew partitioner requires classes_per_client parameter or Ck/labelskewk format")

    # Handle Dirichlet distribution
    if distribution_type.startswith("Dir(") and distribution_type.endswith(")"):
        try:
            # Extract alpha value from "Dir(0.5)" format
            alpha_str = distribution_type[4:-1]  # Remove "Dir(" and ")"
            alpha = float(alpha_str)

            return DirichletPartitioner(
                num_partitions=num_partitions,
                partition_by="label",
                alpha=alpha,
                min_partition_size=10,  # Ensure each partition has at least 10 samples
                self_balancing=True
            )
        except ValueError:
            raise ValueError(f"Invalid Dirichlet alpha value: {alpha_str}")

    raise ValueError(f"Unknown distribution type: {distribution_type}")


def get_distribution_name(distribution_type: str) -> str:
    """Get clean distribution name for file naming.

    Args:
        distribution_type: Type of data distribution

    Returns:
        Clean name suitable for filenames

    Examples:
        >>> get_distribution_name("homo")
        'homo'
        >>> get_distribution_name("Dir(0.5)")
        'Dir0.5'
        >>> get_distribution_name("C(2)")
        'C2'
    """
    if distribution_type.lower() in ["homo", "iid"]:
        return "homo"

    if distribution_type.lower() == "label-skew":
        return "label-skew"

    if distribution_type.lower().startswith("labelskew"):
        k_str = distribution_type.lower().replace("labelskew", "")
        return f"C{k_str}"

    if distribution_type.startswith("C(") and distribution_type.endswith(")"):
        k_str = distribution_type[2:-1]
        return f"C{k_str}"

    if distribution_type.startswith("C") and len(distribution_type) > 1:
        k_str = distribution_type[1:]
        return f"C{k_str}"

    if distribution_type.startswith("Dir(") and distribution_type.endswith(")"):
        alpha_str = distribution_type[4:-1].replace(".", "p")
        return f"Dir{alpha_str}"

    return distribution_type.replace("(", "").replace(")", "").replace(".", "p")
