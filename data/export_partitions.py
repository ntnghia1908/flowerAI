"""Script to export CIFAR-10 partitions to .npy format for faster loading."""

import os
import numpy as np
from pathlib import Path
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from pytorchexample.partitioner import LabelSkewPartitioner


def export_partition_to_npy(dataset, output_dir, partition_name):
    """Export a dataset partition to .npy files.

    Args:
        dataset: HuggingFace Dataset object
        output_dir: Directory to save .npy files
        partition_name: Name of partition (e.g., 'train', 'val')
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert dataset to numpy arrays
    images = []
    labels = []

    for example in dataset:
        images.append(np.array(example['img']))
        labels.append(example['label'])

    images = np.array(images)
    labels = np.array(labels)

    # Save to .npy files
    np.save(os.path.join(output_dir, 'images.npy'), images)
    np.save(os.path.join(output_dir, 'labels.npy'), labels)

    print(f"Saved {partition_name}: {len(labels)} samples, shape={images.shape}")
    return len(labels)


def export_federated_dataset(distribution, num_clients, output_base_dir):
    """Export partitioned CIFAR-10 dataset based on distribution type.

    Args:
        distribution: One of 'homo', 'C1'-'C5', 'Dir0.1', 'Dir0.5', 'Dir1.0', 'Dir10.0'
        num_clients: Number of clients (partitions)
        output_base_dir: Base directory for output
    """
    # Create output directory
    output_dir = os.path.join(output_base_dir, f'cifar10_{distribution}_{num_clients}partition')
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Exporting: {distribution} with {num_clients} clients")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")

    # Create partitioner based on distribution type
    if distribution == 'homo':
        partitioner = IidPartitioner(num_partitions=num_clients)
        use_federated_dataset = True
    elif distribution.startswith('C'):
        # Label Skew: C1-C5 (number of classes per client)
        classes_per_client = int(distribution[1:])
        partitioner = LabelSkewPartitioner(
            num_partitions=num_clients,
            classes_per_client=classes_per_client
        )
        use_federated_dataset = False
    elif distribution.startswith('Dir'):
        # Dirichlet distribution
        alpha = float(distribution[3:].replace('p', '.'))
        partitioner = DirichletPartitioner(
            num_partitions=num_clients,
            partition_by="label",
            alpha=alpha,
            min_partition_size=10
        )
        use_federated_dataset = True
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    # Export client partitions
    client_stats = []

    if use_federated_dataset:
        # Use FederatedDataset for IID and Dirichlet
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner}
        )

        for partition_id in range(num_clients):
            print(f"Processing partition {partition_id}/{num_clients-1}...")

            # Load partition
            partition = fds.load_partition(partition_id)

            # Split into train/val (80/20)
            partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

            # Export train
            train_dir = os.path.join(output_dir, f'partition_{partition_id}', 'train')
            train_count = export_partition_to_npy(
                partition_train_test['train'],
                train_dir,
                f"partition_{partition_id}/train"
            )

            # Export val
            val_dir = os.path.join(output_dir, f'partition_{partition_id}', 'val')
            val_count = export_partition_to_npy(
                partition_train_test['test'],
                val_dir,
                f"partition_{partition_id}/val"
            )

            client_stats.append({
                'partition_id': partition_id,
                'train_samples': train_count,
                'val_samples': val_count
            })

    else:
        # Use LabelSkewPartitioner (custom)
        train_dataset = load_dataset("uoft-cs/cifar10", split="train")

        for partition_id in range(num_clients):
            print(f"Processing partition {partition_id}/{num_clients-1}...")

            # Get assigned classes for this partition
            assigned_classes = partitioner.get_partition_classes(partition_id)

            # Filter dataset by assigned classes
            def filter_by_classes(example):
                return example['label'] in assigned_classes

            filtered_dataset = train_dataset.filter(filter_by_classes)

            # Split into train/val (80/20)
            partition_train_test = filtered_dataset.train_test_split(test_size=0.2, seed=42)

            # Export train
            train_dir = os.path.join(output_dir, f'partition_{partition_id}', 'train')
            train_count = export_partition_to_npy(
                partition_train_test['train'],
                train_dir,
                f"partition_{partition_id}/train"
            )

            # Export val
            val_dir = os.path.join(output_dir, f'partition_{partition_id}', 'val')
            val_count = export_partition_to_npy(
                partition_train_test['test'],
                val_dir,
                f"partition_{partition_id}/val"
            )

            client_stats.append({
                'partition_id': partition_id,
                'train_samples': train_count,
                'val_samples': val_count,
                'assigned_classes': sorted(assigned_classes)
            })

    # Export centralized test set
    print(f"\nProcessing centralized test set...")
    test_dataset = load_dataset("uoft-cs/cifar10", split="test")
    test_dir = os.path.join(output_dir, 'test')
    test_count = export_partition_to_npy(test_dataset, test_dir, "centralized test")

    # Write summary
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"CIFAR-10 Federated Learning Dataset\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Distribution: {distribution}\n")
        f.write(f"Number of clients: {num_clients}\n")
        f.write(f"Test set samples: {test_count}\n\n")

        f.write(f"Client Statistics:\n")
        f.write(f"{'-'*70}\n")

        total_train = 0
        total_val = 0

        for stats in client_stats:
            f.write(f"\nPartition {stats['partition_id']}:\n")
            f.write(f"  Train samples: {stats['train_samples']}\n")
            f.write(f"  Val samples:   {stats['val_samples']}\n")
            if 'assigned_classes' in stats:
                f.write(f"  Assigned classes: {stats['assigned_classes']}\n")

            total_train += stats['train_samples']
            total_val += stats['val_samples']

        f.write(f"\n{'-'*70}\n")
        f.write(f"Total train samples: {total_train}\n")
        f.write(f"Total val samples:   {total_val}\n")
        f.write(f"Total samples:       {total_train + total_val}\n")

    print(f"\n{'='*70}")
    print(f"Export completed!")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*70}\n")


def main():
    """Export all test configurations."""
    import argparse

    parser = argparse.ArgumentParser(description='Export CIFAR-10 partitions to .npy format')
    parser.add_argument('--distribution', type=str, required=True,
                        help='Distribution type: homo, C2-C5, Dir0.1, Dir0.5, Dir1.0, Dir10.0')
    parser.add_argument('--num-clients', type=int, default=6,
                        help='Number of clients (default: 6)')
    parser.add_argument('--output-dir', type=str, default='./data',
                        help='Output base directory (default: ./data)')

    args = parser.parse_args()

    export_federated_dataset(
        distribution=args.distribution,
        num_clients=args.num_clients,
        output_base_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
