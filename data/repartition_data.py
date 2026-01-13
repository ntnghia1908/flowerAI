"""Re-partition CIFAR-10 data with NO replication (disjoint, pathological partition).

This script fixes the data replication issue in C2, C3, C4, C5 distributions.

Usage:
    python repartition_data.py --distribution C2 --verify
    python repartition_data.py --distribution C3 --verify
    python repartition_data.py --distribution C4 --verify
    python repartition_data.py --distribution C5 --verify
    python repartition_data.py --all --verify
"""

import argparse
import numpy as np
from pathlib import Path
from datasets import load_dataset
from collections import defaultdict
import os

# Suppress HuggingFace offline warning
os.environ['HF_DATASETS_OFFLINE'] = '1'


def load_cifar10_data():
    """Load CIFAR-10 train and test datasets.

    Returns:
        tuple: (train_dataset, test_dataset)
    """
    print("Loading CIFAR-10 dataset from HuggingFace...")
    train_dataset = load_dataset("uoft-cs/cifar10", split="train")
    test_dataset = load_dataset("uoft-cs/cifar10", split="test")

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    return train_dataset, test_dataset


def organize_by_class(dataset):
    """Organize dataset samples by class label.

    Args:
        dataset: HuggingFace dataset

    Returns:
        dict: {class_id: list of (image, label) tuples}
    """
    class_data = defaultdict(list)

    for item in dataset:
        img = np.array(item['img'])  # Convert PIL Image to numpy
        label = item['label']
        class_data[label].append((img, label))

    return class_data


def create_pathological_partition(
    class_data,
    num_clients=6,
    classes_per_client=2,
    num_classes=10,
    train_ratio=0.8
):
    """Create disjoint pathological partition (NO replication).

    Based on FedAvg paper (McMahan et al., 2017) pathological partition.
    Each class is split into portions and distributed to clients WITHOUT overlap.

    Args:
        class_data: dict of {class_id: list of (image, label)}
        num_clients: Number of clients (default: 6)
        classes_per_client: Classes per client (2, 3, 4, or 5)
        num_classes: Total classes (default: 10)
        train_ratio: Ratio of train/val split (default: 0.8)

    Returns:
        list: Client datasets, each containing (train_images, train_labels, val_images, val_labels)
    """
    print(f"\nCreating pathological partition:")
    print(f"  Clients: {num_clients}")
    print(f"  Classes per client: {classes_per_client}")
    print(f"  Total classes: {num_classes}")
    print(f"  Train/Val ratio: {train_ratio:.1%}/{1-train_ratio:.1%}")

    # Calculate class assignments
    total_assignments = num_clients * classes_per_client
    appearances_per_class = total_assignments / num_classes

    print(f"  Total class assignments: {total_assignments}")
    print(f"  Average appearances per class: {appearances_per_class:.2f}x")

    # Define class assignments for each client
    class_assignments = get_class_assignments(num_clients, classes_per_client, num_classes)

    # Print class assignments
    print("\nClass assignments:")
    for client_id, classes in enumerate(class_assignments):
        print(f"  Client {client_id}: {classes}")

    # Split each class into portions (stratified sampling)
    class_portions = {}
    for cls in range(num_classes):
        # How many clients need this class?
        count = sum(1 for client_classes in class_assignments if cls in client_classes)

        # Get all samples for this class
        samples = class_data[cls]
        np.random.shuffle(samples)  # Shuffle for randomness

        # Split into portions
        portion_size = len(samples) // count if count > 0 else 0
        portions = []
        for i in range(count):
            start_idx = i * portion_size
            end_idx = (i + 1) * portion_size if i < count - 1 else len(samples)
            portions.append(samples[start_idx:end_idx])

        class_portions[cls] = portions
        print(f"  Class {cls}: {len(samples)} samples -> {count} portions of ~{portion_size} each")

    # Assign portions to clients
    class_portion_idx = {cls: 0 for cls in range(num_classes)}
    client_datasets = []

    print("\nAssigning data to clients:")
    for client_id in range(num_clients):
        client_data = []

        # Collect data from assigned classes
        for cls in class_assignments[client_id]:
            portion = class_portions[cls][class_portion_idx[cls]]
            client_data.extend(portion)
            class_portion_idx[cls] += 1

        # Shuffle client data
        np.random.shuffle(client_data)

        # Split into train/val
        split_idx = int(len(client_data) * train_ratio)
        train_data = client_data[:split_idx]
        val_data = client_data[split_idx:]

        # Separate images and labels
        train_images = np.array([img for img, _ in train_data])
        train_labels = np.array([label for _, label in train_data])
        val_images = np.array([img for img, _ in val_data])
        val_labels = np.array([label for _, label in val_data])

        client_datasets.append((train_images, train_labels, val_images, val_labels))

        print(f"  Client {client_id}: {len(train_data)} train, {len(val_data)} val samples (classes: {class_assignments[client_id]})")

    # Verify totals
    total_train = sum(len(train_imgs) for train_imgs, _, _, _ in client_datasets)
    total_val = sum(len(val_imgs) for _, _, val_imgs, _ in client_datasets)
    total_all = total_train + total_val

    print(f"\nTotals:")
    print(f"  Train: {total_train}")
    print(f"  Val: {total_val}")
    print(f"  Total: {total_all}")
    print(f"  Replication factor: {total_all / 50000:.2f}x")

    if total_all > 50100:  # Allow small rounding error
        print(f"  WARNING: Total exceeds 50k! Replication detected!")

    return client_datasets


def get_class_assignments(num_clients, classes_per_client, num_classes):
    """Get class assignments for each client.

    Implements different strategies based on classes_per_client to avoid
    catastrophic patterns (e.g., binary split for C5).

    Args:
        num_clients: Number of clients
        classes_per_client: Classes per client
        num_classes: Total classes

    Returns:
        list: List of class assignments for each client
    """
    if classes_per_client == 2:
        # C2: Simple rotation
        # Total: 6 × 2 = 12 assignments, 10 classes available
        # Some classes appear 1x, others 2x
        return [
            [0, 1],  # Client 0
            [2, 3],  # Client 1
            [4, 5],  # Client 2
            [6, 7],  # Client 3
            [8, 9],  # Client 4
            [0, 5],  # Client 5 (overlap classes 0, 5 - will use DIFFERENT samples)
        ]

    elif classes_per_client == 3:
        # C3: Rotation with overlap
        # Total: 6 × 3 = 18 assignments, 10 classes
        # Each class appears ~1.8x (some 1x, some 2x)
        return [
            [0, 1, 2],  # Client 0
            [3, 4, 5],  # Client 1
            [6, 7, 8],  # Client 2
            [9, 0, 1],  # Client 3 (overlap: 0, 1)
            [2, 3, 4],  # Client 4 (overlap: 2, 3, 4)
            [5, 6, 7],  # Client 5 (overlap: 5, 6, 7)
        ]

    elif classes_per_client == 4:
        # C4: Balanced rotation
        # Total: 6 × 4 = 24 assignments, 10 classes
        # Each class appears 2.4x (some 2x, some 3x)
        return [
            [0, 1, 2, 3],  # Client 0
            [4, 5, 6, 7],  # Client 1
            [8, 9, 0, 1],  # Client 2 (overlap: 0, 1)
            [2, 3, 4, 5],  # Client 3 (overlap: 2, 3, 4, 5)
            [6, 7, 8, 9],  # Client 4 (overlap: 6, 7, 8, 9)
            [0, 1, 2, 3],  # Client 5 (overlap: 0, 1, 2, 3)
        ]

    elif classes_per_client == 5:
        # C5: CRITICAL - Avoid binary split!
        # Total: 6 × 5 = 30 assignments, 10 classes
        # Each class appears 3.0x
        # Use overlapping rotation to prevent binary split
        return [
            [0, 1, 2, 3, 4],  # Client 0: First half
            [5, 6, 7, 8, 9],  # Client 1: Second half
            [0, 2, 4, 6, 8],  # Client 2: Even classes (CROSS-GROUP!)
            [1, 3, 5, 7, 9],  # Client 3: Odd classes (CROSS-GROUP!)
            [0, 1, 5, 6, 7],  # Client 4: Mixed cross-group
            [2, 3, 4, 8, 9],  # Client 5: Mixed cross-group
        ]

    else:
        raise ValueError(f"Unsupported classes_per_client: {classes_per_client}")


def save_partition(client_datasets, test_dataset, output_dir, distribution_name):
    """Save partitioned data to .npy files.

    Args:
        client_datasets: List of (train_images, train_labels, val_images, val_labels)
        test_dataset: HuggingFace test dataset
        output_dir: Output directory (e.g., "data")
        distribution_name: Distribution name (e.g., "C2")
    """
    num_clients = len(client_datasets)
    output_path = Path(output_dir) / f"cifar10_{distribution_name}_6partition"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving partitions to: {output_path}")

    # Save each client partition
    for client_id, (train_images, train_labels, val_images, val_labels) in enumerate(client_datasets):
        client_dir = output_path / f"partition_{client_id}"
        train_dir = client_dir / "train"
        val_dir = client_dir / "val"

        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        # Save train data
        np.save(train_dir / "images.npy", train_images)
        np.save(train_dir / "labels.npy", train_labels)

        # Save val data
        np.save(val_dir / "images.npy", val_images)
        np.save(val_dir / "labels.npy", val_labels)

        print(f"  Saved partition {client_id}: {len(train_images)} train, {len(val_images)} val")

    # Save centralized test set
    test_dir = output_path / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    test_images = np.array([np.array(item['img']) for item in test_dataset])
    test_labels = np.array([item['label'] for item in test_dataset])

    np.save(test_dir / "images.npy", test_images)
    np.save(test_dir / "labels.npy", test_labels)

    print(f"  Saved centralized test set: {len(test_images)} samples")

    # Create summary file
    create_summary_file(client_datasets, output_path, distribution_name)

    print(f"\n[SUCCESS] Partition saved successfully!")


def create_summary_file(client_datasets, output_path, distribution_name):
    """Create summary.txt file with partition statistics.

    Args:
        client_datasets: List of (train_images, train_labels, val_images, val_labels)
        output_path: Path to output directory
        distribution_name: Distribution name
    """
    summary_path = output_path / "summary.txt"

    with open(summary_path, 'w') as f:
        f.write("CIFAR-10 Federated Learning Dataset\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Distribution: {distribution_name} (FIXED - NO REPLICATION)\n")
        f.write(f"Number of clients: {len(client_datasets)}\n")
        f.write(f"Test set samples: 10000\n\n")
        f.write("Client Statistics:\n")
        f.write("-" * 70 + "\n\n")

        total_train = 0
        total_val = 0

        for client_id, (train_images, train_labels, val_images, val_labels) in enumerate(client_datasets):
            f.write(f"Partition {client_id}:\n")
            f.write(f"  Train samples: {len(train_images)}\n")
            f.write(f"  Val samples:   {len(val_images)}\n")

            # Get unique classes
            unique_classes = sorted(set(train_labels.tolist() + val_labels.tolist()))
            f.write(f"  Assigned classes: {unique_classes}\n\n")

            total_train += len(train_images)
            total_val += len(val_images)

        f.write("-" * 70 + "\n")
        f.write(f"Total train samples: {total_train}\n")
        f.write(f"Total val samples:   {total_val}\n")
        f.write(f"Total samples:       {total_train + total_val}\n")
        f.write(f"Replication factor:  {(total_train + total_val) / 50000:.2f}x\n")

        if total_train + total_val <= 50100:  # Allow small rounding
            f.write("\n[PASS] NO REPLICATION - Fair partition!\n")
        else:
            f.write("\n[WARNING] Replication detected!\n")


def verify_partition(output_dir, distribution_name):
    """Verify that partition has no replication and correct totals.

    Args:
        output_dir: Output directory
        distribution_name: Distribution name

    Returns:
        bool: True if verification passed
    """
    output_path = Path(output_dir) / f"cifar10_{distribution_name}_6partition"

    if not output_path.exists():
        print(f"[ERROR] Partition not found: {output_path}")
        return False

    print(f"\nVerifying partition: {distribution_name}")
    print("-" * 70)

    all_passed = True

    # Check each client partition
    total_train = 0
    total_val = 0
    all_train_samples = []
    all_val_samples = []

    for client_id in range(6):
        client_dir = output_path / f"partition_{client_id}"
        train_dir = client_dir / "train"
        val_dir = client_dir / "val"

        # Load data
        train_images = np.load(train_dir / "images.npy")
        train_labels = np.load(train_dir / "labels.npy")
        val_images = np.load(val_dir / "images.npy")
        val_labels = np.load(val_dir / "labels.npy")

        total_train += len(train_images)
        total_val += len(val_images)

        # Store samples for duplicate check
        for i in range(len(train_images)):
            sample_hash = hash(train_images[i].tobytes())
            all_train_samples.append(sample_hash)

        for i in range(len(val_images)):
            sample_hash = hash(val_images[i].tobytes())
            all_val_samples.append(sample_hash)

        unique_classes = sorted(set(train_labels.tolist() + val_labels.tolist()))
        print(f"  Client {client_id}: {len(train_images)} train, {len(val_images)} val | Classes: {unique_classes}")

    # Check totals
    total_all = total_train + total_val
    replication_factor = total_all / 50000

    print(f"\nTotals:")
    print(f"  Train: {total_train}")
    print(f"  Val: {total_val}")
    print(f"  Total: {total_all}")
    print(f"  Replication factor: {replication_factor:.4f}x")

    # Check for duplicates
    unique_train = len(set(all_train_samples))
    unique_val = len(set(all_val_samples))

    print(f"\nDuplicate check:")
    print(f"  Train samples: {total_train} total, {unique_train} unique")
    print(f"  Val samples: {total_val} total, {unique_val} unique")

    if unique_train < total_train:
        print(f"  [FAIL] Train has {total_train - unique_train} duplicates!")
        all_passed = False
    else:
        print(f"  [PASS] No train duplicates")

    if unique_val < total_val:
        print(f"  [FAIL] Val has {total_val - unique_val} duplicates!")
        all_passed = False
    else:
        print(f"  [PASS] No val duplicates")

    # Check replication factor
    if replication_factor > 1.01:  # Allow 1% tolerance
        print(f"  [FAIL] Replication factor {replication_factor:.2f}x > 1.01")
        all_passed = False
    else:
        print(f"  [PASS] Replication factor <= 1.01")

    # Check test set
    test_dir = output_path / "test"
    test_images = np.load(test_dir / "images.npy")
    test_labels = np.load(test_dir / "labels.npy")

    print(f"\nTest set:")
    print(f"  Samples: {len(test_images)}")

    if len(test_images) != 10000:
        print(f"  [FAIL] Expected 10000 test samples, got {len(test_images)}")
        all_passed = False
    else:
        print(f"  [PASS] 10000 test samples")

    # Final result
    print("\n" + "=" * 70)
    if all_passed:
        print("[PASS] ALL CHECKS PASSED - Partition is FAIR!")
    else:
        print("[FAIL] SOME CHECKS FAILED - Partition has issues!")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Re-partition CIFAR-10 data without replication")
    parser.add_argument("--distribution", type=str, choices=["C2", "C3", "C4", "C5", "all"],
                       help="Distribution to create (C2, C3, C4, C5, or all)")
    parser.add_argument("--output-dir", type=str, default="data",
                       help="Output directory (default: data)")
    parser.add_argument("--verify", action="store_true",
                       help="Verify partition after creation")
    parser.add_argument("--verify-only", action="store_true",
                       help="Only verify existing partitions (no creation)")

    args = parser.parse_args()

    # Verify-only mode
    if args.verify_only:
        distributions = ["C2", "C3", "C4", "C5"] if args.distribution == "all" or args.distribution is None else [args.distribution]

        all_passed = True
        for dist in distributions:
            passed = verify_partition(args.output_dir, dist)
            all_passed = all_passed and passed
            print()

        if all_passed:
            print("[PASS] All verifications PASSED!")
            return 0
        else:
            print("[FAIL] Some verifications FAILED!")
            return 1

    # Create mode
    if args.distribution is None:
        print("Error: --distribution required (unless using --verify-only)")
        return 1

    # Load CIFAR-10 data once
    train_dataset, test_dataset = load_cifar10_data()
    class_data = organize_by_class(train_dataset)

    # Distributions to create
    distributions = {
        "C2": 2,
        "C3": 3,
        "C4": 4,
        "C5": 5
    }

    if args.distribution == "all":
        targets = distributions.items()
    else:
        targets = [(args.distribution, distributions[args.distribution])]

    # Create partitions
    for dist_name, classes_per_client in targets:
        print("\n" + "=" * 70)
        print(f"Creating partition: {dist_name}")
        print("=" * 70)

        client_datasets = create_pathological_partition(
            class_data,
            num_clients=6,
            classes_per_client=classes_per_client,
            num_classes=10,
            train_ratio=0.8
        )

        save_partition(client_datasets, test_dataset, args.output_dir, dist_name)

        if args.verify:
            verify_partition(args.output_dir, dist_name)

    print("\n" + "=" * 70)
    print("[SUCCESS] All partitions created successfully!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
