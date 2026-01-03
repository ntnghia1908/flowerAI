"""Script to export all test case partitions to .npy format."""

from export_partitions import export_federated_dataset

# All test configurations
DISTRIBUTIONS = [
    'homo',      # Homogeneous (IID)
    'C2',        # 2 classes per client
    'C3',        # 3 classes per client
    'C4',        # 4 classes per client
    'C5',        # 5 classes per client
    'Dir0.1',    # Dirichlet alpha=0.1 (very non-IID)
    'Dir0.5',    # Dirichlet alpha=0.5 (moderate non-IID)
    'Dir1.0',    # Dirichlet alpha=1.0 (mild non-IID)
    'Dir10.0',   # Dirichlet alpha=10.0 (nearly IID)
]

NUM_CLIENTS = 6
OUTPUT_DIR = './data'


def main():
    """Export all distributions."""
    print(f"\n{'='*70}")
    print(f"EXPORTING ALL CIFAR-10 PARTITIONS")
    print(f"{'='*70}")
    print(f"Total configurations: {len(DISTRIBUTIONS)}")
    print(f"Clients per config: {NUM_CLIENTS}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*70}\n")

    for i, distribution in enumerate(DISTRIBUTIONS, 1):
        print(f"\n[{i}/{len(DISTRIBUTIONS)}] Processing: {distribution}")
        try:
            export_federated_dataset(
                distribution=distribution,
                num_clients=NUM_CLIENTS,
                output_base_dir=OUTPUT_DIR
            )
        except Exception as e:
            print(f"ERROR exporting {distribution}: {e}")
            continue

    print(f"\n{'='*70}")
    print(f"ALL EXPORTS COMPLETED!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
