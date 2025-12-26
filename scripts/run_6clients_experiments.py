"""Run experiments with 6 clients and 500 rounds.

This script runs FL experiments with:
- 6 clients total
- All 6 clients participate in each round
- 500 rounds
- All combinations of strategies and distributions
"""

import subprocess
import sys
from pathlib import Path
from itertools import product


# Experiment configurations
STRATEGIES = [
    "FedAvg",
    "FedAvgM",
    "FedProx",
    "FedAdam",
    "FedAdagrad",
    "FedYogi",
]

DISTRIBUTIONS = [
    "homo",
    "Dir(10.0)",
    "Dir(1.0)",
    "Dir(0.5)",
    "Dir(0.1)",
    "Dir(0.01)",
]


def get_distribution_name(dist: str) -> str:
    """Convert distribution string to filename-safe format."""
    if dist.lower() in ["homo", "iid"]:
        return "homo"
    return dist.replace("(", "").replace(")", "").replace(".", "p")


def run_experiment(
    strategy: str,
    distribution: str,
    num_rounds: int = 500,
    num_clients: int = 6,
    batch_size: int = 32,
    learning_rate: float = 0.1,
    local_epochs: int = 1,
    use_gpu: bool = False,
):
    """Run a single FL experiment with 6 clients.

    Args:
        strategy: FL strategy name
        distribution: Data distribution type
        num_rounds: Number of FL rounds (default: 500)
        num_clients: Total number of clients (default: 6)
        batch_size: Batch size for training
        learning_rate: Learning rate
        local_epochs: Number of local epochs
        use_gpu: Whether to use GPU
    """
    # All 6 clients participate in each round
    min_train_nodes = num_clients
    min_eval_nodes = num_clients
    fraction_train = 1.0
    fraction_evaluate = 1.0

    # Create experiment name
    dist_name = get_distribution_name(distribution)
    experiment_name = f"{strategy}_{dist_name}_6clients"

    print(f"\n{'='*80}")
    print(f"Running Experiment: {experiment_name}")
    print(f"  Strategy: {strategy}")
    print(f"  Distribution: {distribution}")
    print(f"  Clients: {num_clients} (all participate each round)")
    print(f"  Rounds: {num_rounds}")
    print(f"{'='*80}\n")

    # Prepare flwr run command
    federation = "local-simulation-gpu" if use_gpu else "local-simulation"

    # Build override arguments
    overrides = " ".join([
        f'strategy="{strategy}"',
        f'distribution="{distribution}"',
        f'experiment-name="{experiment_name}"',
        f"num-server-rounds={num_rounds}",
        f"num-clients={num_clients}",
        f"min-train-nodes={min_train_nodes}",
        f"min-evaluate-nodes={min_eval_nodes}",
        f"fraction-train={fraction_train}",
        f"fraction-evaluate={fraction_evaluate}",
        f"batch-size={batch_size}",
        f"learning-rate={learning_rate}",
        f"local-epochs={local_epochs}",
    ])

    # Construct command
    cmd = [
        "flwr", "run", ".",
        federation,
        "--run-config",
        overrides,
    ]

    # Run experiment
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n[PASS] Experiment {experiment_name} completed successfully!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[FAIL] Experiment {experiment_name} failed!")
        print(f"Error: {e}\n")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run FL experiments with 6 clients and 500 rounds"
    )

    # Experiment selection
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=STRATEGIES,
        default=None,
        help="FL strategies to test"
    )
    parser.add_argument(
        "--distributions",
        nargs="+",
        default=None,
        help="Data distributions to test"
    )

    # Quick presets
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all combinations (6 strategies × 6 distributions = 36 experiments)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with 10 rounds (for debugging)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test with 100 rounds"
    )

    # Experiment parameters
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=500,
        help="Number of FL rounds (default: 500)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate (default: 0.1)"
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=1,
        help="Local epochs (default: 1)"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for training"
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt"
    )

    args = parser.parse_args()

    # Determine configurations to run
    if args.quick:
        strategies = ["FedAvg"]
        distributions = ["homo"]
        num_rounds = 10
    elif args.test:
        strategies = ["FedAvg", "FedAvgM"]
        distributions = ["homo", "Dir(0.5)"]
        num_rounds = 100
    elif args.all:
        strategies = STRATEGIES
        distributions = DISTRIBUTIONS
        num_rounds = args.num_rounds
    else:
        strategies = args.strategies or ["FedAvg"]
        distributions = args.distributions or ["homo"]
        num_rounds = args.num_rounds

    # Calculate total experiments
    total_experiments = len(strategies) * len(distributions)

    print(f"\n{'='*80}")
    print(f"EXPERIMENT BATCH CONFIGURATION - 6 CLIENTS")
    print(f"{'='*80}")
    print(f"Strategies: {strategies}")
    print(f"Distributions: {distributions}")
    print(f"Clients: 6 (all participate each round)")
    print(f"Rounds per experiment: {num_rounds}")
    print(f"Total experiments: {total_experiments}")
    print(f"{'='*80}\n")

    # Estimate time
    if num_rounds == 500:
        est_time_per_exp = "40-80 minutes"
        est_total_time = f"{total_experiments * 40}-{total_experiments * 80} minutes"
    elif num_rounds == 100:
        est_time_per_exp = "8-16 minutes"
        est_total_time = f"{total_experiments * 8}-{total_experiments * 16} minutes"
    else:
        est_time_per_exp = "N/A"
        est_total_time = "N/A"

    print(f"Estimated time per experiment: {est_time_per_exp}")
    print(f"Estimated total time: {est_total_time}")
    print(f"{'='*80}\n")

    # Confirm before running
    if not args.yes:
        response = input(f"Run {total_experiments} experiments? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            print("Cancelled.")
            return

    # Run all experiments
    results = []
    for idx, (strategy, dist) in enumerate(product(strategies, distributions), 1):
        print(f"\n[{idx}/{total_experiments}] ", end="")

        success = run_experiment(
            strategy=strategy,
            distribution=dist,
            num_rounds=num_rounds,
            num_clients=6,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            local_epochs=args.local_epochs,
            use_gpu=args.gpu,
        )

        results.append({
            "strategy": strategy,
            "distribution": dist,
            "success": success
        })

    # Print summary
    print(f"\n{'='*80}")
    print(f"EXPERIMENT BATCH SUMMARY")
    print(f"{'='*80}")
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    print(f"Total: {len(results)} | Successful: {successful} | Failed: {failed}")

    if failed > 0:
        print(f"\nFailed experiments:")
        for r in results:
            if not r["success"]:
                print(f"  - {r['strategy']} / {r['distribution']}")

    print(f"{'='*80}\n")

    print("\nResults saved to: results/")
    print("Analyze with: python analyze_results.py")


if __name__ == "__main__":
    main()
