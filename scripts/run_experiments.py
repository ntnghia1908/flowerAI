"""Automated experiment runner for federated learning experiments.

This script runs experiments with different combinations of:
- FL strategies: FedAvg, FedAvgM, FedProx, FedAdam, FedAdagrad, FedYogi
- Data distributions: homo, Dir(10.0), Dir(1.0), Dir(0.5), Dir(0.1), Dir(0.01)
- Number of clients participating (#C): 1, 2, 3, 4, 5

Usage:
    python run_experiments.py --strategies FedAvg FedAvgM --distributions homo Dir(0.5)
    python run_experiments.py --quick  # Run quick test with fewer rounds
    python run_experiments.py --all    # Run all combinations (long!)
"""

import argparse
import subprocess
import sys
from itertools import product
from pathlib import Path


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

# Number of clients to select each round
NUM_CLIENTS_CONFIGS = {
    "C1": {"min_fit": 1, "min_eval": 1},
    "C2": {"min_fit": 2, "min_eval": 2},
    "C3": {"min_fit": 3, "min_eval": 3},
    "C4": {"min_fit": 4, "min_eval": 4},
    "C5": {"min_fit": 5, "min_eval": 5},
}


def get_distribution_name(dist: str) -> str:
    """Convert distribution string to filename-safe format."""
    if dist.lower() in ["homo", "iid"]:
        return "homo"
    return dist.replace("(", "").replace(")", "").replace(".", "p")


def run_experiment(
    strategy: str,
    distribution: str,
    num_clients_config: str,
    num_rounds: int = 500,
    total_clients: int = 6,
    batch_size: int = 32,
    learning_rate: float = 0.1,
    local_epochs: int = 1,
    use_gpu: bool = False,
):
    """Run a single federated learning experiment.

    Args:
        strategy: FL strategy name
        distribution: Data distribution type
        num_clients_config: Client configuration key (e.g., "C5")
        num_rounds: Number of FL rounds
        total_clients: Total number of clients in the federation
        batch_size: Batch size for training
        learning_rate: Learning rate for local training
        local_epochs: Number of local epochs
        use_gpu: Whether to use GPU
    """
    # Get client configuration
    client_config = NUM_CLIENTS_CONFIGS[num_clients_config]
    min_train_nodes = client_config["min_fit"]
    min_eval_nodes = client_config["min_eval"]

    # Calculate fractions
    fraction_train = min_train_nodes / total_clients
    fraction_evaluate = min_eval_nodes / total_clients

    # Create experiment name
    dist_name = get_distribution_name(distribution)
    experiment_name = f"{strategy}_{dist_name}_{num_clients_config}"

    print(f"\n{'='*80}")
    print(f"Running Experiment: {experiment_name}")
    print(f"  Strategy: {strategy}")
    print(f"  Distribution: {distribution}")
    print(f"  Clients per round: {min_train_nodes}/{total_clients}")
    print(f"  Rounds: {num_rounds}")
    print(f"{'='*80}\n")

    # Prepare flwr run command
    federation = "local-simulation-gpu" if use_gpu else "local-simulation"

    # Build override arguments as a single space-separated string
    # String values need quotes, numeric values don't
    overrides = " ".join([
        f'strategy="{strategy}"',
        f'distribution="{distribution}"',
        f'experiment-name="{experiment_name}"',
        f"num-server-rounds={num_rounds}",
        f"num-clients={total_clients}",
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
    parser = argparse.ArgumentParser(
        description="Run federated learning experiments automatically"
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
    parser.add_argument(
        "--client-configs",
        nargs="+",
        choices=list(NUM_CLIENTS_CONFIGS.keys()),
        default=None,
        help="Client configurations to test (C1, C2, etc.)"
    )

    # Quick presets
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all combinations (warning: very long!)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with 10 rounds (for debugging)"
    )
    parser.add_argument(
        "--medium",
        action="store_true",
        help="Medium test with 100 rounds"
    )

    # Experiment parameters
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=500,
        help="Number of FL rounds (default: 500)"
    )
    parser.add_argument(
        "--total-clients",
        type=int,
        default=10,
        help="Total number of clients (default: 10)"
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
        client_configs = ["C5"]
        num_rounds = 10
    elif args.medium:
        strategies = ["FedAvg", "FedAvgM"]
        distributions = ["homo", "Dir(0.5)"]
        client_configs = ["C3", "C5"]
        num_rounds = 100
    elif args.all:
        strategies = STRATEGIES
        distributions = DISTRIBUTIONS
        client_configs = list(NUM_CLIENTS_CONFIGS.keys())
        num_rounds = args.num_rounds
    else:
        strategies = args.strategies or ["FedAvg"]
        distributions = args.distributions or ["homo"]
        client_configs = args.client_configs or ["C5"]
        num_rounds = args.num_rounds

    # Calculate total experiments
    total_experiments = len(strategies) * len(distributions) * len(client_configs)

    print(f"\n{'='*80}")
    print(f"EXPERIMENT BATCH CONFIGURATION")
    print(f"{'='*80}")
    print(f"Strategies: {strategies}")
    print(f"Distributions: {distributions}")
    print(f"Client configs: {client_configs}")
    print(f"Rounds per experiment: {num_rounds}")
    print(f"Total experiments: {total_experiments}")
    print(f"{'='*80}\n")

    # Confirm before running
    if not args.yes:
        response = input(f"Run {total_experiments} experiments? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            print("Cancelled.")
            return

    # Run all experiments
    results = []
    for idx, (strategy, dist, client_config) in enumerate(
        product(strategies, distributions, client_configs), 1
    ):
        print(f"\n[{idx}/{total_experiments}] ", end="")

        success = run_experiment(
            strategy=strategy,
            distribution=dist,
            num_clients_config=client_config,
            num_rounds=num_rounds,
            total_clients=args.total_clients,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            local_epochs=args.local_epochs,
            use_gpu=args.gpu,
        )

        results.append({
            "strategy": strategy,
            "distribution": dist,
            "client_config": client_config,
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
                print(f"  - {r['strategy']} / {r['distribution']} / {r['client_config']}")

    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
