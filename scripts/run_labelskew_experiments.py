"""Script to run label-skew experiments with C=1,2,3,4,5 classes per client.

This script automates running federated learning experiments where each client
has access to only k classes (extreme label skew), with k ranging from 1 to 5.

Usage:
    python scripts/run_labelskew_experiments.py --quick      # Quick test with 10 rounds
    python scripts/run_labelskew_experiments.py --medium     # Medium test with 100 rounds
    python scripts/run_labelskew_experiments.py --full       # Full experiment with 500 rounds
    python scripts/run_labelskew_experiments.py --classes 1 2 3  # Run specific C values
"""

import subprocess
import argparse
import time
from datetime import datetime


def run_experiment(strategy, distribution, num_rounds, experiment_name):
    """Run a single federated learning experiment.

    Args:
        strategy: FL strategy name (e.g., "FedAvg")
        distribution: Data distribution type (e.g., "C(2)")
        num_rounds: Number of FL rounds
        experiment_name: Name for the experiment

    Returns:
        bool: True if experiment succeeded, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"Starting experiment: {experiment_name}")
    print(f"Strategy: {strategy}, Distribution: {distribution}, Rounds: {num_rounds}")
    print(f"{'='*80}\n")

    start_time = time.time()

    # Construct flwr run command
    # All run-config parameters must be in a single string
    # Quote string values to ensure correct parsing
    run_config = (
        f'strategy="{strategy}" '
        f'distribution="{distribution}" '
        f'num-server-rounds={num_rounds} '
        f'experiment-name="{experiment_name}" '
        f'num-clients=6 '
        f'min-train-nodes=6 '
        f'min-evaluate-nodes=6 '
        f'fraction-train=1.0 '
        f'fraction-evaluate=1.0'
    )

    cmd = [
        "flwr", "run", ".",
        "local-simulation",
        "--run-config",
        run_config,
    ]

    try:
        # Run the experiment
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)

        elapsed_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"[OK] Experiment completed: {experiment_name}")
        print(f"  Time elapsed: {elapsed_time/60:.2f} minutes")
        print(f"{'='*80}\n")

        return True

    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"[FAILED] Experiment failed: {experiment_name}")
        print(f"  Error: {e}")
        print(f"  Time elapsed: {elapsed_time/60:.2f} minutes")
        print(f"{'='*80}\n")

        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run label-skew FL experiments with C=1,2,3,4,5"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test: C=2 only, 10 rounds"
    )
    parser.add_argument(
        "--medium",
        action="store_true",
        help="Medium test: C=1,2,3, 100 rounds"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full experiment: C=1,2,3,4,5, 500 rounds"
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs='+',
        help="Specific C values to run (e.g., --classes 1 2 3)"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=500,
        help="Number of FL rounds (default: 500)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="FedAvg",
        help="FL strategy to use (default: FedAvg)"
    )

    args = parser.parse_args()

    # Determine which C values to run
    if args.quick:
        classes_list = [2]
        num_rounds = 10
        print("Quick mode: Testing C=2 with 10 rounds")
    elif args.medium:
        classes_list = [1, 2, 3]
        num_rounds = 100
        print("Medium mode: Testing C=1,2,3 with 100 rounds")
    elif args.full:
        classes_list = [1, 2, 3, 4, 5]
        num_rounds = 500
        print("Full mode: Testing C=1,2,3,4,5 with 500 rounds")
    elif args.classes:
        classes_list = args.classes
        num_rounds = args.rounds
        print(f"Custom mode: Testing C={classes_list} with {num_rounds} rounds")
    else:
        print("No mode specified. Use --quick, --medium, --full, or --classes")
        parser.print_help()
        return

    strategy = args.strategy

    # Track results
    results = []
    total_start = time.time()

    # Run experiments for each C value
    for k in classes_list:
        distribution = f"C{k}"  # Use Ck format (simple, no special chars)
        experiment_name = f"{strategy}_C{k}"

        success = run_experiment(
            strategy=strategy,
            distribution=distribution,
            num_rounds=num_rounds,
            experiment_name=experiment_name
        )

        results.append({
            'experiment': experiment_name,
            'strategy': strategy,
            'distribution': distribution,
            'classes_per_client': k,
            'rounds': num_rounds,
            'success': success
        })

    total_elapsed = time.time() - total_start

    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r['success'])}")
    print(f"Failed: {sum(1 for r in results if not r['success'])}")
    print(f"Total time: {total_elapsed/60:.2f} minutes")
    print("\nResults:")
    for r in results:
        status = "[OK]" if r['success'] else "[FAIL]"
        print(f"  {status} {r['experiment']}: {r['distribution']} ({r['rounds']} rounds)")
    print("="*80)

    # Print instructions
    print("\nNext steps:")
    print("1. Check the 'results/' directory for CSV files")
    print("2. Check the models directory for saved models")
    print("3. Run analysis: python scripts/analyze_results.py --pattern 'FedAvg_C*'")


if __name__ == "__main__":
    main()
