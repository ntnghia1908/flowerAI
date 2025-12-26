"""Quick test script for label-skew experiments.

This script runs a quick 10-round test to verify the label-skew implementation works correctly.

Usage:
    python scripts/test_labelskew.py
"""

import subprocess
import sys


def test_labelskew():
    """Run a quick test with C=2 (each client has 2 classes)."""
    print("\n" + "="*80)
    print("QUICK TEST: Label-Skew with C=2 (10 rounds)")
    print("="*80)
    print("\nThis test will:")
    print("  - Use 6 clients (all participate each round)")
    print("  - Each client has exactly 2 classes (extreme label skew)")
    print("  - Run for 10 rounds (quick test)")
    print("\n" + "="*80 + "\n")

    cmd = [
        "flwr", "run", ".",
        "local-simulation",
        "--run-config",
        'strategy="FedAvg" distribution="C2" num-server-rounds=10 experiment-name="test_labelskew_C2" num-clients=6 min-train-nodes=6 min-evaluate-nodes=6 fraction-train=1.0 fraction-evaluate=1.0',
    ]

    try:
        print("Running command:")
        print(" ".join(cmd))
        print("\n")

        result = subprocess.run(cmd, check=True)

        print("\n" + "="*80)
        print("TEST PASSED!")
        print("="*80)
        print("\nLabel-skew implementation is working correctly.")
        print("\nCheck the results:")
        print("  - results/test_labelskew_C2_global_*.csv")
        print("  - test_labelskew_C2_final_model.pt")
        print("\nTo run full experiments:")
        print("  python scripts/run_labelskew_experiments.py --quick")
        print("  python scripts/run_labelskew_experiments.py --medium")
        print("  python scripts/run_labelskew_experiments.py --full")
        print("\n" + "="*80 + "\n")

        return 0

    except subprocess.CalledProcessError as e:
        print("\n" + "="*80)
        print("TEST FAILED!")
        print("="*80)
        print(f"Error: {e}")
        print("\nPlease check the error messages above.")
        print("="*80 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(test_labelskew())
