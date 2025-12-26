"""Quick test script for experiment framework.

This script runs a quick 10-round experiment to verify everything works.
"""

import subprocess
import sys


def test_basic_experiment():
    """Test basic experiment with FedAvg, homo, 10 rounds."""
    print("\n" + "="*80)
    print("TESTING EXPERIMENT FRAMEWORK")
    print("="*80)
    print("\nRunning quick test: FedAvg, homo, C5, 10 rounds")
    print("This should take ~2-3 minutes...\n")

    # First, backup original pyproject.toml
    print("1. Backing up pyproject.toml...")
    try:
        with open("pyproject.toml", "r") as f:
            original = f.read()
        with open("pyproject.toml.backup", "w") as f:
            f.write(original)
    except Exception as e:
        print(f"Warning: Could not backup pyproject.toml: {e}")

    # Copy experimental config
    print("2. Using experimental configuration...")
    try:
        with open("pyproject_experiment.toml", "r") as f:
            experiment_config = f.read()
        with open("pyproject.toml", "w") as f:
            f.write(experiment_config)
    except Exception as e:
        print(f"Error: Could not load experimental config: {e}")
        return False

    # Run experiment
    print("3. Running experiment...")
    try:
        cmd = [
            "flwr", "run", ".",
            "local-simulation",
            "--run-config",
            'num-server-rounds=10 strategy="FedAvg" distribution="homo" experiment-name="test_FedAvg_homo_C5" min-train-nodes=5 min-evaluate-nodes=5',
        ]

        result = subprocess.run(cmd, check=True)

        print("\n" + "="*80)
        print("[PASS] TEST PASSED!")
        print("="*80)
        print("\nCheck the results/ directory for CSV files:")
        print("  - test_FedAvg_homo_C5_global_*.csv")
        print("  - test_FedAvg_homo_C5_client_*.csv")
        print("  - test_FedAvg_homo_C5_weight_*.csv")
        print("\n")
        return True

    except subprocess.CalledProcessError as e:
        print("\n" + "="*80)
        print("[FAIL] TEST FAILED!")
        print("="*80)
        print(f"Error: {e}")
        return False

    finally:
        # Restore original config
        print("\n4. Restoring original configuration...")
        try:
            with open("pyproject.toml.backup", "r") as f:
                original = f.read()
            with open("pyproject.toml", "w") as f:
                f.write(original)
            print("Configuration restored.\n")
        except Exception as e:
            print(f"Warning: Could not restore original config: {e}\n")


def main():
    """Main test runner."""
    success = test_basic_experiment()

    if success:
        print("Next steps:")
        print("1. Check results/ directory for CSV files")
        print("2. Try running experiments with run_experiments.py")
        print("3. Read EXPERIMENTS_README.md for full documentation")
        return 0
    else:
        print("Please check the error messages above and:")
        print("1. Make sure all dependencies are installed: pip install -e .")
        print("2. Check that pyproject_experiment.toml exists")
        print("3. Verify Flower is properly installed: flwr --version")
        return 1


if __name__ == "__main__":
    sys.exit(main())
