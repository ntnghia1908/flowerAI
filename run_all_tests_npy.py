"""Script to run all test cases with .npy data sequentially."""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

# Set Ray environment variables for Windows compatibility
os.environ['RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO'] = '0'

# All test cases in order (NPY versions)
TEST_CASES = [
    ('homo', 'configs/test_homo_npy.toml', 'Homogeneous (IID) - NPY'),
    ('C2', 'configs/test_C2_npy.toml', 'Label Skew C2 (2 classes/client) - NPY'),
    ('C3', 'configs/test_C3_npy.toml', 'Label Skew C3 (3 classes/client) - NPY'),
    ('C4', 'configs/test_C4_npy.toml', 'Label Skew C4 (4 classes/client) - NPY'),
    ('C5', 'configs/test_C5_npy.toml', 'Label Skew C5 (5 classes/client) - NPY'),
    ('Dir0.1', 'configs/test_Dir0.1_npy.toml', 'Dirichlet(0.1) - Very non-IID - NPY'),
    ('Dir0.5', 'configs/test_Dir0.5_npy.toml', 'Dirichlet(0.5) - Moderate non-IID - NPY'),
    ('Dir1.0', 'configs/test_Dir1.0_npy.toml', 'Dirichlet(1.0) - Mild non-IID - NPY'),
    ('Dir10.0', 'configs/test_Dir10.0_npy.toml', 'Dirichlet(10.0) - Nearly IID - NPY'),
]


def run_test(test_name, config_file, description):
    """Run a single test case."""
    print()
    print("="*70)
    print(f"Test: {test_name}")
    print(f"Description: {description}")
    print(f"Config: {config_file}")
    print("="*70)
    print()

    # Check if config exists
    if not Path(config_file).exists():
        print(f"[FAIL] Error: Config file not found: {config_file}")
        return False

    # Run the experiment
    cmd = ['flwr', 'run', '.', '--run-config', config_file]

    try:
        result = subprocess.run(cmd, check=True)
        print()
        print(f"[OK] Test '{test_name}' completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print()
        print(f"[FAIL] Test '{test_name}' failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print()
        print("[WARNING]  Test interrupted by user")
        return False


def main():
    """Run all test cases with .npy data."""
    start_time = datetime.now()

    print("="*70)
    print("Running All Test Cases with NPY Data (10 rounds each)")
    print(f"Total tests: {len(TEST_CASES)}")
    print(f"Data source: Pre-partitioned .npy files")
    print(f"Expected speedup: 5-10x faster than HuggingFace mode")
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    results = []
    for idx, (name, config, desc) in enumerate(TEST_CASES, 1):
        print()
        print(f"[{idx}/{len(TEST_CASES)}] Starting test: {name}")

        success = run_test(name, config, desc)
        results.append((name, success))

        if not success:
            print()
            print("="*70)
            print(f"[WARNING]  Test '{name}' failed. Continue with next test? (y/n)")
            print("="*70)
            response = input().strip().lower()
            if response != 'y':
                print("Aborting remaining tests.")
                break

    # Print summary
    end_time = datetime.now()
    duration = end_time - start_time

    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total tests run: {len(results)}")
    print(f"Successful: {sum(1 for _, success in results if success)}")
    print(f"Failed: {sum(1 for _, success in results if not success)}")
    print(f"Duration: {duration}")
    print()
    print("Results:")
    for name, success in results:
        status = "[OK] PASS" if success else "[FAIL] FAIL"
        print(f"  {name:15s} - {status}")
    print("="*70)
    print()
    print(f"All results saved in: results/")
    print(f"Data source: Pre-partitioned .npy files (5-10x faster!)")
    print()

    # Check if all passed
    all_passed = all(success for _, success in results)
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("="*70)
        print("[WARNING]  Interrupted by user")
        print("="*70)
        sys.exit(1)
