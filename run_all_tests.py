"""Script to run all test cases sequentially."""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

# All test cases in order
TEST_CASES = [
    ('homo', 'configs/test_homo.toml', 'Homogeneous (IID)'),
    ('C1', 'configs/test_C1.toml', 'Label Skew C1 (1 class/client)'),
    ('C2', 'configs/test_C2.toml', 'Label Skew C2 (2 classes/client)'),
    ('C3', 'configs/test_C3.toml', 'Label Skew C3 (3 classes/client)'),
    ('C4', 'configs/test_C4.toml', 'Label Skew C4 (4 classes/client)'),
    ('C5', 'configs/test_C5.toml', 'Label Skew C5 (5 classes/client)'),
    ('Dir0.1', 'configs/test_Dir0p1.toml', 'Dirichlet(0.1) - Very non-IID'),
    ('Dir0.5', 'configs/test_Dir0p5.toml', 'Dirichlet(0.5) - Moderate non-IID'),
    ('Dir1.0', 'configs/test_Dir1p0.toml', 'Dirichlet(1.0) - Mild non-IID'),
    ('Dir10.0', 'configs/test_Dir10p0.toml', 'Dirichlet(10.0) - Nearly IID'),
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
        print(f"❌ Error: Config file not found: {config_file}")
        return False

    # Run the experiment
    cmd = ['conda', 'run', '-n', 'flwr', 'flwr', 'run', '.', '--run-config', config_file]

    try:
        result = subprocess.run(cmd, check=True)
        print()
        print(f"✅ Test '{test_name}' completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print()
        print(f"❌ Test '{test_name}' failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print()
        print("⚠️  Test interrupted by user")
        return False


def main():
    """Run all test cases."""
    start_time = datetime.now()

    print("="*70)
    print("Running All Test Cases (10 rounds each)")
    print(f"Total tests: {len(TEST_CASES)}")
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
            print(f"⚠️  Test '{name}' failed. Continue with next test? (y/n)")
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
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {name:15s} - {status}")
    print("="*70)
    print()
    print(f"All results saved in: results/")
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
        print("⚠️  Interrupted by user")
        print("="*70)
        sys.exit(1)
