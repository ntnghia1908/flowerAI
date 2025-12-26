"""Script to run a single test case with 10 rounds."""

import subprocess
import sys
import argparse
from pathlib import Path

# Available test cases
TEST_CASES = {
    'homo': {
        'config': 'configs/test_homo.toml',
        'description': 'Homogeneous (IID) - Equal distribution'
    },
    'C1': {
        'config': 'configs/test_C1.toml',
        'description': 'Label Skew C1 - 1 class per client (extreme non-IID)'
    },
    'C2': {
        'config': 'configs/test_C2.toml',
        'description': 'Label Skew C2 - 2 classes per client (severe non-IID)'
    },
    'C3': {
        'config': 'configs/test_C3.toml',
        'description': 'Label Skew C3 - 3 classes per client (moderate non-IID)'
    },
    'C4': {
        'config': 'configs/test_C4.toml',
        'description': 'Label Skew C4 - 4 classes per client (mild non-IID)'
    },
    'C5': {
        'config': 'configs/test_C5.toml',
        'description': 'Label Skew C5 - 5 classes per client (light non-IID)'
    },
    'Dir0.1': {
        'config': 'configs/test_Dir0p1.toml',
        'description': 'Dirichlet(0.1) - Very non-IID'
    },
    'Dir0.5': {
        'config': 'configs/test_Dir0p5.toml',
        'description': 'Dirichlet(0.5) - Moderate non-IID'
    },
    'Dir1.0': {
        'config': 'configs/test_Dir1p0.toml',
        'description': 'Dirichlet(1.0) - Mild non-IID'
    },
    'Dir10.0': {
        'config': 'configs/test_Dir10p0.toml',
        'description': 'Dirichlet(10.0) - Nearly IID'
    },
}


def run_test(test_name):
    """Run a single test case."""
    if test_name not in TEST_CASES:
        print(f"Error: Unknown test case '{test_name}'")
        print(f"\nAvailable test cases:")
        for name, info in TEST_CASES.items():
            print(f"  {name:10s} - {info['description']}")
        return False

    test_info = TEST_CASES[test_name]
    config_file = test_info['config']

    print("="*70)
    print(f"Running Test: {test_name}")
    print(f"Description: {test_info['description']}")
    print(f"Config: {config_file}")
    print("="*70)
    print()

    # Check if config file exists
    if not Path(config_file).exists():
        print(f"Error: Config file not found: {config_file}")
        return False

    # Run the experiment
    cmd = ['conda', 'run', '-n', 'flwr', 'flwr', 'run', '.', '--run-config', config_file]

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print()
        print("="*70)
        print(f"✅ Test '{test_name}' completed successfully!")
        print("="*70)
        return True
    except subprocess.CalledProcessError as e:
        print()
        print("="*70)
        print(f"❌ Test '{test_name}' failed with exit code {e.returncode}")
        print("="*70)
        return False
    except KeyboardInterrupt:
        print()
        print("="*70)
        print("⚠️  Test interrupted by user")
        print("="*70)
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Run a single federated learning test case',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available test cases:
  homo      - Homogeneous (IID) - Equal distribution
  C1        - Label Skew C1 - 1 class per client (extreme non-IID)
  C2        - Label Skew C2 - 2 classes per client (severe non-IID)
  C3        - Label Skew C3 - 3 classes per client (moderate non-IID)
  C4        - Label Skew C4 - 4 classes per client (mild non-IID)
  C5        - Label Skew C5 - 5 classes per client (light non-IID)
  Dir0.1    - Dirichlet(0.1) - Very non-IID
  Dir0.5    - Dirichlet(0.5) - Moderate non-IID
  Dir1.0    - Dirichlet(1.0) - Mild non-IID
  Dir10.0   - Dirichlet(10.0) - Nearly IID

Examples:
  python run_single_test.py homo      # Run homogeneous test
  python run_single_test.py C2        # Run label-skew C2 test
  python run_single_test.py C4        # Run label-skew C4 test
  python run_single_test.py Dir0.5    # Run Dirichlet(0.5) test
        """
    )
    parser.add_argument('test_name', help='Name of the test case to run')

    args = parser.parse_args()

    success = run_test(args.test_name)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
