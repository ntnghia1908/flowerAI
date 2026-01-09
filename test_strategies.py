"""Test each strategy individually with 3 rounds to verify they work."""

import subprocess
import sys

# Ensure UTF-8 encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

STRATEGIES = ["FedAvg", "FedAvgM", "FedProx", "FedAdam", "FedAdagrad", "FedYogi"]

def test_strategy(strategy):
    """Test a single strategy with homo distribution."""
    config = f"configs/{strategy}_homo_npy.toml"

    print(f"\n{'='*70}")
    print(f"Testing: {strategy}")
    print(f"Config: {config}")
    print(f"{'='*70}\n")

    try:
        result = subprocess.run(
            ["flower-simulation", "--app", ".", "--num-supernodes", "6", "--run-config", config],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✅ {strategy} completed successfully\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {strategy} failed with error code {e.returncode}\n")
        return False

def main():
    """Test all strategies."""
    print(f"\n{'='*70}")
    print(f"TESTING ALL STRATEGIES (3 rounds each)")
    print(f"{'='*70}\n")

    results = {}
    for i, strategy in enumerate(STRATEGIES, 1):
        print(f"\n[{i}/{len(STRATEGIES)}] Testing strategy: {strategy}")
        results[strategy] = test_strategy(strategy)

    # Print summary
    print(f"\n{'='*70}")
    print(f"TEST SUMMARY")
    print(f"{'='*70}\n")

    passed = sum(1 for v in results.values() if v)
    failed = len(results) - passed

    for strategy, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {strategy}")

    print(f"\nTotal: {passed} passed, {failed} failed out of {len(STRATEGIES)} strategies\n")

    if failed > 0:
        sys.exit(1)

if __name__ == '__main__':
    main()
