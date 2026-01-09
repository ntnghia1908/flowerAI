"""Run all strategy × distribution experiments."""

import subprocess
import os
import sys
from pathlib import Path
from datetime import datetime

# Set encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Set Ray environment variables for Windows compatibility
os.environ['RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO'] = '0'

STRATEGIES = [ "FedAvg", "FedAvgM", "FedProx", "FedAdam", "FedAdagrad", "FedYogi"]
# STRATEGIES = ["FedAvg"]
DISTRIBUTIONS = ["homo", "C2", "C3", "C4", "C5", "Dir0.1", "Dir0.5", "Dir1.0", "Dir10.0"]

def run_experiment(strategy, distribution):
    """Run a single experiment."""
    config_file = f"configs/{strategy}_{distribution}_npy.toml"

    if not Path(config_file).exists():
        print(f"❌ Config not found: {config_file}")
        return False

    print(f"\n{'='*70}")
    print(f"Running: {strategy} on {distribution}")
    print(f"Config: {config_file}")
    print(f"{'='*70}\n")

    cmd = ['flwr', 'run', '.', '--run-config', config_file]
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ {strategy}_{distribution} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {strategy}_{distribution} failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"⚠️ {strategy}_{distribution} interrupted by user")
        return False

def main():
    """Run all experiments sequentially."""
    start_time = datetime.now()
    results = []

    total = len(STRATEGIES) * len(DISTRIBUTIONS)
    print(f"{'='*70}")
    print(f"Running All Strategy × Distribution Experiments")
    print(f"{'='*70}")
    print(f"Total experiments: {total}")
    print(f"Strategies: {STRATEGIES}")
    print(f"Distributions: {DISTRIBUTIONS}")
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    for idx, strategy in enumerate(STRATEGIES, 1):
        print(f"\n[{idx}/{len(STRATEGIES)}] Testing strategy: {strategy}")
        for distribution in DISTRIBUTIONS:
            success = run_experiment(strategy, distribution)
            results.append((f"{strategy}_{distribution}", success))

            if not success:
                print(f"\n{'='*70}")
                print(f"⚠️  Experiment '{strategy}_{distribution}' failed.")
                print(f"Continue with next experiment? (y/n)")
                print(f"{'='*70}")
                response = input().strip().lower()
                if response != 'y':
                    print("Aborting remaining experiments.")
                    break

    # Print summary
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"Total run: {len(results)}")
    print(f"Passed: {sum(1 for _, s in results if s)}")
    print(f"Failed: {sum(1 for _, s in results if not s)}")
    print(f"Duration: {duration}")
    print("\nResults:")
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {name}")
    print("="*70)
    print(f"\nAll results saved in: results/")
    print("="*70)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n" + "="*70)
        print("⚠️ Interrupted by user")
        print("="*70)
