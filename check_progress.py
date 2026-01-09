"""Check experiment progress every 10 minutes."""

import time
import sys
from pathlib import Path
from datetime import datetime
import subprocess

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

STRATEGIES = ["FedAvg", "FedAvgM", "FedProx", "FedAdam", "FedAdagrad", "FedYogi"]
TOTAL_EXPERIMENTS = 54

def count_completed_experiments():
    """Count completed experiments by strategy."""
    results = {}
    total = 0

    for strategy in STRATEGIES:
        csv_files = list(Path(f"results/{strategy}").glob("*_global_*.csv"))
        count = len(csv_files)
        results[strategy] = count
        total += count

    return results, total

def check_if_running():
    """Check if experiments are still running."""
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            timeout=5
        )
        output = result.stdout.lower()
        return "flower-simulation" in output or "run_all_experiments" in output
    except:
        return False

def print_progress():
    """Print current progress."""
    results, total = count_completed_experiments()
    is_running = check_if_running()

    print("\n" + "="*70)
    print(f"EXPERIMENT PROGRESS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    for strategy in STRATEGIES:
        count = results[strategy]
        bar = "█" * count + "░" * (9 - count)
        status = "✓" if count == 9 else "→" if count > 0 else "○"
        print(f"{status} {strategy:12s}: [{bar}] {count}/9")

    print("-"*70)
    percentage = (total / TOTAL_EXPERIMENTS) * 100
    bar_total = "█" * int(percentage / 2) + "░" * (50 - int(percentage / 2))
    print(f"  TOTAL: [{bar_total}] {total}/{TOTAL_EXPERIMENTS} ({percentage:.1f}%)")

    print("-"*70)
    if is_running:
        print("  Status: RUNNING")
    else:
        print("  Status: STOPPED (Process not found)")

    print("="*70 + "\n")

    return total, is_running

def main():
    """Monitor progress every 10 minutes."""
    print("\nStarting progress monitor (checking every 10 minutes)")
    print("Press Ctrl+C to stop\n")

    check_count = 0

    try:
        while True:
            check_count += 1
            print(f"\n[Check #{check_count}]")

            total, is_running = print_progress()

            # If all completed, exit
            if total >= TOTAL_EXPERIMENTS:
                print("All experiments completed!")
                break

            # If stopped and not all completed, warn
            if not is_running and total < TOTAL_EXPERIMENTS:
                print("WARNING: Process stopped but experiments not complete!")
                print("   You may need to restart: python run_all_experiments.py")

            # Wait 10 minutes
            if total < TOTAL_EXPERIMENTS:
                print(f"\nWaiting 10 minutes before next check...")
                time.sleep(600)  # 10 minutes

    except KeyboardInterrupt:
        print("\n\nProgress monitoring stopped by user")
        print_progress()

if __name__ == "__main__":
    main()
