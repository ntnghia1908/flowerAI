"""Verify all experiment results and metrics logging."""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

STRATEGIES = ["FedAvg", "FedAvgM", "FedProx", "FedAdam", "FedAdagrad", "FedYogi", "FedNova", "SCAFFOLD"]
DISTRIBUTIONS = ["homo", "C2", "C3", "C4", "C5", "Dir0.1", "Dir0.5", "Dir1.0", "Dir10.0"]


def verify_csv_file(csv_path, expected_type):
    """Verify a CSV file has correct structure and data."""
    if not Path(csv_path).exists():
        return False, f"File not found: {csv_path}"

    try:
        df = pd.read_csv(csv_path)

        if expected_type == "client":
            # Check client CSV structure
            required_cols = ['round', 'client_id', 'phase', 'loss', 'accuracy',
                           'precision', 'recall', 'f1', 'num_examples']
            if not all(col in df.columns for col in required_cols):
                return False, f"Missing required columns in {csv_path}"

            # Check for data (should have 6 clients × 20 rounds = 120 rows)
            if len(df) < 100:  # Allow some tolerance
                return False, f"Too few rows ({len(df)}) in {csv_path}"

            # Check non-zero metrics
            if df['accuracy'].max() == 0:
                return False, f"All accuracies are 0 in {csv_path}"

            # Check num_examples
            if df['num_examples'].max() == 0:
                return False, f"All num_examples are 0 in {csv_path}"

        elif expected_type == "global":
            # Check global CSV structure
            required_cols = ['round', 'loss', 'accuracy', 'precision', 'recall',
                           'f1', 'global_accuracy', 'weighted_accuracy']
            if not all(col in df.columns for col in required_cols):
                return False, f"Missing required columns in {csv_path}"

            # Check for data (should have 21 rows: round 0-20)
            if len(df) < 20:
                return False, f"Too few rows ({len(df)}) in {csv_path}"

            # Check non-zero global_accuracy (except round 0)
            if df[df['round'] > 0]['global_accuracy'].max() == 0:
                return False, f"All global_accuracy are 0 in {csv_path}"

            # Check non-zero weighted_accuracy (except round 0)
            if df[df['round'] > 0]['weighted_accuracy'].max() == 0:
                return False, f"All weighted_accuracy are 0 in {csv_path}"

        return True, "OK"

    except Exception as e:
        return False, f"Error reading {csv_path}: {str(e)}"


def get_latest_csv_files(strategy, distribution, csv_type):
    """Get the latest CSV file for a strategy-distribution pair."""
    # Updated to look in strategy subfolders
    pattern = f"results/{strategy}/{strategy}_{distribution}_npy_{csv_type}_*.csv"
    files = sorted(Path('.').glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
    return files[0] if files else None


def main():
    """Verify all experiment results."""
    print("="*80)
    print("VERIFICATION OF ALL EXPERIMENT RESULTS")
    print("="*80)
    print(f"Checking {len(STRATEGIES)} strategies × {len(DISTRIBUTIONS)} distributions = {len(STRATEGIES) * len(DISTRIBUTIONS)} experiments")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print()

    results = []
    total_experiments = 0
    passed_experiments = 0
    failed_experiments = 0

    for strategy in STRATEGIES:
        print(f"\n{'='*80}")
        print(f"Strategy: {strategy}")
        print(f"{'='*80}")

        for distribution in DISTRIBUTIONS:
            total_experiments += 1
            experiment_name = f"{strategy}_{distribution}"

            # Find latest CSV files
            client_csv = get_latest_csv_files(strategy, distribution, "client")
            global_csv = get_latest_csv_files(strategy, distribution, "global")

            if not client_csv:
                status = "FAIL"
                message = "Client CSV not found"
                failed_experiments += 1
            elif not global_csv:
                status = "FAIL"
                message = "Global CSV not found"
                failed_experiments += 1
            else:
                # Verify client CSV
                client_ok, client_msg = verify_csv_file(client_csv, "client")

                # Verify global CSV
                global_ok, global_msg = verify_csv_file(global_csv, "global")

                if client_ok and global_ok:
                    status = "PASS"
                    message = "All metrics OK"
                    passed_experiments += 1

                    # Get final and best metrics
                    df_global = pd.read_csv(global_csv)
                    final_round = df_global.iloc[-1]

                    # Find best round based on global_accuracy
                    best_idx = df_global[df_global['round'] > 0]['global_accuracy'].idxmax()
                    best_round = df_global.loc[best_idx]

                    message += f" | Best (R{int(best_round['round'])}): G_Acc={best_round['global_accuracy']:.4f}"
                    message += f" | Final (R{int(final_round['round'])}): G_Acc={final_round['global_accuracy']:.4f}"
                else:
                    status = "FAIL"
                    message = f"Client: {client_msg} | Global: {global_msg}"
                    failed_experiments += 1

            results.append((experiment_name, status, message))
            status_symbol = "✅" if status == "PASS" else "❌"
            print(f"  {status_symbol} {experiment_name:30s} - {message}")

    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    print(f"Total experiments: {total_experiments}")
    print(f"Passed: {passed_experiments} ({passed_experiments/total_experiments*100:.1f}%)")
    print(f"Failed: {failed_experiments} ({failed_experiments/total_experiments*100:.1f}%)")
    print()

    if failed_experiments > 0:
        print("Failed experiments:")
        for name, status, message in results:
            if status == "FAIL":
                print(f"  ❌ {name}: {message}")
    else:
        print("🎉 All experiments passed verification!")

    print()
    print("Detailed results:")
    print("-"*80)
    for name, status, message in results:
        print(f"  {name:30s} | {status:4s} | {message}")

    print("="*80)
    print(f"Verification completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == '__main__':
    main()
