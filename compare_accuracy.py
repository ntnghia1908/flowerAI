"""Compare accuracy between old (duplicated) and new (fixed) data partitions."""

import pandas as pd
import os
import sys
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Algorithms to compare
ALGORITHMS = ['FedAvg', 'FedAvgM', 'FedProx']
DISTRIBUTIONS = ['C2', 'C3', 'C4', 'C5']

def get_best_accuracy(csv_file):
    """Extract best accuracy from global CSV file."""
    try:
        df = pd.read_csv(csv_file)
        if 'accuracy' in df.columns:
            return df['accuracy'].max()
        return None
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return None

def get_final_accuracy(csv_file):
    """Extract final (last round) accuracy from global CSV file."""
    try:
        df = pd.read_csv(csv_file)
        if 'accuracy' in df.columns:
            return df['accuracy'].iloc[-1]
        return None
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return None

def find_csv_files(results_dir, algorithm, distribution, is_old=False):
    """Find global CSV file for algorithm and distribution."""
    if is_old:
        search_dir = os.path.join(results_dir, algorithm, 'old_C_duplication_data')
    else:
        search_dir = os.path.join(results_dir, algorithm)

    if not os.path.exists(search_dir):
        return None

    # Find global CSV file matching pattern
    pattern = f"{algorithm}_{distribution}_npy_global_*.csv"

    for file in os.listdir(search_dir):
        if file.startswith(f"{algorithm}_{distribution}_npy_global_") and file.endswith('.csv'):
            return os.path.join(search_dir, file)

    return None

def main():
    results_dir = r'C:\Users\DESKSTOP_003\Desktop\flowerAI\results'

    print("=" * 100)
    print("ACCURACY COMPARISON: OLD DATA (Duplicated) vs NEW DATA (Fixed)")
    print("=" * 100)
    print()

    comparison_data = []

    for algorithm in ALGORITHMS:
        print(f"\n{'=' * 100}")
        print(f"Algorithm: {algorithm}")
        print(f"{'=' * 100}\n")

        for dist in DISTRIBUTIONS:
            print(f"  Distribution: {dist}")
            print(f"  {'-' * 90}")

            # Find old and new CSV files
            old_csv = find_csv_files(results_dir, algorithm, dist, is_old=True)
            new_csv = find_csv_files(results_dir, algorithm, dist, is_old=False)

            if old_csv:
                old_best = get_best_accuracy(old_csv)
                old_final = get_final_accuracy(old_csv)
                print(f"    OLD (Duplicated):  Best: {old_best:.2f}%  |  Final: {old_final:.2f}%")
            else:
                old_best = old_final = None
                print(f"    OLD (Duplicated):  [NOT FOUND]")

            if new_csv:
                new_best = get_best_accuracy(new_csv)
                new_final = get_final_accuracy(new_csv)
                print(f"    NEW (Fixed):       Best: {new_best:.2f}%  |  Final: {new_final:.2f}%")
            else:
                new_best = new_final = None
                print(f"    NEW (Fixed):       [NOT FOUND]")

            # Calculate differences
            if old_best and new_best:
                diff_best = new_best - old_best
                diff_final = new_final - old_final if (old_final and new_final) else None

                sign_best = "[+]" if diff_best > 0 else "[-]" if diff_best < 0 else "[=]"
                sign_final = "[+]" if diff_final and diff_final > 0 else "[-]" if diff_final and diff_final < 0 else "[=]"

                if diff_final:
                    print(f"    DIFFERENCE:        Best: {sign_best} {diff_best:+.2f}%  |  Final: {sign_final} {diff_final:+.2f}%")
                else:
                    print(f"    DIFFERENCE:        Best: {sign_best} {diff_best:+.2f}%")

                comparison_data.append({
                    'Algorithm': algorithm,
                    'Distribution': dist,
                    'Old_Best': old_best,
                    'New_Best': new_best,
                    'Diff_Best': diff_best,
                    'Old_Final': old_final,
                    'New_Final': new_final,
                    'Diff_Final': diff_final
                })

            print()

    # Generate summary table
    if comparison_data:
        print(f"\n{'=' * 100}")
        print("SUMMARY TABLE - BEST ACCURACY")
        print(f"{'=' * 100}\n")

        df = pd.DataFrame(comparison_data)

        # Pivot table for best accuracy
        print("Best Accuracy Comparison:")
        print("-" * 100)
        pivot_old = df.pivot(index='Algorithm', columns='Distribution', values='Old_Best')
        pivot_new = df.pivot(index='Algorithm', columns='Distribution', values='New_Best')
        pivot_diff = df.pivot(index='Algorithm', columns='Distribution', values='Diff_Best')

        print("\nOLD DATA (with duplication bug):")
        print(pivot_old.to_string())

        print("\n\nNEW DATA (fixed):")
        print(pivot_new.to_string())

        print("\n\nDIFFERENCE (New - Old):")
        print(pivot_diff.to_string())

        # Final accuracy comparison
        print(f"\n\n{'=' * 100}")
        print("FINAL ROUND ACCURACY")
        print(f"{'=' * 100}\n")

        pivot_old_final = df.pivot(index='Algorithm', columns='Distribution', values='Old_Final')
        pivot_new_final = df.pivot(index='Algorithm', columns='Distribution', values='New_Final')
        pivot_diff_final = df.pivot(index='Algorithm', columns='Distribution', values='Diff_Final')

        print("OLD DATA (with duplication bug):")
        print(pivot_old_final.to_string())

        print("\n\nNEW DATA (fixed):")
        print(pivot_new_final.to_string())

        print("\n\nDIFFERENCE (New - Old):")
        print(pivot_diff_final.to_string())

        # Calculate average improvements
        print(f"\n\n{'=' * 100}")
        print("AVERAGE IMPROVEMENTS")
        print(f"{'=' * 100}\n")

        avg_diff_best = df.groupby('Algorithm')['Diff_Best'].mean()
        avg_diff_final = df.groupby('Algorithm')['Diff_Final'].mean()

        print("Average Best Accuracy Improvement:")
        for alg in avg_diff_best.index:
            sign = "[+]" if avg_diff_best[alg] > 0 else "[-]" if avg_diff_best[alg] < 0 else "[=]"
            print(f"  {alg}: {sign} {avg_diff_best[alg]:+.2f}%")

        print("\nAverage Final Accuracy Improvement:")
        for alg in avg_diff_final.index:
            sign = "[+]" if avg_diff_final[alg] > 0 else "[-]" if avg_diff_final[alg] < 0 else "[=]"
            print(f"  {alg}: {sign} {avg_diff_final[alg]:+.2f}%")

        # Save to CSV
        output_file = os.path.join(results_dir, 'accuracy_comparison.csv')
        df.to_csv(output_file, index=False)
        print(f"\n\n[OK] Comparison data saved to: {output_file}")

if __name__ == '__main__':
    main()
