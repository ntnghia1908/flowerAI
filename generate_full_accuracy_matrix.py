"""Generate comprehensive accuracy matrix for all algorithms and distributions."""

import pandas as pd
import os
import sys
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# All algorithms
ALGORITHMS = ['FedAvg', 'FedAvgM', 'FedProx', 'FedAdam', 'FedAdagrad', 'FedYogi']

# All distributions
DISTRIBUTIONS = ['homo', 'C2', 'C3', 'C4', 'C5', 'Dir0.1', 'Dir0.5', 'Dir1.0', 'Dir10.0']

def get_best_accuracy(csv_file):
    """Extract best accuracy from global CSV file."""
    try:
        df = pd.read_csv(csv_file)
        if 'accuracy' in df.columns:
            return df['accuracy'].max()
        return None
    except Exception as e:
        return None

def get_final_accuracy(csv_file):
    """Extract final (last round) accuracy from global CSV file."""
    try:
        df = pd.read_csv(csv_file)
        if 'accuracy' in df.columns:
            return df['accuracy'].iloc[-1]
        return None
    except Exception as e:
        return None

def get_best_round(csv_file):
    """Extract round number where best accuracy was achieved."""
    try:
        df = pd.read_csv(csv_file)
        if 'accuracy' in df.columns:
            return df['accuracy'].idxmax() + 1  # +1 because rounds start at 1
        return None
    except Exception as e:
        return None

def find_csv_files(results_dir, algorithm, distribution):
    """Find global CSV file for algorithm and distribution."""
    search_dir = os.path.join(results_dir, algorithm)

    if not os.path.exists(search_dir):
        return None

    # Find global CSV file matching pattern
    for file in os.listdir(search_dir):
        if file.startswith(f"{algorithm}_{distribution}_npy_global_") and file.endswith('.csv'):
            return os.path.join(search_dir, file)

    return None

def main():
    results_dir = r'C:\Users\DESKSTOP_003\Desktop\flowerAI\results'

    print("=" * 120)
    print("COMPREHENSIVE ACCURACY MATRIX - ALL ALGORITHMS & DISTRIBUTIONS")
    print("=" * 120)
    print()

    # Collect all data
    all_data = []

    for algorithm in ALGORITHMS:
        for dist in DISTRIBUTIONS:
            csv_file = find_csv_files(results_dir, algorithm, dist)

            if csv_file:
                best_acc = get_best_accuracy(csv_file)
                final_acc = get_final_accuracy(csv_file)
                best_round = get_best_round(csv_file)

                all_data.append({
                    'Algorithm': algorithm,
                    'Distribution': dist,
                    'Best_Accuracy': best_acc,
                    'Final_Accuracy': final_acc,
                    'Best_Round': best_round,
                    'Status': 'Complete'
                })
            else:
                all_data.append({
                    'Algorithm': algorithm,
                    'Distribution': dist,
                    'Best_Accuracy': None,
                    'Final_Accuracy': None,
                    'Best_Round': None,
                    'Status': 'Missing'
                })

    df = pd.DataFrame(all_data)

    # Save raw data
    output_file = os.path.join(results_dir, 'full_accuracy_matrix.csv')
    df.to_csv(output_file, index=False)
    print(f"[OK] Raw data saved to: {output_file}\n")

    # Generate matrices
    print("=" * 120)
    print("1. BEST ACCURACY MATRIX (Transposed)")
    print("=" * 120)
    print()

    pivot_best = df.pivot(index='Distribution', columns='Algorithm', values='Best_Accuracy')
    print(pivot_best.to_string())

    print("\n\n" + "=" * 120)
    print("2. FINAL ROUND ACCURACY MATRIX (Transposed)")
    print("=" * 120)
    print()

    pivot_final = df.pivot(index='Distribution', columns='Algorithm', values='Final_Accuracy')
    print(pivot_final.to_string())

    print("\n\n" + "=" * 120)
    print("3. BEST ROUND NUMBER MATRIX (Transposed)")
    print("=" * 120)
    print()

    pivot_round = df.pivot(index='Distribution', columns='Algorithm', values='Best_Round')
    print(pivot_round.to_string())

    print("\n\n" + "=" * 120)
    print("4. COMPLETION STATUS MATRIX")
    print("=" * 120)
    print()

    pivot_status = df.pivot(index='Distribution', columns='Algorithm', values='Status')
    print(pivot_status.to_string())

    # Statistics
    print("\n\n" + "=" * 120)
    print("5. SUMMARY STATISTICS")
    print("=" * 120)
    print()

    # By Algorithm
    print("Average Best Accuracy by Algorithm:")
    print("-" * 60)
    avg_by_alg = df.groupby('Algorithm')['Best_Accuracy'].agg(['mean', 'std', 'min', 'max'])
    print(avg_by_alg.to_string())

    print("\n\nAverage Final Accuracy by Algorithm:")
    print("-" * 60)
    avg_final_by_alg = df.groupby('Algorithm')['Final_Accuracy'].agg(['mean', 'std', 'min', 'max'])
    print(avg_final_by_alg.to_string())

    # By Distribution
    print("\n\nAverage Best Accuracy by Distribution:")
    print("-" * 60)
    avg_by_dist = df.groupby('Distribution')['Best_Accuracy'].agg(['mean', 'std', 'min', 'max'])
    print(avg_by_dist.to_string())

    print("\n\nAverage Final Accuracy by Distribution:")
    print("-" * 60)
    avg_final_by_dist = df.groupby('Distribution')['Final_Accuracy'].agg(['mean', 'std', 'min', 'max'])
    print(avg_final_by_dist.to_string())

    # Completion rate
    print("\n\n" + "=" * 120)
    print("6. EXPERIMENT COMPLETION RATE")
    print("=" * 120)
    print()

    total_experiments = len(ALGORITHMS) * len(DISTRIBUTIONS)
    completed = df[df['Status'] == 'Complete'].shape[0]
    completion_rate = (completed / total_experiments) * 100

    print(f"Total Experiments: {total_experiments}")
    print(f"Completed: {completed}")
    print(f"Missing: {total_experiments - completed}")
    print(f"Completion Rate: {completion_rate:.2f}%")

    print("\n\nCompletion by Algorithm:")
    print("-" * 60)
    completion_by_alg = df.groupby('Algorithm')['Status'].apply(lambda x: (x == 'Complete').sum())
    for alg in ALGORITHMS:
        count = completion_by_alg.get(alg, 0)
        rate = (count / len(DISTRIBUTIONS)) * 100
        print(f"  {alg:12s}: {count}/{len(DISTRIBUTIONS)} ({rate:.1f}%)")

    print("\n\nCompletion by Distribution:")
    print("-" * 60)
    completion_by_dist = df.groupby('Distribution')['Status'].apply(lambda x: (x == 'Complete').sum())
    for dist in DISTRIBUTIONS:
        count = completion_by_dist.get(dist, 0)
        rate = (count / len(ALGORITHMS)) * 100
        print(f"  {dist:12s}: {count}/{len(ALGORITHMS)} ({rate:.1f}%)")

    # Best performing combinations
    print("\n\n" + "=" * 120)
    print("7. TOP 10 BEST PERFORMING COMBINATIONS")
    print("=" * 120)
    print()

    top10 = df.nlargest(10, 'Best_Accuracy')[['Algorithm', 'Distribution', 'Best_Accuracy', 'Best_Round']]
    print(top10.to_string(index=False))

    print("\n\n" + "=" * 120)
    print("8. TOP 10 WORST PERFORMING COMBINATIONS")
    print("=" * 120)
    print()

    bottom10 = df.nsmallest(10, 'Best_Accuracy')[['Algorithm', 'Distribution', 'Best_Accuracy', 'Best_Round']]
    print(bottom10.to_string(index=False))

    # Algorithm rankings
    print("\n\n" + "=" * 120)
    print("9. ALGORITHM RANKINGS (by Average Best Accuracy)")
    print("=" * 120)
    print()

    rankings = df.groupby('Algorithm')['Best_Accuracy'].mean().sort_values(ascending=False)
    for rank, (alg, acc) in enumerate(rankings.items(), 1):
        print(f"  {rank}. {alg:12s}: {acc:.2f}%")

    print("\n\n[OK] Comprehensive accuracy matrix generated!")

if __name__ == '__main__':
    main()
