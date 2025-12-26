"""Script to analyze and visualize test results."""

import pandas as pd
import matplotlib.pyplot as plt
import glob
from pathlib import Path


def find_latest_results(test_name, metric_type='global'):
    """Find the latest result file for a test."""
    pattern = f"results/{test_name}_{metric_type}_*.csv"
    files = glob.glob(pattern)
    if not files:
        return None
    # Return the latest file
    return max(files, key=lambda x: Path(x).stat().st_mtime)


def load_test_results(test_names):
    """Load global results for multiple tests."""
    results = {}
    for name in test_names:
        file = find_latest_results(name, 'global')
        if file:
            results[name] = pd.read_csv(file)
        else:
            print(f"Warning: No results found for {name}")
    return results


def plot_comparison(results, metric='accuracy', title=None, save_path=None):
    """Plot comparison of a metric across tests."""
    plt.figure(figsize=(12, 6))

    for name, df in results.items():
        plt.plot(df['round'], df[metric], marker='o', label=name, linewidth=2)

    plt.xlabel('Round', fontsize=12)
    plt.ylabel(metric.capitalize(), fontsize=12)
    plt.title(title or f'{metric.capitalize()} Comparison Across Distributions', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.show()


def print_summary(results):
    """Print summary statistics for all tests."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"{'Test Name':<15} {'Final Acc':<12} {'Best Acc':<12} {'Final Loss':<12} {'Final F1':<12}")
    print("-"*80)

    for name, df in sorted(results.items()):
        final_acc = df['accuracy'].iloc[-1]
        best_acc = df['accuracy'].max()
        final_loss = df['loss'].iloc[-1]
        final_f1 = df['f1'].iloc[-1]

        print(f"{name:<15} {final_acc:>11.4f} {best_acc:>11.4f} {final_loss:>11.4f} {final_f1:>11.4f}")

    print("="*80)


def analyze_convergence(results):
    """Analyze convergence speed."""
    print("\n" + "="*80)
    print("CONVERGENCE ANALYSIS")
    print("="*80)
    print(f"{'Test Name':<15} {'Rounds to 50%':<15} {'Improvement':<15} {'Stable?':<10}")
    print("-"*80)

    for name, df in sorted(results.items()):
        # Find round where accuracy first exceeds 50%
        acc_50 = df[df['accuracy'] >= 0.5]
        rounds_to_50 = acc_50.index[0] if len(acc_50) > 0 else "N/A"

        # Calculate improvement (last - first)
        improvement = df['accuracy'].iloc[-1] - df['accuracy'].iloc[0]

        # Check stability (std of last 3 rounds)
        last_3 = df['accuracy'].iloc[-3:]
        stable = "Yes" if last_3.std() < 0.05 else "No"

        print(f"{name:<15} {str(rounds_to_50):<15} {improvement:>14.4f} {stable:<10}")

    print("="*80)


def analyze_client_variance(test_name):
    """Analyze variance across clients."""
    file = find_latest_results(test_name, 'client')
    if not file:
        print(f"No client results found for {test_name}")
        return

    df = pd.read_csv(file)

    print(f"\n{'='*80}")
    print(f"CLIENT ANALYSIS: {test_name}")
    print("="*80)

    # Group by round and calculate variance
    for round_num in df['round'].unique():
        round_df = df[df['round'] == round_num]
        mean_acc = round_df['accuracy'].mean()
        std_acc = round_df['accuracy'].std()
        min_acc = round_df['accuracy'].min()
        max_acc = round_df['accuracy'].max()

        print(f"Round {round_num}: Mean={mean_acc:.4f}, Std={std_acc:.4f}, "
              f"Min={min_acc:.4f}, Max={max_acc:.4f}")

    print("="*80)


def main():
    """Main analysis function."""
    print("="*80)
    print("FEDERATED LEARNING TEST RESULTS ANALYSIS")
    print("="*80)

    # Define test names
    all_tests = ['test_homo', 'test_C1', 'test_C2', 'test_C3', 'test_C4', 'test_C5',
                 'test_Dir0p1', 'test_Dir0p5', 'test_Dir1p0', 'test_Dir10p0']

    # Load results
    print("\nLoading results...")
    results = load_test_results(all_tests)

    if not results:
        print("No results found! Please run tests first.")
        return

    print(f"Loaded {len(results)} test results")

    # Print summary
    print_summary(results)

    # Analyze convergence
    analyze_convergence(results)

    # Plot accuracy comparison
    print("\nGenerating plots...")
    plot_comparison(results, metric='accuracy',
                   title='Global Accuracy Comparison',
                   save_path='results/accuracy_comparison.png')

    # Plot loss comparison
    plot_comparison(results, metric='loss',
                   title='Global Loss Comparison',
                   save_path='results/loss_comparison.png')

    # Plot F1 comparison
    plot_comparison(results, metric='f1',
                   title='Global F1 Score Comparison',
                   save_path='results/f1_comparison.png')

    # Analyze client variance for selected tests
    print("\n")
    for test in ['test_homo', 'test_C2', 'test_Dir0p5']:
        if test in [k for k in results.keys()]:
            analyze_client_variance(test)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("Charts saved to results/ directory")
    print("="*80)


if __name__ == '__main__':
    main()
