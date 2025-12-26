"""Analyze experiment results from CSV files.

This script helps analyze and visualize the results from FL experiments.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob


def find_csv_files(results_dir: str = "results", pattern: str = "*"):
    """Find CSV files in results directory.

    Args:
        results_dir: Directory containing results
        pattern: Pattern to match (e.g., "FedAvg_homo*")

    Returns:
        Dictionary with global, client, and weight CSV paths
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory '{results_dir}' not found!")
        return None

    # Find all CSV files matching pattern
    global_files = sorted(results_path.glob(f"{pattern}_global_*.csv"))
    client_files = sorted(results_path.glob(f"{pattern}_client_*.csv"))
    weight_files = sorted(results_path.glob(f"{pattern}_weight_*.csv"))

    if not global_files:
        print(f"No CSV files found matching pattern '{pattern}'")
        return None

    print(f"Found {len(global_files)} experiment(s):")
    for gf in global_files:
        print(f"  - {gf.stem}")

    return {
        "global": global_files,
        "client": client_files,
        "weight": weight_files,
    }


def plot_global_metrics(csv_path: str, save_plot: bool = True):
    """Plot global metrics from a single experiment.

    Args:
        csv_path: Path to global metrics CSV file
        save_plot: Whether to save the plot as image
    """
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Global Metrics: {Path(csv_path).stem}", fontsize=16)

    # Plot each metric
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1']
    colors = ['red', 'blue', 'green', 'orange', 'purple']

    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        ax.plot(df['round'], df[metric], color=color, linewidth=2)
        ax.set_xlabel('Round')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(metric.capitalize())
        ax.grid(True, alpha=0.3)

    # Plot all metrics together in last subplot
    ax = axes[1, 2]
    for metric, color in zip(metrics[1:], colors[1:]):  # Skip loss
        ax.plot(df['round'], df[metric], label=metric.capitalize(), color=color, linewidth=2)
    ax.set_xlabel('Round')
    ax.set_ylabel('Score')
    ax.set_title('All Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plot:
        plot_path = Path(csv_path).with_suffix('.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")

    plt.show()


def plot_weight_metrics(csv_path: str, save_plot: bool = True):
    """Plot weight change metrics.

    Args:
        csv_path: Path to weight metrics CSV file
        save_plot: Whether to save the plot as image
    """
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Weight Metrics: {Path(csv_path).stem}", fontsize=16)

    metrics = ['weight_norm', 'weight_change', 'weight_relative_change']
    titles = ['Weight Norm', 'Weight Change (Absolute)', 'Weight Change (Relative)']

    for ax, metric, title in zip(axes, metrics, titles):
        ax.plot(df['round'], df[metric], linewidth=2, color='darkblue')
        ax.set_xlabel('Round')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plot:
        plot_path = Path(csv_path).with_suffix('.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")

    plt.show()


def compare_experiments(csv_paths: list, metric: str = 'accuracy'):
    """Compare multiple experiments on a single metric.

    Args:
        csv_paths: List of paths to global metrics CSV files
        metric: Metric to compare ('accuracy', 'loss', 'f1', etc.)
    """
    if not csv_paths:
        print("No CSV files to compare!")
        return

    plt.figure(figsize=(12, 6))

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        exp_name = Path(csv_path).stem.replace('_global', '').split('_20')[0]
        plt.plot(df['round'], df[metric], label=exp_name, linewidth=2)

    plt.xlabel('Round', fontsize=12)
    plt.ylabel(metric.capitalize(), fontsize=12)
    plt.title(f'Comparison: {metric.capitalize()}', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = f"results/comparison_{metric}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to: {save_path}")

    plt.show()


def print_summary(csv_path: str):
    """Print summary statistics for an experiment.

    Args:
        csv_path: Path to global metrics CSV file
    """
    df = pd.read_csv(csv_path)

    print("\n" + "="*80)
    print(f"SUMMARY: {Path(csv_path).stem}")
    print("="*80)

    print("\nFinal Round Metrics:")
    final_row = df.iloc[-1]
    print(f"  Round: {int(final_row['round'])}")
    print(f"  Loss: {final_row['loss']:.4f}")
    print(f"  Accuracy: {final_row['accuracy']:.4f}")
    print(f"  Precision: {final_row['precision']:.4f}")
    print(f"  Recall: {final_row['recall']:.4f}")
    print(f"  F1 Score: {final_row['f1']:.4f}")

    print("\nBest Metrics:")
    print(f"  Best Accuracy: {df['accuracy'].max():.4f} (Round {df.loc[df['accuracy'].idxmax(), 'round']:.0f})")
    print(f"  Best F1 Score: {df['f1'].max():.4f} (Round {df.loc[df['f1'].idxmax(), 'round']:.0f})")
    print(f"  Lowest Loss: {df['loss'].min():.4f} (Round {df.loc[df['loss'].idxmin(), 'round']:.0f})")

    print("\nConvergence:")
    last_10_acc = df.tail(10)['accuracy']
    print(f"  Last 10 rounds accuracy: {last_10_acc.mean():.4f} ± {last_10_acc.std():.4f}")

    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze FL experiment results")

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing result CSV files"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*",
        help="Pattern to match CSV files (e.g., 'FedAvg_homo*')"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple experiments"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="accuracy",
        choices=["loss", "accuracy", "precision", "recall", "f1"],
        help="Metric to compare"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Don't show plots (only print summaries)"
    )

    args = parser.parse_args()

    # Find CSV files
    files = find_csv_files(args.results_dir, args.pattern)
    if not files or not files['global']:
        return

    # Print summaries
    for csv_path in files['global']:
        print_summary(csv_path)

    if args.no_plot:
        return

    # Compare mode
    if args.compare and len(files['global']) > 1:
        print(f"\nComparing {len(files['global'])} experiments on {args.metric}...")
        compare_experiments(files['global'], args.metric)

    # Individual plots
    else:
        for global_csv, weight_csv in zip(files['global'], files['weight']):
            print(f"\nPlotting: {Path(global_csv).stem}")
            plot_global_metrics(global_csv, save_plot=True)
            if weight_csv.exists():
                plot_weight_metrics(weight_csv, save_plot=True)


if __name__ == "__main__":
    main()
