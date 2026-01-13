"""
Visualization script for Comprehensive Accuracy Matrix Report
Generates publication-quality figures for federated learning experiments
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Data from the report
algorithms = ['FedAdagrad', 'FedAdam', 'FedAvg', 'FedAvgM', 'FedProx', 'FedYogi']
distributions = ['homo', 'Dir10.0', 'Dir1.0', 'Dir0.5', 'C5', 'C4', 'Dir0.1', 'C3', 'C2']

# Best Accuracy Matrix (currently Distribution × Algorithm, need to transpose)
best_accuracy_data = [
    [61.02, 59.68, 63.64, 52.31, 64.77, 58.21],  # homo
    [60.70, 60.28, 63.24, 59.29, 62.84, 58.21],  # Dir10.0
    [59.50, 58.38, 62.13, 57.11, 62.80, 57.81],  # Dir1.0
    [58.91, 57.26, 61.33, 56.70, 61.00, 56.04],  # Dir0.5
    [57.85, 50.71, 59.52, 55.07, 58.91, 53.74],  # C5
    [56.58, 53.41, 58.41, 54.27, 58.46, 52.96],  # C4
    [53.68, 49.99, 54.86, 49.23, 54.69, 48.57],  # Dir0.1
    [51.48, 48.12, 53.78, 49.59, 56.04, 48.28],  # C3
    [48.49, 47.92, 51.41, 48.26, 49.53, 47.80],  # C2
]

# Final Round Accuracy Matrix (currently Distribution × Algorithm, need to transpose)
final_accuracy_data = [
    [57.04, 53.14, 57.81, 52.31, 59.46, 51.72],  # homo
    [58.00, 53.38, 59.05, 51.67, 57.86, 51.53],  # Dir10.0
    [56.06, 52.53, 55.27, 47.76, 54.70, 51.47],  # Dir1.0
    [55.20, 51.36, 54.32, 49.29, 55.04, 49.13],  # Dir0.5
    [54.27, 46.27, 53.40, 48.00, 52.02, 48.99],  # C5
    [52.58, 49.89, 51.44, 46.88, 52.15, 48.56],  # C4
    [50.02, 46.88, 50.34, 45.02, 50.29, 45.31],  # Dir0.1
    [47.57, 42.99, 49.14, 45.13, 50.66, 46.10],  # C3
    [47.94, 46.00, 50.54, 45.61, 46.80, 47.38],  # C2
]

# Best Round Number Matrix
best_round_data = [
    [131, 51, 47, 19, 43, 36],    # homo
    [86, 45, 47, 48, 46, 41],     # Dir10.0
    [104, 38, 39, 31, 37, 34],    # Dir1.0
    [115, 46, 51, 42, 44, 38],    # Dir0.5
    [91, 29, 63, 39, 47, 42],     # C5
    [86, 36, 51, 39, 50, 28],     # C4
    [69, 34, 56, 31, 49, 41],     # Dir0.1
    [66, 35, 69, 46, 62, 38],     # C3
    [441, 489, 500, 386, 478, 369], # C2
]

# Convert to numpy arrays for easy transposition
best_accuracy = np.array(best_accuracy_data)
final_accuracy = np.array(final_accuracy_data)
best_round = np.array(best_round_data)

# Algorithm statistics
algo_stats = {
    'Algorithm': algorithms,
    'Avg Best': [58.78, 58.70, 56.47, 53.97, 53.54, 53.51],
    'Avg Final': [53.22, 53.48, 53.19, 49.16, 47.96, 48.91],
    'Max': [64.77, 63.64, 61.02, 60.28, 59.29, 58.21],
}

# Distribution statistics
dist_stats = {
    'Distribution': distributions,
    'Avg Best': [59.94, 60.76, 59.62, 58.54, 55.97, 55.68, 51.84, 51.22, 48.90],
    'Avg Final': [55.25, 55.25, 52.97, 52.39, 50.49, 50.25, 47.98, 46.93, 47.38],
}

# Top 10 combinations
top10 = [
    ('FedProx', 'homo', 64.77),
    ('FedAvg', 'homo', 63.64),
    ('FedAvg', 'Dir10.0', 63.24),
    ('FedProx', 'Dir10.0', 62.84),
    ('FedProx', 'Dir1.0', 62.80),
    ('FedAvg', 'Dir1.0', 62.13),
    ('FedAvg', 'Dir0.5', 61.33),
    ('FedAdagrad', 'homo', 61.02),
    ('FedProx', 'Dir0.5', 61.00),
    ('FedAdagrad', 'Dir10.0', 60.70),
]


def plot_heatmap_transposed(data, title, filename, fmt='.2f', cmap='YlOrRd', vmin=None, vmax=None):
    """
    Plot heatmap with Algorithm as rows and Distribution as columns
    (Transposed from original Distribution × Algorithm format)
    """
    # Transpose: now shape is (6 algorithms, 9 distributions)
    data_transposed = data.T

    fig, ax = plt.subplots(figsize=(14, 6))

    # Create heatmap
    im = ax.imshow(data_transposed, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

    # Set ticks
    ax.set_xticks(np.arange(len(distributions)))
    ax.set_yticks(np.arange(len(algorithms)))
    ax.set_xticklabels(distributions, rotation=45, ha='right')
    ax.set_yticklabels(algorithms)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy (%)', rotation=270, labelpad=20)

    # Add text annotations
    for i in range(len(algorithms)):
        for j in range(len(distributions)):
            value = data_transposed[i, j]
            text_color = 'white' if value > (data_transposed.max() - data_transposed.min())/2 + data_transposed.min() else 'black'
            text = ax.text(j, i, f'{value:.2f}', ha="center", va="center",
                          color=text_color, fontsize=9, weight='bold')

    # Highlight winners (max in each column)
    for j in range(len(distributions)):
        max_idx = np.argmax(data_transposed[:, j])
        rect = Rectangle((j-0.45, max_idx-0.45), 0.9, 0.9,
                        fill=False, edgecolor='green', linewidth=3)
        ax.add_patch(rect)

    ax.set_xlabel('Distribution (sorted by difficulty: easy → hard)', fontsize=12, weight='bold')
    ax.set_ylabel('Algorithm', fontsize=12, weight='bold')
    ax.set_title(title, fontsize=14, weight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def plot_convergence_heatmap():
    """Plot convergence speed (best round number) heatmap - transposed"""
    # Transpose data
    data_transposed = best_round.T

    fig, ax = plt.subplots(figsize=(14, 6))

    # Use reversed colormap (lower round = better = darker)
    im = ax.imshow(data_transposed, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=150)

    # Set ticks
    ax.set_xticks(np.arange(len(distributions)))
    ax.set_yticks(np.arange(len(algorithms)))
    ax.set_xticklabels(distributions, rotation=45, ha='right')
    ax.set_yticklabels(algorithms)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Round Number (lower = faster)', rotation=270, labelpad=20)

    # Add text annotations with special marking for C2
    for i in range(len(algorithms)):
        for j in range(len(distributions)):
            value = data_transposed[i, j]
            # Special formatting for very late convergence (C2)
            if value > 300:
                text = ax.text(j, i, f'{int(value)}', ha="center", va="center",
                              color='darkred', fontsize=8, weight='bold')
            elif value < 50:
                text = ax.text(j, i, f'{int(value)}', ha="center", va="center",
                              color='darkgreen', fontsize=9, weight='bold')
            else:
                text_color = 'white' if value > 75 else 'black'
                text = ax.text(j, i, f'{int(value)}', ha="center", va="center",
                              color=text_color, fontsize=9, weight='bold')

    ax.set_xlabel('Distribution', fontsize=12, weight='bold')
    ax.set_ylabel('Algorithm', fontsize=12, weight='bold')
    ax.set_title('Convergence Speed: Round Where Best Accuracy Was Achieved\nGreen=Fast (<50) | Red=Very Slow (>300)',
                 fontsize=14, weight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('figures/03_convergence_speed_heatmap.png', dpi=300, bbox_inches='tight')
    print("Saved: figures/03_convergence_speed_heatmap.png")
    plt.close()


def plot_algorithm_comparison():
    """Plot algorithm performance comparison"""
    df = pd.DataFrame(algo_stats)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Sort by average best accuracy
    df_sorted = df.sort_values('Avg Best', ascending=True)

    # Plot 1: Average Best Accuracy
    colors = ['#d62728' if x < 55 else '#ff7f0e' if x < 58 else '#2ca02c' for x in df_sorted['Avg Best']]
    axes[0].barh(df_sorted['Algorithm'], df_sorted['Avg Best'], color=colors, edgecolor='black', linewidth=1.5)
    axes[0].set_xlabel('Average Best Accuracy (%)', fontsize=11, weight='bold')
    axes[0].set_title('Average Best Accuracy\n(across 9 distributions)', fontsize=12, weight='bold')
    axes[0].axvline(x=56, color='gray', linestyle='--', alpha=0.5, label='Good threshold (56%)')
    for i, (alg, val) in enumerate(zip(df_sorted['Algorithm'], df_sorted['Avg Best'])):
        axes[0].text(val + 0.3, i, f'{val:.2f}%', va='center', fontsize=10, weight='bold')
    axes[0].set_xlim(50, 60)
    axes[0].legend()

    # Plot 2: Average Final Accuracy
    df_sorted2 = df.sort_values('Avg Final', ascending=True)
    colors2 = ['#d62728' if x < 48 else '#ff7f0e' if x < 51 else '#2ca02c' for x in df_sorted2['Avg Final']]
    axes[1].barh(df_sorted2['Algorithm'], df_sorted2['Avg Final'], color=colors2, edgecolor='black', linewidth=1.5)
    axes[1].set_xlabel('Average Final Accuracy (%)', fontsize=11, weight='bold')
    axes[1].set_title('Average Final Round Accuracy\n(convergence quality)', fontsize=12, weight='bold')
    axes[1].axvline(x=50, color='gray', linestyle='--', alpha=0.5, label='Good threshold (50%)')
    for i, (alg, val) in enumerate(zip(df_sorted2['Algorithm'], df_sorted2['Avg Final'])):
        axes[1].text(val + 0.3, i, f'{val:.2f}%', va='center', fontsize=10, weight='bold')
    axes[1].set_xlim(45, 55)
    axes[1].legend()

    # Plot 3: Maximum Accuracy
    df_sorted3 = df.sort_values('Max', ascending=True)
    colors3 = ['#d62728' if x < 58 else '#ff7f0e' if x < 62 else '#2ca02c' for x in df_sorted3['Max']]
    axes[2].barh(df_sorted3['Algorithm'], df_sorted3['Max'], color=colors3, edgecolor='black', linewidth=1.5)
    axes[2].set_xlabel('Maximum Accuracy (%)', fontsize=11, weight='bold')
    axes[2].set_title('Peak Performance\n(best result achieved)', fontsize=12, weight='bold')
    for i, (alg, val) in enumerate(zip(df_sorted3['Algorithm'], df_sorted3['Max'])):
        axes[2].text(val + 0.3, i, f'{val:.2f}%', va='center', fontsize=10, weight='bold')
    axes[2].set_xlim(55, 66)

    plt.tight_layout()
    plt.savefig('figures/04_algorithm_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: figures/04_algorithm_comparison.png")
    plt.close()


def plot_distribution_difficulty():
    """Plot distribution difficulty ranking"""
    df = pd.DataFrame(dist_stats)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Average Best Accuracy (sorted by difficulty)
    df_sorted = df.sort_values('Avg Best', ascending=True)

    # Color by difficulty
    colors = []
    for dist, val in zip(df_sorted['Distribution'], df_sorted['Avg Best']):
        if val < 51:
            colors.append('#d62728')  # Very Hard (red)
        elif val < 56:
            colors.append('#ff7f0e')  # Hard (orange)
        elif val < 59:
            colors.append('#ffdd57')  # Medium (yellow)
        else:
            colors.append('#2ca02c')  # Easy (green)

    bars1 = axes[0].barh(df_sorted['Distribution'], df_sorted['Avg Best'],
                         color=colors, edgecolor='black', linewidth=1.5)
    axes[0].set_xlabel('Average Best Accuracy (%)', fontsize=12, weight='bold')
    axes[0].set_title('Distribution Difficulty Ranking\n(by Average Best Accuracy)',
                      fontsize=13, weight='bold')

    # Add difficulty labels
    for i, (dist, val) in enumerate(zip(df_sorted['Distribution'], df_sorted['Avg Best'])):
        if val < 51:
            label = 'Very Hard'
        elif val < 56:
            label = 'Hard'
        elif val < 59:
            label = 'Medium'
        else:
            label = 'Easy'
        axes[0].text(val + 0.5, i, f'{val:.2f}% {label}', va='center', fontsize=9, weight='bold')

    axes[0].set_xlim(45, 63)
    axes[0].axvline(x=51, color='red', linestyle='--', alpha=0.3, linewidth=2)
    axes[0].axvline(x=56, color='orange', linestyle='--', alpha=0.3, linewidth=2)
    axes[0].axvline(x=59, color='yellow', linestyle='--', alpha=0.3, linewidth=2)

    # Plot 2: Comparison of Best vs Final
    x = np.arange(len(df_sorted))
    width = 0.35

    bars1 = axes[1].bar(x - width/2, df_sorted['Avg Best'], width,
                        label='Avg Best', color='#2ca02c', edgecolor='black', linewidth=1.2)
    bars2 = axes[1].bar(x + width/2, df_sorted['Avg Final'], width,
                        label='Avg Final', color='#1f77b4', edgecolor='black', linewidth=1.2)

    axes[1].set_xlabel('Distribution', fontsize=12, weight='bold')
    axes[1].set_ylabel('Accuracy (%)', fontsize=12, weight='bold')
    axes[1].set_title('Best vs Final Accuracy by Distribution\n(Gap indicates training instability)',
                      fontsize=13, weight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df_sorted['Distribution'], rotation=45, ha='right')
    axes[1].legend(fontsize=11)
    axes[1].grid(axis='y', alpha=0.3)

    # Add gap annotations for largest gaps
    for i, (best, final) in enumerate(zip(df_sorted['Avg Best'], df_sorted['Avg Final'])):
        gap = best - final
        if gap > 5:  # Show only significant gaps
            axes[1].annotate(f'Δ{gap:.1f}%', xy=(i, (best + final)/2),
                           ha='center', fontsize=8, weight='bold', color='red')

    plt.tight_layout()
    plt.savefig('figures/05_distribution_difficulty.png', dpi=300, bbox_inches='tight')
    print("Saved: figures/05_distribution_difficulty.png")
    plt.close()


def plot_top10_combinations():
    """Plot top 10 algorithm-distribution combinations"""
    fig, ax = plt.subplots(figsize=(12, 7))

    labels = [f"{algo}\n{dist}" for algo, dist, _ in top10]
    values = [acc for _, _, acc in top10]
    colors_map = {
        'FedProx': '#d62728',
        'FedAvg': '#2ca02c',
        'FedAdagrad': '#ff7f0e',
        'FedAvgM': '#9467bd',
        'FedAdam': '#8c564b',
        'FedYogi': '#e377c2',
    }
    colors = [colors_map[algo] for algo, _, _ in top10]

    # Reverse for top-to-bottom display
    labels = labels[::-1]
    values = values[::-1]
    colors = colors[::-1]

    bars = ax.barh(range(len(labels)), values, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + 0.2, i, f'{val:.2f}%', va='center', fontsize=11, weight='bold')

    # Add medal markers
    medal_positions = [len(labels) - 1, len(labels) - 2, len(labels) - 3]
    medals = ['#1', '#2', '#3']
    for pos, medal in zip(medal_positions, medals):
        ax.text(45, pos, medal, fontsize=12, va='center', weight='bold',
                bbox=dict(boxstyle='circle', facecolor='gold' if medal=='#1' else 'silver' if medal=='#2' else '#CD7F32', alpha=0.8))

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Best Accuracy (%)', fontsize=12, weight='bold')
    ax.set_title('Top 10 Algorithm-Distribution Combinations\n(Best Performance Achieved)',
                 fontsize=14, weight='bold', pad=20)
    ax.set_xlim(45, 67)
    ax.grid(axis='x', alpha=0.3)

    # Add legend for algorithms
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, edgecolor='black', label=algo)
                      for algo, color in colors_map.items()]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10, ncol=2)

    plt.tight_layout()
    plt.savefig('figures/06_top10_combinations.png', dpi=300, bbox_inches='tight')
    print("Saved: figures/06_top10_combinations.png")
    plt.close()


def plot_algorithm_vs_distribution_overview():
    """Create comprehensive overview: algorithm performance across distributions"""
    fig, ax = plt.subplots(figsize=(16, 8))

    x = np.arange(len(distributions))
    width = 0.13  # Width of each bar

    colors_map = {
        'FedAdagrad': '#ff7f0e',
        'FedAdam': '#8c564b',
        'FedAvg': '#2ca02c',
        'FedAvgM': '#9467bd',
        'FedProx': '#d62728',
        'FedYogi': '#e377c2',
    }

    for i, algo in enumerate(algorithms):
        offset = width * (i - 2.5)
        values = best_accuracy[:, i]
        bars = ax.bar(x + offset, values, width, label=algo,
                     color=colors_map[algo], edgecolor='black', linewidth=0.8)

    ax.set_xlabel('Distribution (sorted by difficulty: easy → hard)', fontsize=13, weight='bold')
    ax.set_ylabel('Best Accuracy (%)', fontsize=13, weight='bold')
    ax.set_title('Comprehensive Algorithm Performance Across All Distributions\n(Bar Chart View)',
                 fontsize=15, weight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(distributions, rotation=45, ha='right', fontsize=11)
    ax.legend(loc='upper right', fontsize=11, ncol=2)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(45, 67)

    # Add difficulty zones
    ax.axvspan(-0.5, 2.5, alpha=0.1, color='green', label='Easy (>58%)')
    ax.axvspan(2.5, 6.5, alpha=0.1, color='yellow')
    ax.axvspan(6.5, 8.5, alpha=0.1, color='red')

    plt.tight_layout()
    plt.savefig('figures/07_algorithm_vs_distribution_overview.png', dpi=300, bbox_inches='tight')
    print("Saved: figures/07_algorithm_vs_distribution_overview.png")
    plt.close()


def main():
    """Generate all visualizations"""
    import os
    os.makedirs('figures', exist_ok=True)

    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS FOR ACCURACY MATRIX REPORT")
    print("="*60 + "\n")

    # 1. Best Accuracy Heatmap (TRANSPOSED)
    print("1. Creating Best Accuracy Matrix (Transposed: Algorithm x Distribution)...")
    plot_heatmap_transposed(
        best_accuracy,
        'Best Accuracy Matrix (Peak Performance)\nAlgorithm × Distribution | Green Box = Winner',
        'figures/01_best_accuracy_matrix_transposed.png',
        cmap='RdYlGn',
        vmin=45,
        vmax=65
    )

    # 2. Final Round Accuracy Heatmap (TRANSPOSED)
    print("2. Creating Final Round Accuracy Matrix (Transposed: Algorithm x Distribution)...")
    plot_heatmap_transposed(
        final_accuracy,
        'Final Round Accuracy Matrix (Convergence Quality)\nAlgorithm × Distribution | Green Box = Winner',
        'figures/02_final_accuracy_matrix_transposed.png',
        cmap='RdYlGn',
        vmin=42,
        vmax=60
    )

    # 3. Convergence Speed Heatmap
    print("3. Creating Convergence Speed Heatmap...")
    plot_convergence_heatmap()

    # 4. Algorithm Comparison
    print("4. Creating Algorithm Performance Comparison...")
    plot_algorithm_comparison()

    # 5. Distribution Difficulty
    print("5. Creating Distribution Difficulty Analysis...")
    plot_distribution_difficulty()

    # 6. Top 10 Combinations
    print("6. Creating Top 10 Combinations Chart...")
    plot_top10_combinations()

    # 7. Comprehensive Overview
    print("7. Creating Comprehensive Overview...")
    plot_algorithm_vs_distribution_overview()

    print("\n" + "="*60)
    print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("="*60)
    print("\nOutput location: figures/")
    print("\nGenerated files:")
    print("  1. 01_best_accuracy_matrix_transposed.png")
    print("  2. 02_final_accuracy_matrix_transposed.png")
    print("  3. 03_convergence_speed_heatmap.png")
    print("  4. 04_algorithm_comparison.png")
    print("  5. 05_distribution_difficulty.png")
    print("  6. 06_top10_combinations.png")
    print("  7. 07_algorithm_vs_distribution_overview.png")
    print("\nAll figures are publication-ready at 300 DPI\n")


if __name__ == "__main__":
    main()
