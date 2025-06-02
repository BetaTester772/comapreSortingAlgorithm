import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load results
df = pd.read_csv('sorting_performance_results.csv')

# Performance comparison by size using bar plots
for size in df['size'].unique():
    subset = df[df['size'] == size]

    # Prepare data for grouped bar chart
    algorithms = subset['algorithm'].unique()
    distributions = subset['distribution'].unique()

    # Set up the bar plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Number of algorithms and width of bars
    n_algorithms = len(algorithms)
    bar_width = 0.25
    r = np.arange(n_algorithms)

    # Create bars for each distribution
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, Teal, Blue

    for i, dist in enumerate(distributions):
        dist_data = subset[subset['distribution'] == dist]
        # Ensure algorithms are in same order
        times = [dist_data[dist_data['algorithm'] == algo]['avg_time_ns'].iloc[0]
                 if not dist_data[dist_data['algorithm'] == algo].empty else 0
                 for algo in algorithms]

        bars = ax.bar([x + bar_width * i for x in r], times,
                      bar_width, label=f'{dist.replace("_", " ").title()} Data',
                      color=colors[i % len(colors)], alpha=0.8)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    # Customize the plot
    ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Time (ns)', fontsize=12, fontweight='bold')
    ax.set_title(f'Sorting Algorithm Performance - {size:,} Elements',
                 fontsize=14, fontweight='bold')
    ax.set_xticks([x + bar_width for x in r])
    ax.set_xticklabels([algo.replace('_', ' ').title() for algo in algorithms],
                       rotation=45, ha='right')
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    # Use log scale if there's a large difference in values
    max_time = subset['avg_time_ns'].max()
    min_time = subset['avg_time_ns'].min()
    if max_time / min_time > 100:
        ax.set_yscale('log')
        ax.set_ylabel('Average Time (ns) - Log Scale', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'performance_bar_size_{size}.png', dpi=300, bbox_inches='tight')
    plt.show()

# Overall comparison bar chart (all sizes, random data only)
plt.figure(figsize=(14, 8))
random_data = df[df['distribution'] == 'random']

# Pivot for easier plotting
pivot_random = random_data.pivot(index='algorithm', columns='size', values='avg_time_ns')

# Create grouped bar chart
ax = pivot_random.plot(kind='bar', figsize=(14, 8),
                       color=['#FF9999', '#66B2FF', '#99FF99', '#FFB366'],
                       alpha=0.8, width=0.8)

# Customize the plot
ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Time (ns)', fontsize=12, fontweight='bold')
ax.set_title('Sorting Algorithm Performance Comparison (Random Data)',
             fontsize=14, fontweight='bold')
ax.legend(title='Dataset Size', title_fontsize=11, fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Add value labels on bars (optional, for smaller datasets)
if len(pivot_random.columns) <= 4:
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', rotation=90, fontsize=8)

plt.tight_layout()
plt.savefig('overall_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Heatmap of performance
pivot_df = pd.read_csv('performance_summary_table.csv', index_col=[0, 1])
plt.figure(figsize=(12, 10))
sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='YlOrRd',
            cbar_kws={'label': 'Average Time (ns)'})
plt.title('Algorithm Performance Heatmap', fontsize=14, fontweight='bold')
plt.xlabel('Algorithm', fontsize=12, fontweight='bold')
plt.ylabel('Dataset Size & Distribution', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('performance_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Performance ratio comparison (relative to insertion sort)
plt.figure(figsize=(12, 8))
baseline_algo = 'insertion_sort'

for size in sorted(df['size'].unique()):
    size_data = df[df['size'] == size]
    random_size_data = size_data[size_data['distribution'] == 'random']

    if not random_size_data.empty:
        baseline_time = random_size_data[random_size_data['algorithm'] == baseline_algo]['avg_time_ns']

        if not baseline_time.empty:
            baseline_val = baseline_time.iloc[0]
            ratios = []
            algorithms = []

            for algo in random_size_data['algorithm'].unique():
                algo_time = random_size_data[random_size_data['algorithm'] == algo]['avg_time_ns'].iloc[0]
                ratio = baseline_val / algo_time  # Speedup ratio
                ratios.append(ratio)
                algorithms.append(algo.replace('_', ' ').title())

            # Create bar plot for this size
            bars = plt.bar([f"{algo}\n({size:,})" for algo in algorithms], ratios,
                           alpha=0.7, label=f'{size:,} elements')

# Customize the speedup comparison plot
plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Baseline (1x)')
plt.xlabel('Algorithm (Dataset Size)', fontsize=12, fontweight='bold')
plt.ylabel('Speedup Ratio (vs Insertion Sort)', fontsize=12, fontweight='bold')
plt.title('Algorithm Speedup Comparison (Random Data)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('speedup_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Statistical summary
print("Statistical Summary:")
print(df.groupby('algorithm')['avg_time_ns'].describe())
