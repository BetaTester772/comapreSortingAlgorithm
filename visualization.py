import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# Set style for academic paper quality
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Define consistent colors for each algorithm
colors = {
        'insertion_sort': '#FF6B6B',  # 선명한 빨간색
        'selection_sort': '#4ECDC4',  # 청록색
        'merge_sort'    : '#7B68EE',  # 보라색
        'quick_sort'    : '#FFD700'  # 골드색
}


# Function to create experimental setup architecture diagram
def create_architecture_diagram():
    fig, ax = plt.subplots(figsize=(10, 6))

    # Hide axes
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Draw components
    # Input Data
    rect1 = Rectangle((0.5, 4), 2, 1.2, facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(rect1)
    ax.text(1.5, 4.6, 'Input Data\n(10²-10⁶ elements)', ha='center', va='center', fontsize=10, weight='bold')

    # Data Distributions
    rect2 = Rectangle((0.5, 2), 2, 1.5, facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(rect2)
    ax.text(1.5, 2.75, 'Distributions:\n• Random\n• Sorted\n• Reverse', ha='center', va='center', fontsize=9)

    # Sorting Algorithms
    rect3 = Rectangle((3.5, 3), 3, 2, facecolor='lightcoral', edgecolor='black', linewidth=2)
    ax.add_patch(rect3)
    ax.text(5, 4, 'Sorting Algorithms:\n• Insertion Sort\n• Selection Sort\n• Merge Sort\n• Quick Sort',
            ha='center', va='center', fontsize=9)

    # Measurement
    rect4 = Rectangle((7.5, 3), 2, 2, facecolor='lightyellow', edgecolor='black', linewidth=2)
    ax.add_patch(rect4)
    ax.text(8.5, 4, 'Measurement:\n100 iterations\nμs precision', ha='center', va='center', fontsize=9)

    # System Info
    rect5 = Rectangle((3, 0.5), 4, 1, facecolor='lavender', edgecolor='black', linewidth=2)
    ax.add_patch(rect5)
    ax.text(5, 1, 'ARM64 Processor | Ubuntu 24.04 | GCC 13.3.0', ha='center', va='center', fontsize=9, weight='bold')

    # Draw arrows
    ax.arrow(2.5, 4.6, 0.8, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(2.5, 2.75, 0.8, 0.25, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(6.5, 4, 0.8, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(8.5, 3, 0, -1.3, head_width=0.1, head_length=0.1, fc='black', ec='black')

    plt.title('Experimental Setup Architecture and Data Flow', fontsize=14, weight='bold', pad=20)
    plt.tight_layout()
    return fig


# Function to create logarithmic scale bar chart for overall performance
def create_overall_performance_chart(data):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate mean execution time for each algorithm (in microseconds)
    algorithms = ['insertion_sort', 'merge_sort', 'quick_sort', 'selection_sort']
    mean_times = []
    std_times = []

    for algo in algorithms:
        algo_data = data[data['algorithm'] == algo]['avg_time_ns']  # Convert to microseconds
        mean_times.append(algo_data.mean())
        std_times.append(algo_data.std())

    # Create bar positions
    x = np.arange(len(algorithms))

    # Create bars with error bars
    bars = ax.bar(x, mean_times, yerr=std_times, capsize=5,
                  color=[colors[algo] for algo in algorithms],
                  edgecolor='black', linewidth=1.5)

    # Set logarithmic scale
    ax.set_yscale('log')

    # Customize the plot
    ax.set_xlabel('Sorting Algorithm', fontsize=12, weight='bold')
    ax.set_ylabel('Mean Execution Time (microseconds, log scale)', fontsize=12, weight='bold')
    ax.set_title('Overall Performance Comparison Across All Dataset Sizes', fontsize=14, weight='bold')

    # Set x-axis labels
    labels = ['Insertion Sort', 'Merge Sort', 'Quick Sort', 'Selection Sort']
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, mean_times, std_times)):
        height = bar.get_height()
        # 텍스트 위치를 조정하고 배경 추가
        ax.text(bar.get_x() + bar.get_width() / 2., height * 1.05,
                f'{mean:.0f}±{std:.0f}', ha='center', va='bottom', fontsize=15,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # Add grid for better readability
    ax.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    return fig


# Function to create line graph with logarithmic scales
def create_scaling_performance_chart(data):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    distributions = ['random', 'sorted', 'reverse_sorted']
    axes = [ax1, ax2, ax3]

    for ax, dist in zip(axes, distributions):
        dist_data = data[data['distribution'] == dist]

        for algo in ['insertion_sort', 'merge_sort', 'quick_sort', 'selection_sort']:
            algo_data = dist_data[dist_data['algorithm'] == algo]

            # Group by size and calculate mean
            grouped = algo_data.groupby('size')['avg_time_ns'].mean() / 1e6  # Convert to milliseconds

            # Plot only if data exists
            if not grouped.empty:
                ax.plot(grouped.index, grouped.values, 'o-',
                        color=colors[algo], linewidth=2, markersize=6,
                        label=algo.replace('_', ' ').title())

        # Set logarithmic scales
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Customize subplot
        ax.set_xlabel('Dataset Size', fontsize=11, weight='bold')
        ax.set_ylabel('Execution Time (ms, log scale)', fontsize=11, weight='bold')
        ax.set_title(f'{dist.replace("_", " ").title()} Data', fontsize=12, weight='bold')
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend(loc='upper left', fontsize=9)

    fig.suptitle('Algorithm Performance Scaling by Dataset Size and Distribution',
                 fontsize=14, weight='bold', y=1.02)
    plt.tight_layout()
    return fig


# Function to create grouped bar chart for distribution impact
def create_distribution_impact_chart(data):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Filter for 10,000 elements
    data_10k = data[data['size'] == 10000]

    algorithms = ['insertion_sort', 'merge_sort', 'quick_sort', 'selection_sort']
    distributions = ['random', 'sorted', 'reverse_sorted']

    # Prepare data for grouped bar chart
    bar_width = 0.2
    x = np.arange(len(algorithms))

    for i, dist in enumerate(distributions):
        times = []
        for algo in algorithms:
            algo_dist_data = data_10k[(data_10k['algorithm'] == algo) &
                                      (data_10k['distribution'] == dist)]
            if not algo_dist_data.empty:
                mean_time = algo_dist_data['avg_time_ns'].mean() / 1e6  # Convert to ms
                times.append(mean_time)
            else:
                times.append(0)

        offset = (i - 1) * bar_width
        bars = ax.bar(x + offset, times, bar_width,
                      label=dist.replace('_', ' ').title(),
                      edgecolor='black', linewidth=1)

        # Add value labels on bars
        for bar, time in zip(bars, times):
            if time > 0:
                height = bar.get_height()
                if height > 1:
                    label = f'{height:.1f}'
                else:
                    label = f'{height:.3f}'
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        label, ha='center', va='bottom', fontsize=13)

    # Set logarithmic scale for y-axis
    ax.set_yscale('log')

    # Customize the plot
    ax.set_xlabel('Sorting Algorithm', fontsize=12, weight='bold')
    ax.set_ylabel('Execution Time (ms, log scale)', fontsize=12, weight='bold')
    ax.set_title('Impact of Data Distribution on Algorithm Performance (10,000 elements)',
                 fontsize=14, weight='bold')

    # Set x-axis labels
    labels = ['Insertion Sort', 'Merge Sort', 'Quick Sort', 'Selection Sort']
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.legend(loc='best', fontsize=10, title='Data Distribution')
    ax.grid(True, which="both", ls="-", alpha=0.2, axis='y')

    plt.tight_layout()
    return fig


# Function to create theory vs practice scatter plot
def create_theory_practice_gap_chart(data):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Filter for random distribution only
    random_data = data[data['distribution'] == 'random']

    # Define theoretical complexity (operations)
    def theoretical_ops(n, algo):
        if algo in ['insertion_sort', 'selection_sort']:
            return n * n
        else:  # merge_sort, quick_sort
            return n * np.log2(n) if n > 0 else 0

    algorithms = ['insertion_sort', 'selection_sort', 'merge_sort', 'quick_sort']
    markers = ['o', 's', '^', 'D']

    for algo, marker in zip(algorithms, markers):
        algo_data = random_data[random_data['algorithm'] == algo]
        if not algo_data.empty:
            grouped = algo_data.groupby('size')['avg_time_ns'].mean()

            sizes = grouped.index.values
            actual_times = grouped.values / 1e6  # Convert to ms

            # Calculate theoretical operations
            theory_ops = [theoretical_ops(n, algo) / 1e6 for n in sizes]  # Scale down

            # Plot
            ax.scatter(theory_ops, actual_times,
                       color=colors[algo], marker=marker, s=100,
                       label=algo.replace('_', ' ').title(),
                       edgecolors='black', linewidth=1.5)

            # Add size labels
            for size, theory, actual in zip(sizes, theory_ops, actual_times):
                if size >= 1000:
                    ax.annotate(f'{size // 1000}k',
                                (theory, actual),
                                xytext=(5, 5),
                                textcoords='offset points',
                                fontsize=12, alpha=0.7)
                else:
                    ax.annotate(f'{size}',
                                (theory, actual),
                                xytext=(5, 5),
                                textcoords='offset points',
                                fontsize=12, alpha=0.7)

    # Add theoretical line (perfect correlation)
    max_theory = max([theoretical_ops(1000000, 'insertion_sort') / 1e6,
                      theoretical_ops(1000000, 'quick_sort') / 1e6])
    ax.plot([0, max_theory], [0, max_theory], 'k--', alpha=0.3,
            label='Theoretical Prediction')

    # Set logarithmic scales
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Set bounds same for both axes 10^-4 to 10^6
    ax.set_xlim(1e-4, 1e6)
    ax.set_ylim(1e-4, 1e6)

    # Customize the plot
    ax.set_xlabel('Theoretical Operation Count (millions)', fontsize=12, weight='bold')
    ax.set_ylabel('Actual Execution Time (ms)', fontsize=12, weight='bold')
    ax.set_title('Theory vs Practice: Theoretical Complexity vs Actual Performance',
                 fontsize=14, weight='bold')

    ax.legend(loc='upper left')
    ax.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    return fig


# Main function to generate all plots
def generate_all_plots(csv_file_path):
    """
    Generate all plots from the sorting algorithm performance data.

    Parameters:
    csv_file_path (str): Path to the CSV file containing the performance data
    """
    # Read the data
    data = pd.read_csv(csv_file_path)

    # Create all figures
    fig1 = create_architecture_diagram()
    fig2 = create_overall_performance_chart(data)
    fig3 = create_scaling_performance_chart(data)
    fig4 = create_distribution_impact_chart(data)
    fig5 = create_theory_practice_gap_chart(data)

    # Save all figures
    fig1.savefig('fig1_architecture_diagram.png', dpi=300, bbox_inches='tight')
    fig2.savefig('fig2_overall_performance.png', dpi=300, bbox_inches='tight')
    fig3.savefig('fig3_scaling_performance.png', dpi=300, bbox_inches='tight')
    fig4.savefig('fig4_distribution_impact.png', dpi=300, bbox_inches='tight')
    fig5.savefig('fig5_theory_practice_gap.png', dpi=300, bbox_inches='tight')

    print("All figures have been generated and saved!")

    # Display all figures
    plt.show()


# Example usage
if __name__ == "__main__":
    # You need to provide the path to your CSV file
    # The CSV should have columns: algorithm, size, distribution, avg_time_ns

    # Example of how to use:
    generate_all_plots('sorting_performance_results.csv')

    # # If you want to create sample data for testing:
    # sample_data = pd.DataFrame({
    #         'algorithm'   : ['insertion_sort'] * 15 + ['merge_sort'] * 15 +
    #                         ['quick_sort'] * 15 + ['selection_sort'] * 12,
    #         'size'        : [100, 100, 100, 1000, 1000, 1000, 10000, 10000, 10000,
    #                          100000, 100000, 100000, 1000000, 1000000, 1000000] * 3 +
    #                         [100, 100, 100, 1000, 1000, 1000, 10000, 10000, 10000,
    #                          100000, 100000, 100000],
    #         'distribution': ['random', 'sorted', 'reverse_sorted'] * 19,
    #         'avg_time_ns'     : np.random.lognormal(10, 2, 57) * 1000  # Sample data
    # })

    # Uncomment to test with sample data
    # generate_all_plots(sample_data)
