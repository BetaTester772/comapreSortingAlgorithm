import subprocess
import os
import random
import statistics
import time
import pandas as pd
from typing import List


class SortingAlgorithmMonitor:
    """
    Python wrapper for monitoring C++ sorting algorithm performance.
    Generates test data, executes C++ programs via subprocess, and collects timing data.
    """

    def __init__(self, algorithms: List[str] = None):
        """
        Initialize the monitor with sorting algorithms to test.

        Args:
            algorithms: List of algorithm names (default: all four algorithms)
        """
        if algorithms is None:
            self.algorithms = ['merge_sort', 'quick_sort', 'insertion_sort', 'selection_sort']
        else:
            self.algorithms = algorithms

        self.dataset_sizes = [100, 1000, 10000, 100000, 1000000]
        self.data_distributions = ['random', 'sorted', 'reverse_sorted']
        self.test_iterations = 10
        self.results = []

        # Dictionary to store pre-generated datasets
        self.test_datasets = {}

        # Ensure temp directory exists
        os.makedirs('temp', exist_ok=True)

    def generate_test_data(self, size: int, distribution: str) -> List[int]:
        """
        Generate test dataset based on size and distribution type.

        Args:
            size: Number of elements in the dataset
            distribution: Type of data distribution ('random', 'sorted', 'reverse_sorted')

        Returns:
            List of integers representing the test dataset
        """
        if distribution == 'random':
            # Generate random data (seed is controlled at higher level for reproducibility)
            return [random.randint(1, size * 10) for _ in range(size)]
        elif distribution == 'sorted':
            # Generate sorted data
            return list(range(1, size + 1))
        elif distribution == 'reverse_sorted':
            # Generate reverse sorted data
            return list(range(size, 0, -1))
        else:
            raise ValueError(f"Unknown distribution type: {distribution}")

    def pre_generate_all_datasets(self):
        """
        Pre-generate all test datasets to ensure all algorithms use the same inputs.
        This ensures proper variable control for research purposes.
        """
        print("Pre-generating test datasets for controlled experiments...")

        for size in self.dataset_sizes:
            for distribution in self.data_distributions:
                key = (size, distribution)  # Use tuple as key instead of string

                if distribution == 'random':
                    # For random data, generate multiple datasets (one for each iteration)
                    self.test_datasets[key] = []
                    for i in range(self.test_iterations):
                        dataset = self.generate_test_data(size, distribution)
                        self.test_datasets[key].append(dataset)
                    print(f"  Generated {self.test_iterations} random datasets of size {size}")
                else:
                    # For sorted/reverse_sorted, we only need one dataset
                    dataset = self.generate_test_data(size, distribution)
                    self.test_datasets[key] = [dataset]
                    print(f"  Generated 1 {distribution} dataset of size {size}")

        print("Dataset pre-generation complete.\n")

    def get_test_dataset(self, size: int, distribution: str, iteration: int = 0) -> List[int]:
        """
        Retrieve a pre-generated test dataset.

        Args:
            size: Dataset size
            distribution: Data distribution type
            iteration: Iteration number (for random datasets)

        Returns:
            The pre-generated test dataset
        """
        key = (size, distribution)  # Use tuple as key

        if distribution == 'random':
            # Return the specific iteration's dataset
            return self.test_datasets[key][iteration]
        else:
            # Return the single dataset for sorted/reverse_sorted
            return self.test_datasets[key][0]

    def write_input_file(self, data: List[int], filename: str = 'temp/input.txt'):
        """
        Write test data to input file for C++ program.

        Args:
            data: List of integers to write
            filename: Output filename
        """
        with open(filename, 'w') as f:
            f.write(f"{len(data)}\n")
            f.write(" ".join(map(str, data)) + "\n")

    def execute_cpp_algorithm(self, algorithm: str) -> float:
        """
        Execute C++ sorting algorithm and capture timing output.

        Args:
            algorithm: Name of the sorting algorithm

        Returns:
            Execution time in nanoseconds
        """
        try:
            # Execute the C++ program
            result = subprocess.run(
                    [f'./{algorithm}'],
                    input=open('temp/input.txt', 'r').read(),
                    capture_output=True,
                    text=True,
                    timeout=30  # 30 second timeout
            )

            if result.returncode == 0:
                # Extract timing from stdout (should be in nanoseconds)
                timing_line = result.stdout.strip().split('\n')[-1]
                return float(timing_line)
            else:
                print(f"Error executing {algorithm}: {result.stderr}")
                return -1

        except subprocess.TimeoutExpired:
            print(f"Timeout executing {algorithm}")
            return -1
        except Exception as e:
            print(f"Exception executing {algorithm}: {e}")
            return -1

    def run_single_test(self, algorithm: str, size: int, distribution: str) -> List[float]:
        """
        Run multiple iterations of a single test configuration.
        All algorithms will use the same pre-generated datasets for fair comparison.

        Args:
            algorithm: Sorting algorithm to test
            size: Dataset size
            distribution: Data distribution type

        Returns:
            List of execution times in nanoseconds
        """
        times = []

        for iteration in range(self.test_iterations):
            # Get the pre-generated dataset for this iteration
            data = self.get_test_dataset(size, distribution, iteration)

            # Write the dataset to input file
            self.write_input_file(data)

            if distribution == 'random':
                for r_interation in range(self.test_iterations):
                    print(data)
                    # Execute algorithm and collect timing
                    execution_time = self.execute_cpp_algorithm(algorithm)
                    if execution_time > 0:
                        times.append(execution_time)

                    # Small delay to prevent system interference
                    time.sleep(0.01)
            else:
                # Execute algorithm and collect timing
                execution_time = self.execute_cpp_algorithm(algorithm)
                if execution_time > 0:
                    times.append(execution_time)

                # Small delay to prevent system interference
                time.sleep(0.01)

        return times

    def run_comprehensive_test(self):
        """
        Run comprehensive performance testing across all configurations.
        """
        # Pre-generate all datasets first
        self.pre_generate_all_datasets()

        print("Starting comprehensive sorting algorithm performance test...")
        print(f"Testing algorithms: {', '.join(self.algorithms)}")
        print(f"Dataset sizes: {self.dataset_sizes}")
        print(f"Data distributions: {self.data_distributions}")
        print(f"Iterations per test: {self.test_iterations}")
        print("Note: All algorithms use the same pre-generated datasets for fair comparison")
        print("-" * 60)

        total_tests = len(self.algorithms) * len(self.dataset_sizes) * len(self.data_distributions)
        current_test = 0

        for algorithm in self.algorithms:
            print(f"\nTesting {algorithm}...")

            for size in self.dataset_sizes:
                for distribution in self.data_distributions:
                    current_test += 1

                    if distribution == 'random':
                        print(
                            f"  [{current_test}/{total_tests}] Size: {size}, Distribution: {distribution} (using {self.test_iterations} pre-generated datasets)")
                    else:
                        print(f"  [{current_test}/{total_tests}] Size: {size}, Distribution: {distribution}")

                    # Run test iterations using pre-generated datasets
                    times = self.run_single_test(algorithm, size, distribution)

                    if times:
                        # Calculate statistics
                        avg_time = statistics.mean(times)
                        min_time = min(times)
                        max_time = max(times)
                        std_dev = statistics.stdev(times) if len(times) > 1 else 0

                        # Store results
                        self.results.append({
                                'algorithm'   : algorithm,
                                'size'        : size,
                                'distribution': distribution,
                                'avg_time_ns' : avg_time,
                                'min_time_ns' : min_time,
                                'max_time_ns' : max_time,
                                'std_dev_ns'  : std_dev,
                                'iterations'  : len(times),
                                'all_times'   : times
                        })

                        # Format output based on time magnitude
                        if avg_time >= 1e6:
                            avg_str = f"{avg_time / 1e6:.2f} ms"
                            min_str = f"{min_time / 1e6:.2f} ms"
                            max_str = f"{max_time / 1e6:.2f} ms"
                        elif avg_time >= 1e3:
                            avg_str = f"{avg_time / 1e3:.1f} μs"
                            min_str = f"{min_time / 1e3:.1f} μs"
                            max_str = f"{max_time / 1e3:.1f} μs"
                        else:
                            avg_str = f"{avg_time:.0f} ns"
                            min_str = f"{min_time:.0f} ns"
                            max_str = f"{max_time:.0f} ns"

                        print(f"    Average: {avg_str}, Min: {min_str}, Max: {max_str}")

                        # Show variability
                        if len(times) > 1:
                            cv = (std_dev / avg_time) * 100  # Coefficient of variation
                            print(f"    Variability: {cv:.1f}% (std dev: {std_dev:.0f} ns)")
                    else:
                        print(f"    Failed to collect timing data")

    def save_results_to_csv(self, filename: str = 'sorting_performance_results.csv'):
        """
        Save test results to CSV file using pandas.

        Args:
            filename: Output CSV filename
        """
        if not self.results:
            print("No results to save. Run tests first.")
            return

        # Convert results to DataFrame
        df_data = []
        for result in self.results:
            df_data.append({
                    'algorithm'   : result['algorithm'],
                    'size'        : result['size'],
                    'distribution': result['distribution'],
                    'avg_time_ns' : result['avg_time_ns'],
                    'avg_time_ms' : result['avg_time_ns'] / 1_000_000,  # Convert to milliseconds
                    'min_time_ns' : result['min_time_ns'],
                    'max_time_ns' : result['max_time_ns'],
                    'std_dev_ns'  : result['std_dev_ns'],
                    'iterations'  : result['iterations']
            })

        df = pd.DataFrame(df_data)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

        # Print basic statistics
        print(f"\nDataFrame shape: {df.shape}")
        print(f"Total test configurations: {len(df)}")
        print(f"Algorithms tested: {df['algorithm'].unique()}")
        print(f"Dataset sizes: {sorted(df['size'].unique())}")
        print(f"Data distributions: {df['distribution'].unique()}")

    def save_detailed_results(self, filename: str = 'detailed_results.csv'):
        """
        Save detailed results with individual timing measurements using pandas.

        Args:
            filename: Output CSV filename
        """
        detailed_data = []

        for result in self.results:
            for i, time_val in enumerate(result['all_times'], 1):
                detailed_data.append({
                        'algorithm'   : result['algorithm'],
                        'size'        : result['size'],
                        'distribution': result['distribution'],
                        'iteration'   : i,
                        'time_ns'     : time_val,
                        'time_ms'     : time_val / 1_000_000  # Convert to milliseconds
                })

        detailed_df = pd.DataFrame(detailed_data)
        detailed_df.to_csv(filename, index=False)
        print(f"Detailed results saved to {filename}")
        print(f"Total individual measurements: {len(detailed_df)}")

    def analyze_results(self):
        """
        Perform statistical analysis on results using pandas.
        """
        if not self.results:
            print("No results to analyze.")
            return

        # Convert to DataFrame
        df_data = []
        for result in self.results:
            df_data.append({
                    'algorithm'   : result['algorithm'],
                    'size'        : result['size'],
                    'distribution': result['distribution'],
                    'avg_time_ns' : result['avg_time_ns'],  # Use nanoseconds
                    'avg_time_us' : result['avg_time_ns'] / 1_000,  # Add microseconds for readability
                    'std_dev_ns'  : result['std_dev_ns'],
                    'iterations'  : result['iterations']
            })

        df = pd.DataFrame(df_data)

        print("\n" + "=" * 80)
        print("STATISTICAL ANALYSIS USING PANDAS")
        print("=" * 80)

        # Overall statistics (in microseconds for readability)
        print("\n1. Overall Performance Statistics (microseconds):")
        print(df.groupby('algorithm')['avg_time_us'].agg(['mean', 'std', 'min', 'max']).round(2))

        # Performance by dataset size (in nanoseconds)
        print("\n2. Performance by Dataset Size (nanoseconds):")
        size_analysis = df.groupby(['size', 'algorithm'])['avg_time_ns'].mean().unstack()
        print(size_analysis.round(0))

        # Performance by data distribution (in nanoseconds)
        print("\n3. Performance by Data Distribution (nanoseconds):")
        dist_analysis = df.groupby(['distribution', 'algorithm'])['avg_time_ns'].mean().unstack()
        print(dist_analysis.round(0))

        # Find best algorithm for each condition
        print("\n4. Best Algorithm by Condition:")
        for size in sorted(df['size'].unique()):
            print(f"\n  Dataset Size: {size:,}")
            size_df = df[df['size'] == size]
            for dist in sorted(size_df['distribution'].unique()):
                condition_df = size_df[size_df['distribution'] == dist]
                best_algo = condition_df.loc[condition_df['avg_time_ns'].idxmin(), 'algorithm']
                best_time = condition_df['avg_time_ns'].min()

                # Format time appropriately
                if best_time >= 1e6:
                    time_str = f"{best_time / 1e6:.2f} ms"
                elif best_time >= 1e3:
                    time_str = f"{best_time / 1e3:.1f} μs"
                else:
                    time_str = f"{best_time:.0f} ns"

                print(f"    {dist:15}: {best_algo:15} ({time_str})")

        # Calculate speedup ratios
        print("\n5. Speedup Analysis (vs Selection Sort baseline):")
        for size in sorted(df['size'].unique()):
            size_df = df[df['size'] == size]
            selection_sort_times = size_df[size_df['algorithm'] == 'selection_sort']['avg_time_ns']

            if not selection_sort_times.empty:
                baseline = selection_sort_times.mean()
                baseline_str = f"{baseline / 1e3:.1f} μs" if baseline >= 1e3 else f"{baseline:.0f} ns"
                print(f"\n  Dataset Size: {size:,} (baseline: {baseline_str})")

                for algo in sorted(size_df['algorithm'].unique()):
                    algo_times = size_df[size_df['algorithm'] == algo]['avg_time_ns']
                    if not algo_times.empty:
                        avg_time = algo_times.mean()
                        speedup = baseline / avg_time
                        print(f"    {algo:15}: {speedup:6.2f}x speedup")

        return df

    def create_performance_summary(self):
        """
        Create a comprehensive performance summary using pandas.
        """
        if not self.results:
            print("No results to summarize.")
            return

        # Convert to DataFrame
        df_data = []
        for result in self.results:
            df_data.append({
                    'algorithm'   : result['algorithm'],
                    'size'        : result['size'],
                    'distribution': result['distribution'],
                    'avg_time_ns' : result['avg_time_ns'],
                    'std_dev_ns'  : result['std_dev_ns'],
                    'iterations'  : result['iterations']
            })

        df = pd.DataFrame(df_data)

        # Create pivot table for easy comparison (in nanoseconds)
        pivot_table = df.pivot_table(
                index=['size', 'distribution'],
                columns='algorithm',
                values='avg_time_ns',  # Use nanoseconds instead of milliseconds
                aggfunc='mean'
        )

        print("\n" + "=" * 100)
        print("PERFORMANCE SUMMARY TABLE (Average Time in Nanoseconds)")
        print("=" * 100)
        print(pivot_table.round(0))  # Round to whole nanoseconds

        # Save pivot table to CSV (in nanoseconds)
        pivot_table.to_csv('performance_summary_table.csv')
        print(f"\nPerformance summary table saved to 'performance_summary_table.csv'")

        return df, pivot_table

    def save_test_datasets(self, filename_prefix: str = 'test_datasets'):
        """
        Save the pre-generated test datasets for reproducibility and verification.

        Args:
            filename_prefix: Prefix for the dataset files
        """
        print(f"\nSaving test datasets for reproducibility...")

        datasets_info = []

        for key, datasets in self.test_datasets.items():
            size, distribution = key  # Unpack tuple key

            # Save dataset info for CSV
            for i, dataset in enumerate(datasets):
                datasets_info.append({
                        'size'             : int(size),
                        'distribution'     : distribution,
                        'iteration'        : i + 1,
                        'first_10_elements': str(dataset[:10]),
                        'last_10_elements' : str(dataset[-10:]),
                        'min_value'        : min(dataset),
                        'max_value'        : max(dataset),
                        'checksum'         : sum(dataset)  # Simple checksum for verification
                })

        # Save dataset information to CSV
        df_datasets = pd.DataFrame(datasets_info)
        df_datasets.to_csv(f'{filename_prefix}_info.csv', index=False)
        print(f"  Dataset information saved to '{filename_prefix}_info.csv'")

        # Also save the actual datasets in a separate file for complete reproducibility
        with open(f'{filename_prefix}_full.txt', 'w') as f:
            for key, datasets in self.test_datasets.items():
                size, distribution = key  # Unpack tuple key
                f.write(f"# Size: {size}, Distribution: {distribution}\n")
                for i, dataset in enumerate(datasets):
                    f.write(f"## Iteration {i + 1}\n")
                    f.write(' '.join(map(str, dataset)) + '\n')
                    f.write('\n')

        print(f"  Full datasets saved to '{filename_prefix}_full.txt'")

    def print_summary(self):
        """
        Print a summary of test results.
        """
        if not self.results:
            print("No results to summarize.")
            return

        print("\n" + "=" * 80)
        print("SORTING ALGORITHM PERFORMANCE SUMMARY")
        print("=" * 80)

        # Group results by size and distribution
        for size in self.dataset_sizes:
            print(f"\nDataset Size: {size:,} elements")
            print("-" * 40)

            for distribution in self.data_distributions:
                print(f"\n  {distribution.replace('_', ' ').title()} Data:")

                size_dist_results = [r for r in self.results
                                     if r['size'] == size and r['distribution'] == distribution]

                if size_dist_results:
                    # Sort by average time
                    size_dist_results.sort(key=lambda x: x['avg_time_ns'])

                    for i, result in enumerate(size_dist_results, 1):
                        avg_ms = result['avg_time_ns'] / 1_000_000  # Convert to milliseconds
                        print(f"    {i}. {result['algorithm']:15} {avg_ms:8.3f} ms")

    def compile_cpp_programs(self):
        """
        Compile all C++ sorting algorithm programs.
        """
        print("Compiling C++ sorting algorithms...")

        for algorithm in self.algorithms:
            cpp_file = f"sorting_algorithms/{algorithm}.cpp"
            if os.path.exists(cpp_file):
                compile_command = ["g++", "-O2", "-std=c++17", cpp_file, "-o", algorithm]
                result = subprocess.run(compile_command, capture_output=True, text=True)

                if result.returncode == 0:
                    print(f"  ✓ {algorithm} compiled successfully")
                else:
                    print(f"  ✗ Error compiling {algorithm}: {result.stderr}")
            else:
                print(f"  ✗ {cpp_file} not found")


def main():
    """
    Main function to run the sorting algorithm performance test with pandas analysis.
    """
    # set random seed for reproducibility
    random.seed(42)

    # Initialize monitor
    monitor = SortingAlgorithmMonitor()

    # Compile C++ programs
    monitor.compile_cpp_programs()

    # Run comprehensive tests
    monitor.run_comprehensive_test()

    # Save results using pandas
    monitor.save_results_to_csv()
    monitor.save_detailed_results()

    # Save test datasets for reproducibility
    monitor.save_test_datasets()

    # Perform statistical analysis
    df = monitor.analyze_results()

    # Create performance summary
    df, pivot_table = monitor.create_performance_summary()

    # Print traditional summary
    monitor.print_summary()

    print(f"\nTesting completed. Total test configurations: {len(monitor.results)}")
    print("Files generated:")
    print("  - sorting_performance_results.csv (main results)")
    print("  - detailed_results.csv (individual measurements)")
    print("  - performance_summary_table.csv (pivot table)")
    print("  - test_datasets_info.csv (dataset information)")
    print("  - test_datasets_full.txt (full datasets for reproducibility)")

    # Optional: Display basic pandas info
    if 'df' in locals():
        print(f"\nPandas DataFrame Info:")
        print(f"  Shape: {df.shape}")
        print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        print(f"  Data types: {dict(df.dtypes)}")


if __name__ == "__main__":
    main()