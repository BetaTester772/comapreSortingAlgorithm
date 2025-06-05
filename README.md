# Compare Sorting Alrgorithms

this is a **simple** project to compare the performance of different sorting algorithms in C++ for writing in SKKU
English Writing(GEDG001) class

## how to run

1. Clone the repository
   ```bash
   git clone https://github.com/BetaTester772/comapreSortingAlgorithm
   cd comapreSortingAlgorithm
    ```
2. Install dependencies
   ```bash
   sudo apt install python3 g++
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv sync
   ```
3. Run the main script
   ```bash
   uv run main.py
   ```
   or
    ```bash
    uv run monitor.py
    ```

**Note**: If you want to get test_datasets_full.txt, you can just use `git clone` command. But, sometimes, you must use
`git-lfs` and run `git lfs pull` after cloning the repository. Here is the link to
the [git-lfs](https://git-lfs.github.com/) if you don't have it installed.

## algorithms

- Insertion Sort
- Selection Sort
- Merge Sort
- Quick Sort

## test properties

- **Data size**: 100, 1000, 10000, 100000, 1000000
- **Data type**: Random, Sorted, Reversed
- **iterations**: 100

## test environment

- **CPT**: VM.Standard.A1.flex
- **OS**: Ubuntu 24.04
- **GCC**: 13.3.0

## test results
```text
================================================================================
STATISTICAL ANALYSIS USING PANDAS
================================================================================

1. Overall Performance Statistics (microseconds):
                     mean        std   min         max
algorithm                                             
insertion_sort  145083.93  486169.44  0.19  1688528.32
merge_sort       23047.44   44534.51  8.64   145679.01
quick_sort       13110.24   25592.62  4.70    84526.54
selection_sort   34081.20   50352.03  8.99   101862.37

2. Performance by Dataset Size (nanoseconds):
algorithm  insertion_sort   merge_sort  quick_sort  selection_sort
size                                                              
100                2432.0       9241.0      4742.0          9379.0
1000             180666.0      83195.0     39624.0       1020605.0
10000          16933016.0     880593.0    454963.0     101213615.0
100000        844315291.0    9488956.0   5217884.0             NaN
1000000         1028272.0  104775201.0  59834004.0             NaN

3. Performance by Data Distribution (nanoseconds):
algorithm       insertion_sort  merge_sort  quick_sort  selection_sort
distribution                                                          
random             426425462.0  31974100.0  18480510.0      34008842.0
reverse_sorted      11387527.0  18442566.0  10763223.0      34299191.0
sorted                228554.0  18725646.0  10086997.0      33935566.0

4. Best Algorithm by Condition:

  Dataset Size: 100
    random         : insertion_sort  (2.6 μs)
    reverse_sorted : insertion_sort  (4.5 μs)
    sorted         : insertion_sort  (186 ns)

  Dataset Size: 1,000
    random         : quick_sort      (50.6 μs)
    reverse_sorted : quick_sort      (36.0 μs)
    sorted         : insertion_sort  (1.1 μs)

  Dataset Size: 10,000
    random         : quick_sort      (606.9 μs)
    reverse_sorted : quick_sort      (390.1 μs)
    sorted         : insertion_sort  (11.0 μs)

  Dataset Size: 100,000
    random         : quick_sort      (7.21 ms)
    reverse_sorted : quick_sort      (4.37 ms)
    sorted         : insertion_sort  (102.3 μs)

  Dataset Size: 1,000,000
    random         : quick_sort      (84.53 ms)
    reverse_sorted : quick_sort      (49.01 ms)
    sorted         : insertion_sort  (1.03 ms)

5. Speedup Analysis (vs Selection Sort baseline):

  Dataset Size: 100 (baseline: 9.4 μs)
    insertion_sort :   3.86x speedup
    merge_sort     :   1.01x speedup
    quick_sort     :   1.98x speedup
    selection_sort :   1.00x speedup

  Dataset Size: 1,000 (baseline: 1020.6 μs)
    insertion_sort :   5.65x speedup
    merge_sort     :  12.27x speedup
    quick_sort     :  25.76x speedup
    selection_sort :   1.00x speedup

  Dataset Size: 10,000 (baseline: 101213.6 μs)
    insertion_sort :   5.98x speedup
    merge_sort     : 114.94x speedup
    quick_sort     : 222.47x speedup
    selection_sort :   1.00x speedup

====================================================================================================
PERFORMANCE SUMMARY TABLE (Average Time in Nanoseconds)
====================================================================================================
algorithm               insertion_sort   merge_sort  quick_sort  selection_sort
size    distribution                                                           
100     random            2.609000e+03      10302.0      4792.0          9965.0
        reverse_sorted    4.500000e+03       8644.0      4696.0          9178.0
        sorted            1.860000e+02       8778.0      4739.0          8994.0
1000    random            1.821160e+05     104817.0     50620.0       1021615.0
        reverse_sorted    3.587990e+05      72760.0     35977.0       1026028.0
        sorted            1.084000e+03      72008.0     32273.0       1014171.0
10000   random            1.698880e+07    1158852.0    606880.0     100994945.0
        reverse_sorted    3.379928e+07     736813.0    390073.0     101862366.0
        sorted            1.096700e+04     746113.0    367935.0     100783533.0
100000  random            1.688528e+09   12917521.0   7213715.0             NaN
        reverse_sorted             NaN    7723883.0   4371059.0             NaN
        sorted            1.022590e+05    7825465.0   4068878.0             NaN
1000000 random                     NaN  145679010.0  84526543.0             NaN
        reverse_sorted             NaN   83670727.0  49014308.0             NaN
        sorted            1.028272e+06   84975867.0  45961162.0             NaN

Performance summary table saved to 'performance_summary_table.csv'

================================================================================
SORTING ALGORITHM PERFORMANCE SUMMARY
================================================================================

Dataset Size: 100 elements
----------------------------------------

  Random Data:
    1. insertion_sort     0.003 ms
    2. quick_sort         0.005 ms
    3. selection_sort     0.010 ms
    4. merge_sort         0.010 ms

  Sorted Data:
    1. insertion_sort     0.000 ms
    2. quick_sort         0.005 ms
    3. merge_sort         0.009 ms
    4. selection_sort     0.009 ms

  Reverse Sorted Data:
    1. insertion_sort     0.004 ms
    2. quick_sort         0.005 ms
    3. merge_sort         0.009 ms
    4. selection_sort     0.009 ms

Dataset Size: 1,000 elements
----------------------------------------

  Random Data:
    1. quick_sort         0.051 ms
    2. merge_sort         0.105 ms
    3. insertion_sort     0.182 ms
    4. selection_sort     1.022 ms

  Sorted Data:
    1. insertion_sort     0.001 ms
    2. quick_sort         0.032 ms
    3. merge_sort         0.072 ms
    4. selection_sort     1.014 ms

  Reverse Sorted Data:
    1. quick_sort         0.036 ms
    2. merge_sort         0.073 ms
    3. insertion_sort     0.359 ms
    4. selection_sort     1.026 ms

Dataset Size: 10,000 elements
----------------------------------------

  Random Data:
    1. quick_sort         0.607 ms
    2. merge_sort         1.159 ms
    3. insertion_sort    16.989 ms
    4. selection_sort   100.995 ms

  Sorted Data:
    1. insertion_sort     0.011 ms
    2. quick_sort         0.368 ms
    3. merge_sort         0.746 ms
    4. selection_sort   100.784 ms

  Reverse Sorted Data:
    1. quick_sort         0.390 ms
    2. merge_sort         0.737 ms
    3. insertion_sort    33.799 ms
    4. selection_sort   101.862 ms

Dataset Size: 100,000 elements
----------------------------------------

  Random Data:
    1. quick_sort         7.214 ms
    2. merge_sort        12.918 ms
    3. insertion_sort  1688.528 ms

  Sorted Data:
    1. insertion_sort     0.102 ms
    2. quick_sort         4.069 ms
    3. merge_sort         7.825 ms

  Reverse Sorted Data:
    1. quick_sort         4.371 ms
    2. merge_sort         7.724 ms

Dataset Size: 1,000,000 elements
----------------------------------------

  Random Data:
    1. quick_sort        84.527 ms
    2. merge_sort       145.679 ms

  Sorted Data:
    1. insertion_sort     1.028 ms
    2. quick_sort        45.961 ms
    3. merge_sort        84.976 ms

  Reverse Sorted Data:
    1. quick_sort        49.014 ms
    2. merge_sort        83.671 ms

Testing completed. Total test configurations: 51
Files generated:
  - sorting_performance_results.csv (main results)
  - detailed_results.csv (individual measurements)
  - performance_summary_table.csv (pivot table)
  - test_datasets_info.csv (dataset information)
  - test_datasets_full.txt (full datasets for reproducibility)

Pandas DataFrame Info:
  Shape: (51, 6)
  Memory usage: 7.6 KB
  Data types: {'algorithm': dtype('O'), 'size': dtype('int64'), 'distribution': dtype('O'), 'avg_time_ns': dtype('float64'), 'std_dev_ns': dtype('float64'), 'iterations': dtype('int64')}
```

## test analysis
```text
================================================================================
THEORY VS PRACTICE: EMPIRICAL COMPLEXITY ANALYSIS
================================================================================

Complexity Analysis Summary:
--------------------------------------------------------------------------------
Algorithm            Theoretical  Empirical    Gap        R²       Constant Factor
--------------------------------------------------------------------------------
merge_sort           O(n^1.0)      O(n^1.04)         3.9%    0.9999   8.27e+01
quick_sort           O(n^1.0)      O(n^1.06)         6.5%    0.9999   3.40e+01
insertion_sort       O(n^2.0)      O(n^1.94)         3.0%    0.9997   3.11e-01
selection_sort       O(n^2.0)      O(n^2.00)         0.1%    1.0000   9.89e-01

================================================================================
CACHE EFFECTS AND MEMORY LOCALITY ANALYSIS
================================================================================

Cache Efficiency Summary:
------------------------------------------------------------
merge_sort           Efficiency degradation:  -42.0% (Cache-friendly)
quick_sort           Efficiency degradation:  -30.2% (Cache-friendly)
insertion_sort       Efficiency degradation:  -23.8% (Cache-friendly)
selection_sort       Efficiency degradation:    nan% (Cache-friendly)

================================================================================
PRACTICAL PERFORMANCE ZONES FOR ALGORITHM SELECTION
================================================================================

================================================================================
COMPILER OPTIMIZATION IMPACT ANALYSIS
================================================================================

Compiler Optimization Effectiveness:
------------------------------------------------------------
insertion_sort       Scaling efficiency: 117.02% (Well optimized)
merge_sort           Scaling efficiency: 121.64% (Well optimized)
quick_sort           Scaling efficiency: 115.19% (Well optimized)
selection_sort       Scaling efficiency: 99.35% (Well optimized)

================================================================================
GENERATING RESEARCH-ORIENTED ANALYSIS REPORT
================================================================================
✓ Research-oriented report saved to 'result/research_analysis_report.md'
✓ LaTeX tables saved to 'result/latex_tables.tex'
```