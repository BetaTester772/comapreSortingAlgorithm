# Compare Sorting Alrgorithms
this is a **simple** project to compare the performance of different sorting algorithms in C++ for writing in SKKU English Writing(GEDG001) class

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
## test environment
- CPT: VM.Standard.A1.flex
- OS: Ubuntu 24.04
- GCC: 13.3.0

## algorithms
- Insertion Sort
- Selection Sort
- Merge Sort
- Quick Sort