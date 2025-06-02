#include <iostream>
#include <vector>
#include <chrono>

using namespace std;
using namespace std::chrono;

/**
 * Selection Sort Implementation
 * Time Complexity: O(n²) - Best, Average, and Worst Case
 * Space Complexity: O(1) - in-place sorting
 * Stable: No (can be made stable with modifications)
 * In-place: Yes
 *
 * Academic Implementation for Performance Testing
 * Simple algorithm with consistent O(n²) performance regardless of input
 */

void selectionSort(vector<int>& arr) {
    int n = arr.size();

    // Traverse through all array elements
    for (int i = 0; i < n - 1; i++) {
        // Find the minimum element in remaining unsorted array
        int minIndex = i;
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[minIndex]) {
                minIndex = j;
            }
        }

        // Swap the found minimum element with the first element
        if (minIndex != i) {
            int temp = arr[minIndex];
            arr[minIndex] = arr[i];
            arr[i] = temp;
        }
    }
}

vector<int> readInput() {
    int n;
    cin >> n;

    vector<int> arr(n);
    for (int i = 0; i < n; i++) {
        cin >> arr[i];
    }

    return arr;
}

int main() {
    // Read input data
    vector<int> arr = readInput();

    // Start timing
    auto start = high_resolution_clock::now();

    // Perform selection sort
    selectionSort(arr);

    // End timing
    auto end = high_resolution_clock::now();

    // Calculate duration in nanoseconds
    auto duration = duration_cast<nanoseconds>(end - start);

    // Output the execution time in nanoseconds
    cout << duration.count() << endl;

    return 0;
}