#include <iostream>
#include <vector>
#include <chrono>

using namespace std;
using namespace std::chrono;

/**
 * Insertion Sort Implementation
 * Time Complexity: O(n) - Best Case (already sorted), O(n²) - Average and Worst Case
 * Space Complexity: O(1) - in-place sorting
 * Stable: Yes
 * In-place: Yes
 *
 * Academic Implementation for Performance Testing
 * Efficient for small datasets and nearly sorted arrays
 */

void insertionSort(vector<int>& arr) {
    int n = arr.size();

    // Start from the second element (index 1)
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;

        // Move elements of arr[0..i-1] that are greater than key
        // one position ahead of their current position
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }

        // Place key at its correct position
        arr[j + 1] = key;
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

    // Perform insertion sort
    insertionSort(arr);

    // End timing
    auto end = high_resolution_clock::now();

    // Calculate duration in nanoseconds
    auto duration = duration_cast<nanoseconds>(end - start);

    // Output the execution time in nanoseconds
    cout << duration.count() << endl;

    return 0;
}