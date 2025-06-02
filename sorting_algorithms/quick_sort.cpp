#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>

using namespace std;
using namespace std::chrono;

/**
 * Quick Sort Implementation
 * Time Complexity: O(n log n) - Best and Average Case, O(n²) - Worst Case
 * Space Complexity: O(log n) - due to recursion stack
 * Stable: No
 * In-place: Yes
 *
 * Academic Implementation for Performance Testing
 * Uses random pivot selection to avoid worst-case performance
 */

// Random number generator for pivot selection
random_device rd;
mt19937 gen(rd());

void swap(vector<int>& arr, int i, int j) {
    int temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
}

int randomizedPartition(vector<int>& arr, int low, int high) {
    // Choose random pivot and swap with last element
    uniform_int_distribution<> dis(low, high);
    int randomIndex = dis(gen);
    swap(arr, randomIndex, high);

    // Standard partition with last element as pivot
    int pivot = arr[high];
    int i = low - 1; // Index of smaller element

    for (int j = low; j < high; j++) {
        // If current element is smaller than or equal to pivot
        if (arr[j] <= pivot) {
            i++;
            swap(arr, i, j);
        }
    }
    swap(arr, i + 1, high);
    return i + 1;
}

void quickSort(vector<int>& arr, int low, int high) {
    if (low < high) {
        // Partition the array and get pivot index
        int pivotIndex = randomizedPartition(arr, low, high);

        // Recursively sort elements before and after partition
        quickSort(arr, low, pivotIndex - 1);
        quickSort(arr, pivotIndex + 1, high);
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
    int n = arr.size();

    // Start timing
    auto start = high_resolution_clock::now();

    // Perform quick sort
    quickSort(arr, 0, n - 1);

    // End timing
    auto end = high_resolution_clock::now();

    // Calculate duration in nanoseconds
    auto duration = duration_cast<nanoseconds>(end - start);

    // Output the execution time in nanoseconds
    cout << duration.count() << endl;

    return 0;
}