#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>

using namespace std;
using namespace std::chrono;

/**
 * Merge Sort Implementation
 * Time Complexity: O(n log n) - Best, Average, Worst Case
 * Space Complexity: O(n) - requires additional memory for merging
 * Stable: Yes
 *
 * Academic Implementation for Performance Testing
 */

void merge(vector<int>& arr, int left, int mid, int right) {
    // Calculate sizes of two subarrays
    int n1 = mid - left + 1;
    int n2 = right - mid;

    // Create temporary arrays
    vector<int> leftArr(n1);
    vector<int> rightArr(n2);

    // Copy data to temporary arrays
    for (int i = 0; i < n1; i++)
        leftArr[i] = arr[left + i];
    for (int j = 0; j < n2; j++)
        rightArr[j] = arr[mid + 1 + j];

    // Merge the temporary arrays back into arr[left..right]
    int i = 0;    // Initial index of first subarray
    int j = 0;    // Initial index of second subarray
    int k = left; // Initial index of merged subarray

    while (i < n1 && j < n2) {
        if (leftArr[i] <= rightArr[j]) {
            arr[k] = leftArr[i];
            i++;
        } else {
            arr[k] = rightArr[j];
            j++;
        }
        k++;
    }

    // Copy remaining elements of leftArr[], if any
    while (i < n1) {
        arr[k] = leftArr[i];
        i++;
        k++;
    }

    // Copy remaining elements of rightArr[], if any
    while (j < n2) {
        arr[k] = rightArr[j];
        j++;
        k++;
    }
}

void mergeSort(vector<int>& arr, int left, int right) {
    if (left < right) {
        // Find the middle point to divide the array into two halves
        int mid = left + (right - left) / 2;

        // Recursively sort first and second halves
        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);

        // Merge the sorted halves
        merge(arr, left, mid, right);
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

    // Perform merge sort
    mergeSort(arr, 0, n - 1);

    // End timing
    auto end = high_resolution_clock::now();

    // Calculate duration in nanoseconds
    auto duration = duration_cast<nanoseconds>(end - start);

    // Output the execution time in nanoseconds
    cout << duration.count() << endl;

    return 0;
}