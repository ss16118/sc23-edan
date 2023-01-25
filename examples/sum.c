/**
 * Calculates the sum of all the numbers in an array
 * Example:
 * arr = [1, 2, 3]
 * sum = arr[0] + arr[1] + arr[2] = 6
 */
#include <stdio.h>
#define N 4

/**
 * Main computation kernel that calculates the sum
 * of all elements in a given array.
 */
int __attribute__ ((noinline)) kernel(int *arr, int n)
{
    // Array initialization
    int sum = 0;
    // Performs summation
    for (int i = 0; i < n; ++i)
        sum += arr[i];
    return sum;
}

int main(int argc, char **argv)
{
  int arr[N] = {1, 2, 3, 4};
  return kernel(arr, N);
}
