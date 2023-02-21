/**
 * Calculates prefix sum of an array.
 * Example:
 * x      = [1, 2, 3, 4,  5,  6]
 * result = [1, 3, 6, 10, 15, 21]
 */
#include <stdio.h>
#define N 256

/**
 * Main computation kernel that calculates the prefix sum
 * of the given array of n elements. The results are
 * stored in the given array `res`.
 */
int __attribute__ ((noinline)) kernel(int *arr, int *res, int n)
{
  res[0] = arr[0];
  for (int i = 1; i < n; ++i)
    res[i] = arr[i] + res[i - 1];

  return res[0];
}

int main(int argc, char **argv)
{
  // Array initialization
  int arr[N];
  int res[N] = { 0 };

  for (int i = 0; i < N; ++i)
    arr[i] = i;
  
  return kernel(arr, res, N);
}
