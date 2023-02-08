/**
 * Calculates the sum of all the numbers in an array
 * using OpenMP reduction.
 * Example:
 * arr = [1, 2, 3]
 * sum = arr[0] + arr[1] + arr[2] = 6
 */
#include <stdio.h>
#include <omp.h>
#define N 32

int __attribute__ ((noinline)) kernel(int *arr, int n)
{   
    int i, sum = 0;
    #pragma omp parallel
    {
        #pragma omp for reduction(+:sum)
        for (i = 0; i < n; ++i)
            sum += arr[i];
    }
    return sum;
}

int main(int argc, char **argv)
{
  int arr[N];
  for (int i = 0; i < N; ++i)
    arr[i] = i;
  return kernel(arr, N);
}
