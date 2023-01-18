/**
 * Calculates the sum of all the numbers in an array
 * Example:
 * arr = [1, 2, 3]
 * sum = arr[0] + arr[1] + arr[2] = 6
 */

#include <stdio.h>
#define N 5
int main(int argc, char **argv)
{
  // Array initialization
  int arr[N] = {1, 2, 3, 4, 5};
  int sum = 0;
  // Performs summation
  for (int i = 0; i < N; ++i)
    sum += arr[i];
}
