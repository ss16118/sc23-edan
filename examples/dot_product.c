/**
 * An example of dot product of two 2D vectors.
 * Examples:
 * x = [1, 2]
 * y = [3, 4]
 * z = x . y = 1 * 3 + 2 * 4= 11
 */
#include <stdio.h>
#define N 1024

/**
 * Main computation kernel that calculates the dot product
 * of two vectors.
 */
int __attribute__ ((noinline)) kernel(int n, int *a, int *b)
{
  int res = 0;
  for (int i = 0; i < n; ++i)
    res += a[i] * b[i];
    
  return res;
}



int main(int argc, char **argv)
{
  // Vector initialization
  int x[N];
  int y[N];
  for (int i = 0; i < N; ++i)
  {
    x[i] = i;
    y[i] = i;
  }

  return kernel(N, x, y);
}
