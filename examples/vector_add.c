/**
 * An example of addition of two 2D vectors.
 * Example:
 * x = [1, 2]
 * y = [3, 4]
 * z = x + y = [1 + 3, 2 + 4] = [4, 6]
 */
#include <stdio.h>
#define N 1024

/**
 * Main computation kernel that calculates vector addition
 * between two vectors a and b. The result is stored in variable c.
 */
int __attribute__ ((noinline)) kernel(int n, int *a, int *b, int *c)
{
  for (int i = 0; i < n; ++i)
    c[i] = a[i] + b[i];

  return c[0];
}


int main(int argc, char **argv)
{
  // Vector initialization
  int x[N];
  int y[N];
  int z[N];
  for (int i = 0; i < N; ++i)
  {
    x[i] = i;
    y[i] = i;
  }
  
  return kernel(N, x, y ,z);
}
