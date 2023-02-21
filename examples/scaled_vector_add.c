/**
 * An example of addition of two scaled 2D vectors.
 * Example:
 * a = 2
 * b = 3
 * x = [1, 2]
 * y = [3, 4]
 * z = a * x +  b * y = 2 * [1, 2] + 3 * [3, 4] = [11, 16]
 */
#include <stdio.h>
#define N 1024

#define A 2
#define B 3


/**
 * Main computation kernel that calculates scaled vector addition
 * between two vectors a.x and b.y. The result is stored in variable z.
 */
int __attribute__ ((noinline)) kernel(int n, int *x, int *y, int *z, int a, int b)
{
  for (int i = 0; i < n; ++i)
    z[i] = a * x[i] + b * y[i];

  return z[0];
}


int main(int argc, char **argv)
{
  // Vector initialization
  int x[N];
  int y[N];
  int z[N] = { 0 };

  for (int i = 0; i < N; ++i)
  {
    x[i] = i;
    y[i] = i;
  }

  return kernel(N, x, y, z, A, B);
}
