/**
 * An example of the matrix vector multiplication implemented
 * in a recursive approach.
 */
#include <stdio.h>
#define N 1024

/**
 * Main computation kernel that multiplies a square matrix A
 * and a vector x. The result is stored in vector y.
 * Reference:
 * https://www.cl.cam.ac.uk/teaching/1415/AdvAlgo/lec4_ann.pdf
 */
int __attribute__ ((noinline)) kernel(int n, int A[n][n], int *x, int *y)
{
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      y[i] += A[i][j] * x[j];
  
  return y[0];
}


int main(int argc, char **argv)
{
  // Array initialization
  int A[N][N];
  int x[N];
  int y[N] = { 0 };

  return kernel(N, A, x, y);
}
