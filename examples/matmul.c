/**
 * A very simple example of the multiplication of
 * 2 x 2 matrices.
 * Example:
 * A = [[1, 2],
 *      [2, 1]],
 * B = [[1, 0],
 *      [0, 1]]
 * C = A x B = [[1, 2], [2, 1]]
 */
#include <stdio.h>
#define N 2

/**
 * Main computation kernel that multiplies two square matrices
 * A and B of dimension n x n, and stores the result in matrix C.
 */
int __attribute__ ((noinline)) kernel(int n, int A[n][n], int B[n][n], int C[n][n])
{
  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j < n; ++j)
    {
      for (int k = 0; k < n; ++k)
      {
	C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return C[0][0];
}


int main(int argc, char **argv)
{
  // Array initialization
  int A[N][N] = {{1, 2}, {2, 1}};
  int B[N][N] = {{1, 0}, {0, 1}};
  int C[N][N] = { 0 };

  return kernel(N, A, B, C);
}
