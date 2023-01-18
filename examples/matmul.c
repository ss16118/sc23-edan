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

int main(int argc, char **argv)
{
  // Array initialization
  int A[N][N] = {{1, 2}, {2, 1}};
  int B[N][N] = {{1, 0}, {0, 1}};
  int C[N][N] = { 0 };

  // Main computation kernel, multiplies matrices
  // in the most naive approach
  // C = A x B 
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      for (int k = 0; k < N; ++k)
      {
	C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}
