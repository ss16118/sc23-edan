/**
 * A very simple example of the multiplication of
 * 2 x 2 matrices.
 * A = [[1, 2],
 *      [2, 1]],
 * B = [[1, 0],
 *      [0, 1]]
 * C = A x B = [[1, 2], [2, 1]]
 */
#include <stdio.h>

int main(int argc, char **argv)
{
  // Array initialization
  int A[2][2] = {{1, 2}, {2, 1}};
  int B[2][2] = {{1, 0}, {0, 1}};
  int C[2][2] = { 0 };

  // Main computation kernel, multiplies matrices
  // in the most naive approach
  // C = A x B 
  for (int i = 0; i < 2; ++i)
  {
    for (int j = 0; j < 2; ++j)
    {
      for (int k = 0; k < 2; ++k)
      {
	C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}
