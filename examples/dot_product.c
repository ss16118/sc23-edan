/**
 * An example of dot product of two 2D vectors.
 * Examples:
 * x = [1, 2]
 * y = [3, 4]
 * z = x . y = 1 * 3 + 2 * 4= 11
 */
#include <stdio.h>
#define N 2
int main(int argc, char **argv)
{
  // Vector initialization
  int x[N] = { 1, 2 };
  int y[N] = { 3, 4 };
  int z = 0;
  
  // Main computation kernel, computes the dot product of the
  // two vectors
  for (int i = 0; i < N; ++i)
  {
    z += x[i] * y[i];
  }
}
