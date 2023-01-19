/**
 * An example of addition of two 2D vectors.
 * Example:
 * x = [1, 2]
 * y = [3, 4]
 * z = x + y = [1 + 3, 2 + 4] = [4, 6]
 */
#include <stdio.h>
#define N 2


int main(int argc, char **argv)
{
  // Vector initialization
  int x[N] = { 1, 2 };
  int y[N] = { 3, 4 };
  int z[N] = { 0 };


  // Main computation kernel, computes the result of
  // vector addition
  for (int i = 0; i < N; ++i)
  {
    z[i] = x[i] + y[i];
  }
}
