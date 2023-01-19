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
#define N 2

#define a 2
#define b 3

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
    z[i] = a * x[i] + b * y[i];
  }
}
