/**
 * Calculates the sum of all the numbers in an array
 * Example:
 * arr = [1, 2, 3]
 * sum = arr[0] + arr[1] + arr[2] = 6
 */
int __attribute__ ((noinline)) kernel(int *arr, int N)
{
    // Array initialization
    int sum = 0;
    // Performs summation
    for (int i = 0; i < N; ++i)
        sum += arr[i];
    return sum;
}

int main(int argc, char **argv)
{
    int arr[4] = {1, 2, 3, 4};
    return kernel(arr, 4);
}
