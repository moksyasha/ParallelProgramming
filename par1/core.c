#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <math.h>
#include <malloc.h>

#define PI 3.14159265
#define SIZE 10000080

int main()
{
    double* array = NULL;
    array = (double*)malloc((SIZE + 1) * sizeof(double));

    double result = 0.0;

    for (int i = 0; i < SIZE; i++) {
        array[i] = sin(i * (2 * PI / SIZE));
    }

    for (int i = 0; i < SIZE; i++) {
        result += array[i];
    }

    printf("CPU time: %f\n", result);

    free(array);
    return 0;
}
