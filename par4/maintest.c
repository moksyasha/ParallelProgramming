#include <stdlib.h>
#include <stdio.h> 
#include <malloc.h>

int main(int argc, char *argv[]){
    //size of matrix, tolerance, iter_max
    int size = strtol(argv[1], NULL, 10);

    float tol = atof(argv[2]);
    int iter_max = strtol(argv[3], NULL, 10);

    float* A = (float*)malloc(((size + 2)*(size+2)) * sizeof(float));
    float* Anew = (float*)malloc(((size + 2)*(size+2)) * sizeof(float));

    int iter = 0;
    float error = 100.0;
    float *ptr;
    float add_gradient = 10.0 / (size + 1); 
    

    for (int i = 0; i < size + 2; i++){
        A[i*(size+2) + 0] = 10 + add_gradient*i;
        A[i] = 10 + add_gradient*i;
        A[(size+1)*(size+2) + i] = 20 + add_gradient*i;
        A[i*(size+2)+size+1] = 20 + add_gradient*i;

        Anew[i*(size+2) + 0] = A[i*(size+2) + 0];
        Anew[i] = A[i];
        Anew[(size+1)*(size+2) + i] = A[(size+1)*(size+2) + i];
        Anew[i*(size+2)+size+1] = A[i*(size+2)+size+1];
    }

    for (int j = 1; j < size + 1; j++){
        for (int i = 1; i < size + 1; i++){
            Anew[i*(size+2)+j] = 0.25 * (A[(i+1)*(size+2)+j] + A[(i-1)*(size+2)+j] + A[i*(size+2)+j-1] + A[i*(size+2)+j+1]);
            
        }
    }

    for (int i = 0; i < size+2; i++){
        for (int j = 0; j < size + 2; j++){
            printf("%.5f ", Anew[i*(size+2)+j]);
        }
        printf("\n");
    }


    return 0;
}
