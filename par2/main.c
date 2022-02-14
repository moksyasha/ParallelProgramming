#include <stdlib.h>
#include <stdio.h> 

#define max(x, y) ((x) > (y) ? (x) : (y) )
#define min(x, y) ((x) < (y) ? (x) : (y) )

int main(int argc, char *argv[]){
    //size of matrix, tolerance, iter_max
    int size = strtol(argv[1], NULL, 10);
    double tol = atof(argv[2]);
    int iter_max = strtol(argv[1], NULL, 10);

    //Known point values
    double pt11 = 10;
    double pt12 = 20;
    double pt22 = 30;
    double pt21 = 20;

    double A[size][size];
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            //interpolate the values ​​of the auxiliary points R1 and R2 by OX
            double R1 = pt11 * (size - j) / size + pt21 * j / size;
            double R2 = pt12 * (size - j) / size + pt22 * j / size;

            A[i][j] = R1 * (size - i) / size + R2 * i / size;
        }
    }
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            printf("%f ", A[i][j]);
        }
        printf("\n");
    }

    double Anew[size][size];

    //#pragma acc data copy(A) create(Anew)
    {
    int iter = 0;
    double err = 10;
    

    while ((err > tol) && (iter < iter_max)){
        iter++;
        err = 0;

        //#pragma acc kernels
    {
        for (int j = 1; j < size; j++){
            for (int i = 1; i < size; i++){
                Anew[i][j] = 0.25 * (A[i+1][j] + A[i-1][j] + A[i][j-1] + A[i][j+1]);
                err = max(err, Anew[i][j] - A[i][j]);
            }
        }

        for (int i = 0; i < size; i++){
            for (int j = 0; j < size; j++){
                A[i][j] = Anew[i][j];
            }
        }
    }

        if ((iter % 100 == 0) || (iter == 1))
            printf("%d : %f\n", iter, err);
    }
    }

    return 0;

    // iter=0
    // do while ( err > tol .and. iter < iter_max )
    // iter = iter +1
    // err=0._fp_kind
    // do j=1,m
    // do i=1,n
    // Anew(i,j) = .25_fp_kind *( A(i+1,j ) + A(i-1,j ) &
    // +A(i ,j-1) + A(i ,j+1))
    // err = max( err, Anew(i,j)-A(i,j))
    // end do
    // end do

    // A = Anew
    // IF(mod(iter,100)==0 .or. iter == 1) print *, iter, err
    // end do

    return 0;
}
