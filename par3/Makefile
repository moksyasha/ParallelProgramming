SHELL := /bin/bash
USECUBLAS := 1
.PHONY: all clean

all: blas

clean:
	rm -rf *.out
	rm -rf *.o
	rm -rf *.nsys-rep

cpu: main.c
	gcc main.c -o cpu.out
	time ./cpu.out 128 0.000001 1000000

#nvcc -arch=sm_35 -rdc=true -o main.v2 time_propagation_cublas.cu -lcublas -lcublas_device -lcudadevrt
#nvcc maintest.cu -lcublas -lcurand -o blas.out

#pgcc -acc -cudalib=cublas main.c -o blas.out
gpu: main.c
	nvcc maintest.c -lcublas -lcurand -o blas.out
	time ./gpu.out 128 0.000001 1000000

blas: main.c
	pgcc -acc -cudalib=cublas main.c -o blas.out
	time CUDA_VISIBLE_DEVICES=3 ./blas.out 128 0.000001 1000000

nsy: gpu.out
	nsys profile -o testGPU -t openacc,cuda ./gpu.out 128 0.000001 101

#OMP_NUM_THREAD
