SHELL := /bin/bash
USECUBLAS := 1
.PHONY: all clean

all: cuda

clean:
	rm -rf *.out
	rm -rf *.o
	rm -rf *.nsys-rep

cpu: maintest.c
	gcc maintest.c -o cpu.out
	time ./cpu.out 4 0.000001 1000000

cuda: main.cu
	nvcc -arch=sm_70 main.cu -lm -o cuda.out
	time ./cuda.out 128 0.000001 1000000  

test: maintest.c
	gcc maintest.c -o maintest.out
	time ./maintest.out

nsy: gpu.out
	nsys profile -o testGPU -t cuda ./gpu.out

#OMP_NUM_THREAD
