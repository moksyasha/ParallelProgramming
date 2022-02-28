SHELL := /bin/bash
.PHONY: all clean

all: gpu

clean:
	rm -rf *.out
	rm -rf *.o
	rm -rf *.nsys-rep

cpu: main.c
	gcc main.c -o cpu.out
	time ./cpu.out 128 0.000001 1000000

cpun: nonopt.c
	gcc nonopt.c -o cpun.out
	time ./cpun.out 256 0.000001 1000000

gpu: main.c
	pgcc -acc -ta=nvidia main.c -o gpu.out
	time CUDA_VISIBLE_DEVICES=1 ./gpu.out 128 0.000001 1000000

gpun: nonopt.c
	pgcc -acc -ta=tesla nonopt.c -o gpun.out
	time CUDA_VISIBLE_DEVICES=2 ./gpun.out 256 0.000001 1000000

nsy: gpu.out
	nsys profile -o testGPU -t openacc,cuda ./gpu.out 128 0.000001 101

#OMP_NUM_THREAD
