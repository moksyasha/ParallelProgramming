SHELL := /bin/bash
USECUBLAS := 1
.PHONY: all clean

all: cudampi

clean:
	rm -rf *.out
	rm -rf *.o
	rm -rf *.nsys-rep

cudampi:
	nvcc -I/usr/local/openmpi/include -L/usr/local/openmpi/lib -lmpi -arch=sm_70 maincopy.cu -lm -o cudampi.out
	time mpirun -np 4 ./cudampi.out 0.000001 128 1000000

cuda:
	nvcc -I/usr/local/openmpi/include -L/usr/local/openmpi/lib -lmpi -arch=sm_70 maincopy.cu -lm -o cudampi.out
	time mpirun -np 4 ./cudampi.out 0.000001 128 1000000

profile:
	nsys profile -o prof -f true -t cuda,mpi mpirun -np 4 ./a.out 0.000001 1024 1000
