.PHONY: all clean

all: result

clean:
	rm -rf *.out

kernel: kernel.c
	gcc kernel.c -lm -o kernel-output.out

core: core.c
	pgcc -acc -ta=tesla core.c -o core-output.out

result: kernel core
	time -v ./kernel-output
	time -v ./core-output
