all:
	mpicc -c -O3 mpi_apsp.c -o mpi.o
	nvcc -c -O3 apsp_cuda.cu -o cuda.o
	mpicc mpi.o cuda.o -lcudart -L/usr/local/cuda/lib64 -o erg4
	rm mpi.o
	rm cuda.o
