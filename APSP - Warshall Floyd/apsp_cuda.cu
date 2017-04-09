#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>

struct timeval startwtime, endwtime;
double seq_time;

#define THREADS_PER_BLOCK 1024

//First kernel - one matrix cell per GPU thread without using shared memory
__global__ void apsp_1(int *A, int n, int k) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int j = index % n;
	int i = (index - j) / n;

	if (A[index] == -1 && (A[i*n + k] == -1 || A[k*n + j] == -1))
		;
	else if (A[index] == -1 && (A[i*n + k] != -1 && A[k*n + j] != -1))
		A[index] = A[i*n + k] + A[k*n + j];
	else if (A[i*n + k] == -1 || A[k*n + j] == -1)
		;
	else if (A[index] > A[i*n + k] + A[k*n + j])
		A[index] = A[i*n + k] + A[k*n + j];

}

//Second kernel - one matrix cell per GPU thread using shared memory
__global__ void apsp_2(int *A, int n, int k) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int j = index % n;
	int i = (index - j) / n;

	extern __shared__ int sh[];
	

	//Load the k-th row on the shared memory
	int m;
	for (m = 0; m < (n / THREADS_PER_BLOCK) + ((n < THREADS_PER_BLOCK) ? 1 : 0); m++) {
		if (threadIdx.x + m * THREADS_PER_BLOCK < n) {
			sh[threadIdx.x + m * THREADS_PER_BLOCK] = A[k*n + threadIdx.x + m * THREADS_PER_BLOCK];
		}
	}
	__syncthreads();

	int cell = A[index];
	int cell2 = A[i*n + k];

	if (cell == -1 && (cell2 == -1 || sh[j] == -1))
		;
	else if (cell == -1 && (cell2 != -1 && sh[j] != -1))
		cell = cell2 + sh[j];
	else if (cell2 == -1 || sh[j] == -1)
		;
	else if (cell > cell2 + sh[j])
		cell = cell2 + sh[j];

	A[index] = cell;
}


//Each thread handles a row of A
//We check A[i][j], A[i][k] and A[k][j]
//A[i][k] is on the same row so we can retrieve it from the shared memory
__global__ void apsp_3(int *A, int n, int k) {
	int row = threadIdx.x;

	extern __shared__ int sh[];

	//Load k-th row in shared memory
	int m;
	for (m = 0; m < (n / THREADS_PER_BLOCK) + ((n < THREADS_PER_BLOCK) ? 1 : 0); m++) {
		if (threadIdx.x + m * THREADS_PER_BLOCK < n) {
			sh[threadIdx.x + m * THREADS_PER_BLOCK] = A[k*n + threadIdx.x + m * THREADS_PER_BLOCK];
		}
	}
	__syncthreads();

	int cell;
	int cell2 = A[row*n + k];

	//Each thread handles a single row of the matrix
	//We update each cell in the row below
	int col;
	if (blockIdx.x * blockDim.x + threadIdx.x < n) {

		for (col = 0; col < n; col++) {
			cell = A[row*n + col];

			if (cell == -1 && (cell2 == -1 || sh[col] == -1))
				continue;
			else if (cell == -1 && (cell2 != -1 && sh[col] != -1))
				cell = cell2 + sh[col];
			else if (cell2 == -1 || sh[col] == -1)
				continue;
			else if (cell > cell2 + sh[col])
				cell = cell2 + sh[col];

			A[row*n + col] = cell;
		}
	}
}

void apsp_serial(int *A, int n) {
	int i, j, k;

	for (k = 0; k < n; k++) {
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {

				if (A[i*n + j] == -1 && (A[i*n + k] == -1 || A[k*n + j] == -1))
					continue;
				else if (A[i*n + j] == -1 && (A[i*n + k] != -1 && A[k*n + j] != -1))
					A[i*n + j] = A[i*n + k] + A[k*n + j];
				else if (A[i*n + k] == -1 || A[k*n + j] == -1)
					continue;
				else if (A[i*n + j] > A[i*n + k] + A[k*n + j])
					A[i*n + j] = A[i*n + k] + A[k*n + j];
			}
		}
	}
}

void generate_adj(int *A, int n, float p) {
	int i, j;

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {

			if ( rand() > p * RAND_MAX ) A[i*n + j] = -1;
			else						 A[i*n + j] = rand() % 10;
		}
		A[i*n + i] = 0;
	}
}

void test(int *A, int *B, int n) {
  int pass = 1;
  int i, j;

  for (i = 0; i < n; i++) {
  	for (j = 0; j < n; j++) {
  		pass &= ( A[i*n + j] == B[i*n + j] );
  	}
  }

  printf(" TEST %s\n",(pass) ? "PASSed" : "FAILed");
}

int main(int argc, char **argv) {
	if (argc != 3) {
		printf("Function needs two arguments!\n");
		exit(1);
	}

	int n = 1 << atoi(argv[1]);
	float p = (float) atof(argv[2]);
	//printf("%d %d %f", n, THREADS_PER_BLOCK, p);
	int *A  = (int *) malloc(n*n * sizeof(int));

	int *host_A1, *host_A2, *host_A3;
	int *dev_A;

	host_A1 = (int *) malloc(n*n * sizeof(int));
	host_A2 = (int *) malloc(n*n * sizeof(int));
	host_A3 = (int *) malloc(n*n * sizeof(int));
	cudaMalloc(&dev_A, n*n* sizeof(int));

	unsigned int seed = time(NULL);
	srand(seed);

	generate_adj(A, n, p);

	//
	//RUN 1
	//
	gettimeofday (&startwtime, NULL);

	cudaMemcpy(dev_A, A, n*n * sizeof(int), cudaMemcpyHostToDevice);

	int k;

	//Kernel 1
	int blk = (n * n) / THREADS_PER_BLOCK;
	for (k = 0; k < n; k++) {
		apsp_1<<<blk, THREADS_PER_BLOCK>>>(dev_A, n, k);
	}

	cudaMemcpy(host_A1, dev_A, n*n * sizeof(int), cudaMemcpyDeviceToHost);

	gettimeofday (&endwtime, NULL);

	seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
		      + endwtime.tv_sec - startwtime.tv_sec);

  	printf("%f ", seq_time);

  	//
  	//RUN 2
  	//
  	gettimeofday (&startwtime, NULL);

	cudaMemcpy(dev_A, A, n*n * sizeof(int), cudaMemcpyHostToDevice);

	//Kernel 2
	int blk2 = (n * n) / THREADS_PER_BLOCK;
	for (k = 0; k < n; k++) {
		apsp_2<<<blk2, THREADS_PER_BLOCK, n>>>(dev_A, n, k);
	}

	cudaMemcpy(host_A2, dev_A, n*n * sizeof(int), cudaMemcpyDeviceToHost);

	gettimeofday (&endwtime, NULL);

	seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
		      + endwtime.tv_sec - startwtime.tv_sec);

  	printf("%f ", seq_time);

  	//
  	//RUN 3
  	//
  	gettimeofday (&startwtime, NULL);

	cudaMemcpy(dev_A, A, n*n * sizeof(int), cudaMemcpyHostToDevice);

	//Kernel 3
	int blk3 = (n / THREADS_PER_BLOCK) + ((n < THREADS_PER_BLOCK) ? 1 : 0);
	for (k = 0; k < n; k++) {
		apsp_3<<<blk3, THREADS_PER_BLOCK, n>>>(dev_A, n, k);
	}

	cudaMemcpy(host_A3, dev_A, n*n * sizeof(int), cudaMemcpyDeviceToHost);

	gettimeofday (&endwtime, NULL);

	seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
		      + endwtime.tv_sec - startwtime.tv_sec);

  	printf("%f ", seq_time);

  	//SERIAL TEST

  	gettimeofday(&startwtime, NULL);

  	apsp_serial(A, n);

  	gettimeofday(&endwtime, NULL);

  	seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
		      + endwtime.tv_sec - startwtime.tv_sec);

  	printf("%f\n", seq_time);

  	//TESTS
 	test(A, host_A1, n);
 	test(A, host_A2, n);
 	test(A, host_A3, n);

  	return 0;
}
