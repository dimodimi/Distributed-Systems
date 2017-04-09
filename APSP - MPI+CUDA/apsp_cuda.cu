#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 1024


__global__ void apsp_new(int *A, int *k_row, int *k_col, int n, int num_tasks)
{
	int cell = blockIdx.x * blockDim.x + threadIdx.x;
	int j = cell % n;
	int i = (cell - j)/n;
	int temp;
	
	if (cell < (n*n)/num_tasks) {
		temp = k_row[j] + k_col[i];
		if (temp < A[i*n + j])
			A[i*n + j] = temp;
	}
}

__global__ void move_row(int *A, int *k_row, int offset, int n)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j < n) {
		k_row[j] = A[offset*n + j];
	}
}

__global__ void copy_col(int *A, int *dev_col, int k, int size, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		dev_col[i] = A[i*n + k];
	}
}

extern "C" void cpu_to_gpu_copy(int *dev_buf, int *recvbuf, int size) {
	cudaMemcpy(dev_buf, recvbuf, size * sizeof(int), cudaMemcpyHostToDevice);
}

extern "C" void gpu_to_cpu_copy(int *host_buf, int *dev_buf, int size) {
	cudaMemcpy(host_buf, dev_buf, size * sizeof(int), cudaMemcpyDeviceToHost);
}

extern "C" void gpu_malloc(int **array, int size) {
	cudaMalloc(array, size * sizeof(int));
}

extern "C" void gpu_get_row(int *dev_buf, int *dev_row, int *k_row, int offset, int n, int blk) {
	move_row<<<blk, THREADS_PER_BLOCK>>>(dev_buf, dev_row, offset, n);
	cudaMemcpy(k_row, dev_row, n * sizeof(int), cudaMemcpyDeviceToHost);
}

extern "C" void gpu_copy_column(int *dev_buf, int *dev_col, int k, int size, int n, int blk) {
	copy_col<<<blk, THREADS_PER_BLOCK>>>(dev_buf, dev_col, k, size, n);
}

extern "C" void apsp(int *dev_buf, int *dev_row, int *dev_col, int n, int num_tasks, int blk) 
{
	apsp_new<<<blk, THREADS_PER_BLOCK>>>(dev_buf, dev_row, dev_col, n, num_tasks);
}
