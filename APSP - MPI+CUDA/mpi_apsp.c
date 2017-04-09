#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/time.h>

struct timeval startwtime, endwtime;
double seq_time;

#define THREADS_PER_BLOCK 1024
#define Inf 100000

void generate_adj(int *A, int n, float p) 
{
	int i, j;

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {

			if ( rand() > p * RAND_MAX ) A[i*n + j] = Inf;
			else						             A[i*n + j] = rand() % 100;
		}
		A[i*n + i] = 0;
	}
}

void apsp_serial(int *A, int n) 
{
	int i, j, k, temp;

	for (k = 0; k < n; k++) {
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				temp = A[i*n + k] + A[k*n + j];
				if (temp < A[i*n + j]) 
					A[i*n + j] = temp;
			}
		}
	}
}

void test(int *A, int *B, int n) 
{
  int pass = 1;
  int i, j;

  for (i = 0; i < n; i++) {
  	for (j = 0; j < n; j++) {
  		pass &= ( A[i*n + j] == B[i*n + j] );
  	}
  }

  printf("Size %d --- TEST %s\n", n, (pass) ? "PASSed" : "FAILed");
}


int main(int argc, char *argv[])
{
	//Initialize nodes
    MPI_Init(&argc, &argv);

    int self_rank, num_tasks;

    MPI_Comm_rank(MPI_COMM_WORLD, &self_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);

    int n = 1 << atoi(argv[1]);
    float p = (float) atof(argv[2]);

  	unsigned int seed = time(NULL);
    srand(seed);

    int *A = (int *) malloc(n * n * sizeof(int));
    int *B = (int *) malloc(n * n * sizeof(int));
    
	//Generate the adjacency matrix
    if (self_rank == 0)
        generate_adj(A, n, p);

    //This buffer will have the rows assigned to each process
    int *recvbuf = (int *) malloc( (n/num_tasks) * n * sizeof(int) );

    //Root splits the matrix in rows and sends them to their respective node
    MPI_Scatter(A, (n*n)/num_tasks, MPI_INT, recvbuf, (n*n)/num_tasks, MPI_INT, 0, MPI_COMM_WORLD);

    int *k_row    = (int *) malloc(n * sizeof(int));
    int *node_col = (int *) malloc((n/num_tasks) * sizeof(int));

    //dev_row stores the kth row in the GPU, dev_buf stores the block of A assigned to the node and dev_col stores the kth column
    int *dev_row, *dev_buf, *dev_col;
    gpu_malloc(&dev_row, n);
    gpu_malloc(&dev_col, n/num_tasks);
    gpu_malloc(&dev_buf, (n/num_tasks) * n);
    
    if(self_rank == 0)
        gettimeofday (&startwtime, NULL);

    cpu_to_gpu_copy(dev_buf, recvbuf, (n/num_tasks) * n);

    int k, j, sender_task, offset;
    int blk = (n*n/num_tasks) / THREADS_PER_BLOCK + ( (n*n/num_tasks) < THREADS_PER_BLOCK ? 1 : 0 );


    for (k = 0; k < n; k++) {
    	sender_task = (int) k*num_tasks/n;
    	offset = k % (n/num_tasks);

    	 if (self_rank == sender_task) {
		    gpu_get_row(dev_buf, dev_row, k_row, offset, n, (n/THREADS_PER_BLOCK) + (n < THREADS_PER_BLOCK ? 1 : 0));
    	 }

         //Copy the column elemnets from the adjacency matrix that is soon to be changed to an array
	     gpu_copy_column(dev_buf, dev_col, k, n/num_tasks, n, blk);

    	 //Send k-th row to everyone
    	 MPI_Bcast(k_row, n, MPI_INT, sender_task, MPI_COMM_WORLD);

    	 cpu_to_gpu_copy(dev_row, k_row, n);

    	 apsp(dev_buf, dev_row, dev_col, n, num_tasks, blk);

    	 MPI_Barrier(MPI_COMM_WORLD);
    }

    gpu_to_cpu_copy(recvbuf, dev_buf, (n/num_tasks) * n);

    MPI_Gather(recvbuf, (n*n)/num_tasks, MPI_INT, B, (n*n)/num_tasks, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();

    if (self_rank == 0) {
        gettimeofday (&endwtime, NULL);

        seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
            + endwtime.tv_sec - startwtime.tv_sec);

        printf("%f\n", seq_time);

        gettimeofday(&startwtime, NULL);

        apsp_serial(A, n);

        gettimeofday(&endwtime, NULL);

        seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
            + endwtime.tv_sec - startwtime.tv_sec);

        printf("%f\n", seq_time);

        //test(A, B, n);
        free(A);
        free(B);
    }

}
