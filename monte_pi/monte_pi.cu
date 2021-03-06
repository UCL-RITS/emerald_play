// Simple 1-Node Many-GPU-Core Monte-Carlo pi generator

// On each core, generate a random x- and y- coordinate in [0,1)
// Sum-square them, determine if <1
// Reduce on mean
// Result is pi
#include "curand_kernel.h"
#include "stdio.h"
#include <mpi.h>

const int thread_count=32;
const int cycle_count=10000;

// Pseudo-object sample ---------------

struct sample {
	int id;
	curandState randState;
	float x;
	float y;
	int within;
};

__device__ void init_with_seed(sample *self,unsigned long seed)
{
	self->id = threadIdx.x;
    curand_init ( seed, self->id, 0, &self->randState );
	self->x=0;
	self->y=0;
	self->within=0;
}

__device__ void generate(sample * self)
{
    self->x= curand_uniform( &self->randState );
	self->y= curand_uniform( &self->randState );
	self->within= (pow(self->x,2)+pow(self->y,2))<1.0;
}

void init_zero( sample *self)
{
	self->id = 0;
	self->x=0;
	self->y=0;
	self->within=0;
}

void display( sample * asample)
{
	printf("%d : %f, %f (%d)\n",asample->id,asample->x,asample->y,asample->within);
}

//--------------- Pseudo-object CUDA sample array ---

	// Prepare
	dim3 tpb(thread_count,1,1);
	sample *samples_device;
	float *result_device;

__global__ void init_device_sample ( sample *samples, unsigned long seed )
{
	sample *self=&samples[threadIdx.x];
	init_with_seed(self,seed);
}

__global__ void generate_samples (sample * samples)
{
	sample *self=&samples[threadIdx.x];
	generate(self);
}

__global__ void reduce_samples (sample * samples, float * result){
	__shared__ float cache[thread_count];
	int thread_id=threadIdx.x;
	cache[thread_id]=samples[thread_id].within;
	int reduction_index=thread_count/2;
	__syncthreads();
	while (reduction_index!=0){
		if (thread_id<reduction_index){
			cache[thread_id]+=cache[thread_id+reduction_index];
		}
		__syncthreads();
		reduction_index/=2;
	}
	if (thread_id==0) {
		*result=cache[0];
	}
}

void handle_error( cudaError_t error, char* message)
{
	if(error!=cudaSuccess) { 
		fprintf(stderr,"ERROR: %s : %s\n",message,cudaGetErrorString(error)); 
		exit(-1); 
	}
}

void initialise_samples(int rank)
{
	handle_error(cudaMalloc ( &samples_device, thread_count*sizeof( sample ) ),"Allocate device samples");
	handle_error(cudaMalloc (&result_device,sizeof(float)),"Allocate result");
	init_device_sample <<< 1, tpb >>> ( samples_device, time(NULL)*rank );
}

void generate_sample(float *result)
{
	generate_samples <<< 1, tpb >>> ( samples_device);	
	reduce_samples <<<1,tpb>>> (samples_device,result_device);
	handle_error(   cudaMemcpy(result,result_device,sizeof(float),cudaMemcpyDeviceToHost),"Retrieve result");
}

int main( int argc, char** argv) 
{
	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	float result_host=0.0;
	float result_total=10.0;
	float result_mpi=0.0;

	initialise_samples(rank);
	
	for (int cycle=0;cycle<cycle_count;cycle++)
	{
		generate_sample(&result_host);
		result_total+=result_host;
	}
	
	printf("Partial Result on rank %i: %f\n",rank,4.0*result_total/((float) thread_count*cycle_count));
	cudaFree(result_device);
	cudaFree(samples_device);
	// Reduce the MPI-Samples
	
	MPI_Reduce(&result_total,&result_mpi,1,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
	if (rank==0){
		printf("Final Result over %i mpi processes: %f\n",size,4.0*result_mpi/((float) thread_count*cycle_count*size));
	}
	MPI_Finalize();
    return 0;
}