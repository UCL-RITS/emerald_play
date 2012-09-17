// Simple 1-Node Many-GPU-Core Monte-Carlo pi generator

// On each core, generate a random x- and y- coordinate in [0,1)
// Sum-square them, determine if <1
// Reduce on mean
// Result is pi
#include "curand_kernel.h"
#include "stdio.h"

const int thread_count=256;
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
	sample samples_host[thread_count];
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

void generate_sample(float *result)
{
	generate_samples <<< 1, tpb >>> ( samples_device);	
	reduce_samples <<<1,tpb>>> (samples_device,result_device);
	// Retrieve
	handle_error(	cudaMemcpy(samples_host,samples_device,thread_count*sizeof( sample ),cudaMemcpyDeviceToHost),"Retrieve device samples");
	handle_error(   cudaMemcpy(result,result_device,sizeof(float),cudaMemcpyDeviceToHost),"Retrieve result");
	//for (int i=0;i<thread_count;i++){
	//	display(&samples_host[i]);
	//}
}

int main( int argc, char** argv) 
{
	
	float result_host=0.0;
	float result_total=0.0;
	for (int i=0;i<thread_count;i++){
		
		init_zero(&samples_host[i]);
	}
    handle_error(cudaMalloc ( &samples_device, thread_count*sizeof( sample ) ),"Allocate device samples");
	handle_error(cudaMalloc (&result_device,sizeof(float)),"Allocate result");
    
	init_device_sample <<< 1, tpb >>> ( samples_device, time(NULL) );
	
	for (int cycle=0;cycle<cycle_count;cycle++)
	{
		generate_sample(&result_host);
		result_total+=result_host;
	}
	printf("Result: %f\n",4.0*result_total/((float) thread_count*cycle_count));
	cudaFree(result_device);
	cudaFree(samples_device);
    return 0;
}