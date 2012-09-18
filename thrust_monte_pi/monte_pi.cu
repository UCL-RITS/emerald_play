// Simple 1-Node Many-GPU-Core Monte-Carlo pi generator

// On each core, generate a random x- and y- coordinate in [0,1)
// Sum-square them, determine if <1
// Reduce on mean
// Result is pi
#include "curand_kernel.h"
#include <iostream>
#include <stdio.h>
#include <mpi.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/tuple.h>

const int thread_count=32;
const int cycle_count=2;

// Pseudo-object sample ---------------

class Sample {
public:
	__device__ Sample()
	: id(0),x(0),y(0)
	{
	    
	}

	__device__ void seed(unsigned long seed, int index)
	{
		id=index;
		curand_init ( seed, id, 0, &randState );
	}

	__device__ void generate()
	{
	    x= curand_uniform( &randState );
		y= curand_uniform( &randState );
	}

	__host__ __device__ bool within()
	{
		return (pow(x,2)+pow(y,2))<1.0;
	}
	void __host__ __device__ display()
	{
		// Use C-Style io because CUDA 2.0 doesn't support C++ io on devices
		printf("%d : %f, %f (%d)\n",id,x,y,within());
	}
private:
	int id;
	curandState randState;
	double x;
	double y;
};



//--------------- Pseudo-object CUDA sample array ---
class Device_sampler
{
public:
	Device_sampler(int count,int rank)
	: samples(count), rank(rank)
	{
		// seed the samples from their thread number
		// make a zip of samples and their number
		thrust::for_each(
			thrust::make_zip_iterator(thrust::make_tuple(samples.begin(),thrust::make_counting_iterator(0))),
			thrust::make_zip_iterator(thrust::make_tuple(samples.end(),thrust::make_counting_iterator(count))),
			bind_seed(rank)
			);
	
	}
	
	struct bind_seed 
	{
		bind_seed(int rank):rank(rank){}
		__device__ void operator()(thrust::tuple<Sample&,int> t)
		{
			thrust::get<0>(t).seed(rank,thrust::get<1>(t));
		}
		int rank;
	};
	
	// std::mem_fun_ref does not bind as a device function
	// so we need to create our own lambda
	struct bind_generate
	{
		__device__ bool operator()(Sample &sample) const
		{
			sample.generate();
			return sample.within();
		}
	};
		
	double result(){
		return 4.0*thrust::transform_reduce(
			samples.begin(),samples.end(),
			bind_generate(),
			0,thrust::plus<double>())
		/static_cast<double>(samples.size());
	}
private:
	thrust::device_vector<Sample> samples;
	int rank;
};

int main( int argc, char** argv) 
{
	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	double result_total=0.0;
	double result_mpi=0.0;
	
	Device_sampler device_sampler(thread_count,rank);
	for (int cycle=0;cycle<cycle_count;cycle++)
	{
		double result=device_sampler.result();
		printf("Partial Result %i on rank %i: %f\n",cycle,rank,result);
		result_total+=result;
	}
	
	printf("Partial Result on rank %i: %f\n",rank,result_total/static_cast<double>(cycle_count));
	// Reduce the MPI-Samples
	
	MPI_Reduce(&result_total,&result_mpi,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	if (rank==0){
		printf("Final Result over %i mpi processes: %f\n",size,result_mpi/static_cast<double>(cycle_count*size));
	}
	MPI_Finalize();
    return 0;
}