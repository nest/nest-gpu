#include <config.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include "cuda_error.h"

unsigned int *curand_int(curandGenerator_t &gen, size_t n)
{
  unsigned int *dev_data;
  // Allocate n integers on host
  unsigned int *host_data = new unsigned int[n];
  
  // Allocate n integers on device
  CUDA_CALL(cudaMalloc((void **)&dev_data, n*sizeof(unsigned int)));
  // Create pseudo-random number generator

  // Generate n integers on device
  CURAND_CALL(curandGenerate(gen, dev_data, n));
  //cudaDeviceSynchronize();
  // Copy device memory to host
  CUDA_CALL(cudaMemcpy(host_data, dev_data, n*sizeof(unsigned int),
                       cudaMemcpyDeviceToHost));
  // Cleanup
  CUDA_CALL(cudaFree(dev_data));
  
  return host_data;
}

float *curand_uniform(curandGenerator_t &gen, size_t n)
{
  float *dev_data;
  // Allocate n floats on host
  float *host_data = new float[n];
  
  // Allocate n floats on device
  CUDA_CALL(cudaMalloc((void **)&dev_data, n*sizeof(float)));
  // Create pseudo-random number generator

  // Generate n integers on device
  CURAND_CALL(curandGenerateUniform(gen, dev_data, n));
  //cudaDeviceSynchronize();
  // Copy device memory to host
  CUDA_CALL(cudaMemcpy(host_data, dev_data, n*sizeof(float),
                       cudaMemcpyDeviceToHost));
  // Cleanup
  CUDA_CALL(cudaFree(dev_data));
  
  return host_data;
}

float *curand_normal(curandGenerator_t &gen, size_t n, float mean,
		     float stddev)
{
  size_t n1 = ( (n % 2) == 0 ) ? n : n + 1; // round up to multiple of 2
  float *dev_data;
  // Allocate n floats on host
  float *host_data = new float[n];
  
  // Allocate n1 floats on device
  CUDA_CALL(cudaMalloc((void **)&dev_data, n1*sizeof(float)));
  // Create pseudo-random number generator

  // Generate n1 integers on device
  //printf("curandGenerateNormal n1: %d\tmean: %f\tstd: %f\n", (int)n1, mean,
  //	 stddev);
  CURAND_CALL(curandGenerateNormal(gen, dev_data, n1, mean, stddev));
  //cudaDeviceSynchronize();
  // Copy device memory to host
  CUDA_CALL(cudaMemcpy(host_data, dev_data, n*sizeof(float),
                       cudaMemcpyDeviceToHost));
  // Cleanup
  CUDA_CALL(cudaFree(dev_data));
  
  return host_data;
}

