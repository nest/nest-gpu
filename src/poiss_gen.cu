/*
 *  This file is part of NESTGPU.
 *
 *  Copyright (C) 2021 The NEST Initiative
 *
 *  NESTGPU is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NESTGPU is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NESTGPU.  If not, see <http://www.gnu.org/licenses/>.
 *
 */





#include <config.h>
#include <cmath>
#include <iostream>
//#include <stdio.h>
#include <stdint.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>

#include "utilities.h"
#include "nestgpu.h"
#include "neuron_models.h"
#include "poiss_gen.h"
#include "poiss_gen_variables.h"
#include "copass_kernels.h"
#include "connect.h"

extern __constant__ double NESTGPUTime;
extern __constant__ float NESTGPUTimeResolution;
extern __constant__ NodeGroupStruct NodeGroupArray[];
extern __device__ int16_t *NodeGroupMap;

namespace poiss_conn
{
  typedef uint key_t;
  typedef regular_block_array<key_t> array_t;
  key_t **d_poiss_key_array_data_pt;
  array_t *d_poiss_subarray;
  int64_t *d_poiss_num;
  int64_t *d_poiss_sum;
  key_t *d_poiss_thresh;
};

// max delay functor
struct MaxDelay
{
  //template <typename T>
    __device__ __forceinline__
    //T operator()(const T &source_delay_a, const T &source_delay_b) const {
    uint operator()(const uint &source_delay_a, const uint &source_delay_b)
      const {
      uint i_delay_a = source_delay_a & PortMask;
      uint i_delay_b = source_delay_b & PortMask;
        return (i_delay_b > i_delay_a) ? i_delay_b : i_delay_a;
    }
};

__global__ void SetupPoissKernel(curandState *curand_state, uint64_t n_conn,
				 unsigned long long seed)
{
  uint64_t blockId   = (uint64_t)blockIdx.y * gridDim.x + blockIdx.x;
  uint64_t i_conn = blockId * blockDim.x + threadIdx.x;
  if (i_conn<n_conn) {
    curand_init(seed, i_conn, 0, &curand_state[i_conn]);
  }
}


__global__ void PoissGenUpdateKernel(long long time_idx,
				     int n_node, int max_delay,
				     float *param_arr, int n_param,
				     float *mu_arr)
{
  int i_node = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_node<n_node) {
    float *param = param_arr + i_node*n_param;
    double t_rel = NESTGPUTime - origin;
    if ((t_rel>=start) && (t_rel<=stop)) {
      int it = (int)(time_idx % max_delay);
      mu_arr[it*n_node + i_node] = NESTGPUTimeResolution*rate/1000.0;
    }
  }
}

__global__ void PoissGenSubstractFirstNodeIndexKernel(int64_t n_conn,
						      uint *poiss_key_array,
						      int i_node_0)
{
  uint64_t blockId   = (uint64_t)blockIdx.y * gridDim.x + blockIdx.x;
  uint64_t i_conn_rel = blockId * blockDim.x + threadIdx.x;
  if (i_conn_rel >= n_conn) {
    return;
  }
  uint source_delay = poiss_key_array[i_conn_rel];
  int i_source_rel = (source_delay >> MaxPortNBits) - i_node_0;
  int i_delay = source_delay & PortMask;
  poiss_key_array[i_conn_rel] = (i_source_rel << MaxPortNBits) | i_delay; 
}

/*
__global__ void PoissGenSendSpikeKernel(curandState *curand_state, double t,
					float time_step, float *param_arr,
					int n_param,
					DirectConnection *dir_conn_array,
					uint64_t n_dir_conn)
{
  uint64_t blockId   = (uint64_t)blockIdx.y * gridDim.x + blockIdx.x;
  uint64_t i_conn = blockId * blockDim.x + threadIdx.x;
  if (i_conn<n_dir_conn) {
    DirectConnection dir_conn = dir_conn_array[i_conn];
    int irel = dir_conn.irel_source_;
    int i_target = dir_conn.i_target_;
    int port = dir_conn.port_;
    float weight = dir_conn.weight_;
    float delay = dir_conn.delay_;
    float *param = param_arr + irel*n_param;
    double t_rel = t - origin - delay;

    if ((t_rel>=start) && (t_rel<=stop)){
      int n = curand_poisson(curand_state+i_conn, time_step*rate);
      if (n>0) { // //Send direct spike (i_target, port, weight*n);
	/////////////////////////////////////////////////////////////////
	int i_group=NodeGroupMap[i_target];
	int i = port*NodeGroupArray[i_group].n_node_ + i_target
	  - NodeGroupArray[i_group].i_node_0_;
	double d_val = (double)(weight*n);
	atomicAddDouble(&NodeGroupArray[i_group].get_spike_array_[i], d_val); 
	////////////////////////////////////////////////////////////////
      }
    }
  }
}
*/

__global__ void PoissGenSendSpikeKernel(curandState *curand_state,
					long long time_idx,
					float *mu_arr,
					uint *poiss_key_array,
					int64_t n_conn, int64_t i_conn_0,
					int64_t block_size, int n_node,
					int max_delay)
{
  uint64_t blockId   = (uint64_t)blockIdx.y * gridDim.x + blockIdx.x;
  uint64_t i_conn_rel = blockId * blockDim.x + threadIdx.x;
  if (i_conn_rel >= n_conn) {
    return;
  }
  uint source_delay = poiss_key_array[i_conn_rel];
  int i_source = source_delay >> MaxPortNBits;
  int i_delay = source_delay & PortMask;
  int id = (int)((time_idx - i_delay + 1) % max_delay);
  float mu = mu_arr[id*n_node + i_source];
  int n = curand_poisson(curand_state+i_conn_rel, mu);
  if (n>0) {
    int64_t i_conn = i_conn_0 + i_conn_rel;
    int i_block = (int)(i_conn / block_size);
    int64_t i_block_conn = i_conn % block_size;
    connection_struct conn = ConnectionArray[i_block][i_block_conn];
    uint target_port = conn.target_port;
    int i_target = target_port >> MaxPortNBits;
    uint port = target_port & PortMask;
    float weight = conn.weight;

    int i_group=NodeGroupMap[i_target];
    int i = port*NodeGroupArray[i_group].n_node_ + i_target
      - NodeGroupArray[i_group].i_node_0_;
    double d_val = (double)(weight*n);
    atomicAddDouble(&NodeGroupArray[i_group].get_spike_array_[i], d_val);
  }
}

int poiss_gen::Init(int i_node_0, int n_node, int /*n_port*/,
		    int i_group)
{
  BaseNeuron::Init(i_node_0, n_node, 0 /*n_port*/, i_group);
  node_type_ = i_poisson_generator_model;
  n_scal_param_ = N_POISS_GEN_SCAL_PARAM;
  n_param_ = n_scal_param_;
  scal_param_name_ = poiss_gen_scal_param_name;
  has_dir_conn_ = true;
  
  CUDAMALLOCCTRL("&param_arr_",&param_arr_, n_node_*n_param_*sizeof(float));

  SetScalParam(0, n_node, "rate", 0.0);
  SetScalParam(0, n_node, "origin", 0.0);
  SetScalParam(0, n_node, "start", 0.0);
  SetScalParam(0, n_node, "stop", 1.0e30);
  
  return 0;
}

int poiss_gen::Calibrate(double, float)
{
  buildDirectConnections();
  CUDAMALLOCCTRL("&d_curand_state_",&d_curand_state_, n_conn_*sizeof(curandState));

  unsigned int grid_dim_x, grid_dim_y;

  if (n_conn_<65536*1024) { // max grid dim * max block dim
    grid_dim_x = (n_conn_+1023)/1024;
    grid_dim_y = 1;
  }
  else {
    grid_dim_x = 64; // I think it's not necessary to increase it
    if (n_conn_>grid_dim_x*1024*65535) {
      throw ngpu_exception(std::string("Number of direct connections ")
			   + std::to_string(n_conn_) +
			   " larger than threshold "
			   + std::to_string(grid_dim_x*1024*65535));
    }
    grid_dim_y = (n_conn_ + grid_dim_x*1024 -1) / (grid_dim_x*1024);
  }
  dim3 numBlocks(grid_dim_x, grid_dim_y);
  
  unsigned int *d_seed;
  unsigned int h_seed;

  CUDAMALLOCCTRL("&d_seed",&d_seed, sizeof(unsigned int));
  CURAND_CALL(curandGenerate(*random_generator_, d_seed, 1));
  // Copy seed from device memory to host
  gpuErrchk(cudaMemcpy(&h_seed, d_seed, sizeof(unsigned int),
  		       cudaMemcpyDeviceToHost));
  //std::cout << "h_seed: " << h_seed << "\n";

  SetupPoissKernel<<<numBlocks, 1024>>>(d_curand_state_, n_conn_, h_seed);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  
  return 0;
}


int poiss_gen::Update(long long it, double)
{
  PoissGenUpdateKernel<<<(n_node_+1023)/1024, 1024>>>
    (it, n_node_, max_delay_, param_arr_, n_param_, d_mu_arr_);
    DBGCUDASYNC

  return 0;
}

/*
int poiss_gen::SendDirectSpikes(double t, float time_step)
{
  unsigned int grid_dim_x, grid_dim_y;
  
  if (n_dir_conn_<65536*1024) { // max grid dim * max block dim
    grid_dim_x = (n_dir_conn_+1023)/1024;
    grid_dim_y = 1;
  }
  else {
    grid_dim_x = 64; // I think it's not necessary to increase it
    if (n_dir_conn_>grid_dim_x*1024*65535) {
      throw ngpu_exception(std::string("Number of direct connections ")
			   + std::to_string(n_dir_conn_) +
			   " larger than threshold "
			   + std::to_string(grid_dim_x*1024*65535));
    }
    grid_dim_y = (n_dir_conn_ + grid_dim_x*1024 -1) / (grid_dim_x*1024);
  }
  dim3 numBlocks(grid_dim_x, grid_dim_y);
  PoissGenSendSpikeKernel<<<numBlocks, 1024>>>(d_curand_state_, t, time_step,
					       param_arr_, n_param_,
					       d_dir_conn_array_, n_dir_conn_);
  
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  return 0;
}
*/

int poiss_gen::SendDirectSpikes(long long time_idx)
{
  unsigned int grid_dim_x, grid_dim_y;
  
  if (n_conn_<65536*1024) { // max grid dim * max block dim
    grid_dim_x = (n_conn_+1023)/1024;
    grid_dim_y = 1;
  }
  else {
    grid_dim_x = 64; // I think it's not necessary to increase it
    if (n_conn_>grid_dim_x*1024*65535) {
      throw ngpu_exception(std::string("Number of direct connections ")
			   + std::to_string(n_conn_) +
			   " larger than threshold "
			   + std::to_string(grid_dim_x*1024*65535));
    }
    grid_dim_y = (n_conn_ + grid_dim_x*1024 -1) / (grid_dim_x*1024);
  }
  dim3 numBlocks(grid_dim_x, grid_dim_y);
  PoissGenSendSpikeKernel<<<numBlocks, 1024>>>
    (d_curand_state_,
     time_idx, d_mu_arr_, d_poiss_key_array_,
     n_conn_, i_conn0_,
     h_ConnBlockSize, n_node_, max_delay_);

  DBGCUDASYNC

  return 0;
}



namespace poiss_conn
{
  int OrganizeDirectConnections()
  {    
    uint k = KeySubarray.size();
    int64_t n = NConn;
    int64_t block_size = h_ConnBlockSize;
    
    key_t **key_subarray = KeySubarray.data();
    
    
    CUDAMALLOCCTRL("&d_poiss_key_array_data_pt",&d_poiss_key_array_data_pt, k*sizeof(key_t*));
    gpuErrchk(cudaMemcpy(d_poiss_key_array_data_pt, key_subarray,
			 k*sizeof(key_t*), cudaMemcpyHostToDevice));

    array_t h_poiss_subarray[k];
    for (uint i=0; i<k; i++) {
      h_poiss_subarray[i].h_data_pt = key_subarray;
      h_poiss_subarray[i].data_pt = d_poiss_key_array_data_pt; //key_subarray;
      h_poiss_subarray[i].block_size = block_size;
      h_poiss_subarray[i].offset = i * block_size;
      h_poiss_subarray[i].size = i<k-1 ? block_size : n-(k-1)*block_size;
    }

    CUDAMALLOCCTRL("&d_poiss_subarray",&d_poiss_subarray, k*sizeof(array_t));
    gpuErrchk(cudaMemcpyAsync(d_poiss_subarray, h_poiss_subarray,
			      k*sizeof(array_t), cudaMemcpyHostToDevice));

    CUDAMALLOCCTRL("&d_poiss_num",&d_poiss_num, 2*k*sizeof(int64_t));
    CUDAMALLOCCTRL("&d_poiss_sum",&d_poiss_sum, 2*sizeof(int64_t));
  

    CUDAMALLOCCTRL("&d_poiss_thresh",&d_poiss_thresh, 2*sizeof(key_t));

    return 0;
  }
};

int poiss_gen::buildDirectConnections()
{
  uint k = KeySubarray.size();
  int64_t block_size = h_ConnBlockSize;
  
  poiss_conn::key_t **key_subarray = KeySubarray.data();  
  poiss_conn::key_t h_poiss_thresh[2];
  h_poiss_thresh[0] = i_node_0_ << h_MaxPortNBits;
  h_poiss_thresh[1] = (i_node_0_ + n_node_) << h_MaxPortNBits;
  gpuErrchk(cudaMemcpy(poiss_conn::d_poiss_thresh, h_poiss_thresh,
		       2*sizeof(poiss_conn::key_t),
		       cudaMemcpyHostToDevice));
  
  int64_t h_poiss_num[2*k];
  int64_t *d_num0 = &poiss_conn::d_poiss_num[0];
  int64_t *d_num1 = &poiss_conn::d_poiss_num[k];
  int64_t *h_num0 = &h_poiss_num[0];
  int64_t *h_num1 = &h_poiss_num[k];

  search_multi_down<poiss_conn::key_t, poiss_conn::array_t, 1024>
    (poiss_conn::d_poiss_subarray, k, &poiss_conn::d_poiss_thresh[0], d_num0,
     &poiss_conn::d_poiss_sum[0]);
  CUDASYNC
    
  search_multi_down<poiss_conn::key_t, poiss_conn::array_t, 1024>
    (poiss_conn::d_poiss_subarray, k, &poiss_conn::d_poiss_thresh[1], d_num1,
     &poiss_conn::d_poiss_sum[1]);
  CUDASYNC

  gpuErrchk(cudaMemcpy(h_poiss_num, poiss_conn::d_poiss_num,
		       2*k*sizeof(int64_t), cudaMemcpyDeviceToHost));

  i_conn0_ = 0;
  int64_t i_conn1 = 0;
  uint ib0 = 0;
  uint ib1 = 0;
  for (uint i=0; i<k; i++) {
    if (h_num0[i] < block_size) {
      i_conn0_ = block_size*i + h_num0[i];
      ib0 = i;
      break;
    }
  }
  for (uint i=0; i<k; i++) {
    if (h_num1[i] < block_size) {
      i_conn1 = block_size*i + h_num1[i];
      ib1 = i;
      break;
    }
  }
  n_conn_ = i_conn1 - i_conn0_;
  if (n_conn_>0) {
    CUDAMALLOCCTRL("&d_poiss_key_array_",&d_poiss_key_array_, n_conn_*sizeof(key_t));
    
    int64_t offset = 0;
    for (uint ib=ib0; ib<=ib1; ib++) {
      if (ib==ib0 && ib==ib1) {
	gpuErrchk(cudaMemcpy(d_poiss_key_array_, key_subarray[ib] + h_num0[ib],
			     n_conn_*sizeof(key_t), cudaMemcpyDeviceToDevice));
	break;
      }
      else if (ib==ib0) {
	offset = block_size - h_num0[ib];
	gpuErrchk(cudaMemcpy(d_poiss_key_array_, key_subarray[ib] + h_num0[ib],
			     offset*sizeof(key_t),
			     cudaMemcpyDeviceToDevice));
      }
      else if (ib==ib1) {
	gpuErrchk(cudaMemcpy(d_poiss_key_array_ + offset,
			     key_subarray[ib],
			     h_num1[ib]*sizeof(key_t),
			     cudaMemcpyDeviceToDevice));
	break;
      }
      else {
	gpuErrchk(cudaMemcpy(d_poiss_key_array_ + offset,
			     key_subarray[ib],
			     block_size*sizeof(key_t),
			     cudaMemcpyDeviceToDevice));
	offset += block_size;
      }
    }

    unsigned int grid_dim_x, grid_dim_y;
  
    if (n_conn_<65536*1024) { // max grid dim * max block dim
      grid_dim_x = (n_conn_+1023)/1024;
      grid_dim_y = 1;
    }
    else {
      grid_dim_x = 64; // I think it's not necessary to increase it
      if (n_conn_>grid_dim_x*1024*65535) {
	throw ngpu_exception(std::string("Number of direct connections ")
			     + std::to_string(n_conn_) +
			     " larger than threshold "
			     + std::to_string(grid_dim_x*1024*65535));
      }
      grid_dim_y = (n_conn_ + grid_dim_x*1024 -1) / (grid_dim_x*1024);
    }
    dim3 numBlocks(grid_dim_x, grid_dim_y);
    PoissGenSubstractFirstNodeIndexKernel<<<numBlocks, 1024>>>
      (n_conn_, d_poiss_key_array_, i_node_0_);
    DBGCUDASYNC

  }

  // Find maximum delay of poisson direct connections
  uint *d_max_delay; // maximum delay pointer in device memory
  CUDAMALLOCCTRL("&d_max_delay",&d_max_delay, sizeof(int));
  MaxDelay max_op; // comparison operator used by Reduce function 
  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes,
			    d_poiss_key_array_, d_max_delay, n_conn_, max_op,
			    INT_MIN);
  // Allocate temporary storage
  CUDAMALLOCCTRL("&d_temp_storage",&d_temp_storage, temp_storage_bytes);
  // Run reduction
  cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes,
			    d_poiss_key_array_, d_max_delay, n_conn_, max_op,
			    INT_MIN);
  gpuErrchk(cudaMemcpy(&max_delay_, d_max_delay, sizeof(int),
		       cudaMemcpyDeviceToHost));

  // max_delay_ = 200;
  printf("Max delay of direct (poisson generator) connections: %d\n",
	 max_delay_);
  CUDAMALLOCCTRL("&d_mu_arr_",&d_mu_arr_, n_node_*max_delay_*sizeof(float));
  gpuErrchk(cudaMemset(d_mu_arr_, 0, n_node_*max_delay_*sizeof(float)));
  
  /*
  gpuErrchk(cudaFree(d_key_array_data_pt));
  gpuErrchk(cudaFree(d_subarray));
  gpuErrchk(cudaFree(d_num));
  gpuErrchk(cudaFree(d_sum));
  gpuErrchk(cudaFree(d_thresh));
  */
  
  return 0;
}
