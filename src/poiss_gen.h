/*
 *  poiss_gen.h
 *
 *  This file is part of NEST GPU.
 *
 *  Copyright (C) 2021 The NEST Initiative
 *
 *  NEST GPU is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST GPU is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST GPU.  If not, see <http://www.gnu.org/licenses/>.
 *
 */





#ifndef POISSGEN_H
#define POISSGEN_H

#include <iostream>
#include <string>
#include <curand.h>
#include <curand_kernel.h>
#include "cuda_error.h"
#include "node_group.h"
#include "base_neuron.h"
#include "neuron_models.h"
#include "copass_kernels.h"
#include "connect.h"

/*
const int N_POISS_GEN_SCAL_PARAM = 4;
const std::string poiss_gen_scal_param_name[] = {
  "rate",
  "origin"
  "start",
  "stop",
};
*/

/* BeginUserDocs: device, generator

Short description
+++++++++++++++++

Generate spikes with Poisson process statistics

Description
+++++++++++

The poisson_generator simulates a neuron that is firing with Poisson
statistics, i.e. exponentially distributed interspike intervals. It will
generate a `unique` spike train for each of it's targets. If you do not want
this behavior and need the same spike train for all targets, you have to use a
``parrot_neuron`` between the poisson generator and the targets.

Parameters
++++++++++

The following parameters can be set in the status dictionary.

======== ======= =======================================
 rate    Hz      Mean firing rate
 origin  ms      Reference time for start and stop
 start   ms      Activation time, relative to origin
 stop    ms      Deactivation time, relative to origin
======== ======= =======================================


EndUserDocs */

extern __device__ int16_t *NodeGroupMap;
extern __constant__ NodeGroupStruct NodeGroupArray[];

class poiss_gen : public BaseNeuron
{
  curandState *d_curand_state_;
  uint *d_poiss_key_array_;
  int64_t i_conn0_;
  int64_t n_conn_;
  float *d_mu_arr_;
  int max_delay_;
  
 public:
  
  int Init(int i_node_0, int n_node, int n_port, int i_group);

  int Calibrate(double, float);
		
  int Update(long long it, double t1);

  int SendDirectSpikes(long long time_idx);
  
  template <class ConnKeyT, class ConnStructT>
  int SendDirectSpikesTemplate(long long time_idx);
  
  int buildDirectConnections();
  
  template<class ConnKeyT>
  int buildDirectConnectionsTemplate();
};

/*
// max delay functor
struct MaxDelay
{
  template <class ConnKeyT>
  __device__ __forceinline__
  //uint operator()(const uint &source_delay_a, const uint &source_delay_b)
  //const {
  ConnKeyT operator()(const ConnKeyT &conn_key_a,
		      const ConnKeyT &conn_key_b) const {
    int i_delay_a = getConnDelay<ConnKeyT>(conn_key_a);
    int i_delay_b = getConnDelay<ConnKeyT>(conn_key_b);
    return (i_delay_b > i_delay_a) ? i_delay_b : i_delay_a;
  }
};
*/

// max delay functor
template <class ConnKeyT>
struct MaxDelay
{
  __device__ __forceinline__
  //uint operator()(const uint &source_delay_a, const uint &source_delay_b)
  //const {
  ConnKeyT operator()(const ConnKeyT &conn_key_a,
		      const ConnKeyT &conn_key_b) const {
    int i_delay_a = getConnDelay<ConnKeyT>(conn_key_a);
    int i_delay_b = getConnDelay<ConnKeyT>(conn_key_b);
    return (i_delay_b > i_delay_a) ? i_delay_b : i_delay_a;
  }
};

template <class ConnKeyT>
__global__ void PoissGenSubstractFirstNodeIndexKernel(int64_t n_conn,
						      ConnKeyT *poiss_key_array,
						      int i_node_0)
{
  int64_t blockId   = (int64_t)blockIdx.y * gridDim.x + blockIdx.x;
  int64_t i_conn_rel = blockId * blockDim.x + threadIdx.x;
  if (i_conn_rel >= n_conn) {
    return;
  }
  ConnKeyT &conn_key = poiss_key_array[i_conn_rel];
  int i_source_rel = getConnSource<ConnKeyT>(conn_key) - i_node_0;
  setConnSource<ConnKeyT>(conn_key, i_source_rel);
}


template <class ConnKeyT, class ConnStructT>
__global__ void PoissGenSendSpikeKernel(curandState *curand_state,
					long long time_idx,
					float *mu_arr,
					ConnKeyT *poiss_key_array,
					int64_t n_conn, int64_t i_conn_0,
					int64_t block_size, int n_node,
					int max_delay)
{
  int64_t blockId   = (int64_t)blockIdx.y * gridDim.x + blockIdx.x;
  int64_t i_conn_rel = blockId * blockDim.x + threadIdx.x;
  if (i_conn_rel >= n_conn) {
    return;
  }
  ConnKeyT &conn_key = poiss_key_array[i_conn_rel];
  int i_source = getConnSource<ConnKeyT>(conn_key);
  int i_delay = getConnDelay<ConnKeyT>(conn_key);
  int id = (int)((time_idx - i_delay + 1) % max_delay);
  float mu = mu_arr[id*n_node + i_source];
  int n = curand_poisson(curand_state+i_conn_rel, mu);
  if (n>0) {
    int64_t i_conn = i_conn_0 + i_conn_rel;
    int i_block = (int)(i_conn / block_size);
    int64_t i_block_conn = i_conn % block_size;
    ConnStructT &conn_struct =
      ((ConnStructT**)ConnStructArray)[i_block][i_block_conn];

    int i_target = getConnTarget<ConnStructT>(conn_struct);
    int port = getConnPort<ConnKeyT, ConnStructT>(conn_key, conn_struct); 
    float weight = conn_struct.weight;

    int i_group=NodeGroupMap[i_target];
    int i = port*NodeGroupArray[i_group].n_node_ + i_target
      - NodeGroupArray[i_group].i_node_0_;
    double d_val = (double)(weight*n);
    atomicAddDouble(&NodeGroupArray[i_group].get_spike_array_[i], d_val);
  }
}

template <class ConnKeyT, class ConnStructT>
int poiss_gen::SendDirectSpikesTemplate(long long time_idx)
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
  PoissGenSendSpikeKernel<ConnKeyT, ConnStructT><<<numBlocks, 1024>>>
    (d_curand_state_,
     time_idx, d_mu_arr_, d_poiss_key_array_,
     n_conn_, i_conn0_,
     h_ConnBlockSize, n_node_, max_delay_);

  DBGCUDASYNC

  return 0;
}

namespace poiss_conn
{
  extern void *d_poiss_key_array_data_pt;
  extern void *d_poiss_subarray;  
  extern int64_t *d_poiss_num;
  extern int64_t *d_poiss_sum;
  extern void *d_poiss_thresh;

  template<class ConnKeyT>
  int OrganizeDirectConnections()
  {    
    int k = ConnKeyVect.size();
    int64_t n = NConn;
    int64_t block_size = h_ConnBlockSize;
    
    ConnKeyT **conn_key_array = (ConnKeyT**)ConnKeyVect.data();
    
    
    CUDAMALLOCCTRL("&d_poiss_key_array_data_pt",&d_poiss_key_array_data_pt,
		   k*sizeof(ConnKeyT*));
    gpuErrchk(cudaMemcpy(d_poiss_key_array_data_pt, conn_key_array,
			 k*sizeof(ConnKeyT*), cudaMemcpyHostToDevice));

    regular_block_array<ConnKeyT> h_poiss_subarray[k];
    for (int i=0; i<k; i++) {
      h_poiss_subarray[i].h_data_pt = conn_key_array;
      h_poiss_subarray[i].data_pt = (ConnKeyT**)d_poiss_key_array_data_pt;
      h_poiss_subarray[i].block_size = block_size;
      h_poiss_subarray[i].offset = i * block_size;
      h_poiss_subarray[i].size = i<k-1 ? block_size : n-(k-1)*block_size;
    }

    CUDAMALLOCCTRL("&d_poiss_subarray",&d_poiss_subarray,
		   k*sizeof(regular_block_array<ConnKeyT>));
    gpuErrchk(cudaMemcpyAsync(d_poiss_subarray, h_poiss_subarray,
			      k*sizeof(regular_block_array<ConnKeyT>),
			      cudaMemcpyHostToDevice));

    CUDAMALLOCCTRL("&d_poiss_num",&d_poiss_num, 2*k*sizeof(int64_t));
    CUDAMALLOCCTRL("&d_poiss_sum",&d_poiss_sum, 2*sizeof(int64_t));
  

    CUDAMALLOCCTRL("&d_poiss_thresh",&d_poiss_thresh, 2*sizeof(key_t));

    return 0;
  }
};

template<class ConnKeyT>
int poiss_gen::buildDirectConnectionsTemplate()
{
  int k = ConnKeyVect.size();
  int64_t block_size = h_ConnBlockSize;
  
  ConnKeyT **conn_key_array = (ConnKeyT**)ConnKeyVect.data();  
  ConnKeyT h_poiss_thresh[2];
  h_poiss_thresh[0] = 0;
  hostSetConnSource<ConnKeyT>(h_poiss_thresh[0], i_node_0_);
    
  h_poiss_thresh[1] = 0;
  hostSetConnSource<ConnKeyT>(h_poiss_thresh[1], i_node_0_ + n_node_);

  gpuErrchk(cudaMemcpy(poiss_conn::d_poiss_thresh, h_poiss_thresh,
		       2*sizeof(ConnKeyT),
		       cudaMemcpyHostToDevice));
  
  int64_t h_poiss_num[2*k];
  int64_t *d_num0 = &poiss_conn::d_poiss_num[0];
  int64_t *d_num1 = &poiss_conn::d_poiss_num[k];
  int64_t *h_num0 = &h_poiss_num[0];
  int64_t *h_num1 = &h_poiss_num[k];

  search_multi_down<ConnKeyT, regular_block_array<ConnKeyT>, 1024>
    ( (regular_block_array<ConnKeyT>*) poiss_conn::d_poiss_subarray,
      k, &(((ConnKeyT*) poiss_conn::d_poiss_thresh)[0]), d_num0,
     &poiss_conn::d_poiss_sum[0]);
  CUDASYNC
    
  search_multi_down<ConnKeyT, regular_block_array<ConnKeyT>, 1024>
    ( (regular_block_array<ConnKeyT>*) poiss_conn::d_poiss_subarray,
      k, &(((ConnKeyT*) poiss_conn::d_poiss_thresh)[1]), d_num1,
     &poiss_conn::d_poiss_sum[1]);
  CUDASYNC

  gpuErrchk(cudaMemcpy(h_poiss_num, poiss_conn::d_poiss_num,
		       2*k*sizeof(int64_t), cudaMemcpyDeviceToHost));

  
  i_conn0_ = 0;
  int64_t i_conn1 = 0;
  int ib0 = 0;
  int ib1 = 0;
  for (int i=0; i<k; i++) {
    if (h_num0[i] < block_size) {
      i_conn0_ = block_size*i + h_num0[i];
      ib0 = i;
      break;
    }
  }
  for (int i=0; i<k; i++) {
    if (h_num1[i] < block_size) {
      i_conn1 = block_size*i + h_num1[i];
      ib1 = i;
      break;
    }
  }

  n_conn_ = i_conn1 - i_conn0_;
  if (n_conn_>0) {
    CUDAMALLOCCTRL("&d_poiss_key_array_",&d_poiss_key_array_,
		   n_conn_*sizeof(ConnKeyT));
    
    int64_t offset = 0;
    for (int ib=ib0; ib<=ib1; ib++) {
      if (ib==ib0 && ib==ib1) {
	gpuErrchk(cudaMemcpy(d_poiss_key_array_,
			     conn_key_array[ib] + h_num0[ib],
			     n_conn_*sizeof(ConnKeyT),
			     cudaMemcpyDeviceToDevice));
	break;
      }
      else if (ib==ib0) {
	offset = block_size - h_num0[ib];
	gpuErrchk(cudaMemcpy(d_poiss_key_array_,
			     conn_key_array[ib] + h_num0[ib],
			     offset*sizeof(ConnKeyT),
			     cudaMemcpyDeviceToDevice));
      }
      else if (ib==ib1) {
	gpuErrchk(cudaMemcpy(d_poiss_key_array_ + offset,
			     conn_key_array[ib],
			     h_num1[ib]*sizeof(ConnKeyT),
			     cudaMemcpyDeviceToDevice));
	break;
      }
      else {
	gpuErrchk(cudaMemcpy(d_poiss_key_array_ + offset,
			     conn_key_array[ib],
			     block_size*sizeof(ConnKeyT),
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
  int *d_max_delay; // maximum delay pointer in device memory
  CUDAMALLOCCTRL("&d_max_delay",&d_max_delay, sizeof(int));
  MaxDelay<ConnKeyT> max_op; // comparison operator used by Reduce function 
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
  CUDAFREECTRL("d_key_array_data_pt",d_key_array_data_pt);
  CUDAFREECTRL("d_subarray",d_subarray);
  CUDAFREECTRL("d_num",d_num);
  CUDAFREECTRL("d_sum",d_sum);
  CUDAFREECTRL("d_thresh",d_thresh);
  */
  
  return 0;
}

#endif
