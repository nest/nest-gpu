/*
 *  connect.cu
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

#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <climits>
#include <vector>
#include <utility>
#include <cuda.h>
#include <curand.h>
#include <cub/cub.cuh>
#include "cuda_error.h"
#include "copass_kernels.h"
#include "copass_sort.h"
#include "distribution.h"
#include "connect.h"
#include "nestgpu.h"
#include "utilities.h"

//#define OPTIMIZE_FOR_MEMORY

extern __constant__ float NESTGPUTimeResolution;

bool print_sort_err = true;
bool print_sort_cfr = false;
bool compare_with_serial = false;
uint last_i_sub = 0;

uint h_MaxNodeNBits;
__device__ uint MaxNodeNBits;
// maximum number of bits used to represent node index 

uint h_MaxPortNBits;
__device__ uint MaxPortNBits;
// maximum number of bits used to represent receptor port index and delays 

uint h_PortMask;
__device__ uint PortMask;
// bit mask used to extract port index

uint *d_ConnGroupIdx0;
__device__ uint *ConnGroupIdx0;
// ig0 = ConnGroupIdx0[i_spike_buffer] is the index in the whole
// connection-group array of the first connection group outgoing
// from the node i_spike_buffer

int64_t *d_ConnGroupIConn0;
__device__ int64_t *ConnGroupIConn0;
// i_conn0 = ConnGroupIConn0[ig] with ig = 0, ..., Ng
//  is the index in the whole connection array of the first connection
// belonging to the connection group ig

uint *d_ConnGroupDelay;
__device__ uint *ConnGroupDelay;
// ConnGroupDelay[ig]
// delay associated to all connections of the connection group ig
// with ig = 0, ..., Ng

uint tot_conn_group_num;

int64_t NConn; // total number of connections in the whole network

int64_t h_ConnBlockSize = 10000000; // 160000000; //50000000;
__device__ int64_t ConnBlockSize;
// size (i.e. number of connections) of connection blocks 

uint h_MaxDelayNum;


// it seems that there is no relevant advantage in using a constant array
// however better to keep this option ready and commented
std::vector<uint*> KeySubarray;
uint** d_SourceDelayArray;
__device__ uint** SourceDelayArray;
//__constant__ uint* SourceDelayArray[1024];
// Array of source node indexes and delays of all connections
// Source node indexes and delays are merged in a single integer variable
// The most significant MaxNodeNBits are used for the node index 
// the others (less significant) bits are used to represent the delay
// This array is used as a key array for sorting the connections
// in ascending order according to the source node index
// Connections from the same source node are sorted according to
// the delay

// it seems that there is no relevant advantage in using a constant array
// however better to keep this option ready and commented
std::vector<connection_struct*> ConnectionSubarray;
connection_struct** d_ConnectionArray;
__device__ connection_struct** ConnectionArray;
//__constant__ connection_struct* ConnectionArray[1024];
// array of target node indexes, receptor port index, synapse type,
// weight of all connections
// used as a value for key-value sorting of the connections (see above)


enum ConnectionFloatParamIndexes {
  i_weight_param = 0,
  i_delay_param,
  N_CONN_FLOAT_PARAM
};

enum ConnectionIntParamIndexes {
  i_source_param = 0,
  i_target_param,
  i_port_param,
  i_syn_group_param,
  N_CONN_INT_PARAM
};

const std::string ConnectionFloatParamName[N_CONN_FLOAT_PARAM] = {
  "weight",
  "delay"
};

const std::string ConnectionIntParamName[N_CONN_INT_PARAM] = {
  "source",
  "target",
  "port",
  "syn_group"
};


__global__ void setConnGroupNum(int64_t n_compact,
				uint *conn_group_num,
				int64_t *conn_group_idx0_compact,
				int *conn_group_source_compact)
{
  int64_t i_compact = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_compact>=n_compact) return;
  int source = conn_group_source_compact[i_compact];
  uint num = (uint)(conn_group_idx0_compact[i_compact+1]
		    - conn_group_idx0_compact[i_compact]);
  conn_group_num[source] = num;
}


__global__ void setConnGroupIdx0Compact
(uint *key_subarray, int64_t n_block_conn, int *conn_group_idx0_mask,
 int64_t *conn_group_iconn0_mask_cumul, int64_t *conn_group_idx0_mask_cumul,
 int64_t *conn_group_idx0_compact, int *conn_group_source_compact,
 int64_t *iconn0_offset, int64_t *idx0_offset)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>n_block_conn) return;
  if (i_conn<n_block_conn && conn_group_idx0_mask[i_conn]==0) return;
  int64_t i_group = conn_group_iconn0_mask_cumul[i_conn] + *iconn0_offset;
  int64_t i_source_compact = conn_group_idx0_mask_cumul[i_conn]
    + *idx0_offset;
  conn_group_idx0_compact[i_source_compact] = i_group;
  if (i_conn<n_block_conn) {
    int source = key_subarray[i_conn] >> MaxPortNBits;
    conn_group_source_compact[i_source_compact] = source;
  }
}


__global__ void buildConnGroupMask(uint *key_subarray,
				   uint *key_subarray_prev,
				   int64_t n_block_conn,
				   int *conn_group_iconn0_mask,
				   int *conn_group_idx0_mask)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_block_conn) return;
  uint val = key_subarray[i_conn];
  int64_t prev_val;
  int prev_source;
  if (i_conn==0) {
    if (key_subarray_prev != NULL) {
      prev_val = *key_subarray_prev;
      prev_source = prev_val >> MaxPortNBits; 
    }
    else {
      prev_val = -1;      // just to ensure it is different from val
      prev_source = -1;
    }
  }
  else {
    prev_val = key_subarray[i_conn-1];
    prev_source = prev_val >> MaxPortNBits;
  }
  if (val != prev_val) {
    conn_group_iconn0_mask[i_conn] = 1;
    int source = val >> MaxPortNBits; 
    if (source != prev_source) {
      conn_group_idx0_mask[i_conn] = 1;
    }
  }
}

__global__ void buildConnGroupIConn0Mask(uint *key_subarray,
					 uint *key_subarray_prev,
					 int64_t n_block_conn,
					 int *conn_group_iconn0_mask)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_block_conn) return;
  uint val = key_subarray[i_conn];
  int64_t prev_val;
  if (i_conn==0) {
    if (key_subarray_prev != NULL) {
      prev_val = *key_subarray_prev;
    }
    else {
      prev_val = -1;      // just to ensure it is different from val
    }
  }
  else {
    prev_val = key_subarray[i_conn-1];
  }
  if (val != prev_val) {
    conn_group_iconn0_mask[i_conn] = 1;
  }
}

__global__ void setConnGroupIConn0(int64_t n_block_conn,
				   int *conn_group_iconn0_mask,
				   int64_t *conn_group_iconn0_mask_cumul,
				   int64_t *conn_group_iconn0, int64_t i_conn0,
				   int64_t *offset)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_block_conn) return;
  if (conn_group_iconn0_mask[i_conn] != 0) {
    int64_t pos = conn_group_iconn0_mask_cumul[i_conn] + *offset;
    conn_group_iconn0[pos] = i_conn0 + i_conn;
  }
}

__global__ void setConnGroupNewOffset(int64_t *offset, int64_t *add_offset)
{
  *offset = *offset + *add_offset;
}


__global__ void setWeights(connection_struct *conn_subarray, float weight,
			   int64_t n_conn)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) return;
  conn_subarray[i_conn].weight = weight;
}


__global__ void setWeights(connection_struct *conn_subarray, float *arr_val,
			   int64_t n_conn)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) return;
  conn_subarray[i_conn].weight = arr_val[i_conn];
}


__global__ void setDelays(uint *key_subarray, float *arr_val,
			  int64_t n_conn, float time_resolution)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) return;
  int delay = (int)round(arr_val[i_conn]/time_resolution);
  delay = max(delay,1);
  key_subarray[i_conn] = (key_subarray[i_conn] << MaxPortNBits) | delay;
}


__global__ void setDelays(uint *key_subarray, float fdelay,
			  int64_t n_conn, float time_resolution)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) return;
  int delay = (int)round(fdelay/time_resolution);
  delay = max(delay,1);
  key_subarray[i_conn] = (key_subarray[i_conn] << MaxPortNBits) | delay;
}


__global__ void setPort(connection_struct *conn_subarray, uint port,
			   int64_t n_conn)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) return;
  conn_subarray[i_conn].target_port =
    (conn_subarray[i_conn].target_port << MaxPortNBits) | port; 
}


__global__ void setSynGroup(connection_struct *conn_subarray,
			    unsigned char syn_group,
			    int64_t n_conn)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) return;
  conn_subarray[i_conn].syn_group = syn_group; 
}

__global__ void getConnGroupDelay(int64_t block_size,
				  uint **source_delay_array,
				  int64_t *conn_group_iconn0,
				  uint *conn_group_delay,
				  uint conn_group_num)
{
  uint conn_group_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (conn_group_idx >= conn_group_num) return;
  int64_t i_conn = conn_group_iconn0[conn_group_idx];
  uint i_block = (uint)(i_conn / block_size);
  int64_t i_block_conn = i_conn % block_size;
  uint source_delay = source_delay_array[i_block][i_block_conn];
  conn_group_delay[conn_group_idx] = source_delay & PortMask;
}




int allocateNewBlocks(std::vector<uint*> &key_subarray,
		      std::vector<connection_struct*> &conn_subarray,
		      int64_t block_size, uint new_n_block)
{
  // Allocating GPU memory for new connection blocks
  // allocate new blocks if needed
  for (uint ib=key_subarray.size(); ib<new_n_block; ib++) {
    uint *d_key_pt;
    connection_struct *d_connection_pt;
    // allocate GPU memory for new blocks 
    CUDAMALLOCCTRL("&d_key_pt",&d_key_pt, block_size*sizeof(uint));
    CUDAMALLOCCTRL("&d_connection_pt",&d_connection_pt,
			 block_size*sizeof(connection_struct));
    key_subarray.push_back(d_key_pt);
    conn_subarray.push_back(d_connection_pt);
  }

  return 0;
}


int setConnectionWeights(curandGenerator_t &gen, void *d_storage,
			 connection_struct *conn_subarray, int64_t n_conn,
			 SynSpec &syn_spec)
{
  if (syn_spec.weight_distr_ >= DISTR_TYPE_ARRAY   // probability distribution
      && syn_spec.weight_distr_ < N_DISTR_TYPE) {  // or array
    if (syn_spec.weight_distr_ == DISTR_TYPE_ARRAY) {
      gpuErrchk(cudaMemcpy(d_storage, syn_spec.weight_h_array_pt_,
			   n_conn*sizeof(float), cudaMemcpyHostToDevice));    
    }
    else if (syn_spec.weight_distr_ == DISTR_TYPE_NORMAL_CLIPPED) {
      CURAND_CALL(curandGenerateUniform(gen, (float*)d_storage, n_conn));
      randomNormalClipped((float*)d_storage, n_conn, syn_spec.weight_mu_,
			  syn_spec.weight_sigma_, syn_spec.weight_low_,
			  syn_spec.weight_high_);
    }
    else if (syn_spec.weight_distr_==DISTR_TYPE_NORMAL) {
      float low = syn_spec.weight_mu_ - 5.0*syn_spec.weight_sigma_;
      float high = syn_spec.weight_mu_ + 5.0*syn_spec.weight_sigma_;
      CURAND_CALL(curandGenerateUniform(gen, (float*)d_storage, n_conn));
      randomNormalClipped((float*)d_storage, n_conn, syn_spec.weight_mu_,
			  syn_spec.weight_sigma_, low, high);
    }
    else {
      throw ngpu_exception("Invalid connection weight distribution type");
    }
    setWeights<<<(n_conn+1023)/1024, 1024>>>
      (conn_subarray, (float*)d_storage, n_conn);
    DBGCUDASYNC
  }
  else {
    setWeights<<<(n_conn+1023)/1024, 1024>>>
      (conn_subarray, syn_spec.weight_, n_conn);
    DBGCUDASYNC
  }
    
  return 0;
}


int setConnectionDelays(curandGenerator_t &gen, void *d_storage,
			uint *key_subarray, int64_t n_conn,
			SynSpec &syn_spec, float time_resolution)
{
  if (syn_spec.delay_distr_ >= DISTR_TYPE_ARRAY   // probability distribution
      && syn_spec.delay_distr_ < N_DISTR_TYPE) {  // or array
    if (syn_spec.delay_distr_ == DISTR_TYPE_ARRAY) {
      gpuErrchk(cudaMemcpy(d_storage, syn_spec.delay_h_array_pt_,
			   n_conn*sizeof(float), cudaMemcpyHostToDevice));
    }
    else if (syn_spec.delay_distr_ == DISTR_TYPE_NORMAL_CLIPPED) {
      CURAND_CALL(curandGenerateUniform(gen, (float*)d_storage, n_conn));
      randomNormalClipped((float*)d_storage, n_conn, syn_spec.delay_mu_,
			  syn_spec.delay_sigma_, syn_spec.delay_low_,
			  syn_spec.delay_high_);
    }
    else if (syn_spec.delay_distr_ == DISTR_TYPE_NORMAL) {
      float low = syn_spec.delay_mu_ - 5.0*syn_spec.delay_sigma_;
      float high = syn_spec.delay_mu_ + 5.0*syn_spec.delay_sigma_;
      CURAND_CALL(curandGenerateUniform(gen, (float*)d_storage, n_conn));
      randomNormalClipped((float*)d_storage, n_conn, syn_spec.delay_mu_,
			  syn_spec.delay_sigma_, syn_spec.delay_low_,
			  syn_spec.delay_high_);
    }
    else {
      throw ngpu_exception("Invalid connection delay distribution type");
    }

    setDelays<<<(n_conn+1023)/1024, 1024>>>
      (key_subarray, (float*)d_storage, n_conn, time_resolution);
    DBGCUDASYNC

  }
  else {
    setDelays<<<(n_conn+1023)/1024, 1024>>>
      (key_subarray, syn_spec.delay_, n_conn, time_resolution);
    DBGCUDASYNC
  }
  return 0;
}


int organizeConnections(float time_resolution, uint n_node, int64_t n_conn,
			int64_t block_size,
			std::vector<uint*> &key_subarray,
			std::vector<connection_struct*> &conn_subarray)
{
  typedef uint key_t;
  timeval startTV;
  timeval endTV;
  CUDASYNC
  gettimeofday(&startTV, NULL);

  if (n_conn > 0) {
    printf("Allocating auxiliary GPU memory...\n");
    int64_t sort_storage_bytes = 0;
    void *d_sort_storage = NULL;
    copass_sort::sort<uint, connection_struct>(key_subarray.data(),
					       conn_subarray.data(), n_conn,
					       block_size, d_sort_storage,
					       sort_storage_bytes);
    printf("storage bytes: %ld\n", sort_storage_bytes);
    CUDAMALLOCCTRL("&d_sort_storage",&d_sort_storage, sort_storage_bytes);
    
    printf("Sorting...\n");
    copass_sort::sort<uint, connection_struct>(key_subarray.data(),
					       conn_subarray.data(), n_conn,
					       block_size, d_sort_storage,
					       sort_storage_bytes);
    CUDAFREECTRL("d_sort_storage",d_sort_storage);

    size_t storage_bytes = 0;
    size_t storage_bytes1 = 0;
    void *d_storage = NULL;
    printf("Indexing connection groups...\n");
    // It is important to separate number of allocated blocks
    // (determined by key_subarray.size()) from number of blocks
    // on which there are connections, which is determined by n_conn
    // number of used connection blocks
    uint k = (n_conn - 1)  / block_size + 1;
    
    // it seems that there is no relevant advantage in using a constant array
    // however better to keep this option ready and commented
    //gpuErrchk(cudaMemcpyToSymbol(SourceDelayArray, KeySubarray.data(),
    //				 k*sizeof(uint*)));//, cudaMemcpyHostToDevice));
    //gpuErrchk(cudaMemcpyToSymbol(ConnectionArray, ConnectionSubarray.data(),
    //				 k*sizeof(connection_struct*)));
				 //, cudaMemcpyHostToDevice));

    CUDAMALLOCCTRL("&d_SourceDelayArray",&d_SourceDelayArray,
		   k*sizeof(uint*));
    gpuErrchk(cudaMemcpy(d_SourceDelayArray, KeySubarray.data(),
			 k*sizeof(uint*), cudaMemcpyHostToDevice));
  
    CUDAMALLOCCTRL("&d_ConnectionArray",&d_ConnectionArray,
		   k*sizeof(connection_struct*));
    gpuErrchk(cudaMemcpy(d_ConnectionArray, ConnectionSubarray.data(),
			 k*sizeof(connection_struct*), cudaMemcpyHostToDevice));

    //////////////////////////////////////////////////////////////////////
    
    int *d_conn_group_iconn0_mask;
    CUDAMALLOCCTRL("&d_conn_group_iconn0_mask",
		   &d_conn_group_iconn0_mask,
		   block_size*sizeof(int));

    int64_t *d_conn_group_iconn0_mask_cumul;
    CUDAMALLOCCTRL("&d_conn_group_iconn0_mask_cumul",
		   &d_conn_group_iconn0_mask_cumul,
		   (block_size+1)*sizeof(int64_t));
    
    int *d_conn_group_idx0_mask;
    CUDAMALLOCCTRL("&d_conn_group_idx0_mask",
		   &d_conn_group_idx0_mask,
		   block_size*sizeof(int));

    int64_t *d_conn_group_idx0_mask_cumul;
    CUDAMALLOCCTRL("&d_conn_group_idx0_mask_cumul",
		   &d_conn_group_idx0_mask_cumul,
		   (block_size+1)*sizeof(int64_t));

    int64_t *d_conn_group_idx0_compact;
    int64_t reserve_size = n_node<block_size ? n_node : block_size;
    CUDAMALLOCCTRL("&d_conn_group_idx0_compact",
		   &d_conn_group_idx0_compact,
		   (reserve_size+1)*sizeof(int64_t));
  
    int *d_conn_group_source_compact;
    CUDAMALLOCCTRL("&d_conn_group_source_compact",
		   &d_conn_group_source_compact,
		   reserve_size*sizeof(int));
  
    int64_t *d_iconn0_offset;
    CUDAMALLOCCTRL("&d_iconn0_offset", &d_iconn0_offset, sizeof(int64_t));
    gpuErrchk(cudaMemset(d_iconn0_offset, 0, sizeof(int64_t)));
    int64_t *d_idx0_offset;
    CUDAMALLOCCTRL("&d_idx0_offset", &d_idx0_offset, sizeof(int64_t));
    gpuErrchk(cudaMemset(d_idx0_offset, 0, sizeof(int64_t)));

    uint *key_subarray_prev = NULL;
    for (uint ib=0; ib<k; ib++) {
      uint n_block_conn = ib<(k-1) ? block_size : NConn - block_size*(k-1);
      gpuErrchk(cudaMemset(d_conn_group_iconn0_mask, 0,
			   n_block_conn*sizeof(int)));
      buildConnGroupIConn0Mask<<<(n_block_conn+1023)/1024, 1024>>>
	(key_subarray[ib], key_subarray_prev, n_block_conn,
	 d_conn_group_iconn0_mask);
      CUDASYNC;
      
      key_subarray_prev = key_subarray[ib] + block_size - 1;
    
      if (ib==0) {
	// Determine temporary device storage requirements for prefix sum
	cub::DeviceScan::ExclusiveSum(NULL, storage_bytes,
				      d_conn_group_iconn0_mask,
				      d_conn_group_iconn0_mask_cumul,
				      n_block_conn+1);
	// Allocate temporary storage for prefix sum
	CUDAMALLOCCTRL("&d_storage",&d_storage, storage_bytes);
      }
      // Run exclusive prefix sum
      cub::DeviceScan::ExclusiveSum(d_storage, storage_bytes,
				    d_conn_group_iconn0_mask,
				    d_conn_group_iconn0_mask_cumul,
				    n_block_conn+1);

      setConnGroupNewOffset<<<1, 1>>>(d_iconn0_offset,
				      d_conn_group_iconn0_mask_cumul
				      + n_block_conn);

      CUDASYNC;
      
    }
    gpuErrchk(cudaMemcpy(&tot_conn_group_num, d_iconn0_offset,
			 sizeof(int64_t), cudaMemcpyDeviceToHost));
    printf("Total number of connection groups: %d\n", tot_conn_group_num);

    if (tot_conn_group_num > 0) {
      uint *d_conn_group_num;
      CUDAMALLOCCTRL("&d_conn_group_num", &d_conn_group_num,
		     n_node*sizeof(uint));
      gpuErrchk(cudaMemset(d_conn_group_num, 0, sizeof(uint)));
    
      uint *key_subarray_prev = NULL;
      gpuErrchk(cudaMemset(d_iconn0_offset, 0, sizeof(int64_t)));

      CUDAMALLOCCTRL("&d_ConnGroupIConn0",&d_ConnGroupIConn0,
		     (tot_conn_group_num+1)*sizeof(int64_t));

      int64_t n_compact = 0; 
      for (uint ib=0; ib<k; ib++) {
	uint n_block_conn = ib<(k-1) ? block_size : NConn - block_size*(k-1);
	gpuErrchk(cudaMemset(d_conn_group_iconn0_mask, 0,
			     n_block_conn*sizeof(int)));
	gpuErrchk(cudaMemset(d_conn_group_idx0_mask, 0,
			     n_block_conn*sizeof(int)));
	buildConnGroupMask<<<(n_block_conn+1023)/1024, 1024>>>
	  (key_subarray[ib], key_subarray_prev, n_block_conn,
	   d_conn_group_iconn0_mask, d_conn_group_idx0_mask);
	CUDASYNC;
      
	key_subarray_prev = key_subarray[ib] + block_size - 1;
    
	// Run exclusive prefix sum
	cub::DeviceScan::ExclusiveSum(d_storage, storage_bytes,
				      d_conn_group_iconn0_mask,
				      d_conn_group_iconn0_mask_cumul,
				      n_block_conn+1);
	DBGCUDASYNC;
	cub::DeviceScan::ExclusiveSum(d_storage, storage_bytes,
				      d_conn_group_idx0_mask,
				      d_conn_group_idx0_mask_cumul,
				      n_block_conn+1);

	DBGCUDASYNC;
	int64_t i_conn0 = block_size*ib;
	setConnGroupIConn0<<<(n_block_conn+1023)/1024, 1024>>>
	  (n_block_conn, d_conn_group_iconn0_mask,
	   d_conn_group_iconn0_mask_cumul, d_ConnGroupIConn0,
	   i_conn0, d_iconn0_offset);
	CUDASYNC;

	setConnGroupIdx0Compact<<<(n_block_conn+1023)/1024, 1024>>>
	  (key_subarray[ib], n_block_conn, d_conn_group_idx0_mask,
	   d_conn_group_iconn0_mask_cumul, d_conn_group_idx0_mask_cumul,
	   d_conn_group_idx0_compact, d_conn_group_source_compact,
	   d_iconn0_offset, d_idx0_offset);
	CUDASYNC;

	int64_t n_block_compact; 
	gpuErrchk(cudaMemcpy(&n_block_compact, d_conn_group_idx0_mask_cumul
			     + n_block_conn,
			     sizeof(int64_t), cudaMemcpyDeviceToHost));
	//std::cout << "number of nodes with outgoing connections "
	//"in block " << ib << ": " << n_block_compact << "\n";
	n_compact += n_block_compact;
            
	setConnGroupNewOffset<<<1, 1>>>(d_iconn0_offset,
					d_conn_group_iconn0_mask_cumul
					+ n_block_conn);
	setConnGroupNewOffset<<<1, 1>>>(d_idx0_offset,
					d_conn_group_idx0_mask_cumul
					+ n_block_conn);
	CUDASYNC;
      }
      gpuErrchk(cudaMemcpy(d_ConnGroupIConn0+tot_conn_group_num, &NConn,
			   sizeof(int64_t), cudaMemcpyHostToDevice));

      setConnGroupNum<<<(n_compact+1023)/1024, 1024>>>
	(n_compact, d_conn_group_num, d_conn_group_idx0_compact,
	 d_conn_group_source_compact);
      CUDASYNC;

      CUDAMALLOCCTRL("&d_ConnGroupIdx0", &d_ConnGroupIdx0,
		     (n_node+1)*sizeof(uint));
      storage_bytes1 = 0;
      
      // Determine temporary device storage requirements for prefix sum
      cub::DeviceScan::ExclusiveSum(NULL, storage_bytes1,
				    d_conn_group_num,
				    d_ConnGroupIdx0,
				    n_node+1);
      if (storage_bytes1 > storage_bytes) {
	storage_bytes = storage_bytes1;
	CUDAFREECTRL("d_storage",d_storage);
	// Allocate temporary storage for prefix sum
	CUDAMALLOCCTRL("&d_storage",&d_storage, storage_bytes);
      }
      // Run exclusive prefix sum
      cub::DeviceScan::ExclusiveSum(d_storage, storage_bytes,
				    d_conn_group_num,
				    d_ConnGroupIdx0,
				    n_node+1);

      // find maxumum number of connection groups (delays) over all neurons
      uint *d_max_delay_num;
      CUDAMALLOCCTRL("&d_max_delay_num",&d_max_delay_num, sizeof(uint));
    
      storage_bytes1 = 0; 
      // Determine temporary device storage requirements
      cub::DeviceReduce::Max(NULL, storage_bytes1,
			     d_conn_group_num, d_max_delay_num, n_node);
      if (storage_bytes1 > storage_bytes) {
	storage_bytes = storage_bytes1;
	CUDAFREECTRL("d_storage",d_storage);
	// Allocate temporary storage for prefix sum
	CUDAMALLOCCTRL("&d_storage",&d_storage, storage_bytes);
      }
    
      // Run maximum search
      cub::DeviceReduce::Max(d_storage, storage_bytes,
			     d_conn_group_num, d_max_delay_num, n_node);
    
      CUDASYNC;
      gpuErrchk(cudaMemcpy(&h_MaxDelayNum, d_max_delay_num,
			   sizeof(uint), cudaMemcpyDeviceToHost));
      CUDAFREECTRL("d_max_delay_num",d_max_delay_num);

      printf("Maximum number of connection groups (delays) over all nodes: %d\n",
	     h_MaxDelayNum);
    

      ///////////////////////////////////////////////////////////////////
      ///////////////////////////////////////////////////////////////////
      CUDAFREECTRL("d_storage",d_storage); // free temporary allocated storage
      CUDAFREECTRL("d_conn_group_iconn0_mask",d_conn_group_iconn0_mask);
      CUDAFREECTRL("d_conn_group_iconn0_mask_cumul",d_conn_group_iconn0_mask_cumul);
      CUDAFREECTRL("d_iconn0_offset",d_iconn0_offset);
      CUDAFREECTRL("d_conn_group_idx0_mask",d_conn_group_idx0_mask);
      CUDAFREECTRL("d_conn_group_idx0_mask_cumul",d_conn_group_idx0_mask_cumul);
      CUDAFREECTRL("d_idx0_offset",d_idx0_offset);
      CUDAFREECTRL("d_conn_group_idx0_compact",d_conn_group_idx0_compact);
      CUDAFREECTRL("d_conn_group_num",d_conn_group_num);
      
#ifndef OPTIMIZE_FOR_MEMORY
      CUDAMALLOCCTRL("&d_ConnGroupDelay",&d_ConnGroupDelay,
		     tot_conn_group_num*sizeof(uint));

      getConnGroupDelay<<<(tot_conn_group_num+1023)/1024, 1024>>>
	(block_size, d_SourceDelayArray, d_ConnGroupIConn0, d_ConnGroupDelay,
	 tot_conn_group_num);
      DBGCUDASYNC
#endif
	
    }
    else {
      throw ngpu_exception("Number of connections groups must be positive "
			   "for number of connections > 0");   
    }
  }
  else {
    gpuErrchk(cudaMemset(d_ConnGroupIdx0, 0, (n_node+1)*sizeof(uint)));
    h_MaxDelayNum = 0;
  }
  
  gettimeofday(&endTV, NULL);
  long time = (long)((endTV.tv_sec * 1000000.0 + endTV.tv_usec)
		     - (startTV.tv_sec * 1000000.0 + startTV.tv_usec));
  printf("%-40s%.2f ms\n", "Time: ", (double)time / 1000.);
  printf("Done\n");
  
  
  return 0;
}


__global__ void ConnectInitKernel(uint *conn_group_idx0,
				     int64_t *conn_group_iconn0,
				     uint *conn_group_delay,
				     int64_t block_size,
				     uint **source_delay_array,
				     connection_struct **connection_array)
{
  
  ConnGroupIdx0 = conn_group_idx0;
  ConnGroupIConn0 = conn_group_iconn0;
  ConnGroupDelay = conn_group_delay;
  ConnBlockSize = block_size;
  SourceDelayArray = source_delay_array;
  ConnectionArray = connection_array;
}

int ConnectInit()
{
  /*
  uint k = ConnectionSubarray.size();
  uint **d_source_delay_array;
  CUDAMALLOCCTRL("&d_source_delay_array",&d_source_delay_array, k*sizeof(uint*));
  gpuErrchk(cudaMemcpy(d_source_delay_array, KeySubarray.data(),
		       k*sizeof(uint*), cudaMemcpyHostToDevice));
  
  connection_struct **d_connection_array;
  CUDAMALLOCCTRL("&d_connection_array",&d_connection_array, k*sizeof(connection_struct*));
  gpuErrchk(cudaMemcpy(d_connection_array, ConnectionSubarray.data(),
		       k*sizeof(connection_struct*), cudaMemcpyHostToDevice));

  */
  ConnectInitKernel<<<1,1>>>(d_ConnGroupIdx0, d_ConnGroupIConn0,
				d_ConnGroupDelay, h_ConnBlockSize,
				d_SourceDelayArray,
				d_ConnectionArray);
  DBGCUDASYNC

  return 0;
}


__global__ void setMaxNodeNBitsKernel(int max_node_nbits, int max_port_nbits,
				      int port_mask)
{
  MaxNodeNBits = max_node_nbits;
  MaxPortNBits = max_port_nbits;
  PortMask = port_mask;
}

int setMaxNodeNBits(int max_node_nbits)
{
  h_MaxNodeNBits = max_node_nbits;
  h_MaxPortNBits = 32 - h_MaxNodeNBits;
  h_PortMask = (1 << h_MaxPortNBits) - 1;
  setMaxNodeNBitsKernel<<<1,1>>>(h_MaxNodeNBits, h_MaxPortNBits, h_PortMask);
  DBGCUDASYNC

  return 0;
}  

int *sortArray(int *h_arr, int n_elem)
{
  // allocate unsorted and sorted array in device memory
  int *d_arr_unsorted;
  int *d_arr_sorted;
  CUDAMALLOCCTRL("&d_arr_unsorted",&d_arr_unsorted, n_elem*sizeof(int));
  CUDAMALLOCCTRL("&d_arr_sorted",&d_arr_sorted, n_elem*sizeof(int));
  gpuErrchk(cudaMemcpy(d_arr_unsorted, h_arr, n_elem*sizeof(int),
		       cudaMemcpyHostToDevice));
  void *d_storage = NULL;
  size_t storage_bytes = 0;
  // Determine temporary storage requirements for sorting source indexes
  cub::DeviceRadixSort::SortKeys(d_storage, storage_bytes, d_arr_unsorted,
				 d_arr_sorted, n_elem);
  // Allocate temporary storage for sorting
  CUDAMALLOCCTRL("&d_storage",&d_storage, storage_bytes);
  // Run radix sort
  cub::DeviceRadixSort::SortKeys(d_storage, storage_bytes, d_arr_unsorted,
				 d_arr_sorted, n_elem);
  CUDAFREECTRL("d_storage",d_storage);
  CUDAFREECTRL("d_arr_unsorted",d_arr_unsorted);

  return d_arr_sorted;
}

__global__ void setSourceTargetIndexKernel(int64_t n_src_tgt, int  n_source,
					   int n_target, int64_t *d_src_tgt_arr,
					   int *d_src_arr, int *d_tgt_arr)
{
  int64_t i_src_tgt = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; 
  if (i_src_tgt >= n_src_tgt) return;
  int i_src =(int)(i_src_tgt / n_target);
  int i_tgt =(int)(i_src_tgt % n_target);
  int src_id = d_src_arr[i_src];
  int tgt_id = d_tgt_arr[i_tgt];
  int64_t src_tgt_id = ((int64_t)src_id << 32) | tgt_id;
  d_src_tgt_arr[i_src_tgt] = src_tgt_id;
  //printf("i_src_tgt %lld\tsrc_id %d\ttgt_id %d\tsrc_tgt_id %lld\n", 
  //	 i_src_tgt, src_id, tgt_id, src_tgt_id); 
}

// Count number of connections per source-target couple
__global__ void CountConnectionsKernel(int64_t n_conn, int n_source,
				       int n_target, int64_t *src_tgt_arr,
				       int64_t *src_tgt_conn_num,
				       int syn_group)
{
  int64_t i_conn = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; 
  if (i_conn >= n_conn) return;

  uint i_block = (uint)(i_conn / ConnBlockSize);
  int64_t i_block_conn = i_conn % ConnBlockSize;
  connection_struct conn = ConnectionArray[i_block][i_block_conn];
  if (syn_group==-1 || conn.syn_group == syn_group) {
    // First get target node index
    uint target_port = conn.target_port;
    int i_target = target_port >> MaxPortNBits;
    uint source_delay = SourceDelayArray[i_block][i_block_conn];
    int i_source = source_delay >> MaxPortNBits;
    int64_t i_src_tgt = ((int64_t)i_source << 32) | i_target;
    int64_t i_arr = locate(i_src_tgt, src_tgt_arr, n_source*n_target);
    if (src_tgt_arr[i_arr] == i_src_tgt) {
      //printf("i_conn %lld i_source %d i_target %d i_src_tgt %lld "
      //     "i_arr %lld\n", i_conn, i_source, i_target, i_src_tgt, i_arr);
      // (atomic)increase the number of connections for source-target couple
      atomicAdd((unsigned long long *)&src_tgt_conn_num[i_arr], 1);
    }
  }
}


// Fill array of connection indexes
__global__ void SetConnectionsIndexKernel(int64_t n_conn, int n_source,
					  int n_target, int64_t *src_tgt_arr,
					  int64_t *src_tgt_conn_num,
					  int64_t *src_tgt_conn_cumul,
					  int syn_group, int64_t *conn_ids)
{
  int64_t i_conn = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; 
  if (i_conn >= n_conn) return;

  uint i_block = (uint)(i_conn / ConnBlockSize);
  int64_t i_block_conn = i_conn % ConnBlockSize;
  connection_struct conn = ConnectionArray[i_block][i_block_conn];
  if (syn_group==-1 || conn.syn_group == syn_group) {
    // First get target node index
    uint target_port = conn.target_port;
    int i_target = target_port >> MaxPortNBits;
    uint source_delay = SourceDelayArray[i_block][i_block_conn];
    int i_source = source_delay >> MaxPortNBits;
    int64_t i_src_tgt = ((int64_t)i_source << 32) | i_target;
    int64_t i_arr = locate(i_src_tgt, src_tgt_arr, n_source*n_target);
    if (src_tgt_arr[i_arr] == i_src_tgt) {
      //printf("i_conn %lld i_source %d i_target %d i_src_tgt %lld "
      //     "i_arr %lld\n", i_conn, i_source, i_target, i_src_tgt, i_arr);
      // (atomic)increase the number of connections for source-target couple
      int64_t pos =
	atomicAdd((unsigned long long *)&src_tgt_conn_num[i_arr], 1);
      //printf("pos %lld src_tgt_conn_cumul[i_arr] %lld\n",
      //     pos, src_tgt_conn_cumul[i_arr]);
      conn_ids[src_tgt_conn_cumul[i_arr] + pos] = i_conn;
    }
  }
}


int64_t *NESTGPU::GetConnections(int *i_source_pt, int n_source,
				 int *i_target_pt, int n_target,
				 int syn_group, int64_t *n_conn)
{  
  int64_t *h_conn_ids = NULL;
  int64_t *d_conn_ids = NULL;
  int64_t n_src_tgt = (int64_t)n_source * n_target;
  int64_t n_conn_ids = 0;
  
  if (n_src_tgt > 0) {
    // sort source node index array in GPU memory
    int *d_src_arr = sortArray(i_source_pt, n_source);
    // sort target node index array in GPU memory
    int *d_tgt_arr = sortArray(i_target_pt, n_target);
    // Allocate array of combined source-target indexes (src_arr x tgt_arr)
    int64_t *d_src_tgt_arr;
    CUDAMALLOCCTRL("&d_src_tgt_arr",&d_src_tgt_arr, n_src_tgt*sizeof(int64_t));
    // Fill it with combined source-target indexes
    setSourceTargetIndexKernel<<<(n_src_tgt+1023)/1024, 1024>>>
      (n_src_tgt, n_source, n_target, d_src_tgt_arr, d_src_arr, d_tgt_arr);
    // Allocate array of number of connections per source-target couple
    // and initialize it to 0
    int64_t *d_src_tgt_conn_num;
    CUDAMALLOCCTRL("&d_src_tgt_conn_num",&d_src_tgt_conn_num, (n_src_tgt + 1)*sizeof(int64_t));
    gpuErrchk(cudaMemset(d_src_tgt_conn_num, 0,
			 (n_src_tgt + 1)*sizeof(int64_t)));

    // Count number of connections per source-target couple
    CountConnectionsKernel<<<(NConn+1023)/1024, 1024>>>
      (NConn, n_source, n_target, d_src_tgt_arr, d_src_tgt_conn_num, syn_group);
    // Evaluate exclusive sum of connections per source-target couple
    // Allocate array for cumulative sum
    int64_t *d_src_tgt_conn_cumul;
    CUDAMALLOCCTRL("&d_src_tgt_conn_cumul",&d_src_tgt_conn_cumul,
			 (n_src_tgt + 1)*sizeof(int64_t));
    // Determine temporary device storage requirements
    void *d_storage = NULL;
    size_t storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_storage, storage_bytes,
				  d_src_tgt_conn_num,
				  d_src_tgt_conn_cumul,
				  n_src_tgt + 1);
    // Allocate temporary storage
    CUDAMALLOCCTRL("&d_storage",&d_storage, storage_bytes);
    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(d_storage, storage_bytes,
				  d_src_tgt_conn_num,
				  d_src_tgt_conn_cumul,
				  n_src_tgt + 1);
    CUDAFREECTRL("d_storage",d_storage);
    
    // The last element is the total number of required connection Ids
    cudaMemcpy(&n_conn_ids, &d_src_tgt_conn_cumul[n_src_tgt],
	       sizeof(int64_t), cudaMemcpyDeviceToHost);
    
    if (n_conn_ids > 0) {
      // Allocate array of connection indexes
      CUDAMALLOCCTRL("&d_conn_ids",&d_conn_ids, n_conn_ids*sizeof(int64_t));  
      // Set number of connections per source-target couple to 0 again
      gpuErrchk(cudaMemset(d_src_tgt_conn_num, 0,
			   (n_src_tgt + 1)*sizeof(int64_t)));
      // Fill array of connection indexes
      SetConnectionsIndexKernel<<<(NConn+1023)/1024, 1024>>>
	(NConn, n_source, n_target, d_src_tgt_arr, d_src_tgt_conn_num,
	 d_src_tgt_conn_cumul, syn_group, d_conn_ids);

      /// check if allocating with new is more appropriate
      h_conn_ids = (int64_t*)malloc(n_conn_ids*sizeof(int64_t));
      gpuErrchk(cudaMemcpy(h_conn_ids, d_conn_ids,
			   n_conn_ids*sizeof(int64_t),
			   cudaMemcpyDeviceToHost));
	
      CUDAFREECTRL("d_src_tgt_arr",d_src_tgt_arr);
      CUDAFREECTRL("d_src_tgt_conn_num",d_src_tgt_conn_num);
      CUDAFREECTRL("d_src_tgt_conn_cumul",d_src_tgt_conn_cumul);
      CUDAFREECTRL("d_conn_ids",d_conn_ids);
    }
  }
  *n_conn = n_conn_ids;
  
  return h_conn_ids;
}

//////////////////////////////////////////////////////////////////////
// CUDA Kernel that gets all parameters of an array of n_conn connections,
// identified by the indexes conn_ids[i], and puts them in the arrays
// i_source, i_target, port, syn_group, delay, weight
//////////////////////////////////////////////////////////////////////
__global__ void GetConnectionStatusKernel
(int64_t *conn_ids, int64_t n_conn, int *i_source, int *i_target,
 int *port, unsigned char *syn_group, float *delay, float *weight)
{
  int64_t i_arr = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; 
  if (i_arr >= n_conn) return;

   // get connection index, connection block index and index within block
  int64_t i_conn = conn_ids[i_arr];
  uint i_block = (uint)(i_conn / ConnBlockSize);
  int64_t i_block_conn = i_conn % ConnBlockSize;
  // get connection structure
  connection_struct conn = ConnectionArray[i_block][i_block_conn];
  // Get joined target-port parameter, then target index and port index
  uint target_port = conn.target_port;
  i_target[i_arr] = target_port >> MaxPortNBits;
  port[i_arr] = target_port & PortMask;
  // Get weight and synapse group
  weight[i_arr] = conn.weight;
  syn_group[i_arr] = conn.syn_group;
  // Get joined source-delay parameter, then source index and delay
  uint source_delay = SourceDelayArray[i_block][i_block_conn];
  i_source[i_arr] = source_delay >> MaxPortNBits;
  int i_delay = source_delay & PortMask;
  delay[i_arr] = NESTGPUTimeResolution * i_delay;
}

//////////////////////////////////////////////////////////////////////
// CUDA Kernel that gets a float parameter of an array of n_conn connections,
// identified by the indexes conn_ids[i], and puts it in the array
// param_arr
//////////////////////////////////////////////////////////////////////
__global__ void GetConnectionFloatParamKernel
(int64_t *conn_ids, int64_t n_conn, float *param_arr, int i_param)
{
  int64_t i_arr = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; 
  if (i_arr >= n_conn) return;

   // get connection index, connection block index and index within block
  int64_t i_conn = conn_ids[i_arr];
  uint i_block = (uint)(i_conn / ConnBlockSize);
  int64_t i_block_conn = i_conn % ConnBlockSize;
  // get connection structure
  connection_struct conn = ConnectionArray[i_block][i_block_conn];
  switch (i_param) {
  case i_weight_param: {
    param_arr[i_arr] = conn.weight;
    break;
  }
  case i_delay_param: {
    // Get joined source-delay parameter, then delay
    uint source_delay = SourceDelayArray[i_block][i_block_conn];
    int i_delay = source_delay & PortMask;
    param_arr[i_arr] = NESTGPUTimeResolution * i_delay;
    break;
  }
  }
}

//////////////////////////////////////////////////////////////////////
// CUDA Kernel that gets an integer parameter of an array of n_conn connections,
// identified by the indexes conn_ids[i], and puts it in the array
// param_arr
//////////////////////////////////////////////////////////////////////
__global__ void GetConnectionIntParamKernel
(int64_t *conn_ids, int64_t n_conn, int *param_arr, int i_param)
{
  int64_t i_arr = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; 
  if (i_arr >= n_conn) return;

   // get connection index, connection block index and index within block
  int64_t i_conn = conn_ids[i_arr];
  uint i_block = (uint)(i_conn / ConnBlockSize);
  int64_t i_block_conn = i_conn % ConnBlockSize;
  // get connection structure
  connection_struct conn = ConnectionArray[i_block][i_block_conn];
  switch (i_param) {
  case i_source_param: {
    // Get joined source-delay parameter, then source index and delay
    uint source_delay = SourceDelayArray[i_block][i_block_conn];
    param_arr[i_arr] = source_delay >> MaxPortNBits;
    break;
  }
  case i_target_param: {
    // Get joined target-port parameter, then target index
    param_arr[i_arr] = conn.target_port >> MaxPortNBits;
    break;
  }
  case i_port_param: {
    // Get joined target-port parameter, then port index
    param_arr[i_arr] = conn.target_port & PortMask;
    break;
  }
  case i_syn_group_param: {
    // Get synapse group
    param_arr[i_arr] = conn.syn_group;
    break;
  }
  }
}

//////////////////////////////////////////////////////////////////////
// CUDA Kernel that sets a float parameter of an array of n_conn connections,
// identified by the indexes conn_ids[i], using values from the array
// param_arr
//////////////////////////////////////////////////////////////////////
__global__ void SetConnectionFloatParamKernel
(int64_t *conn_ids, int64_t n_conn, float *param_arr, int i_param)
{
  int64_t i_arr = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; 
  if (i_arr >= n_conn) return;

   // get connection index, connection block index and index within block
  int64_t i_conn = conn_ids[i_arr];
  uint i_block = (uint)(i_conn / ConnBlockSize);
  int64_t i_block_conn = i_conn % ConnBlockSize;
  // get connection structure
  connection_struct *conn = &ConnectionArray[i_block][i_block_conn];
  switch (i_param) {
  case i_weight_param: {
    conn->weight = param_arr[i_arr]; 
    break;
  }
  }
}

//////////////////////////////////////////////////////////////////////
// CUDA Kernel that sets a float parameter of an array of n_conn connections,
// identified by the indexes conn_ids[i], to the value val
//////////////////////////////////////////////////////////////////////
__global__ void SetConnectionFloatParamKernel
(int64_t *conn_ids, int64_t n_conn, float val, int i_param)
{
  int64_t i_arr = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; 
  if (i_arr >= n_conn) return;

   // get connection index, connection block index and index within block
  int64_t i_conn = conn_ids[i_arr];
  uint i_block = (uint)(i_conn / ConnBlockSize);
  int64_t i_block_conn = i_conn % ConnBlockSize;
  // get connection structure
  connection_struct *conn = &ConnectionArray[i_block][i_block_conn];
  switch (i_param) {
  case i_weight_param: {
    conn->weight = val; 
    break;
  }
  }
}

//////////////////////////////////////////////////////////////////////
// CUDA Kernel that sets an integer parameter of an array of n_conn connections,
// identified by the indexes conn_ids[i], using values from the array
// param_arr
//////////////////////////////////////////////////////////////////////
__global__ void SetConnectionIntParamKernel
(int64_t *conn_ids, int64_t n_conn, int *param_arr, int i_param)
{
  int64_t i_arr = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; 
  if (i_arr >= n_conn) return;

   // get connection index, connection block index and index within block
  int64_t i_conn = conn_ids[i_arr];
  uint i_block = (uint)(i_conn / ConnBlockSize);
  int64_t i_block_conn = i_conn % ConnBlockSize;
  // get connection structure
  connection_struct *conn = &ConnectionArray[i_block][i_block_conn];
  switch (i_param) {
  case i_target_param: {
    // Get port index from joined target-port parameter
    int i_port = conn->target_port & PortMask;
    // Set joined target-port parameter
    conn->target_port = (param_arr[i_arr] << MaxPortNBits) | i_port;
    break;
  }
  case i_port_param: {
    // Get target index from joined target-port parameter
    int i_target = conn->target_port >> MaxPortNBits;
    // Set joined target-port parameter
    conn->target_port = (i_target << MaxPortNBits) | param_arr[i_arr];
    break;
  }
  case i_syn_group_param: {
    // Set synapse group
    conn->syn_group = param_arr[i_arr]; 
    break;
  }
  }
}

//////////////////////////////////////////////////////////////////////
// CUDA Kernel that sets an integer parameter of an array of n_conn connections,
// identified by the indexes conn_ids[i], to the value val
//////////////////////////////////////////////////////////////////////
__global__ void SetConnectionIntParamKernel
(int64_t *conn_ids, int64_t n_conn, int val, int i_param)
{
  int64_t i_arr = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; 
  if (i_arr >= n_conn) return;

   // get connection index, connection block index and index within block
  int64_t i_conn = conn_ids[i_arr];
  uint i_block = (uint)(i_conn / ConnBlockSize);
  int64_t i_block_conn = i_conn % ConnBlockSize;
  // get connection structure
  connection_struct *conn = &ConnectionArray[i_block][i_block_conn];
  switch (i_param) {
  case i_target_param: {
    // Get port index from joined target-port parameter
    int i_port = conn->target_port & PortMask;
    // Set joined target-port parameter
    conn->target_port = (val << MaxPortNBits) | i_port;
    break;
  }
  case i_port_param: {
    // Get target index from joined target-port parameter
    int i_target = conn->target_port >> MaxPortNBits;
    // Set joined target-port parameter
    conn->target_port = (i_target << MaxPortNBits) | val;
    break;
  }
  case i_syn_group_param: {
    // Set synapse group
    conn->syn_group = val; 
    break;
  }
  }
}


//////////////////////////////////////////////////////////////////////
// Get all parameters of an array of n_conn connections, identified by
// the indexes conn_ids[i], and put them in the arrays
// i_source, i_target, port, syn_group, delay, weight
// NOTE: host arrays should be pre-allocated to store n_conn elements
//////////////////////////////////////////////////////////////////////
int NESTGPU::GetConnectionStatus(int64_t *conn_ids, int64_t n_conn,
				 int *i_source, int *i_target, int *port,
				 unsigned char *syn_group, float *delay,
				 float *weight)
{
  if (n_conn > 0) {
    // declare pointers to arrays in device memory
    int64_t *d_conn_ids;
    int *d_source;
    int *d_target;
    int *d_port;
    unsigned char *d_syn_group;
    float *d_delay;
    float *d_weight;

    // allocate array of connection ids in device memory
    // and copy the ids from host to device array
    CUDAMALLOCCTRL("&d_conn_ids",&d_conn_ids, n_conn*sizeof(int64_t));
    gpuErrchk(cudaMemcpy(d_conn_ids, conn_ids, n_conn*sizeof(int64_t),
			 cudaMemcpyHostToDevice));

    // allocate arrays of connection parameters in device memory
    CUDAMALLOCCTRL("&d_source",&d_source, n_conn*sizeof(int));
    CUDAMALLOCCTRL("&d_target",&d_target, n_conn*sizeof(int));
    CUDAMALLOCCTRL("&d_port",&d_port, n_conn*sizeof(int));
    CUDAMALLOCCTRL("&d_syn_group",&d_syn_group, n_conn*sizeof(unsigned char));
    CUDAMALLOCCTRL("&d_delay",&d_delay, n_conn*sizeof(float));
    CUDAMALLOCCTRL("&d_weight",&d_weight, n_conn*sizeof(float));
    // host arrays
    
    // launch kernel to get connection parameters
    GetConnectionStatusKernel<<<(n_conn+1023)/1024, 1024 >>>
      (d_conn_ids, n_conn, d_source, d_target, d_port, d_syn_group,
       d_delay, d_weight);

    // copy connection parameters from device to host memory
    gpuErrchk(cudaMemcpy(i_source, d_source, n_conn*sizeof(int),
			 cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(i_target, d_target, n_conn*sizeof(int),
			 cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(port, d_port, n_conn*sizeof(int),
			 cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(syn_group, d_syn_group,
			 n_conn*sizeof(unsigned char),
			 cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(delay, d_delay, n_conn*sizeof(float),
			 cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(weight, d_weight, n_conn*sizeof(float),
			 cudaMemcpyDeviceToHost));
  }
  
  return 0;
}


// Get the index of the connection float parameter param_name
// if param_name is not a float parameter, return -1
int NESTGPU::GetConnectionFloatParamIndex(std::string param_name)
{
  for (int i=0; i<N_CONN_FLOAT_PARAM; i++) {
    if (param_name==ConnectionFloatParamName[i]) {
      return i;
    }
  }
  
  return -1;
}

// Get the index of the connection int parameter param_name
// if param_name is not an int parameter, return -1
int NESTGPU::GetConnectionIntParamIndex(std::string param_name)
{
  for (int i=0; i<N_CONN_INT_PARAM; i++) {
    if (param_name==ConnectionIntParamName[i]) {
      return i;
    }
  }
  
  return -1;
}

// Check if param_name is a connection float parameter
int NESTGPU::IsConnectionFloatParam(std::string param_name)
{
  if (GetConnectionFloatParamIndex(param_name) >=0 ) {
    return 1;
  }
  else {
    return 0;
  }
}

// Check if param_name is a connection integer parameter
int NESTGPU::IsConnectionIntParam(std::string param_name)
{
  if (GetConnectionIntParamIndex(param_name) >=0 ) {
    return 1;
  }
  else {
    return 0;
  }
}

//////////////////////////////////////////////////////////////////////
// Get the float parameter param_name of an array of n_conn connections,
// identified by the indexes conn_ids[i], and put it in the array
// h_param_arr
// NOTE: host array should be pre-allocated to store n_conn elements
//////////////////////////////////////////////////////////////////////
int NESTGPU::GetConnectionFloatParam(int64_t *conn_ids, int64_t n_conn,
				     float *h_param_arr,
				     std::string param_name)
{
  // Check if param_name is a connection float parameter
  int i_param = GetConnectionFloatParamIndex(param_name);
  if (i_param < 0) {
    throw ngpu_exception(std::string("Unrecognized connection float parameter ")
			 + param_name);
  }
  if (n_conn > 0) {
    // declare pointers to arrays in device memory
    int64_t *d_conn_ids;
    float *d_arr;
    // allocate array of connection ids in device memory
    // and copy the ids from host to device array
    CUDAMALLOCCTRL("&d_conn_ids",&d_conn_ids, n_conn*sizeof(int64_t));
    gpuErrchk(cudaMemcpy(d_conn_ids, conn_ids, n_conn*sizeof(int64_t),
			 cudaMemcpyHostToDevice));
    
    // allocate connection parameter array in device memory
    CUDAMALLOCCTRL("&d_arr",&d_arr, n_conn*sizeof(float));
    
    // launch kernel to get connection parameters
    GetConnectionFloatParamKernel<<<(n_conn+1023)/1024, 1024 >>>
      (d_conn_ids, n_conn, d_arr, i_param);
    
    // copy connection parameter array from device to host memory
    gpuErrchk(cudaMemcpy(h_param_arr, d_arr, n_conn*sizeof(float),
			 cudaMemcpyDeviceToHost));
    // free allocated device memory
    CUDAFREECTRL("d_conn_ids",d_conn_ids);
    CUDAFREECTRL("d_arr",d_arr);
  }
  
  return 0;
}

//////////////////////////////////////////////////////////////////////
// Get the integer parameter param_name of an array of n_conn connections,
// identified by the indexes conn_ids[i], and put it in the array
// h_param_arr
// NOTE: host array should be pre-allocated to store n_conn elements
//////////////////////////////////////////////////////////////////////
int NESTGPU::GetConnectionIntParam(int64_t *conn_ids, int64_t n_conn,
				   int *h_param_arr,
				   std::string param_name)
{
  // Check if param_name is a connection integer parameter
  int i_param = GetConnectionIntParamIndex(param_name);
  if (i_param < 0) {
    throw ngpu_exception(std::string("Unrecognized connection "
				     "integer parameter ") + param_name);
  }
  if (n_conn > 0) {
    // declare pointers to arrays in device memory
    int64_t *d_conn_ids;
    int *d_arr;
    // allocate array of connection ids in device memory
    // and copy the ids from host to device array
    CUDAMALLOCCTRL("&d_conn_ids",&d_conn_ids, n_conn*sizeof(int64_t));
    gpuErrchk(cudaMemcpy(d_conn_ids, conn_ids, n_conn*sizeof(int64_t),
			 cudaMemcpyHostToDevice));
    
    // allocate connection parameter array in device memory
    CUDAMALLOCCTRL("&d_arr",&d_arr, n_conn*sizeof(int));
    
    // launch kernel to get connection parameters
    GetConnectionIntParamKernel<<<(n_conn+1023)/1024, 1024 >>>
      (d_conn_ids, n_conn, d_arr, i_param);
    
    // copy connection parameter array from device to host memory
    gpuErrchk(cudaMemcpy(h_param_arr, d_arr, n_conn*sizeof(int),
			 cudaMemcpyDeviceToHost));
    // free allocated device memory
    CUDAFREECTRL("d_conn_ids",d_conn_ids);
    CUDAFREECTRL("d_arr",d_arr);
  }
  
  return 0;
}


//////////////////////////////////////////////////////////////////////
// Set the float parameter param_name of an array of n_conn connections,
// identified by the indexes conn_ids[i], using values from a distribution
// or from an array
//////////////////////////////////////////////////////////////////////
int NESTGPU::SetConnectionFloatParamDistr(int64_t *conn_ids, int64_t n_conn,
					  std::string param_name)
{
  // Check if param_name is a connection float parameter
  int i_param = GetConnectionFloatParamIndex(param_name);
  if (i_param < 0) {
    throw ngpu_exception(std::string("Unrecognized connection float parameter ")
			 + param_name);
  }
  if (i_param == i_delay_param) {
    throw ngpu_exception("Connection delay cannot be modified");
  }

  if (n_conn > 0) {
    // declare pointers to arrays in device memory
    int64_t *d_conn_ids;
    // allocate array of connection ids in device memory
    // and copy the ids from host to device array
    CUDAMALLOCCTRL("&d_conn_ids",&d_conn_ids, n_conn*sizeof(int64_t));
    gpuErrchk(cudaMemcpy(d_conn_ids, conn_ids, n_conn*sizeof(int64_t),
			 cudaMemcpyHostToDevice));
    
    // get values from array or distribution
    float *d_arr = distribution_->getArray
      (conn_random_generator_[this_host_][this_host_], n_conn);
    // launch kernel to set connection parameters
    SetConnectionFloatParamKernel<<<(n_conn+1023)/1024, 1024 >>>
      (d_conn_ids, n_conn, d_arr, i_param);
    // free allocated device memory
    CUDAFREECTRL("d_conn_ids",d_conn_ids);
    CUDAFREECTRL("d_arr",d_arr);
  }
  
  return 0;
}

//////////////////////////////////////////////////////////////////////
// Set the float parameter param_name of an array of n_conn connections,
// identified by the indexes conn_ids[i], to the value val
//////////////////////////////////////////////////////////////////////
int NESTGPU::SetConnectionFloatParam(int64_t *conn_ids, int64_t n_conn,
				     float val,
				     std::string param_name)
{
  // Check if param_name is a connection float parameter
  int i_param = GetConnectionFloatParamIndex(param_name);
  if (i_param < 0) {
    throw ngpu_exception(std::string("Unrecognized connection float parameter ")
			 + param_name);
  }
  if (i_param == i_delay_param) {
        throw ngpu_exception("Connection delay cannot be modified");
  }

  if (n_conn > 0) {
    // declare pointers to arrays in device memory
    int64_t *d_conn_ids;
    // allocate array of connection ids in device memory
    // and copy the ids from host to device array
    CUDAMALLOCCTRL("&d_conn_ids",&d_conn_ids, n_conn*sizeof(int64_t));
    gpuErrchk(cudaMemcpy(d_conn_ids, conn_ids, n_conn*sizeof(int64_t),
			 cudaMemcpyHostToDevice));
        
    // launch kernel to set connection parameters
    SetConnectionFloatParamKernel<<<(n_conn+1023)/1024, 1024 >>>
      (d_conn_ids, n_conn, val, i_param);
    // free allocated device memory
    CUDAFREECTRL("d_conn_ids",d_conn_ids);    
  }
  
  return 0;
}


//////////////////////////////////////////////////////////////////////
// Set the integer parameter param_name of an array of n_conn connections,
// identified by the indexes conn_ids[i], using the values from the array
// h_param_arr
//////////////////////////////////////////////////////////////////////
int NESTGPU::SetConnectionIntParamArr(int64_t *conn_ids, int64_t n_conn,
				      int *h_param_arr,
				      std::string param_name)
{
  // Check if param_name is a connection int parameter
  int i_param = GetConnectionIntParamIndex(param_name);
  if (i_param < 0) {
    throw ngpu_exception(std::string("Unrecognized connection int parameter ")
			 + param_name);
  }
  if (i_param == i_source_param) {
    throw ngpu_exception("Connection source node cannot be modified");
  }

  if (n_conn > 0) {
    // declare pointers to arrays in device memory
    int64_t *d_conn_ids;
    int *d_arr;
    // allocate array of connection ids in device memory
    // and copy the ids from host to device array
    CUDAMALLOCCTRL("&d_conn_ids",&d_conn_ids, n_conn*sizeof(int64_t));
    gpuErrchk(cudaMemcpy(d_conn_ids, conn_ids, n_conn*sizeof(int64_t),
			 cudaMemcpyHostToDevice));
    
    // allocate connection parameter array in device memory
    CUDAMALLOCCTRL("&d_arr",&d_arr, n_conn*sizeof(int));

    // copy connection parameter array from host to device memory
    gpuErrchk(cudaMemcpy(d_arr, h_param_arr, n_conn*sizeof(int),
			 cudaMemcpyHostToDevice));
    
    // launch kernel to set connection parameters
    SetConnectionIntParamKernel<<<(n_conn+1023)/1024, 1024 >>>
      (d_conn_ids, n_conn, d_arr, i_param);
    // free allocated device memory
    CUDAFREECTRL("d_conn_ids",d_conn_ids);
    CUDAFREECTRL("d_arr",d_arr);

  }
  
  return 0;
}

//////////////////////////////////////////////////////////////////////
// Set the int parameter param_name of an array of n_conn connections,
// identified by the indexes conn_ids[i], to the value val
//////////////////////////////////////////////////////////////////////
int NESTGPU::SetConnectionIntParam(int64_t *conn_ids, int64_t n_conn,
				   int val, std::string param_name)
{
  // Check if param_name is a connection float parameter
  int i_param = GetConnectionIntParamIndex(param_name);
  if (i_param < 0) {
    throw ngpu_exception(std::string("Unrecognized connection int parameter ")
			 + param_name);
  }
  if (i_param == i_source_param) {
    throw ngpu_exception("Connection source node cannot be modified");
  }

  if (n_conn > 0) {
    // declare pointers to arrays in device memory
    int64_t *d_conn_ids;
    // allocate array of connection ids in device memory
    // and copy the ids from host to device array
    CUDAMALLOCCTRL("&d_conn_ids",&d_conn_ids, n_conn*sizeof(int64_t));
    gpuErrchk(cudaMemcpy(d_conn_ids, conn_ids, n_conn*sizeof(int64_t),
			 cudaMemcpyHostToDevice));
        
    // launch kernel to set connection parameters
    SetConnectionIntParamKernel<<<(n_conn+1023)/1024, 1024 >>>
      (d_conn_ids, n_conn, val, i_param);
    // free allocated device memory
    CUDAFREECTRL("d_conn_ids",d_conn_ids);
  }
  
  return 0;
}

