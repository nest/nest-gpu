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

//#include <time.h>
//#include <sys/time.h>
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
//#include "copass_kernels.h"
//#include "copass_sort.h"
//#include "distribution.h"
#include "connect.h"
#include "nestgpu.h"
//#include "utilities.h"

//#define OPTIMIZE_FOR_MEMORY

//extern __constant__ float NESTGPUTimeResolution;

bool print_sort_err = true;
bool print_sort_cfr = false;
bool compare_with_serial = false;
uint last_i_sub = 0;

int h_MaxNodeNBits;
__device__ int MaxNodeNBits;
// maximum number of bits used to represent node index 

int h_MaxDelayNBits;
__device__ int MaxDelayNBits;
// maximum number of bits used to represent delays

int h_MaxSynNBits;
__device__ int MaxSynNBits;
// maximum number of bits used to represent synapse group index

int h_MaxPortNBits;
__device__ int MaxPortNBits;
// maximum number of bits used to represent receptor port index

int h_MaxPortSynNBits;
__device__ int MaxPortSynNBits;
// maximum number of bits used to represent receptor port index
// and synapse group index


uint h_SourceMask;
__device__ uint SourceMask;
// bit mask used to extract source node index

uint h_DelayMask;
__device__ uint DelayMask;
// bit mask used to extract delay

uint h_TargetMask;
__device__ uint TargetMask;
// bit mask used to extract target node index

uint h_SynMask;
__device__ uint SynMask;
// bit mask used to extract synapse group index

uint h_PortMask;
__device__ uint PortMask;
// bit mask used to extract port index

uint h_PortSynMask;
__device__ uint PortSynMask;
// bit mask used to extract port and synapse group index



iconngroup_t *d_ConnGroupIdx0;
__device__ iconngroup_t *ConnGroupIdx0;
// ig0 = ConnGroupIdx0[i_spike_buffer] is the index in the whole
// connection-group array of the first connection group outgoing
// from the node i_spike_buffer

int64_t *d_ConnGroupIConn0;
__device__ int64_t *ConnGroupIConn0;
// i_conn0 = ConnGroupIConn0[ig] with ig = 0, ..., Ng
//  is the index in the whole connection array of the first connection
// belonging to the connection group ig

int *d_ConnGroupDelay;
__device__ int *ConnGroupDelay;
// ConnGroupDelay[ig]
// delay associated to all connections of the connection group ig
// with ig = 0, ..., Ng

iconngroup_t tot_conn_group_num;

int64_t NConn; // total number of connections in the whole network

int64_t h_ConnBlockSize = 10000000; // 160000000; //50000000;
__device__ int64_t ConnBlockSize;
// size (i.e. number of connections) of connection blocks 

int h_MaxDelayNum;


// it seems that there is no relevant advantage in using a constant array
// however better to keep this option ready and commented
//std::vector<int*> ConnKeyVect;
//int** d_ConnKeyArray;
//__device__ uint** ConnKeyArray;
//__constant__ uint* ConnKeyArray[1024];
std::vector<void*> ConnKeyVect;
void* d_ConnKeyArray;
__device__ void* ConnKeyArray;
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
//std::vector<connection_struct*> ConnStructVect;
//connection_struct** d_ConnStructArray;
//__device__ connection_struct** ConnStructArray;
//__constant__ connection_struct* ConnStructArray[1024];
std::vector<void*> ConnStructVect;
void* d_ConnStructArray;
__device__ void* ConnStructArray;
// array of target node indexes, receptor port index, synapse type,
// weight of all connections
// used as a value for key-value sorting of the connections (see above)



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


__global__ void setConnGroupNum(inode_t n_compact,
				iconngroup_t *conn_group_num,
				iconngroup_t *conn_group_idx0_compact,
				inode_t *conn_group_source_compact)
{
  inode_t i_compact = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_compact>=n_compact) return;
  inode_t source = conn_group_source_compact[i_compact];
  iconngroup_t num = conn_group_idx0_compact[i_compact+1]
    - conn_group_idx0_compact[i_compact];
  conn_group_num[source] = num;
}


__global__ void setConnGroupIConn0(int64_t n_block_conn,
				   int *conn_group_iconn0_mask,
				   iconngroup_t *conn_group_iconn0_mask_cumul,
				   int64_t *conn_group_iconn0, int64_t i_conn0,
				   iconngroup_t *offset)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_block_conn) return;
  if (conn_group_iconn0_mask[i_conn] != 0) {
    iconngroup_t pos = conn_group_iconn0_mask_cumul[i_conn] + *offset;
    conn_group_iconn0[pos] = i_conn0 + i_conn;
  }
}


__global__ void ConnectInitKernel(iconngroup_t *conn_group_idx0,
				  int64_t *conn_group_iconn0,
				  int *conn_group_delay,
				  int64_t block_size,
				  void *conn_key_array,
				  void *conn_struct_array)
{
  ConnGroupIdx0 = conn_group_idx0;
  ConnGroupIConn0 = conn_group_iconn0;
  ConnGroupDelay = conn_group_delay;
  ConnBlockSize = block_size;
  ConnKeyArray = conn_key_array;
  ConnStructArray = conn_struct_array;
}

int ConnectInit()
{
  /*
  int k = ConnStructVect.size();
  int **d_conn_key_array;
  CUDAMALLOCCTRL("&d_conn_key_array",&d_conn_key_array, k*sizeof(int*));
  gpuErrchk(cudaMemcpy(d_conn_key_array, ConnKeyVect.data(),
		       k*sizeof(int*), cudaMemcpyHostToDevice));
  
  connection_struct **d_connection_array;
  CUDAMALLOCCTRL("&d_connection_array",&d_connection_array, k*sizeof(connection_struct*));
  gpuErrchk(cudaMemcpy(d_connection_array, ConnStructVect.data(),
		       k*sizeof(connection_struct*), cudaMemcpyHostToDevice));

  */
  ConnectInitKernel<<<1,1>>>(d_ConnGroupIdx0, d_ConnGroupIConn0,
			     d_ConnGroupDelay, h_ConnBlockSize,
			     d_ConnKeyArray,
			     d_ConnStructArray);
  DBGCUDASYNC
    
    return 0;
}


__global__ void setMaxNodeNBitsKernel(int max_node_nbits,
				      int max_port_syn_nbits,
				      int max_delay_nbits,
				      int max_port_nbits,
				      uint port_syn_mask,
				      uint delay_mask,
				      uint source_mask,
				      uint target_mask,
				      uint port_mask)
{
  MaxNodeNBits = max_node_nbits;
  MaxPortSynNBits = max_port_syn_nbits;
  MaxDelayNBits = max_delay_nbits;
  MaxPortNBits = max_port_nbits;
  PortSynMask = port_syn_mask;
  DelayMask = delay_mask;
  SourceMask = source_mask;
  TargetMask = target_mask;
  PortMask = port_mask;
}

__global__ void setMaxSynNBitsKernel(int max_syn_nbits,
				     int max_port_nbits,
				     uint syn_mask,
				     uint port_mask)
{
  MaxSynNBits = max_syn_nbits;
  MaxPortNBits = max_port_nbits;
  SynMask = syn_mask;
  PortMask = port_mask;
}

int setMaxNodeNBits(int max_node_nbits)
{
  h_MaxNodeNBits = max_node_nbits;
  h_MaxPortSynNBits = 32 - h_MaxNodeNBits;
  h_MaxDelayNBits = h_MaxPortSynNBits;
  h_MaxPortNBits = h_MaxPortSynNBits - h_MaxSynNBits - 1; 
  h_PortSynMask = (1 << h_MaxPortSynNBits) - 1;
  h_DelayMask = h_PortSynMask;
  h_SourceMask = ~h_DelayMask;
  h_TargetMask = h_SourceMask;
  h_PortMask = ((1 << h_MaxPortNBits) - 1) << (h_MaxSynNBits + 1);
  setMaxNodeNBitsKernel<<<1,1>>>
    (h_MaxNodeNBits, h_MaxPortSynNBits, h_MaxDelayNBits, h_MaxPortNBits,
     h_PortSynMask, h_DelayMask, h_SourceMask, h_TargetMask, h_PortMask);
  
  DBGCUDASYNC

  return 0;
}  

int setMaxSynNBits(int max_syn_nbits)
{
  h_MaxSynNBits = max_syn_nbits;
  h_MaxPortNBits = h_MaxPortSynNBits - h_MaxSynNBits - 1; 
  h_SynMask = (1 << h_MaxSynNBits) - 1;
  h_PortMask = ((1 << h_MaxPortNBits) - 1) << (h_MaxSynNBits + 1);
  setMaxSynNBitsKernel<<<1,1>>>(h_MaxSynNBits, h_MaxPortNBits,
				h_SynMask, h_PortMask);
  DBGCUDASYNC

  return 0;
}  


__global__ void setSourceTargetIndexKernel(uint64_t n_src_tgt, uint n_source,
					   uint n_target,
					   uint64_t *d_src_tgt_arr,
					   uint *d_src_arr, uint *d_tgt_arr)
{
  uint64_t i_src_tgt = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; 
  if (i_src_tgt >= n_src_tgt) return;
  uint i_src =(uint)(i_src_tgt / n_target);
  uint i_tgt =(uint)(i_src_tgt % n_target);
  uint src_id = d_src_arr[i_src];
  uint tgt_id = d_tgt_arr[i_tgt];
  uint64_t src_tgt_id = ((uint64_t)src_id << 32) | tgt_id;
  d_src_tgt_arr[i_src_tgt] = src_tgt_id;
  //printf("i_src_tgt %lld\tsrc_id %d\ttgt_id %d\tsrc_tgt_id %lld\n", 
  //	 i_src_tgt, src_id, tgt_id, src_tgt_id); 
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

template
int64_t *NESTGPU::GetConnections<conn12b_key, conn12b_struct>
(inode_t *i_source_pt, inode_t n_source,
 inode_t *i_target_pt, inode_t n_target,
 int syn_group, int64_t *n_conn);

template
int NESTGPU::GetConnectionStatus<conn12b_key, conn12b_struct>
(int64_t *conn_ids, int64_t n_conn,
 inode_t *source, inode_t *target, int *port,
 int *syn_group, float *delay,
 float *weight);

template
int NESTGPU::SetConnectionIntParam<conn12b_key, conn12b_struct>
(int64_t *conn_ids, int64_t n_conn, int val, std::string param_name);

template
int NESTGPU::SetConnectionIntParamArr<conn12b_key, conn12b_struct>
(int64_t *conn_ids, int64_t n_conn, int *h_param_arr,
 std::string param_name);

template
int NESTGPU::SetConnectionFloatParam<conn12b_key, conn12b_struct>
(int64_t *conn_ids, int64_t n_conn, float val,
 std::string param_name);

template
int NESTGPU::SetConnectionFloatParamDistr<conn12b_key, conn12b_struct>
(int64_t *conn_ids, int64_t n_conn, std::string param_name);

template
int NESTGPU::GetConnectionIntParam<conn12b_key, conn12b_struct>
(int64_t *conn_ids, int64_t n_conn, int *h_param_arr,
 std::string param_name);

template
int NESTGPU::GetConnectionFloatParam<conn12b_key, conn12b_struct>
(int64_t *conn_ids, int64_t n_conn, float *h_param_arr,
 std::string param_name);
