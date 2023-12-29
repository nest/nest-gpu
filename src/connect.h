/*
 *  connect.h
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

#ifndef CONNECT_H
#define CONNECT_H

#include <time.h>
#include <sys/time.h>
#include <curand.h>
#include <cub/cub.cuh>
#include <vector>

#include "cuda_error.h"
#include "copass_kernels.h"
#include "copass_sort.h"
#include "connect_spec.h"
#include "nestgpu.h"
#include "distribution.h"
#include "utilities.h"

extern __constant__ float NESTGPUTimeResolution;

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

//struct connection_struct
//{
//  int target_port_syn;
//  float weight;
  //// unsigned char syn_group;
//};

struct conn12b_struct
{
  uint target_port_syn;
  float weight;
};

extern int h_MaxNodeNBits;
extern __device__ int MaxNodeNBits;
// maximum number of bits used to represent node index 

extern int h_MaxDelayNBits;
extern __device__ int MaxDelayNBits;
// maximum number of bits used to represent delays

extern int h_MaxSynNBits;
extern __device__ int MaxSynNBits;
// maximum number of bits used to represent synapse group index

extern int h_MaxPortNBits;
extern __device__ int MaxPortNBits;
// maximum number of bits used to represent receptor port index

extern int h_MaxPortSynNBits;
extern __device__ int MaxPortSynNBits;
// maximum number of bits used to represent receptor port index
// and synapse group index


extern uint h_SourceMask;
extern __device__ uint SourceMask;
// bit mask used to extract source node index

extern uint h_DelayMask;
extern __device__ uint DelayMask;
// bit mask used to extract delay

extern uint h_TargetMask;
extern __device__ uint TargetMask;
// bit mask used to extract target node index

extern uint h_SynMask;
extern __device__ uint SynMask;
// bit mask used to extract synapse group index

extern uint h_PortMask;
extern __device__ uint PortMask;
// bit mask used to extract port index

extern uint h_PortSynMask;
extern __device__ uint PortSynMask;
// bit mask used to extract port and synapse group index

extern iconngroup_t *d_ConnGroupIdx0;
extern __device__ iconngroup_t *ConnGroupIdx0;

extern int64_t *d_ConnGroupIConn0;
extern __device__ int64_t *ConnGroupIConn0;

extern int *d_ConnGroupDelay;
extern __device__ int *ConnGroupDelay;

extern iconngroup_t tot_conn_group_num;

extern int64_t NConn;

extern int64_t h_ConnBlockSize;
extern __device__ int64_t ConnBlockSize;

extern int h_MaxDelayNum;

// it seems that there is no relevant advantage in using a constant array
// however better to keep this option ready and commented
//extern __constant__ uint* ConnKeyArray[];
extern std::vector<void*> ConnKeyVect;
extern void* d_ConnKeyArray;
extern __device__ void* ConnKeyArray;

//extern std::vector<connection_struct*> ConnStructVect;
//extern connection_struct** d_ConnStructArray;
//extern __device__ connection_struct** ConnStructArray;
//extern __constant__ connection_struct* ConnStructArray[];
extern std::vector<void*> ConnStructVect;
extern void* d_ConnStructArray;
extern __device__ void* ConnStructArray;


typedef uint conn12b_key;

//typedef connection_struct conn12b_struct;

template <class ConnKeyT>
__device__ __forceinline__ void setConnDelay
(ConnKeyT &conn_key, int delay);

template <class ConnKeyT>
__device__ __forceinline__ void setConnSource
(ConnKeyT &conn_key, inode_t source);

template <class ConnKeyT>
inline void hostSetConnSource
(ConnKeyT &conn_key, inode_t source);

template <class ConnStructT>
__device__ __forceinline__ void setConnTarget
(ConnStructT &conn_struct, inode_t target);

template <class ConnKeyT, class ConnStructT>
__device__ __forceinline__ void setConnPort
(ConnKeyT &conn_key, ConnStructT &conn_struct, int port);

template <class ConnKeyT, class ConnStructT>
__device__ __forceinline__ void setConnSyn
(ConnKeyT &conn_key, ConnStructT &conn_struct, int syn);


template <class ConnKeyT>
__device__ __forceinline__ int getConnDelay(const ConnKeyT &conn_key);

template <class ConnKeyT>
__device__ __forceinline__ inode_t getConnSource(ConnKeyT &conn_key);

template <class ConnStructT>
__device__ __forceinline__ inode_t getConnTarget(ConnStructT &conn_struct);

template <class ConnKeyT, class ConnStructT>
__device__ __forceinline__ int getConnPort
(ConnKeyT &conn_key, ConnStructT &conn_struct);

template <class ConnKeyT, class ConnStructT>
__device__ __forceinline__ int getConnSyn
(ConnKeyT &conn_key, ConnStructT &conn_struct);

template <class ConnKeyT, class ConnStructT>
__device__ __forceinline__ bool getConnRemoteFlag
(ConnKeyT &conn_key, ConnStructT &conn_struct);

template <class ConnKeyT, class ConnStructT>
__device__ __forceinline__ void clearConnRemoteFlag
(ConnKeyT &conn_key, ConnStructT &conn_struct);


template <>
__device__ __forceinline__ void
setConnDelay<conn12b_key> 
(conn12b_key &conn_key, int delay) {
  conn_key = (conn_key & (~DelayMask)) | delay;
}

template <>
__device__ __forceinline__ void
setConnSource<conn12b_key> 
(conn12b_key &conn_key, inode_t source) {
  conn_key = (conn_key & (~SourceMask)) | (source << MaxDelayNBits);
}

template <>
inline void hostSetConnSource<conn12b_key> 
(conn12b_key &conn_key, inode_t source) {
  conn_key = (conn_key & (~h_SourceMask)) | (source << h_MaxDelayNBits);
}

template <>
__device__ __forceinline__ void
setConnTarget<conn12b_struct> 
(conn12b_struct &conn, inode_t target) {
  conn.target_port_syn = (conn.target_port_syn & (~TargetMask))
    | (target << MaxPortSynNBits);
}

template <>
__device__ __forceinline__ void
setConnPort<conn12b_key, conn12b_struct> 
(conn12b_key &conn_key, conn12b_struct &conn, int port) {
  conn.target_port_syn = (conn.target_port_syn & (~PortMask))
    | (port << (MaxSynNBits + 1));
}

template <>
__device__ __forceinline__ void
setConnSyn<conn12b_key, conn12b_struct> 
(conn12b_key &conn_key, conn12b_struct &conn, int syn) {
  conn.target_port_syn = (conn.target_port_syn & (~SynMask))
    | syn;
}

template <>
__device__ __forceinline__ int
getConnDelay<conn12b_key> 
(const conn12b_key &conn_key) {
  return conn_key & DelayMask;
}

template <>
__device__ __forceinline__ inode_t
getConnSource<conn12b_key> 
(conn12b_key &conn_key) {
  return (conn_key & SourceMask) >> MaxDelayNBits;
}

template <>
__device__ __forceinline__ inode_t
getConnTarget<conn12b_struct> 
(conn12b_struct &conn) {
  return (conn.target_port_syn & TargetMask) >>  MaxPortSynNBits;
}

template <>
__device__ __forceinline__ int
getConnPort<conn12b_key, conn12b_struct> 
(conn12b_key &conn_key, conn12b_struct &conn) {
  return (conn.target_port_syn & PortMask) >> (MaxSynNBits + 1);
}

template <>
__device__ __forceinline__ int
getConnSyn<conn12b_key, conn12b_struct> 
(conn12b_key &conn_key, conn12b_struct &conn) {
  return conn.target_port_syn & SynMask;
}

// TEMPORARY TO BE IMPROVED!!!!
template <>
__device__ __forceinline__ bool
getConnRemoteFlag<conn12b_key, conn12b_struct> 
(conn12b_key &conn_key, conn12b_struct &conn) {
  return (conn.target_port_syn >> MaxSynNBits) & (uint)1;
}

template <>
__device__ __forceinline__ void
clearConnRemoteFlag<conn12b_key, conn12b_struct> 
(conn12b_key &conn_key, conn12b_struct &conn) {
  conn.target_port_syn = conn.target_port_syn &
    ~((uint)1 << MaxSynNBits);
}


int setMaxNodeNBits(int max_node_nbits);

int setMaxSynNBits(int max_syn_nbits);

template<class ConnKeyT, class ConnStructT>
int allocateNewBlocks(std::vector<void*> &conn_key_vect,
		      std::vector<void*> &conn_struct_vect,
		      int64_t block_size, int new_n_block)
{
  // Allocating GPU memory for new connection blocks
  // allocate new blocks if needed
  for (int ib=conn_key_vect.size(); ib<new_n_block; ib++) {
    ConnKeyT *d_key_pt;
    ConnStructT *d_connection_pt;
    // allocate GPU memory for new blocks 
    CUDAMALLOCCTRL("&d_key_pt",&d_key_pt, block_size*sizeof(ConnKeyT));
    CUDAMALLOCCTRL("&d_connection_pt",&d_connection_pt,
			 block_size*sizeof(ConnStructT));
    conn_key_vect.push_back((void*)d_key_pt);
    conn_struct_vect.push_back((void*)d_connection_pt);
  }

  return 0;
}

template<class ConnKeyT>
int freeConnectionKey()
{
  for (uint ib=0; ib<ConnKeyVect.size(); ib++) {
    ConnKeyT *d_key_pt = (ConnKeyT*)ConnKeyVect[ib];
    if (d_key_pt != NULL) {
      CUDAFREECTRL("d_key_pt", d_key_pt);
    }
  }
  return 0;
}

template<class ConnStructT>
__global__ void setWeights(ConnStructT *conn_struct_subarray, float weight,
			   int64_t n_conn)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) return;
  conn_struct_subarray[i_conn].weight = weight;
}

template<class ConnStructT>
__global__ void setWeights(ConnStructT *conn_struct_subarray, float *arr_val,
			   int64_t n_conn)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) return;
  conn_struct_subarray[i_conn].weight = arr_val[i_conn];
}


template<class ConnKeyT>
__global__ void setDelays(ConnKeyT *conn_key_subarray, float *arr_val,
			  int64_t n_conn, float time_resolution)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) return;
  int delay = (int)round(arr_val[i_conn]/time_resolution);
  delay = max(delay,1);
  setConnDelay<ConnKeyT>(conn_key_subarray[i_conn], delay);
}

template<class ConnKeyT>
__global__ void setDelays(ConnKeyT *conn_key_subarray, float fdelay,
			  int64_t n_conn, float time_resolution)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) return;
  int delay = (int)round(fdelay/time_resolution);
  delay = max(delay,1);
  setConnDelay<ConnKeyT>(conn_key_subarray[i_conn], delay);
}


template <class ConnKeyT, class ConnStructT>
__global__ void setPort(ConnKeyT *conn_key_subarray,
			ConnStructT *conn_struct_subarray, int port,
			int64_t n_conn)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) return;
  setConnPort<ConnKeyT, ConnStructT>(conn_key_subarray[i_conn],
				     conn_struct_subarray[i_conn],
				     port);
}


template <class ConnKeyT, class ConnStructT>
__global__ void setSynGroup(ConnKeyT *conn_key_subarray,
			    ConnStructT *conn_struct_subarray,
			    int syn_group, int64_t n_conn)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) return;
  setConnSyn<ConnKeyT, ConnStructT>(conn_key_subarray[i_conn],
				    conn_struct_subarray[i_conn],
				    syn_group);
}


template <class ConnKeyT, class ConnStructT>
__global__ void setPortSynGroup(ConnKeyT *conn_key_subarray,
				ConnStructT *conn_struct_subarray,
				int port,
				int syn_group,
				int64_t n_conn)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) return;
  setConnPort<ConnKeyT, ConnStructT>(conn_key_subarray[i_conn],
				     conn_struct_subarray[i_conn],
				     port);
  setConnSyn<ConnKeyT, ConnStructT>(conn_key_subarray[i_conn],
				    conn_struct_subarray[i_conn],
				    syn_group);  
}


template<class ConnStructT>
int setConnectionWeights(curandGenerator_t &gen, void *d_storage,
			 ConnStructT *conn_struct_subarray, int64_t n_conn,
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
    setWeights<ConnStructT><<<(n_conn+1023)/1024, 1024>>>
      (conn_struct_subarray, (float*)d_storage, n_conn);
    DBGCUDASYNC;
  }
  else {
    setWeights<ConnStructT><<<(n_conn+1023)/1024, 1024>>>
      (conn_struct_subarray, syn_spec.weight_, n_conn);
    DBGCUDASYNC;
  }
    
  return 0;
}


template<class ConnKeyT>
int setConnectionDelays(curandGenerator_t &gen, void *d_storage,
			ConnKeyT *conn_key_subarray, int64_t n_conn,
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

    setDelays<ConnKeyT><<<(n_conn+1023)/1024, 1024>>>
      (conn_key_subarray, (float*)d_storage, n_conn, time_resolution);
    DBGCUDASYNC;

  }
  else {
    setDelays<ConnKeyT><<<(n_conn+1023)/1024, 1024>>>
      (conn_key_subarray, syn_spec.delay_, n_conn, time_resolution);
    DBGCUDASYNC;
  }
  return 0;
}

__global__ void setSourceTargetIndexKernel(uint64_t n_src_tgt, inode_t n_source,
					   inode_t n_target,
					   uint64_t *d_src_tgt_arr,
					   inode_t *d_src_arr,
					   inode_t *d_tgt_arr);

__global__ void setConnGroupNum(inode_t n_compact,
				iconngroup_t *conn_group_num,
				iconngroup_t *conn_group_idx0_compact,
				inode_t *conn_group_source_compact);


__global__ void setConnGroupIConn0(int64_t n_block_conn,
				   int *conn_group_iconn0_mask,
				   iconngroup_t *conn_group_iconn0_mask_cumul,
				   int64_t *conn_group_iconn0, int64_t i_conn0,
				   iconngroup_t *offset);


template <class T>
__global__ void setConnGroupNewOffset(T *offset, T *add_offset)
{
  *offset = *offset + *add_offset;
}


template <class ConnKeyT>
__global__ void buildConnGroupIConn0Mask(ConnKeyT *conn_key_subarray,
					 ConnKeyT *conn_key_subarray_prev,
					 int64_t n_block_conn,
					 int *conn_group_iconn0_mask)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_block_conn) return;
  ConnKeyT val = conn_key_subarray[i_conn];
  ConnKeyT prev_val;
  inode_t prev_source;
  int prev_delay;
  if (i_conn==0) {
    if (conn_key_subarray_prev != NULL) {
      prev_val = *conn_key_subarray_prev;
      prev_source = getConnSource<ConnKeyT>(prev_val);
      prev_delay = getConnDelay<ConnKeyT>(prev_val);
    }
    else {
      prev_source = 0;
      prev_delay = -1;      // just to ensure it is different
    }
  }
  else {
    prev_val = conn_key_subarray[i_conn-1];
    prev_source = getConnSource<ConnKeyT>(prev_val);
    prev_delay = getConnDelay<ConnKeyT>(prev_val);
  }
  inode_t source = getConnSource<ConnKeyT>(val);
  int delay = getConnDelay<ConnKeyT>(val);
  if (source != prev_source || delay != prev_delay) {
    conn_group_iconn0_mask[i_conn] = 1;
  }
}


template <class ConnKeyT>
__global__ void setConnGroupIdx0Compact
(ConnKeyT *conn_key_subarray, int64_t n_block_conn, int *conn_group_idx0_mask,
 iconngroup_t *conn_group_iconn0_mask_cumul,
 inode_t *conn_group_idx0_mask_cumul,
 iconngroup_t *conn_group_idx0_compact, inode_t *conn_group_source_compact,
 iconngroup_t *iconn0_offset, inode_t *idx0_offset)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>n_block_conn) return;
  if (i_conn<n_block_conn && conn_group_idx0_mask[i_conn]==0) return;
  iconngroup_t i_group = conn_group_iconn0_mask_cumul[i_conn] + *iconn0_offset;
  inode_t i_source_compact = conn_group_idx0_mask_cumul[i_conn]
    + *idx0_offset;
  conn_group_idx0_compact[i_source_compact] = i_group;
  if (i_conn<n_block_conn) {
    //int source = conn_key_subarray[i_conn] >> MaxPortSynNBits;
    inode_t source = getConnSource<ConnKeyT>(conn_key_subarray[i_conn]);
    conn_group_source_compact[i_source_compact] = source;
  }
}

template <class ConnKeyT>
__global__ void getConnGroupDelay(int64_t block_size,
				  ConnKeyT **conn_key_array,
				  int64_t *conn_group_iconn0,
				  int *conn_group_delay,
				  iconngroup_t conn_group_num)
{
  iconngroup_t conn_group_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (conn_group_idx >= conn_group_num) return;
  int64_t i_conn = conn_group_iconn0[conn_group_idx];
  int i_block = (int)(i_conn / block_size);
  int64_t i_block_conn = i_conn % block_size;
  ConnKeyT &conn_key = conn_key_array[i_block][i_block_conn];
  conn_group_delay[conn_group_idx] = getConnDelay(conn_key);
}

template <class ConnKeyT>
__global__ void buildConnGroupMask(ConnKeyT *conn_key_subarray,
				   ConnKeyT *conn_key_subarray_prev,
				   int64_t n_block_conn,
				   int *conn_group_iconn0_mask,
				   int *conn_group_idx0_mask)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_block_conn) return;
  ConnKeyT val = conn_key_subarray[i_conn];
  ConnKeyT prev_val;
  inode_t prev_source;
  int prev_delay;
  if (i_conn==0) {
    if (conn_key_subarray_prev != NULL) {
      prev_val = *conn_key_subarray_prev;
      //prev_source = prev_val >> MaxPortSynNBits; 
      prev_source = getConnSource<ConnKeyT>(prev_val);
      prev_delay = getConnDelay<ConnKeyT>(prev_val);
    }
    else {
      prev_source = 0;
      prev_delay = -1;      // just to ensure it is different
    }
  }
  else {
    prev_val = conn_key_subarray[i_conn-1];
    //prev_source = prev_val >> MaxPortSynNBits;
    prev_source = getConnSource<ConnKeyT>(prev_val);
    prev_delay = getConnDelay<ConnKeyT>(prev_val);
  }
  //int source = val >> MaxPortSynNBits;
  inode_t source = getConnSource<ConnKeyT>(val);
  if (source != prev_source || prev_delay<0) {
    conn_group_iconn0_mask[i_conn] = 1;
    conn_group_idx0_mask[i_conn] = 1;
  }
  else {
    int delay = getConnDelay<ConnKeyT>(val);
    if (delay != prev_delay) {
      conn_group_iconn0_mask[i_conn] = 1;
    }
  }
}

template <class ConnKeyT, class ConnStructT>
int organizeConnections(float time_resolution, inode_t n_node, int64_t n_conn,
			int64_t block_size)
{
  timeval startTV;
  timeval endTV;
  CUDASYNC;
  gettimeofday(&startTV, NULL);

  if (n_conn > 0) {
    printf("Allocating auxiliary GPU memory...\n");
    int64_t sort_storage_bytes = 0;
    void *d_sort_storage = NULL;
    copass_sort::sort<ConnKeyT, ConnStructT>
      ((ConnKeyT**)ConnKeyVect.data(), (ConnStructT**)ConnStructVect.data(),
       n_conn, block_size, d_sort_storage, sort_storage_bytes);
    printf("storage bytes: %ld\n", sort_storage_bytes);
    CUDAMALLOCCTRL("&d_sort_storage",&d_sort_storage, sort_storage_bytes);
    
    printf("Sorting...\n");
    copass_sort::sort<ConnKeyT, ConnStructT>
      ((ConnKeyT**)ConnKeyVect.data(), (ConnStructT**)ConnStructVect.data(),
       n_conn, block_size, d_sort_storage, sort_storage_bytes);
    CUDAFREECTRL("d_sort_storage",d_sort_storage);

    size_t storage_bytes = 0;
    size_t storage_bytes1 = 0;
    void *d_storage = NULL;
    printf("Indexing connection groups...\n");
    // It is important to separate number of allocated blocks
    // (determined by conn_key_vect.size()) from number of blocks
    // on which there are connections, which is determined by n_conn
    // number of used connection blocks
    int k = (n_conn - 1)  / block_size + 1;
    
    // it seems that there is no relevant advantage in using a constant array
    // however better to keep this option ready and commented
    //gpuErrchk(cudaMemcpyToSymbol(ConnKeyArray, ConnKeyVect.data(),
    //				 k*sizeof(ConnKeyT*)));
    //, cudaMemcpyHostToDevice));
    //gpuErrchk(cudaMemcpyToSymbol(ConnStructArray, ConnStructVect.data(),
    //				 k*sizeof(ConnStructT*)));
    //, cudaMemcpyHostToDevice));

    CUDAMALLOCCTRL("&d_ConnKeyArray",&d_ConnKeyArray,
		   k*sizeof(ConnKeyT*));
    gpuErrchk(cudaMemcpy(d_ConnKeyArray, (ConnKeyT**)ConnKeyVect.data(),
			 k*sizeof(ConnKeyT*), cudaMemcpyHostToDevice));
  
    CUDAMALLOCCTRL("&d_ConnStructArray",&d_ConnStructArray,
		   k*sizeof(ConnStructT*));
    gpuErrchk(cudaMemcpy(d_ConnStructArray,
			 (ConnStructT**)ConnStructVect.data(),
			 k*sizeof(ConnStructT*), cudaMemcpyHostToDevice));

    //////////////////////////////////////////////////////////////////////
    
    int *d_conn_group_iconn0_mask;
    CUDAMALLOCCTRL("&d_conn_group_iconn0_mask",
		   &d_conn_group_iconn0_mask,
		   block_size*sizeof(int));

    iconngroup_t *d_conn_group_iconn0_mask_cumul;
    CUDAMALLOCCTRL("&d_conn_group_iconn0_mask_cumul",
		   &d_conn_group_iconn0_mask_cumul,
		   (block_size+1)*sizeof(iconngroup_t));
    
    int *d_conn_group_idx0_mask;
    CUDAMALLOCCTRL("&d_conn_group_idx0_mask",
		   &d_conn_group_idx0_mask,
		   block_size*sizeof(int));

    inode_t *d_conn_group_idx0_mask_cumul;
    CUDAMALLOCCTRL("&d_conn_group_idx0_mask_cumul",
		   &d_conn_group_idx0_mask_cumul,
		   (block_size+1)*sizeof(inode_t));

    iconngroup_t *d_conn_group_idx0_compact;
    int64_t reserve_size = n_node<block_size ? n_node : block_size;
    CUDAMALLOCCTRL("&d_conn_group_idx0_compact",
		   &d_conn_group_idx0_compact,
		   (reserve_size+1)*sizeof(iconngroup_t));
  
    inode_t *d_conn_group_source_compact;
    CUDAMALLOCCTRL("&d_conn_group_source_compact",
		   &d_conn_group_source_compact,
		   reserve_size*sizeof(inode_t));
  
    iconngroup_t *d_iconn0_offset;
    CUDAMALLOCCTRL("&d_iconn0_offset", &d_iconn0_offset, sizeof(iconngroup_t));
    gpuErrchk(cudaMemset(d_iconn0_offset, 0, sizeof(iconngroup_t)));
    inode_t *d_idx0_offset;
    CUDAMALLOCCTRL("&d_idx0_offset", &d_idx0_offset, sizeof(inode_t));
    gpuErrchk(cudaMemset(d_idx0_offset, 0, sizeof(inode_t)));

    ConnKeyT *conn_key_subarray_prev = NULL;
    for (int ib=0; ib<k; ib++) {
      int64_t n_block_conn = ib<(k-1) ? block_size : NConn - block_size*(k-1);
      gpuErrchk(cudaMemset(d_conn_group_iconn0_mask, 0,
			   n_block_conn*sizeof(int)));
      buildConnGroupIConn0Mask<ConnKeyT><<<(n_block_conn+1023)/1024, 1024>>>
	((ConnKeyT*)ConnKeyVect[ib], conn_key_subarray_prev,
       n_block_conn, d_conn_group_iconn0_mask);
      CUDASYNC;
      
      conn_key_subarray_prev = (ConnKeyT*)ConnKeyVect[ib] + block_size - 1;
    
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
			 sizeof(iconngroup_t), cudaMemcpyDeviceToHost));
    printf("Total number of connection groups: %d\n", tot_conn_group_num);

    if (tot_conn_group_num > 0) {
      iconngroup_t *d_conn_group_num;
      CUDAMALLOCCTRL("&d_conn_group_num", &d_conn_group_num,
		     n_node*sizeof(iconngroup_t));
      gpuErrchk(cudaMemset(d_conn_group_num, 0, sizeof(iconngroup_t)));
    
      ConnKeyT *conn_key_subarray_prev = NULL;
      gpuErrchk(cudaMemset(d_iconn0_offset, 0, sizeof(iconngroup_t)));

      CUDAMALLOCCTRL("&d_ConnGroupIConn0",&d_ConnGroupIConn0,
		     (tot_conn_group_num+1)*sizeof(int64_t));

      inode_t n_compact = 0; 
      for (int ib=0; ib<k; ib++) {
	int64_t n_block_conn = ib<(k-1) ? block_size :
	  NConn - block_size*(k-1);
	gpuErrchk(cudaMemset(d_conn_group_iconn0_mask, 0,
			     n_block_conn*sizeof(int)));
	gpuErrchk(cudaMemset(d_conn_group_idx0_mask, 0,
			     n_block_conn*sizeof(int)));
	buildConnGroupMask<ConnKeyT><<<(n_block_conn+1023)/1024, 1024>>>
	  ((ConnKeyT*)ConnKeyVect[ib], conn_key_subarray_prev,
	   n_block_conn, d_conn_group_iconn0_mask, d_conn_group_idx0_mask);
	CUDASYNC;
      
	conn_key_subarray_prev = (ConnKeyT*)ConnKeyVect[ib] + block_size - 1;
    
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

	setConnGroupIdx0Compact<ConnKeyT><<<(n_block_conn+1023)/1024, 1024>>>
	  ((ConnKeyT*)ConnKeyVect[ib], n_block_conn, d_conn_group_idx0_mask,
	   d_conn_group_iconn0_mask_cumul, d_conn_group_idx0_mask_cumul,
	   d_conn_group_idx0_compact, d_conn_group_source_compact,
	   d_iconn0_offset, d_idx0_offset);
	CUDASYNC;

	inode_t n_block_compact; 
	gpuErrchk(cudaMemcpy(&n_block_compact, d_conn_group_idx0_mask_cumul
			     + n_block_conn,
			     sizeof(inode_t), cudaMemcpyDeviceToHost));
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
		     (n_node+1)*sizeof(iconngroup_t));
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
      int *d_max_delay_num;
      CUDAMALLOCCTRL("&d_max_delay_num",&d_max_delay_num, sizeof(int));
    
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
			   sizeof(int), cudaMemcpyDeviceToHost));
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
		     tot_conn_group_num*sizeof(int));

      getConnGroupDelay<ConnKeyT><<<(tot_conn_group_num+1023)/1024, 1024>>>
	(block_size, (ConnKeyT **)d_ConnKeyArray, d_ConnGroupIConn0,
	 d_ConnGroupDelay, tot_conn_group_num);
      DBGCUDASYNC;
#endif
	
    }
    else {
      throw ngpu_exception("Number of connections groups must be positive "
			   "for number of connections > 0");   
    }
  }
  else {
    gpuErrchk(cudaMemset(d_ConnGroupIdx0, 0, (n_node+1)*sizeof(iconngroup_t)));
    h_MaxDelayNum = 0;
  }
  
  gettimeofday(&endTV, NULL);
  long time = (long)((endTV.tv_sec * 1000000.0 + endTV.tv_usec)
		     - (startTV.tv_sec * 1000000.0 + startTV.tv_usec));
  printf("%-40s%.2f ms\n", "Time: ", (double)time / 1000.);
  printf("Done\n");
  
  
  return 0;
}
  


int ConnectInit();

__device__ __forceinline__
inode_t GetNodeIndex(inode_t i_node_0, inode_t i_node_rel)
{
  return i_node_0 + i_node_rel;
}

__device__ __forceinline__
inode_t GetNodeIndex(inode_t *i_node_0, inode_t i_node_rel)
{
  return *(i_node_0 + i_node_rel);
}

template <class T, class ConnKeyT>
__global__ void setSource(ConnKeyT *conn_key_subarray, uint *rand_val,
			  int64_t n_conn, T source, inode_t n_source)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) return;
  inode_t i_source = GetNodeIndex(source, rand_val[i_conn]%n_source);
  setConnSource<ConnKeyT>(conn_key_subarray[i_conn], i_source);    
}

template <class T, class ConnStructT>
__global__ void setTarget(ConnStructT *conn_struct_subarray, uint *rand_val,
			  int64_t n_conn, T target, inode_t n_target)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) return;
  inode_t i_target = GetNodeIndex(target, rand_val[i_conn]%n_target);
  setConnTarget<ConnStructT>(conn_struct_subarray[i_conn], i_target);    
}

template <class T1, class T2, class ConnKeyT, class ConnStructT>
__global__ void setOneToOneSourceTarget(ConnKeyT *conn_key_subarray,
					ConnStructT *conn_struct_subarray,
					int64_t n_block_conn,
					int64_t n_prev_conn,
					T1 source, T2 target)
{
  int64_t i_block_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_block_conn>=n_block_conn) return;
  int64_t i_conn = n_prev_conn + i_block_conn;
  inode_t i_source = GetNodeIndex(source, (int)(i_conn));
  inode_t i_target = GetNodeIndex(target, (int)(i_conn));
  setConnSource<ConnKeyT>(conn_key_subarray[i_block_conn], i_source);
  setConnTarget<ConnStructT>(conn_struct_subarray[i_block_conn], i_target);
}

template <class T1, class T2, class ConnKeyT, class ConnStructT>
__global__ void setAllToAllSourceTarget(ConnKeyT *conn_key_subarray,
					ConnStructT *conn_struct_subarray,
					int64_t n_block_conn,
					int64_t n_prev_conn,
					T1 source, inode_t n_source,
					T2 target, inode_t n_target)
{
  int64_t i_block_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_block_conn>=n_block_conn) return;
  int64_t i_conn = n_prev_conn + i_block_conn;
  inode_t i_source = GetNodeIndex(source, (int)(i_conn / n_target));
  inode_t i_target = GetNodeIndex(target, (int)(i_conn % n_target));
  setConnSource<ConnKeyT>(conn_key_subarray[i_block_conn], i_source);    
  setConnTarget<ConnStructT>(conn_struct_subarray[i_block_conn], i_target);
}

template <class T, class ConnStructT>
__global__ void setIndegreeTarget(ConnStructT *conn_struct_subarray,
				  int64_t n_block_conn,
				  int64_t n_prev_conn,
				  T target, int indegree)
{
  int64_t i_block_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_block_conn>=n_block_conn) return;
  int64_t i_conn = n_prev_conn + i_block_conn;
  inode_t i_target = GetNodeIndex(target, (int)(i_conn / indegree));
  setConnTarget<ConnStructT>(conn_struct_subarray[i_block_conn], i_target);
}

template <class T, class ConnKeyT>
__global__ void setOutdegreeSource(ConnKeyT *conn_key_subarray,
				   int64_t n_block_conn,
				   int64_t n_prev_conn,
				   T source, int outdegree)
{
  int64_t i_block_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_block_conn>=n_block_conn) return;
  int64_t i_conn = n_prev_conn + i_block_conn;
  inode_t i_source = GetNodeIndex(source, (int)(i_conn / outdegree));
  setConnSource<ConnKeyT>(conn_key_subarray[i_block_conn], i_source);    
}

template <class T1, class T2, class ConnKeyT, class ConnStructT>
int connect_one_to_one(curandGenerator_t &gen,
		       void *d_storage, float time_resolution,
		       std::vector<void*> &conn_key_vect,
		       std::vector<void*> &conn_struct_vect,
		       int64_t &n_conn, int64_t block_size,
		       T1 source, T2 target,  inode_t n_node,
		       SynSpec &syn_spec)
{
  int64_t old_n_conn = n_conn;
  int64_t n_new_conn = n_node;
  n_conn += n_new_conn; // new number of connections
  int new_n_block = (int)((n_conn + block_size - 1) / block_size);
  allocateNewBlocks<ConnKeyT, ConnStructT>
    (conn_key_vect, conn_struct_vect, block_size, new_n_block);

  //printf("Generating connections with one-to-one rule...\n");
  int64_t n_prev_conn = 0;
  int ib0 = (int)(old_n_conn / block_size);
  for (int ib=ib0; ib<new_n_block; ib++) {
    int64_t n_block_conn; // number of connections in a block
    int64_t i_conn0; // index of first connection in a block
    if (new_n_block == ib0 + 1) {  // all connections are in the same block
      i_conn0 = old_n_conn % block_size;
      n_block_conn = n_new_conn;
    }
    else if (ib == ib0) { // first block
      i_conn0 = old_n_conn % block_size;
      n_block_conn = block_size - i_conn0;
    }
    else if (ib == new_n_block-1) { // last block
      i_conn0 = 0;
      n_block_conn = (n_conn - 1) % block_size + 1;
    }
    else {
      i_conn0 = 0;
      n_block_conn = block_size;
    }
    setOneToOneSourceTarget<T1, T2, ConnKeyT, ConnStructT>
      <<<(n_block_conn+1023)/1024, 1024>>>
      ((ConnKeyT*)conn_key_vect[ib] + i_conn0,
       (ConnStructT*)conn_struct_vect[ib] + i_conn0,
       n_block_conn, n_prev_conn, source, target);
    DBGCUDASYNC;
    CUDASYNC;				
    setConnectionWeights<ConnStructT>
      (gen, d_storage, (ConnStructT*)conn_struct_vect[ib] + i_conn0,
       n_block_conn, syn_spec);
    CUDASYNC;				
    setConnectionDelays<ConnKeyT>
      (gen, d_storage, (ConnKeyT*)conn_key_vect[ib] + i_conn0,
       n_block_conn, syn_spec, time_resolution);
    CUDASYNC;
    setPort<ConnKeyT, ConnStructT><<<(n_block_conn+1023)/1024, 1024>>>
    ((ConnKeyT*)conn_key_vect[ib] + i_conn0,
     (ConnStructT*)conn_struct_vect[ib] + i_conn0,
     syn_spec.port_, n_block_conn);
    DBGCUDASYNC;
    CUDASYNC;
    setSynGroup<ConnKeyT, ConnStructT><<<(n_block_conn+1023)/1024, 1024>>>
      ((ConnKeyT*)conn_key_vect[ib] + i_conn0,
       (ConnStructT*)conn_struct_vect[ib] + i_conn0,
       syn_spec.syn_group_, n_block_conn);
    DBGCUDASYNC;
    CUDASYNC;
    n_prev_conn += n_block_conn;
  }

  return 0;
}


template <class T1, class T2, class ConnKeyT, class ConnStructT>
int connect_all_to_all(curandGenerator_t &gen,
		       void *d_storage, float time_resolution,
		       std::vector<void*> &conn_key_vect,
		       std::vector<void*> &conn_struct_vect,
		       int64_t &n_conn, int64_t block_size,
		       T1 source, inode_t n_source,
		       T2 target, inode_t n_target,
		       SynSpec &syn_spec)
{
  int64_t old_n_conn = n_conn;
  int64_t n_new_conn = n_source*n_target;
  n_conn += n_new_conn; // new number of connections
  int new_n_block = (int)((n_conn + block_size - 1) / block_size);

  allocateNewBlocks<ConnKeyT, ConnStructT>
    (conn_key_vect, conn_struct_vect, block_size, new_n_block);

  //printf("Generating connections with all-to-all rule...\n");
  int64_t n_prev_conn = 0;
  int ib0 = (int)(old_n_conn / block_size);
  for (int ib=ib0; ib<new_n_block; ib++) {
    int64_t n_block_conn; // number of connections in a block
    int64_t i_conn0; // index of first connection in a block
    if (new_n_block == ib0 + 1) {  // all connections are in the same block
      i_conn0 = old_n_conn % block_size;
      n_block_conn = n_new_conn;
    }
    else if (ib == ib0) { // first block
      i_conn0 = old_n_conn % block_size;
      n_block_conn = block_size - i_conn0;
    }
    else if (ib == new_n_block-1) { // last block
      i_conn0 = 0;
      n_block_conn = (n_conn - 1) % block_size + 1;
    }
    else {
      i_conn0 = 0;
      n_block_conn = block_size;
    }

    setAllToAllSourceTarget<T1, T2, ConnKeyT, ConnStructT>
      <<<(n_block_conn+1023)/1024, 1024>>>
      ((ConnKeyT*)conn_key_vect[ib] + i_conn0,
       (ConnStructT*)conn_struct_vect[ib] + i_conn0,
       n_block_conn, n_prev_conn, source, n_source, target, n_target);
    DBGCUDASYNC;
    setConnectionWeights<ConnStructT>
      (gen, d_storage, (ConnStructT*)conn_struct_vect[ib] + i_conn0,
       n_block_conn, syn_spec);

    setConnectionDelays<ConnKeyT>
      (gen, d_storage, (ConnKeyT*)conn_key_vect[ib] + i_conn0,
       n_block_conn, syn_spec, time_resolution);

    setPort<ConnKeyT, ConnStructT><<<(n_block_conn+1023)/1024, 1024>>>
      ((ConnKeyT*)conn_key_vect[ib] + i_conn0,
       (ConnStructT*)conn_struct_vect[ib] + i_conn0, syn_spec.port_,
       n_block_conn);
    DBGCUDASYNC;
    setSynGroup<ConnKeyT, ConnStructT><<<(n_block_conn+1023)/1024, 1024>>>
      ((ConnKeyT*)conn_key_vect[ib] + i_conn0,
       (ConnStructT*)conn_struct_vect[ib] + i_conn0, syn_spec.syn_group_,
       n_block_conn);
    DBGCUDASYNC;

    n_prev_conn += n_block_conn;
  }

  return 0;
}


template <class T1, class T2, class ConnKeyT, class ConnStructT>
int connect_fixed_total_number(curandGenerator_t &gen,
			       void *d_storage, float time_resolution,
			       std::vector<void*> &conn_key_vect,
			       std::vector<void*> &conn_struct_vect,
			       int64_t &n_conn, int64_t block_size,
			       int64_t total_num, T1 source, inode_t n_source,
			       T2 target, inode_t n_target,
			       SynSpec &syn_spec)
{
  if (total_num==0) return 0;
  int64_t old_n_conn = n_conn;
  int64_t n_new_conn = total_num;
  n_conn += n_new_conn; // new number of connections
  int new_n_block = (int)((n_conn + block_size - 1) / block_size);

  allocateNewBlocks<ConnKeyT, ConnStructT>
    (conn_key_vect, conn_struct_vect, block_size, new_n_block);

  //printf("Generating connections with fixed_total_number rule...\n");
  int ib0 = (int)(old_n_conn / block_size);
  for (int ib=ib0; ib<new_n_block; ib++) {
    int64_t n_block_conn; // number of connections in a block
    int64_t i_conn0; // index of first connection in a block
    if (new_n_block == ib0 + 1) {  // all connections are in the same block
        i_conn0 = old_n_conn % block_size;
	n_block_conn =   n_new_conn;
    }
    else if (ib == ib0) { // first block
      i_conn0 = old_n_conn % block_size;
      n_block_conn = block_size - i_conn0;
    }
    else if (ib == new_n_block-1) { // last block
      i_conn0 = 0;
      n_block_conn = (n_conn - 1) % block_size + 1;
    }
    else {
      i_conn0 = 0;
      n_block_conn = block_size;
    }
    // generate random source index in range 0 - n_neuron
    CURAND_CALL(curandGenerate(gen, (uint*)d_storage, n_block_conn));
    setSource<T1, ConnKeyT><<<(n_block_conn+1023)/1024, 1024>>>
      ((ConnKeyT*)conn_key_vect[ib] + i_conn0, (uint*)d_storage, n_block_conn,
       source, n_source);
    DBGCUDASYNC;

    // generate random target index in range 0 - n_neuron
    CURAND_CALL(curandGenerate(gen, (uint*)d_storage, n_block_conn));
    setTarget<T2, ConnStructT><<<(n_block_conn+1023)/1024, 1024>>>
      ((ConnStructT*)conn_struct_vect[ib] + i_conn0, (uint*)d_storage,
       n_block_conn, target, n_target);
    DBGCUDASYNC;

    setConnectionWeights<ConnStructT>
      (gen, d_storage, (ConnStructT*)conn_struct_vect[ib] + i_conn0,
       n_block_conn, syn_spec);

    setConnectionDelays<ConnKeyT>
      (gen, d_storage, (ConnKeyT*)conn_key_vect[ib] + i_conn0, n_block_conn,
       syn_spec, time_resolution);

    setPort<ConnKeyT, ConnStructT><<<(n_block_conn+1023)/1024, 1024>>>
      ((ConnKeyT*)conn_key_vect[ib] + i_conn0,
       (ConnStructT*)conn_struct_vect[ib] + i_conn0, syn_spec.port_,
       n_block_conn);
    DBGCUDASYNC;
    setSynGroup<ConnKeyT, ConnStructT><<<(n_block_conn+1023)/1024, 1024>>>
      ((ConnKeyT*)conn_key_vect[ib] + i_conn0,
       (ConnStructT*)conn_struct_vect[ib] + i_conn0, syn_spec.syn_group_,
       n_block_conn);
    DBGCUDASYNC;

  }

  return 0;
}

template <class T1, class T2, class ConnKeyT, class ConnStructT>
int connect_fixed_indegree(curandGenerator_t &gen,
			   void *d_storage, float time_resolution,
			   std::vector<void*> &conn_key_vect,
			   std::vector<void*> &conn_struct_vect,
			   int64_t &n_conn, int64_t block_size,
			   int indegree, T1 source, inode_t n_source,
			   T2 target, inode_t n_target,
			   SynSpec &syn_spec)
{
  if (indegree<=0) return 0;
  int64_t old_n_conn = n_conn;
  int64_t n_new_conn = n_target*indegree;
  n_conn += n_new_conn; // new number of connections
  int new_n_block = (int)((n_conn + block_size - 1) / block_size);

  allocateNewBlocks<ConnKeyT, ConnStructT>
    (conn_key_vect, conn_struct_vect, block_size, new_n_block);

  //printf("Generating connections with fixed_indegree rule...\n");
  int64_t n_prev_conn = 0;
  int ib0 = (int)(old_n_conn / block_size);
  for (int ib=ib0; ib<new_n_block; ib++) {
    int64_t n_block_conn; // number of connections in a block
    int64_t i_conn0; // index of first connection in a block
    if (new_n_block == ib0 + 1) {  // all connections are in the same block
        i_conn0 = old_n_conn % block_size;
	n_block_conn =   n_new_conn;
    }
    else if (ib == ib0) { // first block
      i_conn0 = old_n_conn % block_size;
      n_block_conn = block_size - i_conn0;
    }
    else if (ib == new_n_block-1) { // last block
      i_conn0 = 0;
      n_block_conn = (n_conn - 1) % block_size + 1;
    }
    else {
      i_conn0 = 0;
      n_block_conn = block_size;
    }
    // generate random source index in range 0 - n_neuron
    CURAND_CALL(curandGenerate(gen, (uint*)d_storage, n_block_conn));
    setSource<T1, ConnKeyT><<<(n_block_conn+1023)/1024, 1024>>>
      ((ConnKeyT*)conn_key_vect[ib] + i_conn0, (uint*)d_storage, n_block_conn,
       source, n_source);
    DBGCUDASYNC;

    setIndegreeTarget<T2, ConnStructT><<<(n_block_conn+1023)/1024, 1024>>>
      ((ConnStructT*)conn_struct_vect[ib] + i_conn0, n_block_conn, n_prev_conn,
       target, indegree);
    DBGCUDASYNC;

    setConnectionWeights<ConnStructT>
      (gen, d_storage, (ConnStructT*)conn_struct_vect[ib] + i_conn0,
       n_block_conn, syn_spec);

    setConnectionDelays<ConnKeyT>
      (gen, d_storage, (ConnKeyT*)conn_key_vect[ib] + i_conn0, n_block_conn,
       syn_spec, time_resolution);

    setPort<ConnKeyT, ConnStructT><<<(n_block_conn+1023)/1024, 1024>>>
      ((ConnKeyT*)conn_key_vect[ib] + i_conn0,
       (ConnStructT*)conn_struct_vect[ib] + i_conn0, syn_spec.port_,
       n_block_conn);
    DBGCUDASYNC;
    setSynGroup<ConnKeyT, ConnStructT><<<(n_block_conn+1023)/1024, 1024>>>
      ((ConnKeyT*)conn_key_vect[ib] + i_conn0,
       (ConnStructT*)conn_struct_vect[ib] + i_conn0, syn_spec.syn_group_,
       n_block_conn);
    DBGCUDASYNC;

    n_prev_conn += n_block_conn;
  }

  return 0;
}

template <class T1, class T2, class ConnKeyT, class ConnStructT>
int connect_fixed_outdegree(curandGenerator_t &gen,
			    void *d_storage, float time_resolution,
			    std::vector<void*> &conn_key_vect,
			    std::vector<void*> &conn_struct_vect,
			    int64_t &n_conn, int64_t block_size,
			    int outdegree, T1 source, inode_t n_source,
			    T2 target, inode_t n_target,
			    SynSpec &syn_spec)
{
  if (outdegree<=0) return 0;
  int64_t old_n_conn = n_conn;
  int64_t n_new_conn = n_source*outdegree;
  n_conn += n_new_conn; // new number of connections
  int new_n_block = (int)((n_conn + block_size - 1) / block_size);

  allocateNewBlocks<ConnKeyT, ConnStructT>
    (conn_key_vect, conn_struct_vect, block_size, new_n_block);

  //printf("Generating connections with fixed_outdegree rule...\n");
  int64_t n_prev_conn = 0;
  int ib0 = (int)(old_n_conn / block_size);
  for (int ib=ib0; ib<new_n_block; ib++) {
    int64_t n_block_conn; // number of connections in a block
    int64_t i_conn0; // index of first connection in a block
    if (new_n_block == ib0 + 1) {  // all connections are in the same block
        i_conn0 = old_n_conn % block_size;
	n_block_conn =   n_new_conn;
    }
    else if (ib == ib0) { // first block
      i_conn0 = old_n_conn % block_size;
      n_block_conn = block_size - i_conn0;
    }
    else if (ib == new_n_block-1) { // last block
      i_conn0 = 0;
      n_block_conn = (n_conn - 1) % block_size + 1;
    }
    else {
      i_conn0 = 0;
      n_block_conn = block_size;
    }

    setOutdegreeSource<T1, ConnKeyT><<<(n_block_conn+1023)/1024, 1024>>>
      ((ConnKeyT*)conn_key_vect[ib] + i_conn0, n_block_conn, n_prev_conn,
       source, outdegree);
    DBGCUDASYNC;

    // generate random target index in range 0 - n_neuron
    CURAND_CALL(curandGenerate(gen, (uint*)d_storage, n_block_conn));
    setTarget<T2, ConnStructT><<<(n_block_conn+1023)/1024, 1024>>>
      ((ConnStructT*)conn_struct_vect[ib] + i_conn0, (uint*)d_storage,
       n_block_conn, target, n_target);
    DBGCUDASYNC;

    setConnectionWeights<ConnStructT>
      (gen, d_storage, (ConnStructT*)conn_struct_vect[ib] + i_conn0,
       n_block_conn, syn_spec);

    setConnectionDelays<ConnKeyT>
      (gen, d_storage, (ConnKeyT*)conn_key_vect[ib] + i_conn0, n_block_conn,
       syn_spec, time_resolution);

    setPort<ConnKeyT, ConnStructT><<<(n_block_conn+1023)/1024, 1024>>>
      ((ConnKeyT*)conn_key_vect[ib] + i_conn0,
       (ConnStructT*)conn_struct_vect[ib] + i_conn0, syn_spec.port_, n_block_conn);
    DBGCUDASYNC;
    setSynGroup<ConnKeyT, ConnStructT><<<(n_block_conn+1023)/1024, 1024>>>
      ((ConnKeyT*)conn_key_vect[ib] + i_conn0,
       (ConnStructT*)conn_struct_vect[ib] + i_conn0, syn_spec.syn_group_,
       n_block_conn);
    DBGCUDASYNC;

    n_prev_conn += n_block_conn;
  }

  return 0;
}

template <class T1, class T2, class ConnKeyT, class ConnStructT>
int NESTGPU::_ConnectOneToOne
(curandGenerator_t &gen, T1 source, T2 target, inode_t n_node,
 SynSpec &syn_spec)
{
  //printf("In new specialized connection one-to-one\n");

  void *d_storage;
  CUDAMALLOCCTRL("&d_storage",&d_storage, h_ConnBlockSize*sizeof(uint));

  connect_one_to_one<T1, T2, ConnKeyT, ConnStructT>
    (gen, d_storage, time_resolution_,
     ConnKeyVect, ConnStructVect, NConn,
     h_ConnBlockSize, source, target, n_node, syn_spec);
  CUDAFREECTRL("d_storage",d_storage);

  return 0;
}

template <class T1, class T2, class ConnKeyT, class ConnStructT>
int NESTGPU::_ConnectAllToAll
(curandGenerator_t &gen, T1 source, inode_t n_source, T2 target,
 inode_t n_target, SynSpec &syn_spec)
{
  //printf("In new specialized connection all-to-all\n");

  void *d_storage;
  CUDAMALLOCCTRL("&d_storage",&d_storage, h_ConnBlockSize*sizeof(uint));

  connect_all_to_all<T1, T2, ConnKeyT, ConnStructT>
    (gen, d_storage, time_resolution_,
     ConnKeyVect, ConnStructVect, NConn,
     h_ConnBlockSize, source, n_source,
     target, n_target, syn_spec);
  CUDAFREECTRL("d_storage",d_storage);

  return 0;
}

template <class T1, class T2, class ConnKeyT, class ConnStructT>
int NESTGPU::_ConnectFixedTotalNumber
(curandGenerator_t &gen, T1 source, inode_t n_source, T2 target,
 inode_t n_target, int64_t total_num, SynSpec &syn_spec)
{
  //printf("In new specialized connection fixed-total-number\n");

  void *d_storage;
  CUDAMALLOCCTRL("&d_storage",&d_storage, h_ConnBlockSize*sizeof(uint));

  connect_fixed_total_number<T1, T2, ConnKeyT, ConnStructT>
    (gen, d_storage, time_resolution_,
     ConnKeyVect, ConnStructVect, NConn,
     h_ConnBlockSize, total_num, source, n_source,
     target, n_target, syn_spec);
  CUDAFREECTRL("d_storage",d_storage);

  return 0;
}

template <class T1, class T2, class ConnKeyT, class ConnStructT>
int NESTGPU::_ConnectFixedIndegree
(curandGenerator_t &gen, T1 source, inode_t n_source, T2 target,
 inode_t n_target, int indegree, SynSpec &syn_spec)
{
  //printf("In new specialized connection fixed-indegree\n");

  void *d_storage;
  CUDAMALLOCCTRL("&d_storage",&d_storage, h_ConnBlockSize*sizeof(uint));

  connect_fixed_indegree<T1, T2, ConnKeyT, ConnStructT>
    (gen, d_storage, time_resolution_,
     ConnKeyVect, ConnStructVect, NConn,
     h_ConnBlockSize, indegree, source, n_source,
     target, n_target, syn_spec);
  CUDAFREECTRL("d_storage",d_storage);

  return 0;
}

template <class T1, class T2, class ConnKeyT, class ConnStructT>
int NESTGPU::_ConnectFixedOutdegree
(curandGenerator_t &gen, T1 source, inode_t n_source, T2 target,
 inode_t n_target, int outdegree, SynSpec &syn_spec)
{
  //printf("In new specialized connection fixed-outdegree\n");

  void *d_storage;
  CUDAMALLOCCTRL("&d_storage",&d_storage, h_ConnBlockSize*sizeof(uint));

  connect_fixed_outdegree<T1, T2, ConnKeyT, ConnStructT>
    (gen, d_storage, time_resolution_,
     ConnKeyVect, ConnStructVect, NConn,
     h_ConnBlockSize, outdegree, source, n_source,
     target, n_target, syn_spec);
  CUDAFREECTRL("d_storage",d_storage);

  return 0;
}


// Count number of connections per source-target couple
template <class ConnKeyT, class ConnStructT>
__global__ void CountConnectionsKernel(int64_t n_conn, inode_t n_source,
				       inode_t n_target, uint64_t *src_tgt_arr,
				       uint64_t *src_tgt_conn_num,
				       int syn_group)
{
  int64_t i_conn = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; 
  if (i_conn >= n_conn) return;

  int i_block = (int)(i_conn / ConnBlockSize);
  int64_t i_block_conn = i_conn % ConnBlockSize;
  ConnStructT &conn_struct =
    ((ConnStructT **)ConnStructArray)[i_block][i_block_conn];
  ConnKeyT &conn_key =
    ((ConnKeyT **)ConnKeyArray)[i_block][i_block_conn];
  // if (syn_group==-1 || conn.syn_group == syn_group) {
  int syn_group1 = getConnSyn<ConnKeyT, ConnStructT>(conn_key, conn_struct);
  if (syn_group==-1 || (syn_group1 == syn_group)) {
    // First get source and target node index
    inode_t i_target = getConnTarget<ConnStructT>(conn_struct);
    inode_t i_source = getConnSource<ConnKeyT>(conn_key);
    uint64_t i_src_tgt = ((int64_t)i_source << 32) | i_target;
    uint64_t i_arr = locate(i_src_tgt, src_tgt_arr, n_source*n_target);
    if (src_tgt_arr[i_arr] == i_src_tgt) {
      //printf("i_conn %lld i_source %d i_target %d i_src_tgt %lld "
      //     "i_arr %lld\n", i_conn, i_source, i_target, i_src_tgt, i_arr);
      // (atomic)increase the number of connections for source-target couple
      atomicAdd((unsigned long long *)&src_tgt_conn_num[i_arr], 1);
    }
  }
}



// Fill array of connection indexes
template <class ConnKeyT, class ConnStructT>
__global__ void SetConnectionsIndexKernel(int64_t n_conn, inode_t n_source,
					  inode_t n_target,
					  uint64_t *src_tgt_arr,
					  uint64_t *src_tgt_conn_num,
					  uint64_t *src_tgt_conn_cumul,
					  int syn_group, int64_t *conn_ids)
{
  int64_t i_conn = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; 
  if (i_conn >= n_conn) return;

  int i_block = (int)(i_conn / ConnBlockSize);
  int64_t i_block_conn = i_conn % ConnBlockSize;
  ConnStructT &conn_struct =
    ((ConnStructT **)ConnStructArray)[i_block][i_block_conn];
  ConnKeyT &conn_key =
    ((ConnKeyT **)ConnKeyArray)[i_block][i_block_conn];
  // if (syn_group==-1 || conn.syn_group == syn_group) {
  int syn_group1 = getConnSyn<ConnKeyT, ConnStructT>(conn_key, conn_struct);
  if (syn_group==-1 || (syn_group1 == syn_group)) {
    // First get source and target node index
    inode_t i_target = getConnTarget<ConnStructT>(conn_struct);
    inode_t i_source = getConnSource<ConnKeyT>(conn_key);
    uint64_t i_src_tgt = ((int64_t)i_source << 32) | i_target;
    uint64_t i_arr = locate(i_src_tgt, src_tgt_arr, n_source*n_target);
    if (src_tgt_arr[i_arr] == i_src_tgt) {
      //printf("i_conn %lld i_source %d i_target %d i_src_tgt %lld "
      //     "i_arr %lld\n", i_conn, i_source, i_target, i_src_tgt, i_arr);
      // (atomic)increase the number of connections for source-target couple
      uint64_t pos =
	atomicAdd((unsigned long long *)&src_tgt_conn_num[i_arr], 1);
      //printf("pos %lld src_tgt_conn_cumul[i_arr] %lld\n",
      //     pos, src_tgt_conn_cumul[i_arr]);
      conn_ids[src_tgt_conn_cumul[i_arr] + pos] = i_conn;
    }
  }
}

//////////////////////////////////////////////////////////////////////
// CUDA Kernel that gets all parameters of an array of n_conn connections,
// identified by the indexes conn_ids[i], and puts them in the arrays
// i_source, i_target, port, syn_group, delay, weight
//////////////////////////////////////////////////////////////////////
template <class ConnKeyT, class ConnStructT>
__global__ void GetConnectionStatusKernel
(int64_t *conn_ids, int64_t n_conn, inode_t *source, inode_t *target,
 int *port, int *syn_group, float *delay, float *weight)
{
  int64_t i_arr = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; 
  if (i_arr >= n_conn) return;

   // get connection index, connection block index and index within block
  int64_t i_conn = conn_ids[i_arr];
  int i_block = (int)(i_conn / ConnBlockSize);
  int64_t i_block_conn = i_conn % ConnBlockSize;
  // get connection structure
  ConnStructT &conn_struct =
    ((ConnStructT **)ConnStructArray)[i_block][i_block_conn];
  ConnKeyT &conn_key =
    ((ConnKeyT **)ConnKeyArray)[i_block][i_block_conn];
  // Get source, target, port, synaptic group and delay
  inode_t i_source = getConnSource<ConnKeyT>(conn_key);
  inode_t i_target = getConnTarget<ConnStructT>(conn_struct);
  int i_port = getConnPort<ConnKeyT, ConnStructT>(conn_key, conn_struct);
  int i_syn_group = getConnSyn<ConnKeyT, ConnStructT>(conn_key, conn_struct);
  int i_delay = getConnDelay<ConnKeyT>(conn_key);
  source[i_arr] = i_source;
  target[i_arr] = i_target;
  port[i_arr] = i_port;
  // Get weight and synapse group
  weight[i_arr] = conn_struct.weight;
  syn_group[i_arr] = i_syn_group;
  delay[i_arr] = NESTGPUTimeResolution * i_delay;
}


//////////////////////////////////////////////////////////////////////
// CUDA Kernel that gets a float parameter of an array of n_conn connections,
// identified by the indexes conn_ids[i], and puts it in the array
// param_arr
//////////////////////////////////////////////////////////////////////
template <class ConnKeyT, class ConnStructT>
__global__ void GetConnectionFloatParamKernel
(int64_t *conn_ids, int64_t n_conn, float *param_arr, int i_param)
{
  int64_t i_arr = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; 
  if (i_arr >= n_conn) return;

   // get connection index, connection block index and index within block
  int64_t i_conn = conn_ids[i_arr];
  int i_block = (int)(i_conn / ConnBlockSize);
  int64_t i_block_conn = i_conn % ConnBlockSize;
  // get connection structure
  ConnStructT &conn_struct =
    ((ConnStructT **)ConnStructArray)[i_block][i_block_conn];
  ConnKeyT &conn_key =
    ((ConnKeyT **)ConnKeyArray)[i_block][i_block_conn];
  switch (i_param) {
  case i_weight_param: {
    param_arr[i_arr] = conn_struct.weight;
    break;
  }
  case i_delay_param: {
    // Get joined source-delay parameter, then delay
    int i_delay = getConnDelay<ConnKeyT>(conn_key);
    param_arr[i_arr] = NESTGPUTimeResolution * i_delay;
    break;
  }
  }
}

template <class ConnKeyT, class ConnStructT>
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
  int i_block = (int)(i_conn / ConnBlockSize);
  int64_t i_block_conn = i_conn % ConnBlockSize;
  // get connection structure
  ConnStructT &conn_struct =
    ((ConnStructT **)ConnStructArray)[i_block][i_block_conn];
  ConnKeyT &conn_key =
    ((ConnKeyT **)ConnKeyArray)[i_block][i_block_conn];
  switch (i_param) {
  case i_source_param: {
    inode_t i_source = getConnSource<ConnKeyT>(conn_key);
    param_arr[i_arr] = i_source;
    break;
  }
  case i_target_param: {
    inode_t i_target = getConnTarget<ConnStructT>(conn_struct);
    param_arr[i_arr] = i_target;
    break;
  }
  case i_port_param: {
    int i_port = getConnPort<ConnKeyT, ConnStructT>(conn_key, conn_struct);
    param_arr[i_arr] = i_port;
    break;
  }
  case i_syn_group_param: {
    // Get synapse group
    int i_syn_group = getConnSyn<ConnKeyT, ConnStructT>(conn_key, conn_struct);
    param_arr[i_arr] = i_syn_group;
    break;
  }
  }
}

template <class ConnStructT>
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
  int i_block = (int)(i_conn / ConnBlockSize);
  int64_t i_block_conn = i_conn % ConnBlockSize;
  // get connection structure
  ConnStructT &conn_struct =
    ((ConnStructT **)ConnStructArray)[i_block][i_block_conn];
  switch (i_param) {
  case i_weight_param: {
    conn_struct.weight = param_arr[i_arr]; 
    break;
  }
  }
}

template <class ConnStructT>
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
  int i_block = (int)(i_conn / ConnBlockSize);
  int64_t i_block_conn = i_conn % ConnBlockSize;
  // get connection structure
  ConnStructT &conn_struct =
    ((ConnStructT **)ConnStructArray)[i_block][i_block_conn];
  switch (i_param) {
  case i_weight_param: {
    conn_struct.weight = val; 
    break;
  }
  }
}

template <class ConnKeyT, class ConnStructT>
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
  int i_block = (int)(i_conn / ConnBlockSize);
  int64_t i_block_conn = i_conn % ConnBlockSize;
  // get connection structure
  ConnStructT &conn_struct =
    ((ConnStructT **)ConnStructArray)[i_block][i_block_conn];
  ConnKeyT &conn_key =
    ((ConnKeyT **)ConnKeyArray)[i_block][i_block_conn];
  switch (i_param) {
  case i_target_param: {
    setConnTarget<ConnStructT>(conn_struct, param_arr[i_arr]);
    break;
  }
  case i_port_param: {
    setConnPort<ConnKeyT, ConnStructT>(conn_key, conn_struct, param_arr[i_arr]);
    break;
  }
  case i_syn_group_param: {
    setConnSyn<ConnKeyT, ConnStructT>(conn_key, conn_struct, param_arr[i_arr]);
    break;
  }
  }
}


//////////////////////////////////////////////////////////////////////
// CUDA Kernel that sets an integer parameter of an array of n_conn connections,
// identified by the indexes conn_ids[i], to the value val
//////////////////////////////////////////////////////////////////////
template <class ConnKeyT, class ConnStructT>
__global__ void SetConnectionIntParamKernel
(int64_t *conn_ids, int64_t n_conn, int val, int i_param)
{
  int64_t i_arr = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; 
  if (i_arr >= n_conn) return;

   // get connection index, connection block index and index within block
  int64_t i_conn = conn_ids[i_arr];
  int i_block = (int)(i_conn / ConnBlockSize);
  int64_t i_block_conn = i_conn % ConnBlockSize;
  // get connection structure
  ConnStructT &conn_struct =
    ((ConnStructT **)ConnStructArray)[i_block][i_block_conn];
  ConnKeyT &conn_key =
    ((ConnKeyT **)ConnKeyArray)[i_block][i_block_conn];
  switch (i_param) {
      case i_target_param: {
    setConnTarget<ConnStructT>(conn_struct, val);
    break;
  }
  case i_port_param: {
    setConnPort<ConnKeyT, ConnStructT>(conn_key, conn_struct, val);
    break;
  }
  case i_syn_group_param: {
    setConnSyn<ConnKeyT, ConnStructT>(conn_key, conn_struct, val);
    break;
  }
  }
}


//////////////////////////////////////////////////////////////////////
// Get the float parameter param_name of an array of n_conn connections,
// identified by the indexes conn_ids[i], and put it in the array
// h_param_arr
// NOTE: host array should be pre-allocated to store n_conn elements
//////////////////////////////////////////////////////////////////////
template <class ConnKeyT, class ConnStructT>
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
    GetConnectionFloatParamKernel<ConnKeyT, ConnStructT>
      <<<(n_conn+1023)/1024, 1024 >>>
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
template <class ConnKeyT, class ConnStructT>
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
    GetConnectionIntParamKernel<ConnKeyT, ConnStructT>
      <<<(n_conn+1023)/1024, 1024 >>>
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
// identified by the indexes conn_ids[i], to the value val
//////////////////////////////////////////////////////////////////////
template <class ConnKeyT, class ConnStructT>
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
    SetConnectionFloatParamKernel<ConnStructT>
      <<<(n_conn+1023)/1024, 1024 >>>
      (d_conn_ids, n_conn, val, i_param);
    // free allocated device memory
    CUDAFREECTRL("d_conn_ids",d_conn_ids);    
  }
  
  return 0;
}


//////////////////////////////////////////////////////////////////////
// Set the float parameter param_name of an array of n_conn connections,
// identified by the indexes conn_ids[i], using values from a distribution
// or from an array
//////////////////////////////////////////////////////////////////////
template <class ConnKeyT, class ConnStructT>
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
    SetConnectionFloatParamKernel<ConnStructT>
      <<<(n_conn+1023)/1024, 1024 >>>
      (d_conn_ids, n_conn, d_arr, i_param);
    // free allocated device memory
    CUDAFREECTRL("d_conn_ids",d_conn_ids);
    CUDAFREECTRL("d_arr",d_arr);
  }
  
  return 0;
}


//////////////////////////////////////////////////////////////////////
// Set the integer parameter param_name of an array of n_conn connections,
// identified by the indexes conn_ids[i], using the values from the array
// h_param_arr
//////////////////////////////////////////////////////////////////////
template <class ConnKeyT, class ConnStructT>
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
    SetConnectionIntParamKernel<ConnKeyT, ConnStructT>
      <<<(n_conn+1023)/1024, 1024 >>>
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
template <class ConnKeyT, class ConnStructT>
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
    SetConnectionIntParamKernel<ConnKeyT, ConnStructT>
      <<<(n_conn+1023)/1024, 1024 >>>
      (d_conn_ids, n_conn, val, i_param);
    // free allocated device memory
    CUDAFREECTRL("d_conn_ids",d_conn_ids);
  }
  
  return 0;
}


template <class ConnKeyT, class ConnStructT>
int64_t *NESTGPU::GetConnections(inode_t *i_source_pt, inode_t n_source,
				  inode_t *i_target_pt, inode_t n_target,
				  int syn_group, int64_t *n_conn)
{
  int64_t *h_conn_ids = NULL;
  int64_t *d_conn_ids = NULL;
  uint64_t n_src_tgt = (uint64_t)n_source * n_target;
  int64_t n_conn_ids = 0;
  
  if (n_src_tgt > 0) {
    //std::cout << "n_src_tgt " << n_src_tgt << "n_source " << n_source
    //	      << "n_target " << n_target << "\n";
    // sort source node index array in GPU memory
    inode_t *d_src_arr = sortArray(i_source_pt, n_source);
    // sort target node index array in GPU memory
    inode_t *d_tgt_arr = sortArray(i_target_pt, n_target);
    // Allocate array of combined source-target indexes (src_arr x tgt_arr)
    uint64_t *d_src_tgt_arr;
    CUDAMALLOCCTRL("&d_src_tgt_arr",&d_src_tgt_arr, n_src_tgt*sizeof(uint64_t));
    // Fill it with combined source-target indexes
    setSourceTargetIndexKernel<<<(n_src_tgt+1023)/1024, 1024>>>
      (n_src_tgt, n_source, n_target, d_src_tgt_arr, d_src_arr, d_tgt_arr);
    // Allocate array of number of connections per source-target couple
    // and initialize it to 0
    uint64_t *d_src_tgt_conn_num;
    CUDAMALLOCCTRL("&d_src_tgt_conn_num",&d_src_tgt_conn_num,
		   (n_src_tgt + 1)*sizeof(uint64_t));
    gpuErrchk(cudaMemset(d_src_tgt_conn_num, 0,
			 (n_src_tgt + 1)*sizeof(uint64_t)));

    // Count number of connections per source-target couple
    CountConnectionsKernel<ConnKeyT, ConnStructT><<<(NConn+1023)/1024, 1024>>>
      (NConn, n_source, n_target, d_src_tgt_arr, d_src_tgt_conn_num, syn_group);
    // Evaluate exclusive sum of connections per source-target couple
    // Allocate array for cumulative sum
    uint64_t *d_src_tgt_conn_cumul;
    CUDAMALLOCCTRL("&d_src_tgt_conn_cumul",&d_src_tgt_conn_cumul,
			 (n_src_tgt + 1)*sizeof(uint64_t));
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
			   (n_src_tgt + 1)*sizeof(uint64_t)));
      // Fill array of connection indexes
      SetConnectionsIndexKernel<ConnKeyT, ConnStructT>
	<<<(NConn+1023)/1024, 1024>>>
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
// Get all parameters of an array of n_conn connections, identified by
// the indexes conn_ids[i], and put them in the arrays
// i_source, i_target, port, syn_group, delay, weight
// NOTE: host arrays should be pre-allocated to store n_conn elements
//////////////////////////////////////////////////////////////////////
template <class ConnKeyT, class ConnStructT>
int NESTGPU::GetConnectionStatus(int64_t *conn_ids, int64_t n_conn,
				 inode_t *source, inode_t *target, int *port,
				 int *syn_group, float *delay,
				 float *weight)
{
  if (n_conn > 0) {
    // declare pointers to arrays in device memory
    int64_t *d_conn_ids;
    inode_t *d_source;
    inode_t *d_target;
    int *d_port;
    int *d_syn_group;
    float *d_delay;
    float *d_weight;

    // allocate array of connection ids in device memory
    // and copy the ids from host to device array
    CUDAMALLOCCTRL("&d_conn_ids",&d_conn_ids, n_conn*sizeof(int64_t));
    gpuErrchk(cudaMemcpy(d_conn_ids, conn_ids, n_conn*sizeof(int64_t),
			 cudaMemcpyHostToDevice));

    // allocate arrays of connection parameters in device memory
    CUDAMALLOCCTRL("&d_source",&d_source, n_conn*sizeof(inode_t));
    CUDAMALLOCCTRL("&d_target",&d_target, n_conn*sizeof(inode_t));
    CUDAMALLOCCTRL("&d_port",&d_port, n_conn*sizeof(int));
    CUDAMALLOCCTRL("&d_syn_group",&d_syn_group, n_conn*sizeof(int));
    CUDAMALLOCCTRL("&d_delay",&d_delay, n_conn*sizeof(float));
    CUDAMALLOCCTRL("&d_weight",&d_weight, n_conn*sizeof(float));
    // host arrays
    
    // launch kernel to get connection parameters
    GetConnectionStatusKernel<ConnKeyT, ConnStructT>
      <<<(n_conn+1023)/1024, 1024 >>>
      (d_conn_ids, n_conn, d_source, d_target, d_port, d_syn_group,
       d_delay, d_weight);

    // copy connection parameters from device to host memory
    gpuErrchk(cudaMemcpy(source, d_source, n_conn*sizeof(inode_t),
			 cudaMemcpyDeviceToHost));
    
    gpuErrchk(cudaMemcpy(target, d_target, n_conn*sizeof(inode_t),
			 cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(port, d_port, n_conn*sizeof(int),
			 cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(syn_group, d_syn_group,
			 n_conn*sizeof(int),
			 cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(delay, d_delay, n_conn*sizeof(float),
			 cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(weight, d_weight, n_conn*sizeof(float),
			 cudaMemcpyDeviceToHost));
  }
  
  return 0;
}






#endif
