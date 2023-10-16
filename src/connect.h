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

#include <curand.h>
#include <vector>

#include "cuda_error.h"
#include "connect_spec.h"
#include "nestgpu.h"

struct connection_struct
{
  int target_port;
  float weight;
  unsigned char syn_group;
};

extern uint h_MaxNodeNBits;
extern __device__ uint MaxNodeNBits;

extern uint h_MaxPortNBits;
extern __device__ uint MaxPortNBits;

extern uint h_PortMask;
extern __device__ uint PortMask;

extern uint *d_ConnGroupIdx0;
extern __device__ uint *ConnGroupIdx0;

extern int64_t *d_ConnGroupIConn0;
extern __device__ int64_t *ConnGroupIConn0;

//extern uint *d_ConnGroupDelay;
extern __device__ uint *ConnGroupDelay;

extern uint tot_conn_group_num;

extern int64_t NConn;

extern int64_t h_ConnBlockSize;
extern __device__ int64_t ConnBlockSize;

extern uint h_MaxDelayNum;

// it seems that there is no relevant advantage in using a constant array
// however better to keep this option ready and commented
extern std::vector<uint*> KeySubarray;
extern uint** d_SourceDelayArray;
extern __device__ uint** SourceDelayArray;
//extern __constant__ uint* SourceDelayArray[];

extern std::vector<connection_struct*> ConnectionSubarray;
extern connection_struct** d_ConnectionArray;
extern __device__ connection_struct** ConnectionArray;
//extern __constant__ connection_struct* ConnectionArray[];

int setMaxNodeNBits(int max_node_nbits);

int allocateNewBlocks(std::vector<uint*> &key_subarray,
		      std::vector<connection_struct*> &conn_subarray,
		      int64_t block_size, uint new_n_block);

int setConnectionWeights(curandGenerator_t &gen, void *d_storage,
			 connection_struct *conn_subarray, int64_t n_conn,
			 SynSpec &syn_spec);

int setConnectionDelays(curandGenerator_t &gen, void *d_storage,
			uint *key_subarray, int64_t n_conn,
			SynSpec &syn_spec, float time_resolution);

__global__ void setPort(connection_struct *conn_subarray, uint port,
			int64_t n_conn);

__global__ void setSynGroup(connection_struct *conn_subarray,
			    unsigned char syn_group, int64_t n_conn);

int organizeConnections(float time_resolution, uint n_node, int64_t n_conn,
			int64_t block_size,
			std::vector<uint*> &key_subarray,
			std::vector<connection_struct*> &conn_subarray);


int ConnectInit();

__device__ __forceinline__
uint GetNodeIndex(int i_node_0, int i_node_rel)
{
  return i_node_0 + i_node_rel;
}

__device__ __forceinline__
uint GetNodeIndex(int *i_node_0, int i_node_rel)
{
  return *(i_node_0 + i_node_rel);
}

template <class T>
__global__ void setSource(uint *key_subarray, uint *rand_val,
			  int64_t n_conn, T source, uint n_source)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) return;
  key_subarray[i_conn] = GetNodeIndex(source, rand_val[i_conn]%n_source);
}

template <class T>
__global__ void setTarget(connection_struct *conn_subarray, uint *rand_val,
			  int64_t n_conn, T target, uint n_target)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) return;
  conn_subarray[i_conn].target_port =
    GetNodeIndex(target, rand_val[i_conn]%n_target);
}

template <class T1, class T2>
__global__ void setOneToOneSourceTarget(uint *key_subarray,
					connection_struct *conn_subarray,
					int64_t n_block_conn,
					int64_t n_prev_conn,
					T1 source, T2 target)
{
  int64_t i_block_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_block_conn>=n_block_conn) return;
  int64_t i_conn = n_prev_conn + i_block_conn;
  uint i_source = GetNodeIndex(source, (int)(i_conn));
  uint i_target = GetNodeIndex(target, (int)(i_conn));
  key_subarray[i_block_conn] = i_source;
  conn_subarray[i_block_conn].target_port = i_target;
}

template <class T1, class T2>
__global__ void setAllToAllSourceTarget(uint *key_subarray,
					connection_struct *conn_subarray,
					int64_t n_block_conn,
					int64_t n_prev_conn,
					T1 source, uint n_source,
					T2 target, uint n_target)
{
  int64_t i_block_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_block_conn>=n_block_conn) return;
  int64_t i_conn = n_prev_conn + i_block_conn;
  uint i_source = GetNodeIndex(source, (int)(i_conn / n_target));
  uint i_target = GetNodeIndex(target, (int)(i_conn % n_target));
  key_subarray[i_block_conn] = i_source;
  conn_subarray[i_block_conn].target_port = i_target;
}

template <class T>
__global__ void setIndegreeTarget(connection_struct *conn_subarray,
				  int64_t n_block_conn,
				  int64_t n_prev_conn,
				  T target, uint indegree)
{
  int64_t i_block_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_block_conn>=n_block_conn) return;
  int64_t i_conn = n_prev_conn + i_block_conn;
  uint i_target = GetNodeIndex(target, (int)(i_conn / indegree));
  conn_subarray[i_block_conn].target_port = i_target;
}

template <class T>
__global__ void setOutdegreeSource(uint *key_subarray,
				   int64_t n_block_conn,
				   int64_t n_prev_conn,
				   T source, uint outdegree)
{
  int64_t i_block_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_block_conn>=n_block_conn) return;
  int64_t i_conn = n_prev_conn + i_block_conn;
  uint i_source = GetNodeIndex(source, (int)(i_conn / outdegree));
  key_subarray[i_block_conn] = i_source;
}

template <class T1, class T2>
int connect_one_to_one(curandGenerator_t &gen,
		       void *d_storage, float time_resolution,
		       std::vector<uint*> &key_subarray,
		       std::vector<connection_struct*> &conn_subarray,
		       int64_t &n_conn, int64_t block_size,
		       T1 source, T2 target,  int n_node,
		       SynSpec &syn_spec)
{
  uint64_t old_n_conn = n_conn;
  uint64_t n_new_conn = n_node;
  n_conn += n_new_conn; // new number of connections
  uint new_n_block = (uint)((n_conn + block_size - 1) / block_size);

  allocateNewBlocks(key_subarray, conn_subarray, block_size, new_n_block);

  //printf("Generating connections with one-to-one rule...\n");
  int64_t n_prev_conn = 0;
  uint ib0 = (uint)(old_n_conn / block_size);
  for (uint ib=ib0; ib<new_n_block; ib++) {
    uint64_t n_block_conn; // number of connections in a block
    uint64_t i_conn0; // index of first connection in a block
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

    setOneToOneSourceTarget<<<(n_block_conn+1023)/1024, 1024>>>
      (key_subarray[ib] + i_conn0, conn_subarray[ib] + i_conn0,
       n_block_conn, n_prev_conn, source, target);
    DBGCUDASYNC
    setConnectionWeights(gen, d_storage, conn_subarray[ib] + i_conn0,
			 n_block_conn, syn_spec);

    setConnectionDelays(gen, d_storage, key_subarray[ib] + i_conn0,
			n_block_conn, syn_spec, time_resolution);

    setPort<<<(n_block_conn+1023)/1024, 1024>>>
      (conn_subarray[ib] + i_conn0, syn_spec.port_, n_block_conn);
    DBGCUDASYNC
    setSynGroup<<<(n_block_conn+1023)/1024, 1024>>>
      (conn_subarray[ib] + i_conn0, syn_spec.syn_group_, n_block_conn);
    DBGCUDASYNC

    n_prev_conn += n_block_conn;
  }

  return 0;
}




template <class T1, class T2>
int connect_all_to_all(curandGenerator_t &gen,
		       void *d_storage, float time_resolution,
		       std::vector<uint*> &key_subarray,
		       std::vector<connection_struct*> &conn_subarray,
		       int64_t &n_conn, int64_t block_size,
		       T1 source, int n_source,
		       T2 target, int n_target,
		       SynSpec &syn_spec)
{
  uint64_t old_n_conn = n_conn;
  uint64_t n_new_conn = n_source*n_target;
  n_conn += n_new_conn; // new number of connections
  uint new_n_block = (uint)((n_conn + block_size - 1) / block_size);

  allocateNewBlocks(key_subarray, conn_subarray, block_size, new_n_block);

  //printf("Generating connections with all-to-all rule...\n");
  int64_t n_prev_conn = 0;
  uint ib0 = (uint)(old_n_conn / block_size);
  for (uint ib=ib0; ib<new_n_block; ib++) {
    uint64_t n_block_conn; // number of connections in a block
    uint64_t i_conn0; // index of first connection in a block
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

    setAllToAllSourceTarget<<<(n_block_conn+1023)/1024, 1024>>>
      (key_subarray[ib] + i_conn0, conn_subarray[ib] + i_conn0,
       n_block_conn, n_prev_conn, source, n_source, target, n_target);
    DBGCUDASYNC
    setConnectionWeights(gen, d_storage, conn_subarray[ib] + i_conn0,
			 n_block_conn, syn_spec);

    setConnectionDelays(gen, d_storage, key_subarray[ib] + i_conn0,
			n_block_conn, syn_spec, time_resolution);

    setPort<<<(n_block_conn+1023)/1024, 1024>>>
      (conn_subarray[ib] + i_conn0, syn_spec.port_, n_block_conn);
    DBGCUDASYNC
    setSynGroup<<<(n_block_conn+1023)/1024, 1024>>>
      (conn_subarray[ib] + i_conn0, syn_spec.syn_group_, n_block_conn);
    DBGCUDASYNC

    n_prev_conn += n_block_conn;
  }

  return 0;
}


template <class T1, class T2>
int connect_fixed_total_number(curandGenerator_t &gen,
			       void *d_storage, float time_resolution,
			       std::vector<uint*> &key_subarray,
			       std::vector<connection_struct*> &conn_subarray,
			       int64_t &n_conn, int64_t block_size,
			       int64_t total_num, T1 source, int n_source,
			       T2 target, int n_target,
			       SynSpec &syn_spec)
{
  if (total_num==0) return 0;
  uint64_t old_n_conn = n_conn;
  uint64_t n_new_conn = total_num;
  n_conn += n_new_conn; // new number of connections
  uint new_n_block = (uint)((n_conn + block_size - 1) / block_size);

  allocateNewBlocks(key_subarray, conn_subarray, block_size, new_n_block);

  //printf("Generating connections with fixed_total_number rule...\n");
  uint ib0 = (uint)(old_n_conn / block_size);
  for (uint ib=ib0; ib<new_n_block; ib++) {
    uint64_t n_block_conn; // number of connections in a block
    uint64_t i_conn0; // index of first connection in a block
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
    //printf("old_n_conn: %ld\n", old_n_conn);
    //printf("n_new_conn: %ld\n", n_new_conn);
    //printf("new_n_block: %d\n", new_n_block);
    //printf("ib: %d\n", ib);
    //printf("n_block_conn: %ld\n", n_block_conn);
    setSource<<<(n_block_conn+1023)/1024, 1024>>>
      (key_subarray[ib] + i_conn0, (uint*)d_storage, n_block_conn,
       source, n_source);
    DBGCUDASYNC

    // generate random target index in range 0 - n_neuron
    CURAND_CALL(curandGenerate(gen, (uint*)d_storage, n_block_conn));
    setTarget<<<(n_block_conn+1023)/1024, 1024>>>
      (conn_subarray[ib] + i_conn0, (uint*)d_storage, n_block_conn,
       target, n_target);
    DBGCUDASYNC

    setConnectionWeights(gen, d_storage, conn_subarray[ib] + i_conn0,
			 n_block_conn, syn_spec);

    setConnectionDelays(gen, d_storage, key_subarray[ib] + i_conn0,
			n_block_conn, syn_spec, time_resolution);

    setPort<<<(n_block_conn+1023)/1024, 1024>>>
      (conn_subarray[ib] + i_conn0, syn_spec.port_, n_block_conn);
    DBGCUDASYNC
    setSynGroup<<<(n_block_conn+1023)/1024, 1024>>>
      (conn_subarray[ib] + i_conn0, syn_spec.syn_group_, n_block_conn);
    DBGCUDASYNC

  }

  return 0;
}

template <class T1, class T2>
int connect_fixed_indegree(curandGenerator_t &gen,
			   void *d_storage, float time_resolution,
			   std::vector<uint*> &key_subarray,
			   std::vector<connection_struct*> &conn_subarray,
			   int64_t &n_conn, int64_t block_size,
			   int indegree, T1 source, int n_source,
			   T2 target, int n_target,
			   SynSpec &syn_spec)
{
  if (indegree<=0) return 0;
  uint64_t old_n_conn = n_conn;
  uint64_t n_new_conn = n_target*indegree;
  n_conn += n_new_conn; // new number of connections
  uint new_n_block = (uint)((n_conn + block_size - 1) / block_size);

  allocateNewBlocks(key_subarray, conn_subarray, block_size, new_n_block);

  //printf("Generating connections with fixed_indegree rule...\n");
  int64_t n_prev_conn = 0;
  uint ib0 = (uint)(old_n_conn / block_size);
  for (uint ib=ib0; ib<new_n_block; ib++) {
    uint64_t n_block_conn; // number of connections in a block
    uint64_t i_conn0; // index of first connection in a block
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
    //printf("old_n_conn: %ld\n", old_n_conn);
    //printf("n_new_conn: %ld\n", n_new_conn);
    //printf("new_n_block: %d\n", new_n_block);
    //printf("ib: %d\n", ib);
    //printf("n_block_conn: %ld\n", n_block_conn);
    setSource<<<(n_block_conn+1023)/1024, 1024>>>
      (key_subarray[ib] + i_conn0, (uint*)d_storage, n_block_conn,
       source, n_source);
    DBGCUDASYNC

    setIndegreeTarget<<<(n_block_conn+1023)/1024, 1024>>>
      (conn_subarray[ib] + i_conn0, n_block_conn, n_prev_conn,
       target, indegree);
    DBGCUDASYNC

    setConnectionWeights(gen, d_storage, conn_subarray[ib] + i_conn0,
			 n_block_conn, syn_spec);

    setConnectionDelays(gen, d_storage, key_subarray[ib] + i_conn0,
			n_block_conn, syn_spec, time_resolution);

    setPort<<<(n_block_conn+1023)/1024, 1024>>>
      (conn_subarray[ib] + i_conn0, syn_spec.port_, n_block_conn);
    DBGCUDASYNC
    setSynGroup<<<(n_block_conn+1023)/1024, 1024>>>
      (conn_subarray[ib] + i_conn0, syn_spec.syn_group_, n_block_conn);
    DBGCUDASYNC

    n_prev_conn += n_block_conn;
  }

  return 0;
}

template <class T1, class T2>
int connect_fixed_outdegree(curandGenerator_t &gen,
			   void *d_storage, float time_resolution,
			   std::vector<uint*> &key_subarray,
			   std::vector<connection_struct*> &conn_subarray,
			   int64_t &n_conn, int64_t block_size,
			   int outdegree, T1 source, int n_source,
			   T2 target, int n_target,
			   SynSpec &syn_spec)
{
  if (outdegree<=0) return 0;
  uint64_t old_n_conn = n_conn;
  uint64_t n_new_conn = n_source*outdegree;
  n_conn += n_new_conn; // new number of connections
  uint new_n_block = (uint)((n_conn + block_size - 1) / block_size);

  allocateNewBlocks(key_subarray, conn_subarray, block_size, new_n_block);

  //printf("Generating connections with fixed_outdegree rule...\n");
  int64_t n_prev_conn = 0;
  uint ib0 = (uint)(old_n_conn / block_size);
  for (uint ib=ib0; ib<new_n_block; ib++) {
    uint64_t n_block_conn; // number of connections in a block
    uint64_t i_conn0; // index of first connection in a block
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

    setOutdegreeSource<<<(n_block_conn+1023)/1024, 1024>>>
      (key_subarray[ib] + i_conn0, n_block_conn, n_prev_conn,
       source, outdegree);
    DBGCUDASYNC

    // generate random target index in range 0 - n_neuron
    CURAND_CALL(curandGenerate(gen, (uint*)d_storage, n_block_conn));
    setTarget<<<(n_block_conn+1023)/1024, 1024>>>
      (conn_subarray[ib] + i_conn0, (uint*)d_storage, n_block_conn,
       target, n_target);
    DBGCUDASYNC

    setConnectionWeights(gen, d_storage, conn_subarray[ib] + i_conn0,
			 n_block_conn, syn_spec);

    setConnectionDelays(gen, d_storage, key_subarray[ib] + i_conn0,
			n_block_conn, syn_spec, time_resolution);

    setPort<<<(n_block_conn+1023)/1024, 1024>>>
      (conn_subarray[ib] + i_conn0, syn_spec.port_, n_block_conn);
    DBGCUDASYNC
    setSynGroup<<<(n_block_conn+1023)/1024, 1024>>>
      (conn_subarray[ib] + i_conn0, syn_spec.syn_group_, n_block_conn);
    DBGCUDASYNC

    n_prev_conn += n_block_conn;
  }

  return 0;
}

template <class T1, class T2>
int NESTGPU::_ConnectOneToOne
(curandGenerator_t &gen, T1 source, T2 target, int n_node, SynSpec &syn_spec)
{
  //printf("In new specialized connection one-to-one\n");

  void *d_storage;
  CUDAMALLOCCTRL("&d_storage",&d_storage, h_ConnBlockSize*sizeof(int));

  connect_one_to_one(gen, d_storage, time_resolution_,
		     KeySubarray, ConnectionSubarray, NConn,
		     h_ConnBlockSize, source, target, n_node, syn_spec);
  CUDAFREECTRL("d_storage",d_storage);

  return 0;
}

template <class T1, class T2>
int NESTGPU::_ConnectAllToAll
(curandGenerator_t &gen, T1 source, int n_source, T2 target, int n_target,
 SynSpec &syn_spec)
{
  //printf("In new specialized connection all-to-all\n");

  void *d_storage;
  CUDAMALLOCCTRL("&d_storage",&d_storage, h_ConnBlockSize*sizeof(int));

  connect_all_to_all(gen, d_storage, time_resolution_,
		     KeySubarray, ConnectionSubarray, NConn,
		     h_ConnBlockSize, source, n_source,
		     target, n_target, syn_spec);
  CUDAFREECTRL("d_storage",d_storage);

  return 0;
}

template <class T1, class T2>
int NESTGPU::_ConnectFixedTotalNumber
(curandGenerator_t &gen, T1 source, int n_source, T2 target, int n_target,
 int total_num, SynSpec &syn_spec)
{
  //printf("In new specialized connection fixed-total-number\n");

  void *d_storage;
  CUDAMALLOCCTRL("&d_storage",&d_storage, h_ConnBlockSize*sizeof(int));

  connect_fixed_total_number(gen, d_storage, time_resolution_,
			     KeySubarray, ConnectionSubarray, NConn,
			     h_ConnBlockSize, total_num, source, n_source,
			     target, n_target, syn_spec);
  CUDAFREECTRL("d_storage",d_storage);

  return 0;
}

template <class T1, class T2>
int NESTGPU::_ConnectFixedIndegree
(curandGenerator_t &gen, T1 source, int n_source, T2 target, int n_target,
 int indegree, SynSpec &syn_spec)
{
  //printf("In new specialized connection fixed-indegree\n");

  void *d_storage;
  CUDAMALLOCCTRL("&d_storage",&d_storage, h_ConnBlockSize*sizeof(int));

  connect_fixed_indegree(gen, d_storage, time_resolution_,
			 KeySubarray, ConnectionSubarray, NConn,
			 h_ConnBlockSize, indegree, source, n_source,
			 target, n_target, syn_spec);
  CUDAFREECTRL("d_storage",d_storage);

  return 0;
}

template <class T1, class T2>
int NESTGPU::_ConnectFixedOutdegree
(curandGenerator_t &gen, T1 source, int n_source, T2 target, int n_target,
 int outdegree, SynSpec &syn_spec)
{
  //printf("In new specialized connection fixed-outdegree\n");

  void *d_storage;
  CUDAMALLOCCTRL("&d_storage",&d_storage, h_ConnBlockSize*sizeof(int));

  connect_fixed_outdegree(gen, d_storage, time_resolution_,
			  KeySubarray, ConnectionSubarray, NConn,
			  h_ConnBlockSize, outdegree, source, n_source,
			  target, n_target, syn_spec);
  CUDAFREECTRL("d_storage",d_storage);

  return 0;
}

#endif
