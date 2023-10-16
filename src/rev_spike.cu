/*
 *  rev_spike.cu
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

#include <config.h>
#include <stdio.h>
#include "spike_buffer.h"
#include "cuda_error.h"
#include "syn_model.h"
#include "connect.h"
#include <cub/cub.cuh>

#define SPIKE_TIME_DIFF_GUARD 15000 // must be less than 16384
#define SPIKE_TIME_DIFF_THR 10000 // must be less than GUARD

extern __constant__ long long NESTGPUTimeIdx;
extern __constant__ float NESTGPUTimeResolution;

unsigned int *d_RevSpikeNum;
unsigned int *d_RevSpikeTarget;
int *d_RevSpikeNConn;

extern __device__ void SynapseUpdate(int syn_group, float *w, float Dt);

__device__ unsigned int *RevSpikeNum;
__device__ unsigned int *RevSpikeTarget;
__device__ int *RevSpikeNConn;

int64_t h_NRevConn; 

int64_t *d_RevConnections; //[i] i=0,..., n_rev_conn - 1;
__device__ int64_t *RevConnections;

int *d_TargetRevConnectionSize; //[i] i=0,..., n_neuron-1;
__device__ int *TargetRevConnectionSize;

int64_t **d_TargetRevConnection; //[i][j] j=0,...,RevConnectionSize[i]-1
__device__ int64_t **TargetRevConnection;


// Count number of reverse connections per target node
__global__ void CountRevConnectionsKernel
(int64_t n_conn, int64_t *target_rev_connection_size_64)
{
  int64_t i_conn = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; 
  if (i_conn >= n_conn) return;

  uint i_block = (uint)(i_conn / ConnBlockSize);
  int64_t i_block_conn = i_conn % ConnBlockSize;
  connection_struct conn = ConnectionArray[i_block][i_block_conn];

  // TO BE IMPROVED BY CHECKING IF THE SYNAPSE TYPE OF THE GROUP
  // REQUIRES REVERSE CONNECTION
  // - Check syn_group of all connections.
  // - If syn_group>0 must create a reverse connection:
  if (conn.syn_group > 0) {
    // First get target node index
    uint target_port = conn.target_port;
    int i_target = target_port >> MaxPortNBits;
    // (atomic)increase the number of reverse connections for target
    atomicAdd((unsigned long long *)&target_rev_connection_size_64[i_target],
	      1);
  }
}



// Fill array of reverse connection indexes
__global__ void SetRevConnectionsIndexKernel
(int64_t n_conn, int *target_rev_connection_size,
 int64_t **target_rev_connection)
{
  int64_t i_conn = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; 
  if (i_conn >= n_conn) return;

  uint i_block = (uint)(i_conn / ConnBlockSize);
  int64_t i_block_conn = i_conn % ConnBlockSize;
  connection_struct conn = ConnectionArray[i_block][i_block_conn];

  // TO BE IMPROVED BY CHECKING IF THE SYNAPSE TYPE OF THE GROUP
  // REQUIRES REVERSE CONNECTION  
  // - Check syn_group of all connections.
  // - If syn_group>0 must create a reverse connection:
  if (conn.syn_group > 0) {
    // First get target node index
    uint target_port = conn.target_port;
    int i_target = target_port >> MaxPortNBits;
    // (atomic)increase the number of reverse connections for target
    int pos = atomicAdd(&target_rev_connection_size[i_target], 1);
    // Evaluate the pointer to the rev connection position in the
    // array of reverse connection indexes
    int64_t *rev_conn_pt = target_rev_connection[i_target] + pos;
    // Fill it with the connection index
    *rev_conn_pt = i_conn;
  }
}

__global__ void SetTargetRevConnectionsPtKernel
    (int n_spike_buffer, int64_t *target_rev_connection_cumul,
     int64_t **target_rev_connection, int64_t *rev_connections)
{
  int i_target = blockIdx.x * blockDim.x + threadIdx.x; 
  if (i_target >= n_spike_buffer) return;
  target_rev_connection[i_target] = rev_connections
    + target_rev_connection_cumul[i_target];
}

__global__ void RevConnectionInitKernel(int64_t *rev_conn,
					int *target_rev_conn_size,
					int64_t **target_rev_conn)
{
  RevConnections = rev_conn;
  TargetRevConnectionSize = target_rev_conn_size;
  TargetRevConnection = target_rev_conn;
}


__global__ void RevSpikeBufferUpdate(unsigned int n_node)
{
  unsigned int i_node = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_node >= n_node) {
    return;
  }
  long long target_spike_time_idx = LastRevSpikeTimeIdx[i_node];
  // Check if a spike reached the input synapses now
  if (target_spike_time_idx!=NESTGPUTimeIdx) {
    return;
  }
  int n_conn = TargetRevConnectionSize[i_node];
  if (n_conn>0) {
    unsigned int pos = atomicAdd(RevSpikeNum, 1);
    RevSpikeTarget[pos] = i_node;
    RevSpikeNConn[pos] = n_conn;
  }
}

__global__ void SetConnectionSpikeTime(unsigned int n_conn,
				       unsigned short time_idx)
{
  unsigned int i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) {
    return;
  }
  ConnectionSpikeTime[i_conn] = time_idx;
}

__global__ void ResetConnectionSpikeTimeUpKernel(unsigned int n_conn)
{
  unsigned int i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) {
    return;
  }
  unsigned short spike_time = ConnectionSpikeTime[i_conn];
  if (spike_time >= 0x8000) {
    ConnectionSpikeTime[i_conn] = 0;
  }
}

__global__ void ResetConnectionSpikeTimeDownKernel(unsigned int n_conn)
{
  unsigned int i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) {
    return;
  }
  unsigned short spike_time = ConnectionSpikeTime[i_conn];
  if (spike_time < 0x8000) {
    ConnectionSpikeTime[i_conn] = 0x8000;
  }
}

__global__ void DeviceRevSpikeInit(unsigned int *rev_spike_num,
				   unsigned int *rev_spike_target,
				   int *rev_spike_n_conn)
{
  RevSpikeNum = rev_spike_num;
  RevSpikeTarget = rev_spike_target;
  RevSpikeNConn = rev_spike_n_conn;
  *RevSpikeNum = 0;
}

__global__ void RevSpikeReset()
{
  *RevSpikeNum = 0;
}
  

int ResetConnectionSpikeTimeUp()
{  
  ResetConnectionSpikeTimeUpKernel
    <<<(NConn+1023)/1024, 1024>>>
    (NConn);
  gpuErrchk( cudaPeekAtLastError() );

  return 0;
}

int ResetConnectionSpikeTimeDown()
{  
  ResetConnectionSpikeTimeDownKernel
    <<<(NConn+1023)/1024, 1024>>>
    (NConn);
  gpuErrchk( cudaPeekAtLastError() );

  return 0;
}



int RevSpikeFree()
{
  CUDAFREECTRL("&d_RevSpikeNum",&d_RevSpikeNum);
  CUDAFREECTRL("&d_RevSpikeTarget",&d_RevSpikeTarget);
  CUDAFREECTRL("&d_RevSpikeNConn",&d_RevSpikeNConn);

  return 0;
}

int RevSpikeInit(uint n_spike_buffers)
{
  //printf("n_spike_buffers: %d\n", n_spike_buffers);

  //////////////////////////////////////////////////////////////////////
  /////// Organize reverse connections (new version)
  // CHECK THE GLOBAL VARIABLES THAT MUST BE CONVERTED TO 64 bit ARRAYS
  //////////////////////////////////////////////////////////////////////  
  // Alloc 64 bit array of number of reverse connections per target node
  // and initialize it to 0
  int64_t *d_target_rev_conn_size_64;
  int64_t *d_target_rev_conn_cumul;
  CUDAMALLOCCTRL("&d_target_rev_conn_size_64",&d_target_rev_conn_size_64,
		       (n_spike_buffers+1)*sizeof(int64_t));
  gpuErrchk(cudaMemset(d_target_rev_conn_size_64, 0,
		       (n_spike_buffers+1)*sizeof(int64_t)));
  // Count number of reverse connections per target node
  CountRevConnectionsKernel<<<(NConn+1023)/1024, 1024>>>
    (NConn, d_target_rev_conn_size_64);
  // Evaluate exclusive sum of reverse connections per target node
  // Allocate array for cumulative sum
  CUDAMALLOCCTRL("&d_target_rev_conn_cumul",&d_target_rev_conn_cumul,
		       (n_spike_buffers+1)*sizeof(int64_t));
  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
				d_target_rev_conn_size_64,
				d_target_rev_conn_cumul,
				n_spike_buffers+1);
  // Allocate temporary storage
  CUDAMALLOCCTRL("&d_temp_storage",&d_temp_storage, temp_storage_bytes);
  // Run exclusive prefix sum
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
				d_target_rev_conn_size_64,
				d_target_rev_conn_cumul,
				n_spike_buffers+1);
  // The last element is the total number of reverse connections
  gpuErrchk(cudaMemcpy(&h_NRevConn, &d_target_rev_conn_cumul[n_spike_buffers],
		       sizeof(int64_t), cudaMemcpyDeviceToHost));
  if (h_NRevConn > 0) {
    // Allocate array of reverse connection indexes
    // CHECK THAT d_RevConnections is of type int64_t array
    CUDAMALLOCCTRL("&d_RevConnections",&d_RevConnections, h_NRevConn*sizeof(int64_t));  
    // For each target node evaluate the pointer
    // to its first reverse connection using the exclusive sum
    // CHECK THAT d_TargetRevConnection is of type int64_t* pointer
    CUDAMALLOCCTRL("&d_TargetRevConnection",&d_TargetRevConnection, n_spike_buffers
			 *sizeof(int64_t*));
    SetTargetRevConnectionsPtKernel<<<(n_spike_buffers+1023)/1024, 1024>>>
      (n_spike_buffers, d_target_rev_conn_cumul,
       d_TargetRevConnection, d_RevConnections);
  
    // alloc 32 bit array of number of reverse connections per target node
    CUDAMALLOCCTRL("&d_TargetRevConnectionSize",&d_TargetRevConnectionSize,
			 n_spike_buffers*sizeof(int));
    // and initialize it to 0
    gpuErrchk(cudaMemset(d_TargetRevConnectionSize, 0,
			 n_spike_buffers*sizeof(int)));
    // Fill array of reverse connection indexes
    SetRevConnectionsIndexKernel<<<(NConn+1023)/1024, 1024>>>
      (NConn, d_TargetRevConnectionSize, d_TargetRevConnection);

    RevConnectionInitKernel<<<1,1>>>
      (d_RevConnections, d_TargetRevConnectionSize, d_TargetRevConnection);

    SetConnectionSpikeTime
      <<<(NConn+1023)/1024, 1024>>>
      (NConn, 0x8000);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    CUDAMALLOCCTRL("&d_RevSpikeNum",&d_RevSpikeNum, sizeof(unsigned int));
  
    CUDAMALLOCCTRL("&d_RevSpikeTarget",&d_RevSpikeTarget,
			 n_spike_buffers*sizeof(unsigned int));
    CUDAMALLOCCTRL("&d_RevSpikeNConn",&d_RevSpikeNConn,
			 n_spike_buffers*sizeof(int));

    DeviceRevSpikeInit<<<1,1>>>(d_RevSpikeNum, d_RevSpikeTarget,
				d_RevSpikeNConn);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }
  
  CUDAFREECTRL("d_temp_storage",d_temp_storage);
  CUDAFREECTRL("d_target_rev_conn_size_64",d_target_rev_conn_size_64);
  CUDAFREECTRL("d_target_rev_conn_cumul",d_target_rev_conn_cumul);

  return 0;
}
