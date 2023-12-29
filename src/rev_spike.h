/*
 *  rev_spike.h
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

#ifndef REVSPIKE_H
#define REVSPIKE_H

//#include "connect.h"
#include "spike_buffer.h"
#include "syn_model.h"
#include "get_spike.h"

extern __constant__ long long NESTGPUTimeIdx;
extern __constant__ float NESTGPUTimeResolution;

extern int64_t h_NRevConn;
extern unsigned int *d_RevSpikeNum;
extern unsigned int *d_RevSpikeTarget;
extern int *d_RevSpikeNConn;

extern __device__ unsigned int *RevSpikeNum;
extern __device__ unsigned int *RevSpikeTarget;
extern __device__ int *RevSpikeNConn;

extern int64_t *d_RevConnections; //[i] i=0,..., n_rev_conn - 1;
extern __device__ int64_t *RevConnections;

extern int *d_TargetRevConnectionSize; //[i] i=0,..., n_neuron-1;
extern __device__ int *TargetRevConnectionSize;

extern int64_t **d_TargetRevConnection; //[i][j] j=0,...,RevConnectionSize[i]-1
extern __device__ int64_t **TargetRevConnection;



__global__ void SetTargetRevConnectionsPtKernel
(int n_spike_buffer, int64_t *target_rev_connection_cumul,
 int64_t **target_rev_connection, int64_t *rev_connections);

__global__ void RevConnectionInitKernel(int64_t *rev_conn,
					int *target_rev_conn_size,
					int64_t **target_rev_conn);

__global__ void SetConnectionSpikeTime(unsigned int n_conn,
				       unsigned short time_idx);

__global__ void DeviceRevSpikeInit(unsigned int *rev_spike_num,
				   unsigned int *rev_spike_target,
				   int *rev_spike_n_conn);

__global__ void RevSpikeReset();

__global__ void RevSpikeBufferUpdate(unsigned int n_node);

int RevSpikeFree();

int ResetConnectionSpikeTimeDown();

int ResetConnectionSpikeTimeUp();

//template<int i_func>
//__device__  __forceinline__ void NestedLoopFunction(int i_spike, int i_syn);

//////////////////////////////////////////////////////////////////////
// This is the function called by the nested loop
// that makes use of positive post-pre spike time difference
template<class ConnKeyT, class ConnStructT>
__device__ __forceinline__ void NestedLoopFunction1
(int i_spike, int i_target_rev_conn)
{
  unsigned int target = RevSpikeTarget[i_spike];
  int64_t i_conn = TargetRevConnection[target][i_target_rev_conn];
  uint i_block = (uint)(i_conn / ConnBlockSize);
  int64_t i_block_conn = i_conn % ConnBlockSize;
  //connection_struct conn = ConnectionArray[i_block][i_block_conn];
  //unsigned char syn_group = conn.target_port_syn & SynMask;
  ConnKeyT &conn_key =
    ((ConnKeyT**)ConnKeyArray)[i_block][i_block_conn];
  ConnStructT &conn_struct =
    ((ConnStructT**)ConnStructArray)[i_block][i_block_conn];
  uint syn_group = getConnSyn<ConnKeyT, ConnStructT>(conn_key, conn_struct);
  
  // TO BE IMPROVED BY CHECKING IF THE SYNAPSE TYPE OF THE GROUP
  // REQUIRES AN UPDATE BASED ON POST-PRE SPIKE TIME DIFFERENCE
  if (syn_group>0) {
    unsigned short spike_time_idx = ConnectionSpikeTime[i_conn];
    unsigned short time_idx = (unsigned short)(NESTGPUTimeIdx & 0xffff);
    unsigned short Dt_int = time_idx - spike_time_idx;

    //printf("rev spike target %d i_target_rev_conn %d "
    //	   "i_conn %lld weight %f syn_group %d "
    //	   "TimeIdx %lld CST %d Dt %d\n",
    //	   target, i_target_rev_conn, i_conn, conn.weight, syn_group,
    //	   NESTGPUTimeIdx, spike_time_idx, Dt_int);
   
    if (Dt_int<MAX_SYN_DT) {
      SynapseUpdate(syn_group,
		    &(conn_struct.weight),
		    NESTGPUTimeResolution*Dt_int);
    }
  }
}


// Count number of reverse connections per target node
template <class ConnKeyT, class ConnStructT>
__global__ void CountRevConnectionsKernel
(int64_t n_conn, int64_t *target_rev_connection_size_64)
{
  int64_t i_conn = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; 
  if (i_conn >= n_conn) return;

  uint i_block = (uint)(i_conn / ConnBlockSize);
  int64_t i_block_conn = i_conn % ConnBlockSize;
  ConnKeyT &conn_key =
    ((ConnKeyT**)ConnKeyArray)[i_block][i_block_conn];
  ConnStructT &conn_struct =
    ((ConnStructT**)ConnStructArray)[i_block][i_block_conn];

  // TO BE IMPROVED BY CHECKING IF THE SYNAPSE TYPE OF THE GROUP
  // REQUIRES REVERSE CONNECTION
  // - Check syn_group of all connections.
  // - If syn_group>0 must create a reverse connection:
  uint syn_group = getConnSyn<ConnKeyT, ConnStructT>(conn_key, conn_struct);
  if (syn_group > 0) {
    // First get target node index
    uint i_target = getConnTarget<ConnStructT>(conn_struct);
    // (atomic)increase the number of reverse connections for target
    atomicAdd((unsigned long long *)&target_rev_connection_size_64[i_target],
	      1);
  }
}

// Fill array of reverse connection indexes
template <class ConnKeyT, class ConnStructT>
__global__ void SetRevConnectionsIndexKernel
(int64_t n_conn, int *target_rev_connection_size,
 int64_t **target_rev_connection)
{
  int64_t i_conn = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; 
  if (i_conn >= n_conn) return;

  uint i_block = (uint)(i_conn / ConnBlockSize);
  int64_t i_block_conn = i_conn % ConnBlockSize;
  ConnKeyT &conn_key =
    ((ConnKeyT**)ConnKeyArray)[i_block][i_block_conn];
  ConnStructT &conn_struct =
    ((ConnStructT**)ConnStructArray)[i_block][i_block_conn];
  
  // TO BE IMPROVED BY CHECKING IF THE SYNAPSE TYPE OF THE GROUP
  // REQUIRES REVERSE CONNECTION  
  // - Check syn_group of all connections.
  // - If syn_group>0 must create a reverse connection:
  uint syn_group = getConnSyn<ConnKeyT, ConnStructT>(conn_key, conn_struct);
  if (syn_group > 0) {
  // First get target node index
  uint i_target = getConnTarget<ConnStructT>(conn_struct);
    // (atomic)increase the number of reverse connections for target
    int pos = atomicAdd(&target_rev_connection_size[i_target], 1);
    // Evaluate the pointer to the rev connection position in the
    // array of reverse connection indexes
    int64_t *rev_conn_pt = target_rev_connection[i_target] + pos;
    // Fill it with the connection index
    *rev_conn_pt = i_conn;
  }
}

template <class ConnKeyT, class ConnStructT>
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
  CountRevConnectionsKernel<ConnKeyT, ConnStructT><<<(NConn+1023)/1024, 1024>>>
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
    SetRevConnectionsIndexKernel<ConnKeyT, ConnStructT>
      <<<(NConn+1023)/1024, 1024>>>
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

#endif
