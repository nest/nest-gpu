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
#include <stdio.h>
#include "spike_buffer.h"
#include "cuda_error.h"
#include "syn_model.h"

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


//////////////////////////////////////////////////////////////////////
// This is the function called by the nested loop
// that makes use of positive post-pre spike time difference
__device__ void NestedLoopFunction1(int i_spike, int i_target_rev_conn)
{
  unsigned int target = RevSpikeTarget[i_spike];
  unsigned int i_conn = TargetRevConnection[target][i_target_rev_conn];
  unsigned char syn_group = ConnectionSynGroup[i_conn];
  if (syn_group>0) {
    float *weight = &ConnectionWeight[i_conn];
    unsigned short spike_time_idx = ConnectionSpikeTime[i_conn];
    unsigned short time_idx = (unsigned short)(NESTGPUTimeIdx & 0xffff);
    unsigned short Dt_int = time_idx - spike_time_idx;
    if (Dt_int<MAX_SYN_DT) {
      SynapseUpdate(syn_group, weight, NESTGPUTimeResolution*Dt_int);
    }
  }
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
  

int ResetConnectionSpikeTimeUp(NetConnection *net_connection)
{  
  ResetConnectionSpikeTimeUpKernel
    <<<(net_connection->StoredNConnections()+1023)/1024, 1024>>>
    (net_connection->StoredNConnections());
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  return 0;
}

int ResetConnectionSpikeTimeDown(NetConnection *net_connection)
{  
  ResetConnectionSpikeTimeDownKernel
    <<<(net_connection->StoredNConnections()+1023)/1024, 1024>>>
    (net_connection->StoredNConnections());
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  return 0;
}

int RevSpikeInit(NetConnection *net_connection)
{
  int n_spike_buffers = net_connection->connection_.size();
  
  SetConnectionSpikeTime
    <<<(net_connection->StoredNConnections()+1023)/1024, 1024>>>
    (net_connection->StoredNConnections(), 0x8000);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  gpuErrchk(cudaMalloc(&d_RevSpikeNum, sizeof(unsigned int)));
  
  gpuErrchk(cudaMalloc(&d_RevSpikeTarget,
		       n_spike_buffers*sizeof(unsigned int)));
  gpuErrchk(cudaMalloc(&d_RevSpikeNConn,
		       n_spike_buffers*sizeof(int)));

  DeviceRevSpikeInit<<<1,1>>>(d_RevSpikeNum, d_RevSpikeTarget,
			      d_RevSpikeNConn);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  return 0;
}


int RevSpikeFree()
{
  gpuErrchk(cudaFree(&d_RevSpikeNum));
  gpuErrchk(cudaFree(&d_RevSpikeTarget));
  gpuErrchk(cudaFree(&d_RevSpikeNConn));

  return 0;
}
