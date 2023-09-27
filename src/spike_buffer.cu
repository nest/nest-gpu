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
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_error.h"
#include "spike_buffer.h"
//#include "connect.h"
#include "send_spike.h"
#include "node_group.h"
#include "connect.h"
#include "spike_mpi.h"

#define LAST_SPIKE_TIME_GUARD 0x70000000

extern __constant__ double NESTGPUTime;
extern __constant__ long long NESTGPUTimeIdx;
extern __constant__ float NESTGPUTimeResolution;
extern __constant__ NodeGroupStruct NodeGroupArray[];
extern __device__ int16_t *NodeGroupMap;

__constant__ bool ExternalSpikeFlag;
__device__ int MaxSpikeBufferSize;
__device__ int NSpikeBuffer;
__device__ int MaxDelayNum;

int h_NSpikeBuffer;
bool ConnectionSpikeTimeFlag;

float *d_LastSpikeHeight; // [NSpikeBuffer];
__device__ float *LastSpikeHeight; //

long long *d_LastSpikeTimeIdx; // [NSpikeBuffer];
__device__ long long *LastSpikeTimeIdx; //

long long *d_LastRevSpikeTimeIdx; // [NSpikeBuffer];
__device__ long long *LastRevSpikeTimeIdx; //

unsigned short *d_ConnectionSpikeTime; // [NConnection];
__device__ unsigned short *ConnectionSpikeTime; //

//////////////////////////////////////////////////////////////////////

int *d_SpikeBufferSize; // [NSpikeBuffer];
__device__ int *SpikeBufferSize; // [NSpikeBuffer];
// SpikeBufferSize[i_spike_buffer];
// where i_spike_buffer is the source node index
// number of spikes stored in the buffer

int *d_SpikeBufferIdx0; // [NSpikeBuffer];
__device__ int *SpikeBufferIdx0; // [NSpikeBuffer];
// SpikeBufferIdx0[i_spike_buffer];
// where i_spike_buffer is the source node index
// index of most recent spike stored in the buffer

int *d_SpikeBufferTimeIdx; // [NSpikeBuffer*MaxSpikeBufferNum];
__device__ int *SpikeBufferTimeIdx; // [NSpikeBuffer*MaxSpikeBufferNum];
// SpikeBufferTimeIdx[i_spike*NSpikeBuffer+i_spike_buffer];
// time index of the spike

int *d_SpikeBufferConnIdx; // [NSpikeBuffer*MaxSpikeBufferNum];
__device__ int *SpikeBufferConnIdx; // [NSpikeBuffer*MaxSpikeBufferNum];
// SpikeBufferConnIdx[i_spike*NSpikeBuffer+i_spike_buffer];
// index of the next connection group that will emit this spike

float *d_SpikeBufferHeight; // [NSpikeBuffer*MaxSpikeBufferNum];
__device__ float *SpikeBufferHeight; // [NSpikeBuffer*MaxSpikeBufferNum];
// SpikeBufferHeight[i_spike*NSpikeBuffer+i_spike_buffer];
// spike height


////////////////////////////////////////////////////////////
// push a new spike in spike buffer of a node
////////////////////////////////////////////////////////////
// i_spike_buffer : node index
// height: spike multiplicity
////////////////////////////////////////////////////////////
__device__ void PushSpike(int i_spike_buffer, float height)
{
  LastSpikeTimeIdx[i_spike_buffer] = NESTGPUTimeIdx;
  LastSpikeHeight[i_spike_buffer] = height;
  int i_group = NodeGroupMap[i_spike_buffer];
  int den_delay_idx;
  float *den_delay_arr = NodeGroupArray[i_group].den_delay_arr_;
  // check if node has dendritic delay
  if (den_delay_arr != NULL) {
    int i_neuron = i_spike_buffer - NodeGroupArray[i_group].i_node_0_;
    int n_param = NodeGroupArray[i_group].n_param_;
    // dendritic delay index is stored in the parameter array
    // den_delay_arr points to the dendritic delay if the first
    // node of the group. The other are separate by steps = n_param
    den_delay_idx = (int)round(den_delay_arr[i_neuron*n_param]
			       /NESTGPUTimeResolution);
    //printf("isb %d\tden_delay_idx: %d\n", i_spike_buffer, den_delay_idx);
  }
  else {
    den_delay_idx = 0;
  }
  // printf("Node %d spikes at time %lld , den_delay_idx: %d\n",
  //	 i_spike_buffer, NESTGPUTimeIdx, den_delay_idx); 
  if (den_delay_idx==0) {
    // last time when spike is sent back to dendrites (e.g. for STDP)
    LastRevSpikeTimeIdx[i_spike_buffer] = NESTGPUTimeIdx;
  }

  if (ExternalSpikeFlag) {
    // if active spike should eventually be sent to remote connections
    //printf("PushExternalSpike i_spike_buffer: %d height: %f\n",
    //	   i_spike_buffer, height);
    PushExternalSpike(i_spike_buffer, height);
  }
  
  // if recording  spike counts is activated, increase counter
  if (NodeGroupArray[i_group].spike_count_ != NULL) {
    int i_node_0 = NodeGroupArray[i_group].i_node_0_;
    NodeGroupArray[i_group].spike_count_[i_spike_buffer-i_node_0]++;
  }

  // check if recording spike times is activated
  int max_n_rec_spike_times = NodeGroupArray[i_group].max_n_rec_spike_times_;
  if (max_n_rec_spike_times != 0) {
    int i_node_rel = i_spike_buffer - NodeGroupArray[i_group].i_node_0_;
    int n_rec_spike_times =
      NodeGroupArray[i_group].n_rec_spike_times_[i_node_rel];
    if (n_rec_spike_times>=max_n_rec_spike_times-1) {
      printf("Maximum number of recorded spike times exceeded"
	     " for spike buffer %d\n", i_spike_buffer);
    }
    else { // record spike time
      NodeGroupArray[i_group].rec_spike_times_
	[i_node_rel*max_n_rec_spike_times + n_rec_spike_times]
	= NESTGPUTime;
      NodeGroupArray[i_group].n_rec_spike_times_[i_node_rel]++;
    }
  }

  // spike should be stored if there are output connections
  // or if dendritic delay is > 0
  if (ConnGroupNum[i_spike_buffer]>0 || den_delay_idx>0) {
    int Ns = SpikeBufferSize[i_spike_buffer]; // n. of spikes in buffer
    if (Ns>=MaxSpikeBufferSize) {
      printf("Maximum number of spikes in spike buffer exceeded"
	     " for spike buffer %d\n", i_spike_buffer);
      //exit(0);
      return;
    }
    ///////////////////////////////////
    // push_front new spike in buffer
    //////////////////////////////////
    SpikeBufferSize[i_spike_buffer]++; // increase n. of spikes in buffer
    // the index of the most recent spike is0 should be decreased by 1
    int is0 = (SpikeBufferIdx0[i_spike_buffer] + MaxSpikeBufferSize - 1)
      % MaxSpikeBufferSize;
    SpikeBufferIdx0[i_spike_buffer] = is0;
    int i_arr = is0*NSpikeBuffer+i_spike_buffer; // spike index in array
    SpikeBufferTimeIdx[i_arr] = 0; // time index is initialized to 0
    SpikeBufferConnIdx[i_arr] = 0; // connect. group index is initialized to 0
    SpikeBufferHeight[i_arr] = height; // spike multiplicity
  }
}

////////////////////////////////////////////////////////////
// Update spike buffer of a node
////////////////////////////////////////////////////////////
__global__ void SpikeBufferUpdate()
{
  int i_spike_buffer = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_spike_buffer>=NSpikeBuffer) return;
  
  int i_group=NodeGroupMap[i_spike_buffer];
  int den_delay_idx;
  float *den_delay_arr = NodeGroupArray[i_group].den_delay_arr_;
  // check if node has dendritic delay
  if (den_delay_arr != NULL) {
    int i_neuron = i_spike_buffer - NodeGroupArray[i_group].i_node_0_;
    int n_param = NodeGroupArray[i_group].n_param_;
    // dendritic delay index is stored in the parameter array
    // den_delay_arr points to the dendritic delay if the first
    // node of the group. The other are separate by steps = n_param
    den_delay_idx = (int)round(den_delay_arr[i_neuron*n_param]
			       /NESTGPUTimeResolution);
    //printf("isb update %d\tden_delay_idx: %d\n", i_spike_buffer, den_delay_idx);
  }
  else {
    den_delay_idx = 0;
  }
  // flag for sending spikes back through dendrites (e.g. for STDP)
  bool rev_spike = false;
  int is0 = SpikeBufferIdx0[i_spike_buffer]; // index of most recent spike
  int Ns = SpikeBufferSize[i_spike_buffer]; // n. of spikes in buffer
  for (int is=0; is<Ns; is++) {
    int is1 = (is0  + is)%MaxSpikeBufferSize;
    int i_arr = is1*NSpikeBuffer+i_spike_buffer; // spike index in array
    int i_conn = SpikeBufferConnIdx[i_arr];
    int spike_time_idx = SpikeBufferTimeIdx[i_arr];
    //if (i_spike_buffer==1) {
    //printf("is %d st %d dd %d\n", is, spike_time_idx, den_delay_idx);
    //}
    if (spike_time_idx+1 == den_delay_idx) {
      rev_spike = true;
    }
    // connection index in array
    //int i_conn_arr = i_conn*NSpikeBuffer+i_spike_buffer;
    int ig = ConnGroupIdx0[i_spike_buffer] + i_conn;
    // if spike time matches connection group delay deliver it
    // to global spike array
    if (i_conn<ConnGroupNum[i_spike_buffer] &&
	spike_time_idx+1 == ConnGroupDelay[ig]) {
      // spike time matches connection group delay
      float height = SpikeBufferHeight[i_arr]; // spike multiplicity
      // deliver spike
      SendSpike(i_spike_buffer, i_conn, height, ConnGroupNConn[ig]);
      // increase index of the next conn. group that will emit this spike
      i_conn++;
      SpikeBufferConnIdx[i_arr] = i_conn;
    }
    // Check if the oldest spike should be removed from the buffer:
    // check if it is the oldest spike of the buffer
    // and if its connection group index is over the last connection group
    // and if spike time is greater than the dendritic delay
    if (is==Ns-1 && i_conn>=ConnGroupNum[i_spike_buffer]
	&& spike_time_idx+1>=den_delay_idx) {
      // in this case we don't need any more to keep track of the oldest spike
      SpikeBufferSize[i_spike_buffer]--; // so remove it from buffer
    }
    else {
      SpikeBufferTimeIdx[i_arr]++;
      // increase time index
    }
  }

  if (rev_spike) {
    LastRevSpikeTimeIdx[i_spike_buffer] = NESTGPUTimeIdx+1;
  }
}

__global__ void InitLastSpikeTimeIdx(unsigned int n_spike_buffers,
				       int spike_time_idx)
{
  unsigned int i_spike_buffer = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_spike_buffer>=n_spike_buffers) {
    return;
  }
  LastSpikeTimeIdx[i_spike_buffer] = spike_time_idx;
  LastRevSpikeTimeIdx[i_spike_buffer] = spike_time_idx;
}


int SpikeBufferInit(uint n_spike_buffers, int max_spike_buffer_size)
{
  //unsigned int n_spike_buffers = net_connection->connection_.size();
  h_NSpikeBuffer = n_spike_buffers;
  int max_delay_num = h_MaxDelayNum;
  //printf("mdn: %d\n", max_delay_num);
  
  gpuErrchk(cudaMalloc(&d_LastSpikeTimeIdx, n_spike_buffers*sizeof(long long)));
  gpuErrchk(cudaMalloc(&d_LastSpikeHeight, n_spike_buffers*sizeof(float)));
  gpuErrchk(cudaMalloc(&d_LastRevSpikeTimeIdx, n_spike_buffers
		       *sizeof(long long)));
  
  gpuErrchk(cudaMalloc(&d_SpikeBufferSize, n_spike_buffers*sizeof(int)));
  gpuErrchk(cudaMalloc(&d_SpikeBufferIdx0, n_spike_buffers*sizeof(int)));
  gpuErrchk(cudaMalloc(&d_SpikeBufferTimeIdx,
		       n_spike_buffers*max_spike_buffer_size*sizeof(int)));
  gpuErrchk(cudaMalloc(&d_SpikeBufferConnIdx,
		       n_spike_buffers*max_spike_buffer_size*sizeof(int)));
  gpuErrchk(cudaMalloc(&d_SpikeBufferHeight,
		       n_spike_buffers*max_spike_buffer_size*sizeof(float)));
  gpuErrchk(cudaMemset(d_SpikeBufferSize, 0, n_spike_buffers*sizeof(int)));
  gpuErrchk(cudaMemset(d_SpikeBufferIdx0, 0, n_spike_buffers*sizeof(int)));

  if (ConnectionSpikeTimeFlag){
    //h_conn_spike_time = new unsigned short[n_conn];
    gpuErrchk(cudaMalloc(&d_ConnectionSpikeTime,
			 NConn*sizeof(unsigned short)));
    //gpuErrchk(cudaMemset(d_ConnectionSpikeTime, 0,
    //			 n_conn*sizeof(unsigned short)));
  }

  /*
  if(ConnectionSpikeTimeFlag) {
    cudaMemcpy(d_ConnectionGroupTargetSpikeTime,
	       h_ConnectionGroupTargetSpikeTime,
	       n_spike_buffers*max_delay_num*sizeof(unsigned short*),
	       cudaMemcpyHostToDevice);
  }
  */
  
  DeviceSpikeBufferInit<<<1,1>>>(n_spike_buffers, max_delay_num,
			   max_spike_buffer_size,
			   d_LastSpikeTimeIdx, d_LastSpikeHeight,	 
			   d_ConnectionSpikeTime,
			   d_SpikeBufferSize, d_SpikeBufferIdx0,
			   d_SpikeBufferTimeIdx,
			   d_SpikeBufferConnIdx, d_SpikeBufferHeight,
			   d_LastRevSpikeTimeIdx
				 );
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  
  InitLastSpikeTimeIdx
    <<<(n_spike_buffers+1023)/1024, 1024>>>
    (n_spike_buffers, LAST_SPIKE_TIME_GUARD);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  gpuErrchk(cudaMemset(d_LastSpikeHeight, 0,
		       n_spike_buffers*sizeof(unsigned short)));
  
  return 0;
}

__global__ void DeviceSpikeBufferInit(int n_spike_buffers, int max_delay_num,
				int max_spike_buffer_size,
				long long *last_spike_time_idx,
				float *last_spike_height,
				unsigned short *conn_spike_time,
				int *spike_buffer_size, int *spike_buffer_idx0,
				int *spike_buffer_time,
				int *spike_buffer_conn,
				float *spike_buffer_height,
				long long *last_rev_spike_time_idx)
{
  NSpikeBuffer = n_spike_buffers;
  MaxDelayNum = max_delay_num;
  MaxSpikeBufferSize = max_spike_buffer_size;
  LastSpikeTimeIdx = last_spike_time_idx;
  LastSpikeHeight = last_spike_height;
  ConnectionSpikeTime = conn_spike_time;
  SpikeBufferSize = spike_buffer_size;
  SpikeBufferIdx0 = spike_buffer_idx0;
  SpikeBufferTimeIdx = spike_buffer_time;
  SpikeBufferConnIdx = spike_buffer_conn;
  SpikeBufferHeight = spike_buffer_height;
  LastRevSpikeTimeIdx = last_rev_spike_time_idx;
}

