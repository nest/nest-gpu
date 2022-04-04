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
#include <stdlib.h>
#include <list>

#include "cuda_error.h"
#include "spike_buffer.h"
#include "getRealTime.h"

#include "spike_mpi.h"
#include "connect_mpi.h"

__device__ int locate(int val, int *data, int n);

__global__ void PushSpikeFromRemote(int n_spikes, int *spike_buffer_id,
           float *spike_height)
{
  int i_spike = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_spike<n_spikes) {
    int isb = spike_buffer_id[i_spike];
    float height = spike_height[i_spike];
    PushSpike(isb, height);
  }
}

__global__ void PushSpikeFromRemote(int n_spikes, int *spike_buffer_id)
{
  int i_spike = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_spike<n_spikes) {
    int isb = spike_buffer_id[i_spike];
    PushSpike(isb, 1.0);
  }
}

__global__ void AddOffset(int n_spikes, int *spike_buffer_id,
			  int i_remote_node_0)
{
  int i_spike = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_spike<n_spikes) {
    spike_buffer_id[i_spike] += i_remote_node_0;
  }
}

__constant__ bool NESTGPUMpiFlag;

#ifdef HAVE_MPI

__device__ int NExternalTargetHost;
__device__ int MaxSpikePerHost;

int *d_ExternalSpikeNum;
__device__ int *ExternalSpikeNum;

int *d_ExternalSpikeSourceNode; // [MaxSpikeNum];
__device__ int *ExternalSpikeSourceNode;

float *d_ExternalSpikeHeight; // [MaxSpikeNum];
__device__ float *ExternalSpikeHeight;

int *d_ExternalTargetSpikeNum;
__device__ int *ExternalTargetSpikeNum;

int *d_ExternalTargetSpikeNodeId;
__device__ int *ExternalTargetSpikeNodeId;

float *d_ExternalTargetSpikeHeight;
__device__ float *ExternalTargetSpikeHeight;

int *d_NExternalNodeTargetHost;
__device__ int *NExternalNodeTargetHost;

int **d_ExternalNodeTargetHostId;
__device__ int **ExternalNodeTargetHostId;

int **d_ExternalNodeId;
__device__ int **ExternalNodeId;

//int *d_ExternalSourceSpikeNum;
//__device__ int *ExternalSourceSpikeNum;

int *d_ExternalSourceSpikeNodeId;
__device__ int *ExternalSourceSpikeNodeId;

float *d_ExternalSourceSpikeHeight;
__device__ float *ExternalSourceSpikeHeight;

int *d_ExternalTargetSpikeCumul;
int *d_ExternalTargetSpikeNodeIdJoin;

int *h_ExternalTargetSpikeNum;
int *h_ExternalTargetSpikeCumul;
int *h_ExternalSourceSpikeNum;
int *h_ExternalTargetSpikeNodeId;
int *h_ExternalSourceSpikeNodeId;

//int *h_ExternalSpikeNodeId;

float *h_ExternalSpikeHeight;

MPI_Request *recv_mpi_request;

__device__ void PushExternalSpike(int i_source, float height)
{
  int pos = atomicAdd(ExternalSpikeNum, 1);
  if (pos>=MaxSpikePerHost) {
    printf("Number of spikes larger than MaxSpikePerHost: %d\n", MaxSpikePerHost);
    *ExternalSpikeNum = MaxSpikePerHost;
    return;
  }
  ExternalSpikeSourceNode[pos] = i_source;
  ExternalSpikeHeight[pos] = height;
}

__global__ void SendExternalSpike()
{
  int i_spike = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_spike < *ExternalSpikeNum) {
    int i_source = ExternalSpikeSourceNode[i_spike];
    float height = ExternalSpikeHeight[i_spike];
    int Nth = NExternalNodeTargetHost[i_source];
      
    for (int ith=0; ith<Nth; ith++) {
      int target_host_id = ExternalNodeTargetHostId[i_source][ith];
      int remote_node_id = ExternalNodeId[i_source][ith];
      int pos = atomicAdd(&ExternalTargetSpikeNum[target_host_id], 1);
      ExternalTargetSpikeNodeId[target_host_id*MaxSpikePerHost + pos]
	= remote_node_id;
      ExternalTargetSpikeHeight[target_host_id*MaxSpikePerHost + pos]
	= height;
    }
  }
}

__global__ void ExternalSpikeReset()
{
  *ExternalSpikeNum = 0;
  for (int ith=0; ith<NExternalTargetHost; ith++) {
    ExternalTargetSpikeNum[ith] = 0;
  }
}

int ConnectMpi::ExternalSpikeInit(int n_node, int n_hosts, int max_spike_per_host)
{
  SendSpikeToRemote_MPI_time_ = 0;
  RecvSpikeFromRemote_MPI_time_ = 0;
  SendSpikeToRemote_CUDAcp_time_ = 0;
  RecvSpikeFromRemote_CUDAcp_time_ = 0;
  JoinSpike_time_ = 0;

  int *h_NExternalNodeTargetHost = new int[n_node];
  int **h_ExternalNodeTargetHostId = new int*[n_node];
  int **h_ExternalNodeId = new int*[n_node];
  
  //h_ExternalSpikeNodeId = new int[max_spike_per_host];
  h_ExternalTargetSpikeNum = new int [n_hosts];
  h_ExternalTargetSpikeCumul = new int[n_hosts+1];
  h_ExternalSourceSpikeNum = new int[n_hosts];
  h_ExternalTargetSpikeNodeId = new int[n_hosts*(max_spike_per_host + 1)];
  h_ExternalSourceSpikeNodeId = new int[n_hosts*(max_spike_per_host + 1)];

  h_ExternalSpikeHeight = new float[max_spike_per_host];

  recv_mpi_request = new MPI_Request[n_hosts];
 
  gpuErrchk(cudaMalloc(&d_ExternalSpikeNum, sizeof(int)));
  gpuErrchk(cudaMalloc(&d_ExternalSpikeSourceNode,
		       max_spike_per_host*sizeof(int)));
  gpuErrchk(cudaMalloc(&d_ExternalSpikeHeight, max_spike_per_host*sizeof(int)));
  gpuErrchk(cudaMalloc(&d_ExternalTargetSpikeNum, n_hosts*sizeof(int)));

  //printf("n_hosts, max_spike_per_host: %d %d\n", n_hosts, max_spike_per_host);

  gpuErrchk(cudaMalloc(&d_ExternalTargetSpikeNodeId,
		       n_hosts*max_spike_per_host*sizeof(int)));
  gpuErrchk(cudaMalloc(&d_ExternalTargetSpikeHeight,
		       n_hosts*max_spike_per_host*sizeof(float)));
  //gpuErrchk(cudaMalloc(&d_ExternalSourceSpikeNum, n_hosts*sizeof(int)));
  gpuErrchk(cudaMalloc(&d_ExternalSourceSpikeNodeId, n_hosts*
		       max_spike_per_host*sizeof(int)));
  gpuErrchk(cudaMalloc(&d_ExternalSourceSpikeHeight, n_hosts*
		       max_spike_per_host*sizeof(float)));

  gpuErrchk(cudaMalloc(&d_ExternalTargetSpikeNodeIdJoin,
		       n_hosts*max_spike_per_host*sizeof(int)));
  gpuErrchk(cudaMalloc(&d_ExternalTargetSpikeCumul, (n_hosts+1)*sizeof(int)));

  gpuErrchk(cudaMalloc(&d_NExternalNodeTargetHost, n_node*sizeof(int)));
  gpuErrchk(cudaMalloc(&d_ExternalNodeTargetHostId, n_node*sizeof(int*)));
  gpuErrchk(cudaMalloc(&d_ExternalNodeId, n_node*sizeof(int*)));
 
  for (int i_source=0; i_source<n_node; i_source++) {
    std::vector< ExternalConnectionNode > *conn = &extern_connection_[i_source];
    int Nth = conn->size();
    h_NExternalNodeTargetHost[i_source] = Nth;
    if (Nth>0) {
       gpuErrchk(cudaMalloc(&h_ExternalNodeTargetHostId[i_source],
   			 Nth*sizeof(int)));
       gpuErrchk(cudaMalloc(&h_ExternalNodeId[i_source], Nth*sizeof(int)));
       int *target_host_arr = new int[Nth];
       int *node_id_arr = new int[Nth];
       for (int ith=0; ith<Nth; ith++) {
         target_host_arr[ith] = conn->at(ith).target_host_id;
         node_id_arr[ith] = conn->at(ith).remote_node_id;
       }
       cudaMemcpy(h_ExternalNodeTargetHostId[i_source], target_host_arr,
   	       Nth*sizeof(int), cudaMemcpyHostToDevice);
       cudaMemcpy(h_ExternalNodeId[i_source], node_id_arr,
   	       Nth*sizeof(int), cudaMemcpyHostToDevice);
       delete[] target_host_arr;
       delete[] node_id_arr;
     }
  }
  cudaMemcpy(d_NExternalNodeTargetHost, h_NExternalNodeTargetHost,
	     n_node*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ExternalNodeTargetHostId, h_ExternalNodeTargetHostId,
	     n_node*sizeof(int*), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ExternalNodeId, h_ExternalNodeId,
	     n_node*sizeof(int*), cudaMemcpyHostToDevice);

  DeviceExternalSpikeInit<<<1,1>>>(n_hosts, max_spike_per_host,
				   d_ExternalSpikeNum,
				   d_ExternalSpikeSourceNode,
				   d_ExternalSpikeHeight,
				   d_ExternalTargetSpikeNum,
				   d_ExternalTargetSpikeNodeId,
				   d_ExternalTargetSpikeHeight,
				   d_NExternalNodeTargetHost,
				   d_ExternalNodeTargetHostId,
				   d_ExternalNodeId
				   );
  delete[] h_NExternalNodeTargetHost;
  delete[] h_ExternalNodeTargetHostId;
  delete[] h_ExternalNodeId;

  return 0;
}

__global__ void DeviceExternalSpikeInit(int n_hosts,
					int max_spike_per_host,
					int *ext_spike_num,
					int *ext_spike_source_node,
					float *ext_spike_height,
					int *ext_target_spike_num,
					int *ext_target_spike_node_id,
					float *ext_target_spike_height,
					int *n_ext_node_target_host,
					int **ext_node_target_host_id,
					int **ext_node_id
					)
  
{
  NExternalTargetHost = n_hosts;
  MaxSpikePerHost =  max_spike_per_host;
  ExternalSpikeNum = ext_spike_num;
  ExternalSpikeSourceNode = ext_spike_source_node;
  ExternalSpikeHeight = ext_spike_height;
  ExternalTargetSpikeNum = ext_target_spike_num;
  ExternalTargetSpikeNodeId = ext_target_spike_node_id;
  ExternalTargetSpikeHeight = ext_target_spike_height;
  NExternalNodeTargetHost = n_ext_node_target_host;
  ExternalNodeTargetHostId = ext_node_target_host_id;
  ExternalNodeId = ext_node_id;
  *ExternalSpikeNum = 0;
  for (int ith=0; ith<NExternalTargetHost; ith++) {
    ExternalTargetSpikeNum[ith] = 0;
  }  
}

int ConnectMpi::SendSpikeToRemote(int n_hosts, int max_spike_per_host)
{
    MPI_Request request;
  int mpi_id, tag = 1;  // id is already in the class, remove
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_id);

  double time_mark = getRealTime();
  
  SendSpikeToRemote_CUDAcp_time_ += (getRealTime() - time_mark);

  JoinSpikes(n_hosts, max_spike_per_host);

  time_mark = getRealTime();
  for (int ih=0; ih<n_hosts; ih++) {
    if (ih == mpi_id) {
      continue;
    }
    // unpack spikes; array_idx is the first index of block for host ih
    int array_idx = h_ExternalTargetSpikeCumul[ih];
    int n_spikes = h_ExternalTargetSpikeCumul[ih+1] - array_idx;
    //printf("MPI_Send (src,tgt,nspike): %d %d %d\n", mpi_id, ih, n_spike);
    // tolti controlli, GPU_DIRECT
    MPI_Isend(&h_ExternalTargetSpikeNodeId[array_idx],
	      n_spikes, MPI_INT, ih, tag, MPI_COMM_WORLD, &request);
    //printf("MPI_Send nspikes (src,tgt,nspike): "
    //	   "%d %d %d\n", mpi_id, ih, n_spikes);
    //printf("MPI_Send 1st-neuron-idx (src,tgt,idx): "
    //	   "%d %d %d\n", mpi_id, ih,
    //	   h_ExternalTargetSpikeNodeId[array_idx]);
  }
  SendSpikeToRemote_MPI_time_ += (getRealTime() - time_mark);
  
  return 0;
}

int ConnectMpi::RecvSpikeFromRemote(int n_hosts, int max_spike_per_host)
				    
{
  std::list<int> recv_list;
  std::list<int>::iterator list_it;
  
  MPI_Status Stat;
  int mpi_id, tag = 1; // id is already in the class, remove
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_id);

  double time_mark = getRealTime();
  // tolto GPU direct
  for (int i_host=0; i_host<n_hosts; i_host++) {
    if (i_host == mpi_id) continue;
    recv_list.push_back(i_host);
    MPI_Irecv(&h_ExternalSourceSpikeNodeId[i_host*max_spike_per_host],
	      max_spike_per_host, MPI_INT, i_host, tag, MPI_COMM_WORLD,
	      &recv_mpi_request[i_host]);
  }
  
  while (recv_list.size()>0) {
    for (list_it=recv_list.begin(); list_it!=recv_list.end(); ++list_it) {
      int i_host = *list_it;
      int flag;
      MPI_Test(&recv_mpi_request[i_host], &flag, &Stat);
      if (flag) {
	int count;
	MPI_Get_count(&Stat, MPI_INT, &count);
	h_ExternalSourceSpikeNum[i_host] = count;
	recv_list.erase(list_it);
	break;
      }
    }
  }
  
  for (int i_host=0; i_host<n_hosts; i_host++) {
    if (i_host == mpi_id) {
      h_ExternalSourceSpikeNum[i_host] = 0;
      continue;
    }
    // check of number of received spikes
    int n_spikes = h_ExternalSourceSpikeNum[i_host];
    
    //printf("MPI_Recv nspikes (src,tgt,nspikes): %d %d %d\n", i_host,
    //	 mpi_id, n_spikes);
    //printf("MPI_Recv 1st-neuron-idx (src,tgt,idx): %d %d %d\n", i_host,
    //	 mpi_id, h_ExternalSourceSpikeNodeId[i_host*max_spike_per_host]);
  }
  RecvSpikeFromRemote_MPI_time_ += (getRealTime() - time_mark);
  
  return 0;
}

int ConnectMpi::CopySpikeFromRemote(int n_hosts, int max_spike_per_host,
				    int i_remote_node_0)
{
  double time_mark = getRealTime();
  int n_spike_tot = 0;
  for (int i_host=0; i_host<n_hosts; i_host++) {
    int n_spike = h_ExternalSourceSpikeNum[i_host];
    for (int i_spike=0; i_spike<n_spike; i_spike++) {
      h_ExternalSourceSpikeNodeId[n_spike_tot] =
	h_ExternalSourceSpikeNodeId[i_host*(max_spike_per_host + 1) + 1
				    + i_spike];
      n_spike_tot++;
    }
  }
  JoinSpike_time_ += (getRealTime() - time_mark);

  if (n_spike_tot>0) {
    time_mark = getRealTime();
    gpuErrchk(cudaMemcpy(d_ExternalSourceSpikeNodeId,
			 h_ExternalSourceSpikeNodeId,
			 n_spike_tot*sizeof(int), cudaMemcpyHostToDevice));
    RecvSpikeFromRemote_CUDAcp_time_ += (getRealTime() - time_mark);
    // tolto controllo flag spike height ed eventuale ricezione
    AddOffset<<<(n_spike_tot+1023)/1024, 1024>>>
      (n_spike_tot, d_ExternalSourceSpikeNodeId, i_remote_node_0);
    gpuErrchk( cudaPeekAtLastError() );
    cudaDeviceSynchronize();
    PushSpikeFromRemote<<<(n_spike_tot+1023)/1024, 1024>>>
      (n_spike_tot, d_ExternalSourceSpikeNodeId);
    gpuErrchk( cudaPeekAtLastError() );
    cudaDeviceSynchronize();
  }
  
  return n_spike_tot;
}



__global__ void JoinSpikeKernel(int n_hosts, int *ExternalTargetSpikeNum,
				int *ExternalTargetSpikeCumul,
                                int *ExternalTargetSpikeNodeId,
                                int *ExternalTargetSpikeNodeIdJoin,
                                int max_spike_per_host)
{
  const int i_host = blockIdx.x;
  if (i_host<n_hosts && ExternalTargetSpikeCumul[n_hosts]>0) {
    const int n_spikes = ExternalTargetSpikeNum[i_host];
    for (int i_spike = threadIdx.x; i_spike < n_spikes; i_spike += blockDim.x) {
      array_idx = ExternalTargetSpikeCumul[i_host] + i_spike;
      ExternalTargetSpikeNodeIdJoin[array_idx] =
        ExternalTargetSpikeNodeId[i_host*max_spike_per_host + i_spike];
    }
  }
}


int ConnectMpi::JoinSpikes(int n_hosts, int max_spike_per_host)
{
  double time_mark = getRealTime();
  
  time_mark = getRealTime();
  /*
  h_ExternalTargetSpikeCumul[0] = 0;
  for (int ih=0; ih<n_hosts; ih++) {
    h_ExternalTargetSpikeCumul[ih+1] = h_ExternalTargetSpikeCumul[ih]
      + h_ExternalTargetSpikeNum[ih];
  }
  int n_spike_tot = h_ExternalTargetSpikeCumul[n_hosts];
  
  gpuErrchk(cudaMemcpy(d_ExternalTargetSpikeCumul, h_ExternalTargetSpikeCumul,
                     (n_hosts+1)*sizeof(int), cudaMemcpyHostToDevice));
  */
  prefix_scan(d_ExternalTargetSpikeCumul, d_ExternalTargetSpikeNum, n_hosts+1,
  	      true);
  gpuErrchk(cudaMemcpyAsync(h_ExternalTargetSpikeCumul,
			    d_ExternalTargetSpikeCumul,
			    (n_hosts+1)*sizeof(int), cudaMemcpyDeviceToHost));
  //if (n_spike_tot>0) {
  JoinSpikeKernel<<<n_hosts, 1024>>>(n_hosts,
				     d_ExternalTargetSpikeNum,
				     d_ExternalTargetSpikeCumul,
				     d_ExternalTargetSpikeNodeId,
				     d_ExternalTargetSpikeNodeIdJoin,
				     max_spike_per_host);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() ); // necessary to access n_spike_tot
  int n_spike_tot = h_ExternalTargetSpikeCumul[n_hosts];
  gpuErrchk(cudaMemcpy(h_ExternalTargetSpikeNodeId,
		       d_ExternalTargetSpikeNodeIdJoin,
		       n_spike_tot*sizeof(int),
		       cudaMemcpyDeviceToHost));
  }
  JoinSpike_time_ += (getRealTime() - time_mark);
  
  return n_spike_tot;
}

#endif
