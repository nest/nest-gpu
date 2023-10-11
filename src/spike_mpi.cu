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
#include "utilities.h"
#include "spike_buffer.h"
#include "getRealTime.h"

#include "spike_mpi.h"

#include "connect_mpi.h"
#include "scan.h"
#include "utilities.h"
#include "remote_connect.h"

// Simple kernel for pushing remote spikes in local spike buffers
// Version with spike multiplicity array (spike_height) 
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

// Simple kernel for pushing remote spikes in local spike buffers
// Version without spike multiplicity array (spike_height) 
__global__ void PushSpikeFromRemote(int n_spikes, int *spike_buffer_id)
{
  int i_spike = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_spike<n_spikes) {
    int isb = spike_buffer_id[i_spike];
    PushSpike(isb, 1.0);
  }
}

// convert node group indexes to spike buffer indexes
// by adding the index of the first node of the node group  
//__global__ void AddOffset(int n_spikes, int *spike_buffer_id,
//			  int i_remote_node_0)
//{
//  int i_spike = threadIdx.x + blockIdx.x * blockDim.x;
//  if (i_spike<n_spikes) {
//    spike_buffer_id[i_spike] += i_remote_node_0;
//  }
//}

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

//int *d_NExternalNodeTargetHost;
__device__ int *NExternalNodeTargetHost;

//int **d_ExternalNodeTargetHostId;
__device__ int **ExternalNodeTargetHostId;

//int **d_ExternalNodeId;
__device__ int **ExternalNodeId;

//int *d_ExternalSourceSpikeNum;
//__device__ int *ExternalSourceSpikeNum;

int *d_ExternalSourceSpikeNodeId;
__device__ int *ExternalSourceSpikeNodeId;

float *d_ExternalSourceSpikeHeight;
__device__ float *ExternalSourceSpikeHeight;

int *d_ExternalTargetSpikeCumul;
int *d_ExternalTargetSpikeNodeIdJoin;
int *d_ExternalSourceSpikeCumul;

int *h_ExternalTargetSpikeNum;
int *h_ExternalTargetSpikeCumul;
int *h_ExternalSourceSpikeNum;
int *h_ExternalSourceSpikeCumul;
int *h_ExternalTargetSpikeNodeId;
int *h_ExternalSourceSpikeNodeId;

//int *h_ExternalSpikeNodeId;

float *h_ExternalSpikeHeight;

MPI_Request *recv_mpi_request;

// Push in a dedicated array the spikes that must be sent externally
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

// Properly organize the spikes that must be sent externally
__global__ void SendExternalSpike()
{
  int i_spike = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_spike < *ExternalSpikeNum) {
    //printf("ExternalSpikeNum: %d\ti_spike: %d\n", *ExternalSpikeNum, i_spike);
    int i_source = ExternalSpikeSourceNode[i_spike];
    //printf("i_source: %d\n", i_source);
    float height = ExternalSpikeHeight[i_spike];
    //printf("height: %f\n", height);
    int Nth = NExternalNodeTargetHost[i_source];
    //printf("Nth: %d\n", Nth);
    
    for (int ith=0; ith<Nth; ith++) {
      //printf("ith: %d\n", ith);
      int target_host_id = ExternalNodeTargetHostId[i_source][ith];
      //printf("target_host_id: %d\n", target_host_id);
      int remote_node_id = ExternalNodeId[i_source][ith];
      //printf("remote_node_id: %d\n", remote_node_id);
      int pos = atomicAdd(&ExternalTargetSpikeNum[target_host_id], 1);
      //printf("pos: %d\n", pos);
      ExternalTargetSpikeNodeId[target_host_id*MaxSpikePerHost + pos]
	= remote_node_id;
      //printf("ExternalTargetSpikeNodeId assigned\n");
      ExternalTargetSpikeHeight[target_host_id*MaxSpikePerHost + pos]
	= height;
      //printf("ExternalTargetSpikeHeight assigned\n");
    }
  }
}

// reset external spike counters
__global__ void ExternalSpikeReset()
{
  *ExternalSpikeNum = 0;
  for (int ith=0; ith<NExternalTargetHost; ith++) {
    ExternalTargetSpikeNum[ith] = 0;
  }
}

// initialize external spike arrays
int NESTGPU::ExternalSpikeInit(int n_hosts, int max_spike_per_host)
{
  SendSpikeToRemote_MPI_time_ = 0;
  RecvSpikeFromRemote_MPI_time_ = 0;
  SendSpikeToRemote_CUDAcp_time_ = 0;
  RecvSpikeFromRemote_CUDAcp_time_ = 0;
  JoinSpike_time_ = 0;

  //int *h_NExternalNodeTargetHost = new int[n_node];
  //int **h_ExternalNodeTargetHostId = new int*[n_node];
  //int **h_ExternalNodeId = new int*[n_node];
  
  //h_ExternalSpikeNodeId = new int[max_spike_per_host];
  h_ExternalTargetSpikeNum = new int [n_hosts];
  h_ExternalTargetSpikeCumul = new int[n_hosts+1];
  h_ExternalSourceSpikeNum = new int[n_hosts];
  h_ExternalSourceSpikeCumul = new int[n_hosts+1];
  h_ExternalTargetSpikeNodeId = new int[n_hosts*(max_spike_per_host + 1)];
  h_ExternalSourceSpikeNodeId = new int[n_hosts*(max_spike_per_host + 1)];

  h_ExternalSpikeHeight = new float[max_spike_per_host];

  recv_mpi_request = new MPI_Request[n_hosts];
 
  CUDAMALLOCCTRL("&d_ExternalSpikeNum",&d_ExternalSpikeNum, sizeof(int));
  CUDAMALLOCCTRL("&d_ExternalSpikeSourceNode",&d_ExternalSpikeSourceNode,
		       max_spike_per_host*sizeof(int));
  CUDAMALLOCCTRL("&d_ExternalSpikeHeight",&d_ExternalSpikeHeight, max_spike_per_host*sizeof(int));
  CUDAMALLOCCTRL("&d_ExternalTargetSpikeNum",&d_ExternalTargetSpikeNum, n_hosts*sizeof(int));

  //printf("n_hosts, max_spike_per_host: %d %d\n", n_hosts, max_spike_per_host);

  CUDAMALLOCCTRL("&d_ExternalTargetSpikeNodeId",&d_ExternalTargetSpikeNodeId,
		       n_hosts*max_spike_per_host*sizeof(int));
  CUDAMALLOCCTRL("&d_ExternalTargetSpikeHeight",&d_ExternalTargetSpikeHeight,
		       n_hosts*max_spike_per_host*sizeof(float));
  //CUDAMALLOCCTRL("&d_ExternalSourceSpikeNum",&d_ExternalSourceSpikeNum, n_hosts*sizeof(int));
  CUDAMALLOCCTRL("&d_ExternalSourceSpikeNodeId",&d_ExternalSourceSpikeNodeId, n_hosts*
		       max_spike_per_host*sizeof(int));
  CUDAMALLOCCTRL("&d_ExternalSourceSpikeHeight",&d_ExternalSourceSpikeHeight, n_hosts*
		       max_spike_per_host*sizeof(float));

  CUDAMALLOCCTRL("&d_ExternalTargetSpikeNodeIdJoin",&d_ExternalTargetSpikeNodeIdJoin,
		       n_hosts*max_spike_per_host*sizeof(int));
  CUDAMALLOCCTRL("&d_ExternalTargetSpikeCumul",&d_ExternalTargetSpikeCumul, (n_hosts+1)*sizeof(int));

  CUDAMALLOCCTRL("&d_ExternalSourceSpikeCumul",&d_ExternalSourceSpikeCumul, (n_hosts+1)*sizeof(int));
  
  //CUDAMALLOCCTRL("&d_NExternalNodeTargetHost",&d_NExternalNodeTargetHost, n_node*sizeof(int));
  //CUDAMALLOCCTRL("&d_ExternalNodeTargetHostId",&d_ExternalNodeTargetHostId, n_node*sizeof(int*));
  //CUDAMALLOCCTRL("&d_ExternalNodeId",&d_ExternalNodeId, n_node*sizeof(int*));

  /*
  for (int i_source=0; i_source<n_node; i_source++) {
    std::vector< ExternalConnectionNode > *conn = &extern_connection_[i_source];
    int Nth = conn->size();
    h_NExternalNodeTargetHost[i_source] = Nth;
    if (Nth>0) {
       CUDAMALLOCCTRL("&h_ExternalNodeTargetHostId[i_source]",&h_ExternalNodeTargetHostId[i_source],
   			 Nth*sizeof(int));
       CUDAMALLOCCTRL("&h_ExternalNodeId[i_source]",&h_ExternalNodeId[i_source], Nth*sizeof(int));
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
  */
  //std::cout << "DeviceExternalSpikeInit\n";
  //std::cout << "init d_n_target_hosts: " << d_n_target_hosts << "\n";
  DeviceExternalSpikeInit<<<1,1>>>(n_hosts, max_spike_per_host,
				   d_ExternalSpikeNum,
				   d_ExternalSpikeSourceNode,
				   d_ExternalSpikeHeight,
				   d_ExternalTargetSpikeNum,
				   d_ExternalTargetSpikeNodeId,
				   d_ExternalTargetSpikeHeight,
				   d_n_target_hosts,
				   d_node_target_hosts,
				   d_node_target_host_i_map
				   );
  //delete[] h_NExternalNodeTargetHost;
  //delete[] h_ExternalNodeTargetHostId;
  //delete[] h_ExternalNodeId;

  return 0;
}

// initialize external spike array pointers in the GPU
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


// Send spikes to remote MPI processes
int NESTGPU::SendSpikeToRemote(int n_hosts, int max_spike_per_host)
{
  MPI_Request request;
  int mpi_id, tag = 1;  // id is already in the class, can be removed
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_id);

  double time_mark = getRealTime();
  gpuErrchk(cudaMemcpy(h_ExternalTargetSpikeNum, d_ExternalTargetSpikeNum,
		       n_hosts*sizeof(int), cudaMemcpyDeviceToHost));
  SendSpikeToRemote_CUDAcp_time_ += (getRealTime() - time_mark);

  // pack the spikes in GPU memory and copy them to CPU
  int n_spike_tot = JoinSpikes(n_hosts, max_spike_per_host);

  time_mark = getRealTime();
  // copy spikes from GPU to CPU memory
  gpuErrchk(cudaMemcpy(h_ExternalTargetSpikeNodeId,
		       d_ExternalTargetSpikeNodeIdJoin,
		       n_spike_tot*sizeof(int),
		       cudaMemcpyDeviceToHost));
  SendSpikeToRemote_CUDAcp_time_ += (getRealTime() - time_mark);
  time_mark = getRealTime();

  // loop on remote MPI proc
  for (int ih=0; ih<n_hosts; ih++) {
    if (ih == mpi_id) { // skip self MPI proc
      continue;
    }
    // get index and size of spike packet that must be sent to MPI proc ih
    // array_idx is the first index of the packet for host ih
    int array_idx = h_ExternalTargetSpikeCumul[ih];
    int n_spikes = h_ExternalTargetSpikeCumul[ih+1] - array_idx;
    //printf("MPI_Send (src,tgt,nspike): %d %d %d\n", mpi_id, ih, n_spike);
    
    // nonblocking sent of spike packet to MPI proc ih
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

// Receive spikes from remote MPI processes
int NESTGPU::RecvSpikeFromRemote(int n_hosts, int max_spike_per_host)
  
{
  std::list<int> recv_list;
  std::list<int>::iterator list_it;
  
  MPI_Status Stat;
  int mpi_id, tag = 1; // id is already in the class, can be removed
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_id);
  
  double time_mark = getRealTime();
  
  // loop on remote MPI proc
  for (int i_host=0; i_host<n_hosts; i_host++) {
    if (i_host == mpi_id) continue; // skip self MPI proc
    recv_list.push_back(i_host); // insert MPI proc in list
    // start nonblocking MPI receive from MPI proc i_host
    MPI_Irecv(&h_ExternalSourceSpikeNodeId[i_host*max_spike_per_host],
	      max_spike_per_host, MPI_INT, i_host, tag, MPI_COMM_WORLD,
	      &recv_mpi_request[i_host]);
  }
  
  // loop until list is empty, i.e. until receive is complete
  // from all MPI proc
  while (recv_list.size()>0) {
    // loop on all hosts in the list
    for (list_it=recv_list.begin(); list_it!=recv_list.end(); ++list_it) {
      int i_host = *list_it;
      int flag;
      // check if receive is complete
      MPI_Test(&recv_mpi_request[i_host], &flag, &Stat);
      if (flag) {
	int count;
	// get spike count
	MPI_Get_count(&Stat, MPI_INT, &count);
	h_ExternalSourceSpikeNum[i_host] = count;
	// when receive is complete remove MPI proc from list
	recv_list.erase(list_it);
	break;
      }
    }
  }  
  h_ExternalSourceSpikeNum[mpi_id] = 0;
  RecvSpikeFromRemote_MPI_time_ += (getRealTime() - time_mark);
  
  return 0;
}

// pack spikes received from remote MPI processes
// and copy them to GPU memory
int NESTGPU::CopySpikeFromRemote(int n_hosts, int max_spike_per_host)
{
  double time_mark = getRealTime();
  int n_spike_tot = 0;
  h_ExternalSourceSpikeCumul[0] = 0;
  // loop on MPI proc
  for (int i_host=0; i_host<n_hosts; i_host++) {
    int n_spike = h_ExternalSourceSpikeNum[i_host];
    h_ExternalSourceSpikeCumul[i_host+1] =
      h_ExternalSourceSpikeCumul[i_host] + n_spike;
    for (int i_spike=0; i_spike<n_spike; i_spike++) {
      // pack spikes received from remote MPI processes
      h_ExternalSourceSpikeNodeId[n_spike_tot] =
	h_ExternalSourceSpikeNodeId[i_host*max_spike_per_host + i_spike];
      n_spike_tot++;
    }
  }
  JoinSpike_time_ += (getRealTime() - time_mark);
  
  if (n_spike_tot>0) {
    time_mark = getRealTime();
    // Memcopy will be synchronized    
    // copy to GPU memory cumulative sum of number of spikes per source host
    gpuErrchk(cudaMemcpyAsync(d_ExternalSourceSpikeCumul,
			      h_ExternalSourceSpikeCumul,
			      (n_hosts+1)*sizeof(int), cudaMemcpyHostToDevice));
    // copy to GPU memory packed spikes from remote MPI proc
    gpuErrchk(cudaMemcpyAsync(d_ExternalSourceSpikeNodeId,
			      h_ExternalSourceSpikeNodeId,
			      n_spike_tot*sizeof(int), cudaMemcpyHostToDevice));
    RecvSpikeFromRemote_CUDAcp_time_ += (getRealTime() - time_mark);
    // convert node map indexes to spike buffer indexes
    MapIndexToSpikeBufferKernel<<<n_hosts, 1024>>>(n_hosts,
						   d_ExternalSourceSpikeCumul,
						   d_ExternalSourceSpikeNodeId);
    // convert node group indexes to spike buffer indexes
    // by adding the index of the first node of the node group  
    //AddOffset<<<(n_spike_tot+1023)/1024, 1024>>>
    //  (n_spike_tot, d_ExternalSourceSpikeNodeId, i_remote_node_0);
    gpuErrchk( cudaPeekAtLastError() );
    cudaDeviceSynchronize();
    // push remote spikes in local spike buffers
    PushSpikeFromRemote<<<(n_spike_tot+1023)/1024, 1024>>>
      (n_spike_tot, d_ExternalSourceSpikeNodeId);
    gpuErrchk( cudaPeekAtLastError() );
    cudaDeviceSynchronize();
  }
  
  return n_spike_tot;
}

// pack the spikes in GPU memory that must be sent externally
__global__ void JoinSpikeKernel(int n_hosts, int *ExternalTargetSpikeCumul,
				int *ExternalTargetSpikeNodeId,
				int *ExternalTargetSpikeNodeIdJoin,
				int n_spike_tot, int max_spike_per_host)
{
  // parallel implementation of nested loop
  // outer loop index i_host = 0, ... , n_hosts
  // inner loop index i_spike = 0, ... , ExternalTargetSpikeNum[i_host];
  // array_idx is the index in the packed spike array
  int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (array_idx<n_spike_tot) {
    int i_host = locate(array_idx, ExternalTargetSpikeCumul, n_hosts + 1);
    while ((i_host < n_hosts) && (ExternalTargetSpikeCumul[i_host+1]
				  == ExternalTargetSpikeCumul[i_host])) {
      i_host++;
      if (i_host==n_hosts) return;
    }
    int i_spike = array_idx - ExternalTargetSpikeCumul[i_host];
    // packed spike array
    ExternalTargetSpikeNodeIdJoin[array_idx] =
      ExternalTargetSpikeNodeId[i_host*max_spike_per_host + i_spike];
  }
}

// pack the spikes in GPU memory that must be sent externally
// and copy them to CPU memory
int NESTGPU::JoinSpikes(int n_hosts, int max_spike_per_host)
{
  double time_mark = getRealTime();
  // the index in the packed array can be computed from the MPI proc index
  // and from the spike index using  a cumulative sum (prefix scan)
  // of the number of spikes per MPI proc
  // the cumulative sum is done both in CPU and in GPU
  prefix_scan(d_ExternalTargetSpikeCumul, d_ExternalTargetSpikeNum, n_hosts+1,
  	      true);
  h_ExternalTargetSpikeCumul[0] = 0;
  for (int ih=0; ih<n_hosts; ih++) {
    h_ExternalTargetSpikeCumul[ih+1] = h_ExternalTargetSpikeCumul[ih]
      + h_ExternalTargetSpikeNum[ih];
  }
  int n_spike_tot = h_ExternalTargetSpikeCumul[n_hosts];

  if (n_spike_tot>0) {
    // pack the spikes in GPU memory
    JoinSpikeKernel<<<(n_spike_tot+1023)/1024, 1024>>>(n_hosts,
		     d_ExternalTargetSpikeCumul,
		     d_ExternalTargetSpikeNodeId,
		     d_ExternalTargetSpikeNodeIdJoin,
		     n_spike_tot, max_spike_per_host);

    gpuErrchk( cudaPeekAtLastError() );
  }

  JoinSpike_time_ += (getRealTime() - time_mark);
  
  return n_spike_tot;
}

int NESTGPU::ConnectMpiInit(int argc, char *argv[])
{
#ifdef HAVE_MPI
  CheckUncalibrated("MPI connections cannot be initialized after calibration");
  int initialized;
  MPI_Initialized(&initialized);
  if (!initialized) {
    MPI_Init(&argc,&argv);
  }
  int n_hosts;
  int this_host;
  MPI_Comm_size(MPI_COMM_WORLD, &n_hosts);
  MPI_Comm_rank(MPI_COMM_WORLD, &this_host);
  mpi_flag_ = true;
  setHostNum(n_hosts);
  setThisHost(this_host);
  RemoteConnectionMapInit(n_hosts);
  
  return 0;
#else
  throw ngpu_exception("MPI is not available in your build");
#endif
}


int NESTGPU::MpiFinalize()
{
#ifdef HAVE_MPI
  if (mpi_flag_) {
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized) {
      MPI_Finalize();
    }
  }
  
  return 0;
#else
  throw ngpu_exception("MPI is not available in your build");
#endif
}



#endif

