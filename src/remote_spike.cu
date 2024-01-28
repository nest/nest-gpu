/*
 *  remote_spike.cu
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

__constant__ bool have_remote_spike_height;

#include <config.h>

#include <stdio.h>
#include <stdlib.h>

#include "cuda_error.h"
#include "getRealTime.h"
#include "spike_buffer.h"
#include "utilities.h"

#include "remote_spike.h"

#include "nestgpu.h"
#include "remote_connect.h"
#include "scan.h"
#include "utilities.h"

// Simple kernel for pushing remote spikes in local spike buffers
// Version without spike multiplicity array (spike_height)
__global__ void
PushSpikeFromRemote( uint n_spikes, uint* spike_buffer_id )
{
  uint i_spike = threadIdx.x + blockIdx.x * blockDim.x;
  if ( i_spike < n_spikes )
  {
    uint isb = spike_buffer_id[ i_spike ];
    PushSpike( isb, 1.0 );
  }
}

__device__ uint NExternalTargetHost;
__device__ uint MaxSpikePerHost;

uint* d_ExternalSpikeNum;
__device__ uint* ExternalSpikeNum;

uint* d_ExternalSpikeSourceNode; // [MaxSpikeNum];
__device__ uint* ExternalSpikeSourceNode;

float* d_ExternalSpikeHeight; // [MaxSpikeNum];
__device__ float* ExternalSpikeHeight;

uint* d_ExternalTargetSpikeNum;
__device__ uint* ExternalTargetSpikeNum;

uint* d_ExternalTargetSpikeNodeId;
__device__ uint* ExternalTargetSpikeNodeId;

float* d_ExternalTargetSpikeHeight;
__device__ float* ExternalTargetSpikeHeight;

// uint *d_NExternalNodeTargetHost;
__device__ uint* NExternalNodeTargetHost;

// uint **d_ExternalNodeTargetHostId;
__device__ uint** ExternalNodeTargetHostId;

// uint **d_ExternalNodeId;
__device__ uint** ExternalNodeId;

// uint *d_ExternalSourceSpikeNum;
//__device__ uint *ExternalSourceSpikeNum;

uint* d_ExternalSourceSpikeNodeId;
__device__ uint* ExternalSourceSpikeNodeId;

float* d_ExternalSourceSpikeHeight;
__device__ float* ExternalSourceSpikeHeight;

uint* d_ExternalTargetSpikeIdx0;
__device__ uint* ExternalTargetSpikeIdx0;
uint* h_ExternalTargetSpikeIdx0;

uint* d_ExternalSourceSpikeIdx0;

uint* h_ExternalTargetSpikeNum;
uint* h_ExternalSourceSpikeNum;
uint* h_ExternalSourceSpikeIdx0;
uint* h_ExternalTargetSpikeNodeId;
uint* h_ExternalSourceSpikeNodeId;

// uint *h_ExternalSpikeNodeId;

float* h_ExternalSpikeHeight;

// Push in a dedicated array the spikes that must be sent externally
__device__ void
PushExternalSpike( uint i_source, float height )
{
  uint pos = atomicAdd( ExternalSpikeNum, 1 );
  if ( pos >= MaxSpikePerHost )
  {
    printf( "Number of spikes larger than MaxSpikePerHost: %d\n", MaxSpikePerHost );
    *ExternalSpikeNum = MaxSpikePerHost;
    return;
  }
  ExternalSpikeSourceNode[ pos ] = i_source;
  ExternalSpikeHeight[ pos ] = height;
}

// Push in a dedicated array the spikes that must be sent externally
// (version without spike height)
__device__ void
PushExternalSpike( uint i_source )
{
  uint pos = atomicAdd( ExternalSpikeNum, 1 );
  if ( pos >= MaxSpikePerHost )
  {
    printf( "Number of spikes larger than MaxSpikePerHost: %d\n", MaxSpikePerHost );
    *ExternalSpikeNum = MaxSpikePerHost;
    return;
  }
  ExternalSpikeSourceNode[ pos ] = i_source;
}

// Count the spikes that must be sent externally for each target host
__global__ void
countExternalSpikesPerTargetHost()
{
  const uint i_spike = blockIdx.x;
  if ( i_spike < *ExternalSpikeNum )
  {
    // printf("ExternalSpikeNum: %d\ti_spike: %d\n", *ExternalSpikeNum,
    // i_spike);
    uint i_source = ExternalSpikeSourceNode[ i_spike ];
    // printf("i_source: %d\n", i_source);
    uint Nth = NExternalNodeTargetHost[ i_source ];
    // printf("Nth: %d\n", Nth);

    for ( uint ith = threadIdx.x; ith < Nth; ith += blockDim.x )
    {
      // printf("ith: %d\n", ith);
      uint target_host_id = ExternalNodeTargetHostId[ i_source ][ ith ];
      // printf("target_host_id: %d\n", target_host_id);
      // uint remote_node_id = ExternalNodeId[i_source][ith];
      // printf("remote_node_id: %d\n", remote_node_id);
      // uint pos =
      atomicAdd( &ExternalTargetSpikeNum[ target_host_id ], 1 );
      // printf("pos: %d\n", pos);
    }
  }
}

// Organize the spikes that must be sent externally for each target host
__global__ void
organizeExternalSpikesPerTargetHost()
{
  const uint i_spike = blockIdx.x;
  if ( i_spike < *ExternalSpikeNum )
  {
    // printf("ExternalSpikeNum: %d\ti_spike: %d\n", *ExternalSpikeNum,
    // i_spike);
    uint i_source = ExternalSpikeSourceNode[ i_spike ];
    // printf("i_source: %d\n", i_source);
    uint Nth = NExternalNodeTargetHost[ i_source ];
    // printf("Nth: %d\n", Nth);

    for ( uint ith = threadIdx.x; ith < Nth; ith += blockDim.x )
    {
      // printf("ith: %d\n", ith);
      uint target_host_id = ExternalNodeTargetHostId[ i_source ][ ith ];
      // printf("target_host_id: %d\n", target_host_id);
      uint remote_node_id = ExternalNodeId[ i_source ][ ith ];
      // printf("remote_node_id: %d\n", remote_node_id);
      uint pos = atomicAdd( &ExternalTargetSpikeNum[ target_host_id ], 1 );
      // printf("pos: %d\n", pos);
      uint i_arr = ExternalTargetSpikeIdx0[ target_host_id ] + pos;
      ExternalTargetSpikeNodeId[ i_arr ] = remote_node_id;
      if ( have_remote_spike_height )
      {
        float height = ExternalSpikeHeight[ i_spike ];
        // printf("height: %f\n", height);
        ExternalTargetSpikeHeight[ i_arr ] = height;
        // printf("ExternalTargetSpikeHeight assigned\n");
      }
    }
  }
}

// reset external spike counters
int
NESTGPU::ExternalSpikeReset()
{
  gpuErrchk( cudaMemset( d_ExternalSpikeNum, 0, sizeof( uint ) ) );
  gpuErrchk( cudaMemset( d_ExternalTargetSpikeNum, 0, n_hosts_ * sizeof( uint ) ) );

  return 0;
}

// initialize external spike arrays
int
NESTGPU::ExternalSpikeInit()
{
  SendSpikeToRemote_comm_time_ = 0;
  RecvSpikeFromRemote_comm_time_ = 0;

  SendSpikeToRemote_CUDAcp_time_ = 0;
  RecvSpikeFromRemote_CUDAcp_time_ = 0;

  // uint *h_NExternalNodeTargetHost = new uint[n_node];
  // uint **h_ExternalNodeTargetHostId = new uint*[n_node];
  // uint **h_ExternalNodeId = new uint*[n_node];

  h_ExternalTargetSpikeIdx0 = new uint[ n_hosts_ + 1 ];
  // h_ExternalSpikeNodeId = new uint[max_spike_per_host_];
  h_ExternalTargetSpikeNum = new uint[ n_hosts_ ];
  h_ExternalSourceSpikeNum = new uint[ n_hosts_ ];
  h_ExternalSourceSpikeIdx0 = new uint[ n_hosts_ + 1 ];
  h_ExternalTargetSpikeNodeId = new uint[ max_remote_spike_num_ ];
  h_ExternalSourceSpikeNodeId = new uint[ max_remote_spike_num_ ];

  CUDAMALLOCCTRL( "&d_ExternalSpikeNum", &d_ExternalSpikeNum, sizeof( uint ) );
  CUDAMALLOCCTRL( "&d_ExternalSpikeSourceNode", &d_ExternalSpikeSourceNode, max_spike_per_host_ * sizeof( uint ) );

  if ( remote_spike_height_ )
  {
    h_ExternalSpikeHeight = new float[ max_spike_per_host_ ];
    CUDAMALLOCCTRL( "&d_ExternalSpikeHeight", &d_ExternalSpikeHeight, max_spike_per_host_ * sizeof( float ) );
    CUDAMALLOCCTRL(
      "&d_ExternalTargetSpikeHeight", &d_ExternalTargetSpikeHeight, max_remote_spike_num_ * sizeof( float ) );
    CUDAMALLOCCTRL(
      "&d_ExternalSourceSpikeHeight", &d_ExternalSourceSpikeHeight, max_remote_spike_num_ * sizeof( float ) );
  }

  CUDAMALLOCCTRL( "&d_ExternalTargetSpikeNum", &d_ExternalTargetSpikeNum, n_hosts_ * sizeof( uint ) );

  // printf("n_hosts, max_spike_per_host: %d %d\n", n_hosts,
  // max_spike_per_host);

  CUDAMALLOCCTRL(
    "&d_ExternalTargetSpikeNodeId", &d_ExternalTargetSpikeNodeId, max_remote_spike_num_ * sizeof( uint ) );

  // CUDAMALLOCCTRL("&d_ExternalSourceSpikeNum",&d_ExternalSourceSpikeNum,
  // n_hosts*sizeof(int));
  CUDAMALLOCCTRL(
    "&d_ExternalSourceSpikeNodeId", &d_ExternalSourceSpikeNodeId, max_remote_spike_num_ * sizeof( uint ) );
  CUDAMALLOCCTRL( "&d_ExternalTargetSpikeIdx0", &d_ExternalTargetSpikeIdx0, ( n_hosts_ + 1 ) * sizeof( uint ) );

  CUDAMALLOCCTRL( "&d_ExternalSourceSpikeIdx0", &d_ExternalSourceSpikeIdx0, ( n_hosts_ + 1 ) * sizeof( uint ) );

  // CUDAMALLOCCTRL("&d_NExternalNodeTargetHost",&d_NExternalNodeTargetHost,
  // n_node*sizeof(uint));
  // CUDAMALLOCCTRL("&d_ExternalNodeTargetHostId",&d_ExternalNodeTargetHostId,
  // n_node*sizeof(uint*));
  // CUDAMALLOCCTRL("&d_ExternalNodeId",&d_ExternalNodeId,
  // n_node*sizeof(uint*));

  if ( remote_spike_height_ )
  {
    DeviceExternalSpikeInit<<< 1, 1 >>>( n_hosts_,
      max_spike_per_host_,
      d_ExternalSpikeNum,
      d_ExternalSpikeSourceNode,
      d_ExternalSpikeHeight,
      d_ExternalTargetSpikeNum,
      d_ExternalTargetSpikeIdx0,
      d_ExternalTargetSpikeNodeId,
      d_ExternalTargetSpikeHeight,
      conn_->getDevNTargetHosts(),
      conn_->getDevNodeTargetHosts(),
      conn_->getDevNodeTargetHostIMap() );
  }
  else
  {
    DeviceExternalSpikeInit<<< 1, 1 >>>( n_hosts_,
      max_spike_per_host_,
      d_ExternalSpikeNum,
      d_ExternalSpikeSourceNode,
      d_ExternalTargetSpikeNum,
      d_ExternalTargetSpikeIdx0,
      d_ExternalTargetSpikeNodeId,
      conn_->getDevNTargetHosts(),
      conn_->getDevNodeTargetHosts(),
      conn_->getDevNodeTargetHostIMap() );
  }
  // delete[] h_NExternalNodeTargetHost;
  // delete[] h_ExternalNodeTargetHostId;
  // delete[] h_ExternalNodeId;

  return 0;
}

// initialize external spike array pointers in the GPU
__global__ void
DeviceExternalSpikeInit( uint n_hosts,
  uint max_spike_per_host,
  uint* ext_spike_num,
  uint* ext_spike_source_node,
  float* ext_spike_height,
  uint* ext_target_spike_num,
  uint* ext_target_spike_idx0,
  uint* ext_target_spike_node_id,
  float* ext_target_spike_height,
  uint* n_ext_node_target_host,
  uint** ext_node_target_host_id,
  uint** ext_node_id )

{
  NExternalTargetHost = n_hosts;
  MaxSpikePerHost = max_spike_per_host;
  ExternalSpikeNum = ext_spike_num;
  ExternalSpikeSourceNode = ext_spike_source_node;
  ExternalSpikeHeight = ext_spike_height;
  ExternalTargetSpikeNum = ext_target_spike_num;
  ExternalTargetSpikeIdx0 = ext_target_spike_idx0, ExternalTargetSpikeNodeId = ext_target_spike_node_id;
  ExternalTargetSpikeHeight = ext_target_spike_height;
  NExternalNodeTargetHost = n_ext_node_target_host;
  ExternalNodeTargetHostId = ext_node_target_host_id;
  ExternalNodeId = ext_node_id;
  *ExternalSpikeNum = 0;
  for ( uint ith = 0; ith < NExternalTargetHost; ith++ )
  {
    ExternalTargetSpikeNum[ ith ] = 0;
  }
}

// initialize external spike array pointers in the GPU
// (version without spike height)
__global__ void
DeviceExternalSpikeInit( uint n_hosts,
  uint max_spike_per_host,
  uint* ext_spike_num,
  uint* ext_spike_source_node,
  uint* ext_target_spike_num,
  uint* ext_target_spike_idx0,
  uint* ext_target_spike_node_id,
  uint* n_ext_node_target_host,
  uint** ext_node_target_host_id,
  uint** ext_node_id )

{
  NExternalTargetHost = n_hosts;
  MaxSpikePerHost = max_spike_per_host;
  ExternalSpikeNum = ext_spike_num;
  ExternalSpikeSourceNode = ext_spike_source_node;
  ExternalSpikeHeight = nullptr;
  ExternalTargetSpikeNum = ext_target_spike_num;
  ExternalTargetSpikeIdx0 = ext_target_spike_idx0, ExternalTargetSpikeNodeId = ext_target_spike_node_id;
  ExternalTargetSpikeHeight = nullptr;
  NExternalNodeTargetHost = n_ext_node_target_host;
  ExternalNodeTargetHostId = ext_node_target_host_id;
  ExternalNodeId = ext_node_id;
  *ExternalSpikeNum = 0;
  for ( uint ith = 0; ith < NExternalTargetHost; ith++ )
  {
    ExternalTargetSpikeNum[ ith ] = 0;
  }
}

int
NESTGPU::organizeExternalSpikes( int n_ext_spikes )
{
  countExternalSpikesPerTargetHost<<< n_ext_spikes, 1024 >>>();
  CUDASYNC;
  prefix_scan( ( int* ) d_ExternalTargetSpikeIdx0, ( int* ) d_ExternalTargetSpikeNum, n_hosts_ + 1, true );
  DBGCUDASYNC;
  gpuErrchk( cudaMemset( d_ExternalTargetSpikeNum, 0, n_hosts_ * sizeof( uint ) ) );
  organizeExternalSpikesPerTargetHost<<< n_ext_spikes, 1024 >>>();
  CUDASYNC;

  return 0;
}

// pack spikes received from remote hosts
// and copy them to GPU memory
int
NESTGPU::CopySpikeFromRemote()
{
  int n_spike_tot = 0;
  h_ExternalSourceSpikeIdx0[ 0 ] = 0;
  // loop on hosts
  for ( int i_host = 0; i_host < n_hosts_; i_host++ )
  {
    int n_spike = h_ExternalSourceSpikeNum[ i_host ];
    h_ExternalSourceSpikeIdx0[ i_host + 1 ] = h_ExternalSourceSpikeIdx0[ i_host ] + n_spike;
    for ( int i_spike = 0; i_spike < n_spike; i_spike++ )
    {
      // pack spikes received from remote hosts
      h_ExternalSourceSpikeNodeId[ n_spike_tot ] =
        h_ExternalSourceSpikeNodeId[ i_host * max_spike_per_host_ + i_spike ];
      n_spike_tot++;
    }
  }

  if ( n_spike_tot >= max_remote_spike_num_ )
  {
    throw ngpu_exception( std::string( "Number of spikes to be received remotely " ) + std::to_string( n_spike_tot )
      + " larger than limit " + std::to_string( max_remote_spike_num_ ) );
  }

  if ( n_spike_tot > 0 )
  {
    double time_mark = getRealTime();
    // Memcopy will be synchronized
    // copy to GPU memory cumulative sum of number of spikes per source host
    gpuErrchk( cudaMemcpyAsync( d_ExternalSourceSpikeIdx0,
      h_ExternalSourceSpikeIdx0,
      ( n_hosts_ + 1 ) * sizeof( uint ),
      cudaMemcpyHostToDevice ) );
    DBGCUDASYNC;
    // copy to GPU memory packed spikes from remote hosts
    gpuErrchk( cudaMemcpyAsync( d_ExternalSourceSpikeNodeId,
      h_ExternalSourceSpikeNodeId,
      n_spike_tot * sizeof( uint ),
      cudaMemcpyHostToDevice ) );
    DBGCUDASYNC;
    RecvSpikeFromRemote_CUDAcp_time_ += ( getRealTime() - time_mark );
    // convert node map indexes to spike buffer indexes
    MapIndexToSpikeBufferKernel<<< n_hosts_, 1024 >>>(
      n_hosts_, d_ExternalSourceSpikeIdx0, d_ExternalSourceSpikeNodeId );
    DBGCUDASYNC;
    // convert node group indexes to spike buffer indexes
    // by adding the index of the first node of the node group
    // AddOffset<<<(n_spike_tot+1023)/1024, 1024 >>>
    //  (n_spike_tot, d_ExternalSourceSpikeNodeId, i_remote_node_0);
    // gpuErrchk( cudaPeekAtLastError() );
    // cudaDeviceSynchronize();
    // push remote spikes in local spike buffers
    PushSpikeFromRemote<<< ( n_spike_tot + 1023 ) / 1024, 1024 >>>( n_spike_tot, d_ExternalSourceSpikeNodeId );
    DBGCUDASYNC;
  }

  return n_spike_tot;
}
