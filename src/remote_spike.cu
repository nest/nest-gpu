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

__constant__ bool have_remote_spike_mul;

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

// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
/*
(
 uint* d_NExternalNodeTargetHostGroup,
 uint** d_ExternalNodeTargetHostGroupId,
 uint** d_ExternalHostGroupNodeId,
 uint* d_ExternalHostGroupSpikeNum,
 uint* d_ExternalHostGroupSpikeIdx0,
 uint* d_ExternalHostGroupSpikeNodeId,
 float* d_ExternalHostGroupSpikeMul
 ) 
*/
// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx




// number of host groups to which each node should send spikes
// NExternalNodeTargetHostGroup[ i_source ]
__device__ uint* NExternalNodeTargetHostGroup; // [n_local_nodes ]
uint* d_NExternalNodeTargetHostGroup;

// local id of the host group to which each node should send spikes
// ExternalNodeTargetHostGroupId[ i_source ][ itg ] itg = 0, ..., NExternalNodeTargetHostGroup[i_source]-1 
__device__ uint** ExternalNodeTargetHostGroupId; // [n_local_nodes ][num. of target host groups of the node ]
uint** d_ExternalNodeTargetHostGroupId;

// positions of local source nodes in host group node map
// ExternalHostGroupNodeId[ i_source ][ itg ] = host_group_local_source_node_map[group_local_id][i_source]
__device__ uint** ExternalHostGroupNodeId; // [n_local_nodes ][num. of target host groups of the node ]
uint** d_ExternalHostGroupNodeId;

// number of spikes that must be send to each host group
// ExternalHostGroupSpikeNum[ group_local_id ]
__device__ uint* ExternalHostGroupSpikeNum; // [ num. of local host groups ]
uint* d_ExternalHostGroupSpikeNum;
uint* h_ExternalHostGroupSpikeNum;

// index of the first spike of the spike subarray that must be sent to the host group group_local_id
// (the spikes to be sent to all host groups are flattened on a single one-dimensional array)
// ExternalHostGroupSpikeIdx0[ group_local_id ]
__device__ uint* ExternalHostGroupSpikeIdx0; // [ num. of local host groups ]
uint* d_ExternalHostGroupSpikeIdx0;
uint* h_ExternalHostGroupSpikeIdx0;

// flattened one-dimensional array of spikes that must be sent to all host groups
__device__ uint* ExternalHostGroupSpikeNodeId; // [MaxSpikeNum];
uint* d_ExternalHostGroupSpikeNodeId;
uint* h_ExternalHostGroupSpikeNodeId;

// flattened one-dimensional array of multiplicity of spikes that must be sent to all host groups
__device__ float* ExternalHostGroupSpikeMul; // [MaxSpikeNum];
float* d_ExternalHostGroupSpikeMul;
float* h_ExternalHostGroupSpikeMul;

__device__ uint NExternalTargetHost;
__device__ uint MaxSpikePerHost;

uint* d_ExternalSpikeNum;
__device__ uint* ExternalSpikeNum;

uint* d_ExternalSpikeSourceNode; // [MaxSpikeNum];
__device__ uint* ExternalSpikeSourceNode;

float* d_ExternalSpikeMul; // [MaxSpikeNum];
__device__ float* ExternalSpikeMul;

uint* d_ExternalTargetSpikeNum;
__device__ uint* ExternalTargetSpikeNum;

uint* d_ExternalTargetSpikeNodeId;
__device__ uint* ExternalTargetSpikeNodeId;

float* d_ExternalTargetSpikeMul;
__device__ float* ExternalTargetSpikeMul;

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

float* d_ExternalSourceSpikeMul;
__device__ float* ExternalSourceSpikeMul;

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

float* h_ExternalSpikeMul;

// Simple kernel for pushing remote spikes in local spike buffers
// Version without spike multiplicity array
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

// Push in a dedicated array the spikes that must be sent externally
__device__ void
PushExternalSpike( uint i_source, float mul )
{
  uint pos = atomicAdd( ExternalSpikeNum, 1 );
  if ( pos >= MaxSpikePerHost )
  {
    printf( "Number of spikes larger than MaxSpikePerHost: %d\n", MaxSpikePerHost );
    *ExternalSpikeNum = MaxSpikePerHost;
    return;
  }
  ExternalSpikeSourceNode[ pos ] = i_source;
  ExternalSpikeMul[ pos ] = mul;
}

// Push in a dedicated array the spikes that must be sent externally
// (version without spike mul)
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
      if ( have_remote_spike_mul )
      {
        float mul = ExternalSpikeMul[ i_spike ];
        // printf("mul: %f\n", mul);
        ExternalTargetSpikeMul[ i_arr ] = mul;
        // printf("ExternalTargetSpikeMul assigned\n");
      }
    }
  }
}


// Count the spikes that must be sent externally for each target host group
__global__ void
countExternalSpikesPerTargetHostGroup()
{
  const uint i_spike = blockIdx.x;
  if ( i_spike < *ExternalSpikeNum )
  {
    // printf("ExternalSpikeNum: %d\ti_spike: %d\n", *ExternalSpikeNum,
    // i_spike);
    uint i_source = ExternalSpikeSourceNode[ i_spike ];
    // printf("i_source: %d\n", i_source);
    uint Nhg = NExternalNodeTargetHostGroup[ i_source ];
    // printf("Nhg: %d\n", Nhg);

    for ( uint ihg = threadIdx.x; ihg < Nhg; ihg += blockDim.x )
    {
      // printf("ihg: %d\n", ihg);
      uint group_local_id = ExternalNodeTargetHostGroupId[ i_source ][ ihg ];
      // printf("group_local_id: %d\n", group_local_id);
      // uint hg_node_id = ExternalHostGroupNodeId[i_source][ihg];
      // printf("hg_node_id: %d\n", hg_node_id);
      // uint pos =
      atomicAdd( &ExternalHostGroupSpikeNum[ group_local_id ], 1 );
      // printf("pos: %d\n", pos);
    }
  }
}

// Organize the spikes that must be sent externally for each target host group
__global__ void
organizeExternalSpikesPerTargetHostGroup()
{
  const uint i_spike = blockIdx.x;
  if ( i_spike < *ExternalSpikeNum )
  {
    // printf("ExternalSpikeNum: %d\ti_spike: %d\n", *ExternalSpikeNum,
    // i_spike);
    uint i_source = ExternalSpikeSourceNode[ i_spike ];
    // printf("i_source: %d\n", i_source);
    uint Nhg = NExternalNodeTargetHostGroup[ i_source ];
    // printf("Nhg: %d\n", Nhg);

    for ( uint ihg = threadIdx.x; ihg < Nhg; ihg += blockDim.x )
    {
      // printf("ihg: %d\n", ihg);
      uint group_local_id = ExternalNodeTargetHostGroupId[ i_source ][ ihg ];
      // printf("group_local_id: %d\n", group_local_id);
      uint hg_node_id = ExternalHostGroupNodeId[ i_source ][ ihg ];
      // printf("hg_node_id: %d\n", hg_node_id);
      uint pos = atomicAdd( &ExternalHostGroupSpikeNum[ group_local_id ], 1 );
      // printf("pos: %d\n", pos);
      uint i_arr = ExternalHostGroupSpikeIdx0[ group_local_id ] + pos;
      ExternalHostGroupSpikeNodeId[ i_arr ] = hg_node_id;
      if ( have_remote_spike_mul )
      {
        float mul = ExternalSpikeMul[ i_spike ];
        // printf("mul: %f\n", mul);
        ExternalHostGroupSpikeMul[ i_arr ] = mul;
        // printf("ExternalHostGroupSpikeMul assigned\n");
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
  gpuErrchk( cudaMemset( d_ExternalHostGroupSpikeNum, 0, conn_->getHostGroup().size() * sizeof( uint ) ) );

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

  uint n_node = GetNLocalNodes(); // number of nodes
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

  h_ExternalHostGroupSpikeNodeId = new uint[ max_remote_spike_num_ ];

  CUDAMALLOCCTRL( "&d_ExternalSpikeNum", &d_ExternalSpikeNum, sizeof( uint ) );
  CUDAMALLOCCTRL( "&d_ExternalSpikeSourceNode", &d_ExternalSpikeSourceNode, max_spike_per_host_ * sizeof( uint ) );

  if ( remote_spike_mul_ )
  {
    h_ExternalSpikeMul = new float[ max_spike_per_host_ ];
    CUDAMALLOCCTRL( "&d_ExternalSpikeMul", &d_ExternalSpikeMul, max_spike_per_host_ * sizeof( float ) );
    CUDAMALLOCCTRL( "&d_ExternalTargetSpikeMul", &d_ExternalTargetSpikeMul, max_remote_spike_num_ * sizeof( float ) );
    CUDAMALLOCCTRL( "&d_ExternalSourceSpikeMul", &d_ExternalSourceSpikeMul, max_remote_spike_num_ * sizeof( float ) );
    CUDAMALLOCCTRL( "&d_ExternalHostGroupSpikeMul", &d_ExternalHostGroupSpikeMul, max_remote_spike_num_ * sizeof( float ) );
  }

  CUDAMALLOCCTRL( "&d_ExternalTargetSpikeNum", &d_ExternalTargetSpikeNum, n_hosts_ * sizeof( uint ) );

  // printf("n_hosts, max_spike_per_host: %d %d\n", n_hosts,
  // max_spike_per_host);

  CUDAMALLOCCTRL(
    "&d_ExternalTargetSpikeNodeId", &d_ExternalTargetSpikeNodeId, max_remote_spike_num_ * sizeof( uint ) );
  CUDAMALLOCCTRL(
    "&d_ExternalHostGroupSpikeNodeId", &d_ExternalTargetSpikeNodeId, max_remote_spike_num_ * sizeof( uint ) );

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

  std::vector< std::vector< int > > &host_group = conn_->getHostGroup();
  uint nhg = host_group.size();
  h_ExternalHostGroupSpikeIdx0 = new uint[ nhg + 1 ];
  h_ExternalHostGroupSpikeNum = new uint[ nhg ];
  CUDAMALLOCCTRL( "&d_ExternalHostGroupSpikeIdx0", &d_ExternalHostGroupSpikeIdx0, ( nhg + 1 ) * sizeof( uint ) );
  CUDAMALLOCCTRL( "&d_ExternalHostGroupSpikeNum", &d_ExternalHostGroupSpikeNum, nhg * sizeof( uint ) );
  
  std::vector< std::unordered_set < int > > &node_target_host_group = conn_->getNodeTargetHostGroup();
  std::vector< std::vector< uint > > &host_group_local_source_node_map = conn_->getHostGroupLocalSourceNodeMap();

  std::vector < uint > n_node_target_host_group(n_node, 0);
  uint ntg_tot = 0;
  for (inode_t i_node = 0; i_node<n_node; i_node++) {
    uint ntg = node_target_host_group[i_node].size();
    n_node_target_host_group[i_node] = ntg;
    ntg_tot += ntg;
  }
  std::vector < uint > node_target_host_group_flat(ntg_tot, 0);
  std::vector < uint > host_group_node_id_flat(ntg_tot, 0);
  std::vector < uint* > hd_ExternalNodeTargetHostGroupId(n_node, nullptr);
  std::vector < uint* > hd_ExternalHostGroupNodeId(n_node, nullptr);
  CUDAMALLOCCTRL("&hd_ExternalNodeTargetHostGroupId[0]", &hd_ExternalNodeTargetHostGroupId[0], ntg_tot*sizeof(uint));
  CUDAMALLOCCTRL("&hd_ExternalHostGroupNodeId[0]", &hd_ExternalHostGroupNodeId[0], ntg_tot*sizeof(uint));
  //uint pos = 0;
  auto node_target_host_group_it = node_target_host_group_flat.begin();
  uint *host_group_node_id_pt = &host_group_node_id_flat[0];
  for (inode_t i_node = 0; i_node<n_node; i_node++) {
    uint ntg = n_node_target_host_group[i_node];
    std::copy(node_target_host_group[i_node].begin(), node_target_host_group[i_node].end(), node_target_host_group_it);
    for (uint group_local_id=0; group_local_id<ntg; group_local_id++) {
      inode_t node_pos = host_group_local_source_node_map[group_local_id][i_node];
      *(host_group_node_id_pt + group_local_id) = node_pos;
    }
    node_target_host_group_it += ntg;
    host_group_node_id_pt += ntg;
    if (i_node < n_node - 1) {
      hd_ExternalNodeTargetHostGroupId[i_node + 1] = hd_ExternalNodeTargetHostGroupId[i_node] + ntg;
      hd_ExternalHostGroupNodeId[i_node + 1] = hd_ExternalHostGroupNodeId[i_node] + ntg; 
    }
  }
  gpuErrchk( cudaMemcpy( hd_ExternalNodeTargetHostGroupId[0], &node_target_host_group_flat[0], ntg_tot * sizeof( uint ),
			 cudaMemcpyHostToDevice ) );
  gpuErrchk( cudaMemcpy( hd_ExternalHostGroupNodeId[0], &host_group_node_id_flat[0], ntg_tot * sizeof( uint ),
			 cudaMemcpyHostToDevice ) );
  CUDAMALLOCCTRL("&d_ExternalNodeTargetHostGroupId", &d_ExternalNodeTargetHostGroupId, n_node*sizeof(uint*));
  gpuErrchk( cudaMemcpy( d_ExternalNodeTargetHostGroupId, &hd_ExternalNodeTargetHostGroupId[0], n_node*sizeof( uint* ),
			 cudaMemcpyHostToDevice ) );
  CUDAMALLOCCTRL("&d_ExternalHostGroupNodeId", &d_ExternalHostGroupNodeId, n_node*sizeof(uint*));
  gpuErrchk( cudaMemcpy( d_ExternalHostGroupNodeId, &hd_ExternalHostGroupNodeId[0], n_node*sizeof( uint* ),
			 cudaMemcpyHostToDevice ) );
  CUDAMALLOCCTRL("&d_NExternalNodeTargetHostGroup", &d_NExternalNodeTargetHostGroup, n_node*sizeof(uint));
  gpuErrchk( cudaMemcpy( d_NExternalNodeTargetHostGroup, &n_node_target_host_group[0], n_node*sizeof(uint),
			 cudaMemcpyHostToDevice ) );
  
  
  if ( remote_spike_mul_ )
  {
    DeviceExternalSpikeInit<<< 1, 1 >>>( n_hosts_,
      max_spike_per_host_,
      d_ExternalSpikeNum,
      d_ExternalSpikeSourceNode,
      d_ExternalSpikeMul,
      d_ExternalTargetSpikeNum,
      d_ExternalTargetSpikeIdx0,
      d_ExternalTargetSpikeNodeId,
      d_ExternalTargetSpikeMul,
      conn_->getDevNTargetHosts(),
      conn_->getDevNodeTargetHosts(),
      conn_->getDevNodeTargetHostIMap(),
      d_NExternalNodeTargetHostGroup,
      d_ExternalNodeTargetHostGroupId,
      d_ExternalHostGroupNodeId,
      d_ExternalHostGroupSpikeNum,
      d_ExternalHostGroupSpikeIdx0,
      d_ExternalHostGroupSpikeNodeId,
      d_ExternalHostGroupSpikeMul );

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
      conn_->getDevNodeTargetHostIMap(),
      d_NExternalNodeTargetHostGroup,
      d_ExternalNodeTargetHostGroupId,
      d_ExternalHostGroupNodeId,
      d_ExternalHostGroupSpikeNum,
      d_ExternalHostGroupSpikeIdx0,
      d_ExternalHostGroupSpikeNodeId );
  }
  ExternalSpikeReset();
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
  float* ext_spike_mul,
  uint* ext_target_spike_num,
  uint* ext_target_spike_idx0,
  uint* ext_target_spike_node_id,
  float* ext_target_spike_mul,
  uint* n_ext_node_target_host,
  uint** ext_node_target_host_id,
  uint** ext_node_id,
  uint* n_ext_node_target_host_group,
  uint** ext_node_target_host_group_id,
  uint** ext_host_group_node_id,
  uint* ext_host_group_spike_num,
  uint* ext_host_group_spike_idx0,
  uint* ext_host_group_spike_node_id,
  float* ext_host_group_spike_mul
 )

{
  NExternalTargetHost = n_hosts;
  MaxSpikePerHost = max_spike_per_host;
  ExternalSpikeNum = ext_spike_num;
  ExternalSpikeSourceNode = ext_spike_source_node;
  ExternalSpikeMul = ext_spike_mul;
  ExternalTargetSpikeNum = ext_target_spike_num;
  ExternalTargetSpikeIdx0 = ext_target_spike_idx0;
  ExternalTargetSpikeNodeId = ext_target_spike_node_id;
  ExternalTargetSpikeMul = ext_target_spike_mul;
  NExternalNodeTargetHost = n_ext_node_target_host;
  ExternalNodeTargetHostId = ext_node_target_host_id;
  ExternalNodeId = ext_node_id;
  *ExternalSpikeNum = 0;
  
  NExternalNodeTargetHostGroup = n_ext_node_target_host_group; 
  ExternalNodeTargetHostGroupId = ext_node_target_host_group_id;
  ExternalHostGroupNodeId = ext_host_group_node_id;
  ExternalHostGroupSpikeNum = ext_host_group_spike_num; 
  ExternalHostGroupSpikeIdx0 = ext_host_group_spike_idx0;
  ExternalHostGroupSpikeNodeId = ext_host_group_spike_node_id;
  ExternalHostGroupSpikeMul = ext_host_group_spike_mul;
}

// initialize external spike array pointers in the GPU
// (version without spike multiplicity)
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
  uint** ext_node_id,
  uint* n_ext_node_target_host_group,
  uint** ext_node_target_host_group_id,
  uint** ext_host_group_node_id,
  uint* ext_host_group_spike_num,
  uint* ext_host_group_spike_idx0,
  uint* ext_host_group_spike_node_id
 )			 

{
  NExternalTargetHost = n_hosts;
  MaxSpikePerHost = max_spike_per_host;
  ExternalSpikeNum = ext_spike_num;
  ExternalSpikeSourceNode = ext_spike_source_node;
  ExternalSpikeMul = nullptr;
  ExternalTargetSpikeNum = ext_target_spike_num;
  ExternalTargetSpikeIdx0 = ext_target_spike_idx0;
  ExternalTargetSpikeNodeId = ext_target_spike_node_id;
  ExternalTargetSpikeMul = nullptr;
  NExternalNodeTargetHost = n_ext_node_target_host;
  ExternalNodeTargetHostId = ext_node_target_host_id;
  ExternalNodeId = ext_node_id;
  *ExternalSpikeNum = 0;
  
  NExternalNodeTargetHostGroup = n_ext_node_target_host_group; 
  ExternalNodeTargetHostGroupId = ext_node_target_host_group_id;
  ExternalHostGroupNodeId = ext_host_group_node_id;
  ExternalHostGroupSpikeNum = ext_host_group_spike_num; 
  ExternalHostGroupSpikeIdx0 = ext_host_group_spike_idx0;
  ExternalHostGroupSpikeNodeId = ext_host_group_spike_node_id;
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

  // probably missing some condition checking if host groups and/or point-to-point MPI communication are actually used
  countExternalSpikesPerTargetHostGroup<<< n_ext_spikes, 1024 >>>();
  CUDASYNC;
  prefix_scan( ( int* ) d_ExternalHostGroupSpikeIdx0, ( int* ) d_ExternalHostGroupSpikeNum, conn_->getHostGroup().size() + 1, true );
  DBGCUDASYNC;
  gpuErrchk( cudaMemset( d_ExternalHostGroupSpikeNum, 0, conn_->getHostGroup().size() * sizeof( uint ) ) );
  organizeExternalSpikesPerTargetHostGroup<<< n_ext_spikes, 1024 >>>();
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
    // convert node map indexes to image node indexes
    MapIndexToImageNodeKernel<<< n_hosts_, 1024 >>>(
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
