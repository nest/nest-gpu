/*
 *  remote_spike.h
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

#ifndef REMOTE_SPIKE_H
#define REMOTE_SPIKE_H

extern __constant__ bool have_remote_spike_mul;

__global__ void PushSpikeFromRemote( uint n_spikes, uint* spike_buffer_id, float* spike_mul );

__global__ void PushSpikeFromRemote( uint n_spikes, uint* spike_buffer_id );


// number of spikes that must be send to each host group
// ExternalHostGroupSpikeNum[ group_local_id ]
extern std::vector<uint> h_ExternalHostGroupSpikeNum;

// index of the first spike of the spike subarray that must be sent to the host group group_local_id
// (the spikes to be sent to all host groups are flattened on a single one-dimensional array)
// ExternalHostGroupSpikeIdx0[ group_local_id ]
extern uint* d_ExternalHostGroupSpikeIdx0;
extern std::vector<uint> h_ExternalHostGroupSpikeIdx0;

// flattened one-dimensional array of spikes that must be sent to all host groups
extern uint* d_ExternalHostGroupSpikeNodeId;
extern std::vector<uint> h_ExternalHostGroupSpikeNodeId;


extern __device__ uint NExternalTargetHost;
extern __device__ uint MaxSpikePerHost;

extern uint* d_ExternalSpikeNum;
extern __device__ uint* ExternalSpikeNum;

extern uint* d_ExternalSpikeSourceNode; // [MaxSpikeNum];
extern __device__ uint* ExternalSpikeSourceNode;

extern float* d_ExternalSpikeMul; // [MaxSpikeNum];
extern __device__ float* ExternalSpikeMul;

extern uint* d_ExternalTargetSpikeNum;
extern __device__ uint* ExternalTargetSpikeNum;

extern uint* d_ExternalTargetSpikeNodeId;
extern __device__ uint* ExternalTargetSpikeNodeId;

extern float* d_ExternalTargetSpikeMul;
extern __device__ float* ExternalTargetSpikeMul;

// extern uint *d_NExternalNodeTargetHost;
extern __device__ uint* NExternalNodeTargetHost;

// extern uint **d_ExternalNodeTargetHostId;
extern __device__ uint** ExternalNodeTargetHostId;

// extern uint **d_ExternalNodeId;
extern __device__ uint** ExternalNodeId;

// extern int **d_ExternalSourceSpikeNum;
// extern __device__ int **ExternalSourceSpikeNum;

extern uint* d_ExternalSourceSpikeNodeId;
extern __device__ uint* ExternalSourceSpikeNodeId;

extern float* d_ExternalSourceSpikeMul;
extern __device__ float* ExternalSourceSpikeMul;

extern uint* d_ExternalTargetSpikeIdx0;
extern __device__ uint* ExternalTargetSpikeIdx0;
extern std::vector<uint> h_ExternalTargetSpikeIdx0;

extern uint* d_ExternalSourceSpikeIdx0;

extern std::vector<uint> h_ExternalTargetSpikeNum;
extern std::vector< std::vector< int > > h_ExternalSourceSpikeNum;
extern std::vector< uint > h_ExternalSourceSpikeIdx0;
extern std::vector< uint > h_ExternalTargetSpikeNodeId;
extern std::vector< std::vector < uint > > h_ExternalSourceSpikeNodeId;

extern std::vector< int > h_ExternalSourceSpikeDispl;

// extern uint *h_ExternalSpikeNodeId;

extern std::vector < float > h_ExternalSpikeMul;

__device__ void PushExternalSpike( uint i_source, float mul );

__device__ void PushExternalSpike( uint i_source );

__global__ void countExternalSpikesPerTargetHost();

__global__ void organizeExternalSpikesPerTargetHost();

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
 );

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
 );


#endif
