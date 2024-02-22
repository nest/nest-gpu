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

// extern uint *d_ExternalSourceSpikeNum;
// extern __device__ uint *ExternalSourceSpikeNum;

extern uint* d_ExternalSourceSpikeNodeId;
extern __device__ uint* ExternalSourceSpikeNodeId;

extern float* d_ExternalSourceSpikeMul;
extern __device__ float* ExternalSourceSpikeMul;

extern uint* d_ExternalTargetSpikeIdx0;
extern __device__ uint* ExternalTargetSpikeIdx0;
extern uint* h_ExternalTargetSpikeIdx0;

extern uint* d_ExternalSourceSpikeIdx0;

extern uint* h_ExternalTargetSpikeNum;
extern uint* h_ExternalSourceSpikeNum;
extern uint* h_ExternalSourceSpikeIdx0;
extern uint* h_ExternalTargetSpikeNodeId;
extern uint* h_ExternalSourceSpikeNodeId;

// extern uint *h_ExternalSpikeNodeId;

extern float* h_ExternalSpikeMul;

__device__ void PushExternalSpike( uint i_source, float mul );

__device__ void PushExternalSpike( uint i_source );

__global__ void countExternalSpikesPerTargetHost();

__global__ void organizeExternalSpikesPerTargetHost();

__global__ void DeviceExternalSpikeInit( uint n_hosts,
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
  uint** ext_node_id );

__global__ void DeviceExternalSpikeInit( uint n_hosts,
  uint max_spike_per_host,
  uint* ext_spike_num,
  uint* ext_spike_source_node,
  uint* ext_target_spike_num,
  uint* ext_target_spike_idx0,
  uint* ext_target_spike_node_id,
  uint* n_ext_node_target_host,
  uint** ext_node_target_host_id,
  uint** ext_node_id );

#endif
