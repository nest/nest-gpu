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





#ifndef SPIKEBUFFERH
#define SPIKEBUFFERH
#include "connect.h"

extern __device__ int MaxSpikeBufferSize;
extern __device__ int NSpikeBuffer;
extern __device__ int MaxDelayNum;

extern bool ConnectionSpikeTimeFlag;

extern float *d_LastSpikeHeight; // [NSpikeBuffer];
extern __device__ float *LastSpikeHeight; //

extern long long *d_LastSpikeTimeIdx; // [NSpikeBuffer];
extern __device__ long long *LastSpikeTimeIdx; //

extern long long *d_LastRevSpikeTimeIdx; // [NSpikeBuffer];
extern __device__ long long *LastRevSpikeTimeIdx; //

extern float *d_ConnectionWeight; // [NConnection];
extern __device__ float *ConnectionWeight; //

extern unsigned char *d_ConnectionSynGroup; // [NConnection];
extern __device__ unsigned char *ConnectionSynGroup; //

extern unsigned short *d_ConnectionSpikeTime; // [NConnection];
extern __device__ unsigned short *ConnectionSpikeTime; //

extern int *d_ConnectionGroupSize;
extern __device__ int *ConnectionGroupSize;
// Output connections from the source node are organized in groups
// All connection of a group have the same delay

extern int *d_ConnectionGroupDelay;
extern __device__ int *ConnectionGroupDelay;
// delay associated to all connections of this group

extern int *d_ConnectionGroupTargetSize;
extern __device__ int *ConnectionGroupTargetSize;
// number of output connections in the group i_delay

extern unsigned int **d_ConnectionGroupTargetNode;
extern __device__ unsigned int **ConnectionGroupTargetNode;
// is a pointer to an integer array of size ConnectionGroupTargetSize
// that contains the indexes of the target nodes

extern unsigned char **d_ConnectionGroupTargetSynGroup;
extern __device__ unsigned char **ConnectionGroupTargetSynGroup;
// Connection target synapse group

extern float **h_ConnectionGroupTargetWeight;
extern float **d_ConnectionGroupTargetWeight;
extern __device__ float **ConnectionGroupTargetWeight;
// Connection weight

extern unsigned short **d_ConnectionGroupTargetSpikeTime;
extern __device__ unsigned short **ConnectionGroupTargetSpikeTime;
// Connection last spike time index

//////////////////////////////////////////////////////////////////////

extern int *d_SpikeBufferSize;
extern __device__ int *SpikeBufferSize;
// number of spikes stored in the buffer

extern int *d_SpikeBufferTimeIdx;
extern __device__ int *SpikeBufferTimeIdx;
// time index of the spike

extern int *d_SpikeBufferConnIdx;
extern __device__ int *SpikeBufferConnIdx;
// index of the next connection group that will emit this spike

extern float *d_SpikeBufferHeight;
extern __device__ float *SpikeBufferHeight;
// spike height


extern unsigned int *d_RevConnections; //[i] i=0,..., n_rev_conn - 1;
extern __device__ unsigned int *RevConnections;

extern int *d_TargetRevConnectionSize; //[i] i=0,..., n_neuron-1;
extern __device__ int *TargetRevConnectionSize;

extern unsigned int **d_TargetRevConnection; //[i][j] j<=RevConnectionSize[i]-1
extern __device__ unsigned int **TargetRevConnection;


__device__ void PushSpike(int i_spike_buffer, float height);

__global__ void SpikeBufferUpdate();

__global__ void DeviceSpikeBufferInit(int n_spike_buffers, int max_delay_num,
				int max_spike_buffer_size,
				long long *last_spike_time_idx,
				float *last_spike_height,
				float *conn_weight,
				unsigned char *conn_syn_group,
				unsigned short *conn_spike_time,      
				int *conn_group_size, int *conn_group_delay,
				int *conn_group_target_size,
				unsigned int **conn_group_target_node,
				unsigned char **conn_group_target_syn_group,
				float **conn_group_target_weight,
				unsigned short **conn_group_target_spike_time,
				int *spike_buffer_size, int *spike_buffer_time,
				int *spike_buffer_conn,
				float *spike_buffer_height,
				unsigned int *rev_conn,
				int *target_rev_conn_size,
				unsigned int **target_rev_conn,
				long long *last_rev_spike_time_idx);

int SpikeBufferInit(NetConnection *net_connection, int max_spike_buffer_size);

#endif
