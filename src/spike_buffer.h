/*
 *  spike_buffer.h
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





#ifndef SPIKEBUFFER_H
#define SPIKEBUFFER_H
//#include "connect.h"

extern __constant__ bool ExternalSpikeFlag;
extern __device__ int MaxSpikeBufferSize;
extern __device__ int NSpikeBuffer;
extern __device__ int MaxDelayNum;

extern int h_NSpikeBuffer;
extern bool ConnectionSpikeTimeFlag;

extern float *d_LastSpikeHeight; // [NSpikeBuffer];
extern __device__ float *LastSpikeHeight; //

extern long long *d_LastSpikeTimeIdx; // [NSpikeBuffer];
extern __device__ long long *LastSpikeTimeIdx; //

extern long long *d_LastRevSpikeTimeIdx; // [NSpikeBuffer];
extern __device__ long long *LastRevSpikeTimeIdx; //

extern unsigned short *d_ConnectionSpikeTime; // [NConnection];
extern __device__ unsigned short *ConnectionSpikeTime; //


extern int *d_SpikeBufferSize;
extern __device__ int *SpikeBufferSize;
// number of spikes stored in the buffer

extern int *d_SpikeBufferIdx0;
extern __device__ int *SpikeBufferIdx0;
// index of most recent spike stored in the buffer

extern int *d_SpikeBufferTimeIdx;
extern __device__ int *SpikeBufferTimeIdx;
// time index of the spike

extern int *d_SpikeBufferConnIdx;
extern __device__ int *SpikeBufferConnIdx;
// index of the next connection group that will emit this spike

extern float *d_SpikeBufferHeight;
extern __device__ float *SpikeBufferHeight;
// spike height


__device__ void PushSpike(int i_spike_buffer, float height);

__global__ void SpikeBufferUpdate();

__global__ void DeviceSpikeBufferInit(int n_spike_buffers, int max_delay_num,
				int max_spike_buffer_size,
				long long *last_spike_time_idx,
				float *last_spike_height,
				unsigned short *conn_spike_time,      
				int *spike_buffer_size, int *spike_buffer_idx0,
				int *spike_buffer_time,
				int *spike_buffer_conn,
				float *spike_buffer_height,
                                long long *last_rev_spike_time_idx);

int SpikeBufferInit(uint n_spike_buffers, int max_spike_buffer_size);

#endif
