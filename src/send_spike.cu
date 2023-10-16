/*
 *  send_spike.cu
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





#include <config.h>
#include <stdio.h>
#include "send_spike.h"
#include "cuda_error.h"

int *d_SpikeNum;
int *d_SpikeSourceIdx;
int *d_SpikeConnIdx;
float *d_SpikeHeight;
int *d_SpikeTargetNum;

__device__ int MaxSpikeNum;
__device__ int *SpikeNum;
__device__ int *SpikeSourceIdx;
__device__ int *SpikeConnIdx;
__device__ float *SpikeHeight;
__device__ int *SpikeTargetNum;

__device__ void SendSpike(int i_source, int i_conn, float height,
			  int target_num)
{
  int pos = atomicAdd(SpikeNum, 1);
  if (pos>=MaxSpikeNum) {
    printf("Number of spikes larger than MaxSpikeNum: %d\n", MaxSpikeNum);
    *SpikeNum = MaxSpikeNum;
    return;
  }
  SpikeSourceIdx[pos] = i_source;
  SpikeConnIdx[pos] = i_conn;
  SpikeHeight[pos] = height;
  SpikeTargetNum[pos] = target_num;
}

__global__ void DeviceSpikeInit(int *spike_num, int *spike_source_idx,
				int *spike_conn_idx, float *spike_height,
				int *spike_target_num,
				int max_spike_num)
{
  SpikeNum = spike_num;
  SpikeSourceIdx = spike_source_idx;
  SpikeConnIdx = spike_conn_idx;
  SpikeHeight = spike_height;
  SpikeTargetNum = spike_target_num;
  MaxSpikeNum = max_spike_num;
  *SpikeNum = 0;
}


void SpikeInit(int max_spike_num)
{
  //h_SpikeTargetNum = new int[PrefixScan::AllocSize];

  CUDAMALLOCCTRL("&d_SpikeNum",&d_SpikeNum, sizeof(int));
  CUDAMALLOCCTRL("&d_SpikeSourceIdx",&d_SpikeSourceIdx, max_spike_num*sizeof(int));
  CUDAMALLOCCTRL("&d_SpikeConnIdx",&d_SpikeConnIdx, max_spike_num*sizeof(int));
  CUDAMALLOCCTRL("&d_SpikeHeight",&d_SpikeHeight, max_spike_num*sizeof(float));
  CUDAMALLOCCTRL("&d_SpikeTargetNum",&d_SpikeTargetNum, max_spike_num*sizeof(int));
  //printf("here: SpikeTargetNum size: %d", max_spike_num);
  DeviceSpikeInit<<<1,1>>>(d_SpikeNum, d_SpikeSourceIdx, d_SpikeConnIdx,
			   d_SpikeHeight, d_SpikeTargetNum, max_spike_num);
  gpuErrchk( cudaPeekAtLastError() );
}

__global__ void SpikeReset()
{
  *SpikeNum = 0;
}
