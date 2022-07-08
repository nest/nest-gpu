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

#ifndef REVSPIKE_H
#define REVSPIKE_H

#include "connect.h"
#include "spike_buffer.h"
#include "syn_model.h"
#include "get_spike.h"

extern unsigned int *d_RevSpikeNum;
extern unsigned int *d_RevSpikeTarget;
extern int *d_RevSpikeNConn;
extern __device__ unsigned int *RevSpikeTarget;
extern __device__ unsigned int **TargetRevConnection;
extern __constant__ long long NESTGPUTimeIdx;
extern __constant__ float NESTGPUTimeResolution;

__global__ void RevSpikeReset();

__global__ void RevSpikeBufferUpdate(unsigned int n_node);

int RevSpikeInit(NetConnection *net_connection);

int RevSpikeFree();

int ResetConnectionSpikeTimeDown(NetConnection *net_connection);

int ResetConnectionSpikeTimeUp(NetConnection *net_connection);

template<int i_func>
__device__  __forceinline__ void NestedLoopFunction(int i_spike, int i_syn);

//////////////////////////////////////////////////////////////////////
// This is the function called by the nested loop
// that makes use of positive post-pre spike time difference
template<>
__device__ __forceinline__ void NestedLoopFunction<1>
(int i_spike, int i_target_rev_conn)
{
  unsigned int target = RevSpikeTarget[i_spike];
  unsigned int i_conn = TargetRevConnection[target][i_target_rev_conn];
  unsigned char syn_group = ConnectionSynGroup[i_conn];
  if (syn_group>0) {
    float *weight = &ConnectionWeight[i_conn];
    unsigned short spike_time_idx = ConnectionSpikeTime[i_conn];
    unsigned short time_idx = (unsigned short)(NESTGPUTimeIdx & 0xffff);
    unsigned short Dt_int = time_idx - spike_time_idx;
    if (Dt_int<MAX_SYN_DT) {
      SynapseUpdate(syn_group, weight, NESTGPUTimeResolution*Dt_int);
    }
  }
}

#endif
