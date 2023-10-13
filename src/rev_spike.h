/*
 *  rev_spike.h
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

#ifndef REVSPIKE_H
#define REVSPIKE_H

//#include "connect.h"
#include "spike_buffer.h"
#include "syn_model.h"
#include "get_spike.h"

extern int64_t h_NRevConn;
extern unsigned int *d_RevSpikeNum;
extern unsigned int *d_RevSpikeTarget;
extern int *d_RevSpikeNConn;
extern __device__ unsigned int *RevSpikeTarget;
extern __device__ int64_t **TargetRevConnection;
extern __constant__ long long NESTGPUTimeIdx;
extern __constant__ float NESTGPUTimeResolution;

__global__ void RevSpikeReset();

__global__ void RevSpikeBufferUpdate(unsigned int n_node);

int RevSpikeInit(uint n_spike_buffers);

int RevSpikeFree();

int ResetConnectionSpikeTimeDown();

int ResetConnectionSpikeTimeUp();

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
  int64_t i_conn = TargetRevConnection[target][i_target_rev_conn];
  uint i_block = (uint)(i_conn / ConnBlockSize);
  int64_t i_block_conn = i_conn % ConnBlockSize;
  connection_struct conn = ConnectionArray[i_block][i_block_conn];
  unsigned char syn_group = conn.syn_group;
  
  // TO BE IMPROVED BY CHECKING IF THE SYNAPSE TYPE OF THE GROUP
  // REQUIRES AN UPDATE BASED ON POST-PRE SPIKE TIME DIFFERENCE
  if (syn_group>0) {
    unsigned short spike_time_idx = ConnectionSpikeTime[i_conn];
    unsigned short time_idx = (unsigned short)(NESTGPUTimeIdx & 0xffff);
    unsigned short Dt_int = time_idx - spike_time_idx;

    //printf("rev spike target %d i_target_rev_conn %d "
    //	   "i_conn %lld weight %f syn_group %d "
    //	   "TimeIdx %lld CST %d Dt %d\n",
    //	   target, i_target_rev_conn, i_conn, conn.weight, syn_group,
    //	   NESTGPUTimeIdx, spike_time_idx, Dt_int);
   
    if (Dt_int<MAX_SYN_DT) {
      SynapseUpdate(syn_group,
		    &(ConnectionArray[i_block][i_block_conn].weight),
		    NESTGPUTimeResolution*Dt_int);
    }
  }
}

#endif
