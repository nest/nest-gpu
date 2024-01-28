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

#include "conn12b.h"
#include "conn16b.h"
#include "connect.h"
#include "get_spike.h"
#include "spike_buffer.h"
#include "syn_model.h"

extern __constant__ long long NESTGPUTimeIdx;
extern __constant__ float NESTGPUTimeResolution;

extern __device__ unsigned int* RevSpikeNum;
extern __device__ unsigned int* RevSpikeTarget;
extern __device__ int* RevSpikeNConn;

extern int64_t* d_RevConnections; //[i] i=0,..., n_rev_conn - 1;
extern __device__ int64_t* RevConnections;

extern int* d_TargetRevConnectionSize; //[i] i=0,..., n_neuron-1;
extern __device__ int* TargetRevConnectionSize;

extern int64_t** d_TargetRevConnection; //[i][j] j=0,...,RevConnectionSize[i]-1
extern __device__ int64_t** TargetRevConnection;

__global__ void revSpikeReset();

__global__ void revSpikeBufferUpdate( unsigned int n_node );

int revSpikeFree();

int resetConnectionSpikeTimeDown();

int resetConnectionSpikeTimeUp();

//////////////////////////////////////////////////////////////////////
// This is the function called by the nested loop
// that makes use of positive post-pre spike time difference
template < class ConnKeyT, class ConnStructT >
__device__ __forceinline__ void
NestedLoopFunction1( int i_spike, int i_target_rev_conn )
{
  unsigned int target = RevSpikeTarget[ i_spike ];
  int64_t i_conn = TargetRevConnection[ target ][ i_target_rev_conn ];
  uint i_block = ( uint ) ( i_conn / ConnBlockSize );
  int64_t i_block_conn = i_conn % ConnBlockSize;
  // connection_struct conn = ConnectionArray[i_block][i_block_conn];
  // unsigned char syn_group = conn.target_port_syn & SynMask;
  ConnKeyT& conn_key = ( ( ConnKeyT** ) ConnKeyArray )[ i_block ][ i_block_conn ];
  ConnStructT& conn_struct = ( ( ConnStructT** ) ConnStructArray )[ i_block ][ i_block_conn ];
  uint syn_group = getConnSyn< ConnKeyT, ConnStructT >( conn_key, conn_struct );

  // TO BE IMPROVED BY CHECKING IF THE SYNAPSE TYPE OF THE GROUP
  // REQUIRES AN UPDATE BASED ON POST-PRE SPIKE TIME DIFFERENCE
  if ( syn_group > 0 )
  {
    unsigned short spike_time_idx = ConnectionSpikeTime[ i_conn ];
    unsigned short time_idx = ( unsigned short ) ( NESTGPUTimeIdx & 0xffff );
    unsigned short Dt_int = time_idx - spike_time_idx;

    // printf("rev spike target %d i_target_rev_conn %d "
    //	   "i_conn %lld weight %f syn_group %d "
    //	   "TimeIdx %lld CST %d Dt %d\n",
    //	   target, i_target_rev_conn, i_conn, conn.weight, syn_group,
    //	   NESTGPUTimeIdx, spike_time_idx, Dt_int);

    if ( Dt_int < MAX_SYN_DT )
    {
      SynapseUpdate( syn_group, &( conn_struct.weight ), NESTGPUTimeResolution * Dt_int );
    }
  }
}

template < int i_func >
__device__ __forceinline__ void NestedLoopFunction( int i_spike, int i_syn );

//////////////////////////////////////////////////////////////////////
// This is the function called by the nested loop
// that makes use of positive post-pre spike time difference.
// Include more integer template specializations
// for different connection types
template <>
__device__ __forceinline__ void
NestedLoopFunction< 1 >( int i_spike, int i_target_rev_conn )
{
  NestedLoopFunction1< conn12b_key, conn12b_struct >( i_spike, i_target_rev_conn );
}

template <>
__device__ __forceinline__ void
NestedLoopFunction< 3 >( int i_spike, int i_target_rev_conn )
{
  NestedLoopFunction1< conn16b_key, conn16b_struct >( i_spike, i_target_rev_conn );
}

#endif
