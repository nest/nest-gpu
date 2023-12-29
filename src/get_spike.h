/*
 *  get_spike.h
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


#ifndef GETSPIKE_H
#define GETSPIKE_H
#include "utilities.h"
#include "send_spike.h"
#include "connect.h"
#include "node_group.h"
#include "spike_buffer.h"

extern __constant__ NodeGroupStruct NodeGroupArray[];
extern __device__ int16_t *NodeGroupMap;
extern __constant__ float NESTGPUTimeResolution;
extern __constant__ long long NESTGPUTimeIdx;

//template<int i_func>
//__device__  __forceinline__ void NestedLoopFunction(int i_spike, int i_syn);

//////////////////////////////////////////////////////////////////////
// This is the function called by the nested loop
// that collects the spikes
template<class ConnKeyT, class ConnStructT>
__device__  __forceinline__ void NestedLoopFunction0(int i_spike, int i_syn)
{
  int i_source = SpikeSourceIdx[i_spike];
  int i_source_conn_group = SpikeConnIdx[i_spike];
  float height = SpikeHeight[i_spike];
  int ig = ConnGroupIdx0[i_source] + i_source_conn_group;

  int64_t i_conn = ConnGroupIConn0[ig] + i_syn;
  uint i_block = (uint)(i_conn / ConnBlockSize);
  int64_t i_block_conn = i_conn % ConnBlockSize;
  //connection_struct conn = ConnectionArray[i_block][i_block_conn];
  //uint target_port_syn = conn.target_port_syn;
  //int i_target = target_port_syn >> MaxPortSynNBits;
  //uint port = (target_port_syn & PortSynMask) >> MaxSynNBits;
  //unsigned char syn_group = target_port_syn & SynMask;
  ConnKeyT &conn_key =
    ((ConnKeyT**)ConnKeyArray)[i_block][i_block_conn];
  ConnStructT &conn_struct =
    ((ConnStructT**)ConnStructArray)[i_block][i_block_conn];
  inode_t i_target = getConnTarget<ConnStructT>(conn_struct);
  int port = getConnPort<ConnKeyT, ConnStructT>(conn_key, conn_struct);
  int syn_group = getConnSyn<ConnKeyT, ConnStructT>(conn_key, conn_struct);
  float weight = conn_struct.weight;
  //printf("ok target: %d\tport: %d\t syn_group: %d\tweight-0.0005: %.7e\n",
  //	 i_target, port, syn_group, weight-0.0005);

  //printf("handles spike %d src %d conn %ld syn %d target %d"
  //	 " port %d weight %f syn_group %d\n",
  //	 i_spike, i_source, i_conn, i_syn, i_target,
  //	 port, weight, syn_group);
  
  /////////////////////////////////////////////////////////////////
  int i_group=NodeGroupMap[i_target];
  int64_t i = (int64_t)port*NodeGroupArray[i_group].n_node_ + i_target
    - NodeGroupArray[i_group].i_node_0_;
  double d_val = (double)(height*weight);

  atomicAddDouble(&NodeGroupArray[i_group].get_spike_array_[i], d_val);
  // TO BE IMPROVED BY CHECKING IF THE SYNAPSE TYPE OF THE GROUP
  // REQUIRES AN UPDATE BASED ON POST-PRE SPIKE TIME DIFFERENCE
  if (syn_group>0) {
    //ConnectionGroupTargetSpikeTime[i_conn*NSpikeBuffer+i_source][i_syn]
    ConnectionSpikeTime[i_conn]
      = (unsigned short)(NESTGPUTimeIdx & 0xffff);
    
    long long Dt_int = NESTGPUTimeIdx - LastRevSpikeTimeIdx[i_target];

    //    printf("spike src %d target %d weight %f syn_group %d "
    //	   "TimeIdx %lld LRST %lld Dt %lld\n",
    //	   i_source, i_target, weight, syn_group,
    //	   NESTGPUTimeIdx, LastRevSpikeTimeIdx[i_target], Dt_int);
    
     if (Dt_int>0 && Dt_int<MAX_SYN_DT) {
       SynapseUpdate(syn_group, &(conn_struct.weight),
		     -NESTGPUTimeResolution*Dt_int);
    }
  }
  ////////////////////////////////////////////////////////////////
}
///////////////


__global__ void GetSpikes(double *spike_array, int array_size, int n_port,
			  int n_var,
			  float *port_weight_arr,
			  int port_weight_arr_step,
			  int port_weight_port_step, //float *y_arr);
			  float *port_input_arr,
			  int port_input_arr_step,
			  int port_input_port_step);

#endif
