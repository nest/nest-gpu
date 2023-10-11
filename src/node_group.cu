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





#include <config.h>
#include <iostream>

#include "cuda_error.h"
#include "node_group.h"
#include "nestgpu.h"

__constant__ NodeGroupStruct NodeGroupArray[MAX_N_NODE_GROUPS];
__device__ int16_t *NodeGroupMap;

__global__
void NodeGroupMapInit(int16_t *node_group_map)
{
  NodeGroupMap = node_group_map;
}

int NESTGPU::NodeGroupArrayInit()
{
  CUDAMALLOCCTRL("&d_node_group_map_",&d_node_group_map_,
		       node_group_map_.size()*sizeof(int16_t));

  std::vector<NodeGroupStruct> ngs_vect;
  for (unsigned int i=0; i<node_vect_.size(); i++) {
    NodeGroupStruct ngs;
    ngs.node_type_ = node_vect_[i]->node_type_;
    ngs.i_node_0_ = node_vect_[i]->i_node_0_;
    ngs.n_node_ = node_vect_[i]->n_node_;
    ngs.n_port_ = node_vect_[i]->n_port_;
    ngs.n_param_ = node_vect_[i]->n_param_;
    ngs.get_spike_array_ = node_vect_[i]->get_spike_array_;

    ngs.spike_count_ = node_vect_[i]->spike_count_;
    ngs.rec_spike_times_ = node_vect_[i]->rec_spike_times_;
    ngs.n_rec_spike_times_ = node_vect_[i]->n_rec_spike_times_;
    ngs.max_n_rec_spike_times_ = node_vect_[i]->max_n_rec_spike_times_;
    ngs.den_delay_arr_ = node_vect_[i]->den_delay_arr_;
    
    ngs_vect.push_back(ngs);
  }

  //  gpuErrchk( cudaPeekAtLastError() );
  //  gpuErrchk( cudaDeviceSynchronize() );

  //  std::cout << this_host_ << "ngs_vect.data(): " << ngs_vect.data() << "\n";
  //  std::cout << this_host_ << "ngs_vect.size(): " << ngs_vect.size() << "\n";
  //  std::cout << this_host_ << "NodeGroupArray: " << NodeGroupArray << "\n";

  if (ngs_vect.size() > MAX_N_NODE_GROUPS) {
    throw ngpu_exception("Number of neuron groups larger than limit.");
  }
  
  gpuErrchk(cudaMemcpyToSymbolAsync(NodeGroupArray, ngs_vect.data(),
			       ngs_vect.size()*sizeof(NodeGroupStruct)));

  // Memcopy will be synchronized with NodeGroupMapInit kernel
  gpuErrchk(cudaMemcpyAsync(d_node_group_map_, node_group_map_.data(),
		       node_group_map_.size()*sizeof(int16_t),
		       cudaMemcpyHostToDevice));
  // temporary
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  NodeGroupMapInit<<<1, 1>>>(d_node_group_map_);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  return 0;
}

double *NESTGPU::InitGetSpikeArray (int n_node, int n_port)
{
  double *d_get_spike_array = NULL;
  if (n_node*n_port > 0) {
    CUDAMALLOCCTRL("&d_get_spike_array",&d_get_spike_array, n_node*n_port
			 *sizeof(double));
  }
  
  return d_get_spike_array;
}

int NESTGPU::FreeNodeGroupMap()
{
  gpuErrchk(cudaFree(d_node_group_map_));
	    
  return 0;
}
