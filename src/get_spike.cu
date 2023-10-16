/*
 *  get_spike.cu
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

#include "nestgpu.h"
#include "node_group.h"
#include "send_spike.h"
#include "spike_buffer.h"
#include "cuda_error.h"
#include "connect.h"


// improve using a grid
/*
__global__ void GetSpikes(double *spike_array, int array_size, int n_port,
			  int n_var,
			  float *port_weight_arr,
			  int port_weight_arr_step,
			  int port_weight_port_step,
			  float *port_input_arr,
			  int port_input_arr_step,
			  int port_input_port_step)
{
  int i_array = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_array < array_size*n_port) {
     int i_target = i_array % array_size;
     int port = i_array / array_size;
     int port_input = i_target*port_input_arr_step
       + port_input_port_step*port;
     int port_weight = i_target*port_weight_arr_step
       + port_weight_port_step*port;
     double d_val = (double)port_input_arr[port_input]
       + spike_array[i_array]
       * port_weight_arr[port_weight];

     port_input_arr[port_input] = (float)d_val;
  }
}
*/

__global__ void GetSpikes(double *spike_array, int array_size, int n_port,
			  int n_var,
			  float *port_weight_arr,
			  int port_weight_arr_step,
			  int port_weight_port_step,
			  float *port_input_arr,
			  int port_input_arr_step,
			  int port_input_port_step)
{
  int i_target = blockIdx.x*blockDim.x+threadIdx.x;
  int port = blockIdx.y*blockDim.y+threadIdx.y;

  if (i_target < array_size && port<n_port) {
    int i_array = port*array_size + i_target;
    int port_input = i_target*port_input_arr_step
      + port_input_port_step*port;
    int port_weight = i_target*port_weight_arr_step
      + port_weight_port_step*port;
    double d_val = (double)port_input_arr[port_input]
      + spike_array[i_array]
      * port_weight_arr[port_weight];
    
    port_input_arr[port_input] = (float)d_val;
  }
}


int NESTGPU::ClearGetSpikeArrays()
{
  for (unsigned int i=0; i<node_vect_.size(); i++) {
    BaseNeuron *bn = node_vect_[i];
    if (bn->get_spike_array_ != NULL) {
      gpuErrchk(cudaMemsetAsync(bn->get_spike_array_, 0, bn->n_node_*bn->n_port_
			   *sizeof(double)));
    }
  }
  
  return 0;
}

int NESTGPU::FreeGetSpikeArrays()
{
  for (unsigned int i=0; i<node_vect_.size(); i++) {
    BaseNeuron *bn = node_vect_[i];
    if (bn->get_spike_array_ != NULL) {
      CUDAFREECTRL("bn->get_spike_array_",bn->get_spike_array_);
    }
  }
  
  return 0;
}
