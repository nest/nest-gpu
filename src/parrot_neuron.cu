/*
 *  parrot_neuron.cu
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
#include <cmath>
#include <iostream>
#include <string>
//#include <stdio.h>

#include "cuda_error.h"
#include "nestgpu.h"
#include "neuron_models.h"
#include "parrot_neuron.h"
#include "spike_buffer.h"
//#include "parrot_neuron_variables.h"

enum {
  i_parrot_neuron_hold_spike_height=0,
  i_parrot_neuron_den_delay,
  N_PARROT_NEURON_SCAL_PARAM
};

const std::string parrot_neuron_scal_param_name[N_PARROT_NEURON_SCAL_PARAM]
= {"hold_spike_height", "den_delay"};

enum {
  i_parrot_neuron_input_spike_height=0,
  i_parrot_neuron_dummy_input,
  i_parrot_neuron_V,
  N_PARROT_NEURON_SCAL_VAR
};

const std::string parrot_neuron_scal_var_name[N_PARROT_NEURON_SCAL_VAR]
= {"input_spike_height", "dummy_input", "V"};


__global__
void parrot_neuron_UpdateKernel(int i_node_0, int n_node, float *var_arr,
				float *param_arr, int n_var, int n_param)
{
  int irel_node = threadIdx.x + blockIdx.x * blockDim.x;
  if (irel_node < n_node) {
    float *input_spike_height_pt = var_arr + irel_node*n_var
      + i_parrot_neuron_input_spike_height;
    float *V_pt = var_arr + irel_node*n_var + i_parrot_neuron_V;
    float *hold_spike_height_pt = param_arr + irel_node*n_param +
      i_parrot_neuron_hold_spike_height;
    int i_node = i_node_0 + irel_node;
    float spike_height = *input_spike_height_pt;
    *V_pt = spike_height;
    if (spike_height != 0.0) {
      if (*hold_spike_height_pt==0.0) {
	spike_height = 1.0;
      }
      *input_spike_height_pt = 0;
      PushSpike(i_node, spike_height);
    }
  }
}


int parrot_neuron::Init(int i_node_0, int n_node, int /*n_port*/,
			int i_group)
{
  BaseNeuron::Init(i_node_0, n_node, 2 /*n_port*/, i_group);
  node_type_ = i_parrot_neuron_model;

  n_scal_var_ = N_PARROT_NEURON_SCAL_VAR;
  n_var_ = n_scal_var_;
  scal_var_name_ = parrot_neuron_scal_var_name;
  
  n_scal_param_ = N_PARROT_NEURON_SCAL_PARAM;
  n_param_ = n_scal_param_;
  scal_param_name_ = parrot_neuron_scal_param_name;

  CUDAMALLOCCTRL("&var_arr_",&var_arr_, n_node_*n_var_*sizeof(float));

  CUDAMALLOCCTRL("&param_arr_",&param_arr_, n_node_*n_param_*sizeof(float));

  SetScalParam(0, n_node, "hold_spike_height", 0.0);

  SetScalParam(0, n_node, "den_delay", 0.0);

  SetScalVar(0, n_node, "input_spike_height", 0.0);

  SetScalVar(0, n_node, "dummy_input", 0.0);

  SetScalVar(0, n_node, "V", 0.0);

  // multiplication factor of input signal is always 1 for all nodes
  float input_weight[] = {1.0, 0.0};
  CUDAMALLOCCTRL("&port_weight_arr_",&port_weight_arr_, 2*sizeof(float));
  gpuErrchk(cudaMemcpy(port_weight_arr_, input_weight,
			 2*sizeof(float), cudaMemcpyHostToDevice));
  port_weight_arr_step_ = 0;
  port_weight_port_step_ = 1;
  
  // input signal is stored in input_spike_height
  port_input_arr_ = GetVarArr() + GetScalVarIdx("input_spike_height");
  port_input_arr_step_ = n_var_;
  port_input_port_step_ = 1;

  den_delay_arr_ =  GetParamArr() + GetScalParamIdx("den_delay");

  return 0;
}

int parrot_neuron::Update(long long /*i_time*/, double /*t1*/)
{
  parrot_neuron_UpdateKernel<<<(n_node_+1023)/1024, 1024>>>
    (i_node_0_, n_node_, var_arr_, param_arr_, n_var_, n_param_);
  //gpuErrchk( cudaPeekAtLastError() );
  //gpuErrchk( cudaDeviceSynchronize() );

  return 0;
}

int parrot_neuron::Free()
{
  CUDAFREECTRL("var_arr_",var_arr_);
  CUDAFREECTRL("param_arr_",param_arr_);	    

  return 0;
}

parrot_neuron::~parrot_neuron()
{
  if (n_node_>0) {
    Free();
  }
}
