/*
 *  spike_detector.cu
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
#include "spike_detector.h"
				    //#include "spike_buffer.h"
//#include "parrot_neuron_variables.h"

enum {
  i_spike_detector_hold_spike_height=0,
  N_SPIKE_DETECTOR_SCAL_PARAM
};

const std::string spike_detector_scal_param_name[N_SPIKE_DETECTOR_SCAL_PARAM]
= {"hold_spike_height"};

enum {
  i_spike_detector_input_spike_height=0,
  i_spike_detector_spike_height,
  N_SPIKE_DETECTOR_SCAL_VAR
};

const std::string spike_detector_scal_var_name[N_SPIKE_DETECTOR_SCAL_VAR]
= {"input_spike_height", "spike_height"};


__global__
void spike_detector_UpdateKernel(int i_node_0, int n_node, float *var_arr,
				float *param_arr, int n_var, int n_param)
{
  int irel_node = threadIdx.x + blockIdx.x * blockDim.x;
  if (irel_node < n_node) {
    float *input_spike_height_pt = var_arr + irel_node*n_var
      + i_spike_detector_input_spike_height;
    float *spike_height_pt = var_arr + irel_node*n_var
      + i_spike_detector_spike_height;
    float *hold_spike_height_pt = param_arr + irel_node*n_param +
      i_spike_detector_hold_spike_height;
    //int i_node = i_node_0 + irel_node;
    float spike_height = *input_spike_height_pt;
    if (spike_height != 0.0) {
      if (*hold_spike_height_pt==0.0) {
	spike_height = 1.0;
      }
      *input_spike_height_pt = 0;
    }
    *spike_height_pt = spike_height;
  }
}


int spike_detector::Init(int i_node_0, int n_node, int /*n_port*/,
			int i_group)
{
  BaseNeuron::Init(i_node_0, n_node, 1 /*n_port*/, i_group);
  node_type_ = i_spike_detector_model;

  n_scal_var_ = N_SPIKE_DETECTOR_SCAL_VAR;
  n_var_ = n_scal_var_;
  scal_var_name_ = spike_detector_scal_var_name;
  
  n_scal_param_ = N_SPIKE_DETECTOR_SCAL_PARAM;
  n_param_ = n_scal_param_;
  scal_param_name_ = spike_detector_scal_param_name;

  CUDAMALLOCCTRL("&var_arr_",&var_arr_, n_node_*n_var_*sizeof(float));

  CUDAMALLOCCTRL("&param_arr_",&param_arr_, n_node_*n_param_*sizeof(float));

  SetScalParam(0, n_node, "hold_spike_height", 1.0);

  SetScalVar(0, n_node, "input_spike_height", 0.0);

  SetScalVar(0, n_node, "spike_height", 0.0);

  // multiplication factor of input signal is always 1 for all nodes
  float input_weight = 1.0;
  CUDAMALLOCCTRL("&port_weight_arr_",&port_weight_arr_, sizeof(float));
  gpuErrchk(cudaMemcpy(port_weight_arr_, &input_weight,
			 sizeof(float), cudaMemcpyHostToDevice));
  port_weight_arr_step_ = 0;
  port_weight_port_step_ = 0;
  
  // input signal is stored in input_spike_height
  port_input_arr_ = GetVarArr() + GetScalVarIdx("input_spike_height");
  port_input_arr_step_ = n_var_;
  port_input_port_step_ = n_port_var_;
  
  return 0;
}

int spike_detector::Update(long long /*i_time*/, double /*t1*/)
{
  spike_detector_UpdateKernel<<<(n_node_+1023)/1024, 1024>>>
    (i_node_0_, n_node_, var_arr_, param_arr_, n_var_, n_param_);
  //gpuErrchk( cudaPeekAtLastError() );
  //gpuErrchk( cudaDeviceSynchronize() );

  return 0;
}

int spike_detector::Free()
{
  CUDAFREECTRL("var_arr_",var_arr_);
  CUDAFREECTRL("param_arr_",param_arr_);	    

  return 0;
}

spike_detector::~spike_detector()
{
  if (n_node_>0) {
    Free();
  }
}
