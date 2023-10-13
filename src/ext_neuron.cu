/*
 *  ext_neuron.cu
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
#include "ext_neuron.h"

using namespace ext_neuron_ns;

__global__ void UpdateExtNeuron(float *port_input_pt, float *port_value_pt,
				int n_node, int n_var, int n_port_var,
				int n_port)
{
  int i_thread = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_thread<n_node*n_port) {
    int i_port = i_thread%n_port;
    int i_node = i_thread/n_port;
    float *pip = port_input_pt + i_node*n_var + n_port_var*i_port;
    //printf("port %d node %d pip %f\n", i_port, i_node, *pip);
    port_value_pt[i_node*n_var + n_port_var*i_port]
      = *pip;
    *pip = 0.0;    
  }
}

ext_neuron::~ext_neuron()
{
  FreeVarArr();
  FreeParamArr();
}

int ext_neuron::Init(int i_node_0, int n_node, int n_port,
			 int i_group) {
  BaseNeuron::Init(i_node_0, n_node, n_port, i_group);
  node_type_ = i_ext_neuron_model;
  ext_neuron_flag_ = true;
  n_scal_var_ = N_SCAL_VAR;
  n_port_var_ = N_PORT_VAR;
  n_scal_param_ = N_SCAL_PARAM;
  n_port_param_ = N_PORT_PARAM;

  n_var_ = n_scal_var_ + n_port_var_*n_port;
  n_param_ = n_scal_param_ + n_port_param_*n_port;
  AllocParamArr();
  AllocVarArr();

  scal_var_name_ = NULL; //ext_neuron_scal_var_name;
  port_var_name_= ext_neuron_port_var_name;
  scal_param_name_ = ext_neuron_scal_param_name;
  port_param_name_ = ext_neuron_port_param_name;

  port_weight_arr_ = GetParamArr() + n_scal_param_
    + GetPortParamIdx("port_weight");
  port_weight_arr_step_ = n_param_;
  port_weight_port_step_ = n_port_param_;

  port_input_arr_ = GetVarArr() + n_scal_var_
    + GetPortVarIdx("port_input");
  port_input_arr_step_ = n_var_;
  port_input_port_step_ = n_port_var_;
  den_delay_arr_ =  GetParamArr() + GetScalParamIdx("den_delay");

  SetScalParam(0, n_node, "den_delay", 0.0);

  for (int i=0; i<n_port; i++) {
    port_weight_vect_.push_back(1.0);
    port_input_vect_.push_back(0.0);
  }
  SetPortParam(0, n_node, "port_weight", port_weight_vect_.data(), n_port);
  SetPortVar(0, n_node, "port_input", port_input_vect_.data(), n_port);
  
  return 0;
}

int ext_neuron::Update(long long it, double t1) {
  // std::cout << "Ext neuron update\n";
  float *port_input_pt =  GetVarPt(0, "port_input", 0);
  float *port_value_pt =  GetVarPt(0, "port_value", 0);
  
  UpdateExtNeuron<<<(n_node_*n_port_+1023)/1024, 1024>>>
    (port_input_pt, port_value_pt, n_node_, n_var_, n_port_var_, n_port_);
  //gpuErrchk( cudaDeviceSynchronize() );
  
  return 0;
}

int ext_neuron::Free()
{
  FreeVarArr();  
  FreeParamArr();
  
  return 0;
}

float *ext_neuron::GetExtNeuronInputSpikes(int *n_node, int *n_port)
{
  if ((int)ext_neuron_input_spikes_.size()<n_node_*n_port_) {
    ext_neuron_input_spikes_.resize(n_node_*n_port_, 0.0);
  }
  *n_node = n_node_;
  *n_port = n_port_;
  float *var_arr = GetPortVar(0, n_node_, "port_value");
  ext_neuron_input_spikes_.assign(var_arr, var_arr+n_node_*n_port_);
  free(var_arr);
  
  return ext_neuron_input_spikes_.data();
}
