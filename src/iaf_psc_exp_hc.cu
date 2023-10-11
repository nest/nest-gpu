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
#include <cmath>
#include <iostream>
#include "iaf_psc_exp_hc.h"
#include "spike_buffer.h"

using namespace iaf_psc_exp_hc_ns;

extern __constant__ float NESTGPUTimeResolution;

#define I_syn var[i_I_syn]
#define V_m_rel var[i_V_m_rel]
#define refractory_step var[i_refractory_step]
#define I_e param[i_I_e]

#include "iaf_psc_exp_hc_params.h"

__global__ void iaf_psc_exp_hc_Update(int n_node, int i_node_0,
					float *var_arr, float *param_arr,
					int n_var, int n_param)
{
  int i_neuron = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_neuron<n_node) {
    float *var = var_arr + n_var*i_neuron;
    float *param = param_arr + n_param*i_neuron;
    
    if ( refractory_step > 0.0 ) {
      // neuron is absolute refractory
      refractory_step -= 1.0;
    }
    else { // neuron is not refractory, so evolve V
      V_m_rel = V_m_rel * P22 + I_syn * P21 + I_e * P20;
    }
    // exponential decaying PSC
    I_syn *= P11;
    
    if (V_m_rel >= Theta_rel ) { // threshold crossing
      PushSpike(i_node_0 + i_neuron, 1.0);
      V_m_rel = V_reset_rel;
      refractory_step = n_refractory_steps;
    }    
  }
}

iaf_psc_exp_hc::~iaf_psc_exp_hc()
{
  FreeVarArr();
  FreeParamArr();
}

int iaf_psc_exp_hc::Init(int i_node_0, int n_node, int /*n_port*/,
			   int i_group)
{
  BaseNeuron::Init(i_node_0, n_node, 1 /*n_port*/, i_group);
  node_type_ = i_iaf_psc_exp_hc_model;

  n_scal_var_ = N_SCAL_VAR;
  n_var_ = n_scal_var_;
  n_scal_param_ = N_SCAL_PARAM;
  n_param_ = n_scal_param_;
  
  AllocParamArr();
  AllocVarArr();

  scal_var_name_ = iaf_psc_exp_hc_scal_var_name;
  scal_param_name_ = iaf_psc_exp_hc_scal_param_name;

  SetScalParam(0, n_node, "I_e", 0.0 );              // in pA

  SetScalVar(0, n_node, "I_syn", 0.0 );
  SetScalVar(0, n_node, "V_m_rel", 0.0 ); // in mV
  SetScalVar(0, n_node, "refractory_step", 0 );

  // multiplication factor of input signal is always 1 for all nodes
  float input_weight = 1.0;
  CUDAMALLOCCTRL("&port_weight_arr_",&port_weight_arr_, sizeof(float));
  gpuErrchk(cudaMemcpy(port_weight_arr_, &input_weight,
			 sizeof(float), cudaMemcpyHostToDevice));
  port_weight_arr_step_ = 0;
  port_weight_port_step_ = 0;
  
  // input spike signal is stored in I_syn
  port_input_arr_ = GetVarArr() + GetScalVarIdx("I_syn");
  port_input_arr_step_ = n_var_;
  port_input_port_step_ = 0;

  return 0;
}

int iaf_psc_exp_hc::Update(long long it, double t1)
{
  // std::cout << "iaf_psc_exp_hc neuron update\n";
  iaf_psc_exp_hc_Update<<<(n_node_+1023)/1024, 1024>>>
    (n_node_, i_node_0_, var_arr_, param_arr_, n_var_, n_param_);
  //gpuErrchk( cudaDeviceSynchronize() );
  
  return 0;
}

int iaf_psc_exp_hc::Free()
{
  FreeVarArr();  
  FreeParamArr();
  
  return 0;
}

