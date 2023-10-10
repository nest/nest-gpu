/*
 *  iaf_psc_alpha.cu
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





// adapted from:
// https://github.com/nest/nest-simulator/blob/master/models/iaf_psc_alpha.cpp

#include <config.h>
#include <cmath>
#include <iostream>
#include "iaf_psc_alpha.h"
#include "propagator_stability.h"
#include "spike_buffer.h"

using namespace iaf_psc_alpha_ns;

extern __constant__ float NESTGPUTimeResolution;
extern __device__ double propagator_31(double, double, double, double);
extern __device__ double propagator_32(double, double, double, double);

#define I_ex var[i_I_ex]
#define I_in var[i_I_in]
#define dI_ex var[i_dI_ex]
#define dI_in var[i_dI_in]
#define V_m_rel var[i_V_m_rel]
#define refractory_step var[i_refractory_step]

#define tau_m param[i_tau_m]
#define C_m param[i_C_m]
#define E_L param[i_E_L]
#define I_e param[i_I_e]
#define Theta_rel param[i_Theta_rel]
#define V_reset_rel param[i_V_reset_rel]
#define tau_ex param[i_tau_ex]
#define tau_in param[i_tau_in]
#define t_ref param[i_t_ref]
#define den_delay param[i_den_delay]

#define P11ex param[i_P11ex]
#define P11in param[i_P11in]
#define P21ex param[i_P21ex]
#define P21in param[i_P21in]
#define P22ex param[i_P22ex]
#define P22in param[i_P22in]
#define P31ex param[i_P31ex]
#define P31in param[i_P31in]
#define P32ex param[i_P32ex]
#define P32in param[i_P32in]
#define P30 param[i_P30]
#define P33 param[i_P33]
#define expm1_tau_m param[i_expm1_tau_m]
#define EPSCInitialValue param[i_EPSCInitialValue]
#define IPSCInitialValue param[i_IPSCInitialValue]


__global__ void iaf_psc_alpha_Calibrate(int n_node, float *param_arr,
				      int n_param, float h)
{
  int i_neuron = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_neuron<n_node) {
    float *param = param_arr + n_param*i_neuron;
    
    P11ex = P22ex = exp( -h / tau_ex );
    P11in = P22in = exp( -h / tau_in );
    P33 = exp( -h / tau_m );
    expm1_tau_m = expm1( -h / tau_m );

    P30 = -tau_m / C_m * expm1( -h / tau_m );
    P21ex = h * P11ex;
    P21in = h * P11in;

    P31ex = (float)propagator_31( tau_ex, tau_m, C_m, h );
    P32ex = (float)propagator_32( tau_ex, tau_m, C_m, h );
    P31in = (float)propagator_31( tau_in, tau_m, C_m, h );
    P32in = (float)propagator_32( tau_in, tau_m, C_m, h );

    EPSCInitialValue = M_E / tau_ex;
    IPSCInitialValue = M_E / tau_in;

  }
}


__global__ void iaf_psc_alpha_Update(int n_node, int i_node_0, float *var_arr,
				   float *param_arr, int n_var, int n_param)
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
      V_m_rel = P30 * I_e + P31ex * dI_ex + P32ex * I_ex
               + P31in * dI_in + P32in * I_in + expm1_tau_m * V_m_rel + V_m_rel;
    }
  
    // alpha shape PSCs
    I_ex = P21ex * dI_ex + P22ex * I_ex;
    dI_ex *= P11ex;

    I_in = P21in * dI_in + P22in * I_in;
    dI_in *= P11in;

    if (V_m_rel >= Theta_rel ) { // threshold crossing
      PushSpike(i_node_0 + i_neuron, 1.0);
      V_m_rel = V_reset_rel;
      refractory_step = (int)round(t_ref/NESTGPUTimeResolution);
    }
  }
}

iaf_psc_alpha::~iaf_psc_alpha()
{
  FreeVarArr();
  FreeParamArr();
}

int iaf_psc_alpha::Init(int i_node_0, int n_node, int /*n_port*/,
			 int i_group, unsigned long long *seed)
{
  BaseNeuron::Init(i_node_0, n_node, 2 /*n_port*/, i_group, seed);
  node_type_ = i_iaf_psc_alpha_model;

  n_scal_var_ = N_SCAL_VAR;
  n_var_ = n_scal_var_;
  n_scal_param_ = N_SCAL_PARAM;
  n_param_ = n_scal_param_;
  
  AllocParamArr();
  AllocVarArr();

  scal_var_name_ = iaf_psc_alpha_scal_var_name;
  scal_param_name_ = iaf_psc_alpha_scal_param_name;

  SetScalParam(0, n_node, "tau_m", 10.0 );           // in ms
  SetScalParam(0, n_node, "C_m", 250.0 );            // in pF
  SetScalParam(0, n_node, "E_L", -70.0 );            // in mV
  SetScalParam(0, n_node, "I_e", 0.0 );              // in pA
  SetScalParam(0, n_node, "Theta_rel", -55.0 - (-70.0) );   // relative to E_L_
  SetScalParam(0, n_node, "V_reset_rel", -70.0 - (-70.0) ); // relative to E_L_
  SetScalParam(0, n_node, "tau_syn_ex", 2.0 );           // in ms
  SetScalParam(0, n_node, "tau_syn_in", 2.0 );           // in ms
  SetScalParam(0, n_node, "t_ref",  2.0 );           // in ms
  SetScalParam(0, n_node, "den_delay", 0.0);         // in ms
  SetScalParam(0, n_node, "P11ex", 0.0);
  SetScalParam(0, n_node, "P11in", 0.0);
  SetScalParam(0, n_node, "P21ex", 0.0);
  SetScalParam(0, n_node, "P21in", 0.0);
  SetScalParam(0, n_node, "P22ex", 0.0);
  SetScalParam(0, n_node, "P22in", 0.0);
  SetScalParam(0, n_node, "P31ex", 0.0);
  SetScalParam(0, n_node, "P31in", 0.0);
  SetScalParam(0, n_node, "P32ex", 0.0);
  SetScalParam(0, n_node, "P32in", 0.0);
  SetScalParam(0, n_node, "P30", 0.0);
  SetScalParam(0, n_node, "P33", 0.0);
  SetScalParam(0, n_node, "EPSCInitialValue", 0.0);
  SetScalParam(0, n_node, "IPSCInitialValue", 0.0);

  SetScalVar(0, n_node, "I_syn_ex", 0.0 );
  SetScalVar(0, n_node, "dI_ex", 0.0 );
  SetScalVar(0, n_node, "I_syn_in", 0.0 );
  SetScalVar(0, n_node, "dI_in", 0.0 );
  SetScalVar(0, n_node, "V_m_rel", -70.0 - (-70.0) ); // in mV, relative to E_L
  SetScalVar(0, n_node, "refractory_step", 0 );
  
  port_weight_arr_ = GetParamArr() + GetScalParamIdx("EPSCInitialValue");
  port_weight_arr_step_ = n_param_;
  port_weight_port_step_ = 1;

  port_input_arr_ = GetVarArr() + GetScalVarIdx("dI_ex");
  port_input_arr_step_ = n_var_;
  port_input_port_step_ = 1;

  den_delay_arr_ =  GetParamArr() + GetScalParamIdx("den_delay");
  
  return 0;
}

int iaf_psc_alpha::Update(long long it, double t1)
{
  // std::cout << "iaf_psc_alpha neuron update\n";
  iaf_psc_alpha_Update<<<(n_node_+1023)/1024, 1024>>>
    (n_node_, i_node_0_, var_arr_, param_arr_, n_var_, n_param_);
  // gpuErrchk( cudaDeviceSynchronize() );
  
  return 0;
}

int iaf_psc_alpha::Free()
{
  FreeVarArr();  
  FreeParamArr();
  
  return 0;
}

int iaf_psc_alpha::Calibrate(double, float time_resolution)
{
  iaf_psc_alpha_Calibrate<<<(n_node_+1023)/1024, 1024>>>
    (n_node_, param_arr_, n_param_, time_resolution);

  return 0;
}
