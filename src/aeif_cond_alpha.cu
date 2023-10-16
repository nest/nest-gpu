/*
 *  aeif_cond_alpha.cu
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
#include "aeif_cond_alpha_kernel.h"
#include "rk5.h"
#include "aeif_cond_alpha.h"

namespace aeif_cond_alpha_ns
{

__device__
void NodeInit(int n_var, int n_param, double x, float *y, float *param,
	      aeif_cond_alpha_rk5 data_struct)
{
  //int array_idx = threadIdx.x + blockIdx.x * blockDim.x;

  V_th = -50.4;
  Delta_T = 2.0;
  g_L = 30.0;
  E_L = -70.6;
  C_m = 281.0;
  a = 4.0;
  b = 80.5;
  tau_w = 144.0;
  I_e = 0.0;
  V_peak = 0.0;
  V_reset = -60.0;
  t_ref = 0.0;
  den_delay = 0.0;
  E_rev_ex = 0.0;
  E_rev_in = -85.0;
  tau_syn_ex = 0.2;
  tau_syn_in = 2.0;

  V_m = E_L;
  w = 0;
  refractory_step = 0;
  g_ex = 0;
  g_in = 0;
  g1_ex = 0;
  g1_in = 0;
}

__device__
void NodeCalibrate(int n_var, int n_param, double x, float *y,
		       float *param, aeif_cond_alpha_rk5 data_struct)
{
  //int array_idx = threadIdx.x + blockIdx.x * blockDim.x;

  refractory_step = 0;
  // use normalization for alpha function
  g0_ex = M_E / tau_syn_ex;
  g0_in = M_E / tau_syn_in;
}

}
			    
__device__
void NodeInit(int n_var, int n_param, double x, float *y,
	     float *param, aeif_cond_alpha_rk5 data_struct)
{
    aeif_cond_alpha_ns::NodeInit(n_var, n_param, x, y, param, data_struct);
}

__device__
void NodeCalibrate(int n_var, int n_param, double x, float *y,
		  float *param, aeif_cond_alpha_rk5 data_struct)

{
    aeif_cond_alpha_ns::NodeCalibrate(n_var, n_param, x, y, param, data_struct);
}

using namespace aeif_cond_alpha_ns;

int aeif_cond_alpha::Init(int i_node_0, int n_node, int n_port,
			 int i_group) {
  BaseNeuron::Init(i_node_0, n_node, n_port, i_group);
  node_type_ = i_aeif_cond_alpha_model;
  n_scal_var_ = N_SCAL_VAR;
  n_var_ = n_scal_var_;
  n_scal_param_ = N_SCAL_PARAM;
  n_param_ = n_scal_param_;
  n_group_param_ = N_GROUP_PARAM;

  group_param_ = new float[N_GROUP_PARAM];
  
  scal_var_name_ = aeif_cond_alpha_scal_var_name;
  scal_param_name_ = aeif_cond_alpha_scal_param_name;
  group_param_name_ = aeif_cond_alpha_group_param_name;
  //rk5_data_struct_.node_type_ = i_aeif_cond_alpha_model;
  rk5_data_struct_.i_node_0_ = i_node_0_;

  SetGroupParam("h_min_rel", 1.0e-3);
  SetGroupParam("h0_rel",  1.0e-2);
  h_ = h0_rel_* 0.1;
  
  rk5_.Init(n_node, n_var_, n_param_, 0.0, h_, rk5_data_struct_);
  var_arr_ = rk5_.GetYArr();
  param_arr_ = rk5_.GetParamArr();

  port_weight_arr_ = GetParamArr() + GetScalParamIdx("g0_ex");
  port_weight_arr_step_ = n_param_;
  port_weight_port_step_ = 1;

  port_input_arr_ = GetVarArr() + GetScalVarIdx("g1_ex");
  port_input_arr_step_ = n_var_;
  port_input_port_step_ = 1;
  den_delay_arr_ =  GetParamArr() + GetScalParamIdx("den_delay");

  return 0;
}

int aeif_cond_alpha::Calibrate(double time_min, float time_resolution)
{
  h_min_ = h_min_rel_* time_resolution;
  h_ = h0_rel_* time_resolution;
  rk5_.Calibrate(time_min, h_, rk5_data_struct_);
  
  return 0;
}

int aeif_cond_alpha::Update(long long it, double t1) {
  rk5_.Update<N_SCAL_VAR, N_SCAL_PARAM>(t1, h_min_, rk5_data_struct_);

  return 0;
}
