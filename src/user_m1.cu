/*
 *  user_m1.cu
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
#include "user_m1_kernel.h"
#include "rk5.h"
#include "user_m1.h"

namespace user_m1_ns
{

__device__
void NodeInit(int n_var, int n_param, double x, float *y, float *param,
	      user_m1_rk5 data_struct)
{
  //int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  int n_port = (n_var-N_SCAL_VAR)/N_PORT_VAR;

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

  V_m = E_L;
  w = 0;
  refractory_step = 0;
  for (int i = 0; i<n_port; i++) {
    g(i) = 0;
    g1(i) = 0;
    E_rev(i) = 0.0;
    tau_decay(i) = 20.0;
    tau_rise(i) = 2.0;
  }
}

__device__
void NodeCalibrate(int n_var, int n_param, double x, float *y,
		       float *param, user_m1_rk5 data_struct)
{
  //int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  int n_port = (n_var-N_SCAL_VAR)/N_PORT_VAR;

  refractory_step = 0;
  for (int i = 0; i<n_port; i++) {
    // denominator is computed here to check that it is != 0
    float denom1 = tau_decay(i) - tau_rise(i);
    float denom2 = 0;
    if (denom1 != 0) {
      // peak time
      float t_p = tau_decay(i)*tau_rise(i)
	*log(tau_decay(i)/tau_rise(i)) / denom1;
      // another denominator is computed here to check that it is != 0
      denom2 = exp(-t_p / tau_decay(i))
	- exp(-t_p / tau_rise(i));
    }
    if (denom2 == 0) { // if rise time == decay time use alpha function
      // use normalization for alpha function in this case
      g0(i) = M_E / tau_decay(i);
    }
    else { // if rise time != decay time use beta function
      g0(i) // normalization factor for conductance
	= ( 1. / tau_rise(i) - 1. / tau_decay(i) ) / denom2;
    }
  }
}

}

__device__
void NodeInit(int n_var, int n_param, double x, float *y,
	     float *param, user_m1_rk5 data_struct)
{
    user_m1_ns::NodeInit(n_var, n_param, x, y, param, data_struct);
}

__device__
void NodeCalibrate(int n_var, int n_param, double x, float *y,
		  float *param, user_m1_rk5 data_struct)

{
    user_m1_ns::NodeCalibrate(n_var, n_param, x, y, param, data_struct);
}

using namespace user_m1_ns;

int user_m1::Init(int i_node_0, int n_node, int n_port,
			 int i_group, unsigned long long *seed) {
  BaseNeuron::Init(i_node_0, n_node, n_port, i_group, seed);
  node_type_ = i_user_m1_model;
  n_scal_var_ = N_SCAL_VAR;
  n_port_var_ = N_PORT_VAR;
  n_scal_param_ = N_SCAL_PARAM;
  n_port_param_ = N_PORT_PARAM;
  n_group_param_ = N_GROUP_PARAM;

  n_var_ = n_scal_var_ + n_port_var_*n_port;
  n_param_ = n_scal_param_ + n_port_param_*n_port;

  group_param_ = new float[N_GROUP_PARAM];

  scal_var_name_ = user_m1_scal_var_name;
  port_var_name_= user_m1_port_var_name;
  scal_param_name_ = user_m1_scal_param_name;
  port_param_name_ = user_m1_port_param_name;
  group_param_name_ = user_m1_group_param_name;
  //rk5_data_struct_.node_type_ = i_user_m1_model;
  rk5_data_struct_.i_node_0_ = i_node_0_;

  SetGroupParam("h_min_rel", 1.0e-3);
  SetGroupParam("h0_rel",  1.0e-2);
  h_ = h0_rel_* 0.1;

  rk5_.Init(n_node, n_var_, n_param_, 0.0, h_, rk5_data_struct_);
  var_arr_ = rk5_.GetYArr();
  param_arr_ = rk5_.GetParamArr();

  port_weight_arr_ = GetParamArr() + n_scal_param_
    + GetPortParamIdx("g0");
  port_weight_arr_step_ = n_param_;
  port_weight_port_step_ = n_port_param_;

  port_input_arr_ = GetVarArr() + n_scal_var_
    + GetPortVarIdx("g1");
  port_input_arr_step_ = n_var_;
  port_input_port_step_ = n_port_var_;
  den_delay_arr_ =  GetParamArr() + GetScalParamIdx("den_delay");

  return 0;
}

int user_m1::Calibrate(double time_min, float time_resolution)
{
  h_min_ = h_min_rel_* time_resolution;
  h_ = h0_rel_* time_resolution;
  rk5_.Calibrate(time_min, h_, rk5_data_struct_);

  return 0;
}

template <>
int user_m1::UpdateNR<0>(long long it, double t1)
{
  return 0;
}

int user_m1::Update(long long it, double t1) {
  UpdateNR<MAX_PORT_NUM>(it, t1);

  return 0;
}
