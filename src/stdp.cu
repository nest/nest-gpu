/*
 *  stdp.cu
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
#include <iostream>
#include "ngpu_exception.h"
#include "cuda_error.h"
#include "stdp.h"

using namespace stdp_ns;

__device__ void STDPUpdate(float *weight_pt, float Dt, float *param)
{
  //printf("Dt: %f\n", Dt);
  double tau_plus = param[i_tau_plus];
  double tau_minus = param[i_tau_minus];
  double lambda = param[i_lambda];
  double alpha = param[i_alpha];
  double mu_plus = param[i_mu_plus];
  double mu_minus = param[i_mu_minus];
  double Wmax = param[i_Wmax];
  //double den_delay = param[i_den_delay];

  double w = *weight_pt;
  double w1;
  //Dt += den_delay;
  if (Dt>=0) {
    double fact = lambda*exp(-(double)Dt/tau_plus);
    w1 = w + fact*Wmax*pow(1.0 - w/Wmax, mu_plus);
  }
  else {
    double fact = -alpha*lambda*exp((double)Dt/tau_minus);
    w1 = w + fact*Wmax*pow(w/Wmax, mu_minus);
  }

  w1 = w1 >0.0 ? w1 : 0.0;
  w1 = w1 < Wmax ? w1 : Wmax;
  *weight_pt = (float)w1;
}

int STDP::Init()
{
  type_ = i_stdp_model;
  n_param_ = N_PARAM;
  param_name_ = stdp_param_name;
  gpuErrchk(cudaMalloc(&d_param_arr_, n_param_*sizeof(float)));
  SetParam("tau_plus", 20.0);
  SetParam("tau_minus", 20.0);
  SetParam("lambda", 1.0e-4);
  SetParam("alpha", 1.0);
  SetParam("mu_plus", 1.0);
  SetParam("mu_minus", 1.0);
  SetParam("Wmax", 100.0);
  //SetParam("den_delay", 0.0);

  return 0;
}
