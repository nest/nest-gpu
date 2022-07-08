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


#ifndef STDP_H
#define STDP_H
#include <cmath>


namespace stdp_ns
{
  enum ParamIndexes {
    i_tau_plus = 0, i_tau_minus, i_lambda, i_alpha, i_mu_plus, i_mu_minus,
    i_Wmax, // i_den_delay,
    N_PARAM
  };

  const std::string stdp_param_name[N_PARAM] = {
    "tau_plus", "tau_minus", "lambda", "alpha", "mu_plus", "mu_minus", "Wmax"
    //, "den_delay"
  };



  __device__ __forceinline__ void STDPUpdate(float *weight_pt, float Dt,
					     float *param)
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
}


#endif
