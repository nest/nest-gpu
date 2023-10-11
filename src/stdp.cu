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
#include <stdio.h>
#include <iostream>
#include "ngpu_exception.h"
#include "cuda_error.h"
#include "stdp.h"
#include "syn_model.h"

using namespace stdp_ns;

int STDP::Init()
{
  type_ = i_stdp_model;
  n_param_ = N_PARAM;
  param_name_ = stdp_param_name;
  CUDAMALLOCCTRL("&d_param_arr_",&d_param_arr_, n_param_*sizeof(float));
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
