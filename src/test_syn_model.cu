/*
 *  test_syn_model.cu
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
#include "test_syn_model.h"

using namespace test_syn_model_ns;

__device__ void TestSynModelUpdate(float *w, float Dt, float *param)
{
  float fact = param[0];
  float offset = param[1];
  *w += offset + fact*Dt;
}

int TestSynModel::Init()
{
  type_ = i_test_syn_model;
  n_param_ = N_PARAM;
  param_name_ = test_syn_model_param_name;
  CUDAMALLOCCTRL("&d_param_arr_",&d_param_arr_, n_param_*sizeof(float));
  SetParam("fact", 0.1);
  SetParam("offset", 0.0);

  return 0;
}
