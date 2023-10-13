/*
 *  rk5.cu
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
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <curand.h>
#include <curand_kernel.h>
#include "rk5.h"

__constant__ float c2 = 0.2;
__constant__ float c3 = 0.3;
__constant__ float c4 = 0.6;
__constant__ float c5 = 1.0;
__constant__ float c6 = 0.875;
__constant__ float a21 = 0.2;
__constant__ float a31 = 3.0/40.0;
__constant__ float a32 = 9.0/40.0;
__constant__ float a41 = 0.3;
__constant__ float a42 = -0.9;
__constant__ float a43 = 1.2;
__constant__ float a51 = -11.0/54.0;
__constant__ float a52 = 2.5;
__constant__ float a53 = -70.0/27.0;
__constant__ float a54 = 35.0/27.0;
__constant__ float a61 = 1631.0/55296.0;
__constant__ float a62 = 175.0/512.0;
__constant__ float a63 = 575.0/13824.0;
__constant__ float a64 = 44275.0/110592.0;
__constant__ float a65 = 253.0/4096.0;

__constant__ float a71 = 37.0/378.0;
__constant__ float a73 = 250.0/621.0;
__constant__ float a74 = 125.0/594.0;
__constant__ float a76 = 512.0/1771.0;

__constant__ float e1 = 37.0/378.0 - 2825.0/27648.0;
__constant__ float e3 = 250.0/621.0 - 18575.0/48384.0;
__constant__ float e4 = 125.0/594.0 - 13525.0/55296.0;
__constant__ float e5 = -277.00/14336.0;
__constant__ float e6 = 512.0/1771.0 - 0.25;

__constant__ float eps = 1.0e-6;
__constant__ float coeff = 0.9;
__constant__ float exp_inc = -0.2;
__constant__ float exp_dec = -0.25;
__constant__ float err_min = 1.889568e-4; //(5/coeff)^(1/exp_inc)
__constant__ float scal_min = 1.0e-1;

__global__ void SetFloatArray(float *arr, int n_elem, int step, float val)
{
  int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (array_idx<n_elem) {
    arr[array_idx*step] = val;
  }
}
