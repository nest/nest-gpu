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
#include <stdlib.h>

				    //#include "cuda_error_nl.h"
#include "cuda_error.h"
#include "nested_loop.h"

//////////////////////////////////////////////////////////////////////
// declare here the functions called by the nested loop 
__device__ void NestedLoopFunction0(int ix, int iy);
__device__ void NestedLoopFunction1(int ix, int iy);
//////////////////////////////////////////////////////////////////////

__device__ int locate(int val, int *data, int n)
{
  int i_left = 0;
  int i_right = n-1;
  int i = (i_left+i_right)/2;
  while(i_right-i_left>1) {
    if (data[i] > val) i_right = i;
    else if (data[i]<val) i_left = i;
    else break;
    i=(i_left+i_right)/2;
  }

  return i;
}

__global__ void NestedLoopKernel0(int Nx, int *Ny)
{
  const int ix = blockIdx.x;
  if (ix<Nx) {
    const int nyix = Ny[ix];
    for (int iy = threadIdx.x; iy < nyix; iy += blockDim.x){
      NestedLoopFunction0(ix, iy);
    }
  }
}

__global__ void NestedLoopKernel1(int Nx, int *Ny)
{
  const int ix = blockIdx.x;
  if (ix<Nx) {
    const int nyix = Ny[ix];
    for (int iy = threadIdx.x; iy < nyix; iy += blockDim.x){
      NestedLoopFunction1(ix, iy);
    }
  }
}

//////////////////////////////////////////////////////////////////////
int NestedLoop(int Nx, int *d_Ny, int i_func)
{
    switch (i_func) {
    case 0:
      NestedLoopKernel0<<<Nx, 1024>>>
	(Nx, d_Ny);
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
      break;
    case 1:
      NestedLoopKernel1<<<Nx, 1024>>>
	(Nx, d_Ny);
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
      break;
    default:
      throw ngpu_exception("unknown nested loop function");
    }

  return 0;
}

