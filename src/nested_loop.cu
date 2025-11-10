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

//TMP
#include "getRealTime.h"
//

//////////////////////////////////////////////////////////////////////
// declare here the functions called by the nested loop 
__device__ void NestedLoopFunction0(int ix, int iy);
__device__ void NestedLoopFunction1(int ix, int iy);
//////////////////////////////////////////////////////////////////////

namespace NestedLoop
{
  PrefixScan prefix_scan_;
  int *d_Ny_cumul_sum_;   
}

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

__global__ void CumulSumNestedLoopKernel0(int Nx, int *Ny_cumul_sum,
					 int Ny_sum)
{
  int blockId   = blockIdx.y * gridDim.x + blockIdx.x;
  int array_idx = blockId * blockDim.x + threadIdx.x;
  if (array_idx<Ny_sum) {
    int ix = locate(array_idx, Ny_cumul_sum, Nx + 1);
    int iy = (int)(array_idx - Ny_cumul_sum[ix]);
    NestedLoopFunction0(ix, iy);
  }
}

__global__ void CumulSumNestedLoopKernel1(int Nx, int *Ny_cumul_sum,
					 int Ny_sum)
{
  int blockId   = blockIdx.y * gridDim.x + blockIdx.x;
  int array_idx = blockId * blockDim.x + threadIdx.x;
  if (array_idx<Ny_sum) {
    int ix = locate(array_idx, Ny_cumul_sum, Nx + 1);
    int iy = (int)(array_idx - Ny_cumul_sum[ix]);
    NestedLoopFunction1(ix, iy);
  }
}

//////////////////////////////////////////////////////////////////////
int NestedLoop::Init()
{
  //prefix_scan_.Init();
  gpuErrchk(cudaMalloc(&d_Ny_cumul_sum_,
			  PrefixScan::AllocSize*sizeof(int)));
  
  return 0;
}

//////////////////////////////////////////////////////////////////////
int NestedLoop::Run(int Nx, int *d_Ny, int i_func)
{
  return CumulSumNestedLoop(Nx, d_Ny, i_func);
}


//////////////////////////////////////////////////////////////////////
int NestedLoop::CumulSumNestedLoop(int Nx, int *d_Ny, int i_func)
{
  //TMP
  //double time_mark=getRealTime();
  //
  prefix_scan_.Scan(d_Ny_cumul_sum_, d_Ny, Nx+1);
  //TMP
  //printf("pst: %lf\n", getRealTime()-time_mark);
  //	 
  int Ny_sum;
  gpuErrchk(cudaMemcpy(&Ny_sum, &d_Ny_cumul_sum_[Nx],
			  sizeof(int), cudaMemcpyDeviceToHost));

  //printf("CSNL: %d %d\n", Nx, Ny_sum);
  
  //printf("Ny_sum %u\n", Ny_sum);
  //temporary - remove
  /*
  if (Ny_sum==0) {
    printf("Nx %d\n", Nx);
    for (int i=0; i<Nx+1; i++) {
      int psum;
      gpuErrchk(cudaMemcpy(&psum, &d_Ny_cumul_sum_[i],
  			      sizeof(int), cudaMemcpyDeviceToHost));
      printf("%d %d\n", i, psum);
    }
  }
  */    
  ////
  if(Ny_sum>0) {
    int grid_dim_x, grid_dim_y;
    if (Ny_sum<65536*1024) { // max grid dim * max block dim
      grid_dim_x = (Ny_sum+1023)/1024;
      grid_dim_y = 1;
    }
    else {
      grid_dim_x = 32; // I think it's not necessary to increase it
      if (Ny_sum>grid_dim_x*1024*65535) {
	throw ngpu_exception(std::string("Ny sum ") + std::to_string(Ny_sum) +
			     " larger than threshold "
			     + std::to_string(grid_dim_x*1024*65535));
      }
      grid_dim_y = (Ny_sum + grid_dim_x*1024 -1) / (grid_dim_x*1024);
    }
    dim3 numBlocks(grid_dim_x, grid_dim_y);
    //TMP
    //double time_mark=getRealTime();
    //
    switch (i_func) {
    case 0:
      CumulSumNestedLoopKernel0<<<numBlocks, 1024>>>
	(Nx, d_Ny_cumul_sum_, Ny_sum);
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
      break;
    case 1:
      CumulSumNestedLoopKernel1<<<numBlocks, 1024>>>
	(Nx, d_Ny_cumul_sum_, Ny_sum);
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
      break;
    default:
      throw ngpu_exception("unknown nested loop function");
    }

    //TMP
    //printf("cst: %lf\n", getRealTime()-time_mark);
    //
  }
    
  return 0;
}

