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

#include "get_spike.h"
#include "rev_spike.h"

extern int *d_Ny_cumul_sum_;

template<int i_func> 
__global__ void CumulSumNestedLoopKernel(int Nx, int *Ny_cumul_sum,
					 int Ny_sum)
{
  int blockId   = blockIdx.y * gridDim.x + blockIdx.x;
  int array_idx = blockId * blockDim.x + threadIdx.x;
  if (array_idx<Ny_sum) {
    int ix = locate(array_idx, Ny_cumul_sum, Nx + 1);
    int iy = (int)(array_idx - Ny_cumul_sum[ix]);
    NestedLoopFunction<i_func>(ix, iy);
  }
}




#ifndef NESTEDLOOP_H
#define  NESTEDLOOP_H

#include "prefix_scan.h"

namespace NestedLoop
{
  extern PrefixScan prefix_scan_;
  
  int Init();
  
  template<int i_func>
    int Run(int Nx, int *d_Ny);

  template<int i_func>
  int CumulSumNestedLoop(int Nx, int *d_Ny);  

  int Free();
}

//////////////////////////////////////////////////////////////////////
template<int i_func>
int NestedLoop::Run(int Nx, int *d_Ny)
{
  return CumulSumNestedLoop<i_func>(Nx, d_Ny);
}


//////////////////////////////////////////////////////////////////////
template<int i_func>
int NestedLoop::CumulSumNestedLoop(int Nx, int *d_Ny)
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
    CumulSumNestedLoopKernel<i_func><<<numBlocks, 1024>>>
    (Nx, d_Ny_cumul_sum_, Ny_sum);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    //TMP
    //printf("cst: %lf\n", getRealTime()-time_mark);
    //
  }
    
  return 0;
}


#endif
