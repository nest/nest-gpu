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
#include "utilities.h"
#include "syn_model.h"
#include "nested_loop.h"

const int Ny_arr_size_ = 24;
int Ny_th_arr_[] = {
  355375,
  215546,
  48095,
  29171,
  29171,
  10731,
  10731,
  17693,
  10731,
  6509,
  6509,
  3948,
  2395,
  1452,
  881,
  534,
  324,
  197,
  119,
  119,
  119,
  72,
  72,
  72
};

namespace NestedLoop
{
  //#include "Ny_th.h"
  void *d_sort_storage_;
  size_t sort_storage_bytes_;
  void *d_reduce_storage_;
  size_t reduce_storage_bytes_;

  int Nx_max_;
  int *d_max_Ny_;
  int *d_sorted_Ny_;

  int *d_idx_;
  int *d_sorted_idx_;

  int block_dim_x_;
  int block_dim_y_;
  int frame_area_;
  float x_lim_;
}

//TMP
#include "getRealTime.h"
//

//////////////////////////////////////////////////////////////////////
// declare here the functions called by the nested loop 
//__device__ void NestedLoopFunction0(int ix, int iy);
//__device__ void NestedLoopFunction1(int ix, int iy);
//////////////////////////////////////////////////////////////////////
extern __constant__ long long NESTGPUTimeIdx;
extern __constant__ float NESTGPUTimeResolution;
extern __constant__ NodeGroupStruct NodeGroupArray[];
extern __device__ int16_t *NodeGroupMap;



namespace NestedLoop
{
  int *d_Ny_cumul_sum_;
  PrefixScan prefix_scan_;
}

//////////////////////////////////////////////////////////////////////
int NestedLoop::Init()
{
  //return Init(65536*1024);
  return Init(128*1024);
}

//////////////////////////////////////////////////////////////////////
int NestedLoop::Init(int Nx_max)
{
  //prefix_scan_.Init();
  CUDAMALLOCCTRL("&d_Ny_cumul_sum_",&d_Ny_cumul_sum_,
			  PrefixScan::AllocSize*sizeof(int));

  if (Nx_max <= 0) return 0;

  block_dim_x_ = 32;
  block_dim_y_ = 32;
  frame_area_ = 65536*64;
  x_lim_ = 0.75;
  Nx_max_ = Nx_max;

  CUDAMALLOCCTRL("&d_max_Ny_",&d_max_Ny_, sizeof(int));  
  CUDAMALLOCCTRL("&d_sorted_Ny_",&d_sorted_Ny_, Nx_max*sizeof(int));
  CUDAMALLOCCTRL("&d_idx_",&d_idx_, Nx_max*sizeof(int));
  CUDAMALLOCCTRL("&d_sorted_idx_",&d_sorted_idx_, Nx_max*sizeof(int));

  int *h_idx = new int[Nx_max];
  for(int i=0; i<Nx_max; i++) {
    h_idx[i] = i;
  }  
  gpuErrchk(cudaMemcpy(d_idx_, h_idx, Nx_max*sizeof(int),
			  cudaMemcpyHostToDevice));
  delete[] h_idx;
    
  // Determine temporary storage requirements for RadixSort
  d_sort_storage_ = NULL;
  sort_storage_bytes_ = 0;
  cub::DeviceRadixSort::SortPairs(d_sort_storage_, sort_storage_bytes_,
				  d_sorted_Ny_, d_sorted_Ny_, d_idx_,
				  d_sorted_idx_, Nx_max);
  // Determine temporary device storage requirements for Reduce
  d_reduce_storage_ = NULL;
  reduce_storage_bytes_ = 0;
  int *d_Ny = NULL;
  cub::DeviceReduce::Max(d_reduce_storage_, reduce_storage_bytes_, d_Ny,
			 d_max_Ny_, Nx_max);

  // Allocate temporary storage
  CUDAMALLOCCTRL("&d_sort_storage_",&d_sort_storage_, sort_storage_bytes_);
  CUDAMALLOCCTRL("&d_reduce_storage_",&d_reduce_storage_, reduce_storage_bytes_);

  return 0;
}


