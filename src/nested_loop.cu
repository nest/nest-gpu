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
#include "nested_loop.h"


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
extern __device__ signed char *NodeGroupMap;

int *d_Ny_cumul_sum_;

namespace NestedLoop
{
  PrefixScan prefix_scan_;
}

//////////////////////////////////////////////////////////////////////
int NestedLoop::Init()
{
  //prefix_scan_.Init();
  gpuErrchk(cudaMalloc(&d_Ny_cumul_sum_,
			  PrefixScan::AllocSize*sizeof(int)));
  
  return 0;
}


