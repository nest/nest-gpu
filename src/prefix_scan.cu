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
#include "prefix_scan.h"
#include "scan.h"

const unsigned int PrefixScan::AllocSize = 13 * 1048576 / 2;

int PrefixScan::Init()
{
  //printf("Initializing CUDA-C scan...\n\n");
  //initScan();
  
  return 0;
}

int PrefixScan::Scan(int *d_Output, int *d_Input, int n)
{
  prefix_scan(d_Output, d_Input, n, true);

  return 0;
}

int PrefixScan::Free()
{
  //closeScan();
  //gpuErrchk(cudaFree(d_Output));
  //gpuErrchk(cudaFree(d_Input));
  
  return 0;
}
