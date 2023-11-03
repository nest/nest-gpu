/*
 *  prefix_scan.cu
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
  //CUDAFREECTRL("d_Output",d_Output);
  //CUDAFREECTRL("d_Input",d_Input);
  
  return 0;
}
