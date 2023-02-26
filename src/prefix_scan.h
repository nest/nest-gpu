/*
 *  prefix_scan.h
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





#ifndef PREFIXSCAN_H
#define PREFIXSCAN_H

class PrefixScan
{
 public:
  static const unsigned int AllocSize;

  /*
  uint *d_Input;

  uint *d_Output;

  uint *h_Input;

  uint *h_OutputCPU;

  uint *h_OutputGPU;
  */
  
  int Init();

  int Scan(int *d_Output, int *d_Input, int n);

  int Free();
};

#endif
