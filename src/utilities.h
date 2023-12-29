/*
 *  utilities.h
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

#ifndef UTILITIES_H
#define UTILITIES_H

#include <cub/cub.cuh>
#include "cuda_error.h"

__device__  __forceinline__ double atomicAddDouble(double* address, double val)
{
    unsigned long long int* address_as_ull =
                                          (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + 
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

template <class T1, class T2>
__device__  __forceinline__ T2 locate(T1 val, T1 *data, T2 n)
{
  T2 i_left = 0;
  T2 i_right = n;
  T2 i = (i_left+i_right)/2;
  while(i_right-i_left>1) {
    if (data[i] > val) i_right = i;
    else if (data[i]<val) i_left = i;
    else break;
    i=(i_left+i_right)/2;
  }

  return i;
}

int IntPow(int x, unsigned int p);


template <class T>
T *sortArray(T *h_arr, int n_elem)
{
  // allocate unsorted and sorted array in device memory
  T *d_arr_unsorted;
  T *d_arr_sorted;
  CUDAMALLOCCTRL("&d_arr_unsorted",&d_arr_unsorted, n_elem*sizeof(T));
  CUDAMALLOCCTRL("&d_arr_sorted",&d_arr_sorted, n_elem*sizeof(T));
  gpuErrchk(cudaMemcpy(d_arr_unsorted, h_arr, n_elem*sizeof(T),
		       cudaMemcpyHostToDevice));
  void *d_storage = NULL;
  size_t storage_bytes = 0;
  // Determine temporary storage requirements for sorting source indexes
  cub::DeviceRadixSort::SortKeys(d_storage, storage_bytes, d_arr_unsorted,
				 d_arr_sorted, n_elem);
  // Allocate temporary storage for sorting
  CUDAMALLOCCTRL("&d_storage",&d_storage, storage_bytes);
  // Run radix sort
  cub::DeviceRadixSort::SortKeys(d_storage, storage_bytes, d_arr_unsorted,
				 d_arr_sorted, n_elem);
  CUDAFREECTRL("d_storage",d_storage);
  CUDAFREECTRL("d_arr_unsorted",d_arr_unsorted);

  return d_arr_sorted;
}


#endif
