/*
Copyright (C) 2022 Bruno Golosio
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include "copass_kernels.h"

//#define PRINT_VRB

unsigned int nextPowerOf2(unsigned int n) 
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}


/*
//template <class T>
void cudaReusableAlloc(void *d_storage, int64_t &st_bytes,
		       void **variable_pt, const int64_t &num_elems,
		       const size_t &elem_size)
{
  int64_t align_bytes = elem_size;
  int64_t align_mask = ~(align_bytes - 1);
  int64_t allocation_offset = (st_bytes + align_bytes - 1) & align_mask;
  st_bytes = allocation_offset + num_elems*elem_size;
  if (d_storage != NULL) {
    *variable_pt = (void*)((char*)d_storage + allocation_offset);
  }
}
*/

// atomically set old_index = *arg_max_pt, 
// check whether array[index]>array[old_index].
// If it is true, set *arg_max_pt=index
__device__ int atomicArgMax(position_t *array, int *arg_max_pt, int index)
{
  int old_index = *arg_max_pt;
  int assumed_index;
  do {
    if (array[old_index]>=array[index]) {
      break;
    }
    assumed_index = old_index;
    old_index = atomicCAS(arg_max_pt, assumed_index, index);
  } while (assumed_index != old_index);
  
  return old_index;
}


__global__ void copass_last_step_kernel(position_t *part_size, position_t *m_d,
					uint k, position_t tot_diff,
					position_t *diff,
					position_t *diff_cumul,
					position_t *num_down)
{
  int i=threadIdx.x;
  if (i >= k) return;
  position_t nd = *num_down;
  
  if (i < nd) {
    part_size[i] = m_d[i] + diff[i];
  }
  else if (i == nd) {
    part_size[i] = m_d[i] + tot_diff - diff_cumul[i];
  }
  else {
    part_size[i] = m_d[i];
  }
#ifdef PRINT_VRB
  printf("kernel i: %d\tm_d: %ld\tpart_size: %ld\n", i, m_d[i], part_size[i]);
#endif
}


__global__ void case2_inc_partitions_kernel(position_t *part_size,
					    int *sorted_extra_elem_idx,
					    position_t tot_diff)
{
  int i_elem = threadIdx.x;
  if (i_elem >= tot_diff) return;
  int i = sorted_extra_elem_idx[i_elem];
  part_size[i]++;
}

void GPUMemCpyOverlap(char *t_addr, char *s_addr, position_t size)
{
  position_t diff = (position_t)(t_addr - s_addr);
  if (diff==0) return;
  if (diff<0) {
    printf("GPUMemCpyOvelap error: translation cannot be <0\n");
    exit(0);
  }
  if (diff>=size) {
    gpuErrchk(cudaMemcpyAsync(t_addr, s_addr, size, cudaMemcpyDeviceToDevice));
  }
  int nb = (int)((size + diff - 1)/diff);
  for (int ib=nb-1; ib>=0; ib--) {
    position_t b_size = ib<nb-1 ? diff : size - diff*(nb - 1);
    char *s_b_addr = s_addr + diff*ib;
    char *t_b_addr = s_b_addr + diff;
    gpuErrchk(cudaMemcpyAsync(t_b_addr, s_b_addr, b_size,
			      cudaMemcpyDeviceToDevice));
  }
}

void GPUMemCpyBuffered(char *t_addr, char *s_addr, position_t size,
		       char *d_buffer, position_t buffer_size)
{
  position_t diff = (position_t)(t_addr - s_addr);
  if (diff==0) return;
  if (diff<0) {
    printf("GPUMemCpyBuffer error: translation cannot be <0\n");
    exit(0);
  }
  if (diff>=size) {
    gpuErrchk(cudaMemcpyAsync(t_addr, s_addr, size, cudaMemcpyDeviceToDevice));
    return;
  }
  if (diff>buffer_size/2) {
    GPUMemCpyOverlap(t_addr, s_addr, size);
    return;
  }
  int nb = (int)((size + buffer_size - 1)/buffer_size);
  for (int ib=nb-1; ib>=0; ib--) {
    position_t b_size = ib<nb-1 ? buffer_size : size - buffer_size*(nb - 1);
    char *s_b_addr = s_addr + buffer_size*ib;
    char *t_b_addr = s_b_addr + diff;
    gpuErrchk(cudaMemcpyAsync(d_buffer, s_b_addr, b_size,
			      cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpyAsync(t_b_addr, d_buffer, b_size,
			      cudaMemcpyDeviceToDevice));
  }
}

