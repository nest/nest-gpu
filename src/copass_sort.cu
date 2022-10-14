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

#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <climits>
#include <vector>
#include <utility>
#include "cuda_error.h"
#include "copass_kernels.h"
#include "copass_sort.h"

const bool print_gpu_cpu_vrb = false;

namespace copass_sort
{
  uint k_;
  position_t block_size_;
  void *d_aux_array_key_pt_;
  void *d_aux_array_value_pt_;
  position_t *h_part_size_;
  position_t *d_part_size_;
}




int copass_sort::last_step(position_t *local_d_m_d, position_t *local_d_m_u,
			   position_t *local_d_sum_m_d,
			   position_t local_h_sum_m_d,
			   position_t tot_part_size,
			   uint k, uint kp_next_pow_2,
			   position_t *d_part_size, position_t *d_diff,
			   position_t *d_diff_cumul, position_t *h_diff,
			   position_t *h_diff_cumul, position_t *d_num_down)
{
    diffKernel<<<1, k>>>(d_diff, local_d_m_u, local_d_m_d, k);
    DBGCUDASYNC
    prefix_scan<position_t, 1024><<<1, 512>>>
      (d_diff, d_diff_cumul, k+1, kp_next_pow_2);
    DBGCUDASYNC

    gpuErrchk(cudaMemcpyAsync(h_diff, d_diff, k*sizeof(position_t),
			      cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_diff_cumul, d_diff_cumul,
			 (k + 1)*sizeof(position_t),
			 cudaMemcpyDeviceToHost));
    if (print_gpu_cpu_vrb) {
      printf("h_diff: ");
      for (uint i=0; i<k; i++) {
	printf("%ld ", h_diff[i]);
      }
      printf("\n");
      printf("h_diff_cumul: ");
      for (uint i=0; i<k+1; i++) {
	printf("%ld ", h_diff_cumul[i]);
      }
      printf("\n");
    }
    position_t tot_diff = tot_part_size - local_h_sum_m_d;
    search_down<position_t, 1024><<<1, 1024>>>
      (d_diff_cumul+1, k, tot_diff, d_num_down);

    copass_last_step_kernel<<<1, 1024>>>(d_part_size, local_d_m_d, k,
					 tot_diff, d_diff, d_diff_cumul,
					 d_num_down);
    DBGCUDASYNC

    return 0;
}

position_t *copass_sort::get_part_size()
{
  gpuErrchk(cudaMemcpy(h_part_size_, d_part_size_, k_*sizeof(position_t),
		       cudaMemcpyDeviceToHost));
  return h_part_size_;
}

