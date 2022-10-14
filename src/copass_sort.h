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

#ifndef COPASS_SORT_H
#define COPASS_SORT_H

#include "cuda_error.h"
#include "copass_kernels.h"

extern const bool print_gpu_cpu_vrb;

extern bool compare_with_serial;
extern uint last_i_sub;

namespace copass_sort
{
  //////////////////////////////////////////
  // temporary, for testing
  extern uint k_;
  extern position_t block_size_;
  extern void *d_aux_array_key_pt_;
  extern void *d_aux_array_value_pt_;
  extern position_t *h_part_size_;
  extern position_t *d_part_size_;
  ////////////////////////////////////////////////////
    
  template<class KeyT, class ElementT, class KeyArrayT, class ArrayT,
	   class AuxArrayT>
  int sort_template(KeyArrayT key_array, ArrayT *h_subarray,
		    uint k, position_t block_size,
		    void *d_storage, int64_t &st_bytes);

  template <class ElementT, class ArrayT, class AuxArrayT>
  int extract_partitions(ArrayT *d_subarray, uint k,
			 uint k_next_pow_2,
			 position_t *d_part_size, position_t *d_part_size_cumul,
			 AuxArrayT *d_aux_array);  
  
  int last_step(position_t *local_d_m_d, position_t *local_d_m_u,
		position_t *local_d_sum_m_d,
		position_t local_h_sum_m_d,
		position_t tot_part_size,
		uint k, uint kp_next_pow_2,
		position_t *d_part_size, position_t *d_diff,
		position_t *d_diff_cumul, position_t *h_diff,
		position_t *h_diff_cumul, position_t *d_num_down);

template <class KeyT, class ArrayT>
int last_step_case2(ArrayT *d_subarray,
		    position_t tot_part_size,
		    uint k, position_t *d_part_size,
		    position_t *d_m_d, position_t *d_m_u,
		    position_t h_sum_m_d,
		    KeyT *d_extra_elem,
		    KeyT *h_extra_elem,
		    int *d_extra_elem_idx,
		    int *h_extra_elem_idx,
		    int *d_n_extra_elems);
  
  ////////////// Temporary for checking
  template <class KeyT>
  KeyT *get_aux_array_keys();

  template <class ValueT>
  ValueT *get_aux_array_values();

  position_t *get_part_size();

  template <class KeyT>
  int alloc(position_t n, position_t block_size);
  
  template <class KeyT>
  int sort(KeyT *d_keys);

};


  
template <class ElementT, class ArrayT, class AuxArrayT>
int copass_sort::extract_partitions(ArrayT *d_subarray, uint k,
				    uint k_next_pow_2,
				    position_t *d_part_size,
				    position_t *d_part_size_cumul,
				    AuxArrayT d_aux_array)
{
  prefix_scan<position_t, 1024><<<1, 512>>>(d_part_size, d_part_size_cumul, k,
				k_next_pow_2);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  
  extract_partitions_kernel<ElementT, ArrayT, AuxArrayT><<< k, 1024 >>>
    (d_subarray, k, d_part_size, d_part_size_cumul, d_aux_array);
  
  DBGCUDASYNC
  //gpuErrchk(cudaPeekAtLastError());
  //gpuErrchk(cudaDeviceSynchronize());
  
  return 0;
}


template <class KeyT, class ArrayT>
int copass_sort::last_step_case2(ArrayT *d_subarray,
				 position_t tot_part_size,
				 uint k, position_t *d_part_size,
				 position_t *d_m_d, position_t *d_m_u,
				 position_t h_sum_m_d,
				 KeyT *d_extra_elem,
				 KeyT *h_extra_elem,
				 int *d_extra_elem_idx,
				 int *h_extra_elem_idx,
				 int *d_n_extra_elems)
{
  gpuErrchk(cudaMemcpy(d_part_size, d_m_d, k*sizeof(position_t),
		       cudaMemcpyDeviceToDevice));
  
  position_t tot_diff = tot_part_size - h_sum_m_d;
  // printf("kernel tot_diff: %ld\n", tot_diff);
  
  if (tot_diff > 0) {
    case2_extra_elems_kernel<KeyT, ArrayT><<<1, 1024>>>
      (d_subarray, k, d_m_d, d_m_u,
       d_extra_elem, d_extra_elem_idx, d_n_extra_elems);
    
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    int n_extra_elems;
    gpuErrchk(cudaMemcpy(&n_extra_elems, d_n_extra_elems, sizeof(int),
			 cudaMemcpyDeviceToHost));
    if (n_extra_elems < tot_diff) {
      printf("Error in copass_last_step_case2_gpu. Not enough extra elements"
	     " to complete partitions\n");
      exit(EXIT_FAILURE);
    }

    //// !!!!!!!!! temporarily sort in CPU side using std::sort
    //// replace with cub sort directly in the GPU
    gpuErrchk(cudaMemcpy(h_extra_elem, d_extra_elem,
			 n_extra_elems*sizeof(KeyT),
			 cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_extra_elem_idx, d_extra_elem_idx,
			 n_extra_elems*sizeof(int),
			 cudaMemcpyDeviceToHost));
    // build pair vector
    std::vector<std::pair<KeyT, int>> extra_elem_and_idx;
    for (int i=0; i<n_extra_elems; i++) {
      std::pair<KeyT, int> p(h_extra_elem[i], h_extra_elem_idx[i]);
      extra_elem_and_idx.push_back(p);
    }
    // sort pair
    std::sort(extra_elem_and_idx.begin(), extra_elem_and_idx.end());
    //, [](auto &left, auto &right) {return left.second < right.second;);
    // extract indexes from sorted vector
    for (int i=0; i<n_extra_elems; i++) {
      h_extra_elem_idx[i] = extra_elem_and_idx[i].second;
    }
    
    gpuErrchk(cudaMemcpy(d_extra_elem_idx, h_extra_elem_idx,
			 n_extra_elems*sizeof(int),
			 cudaMemcpyHostToDevice));

    /////////////////////////////////////////////////
    
    case2_inc_partitions_kernel<<<1, 1024>>>(d_part_size,
					     d_extra_elem_idx, tot_diff);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
  }
  
  return 0;
}

template<class KeyT, class ElementT, class KeyArrayT, class ArrayT,
	 class AuxArrayT>
int copass_sort::sort_template(KeyArrayT key_array, ArrayT *h_subarray,
			       uint k, position_t block_size,
			       void *d_storage, int64_t &st_bytes)
{
  //////////////////////////////////////////////////////////////////////
  // uint k; // number of subarrays
  // position_t block_size;  // size of auxiliary array for storage
  //////////////////////////////////////////////////////////////////////
  const int buffer_fract = 5;
  
  ArrayT *d_subarray;
  
  AuxArrayT d_aux_array;

  position_t *h_part_size; // size of extracted partitions
  position_t *d_part_size;
  position_t *d_part_size_cumul;
  
  position_t *d_m_u;
  position_t *d_m_d;
  position_t *d_mu_u;
  position_t *d_mu_d;
  position_t *d_sum_m_u;
  position_t *d_sum_m_d;
  position_t *d_sum_mu_u;
  position_t *d_sum_mu_d;
  
  KeyT *d_t_u;
  KeyT *d_t_d;

  position_t *h_m_u;
  position_t *h_m_d;
  position_t h_sum_m_u;
  position_t h_sum_m_d;
  
  position_t *h_mu_u;
  position_t *h_mu_d;
  position_t h_sum_mu_u;
  position_t h_sum_mu_d;
  
  position_t *d_diff;
  position_t *d_diff_cumul;
  
  position_t *h_diff;
  position_t *h_diff_cumul;

  position_t *d_num_down;

  position_t *d_max_diff;
  int *d_arg_max;

  KeyT *d_t_tilde;

  uint k_next_pow_2;
  uint kp_next_pow_2;

  KeyT *d_extra_elem;
  KeyT *h_extra_elem;
  
  int *d_extra_elem_idx;
  int *h_extra_elem_idx;

  int *d_n_extra_elems;

  char *d_buffer;
  
  array_GPUMalloc(d_storage, st_bytes, d_aux_array, block_size);
  
  position_t buffer_size = block_size/buffer_fract;
  cudaReusableAlloc(d_storage, st_bytes, &d_buffer, buffer_size, sizeof(char));

  h_part_size  = new position_t[k];
  cudaReusableAlloc(d_storage, st_bytes, &d_part_size, k, sizeof(position_t));

  cudaReusableAlloc(d_storage, st_bytes, &d_part_size_cumul,
		    (k + 1), sizeof(position_t));

  cudaReusableAlloc(d_storage, st_bytes, &d_m_u, k, sizeof(position_t));
  cudaReusableAlloc(d_storage, st_bytes, &d_m_d, k, sizeof(position_t));
  cudaReusableAlloc(d_storage, st_bytes, &d_mu_u, k, sizeof(position_t));
  cudaReusableAlloc(d_storage, st_bytes, &d_mu_d, k, sizeof(position_t));

  cudaReusableAlloc(d_storage, st_bytes, &d_sum_m_u, 1, sizeof(position_t));
  cudaReusableAlloc(d_storage, st_bytes, &d_sum_m_d, 1, sizeof(position_t));

  cudaReusableAlloc(d_storage, st_bytes, &d_sum_mu_u, 1, sizeof(position_t));
  cudaReusableAlloc(d_storage, st_bytes, &d_sum_mu_d, 1, sizeof(position_t));

  cudaReusableAlloc(d_storage, st_bytes, &d_t_u, 1, sizeof(KeyT));
  cudaReusableAlloc(d_storage, st_bytes, &d_t_d, 1, sizeof(KeyT));

  h_m_u = new position_t[k];
  h_m_d = new position_t[k];
  
  h_mu_u = new position_t[k];
  h_mu_d = new position_t[k];

  // use one more element (k+1) to avoid illegal memory access of
  // subsequent use of the arrays in prefix scan
  cudaReusableAlloc(d_storage, st_bytes, &d_diff, k + 1, sizeof(position_t));
  cudaReusableAlloc(d_storage, st_bytes, &d_diff_cumul, k + 1,
		    sizeof(position_t));

  h_diff = new position_t[k];
  h_diff_cumul = new position_t[k+1];

  cudaReusableAlloc(d_storage, st_bytes, &d_num_down, 1, sizeof(position_t));

  cudaReusableAlloc(d_storage, st_bytes, &d_max_diff, 1, sizeof(position_t));
  cudaReusableAlloc(d_storage, st_bytes, &d_arg_max, 1, sizeof(int));

  cudaReusableAlloc(d_storage, st_bytes, &d_t_tilde, 1, sizeof(KeyT));

  k_next_pow_2 = nextPowerOf2(k);
  kp_next_pow_2 = nextPowerOf2(k+1);

  cudaReusableAlloc(d_storage, st_bytes, &d_extra_elem, k, sizeof(KeyT));
  h_extra_elem = new KeyT[k];  

  cudaReusableAlloc(d_storage, st_bytes, &d_extra_elem_idx, k, sizeof(int));
  h_extra_elem_idx = new int[k];  

  cudaReusableAlloc(d_storage, st_bytes, &d_n_extra_elems, 1, sizeof(int));

  cudaReusableAlloc(d_storage, st_bytes, &d_subarray, k, sizeof(ArrayT));

  // if d_storage==NULL this function should only evaluate the storage bytes
  if (d_storage == NULL) {
    int64_t align_bytes = 256;
    int64_t align_mask = ~(align_bytes - 1);

    st_bytes = (st_bytes + align_bytes - 1) & align_mask;
    
    return 0;
  }

  gpuErrchk(cudaMemcpyAsync(d_subarray, h_subarray,
			    k*sizeof(ArrayT), cudaMemcpyHostToDevice));

  ///// TEMPORARY, FOR TESTING
  k_ = k;
  block_size_ = block_size;
  h_part_size_ = h_part_size;
  d_part_size_ = d_part_size;
  d_aux_array_key_pt_ = getKeyPt(d_aux_array);
  d_aux_array_value_pt_ = getValuePt(d_aux_array);
  
  //////////////////// serve???????!!!!!!!!!!
  position_t tot_part_size = block_size;

  ArrayT target_array[k-1];
  for (uint i=0; i<k-1; i++) {
    target_array[i] = h_subarray[i];
    for (uint j=i+1; j<k; j++) {
      target_array[i].size += h_subarray[j].size;
    }
  }

  //////////////////////////////////////////////////////////
  // LOOP SHOULD START HERE
  //////////////////////////////////////////////////////////
  for (uint i_sub=0; i_sub<k-1; i_sub++) {
    threshold_range_kernel<KeyT, ArrayT, 1024> <<<1, k>>>
      (d_subarray, block_size, k, d_t_u, d_t_d);

    //DBGCUDASYNC
    CUDASYNC
    search_multi_down<KeyT, ArrayT, 1024>
      (d_subarray, k, d_t_u, d_m_u, d_sum_m_u);
    CUDASYNC
    search_multi_up<KeyT, ArrayT, 1024>
      (d_subarray, k, d_t_d, d_m_d, d_sum_m_d);
    CUDASYNC
    gpuErrchk(cudaMemcpyAsync(&h_sum_m_u, d_sum_m_u, sizeof(position_t),
			      cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&h_sum_m_d, d_sum_m_d, sizeof(position_t),
			 cudaMemcpyDeviceToHost));
    if (print_gpu_cpu_vrb)
      printf("kernel sum_m_u: %ld\tsum_m_d: %ld\n", h_sum_m_u, h_sum_m_d);
    /////////////////////////////////////////////////////////////
    if (block_size >= h_sum_m_u) { // m_u -> m_d
      search_multi_up<KeyT, ArrayT, 1024>
	(d_subarray, k, d_t_u, d_mu_u, d_sum_mu_u);
      /////////////////////
      gpuErrchk(cudaMemcpyAsync(h_m_u, d_m_u, k*sizeof(position_t),
			   cudaMemcpyDeviceToHost));
      gpuErrchk(cudaMemcpyAsync(h_mu_u, d_mu_u, k*sizeof(position_t),
			   cudaMemcpyDeviceToHost));
      if (print_gpu_cpu_vrb) {
	CUDASYNC
	printf("last step gpu cond 0 h_m_u: ");
	for (uint i=0; i<k; i++) {
	  printf("%ld ", h_m_u[i]);
	}
	printf("\n");
	printf("last step gpu cond 0 h_mu_u: ");
	for (uint i=0; i<k; i++) {
	  printf("%ld ", h_mu_u[i]);
	}
	printf("\n");
      }
      last_step(d_m_u, d_mu_u, d_sum_m_u, h_sum_m_u,
		tot_part_size, k, kp_next_pow_2, d_part_size, d_diff,
		d_diff_cumul, h_diff, h_diff_cumul, d_num_down);
      if (print_gpu_cpu_vrb) {
	CUDASYNC
	printf("Kernel Final step condition 0\n");
	printf("Kernel total partition size before final step: %ld\n", h_sum_m_u);
      }
    }

    //////////////////////////////////////////////////////////////
    else if (block_size <= h_sum_m_d) {
      search_multi_down<KeyT, ArrayT, 1024>
	(d_subarray, k, d_t_d, d_mu_d, d_sum_mu_d);
      /////////////////////
      gpuErrchk(cudaMemcpyAsync(h_mu_d, d_mu_d, k*sizeof(position_t),
				cudaMemcpyDeviceToHost));
      gpuErrchk(cudaMemcpyAsync(h_m_d, d_m_d, k*sizeof(position_t),
				cudaMemcpyDeviceToHost));
      gpuErrchk(cudaMemcpy(&h_sum_mu_d, d_sum_mu_d, sizeof(position_t),
			   cudaMemcpyDeviceToHost));
      if (print_gpu_cpu_vrb) {
	printf("last step gpu cond 1 h_mu_d: ");
	for (uint i=0; i<k; i++) {
	  printf("%ld ", h_mu_d[i]);
	}
	printf("\n");
	printf("last step gpu cond 1 h_m_d: ");
	for (uint i=0; i<k; i++) {
	  printf("%ld ", h_m_d[i]);
	}
	printf("\n");
      }
      last_step(d_mu_d, d_m_d, d_sum_mu_d, h_sum_mu_d,
		tot_part_size, k, kp_next_pow_2, d_part_size, d_diff,
		d_diff_cumul, h_diff, h_diff_cumul, d_num_down);
      if (print_gpu_cpu_vrb) {
	CUDASYNC
	printf("Kernel Final step condition 1\n");
	printf("Kernel total partition size before final step: %ld\n",
	       h_sum_mu_d);
      }
    }
    else {
      for(;;) {
	max_diff_kernel<ArrayT, 1024><<<1, 1024>>>
	  (d_m_u, d_m_d, k, d_subarray, d_max_diff, d_arg_max);
	DBGCUDASYNC
	position_t h_max_diff;
	gpuErrchk(cudaMemcpy(&h_max_diff, d_max_diff, sizeof(position_t),
			     cudaMemcpyDeviceToHost));
	if (h_max_diff<=1) {
	  gpuErrchk(cudaMemcpy(&h_sum_m_d, d_sum_m_d, sizeof(position_t),
			       cudaMemcpyDeviceToHost));
	  last_step_case2<KeyT, ArrayT>
	    (d_subarray, tot_part_size, k, d_part_size,
	     d_m_d, d_m_u, h_sum_m_d, d_extra_elem,
	     h_extra_elem, d_extra_elem_idx, h_extra_elem_idx,
	     d_n_extra_elems);
	  if (print_gpu_cpu_vrb) {
	    CUDASYNC
	    printf("Kernel final step condition 2\n");
	    printf("Total partition size before final step: %ld\n", h_sum_m_d);
	  }
	  break;
	}
	eval_t_tilde_kernel<KeyT, ArrayT><<< 1, 1 >>>
	  (d_subarray, d_m_u, d_m_d, d_arg_max, d_t_tilde);
	DBGCUDASYNC
	  
	search_multi_up<KeyT, ArrayT, 1024>
	  (d_subarray, k, d_t_tilde, d_mu_u,
	   d_sum_mu_u);
	search_multi_down<KeyT, ArrayT, 1024>
	  (d_subarray, k, d_t_tilde, d_mu_d,
	   d_sum_mu_d);
	gpuErrchk(cudaMemcpyAsync(&h_sum_mu_u, d_sum_mu_u, sizeof(position_t),
				  cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&h_sum_mu_d, d_sum_mu_d, sizeof(position_t),
			     cudaMemcpyDeviceToHost));
	if (block_size < h_sum_mu_d) {
	  gpuErrchk(cudaMemcpyAsync(d_m_u, d_mu_d, k*sizeof(position_t),
				    cudaMemcpyDeviceToDevice));
	  gpuErrchk(cudaMemcpyAsync(d_sum_m_u, d_sum_mu_d, sizeof(position_t),
			       cudaMemcpyDeviceToDevice));
	}
	else if (block_size > h_sum_mu_u) {
	  gpuErrchk(cudaMemcpyAsync(d_m_d, d_mu_u, k*sizeof(position_t),
				    cudaMemcpyDeviceToDevice));
	  gpuErrchk(cudaMemcpyAsync(d_sum_m_d, d_sum_mu_u, sizeof(position_t),
				    cudaMemcpyDeviceToDevice));
	}
	else { // sum_mu_d <= tot_part_size <= sum_mu_u
	  last_step(d_mu_d, d_mu_u, d_sum_mu_d, h_sum_mu_d,
		    tot_part_size, k, kp_next_pow_2, d_part_size,
		    d_diff, d_diff_cumul, h_diff, h_diff_cumul,
		    d_num_down);
	  if (print_gpu_cpu_vrb) {
	    CUDASYNC
	    printf("Kernel final step condition 3\n");
	    printf("Kernel total part size before final step: %ld\n", h_sum_mu_d);
	  }
	  break;
	}
      }
    }

    extract_partitions<ElementT, ArrayT>
      (d_subarray, k, k_next_pow_2, d_part_size,
       d_part_size_cumul, d_aux_array);

    //////////////////////////////////////////////////////////////////////
    //// USE THE INDEX OF THE ITERATION ON the k -1 target arrays
    gpuErrchk(cudaMemcpy(h_part_size, d_part_size_, k*sizeof(position_t),
			 cudaMemcpyDeviceToHost));

    repack(h_subarray, k, h_part_size, d_buffer, buffer_size);
    gpuErrchk(cudaMemcpyAsync(d_subarray, h_subarray,
			      k*sizeof(ArrayT), cudaMemcpyHostToDevice));
    if (compare_with_serial && i_sub==last_i_sub) return 0;

    CopyArray<ElementT, ArrayT, AuxArrayT>
      <<< (block_size + 1023) / 1024, 1024 >>>
      (target_array[i_sub], d_aux_array);
  }
  
  return 0;
}
//////////////////////////////////////////////////////////////////////


template <class KeyT>
int copass_sort::sort(KeyT *d_keys, position_t n, position_t block_size,
		      void *d_storage, int64_t &st_bytes)
{
  st_bytes = 0;
  uint k = (uint)((n + block_size - 1) / block_size); // number of subarrays

  contiguous_array<KeyT> h_subarray[k];
  contiguous_array<KeyT> array_block[k];
  for (uint i=0; i<k; i++) {
    h_subarray[i].data_pt = d_keys;
    h_subarray[i].offset = i * block_size;
    h_subarray[i].size = i<k-1 ? block_size : n-(k-1)*block_size;
    array_block[i] = h_subarray[i];
  }

  int64_t ext_st_bytes = 0;
  for (uint i=0; i<k; i++) {
    array_GPUSort(h_subarray[i], d_storage, ext_st_bytes);
    if (d_storage == NULL) break;
  }

  contiguous_array<KeyT> key_array;
  key_array.data_pt = d_keys;
  key_array.offset = 0;
  key_array.size = n;

  sort_template<KeyT, KeyT, contiguous_array<KeyT>,
		contiguous_array<KeyT>, contiguous_array<KeyT> >
    (key_array, h_subarray, k, block_size, d_storage, st_bytes);

  st_bytes = max(st_bytes, ext_st_bytes);

  if (d_storage==NULL || compare_with_serial) return 0;

  for (uint i=0; i<k; i++) {
    array_GPUSort(array_block[i], d_storage, ext_st_bytes);
  }

  return 0;
}

////////////// Temporary for checking !!!!!!!!!!!!!!!!!
template <class KeyT>
KeyT *copass_sort::get_aux_array_keys()
{
  KeyT *h_aux_array_keys = new KeyT[block_size_];
  gpuErrchk(cudaMemcpy(h_aux_array_keys, d_aux_array_key_pt_,
		       block_size_*sizeof(KeyT),
		       cudaMemcpyDeviceToHost));
  return h_aux_array_keys;
}

////////////// Temporary for checking !!!!!!!!!!!!!!!!!
template <class ValueT>
ValueT *copass_sort::get_aux_array_values()
{
  ValueT *h_aux_array_values = new ValueT[block_size_];
  gpuErrchk(cudaMemcpy(h_aux_array_values, d_aux_array_value_pt_,
		       block_size_*sizeof(ValueT),
		       cudaMemcpyDeviceToHost));
  return h_aux_array_values;
}


template <class KeyT, class ValueT>
int copass_sort::sort(KeyT *d_keys, ValueT *d_values, position_t n,
		      position_t block_size,
		      void *d_storage, int64_t &st_bytes)
{
  st_bytes = 0;
  uint k = (uint)((n + block_size - 1) / block_size); // number of subarrays

  contiguous_key_value<KeyT, ValueT> h_subarray[k];
  contiguous_key_value<KeyT, ValueT> array_block[k];
  for (uint i=0; i<k; i++) {
    h_subarray[i].key_pt = d_keys;
    h_subarray[i].value_pt = d_values;
    h_subarray[i].offset = i * block_size;
    h_subarray[i].size = i<k-1 ? block_size : n-(k-1)*block_size;
    array_block[i] = h_subarray[i];
  }
  
  int64_t ext_st_bytes = 0;
  for (uint i=0; i<k; i++) {
    array_GPUSort(array_block[i], d_storage, ext_st_bytes);
    if (d_storage == NULL) break;
  }

  contiguous_array<KeyT> key_array;
  key_array.data_pt = d_keys;
  key_array.offset = 0;
  key_array.size = n;

  sort_template<KeyT, key_value<KeyT, ValueT>,
		contiguous_array<KeyT>, contiguous_key_value<KeyT, ValueT>,
		contiguous_key_value<KeyT, ValueT> >
    (key_array, h_subarray, k, block_size, d_storage, st_bytes);

  st_bytes = max(st_bytes, ext_st_bytes);

  if (d_storage==NULL || compare_with_serial) return 0;

  for (uint i=0; i<k; i++) {
    //array_GPUSort(h_subarray[i], d_storage, ext_st_bytes);
    array_GPUSort(array_block[i], d_storage, ext_st_bytes);
  }

  return 0;
}


template <class KeyT>
int copass_sort::sort(KeyT **key_subarray, position_t n, position_t block_size,
		      void *d_storage, int64_t &st_bytes)
{
  st_bytes = 0;
  uint k = (uint)((n + block_size - 1) / block_size); // number of subarrays

  regular_block_array<KeyT> h_key_array;
  regular_block_array<KeyT> d_key_array;

  h_key_array.data_pt = key_subarray;
  h_key_array.block_size = block_size;
  h_key_array.offset = 0;
  h_key_array.size = n;

  int64_t ext_st_bytes = 0;
  for (uint i=0; i<k; i++) {
    contiguous_array<KeyT> key_block = getBlock(h_key_array, i);
    array_GPUSort(key_block, d_storage, ext_st_bytes);
    if (d_storage == NULL) break;
  }

  KeyT **d_key_array_data_pt = NULL;
  cudaReusableAlloc(d_storage, st_bytes, &d_key_array_data_pt,
		    k, sizeof(KeyT*));
  if (d_storage != NULL) {
    gpuErrchk(cudaMemcpy(d_key_array_data_pt, key_subarray,
			 k*sizeof(KeyT*), cudaMemcpyHostToDevice));
  }

  d_key_array.data_pt = d_key_array_data_pt; //key_subarray;
  d_key_array.block_size = block_size;
  d_key_array.offset = 0;
  d_key_array.size = n;

  regular_block_array<KeyT> h_subarray[k];
  for (uint i=0; i<k; i++) {
    h_subarray[i].h_data_pt = key_subarray;
    h_subarray[i].data_pt = d_key_array_data_pt; //key_subarray;
    h_subarray[i].block_size = block_size;
    h_subarray[i].offset = i * block_size;
    h_subarray[i].size = i<k-1 ? block_size : n-(k-1)*block_size;
  }
  
  sort_template<KeyT, KeyT, regular_block_array<KeyT>,
		regular_block_array<KeyT>, contiguous_array <KeyT> >
    (d_key_array, h_subarray, k, block_size, d_storage, st_bytes);

  st_bytes = max(st_bytes, ext_st_bytes);

  if (d_storage==NULL || compare_with_serial) return 0;
  
  for (uint i=0; i<k; i++) {
    contiguous_array<KeyT> key_block = getBlock(h_key_array, i);
    array_GPUSort(key_block, d_storage, ext_st_bytes);
  }

  return 0;
}

template <class KeyT, class ValueT>
int copass_sort::sort(KeyT **key_subarray, ValueT **value_subarray,
		      position_t n, position_t block_size,
		      void *d_storage, int64_t &st_bytes)
{
  st_bytes = 0;
  uint k = (uint)((n + block_size - 1) / block_size); // number of subarrays

  regular_block_key_value<KeyT, ValueT> h_key_value;
  regular_block_array<KeyT> d_key_array;

  h_key_value.key_pt = key_subarray;
  h_key_value.value_pt = value_subarray;
  h_key_value.block_size = block_size;
  h_key_value.offset = 0;
  h_key_value.size = n;
  
  int64_t ext_st_bytes = 0;
  for (uint i=0; i<k; i++) {
    contiguous_key_value<KeyT, ValueT> key_value_block =
      getBlock(h_key_value, i);
    array_GPUSort(key_value_block, d_storage, ext_st_bytes);
    if (d_storage == NULL) break;
  }

  KeyT **d_key_array_data_pt = NULL;
  cudaReusableAlloc(d_storage, st_bytes, &d_key_array_data_pt,
		    k, sizeof(KeyT*));
  

  if (d_storage != NULL) {
    gpuErrchk(cudaMemcpy(d_key_array_data_pt, key_subarray,
			 k*sizeof(KeyT*), cudaMemcpyHostToDevice));
  }
  ValueT **d_value_array_data_pt = NULL;
  cudaReusableAlloc(d_storage, st_bytes, &d_value_array_data_pt,
		    k, sizeof(ValueT*));
    
  if (d_storage != NULL) {
    gpuErrchk(cudaMemcpy(d_value_array_data_pt, value_subarray,
			 k*sizeof(ValueT*), cudaMemcpyHostToDevice));
  }

  d_key_array.data_pt = d_key_array_data_pt;
  d_key_array.block_size = block_size;
  d_key_array.offset = 0;
  d_key_array.size = n;

  regular_block_key_value<KeyT, ValueT> h_subarray[k];
  for (uint i=0; i<k; i++) {
    h_subarray[i].h_key_pt = key_subarray;
    h_subarray[i].h_value_pt = value_subarray;
    h_subarray[i].key_pt = d_key_array_data_pt;
    h_subarray[i].value_pt = d_value_array_data_pt;
    h_subarray[i].block_size = block_size;
    h_subarray[i].offset = i * block_size;
    h_subarray[i].size = i<k-1 ? block_size : n-(k-1)*block_size;
  }

  sort_template<KeyT, key_value<KeyT, ValueT>, regular_block_array<KeyT>,
		regular_block_key_value<KeyT, ValueT>,
		contiguous_key_value <KeyT, ValueT> >
    (d_key_array, h_subarray, k, block_size, d_storage, st_bytes);

  st_bytes = max(st_bytes, ext_st_bytes);

  if (d_storage==NULL || compare_with_serial) return 0;
    
  for (uint i=0; i<k; i++) {
    contiguous_key_value<KeyT, ValueT> key_value_block =
      getBlock(h_key_value, i);
    array_GPUSort(key_value_block, d_storage, ext_st_bytes);
  }

  return 0;
}

#endif
