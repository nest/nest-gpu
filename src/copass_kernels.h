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

#ifndef COPASS_KERNEL_H
#define COPASS_KERNEL_H
#include <vector>
#include <utility>
#include <algorithm>
#include <cub/device/device_radix_sort.cuh>

#include "cuda_error.h"

//#define PRINT_VRB

typedef int64_t position_t;
typedef unsigned long long int uposition_t;
//typedef int32_t position_t;
//typedef unsigned int uposition_t;

void GPUMemCpyOverlap(char *t_addr, char *s_addr, position_t size);
void GPUMemCpyBuffered(char *t_addr, char *s_addr, position_t size,
		     char *d_buffer, position_t buffer_size);

template <class T>
void cudaReusableAlloc(void *d_storage, int64_t &st_bytes,
		       T** variable_pt, const int64_t &num_elems,
		       const size_t &elem_size)
{
  int64_t align_bytes = elem_size;
  int64_t align_mask = ~(align_bytes - 1);
  int64_t allocation_offset = (st_bytes + align_bytes - 1) & align_mask;
  st_bytes = allocation_offset + num_elems*elem_size;
  if (d_storage != NULL) {
    *variable_pt = (T*)((char*)d_storage + allocation_offset);
  }
  else {
    *variable_pt = NULL;
  }
}


template <class KeyT, class ValueT>
struct key_value
{
  KeyT key;
  ValueT value;
};

template <class KeyT, class ValueT>
struct contiguous_key_value
{
  KeyT* key_pt;
  ValueT* value_pt;
  position_t offset;
  position_t size;
};

template <class KeyT, class ValueT>
struct regular_block_key_value
{
  KeyT** h_key_pt;
  ValueT** h_value_pt;
  KeyT** key_pt;
  ValueT** value_pt;
  position_t block_size;
  position_t offset;
  position_t size;
};

//namespace array
//{

template <class KeyT, class ValueT>
__device__ key_value<KeyT, ValueT>
getElem(contiguous_key_value<KeyT, ValueT> &arr, position_t i)
{
  key_value<KeyT, ValueT> kv;
  kv.key = *(arr.key_pt + arr.offset + i);
  kv.value = *(arr.value_pt + arr.offset + i);
  return kv;
}

template <class KeyT, class ValueT>
__device__ KeyT getKey(contiguous_key_value<KeyT, ValueT> &arr, position_t i)
{
  return *(arr.key_pt + arr.offset + i);
}

template <class KeyT, class ValueT>
KeyT *getKeyPt(contiguous_key_value<KeyT, ValueT> &arr)
{
  return arr.key_pt + arr.offset;
}

template <class KeyT, class ValueT>
ValueT *getValuePt(contiguous_key_value<KeyT, ValueT> &arr)
{
  return arr.value_pt + arr.offset;
}

template <class KeyT, class ValueT>
__device__ void setElem(contiguous_key_value<KeyT, ValueT> &arr, position_t i,
			const key_value<KeyT, ValueT> &kv)
{
  *(arr.key_pt + arr.offset + i) = kv.key;
  *(arr.value_pt + arr.offset + i) = kv.value;
}


template <class KeyT, class ValueT>
void array_GPUMalloc(void *d_storage, int64_t &st_bytes,
		     contiguous_key_value<KeyT, ValueT> &arr, position_t size)
{
  cudaReusableAlloc(d_storage, st_bytes, &(arr.key_pt), size, sizeof(KeyT));
  cudaReusableAlloc(d_storage, st_bytes, &(arr.value_pt), size, sizeof(ValueT));
  arr.offset = 0;
  arr.size = size;
}

template <class KeyT, class ValueT>
void array_GPUFree(contiguous_key_value<KeyT, ValueT> &arr, position_t size)
{
  CUDAFREECTRL("arr.key_pt",arr.key_pt);
  CUDAFREECTRL("arr.value_pt",arr.value_pt);
  arr.offset = 0;
  arr.size = 0;
}

template <class KeyT, class ValueT>
void array_Malloc(contiguous_key_value<KeyT, ValueT> &arr, position_t size)
{
  arr.key_pt = new KeyT[size];
  arr.value_pt = new ValueT[size];
  arr.offset = 0;
  arr.size = size;
}

template <class KeyT, class ValueT>
void array_Free(contiguous_key_value<KeyT, ValueT> &arr, position_t size)
{
  delete[] arr.key_pt;
  delete[] arr.value_pt;
  arr.offset = 0;
  arr.size = 0;
}

// note: this does not allocate memory fo target array,
// use array_Malloc for that 
template <class KeyT, class ValueT>
void array_GPUtoCPUCopyContent(contiguous_key_value<KeyT, ValueT> &target_arr,
			       contiguous_key_value<KeyT, ValueT> &source_arr)
{
  gpuErrchk(cudaMemcpy(target_arr.key_pt + target_arr.offset,
		       source_arr.key_pt + source_arr.offset,
		       source_arr.size*sizeof(KeyT),
		       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(target_arr.value_pt + target_arr.offset,
		       source_arr.value_pt + source_arr.offset,
		       source_arr.size*sizeof(ValueT),
		       cudaMemcpyDeviceToHost));
  target_arr.size = source_arr.size;
}

// note: this does not allocate memory fo target array,
// use array_GPUMalloc for that 
template <class KeyT, class ValueT>
void array_CPUtoGPUCopyContent(contiguous_key_value<KeyT, ValueT> &target_arr,
			       contiguous_key_value<KeyT, ValueT> &source_arr)
{
  gpuErrchk(cudaMemcpy(target_arr.key_pt + target_arr.offset,
		       source_arr.key_pt + source_arr.offset,
		       source_arr.size*sizeof(KeyT),
		       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(target_arr.value_pt + target_arr.offset,
		       source_arr.value_pt + source_arr.offset,
		       source_arr.size*sizeof(ValueT),
		       cudaMemcpyHostToDevice));
  target_arr.size = source_arr.size;
}

template <class KeyT, class ValueT>
void array_Sort(contiguous_key_value<KeyT, ValueT> &arr)
{
  // build pair vector
  std::vector<std::pair<KeyT, ValueT>> kv;
  position_t i0 = arr.offset;
  for (position_t i=0; i<arr.size; i++) {
    std::pair<KeyT, ValueT> p(arr.key_pt[i0+i], arr.value_pt[i0+i]);
    kv.push_back(p);
  }
  // sort pair
  std::sort(kv.begin(), kv.end(),
	    [](auto &left, auto &right) {
	      return left.first < right.first;});
  // extract elements from sorted vector
  for (position_t i=0; i<arr.size; i++) {
    arr.key_pt[i0+i] = kv[i].first;
    arr.value_pt[i0+i] = kv[i].second;
  }
}

template <class KeyT, class ValueT>
void array_GPUSort(contiguous_key_value<KeyT, ValueT> &arr_in,
		   void *d_storage, int64_t &ext_st_bytes)
{
  ext_st_bytes = 0;
  int num_elems = arr_in.size;
  contiguous_key_value<KeyT, ValueT> arr_out;
  arr_out.offset = 0;
  arr_out.size = num_elems;
  cudaReusableAlloc(d_storage, ext_st_bytes, &arr_out.key_pt, num_elems,
		    sizeof(KeyT));
  cudaReusableAlloc(d_storage, ext_st_bytes, &arr_out.value_pt, num_elems,
		    sizeof(ValueT));
  // the following is just for memory alignement
  void *dummy_pt;
  cudaReusableAlloc(d_storage, ext_st_bytes, &dummy_pt, 1, 256);

  size_t sort_storage_bytes = 0;  
  cub::DeviceRadixSort::SortPairs(NULL, sort_storage_bytes,
				  arr_in.key_pt + arr_in.offset,
				  arr_out.key_pt,
				  arr_in.value_pt + arr_in.offset,
				  arr_out.value_pt, num_elems);

  if (d_storage != NULL) {
    void *d_sort_storage = (void*)((char*)d_storage + ext_st_bytes);
    cub::DeviceRadixSort::SortPairs(d_sort_storage, sort_storage_bytes,
				    arr_in.key_pt + arr_in.offset,
				    arr_out.key_pt,
				    arr_in.value_pt + arr_in.offset,
				    arr_out.value_pt, num_elems);
    
    gpuErrchk(cudaMemcpyAsync(arr_in.key_pt + arr_in.offset, arr_out.key_pt,
			      num_elems*sizeof(KeyT),
			      cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(arr_in.value_pt + arr_in.offset, arr_out.value_pt,
			 num_elems*sizeof(ValueT),
			 cudaMemcpyDeviceToDevice));
  }
  
  ext_st_bytes += sort_storage_bytes;
}

template <class KeyT, class ValueT>
__device__ key_value<KeyT, ValueT>
getElem(regular_block_key_value<KeyT, ValueT> &arr, position_t i)
{
  key_value<KeyT, ValueT> kv;
  position_t position = arr.offset + i;
  kv.key = arr.key_pt[position / arr.block_size][position % arr.block_size];
  kv.value = arr.value_pt[position / arr.block_size][position % arr.block_size];
  return kv;
}

template <class KeyT, class ValueT>
__device__ KeyT getKey(regular_block_key_value<KeyT, ValueT> &arr, position_t i)
{
  position_t position = arr.offset + i;
  return arr.key_pt[position / arr.block_size][position % arr.block_size];
}

template <class KeyT, class ValueT>
KeyT *getKeyPt(regular_block_key_value<KeyT, ValueT> &arr)
{
  position_t position = arr.offset;
  return &arr.key_pt[position / arr.block_size][position % arr.block_size];
}

template <class KeyT, class ValueT>
ValueT *getValuePt(regular_block_key_value<KeyT, ValueT> &arr)
{
  position_t position = arr.offset;
  return &arr.value_pt[position / arr.block_size][position % arr.block_size];
}

template <class KeyT, class ValueT>
__device__ void setElem(regular_block_key_value<KeyT, ValueT> &arr,
			position_t i,
			const key_value<KeyT, ValueT> &kv)
{
  position_t position = arr.offset + i;
  arr.key_pt[position / arr.block_size][position % arr.block_size] = kv.key;
  arr.value_pt[position / arr.block_size][position % arr.block_size] = kv.value;
}

template <class KeyT, class ValueT>
contiguous_key_value<KeyT, ValueT> getBlock
(regular_block_key_value <KeyT, ValueT> &arr, int i_block)
{
  contiguous_key_value<KeyT, ValueT> c_arr;
  c_arr.key_pt = arr.key_pt[i_block];
  c_arr.value_pt = arr.value_pt[i_block];
  c_arr.offset = 0;

  position_t diff = arr.size - i_block*arr.block_size;   
  if (diff <= 0) {
    printf("i_block out of range in getBlock\n");
    exit(0);
  }
  c_arr.size = min(diff, arr.block_size);
  
  return c_arr;
}

/////////////////////////////////////////////////////////////////

template <class ElementT>
struct contiguous_array
{
  ElementT* data_pt;
  position_t offset;
  position_t size;
};

template <class ElementT>
struct regular_block_array
{
  ElementT** h_data_pt;
  ElementT** data_pt;
  position_t block_size;
  position_t offset;
  position_t size;
};

template <class ElementT>
__device__ ElementT getElem(contiguous_array<ElementT> &arr, position_t i)
{
  return *(arr.data_pt + arr.offset + i);
}

template <class ElementT>
__device__ ElementT getKey(contiguous_array<ElementT> &arr, position_t i)
{
  return *(arr.data_pt + arr.offset + i);
}

template <class ElementT>
ElementT *getKeyPt(contiguous_array<ElementT> &arr)
{
  return arr.data_pt + arr.offset;
}

template <class ElementT>
ElementT *getValuePt(contiguous_array<ElementT> &arr)
{
  return NULL;
}

template <class ElementT>
__device__ void setElem(contiguous_array<ElementT> &arr, position_t i,
			const ElementT &val)
{
  *(arr.data_pt + arr.offset + i) = val;
}
////////////////////////////////////////////////////////////

template <class ElementT>
void array_GPUMalloc(void *d_storage, int64_t &st_bytes,
		     contiguous_array<ElementT> &arr, position_t size)
{
  cudaReusableAlloc(d_storage, st_bytes, &(arr.data_pt), size,
		    sizeof(ElementT));
  arr.offset = 0;
  arr.size = size;
}

template <class ElementT>
void array_GPUFree(contiguous_array<ElementT> &arr, position_t size)
{
  CUDAFREECTRL("arr.data_pt",arr.data_pt);
  arr.offset = 0;
  arr.size = 0;
}
template <class ElementT>

void array_Malloc(contiguous_array<ElementT> &arr, position_t size)
{
  arr.data_pt = new ElementT[size];
  arr.offset = 0;
  arr.size = size;
}

template <class ElementT>
void array_Free(contiguous_array<ElementT> &arr, position_t size)
{
  delete[] arr.data_pt;
  arr.offset = 0;
  arr.size = 0;
}

// note: this does not allocate memory fo target array,
// use array_Malloc for that 
template <class ElementT>
void array_GPUtoCPUCopyContent(contiguous_array<ElementT> &target_arr,
			       contiguous_array<ElementT> &source_arr)
{
  gpuErrchk(cudaMemcpy(target_arr.data_pt + target_arr.offset,
		       source_arr.data_pt + source_arr.offset,
		       source_arr.size*sizeof(ElementT),
		       cudaMemcpyDeviceToHost));
  target_arr.size = source_arr.size;
}

// note: this does not allocate memory fo target array,
// use array_GPUMalloc for that 
template <class ElementT>
void array_CPUtoGPUCopyContent(contiguous_array<ElementT> &target_arr,
			       contiguous_array<ElementT> &source_arr)
{
  gpuErrchk(cudaMemcpy(target_arr.data_pt + target_arr.offset,
		       source_arr.data_pt + source_arr.offset,
		       source_arr.size*sizeof(ElementT),
		       cudaMemcpyHostToDevice));
  target_arr.size = source_arr.size;
}

template <class ElementT>
void array_Sort(contiguous_array<ElementT> &arr)
{
  // sort array
  std::sort(arr.data_pt+arr.offset, arr.data_pt+arr.offset+arr.size);
  // extract elements from sorted vector
}

template <class ElementT>
void array_GPUSort(contiguous_array<ElementT> &arr_in,
		   void *d_storage, int64_t &ext_st_bytes)
{
  ext_st_bytes = 0;
  int num_elems = arr_in.size;
  contiguous_array<ElementT> arr_out;
  arr_out.offset = 0;
  arr_out.size = num_elems;
  cudaReusableAlloc(d_storage, ext_st_bytes, &arr_out.data_pt, num_elems,
		    sizeof(ElementT));
  // the following is just for memory alignement
  void *dummy_pt;
  cudaReusableAlloc(d_storage, ext_st_bytes, &dummy_pt, 1, 256);

  size_t sort_storage_bytes = 0;  
  cub::DeviceRadixSort::SortKeys(NULL, sort_storage_bytes,
				 arr_in.data_pt + arr_in.offset,
				 arr_out.data_pt, num_elems);

  if (d_storage != NULL) {
    void *d_sort_storage = (void*)((char*)d_storage + ext_st_bytes);
    cub::DeviceRadixSort::SortKeys(d_sort_storage, sort_storage_bytes,
				   arr_in.data_pt + arr_in.offset,
				   arr_out.data_pt, num_elems);

    gpuErrchk(cudaMemcpy(arr_in.data_pt + arr_in.offset, arr_out.data_pt,
			 num_elems*sizeof(ElementT),
			 cudaMemcpyDeviceToDevice));
  }
  
  ext_st_bytes += sort_storage_bytes;
}

////////////////////////////////////////////////////////

template <class ElementT>
contiguous_array<ElementT> getBlock(regular_block_array<ElementT> &arr,
				    int i_block)
{
  contiguous_array<ElementT> c_arr;
  c_arr.data_pt = arr.data_pt[i_block];
  c_arr.offset = 0;

  position_t diff = arr.size - i_block*arr.block_size;   
  if (diff <= 0) {
    printf("i_block out of range in getBlock\n");
    exit(0);
  }
  c_arr.size = min(diff, arr.block_size);
  
  return c_arr;
}


template <class ElementT>
__device__ ElementT getElem(regular_block_array<ElementT> &arr, position_t i)
{
  position_t position = arr.offset + i;
  return arr.data_pt[position / arr.block_size][position % arr.block_size];
}

template <class ElementT>
__device__ ElementT getKey(regular_block_array<ElementT> &arr, position_t i)
{
  position_t position = arr.offset + i;
  return arr.data_pt[position / arr.block_size][position % arr.block_size];
}

template <class ElementT>
ElementT *getKeyPt(regular_block_array<ElementT> &arr)
{
  position_t position = arr.offset;
  return &arr.data_pt[position / arr.block_size][position % arr.block_size];
}

template <class ElementT>
ElementT *getValuePt(regular_block_array<ElementT> &arr)
{
  return NULL;
}

template <class ElementT>
__device__ void setElem(regular_block_array<ElementT> &arr, position_t i,
			const ElementT &val)
{
  position_t position = arr.offset + i;
  arr.data_pt[position / arr.block_size][position % arr.block_size] = val;
}

//////////////////////////////////////////////////////////////////////

unsigned int nextPowerOf2(unsigned int n);

// atomically set old_index = *arg_max_pt, 
// check whether array[index]>array[old_index].
// If it is true, set *arg_max_pt=index
__device__ int atomicArgMax(position_t *array, int *arg_max_pt, int index);


// find difference between two arrays of type T and specified size
template<class T>
__global__ void diffKernel(T* c, const T* a, const T* b, position_t size)
{
  position_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    c[i] = a[i] - b[i];
  }
}


__global__ void copass_last_step_kernel(position_t *part_size, position_t *m_d,
					uint k, position_t tot_diff,
					position_t *diff,
					position_t *diff_cumul,
					position_t *num_down);

//////////////////////////////////////////////////
__global__ void case2_inc_partitions_kernel(position_t *part_size,
					    int *sorted_extra_elem_idx,
					    position_t tot_diff);


// find the number of elements <= val
// in a sorted array array[i+1]>=array[i]
template <class KeyT, class ArrayT, uint bsize>
__device__ void search_block_up(ArrayT array, position_t size, KeyT val,
				position_t *num_up)
{
  __shared__ KeyT shared_array[bsize+1];
  __shared__ position_t left;
  __shared__ position_t right;

  int tid = threadIdx.x;
  if (size==0 || getKey(array, 0) > val) {
    if (tid == 0) {
      *num_up = 0;
    }
    return;
  }
  else if (getKey(array, size-1) <= val) {
    if (tid == 0) {
      *num_up = size;
    }
    return;
  }

  if (tid == 0) {
    left = 0;
    right = size - 1;
  }
  
  position_t step = size - 1;

  //if (tid == 0) {
  //  printf("bid:%d tid:0 step:%ld size:%ld\n", blockIdx.x, step, size);
  //  printf("arr[n-1]: %d arr[n-2] %d val %d\n", getKey(array, size-1),
  //      getKey(array, size-2), val);
  //}
  __syncthreads();
  while (step>1 && (right-left)>1) {
    position_t pos;
    position_t new_step = (step + blockDim.x - 1) / blockDim.x;
    int n_steps = (int)((step + new_step - 1) / new_step);
    step = new_step;
    if (tid == 0) {
      pos = left;
      shared_array[0] = getKey(array, left);
      shared_array[n_steps] = getKey(array, right);
      //printf("bid:%d tid:0 n_steps:%d sa:%d right:%ld arr:%d step: %ld\n",
      //     blockIdx.x, n_steps, (int)shared_array[n_steps], right,
      //     (int)getKey(array, right), step);
    }
    else if (tid < n_steps) {
      pos = left + step*tid;
      if ((right-pos) >= 1) {
	shared_array[tid] = getKey(array, pos);
	//printf("bid:%d tid:%ld sa:%ld pos:%ld arr:%ld\n", blockIdx.x, tid,
	//           shared_array[tid], pos, array[pos]);
      }
    }
    __syncthreads();
    if ((tid < n_steps) && ((right-pos) >= 1)
	&& (shared_array[tid] <= val)
	&& (shared_array[tid+1] > val)) {
      left = pos;
      right = min(pos + step, right);
      //printf("bid:%d good tid:%d sa0:%d sa1:%d l:%ld r:%ld\n", blockIdx.x,
      //     tid, (int)shared_array[tid], (int)shared_array[tid+1],
      //     left, right);
    }
    __syncthreads();
  }

  if (threadIdx.x==0) {
    *num_up = right;
    //printf("Kernel block: %ld\tnum_up: %ld\n", blockIdx.x, right);
    //printf("bid: %ld\tleft: %ld\tright: %ld\n", blockIdx.x, left, right);
  }
}

template <class KeyT, class ArrayT, uint bsize>
__global__ void search_multi_up_kernel(ArrayT *subarray,
				       KeyT *val_pt, position_t *num_up,
				       position_t *sum_num_up)
{
  int bid = blockIdx.x;
  KeyT val = *val_pt;
  search_block_up<KeyT, ArrayT, bsize>
    (subarray[bid], subarray[bid].size, val, &num_up[bid]);
  if (threadIdx.x==0) {
    atomicAdd((uposition_t*)sum_num_up, num_up[bid]);
    //printf("bid: %ld\tm_d: %ld\n", blockIdx.x, num_up[bid]);
  }
}

// find the number of elements < val
// in a sorted array array[i+1]>=array[i]
template <class KeyT, class ArrayT, uint bsize>
__device__ void search_block_down(ArrayT array, position_t size, KeyT val,
				  position_t *num_down)
{
  __shared__ KeyT shared_array[bsize+1];
  __shared__ position_t left;
  __shared__ position_t right;

  int tid = threadIdx.x;
  if (size==0 || getKey(array, 0) >= val) {
    if (tid == 0) {
      *num_down = 0;
    }
    return;
  }
  else if (getKey(array, size-1) < val) {
    if (tid == 0) {
      *num_down = size;
    }
    return;
  }

  if (tid == 0) {
    left = 0;
    right = size - 1;
  }
  
  position_t step = size - 1;

  //if (tid == 0) {
  //  printf("bid:%d tid:0 step:%ld size:%ld\n", blockIdx.x, step, size);
  //  printf("arr[n-1]: %d arr[n-2] %d val %d\n", getKey(array, size-1),
  //	   getKey(array, size-2), val);
  //}
  __syncthreads();
  while(step>1 && (right-left)>1) {
    position_t pos;
    position_t new_step = (step + blockDim.x - 1) / blockDim.x;
    int n_steps = (int)((step + new_step - 1) / new_step);
    step = new_step;
    if (tid == 0) {
      pos = left;
      shared_array[0] = getKey(array, left);
      shared_array[n_steps] = getKey(array, right);
      //printf("bid:%d tid:0 n_steps:%d sa:%d right:%ld arr:%d step: %ld\n",
      //     blockIdx.x, n_steps, (int)shared_array[n_steps], right,
      //     (int)getKey(array, right), step);
    }
    else if (tid < n_steps) {
      pos = left + step*tid;
      if ((right-pos) >= 1) {
	shared_array[tid] = getKey(array, pos);
	//printf("bid:%d tid:%ld sa:%ld pos:%ld arr:%ld\n", blockIdx.x, tid,
	//	     shared_array[tid], pos, array[pos]);
      }
    }
    __syncthreads();
    if ((tid < n_steps) && ((right-pos) >= 1)
	&& (shared_array[tid] < val)
	&& (shared_array[tid+1] >= val)) {
      left = pos;
      right = min(pos + step, right);
      //printf("bid:%d good tid:%d sa0:%d sa1:%d l:%ld r:%ld\n", blockIdx.x,
      //     tid, (int)shared_array[tid], (int)shared_array[tid+1],
      //     left, right);
    }
    __syncthreads();
  }

  if (threadIdx.x==0) {
    *num_down = right;
    //printf("Kernel block: %ld\tnum_up: %ld\n", blockIdx.x, right);
    //printf("bid: %ld\tleft: %ld\tright: %ld\n", blockIdx.x, left, right);
  }
}

template <class KeyT, class ArrayT, uint bsize>
__global__ void search_multi_down_kernel(ArrayT *subarray,
					 KeyT *val_pt, position_t *num_down,
					 position_t *sum_num_down)
{
  int bid = blockIdx.x;
  KeyT val = *val_pt;
  search_block_down<KeyT, ArrayT, bsize>
    (subarray[bid], subarray[bid].size, val, &num_down[bid]);
  if (threadIdx.x==0) {
    atomicAdd((uposition_t*)sum_num_down, num_down[bid]);
    //printf("bid: %ld\tm_u: %ld\n", blockIdx.x, num_down[bid]);
  }
}

////////////////////////////////////////////////////////////
// find the maximum of m_u[i] - m_d[i], i=0,...,size-1
template <class ArrayT, uint bsize>
__global__ void max_diff_kernel(position_t *m_u, position_t *m_d,
				position_t size, ArrayT* subarray,
				position_t *max_diff,
				int *arg_max)
{
  __shared__ position_t diff_array[bsize];
  __shared__ int shared_arg_max;

  int i = threadIdx.x;
  if (i >= size) return;
  position_t sub_size = subarray[i].size;
  if (i == 0) {
    shared_arg_max = 0; // index of maximum difference
    if (sub_size <= 0) {
      diff_array[0] = -1;
    }
  }

  if (sub_size > 0) {
    diff_array[i] = m_u[i] - m_d[i];
  }
  __syncthreads();
  if (sub_size > 0) {
#ifdef PRINT_VRB
    printf("diff i: %d m_u:%ld m_d:%ld diff_array:%ld\n", i, m_u[i], m_d[i],
	   diff_array[i]);
#endif
    atomicArgMax(diff_array, &shared_arg_max, i);
  }
  __syncthreads();
  
  if (i == 0) {
    *max_diff = diff_array[shared_arg_max];
    *arg_max = shared_arg_max;
#ifdef PRINT_VRB
    printf("Kernel max_diff: %ld\targ_max: %d\n", *max_diff, *arg_max);
#endif
  }
}  


// check array element type, maybe replace with position_t
template <class ElementT, int bsize>
__global__ void prefix_scan(ElementT *array_in, ElementT *array_out,
			    uint k, uint n)
{
  __shared__ ElementT shared_arr[bsize];
  int tid = threadIdx.x;
  
  if (2*tid+1 >= 2*n) return;
  
  int offset = 1; 

   // copy input array to shared memory
  if (2*tid < k) {
    shared_arr[2*tid] = array_in[2*tid];
  }
  else {
    shared_arr[2*tid] = 0;
  }
  if ((2*tid+1) < k) {
    shared_arr[2*tid+1] = array_in[2*tid+1];
  }
  else {
    shared_arr[2*tid+1] = 0;
  }
  
  for (int d=n>>1; d>0; d>>=1) {
    __syncthreads();
    if (tid < d)    { 
      int a = offset*(2*tid+1)-1;
      int b = offset*(2*tid+2)-1;
      shared_arr[b] += shared_arr[a];
    }
    offset *= 2;
  } 
  if (tid == 0) {
    shared_arr[n - 1] = 0;
  }
  
  for (int d=1; d<n; d*=2) {
    offset >>= 1;
    __syncthreads();
    if (tid < d)      { 
      int a = offset*(2*tid+1)-1;
      int b = offset*(2*tid+2)-1; 
      ElementT t = shared_arr[a];
      shared_arr[a] = shared_arr[b];
      shared_arr[b] += t;
    }
  }
  __syncthreads(); 
   if (2*tid < k+1) {
    array_out[2*tid] = shared_arr[2*tid];
  }
  if (2*tid < k) {
    array_out[2*tid+1] = shared_arr[2*tid+1];
  }
} 



// trova num. di elementi dell'array < val
// in un array ordinato array[i+1]>=array[i]
template <class ElementT, uint bsize>
__global__ void search_down(ElementT *array, position_t size,
			    ElementT val, position_t *num_down)
{
  contiguous_array<ElementT> arr;
  arr.data_pt = array;
  arr.offset = 0;
  search_block_down<ElementT, contiguous_array<ElementT>, bsize>
    (arr, size, val, num_down);
}

// trova num. di elementi dell'array <= val
// in un array ordinato array[i+1]>=array[i]
template <class ElementT, uint bsize>
__global__ void search_up(ElementT *array, position_t size,
			    ElementT val, position_t *num_up)
{
  contiguous_array<ElementT> arr;
  arr.data_pt = array;
  arr.offset = 0;
  search_block_up<ElementT, contiguous_array<ElementT>, bsize>
    (arr, size, val, num_up);
}


template <class KeyT, class ArrayT, uint bsize>
int search_multi_up(ArrayT *d_subarray, uint k,
		    KeyT *d_val_pt, position_t *d_num_up,
		    position_t *d_sum_num_up)
{
  gpuErrchk(cudaMemsetAsync(d_sum_num_up, 0, sizeof(position_t)));
  search_multi_up_kernel <KeyT, ArrayT, bsize> <<<k, bsize>>>
    (d_subarray, d_val_pt, d_num_up, d_sum_num_up);
  DBGCUDASYNC
  
  return 0;
}

template <class KeyT, class ArrayT, uint bsize>
int search_multi_down(ArrayT *d_subarray,
		      uint k, KeyT *d_val_pt, position_t *d_num_down,
		      position_t *d_sum_num_down)
{
  gpuErrchk(cudaMemsetAsync(d_sum_num_down, 0, sizeof(position_t)));
  search_multi_down_kernel <KeyT, ArrayT, bsize> <<<k, bsize>>>
    (d_subarray, d_val_pt, d_num_down, d_sum_num_down);

  DBGCUDASYNC
  
  return 0;
}

// atomically set old_index = *arg_max_pt, 
// check whether array[index]>array[old_index].
// If it is true, set *arg_max_pt=index
template<class KeyT>
__device__ int atomicKeyArgMax(KeyT *array, int *arg_max_pt, int index)
{
  int old_index = *arg_max_pt;
  int assumed_index;
  do {
    if (old_index>=0 && array[old_index]>=array[index]) {
      break;
    }
    assumed_index = old_index;
    old_index = atomicCAS(arg_max_pt, assumed_index, index);
  } while (assumed_index != old_index);
  
  return old_index;
}

// atomically set old_index = *arg_min_pt, 
// check whether array[index]<array[old_index].
// If it is true, set *arg_min_pt=index
template<class KeyT>
__device__ int atomicKeyArgMin(KeyT *array, int *arg_min_pt, int index)
{
  int old_index = *arg_min_pt;
  int assumed_index;
  do {
    if (old_index>=0 && array[old_index]<=array[index]) {
      break;
    }
    assumed_index = old_index;
    old_index = atomicCAS(arg_min_pt, assumed_index, index);
  } while (assumed_index != old_index);
  
  return old_index;
}


template<class KeyT, class ArrayT, uint bsize>
__global__ void threshold_range_kernel(ArrayT* subarray,
				       position_t tot_part_size,
				       uint k, KeyT *t_u, KeyT *t_d)
{
  __shared__ KeyT shared_t_u[bsize];
  __shared__ KeyT shared_t_d[bsize];
  __shared__ int shared_arg_max;
  __shared__ int shared_arg_min;

  if (threadIdx.x==0) {
    shared_arg_max = -1;
    shared_arg_min = -1;
  }
  __syncthreads();

#ifdef PRINT_VRB
  bool print_vrb = (threadIdx.x==0);
#endif
  int i=threadIdx.x;

  position_t sub_size;
  if (i < k) {
    //printf("i: %d\t sa pt: %lld\n", i, (long long int)subarray[i].data_pt);
    sub_size = subarray[i].size;
    if (sub_size > 0) {
      position_t m0_u = tot_part_size;
      // (tot_part_size + k -2) / (k-1); // ceil (tot_part_size / k)
      position_t m0_d = tot_part_size / k; // floor (tot_part_size / k)
#ifdef PRINT_VRB
      if (print_vrb) printf("tot_part_size: %ld\n", tot_part_size);
      if (print_vrb) printf("m0_u: %ld\tm0_d: %ld\n", m0_u, m0_d);
#endif
      // find the maximum of subarray[i][m_u]
      // and the minimum of  subarray[i][m_d]
      
      // if the indexes are out of range put them in range
      position_t m1_u = min(m0_u, sub_size);
      position_t m1_d = min(m0_d, sub_size);
      m1_u = max(m1_u - 1, (position_t)0);
      m1_d = max(m1_d - 1, (position_t)0);
#ifdef PRINT_VRB
      printf("i: %d\tm1_u: %ld\tm1_d: %ld\tsubarray_size: %ld\n", i,
      	     m1_u, m1_d, sub_size);
#endif
      // update upper and lower limit of threshold range      
      shared_t_u[i] = getKey(subarray[i], m1_u);
      shared_t_d[i] = getKey(subarray[i], m1_d);
#ifdef PRINT_VRB
      printf("i: %d\tshared_t_u: %d\tshared_t_d: %d\n", i, shared_t_u[i],
	     shared_t_d[i]);
#endif
    }
  }
#ifdef PRINT_VRB
  __syncthreads();
  if (i==0) {
    for (int j=0; j<k; j++) {
      printf("j: %d\tshared_t_u: %d\tshared_t_d: %d\n", j, shared_t_u[j],
	     shared_t_d[j]);
    }
  }
#endif
  __syncthreads();
  ///// creare template di atomicKeyArgMax per tipi generici a 32 e 64 bit
  //// usare anche il verso (ascending/descending) e la fz di confronto
  //// isBefore, isAfter, isNotAfter,...
  if (i < k && sub_size > 0) {
    atomicKeyArgMax(shared_t_u, &shared_arg_max, i);
    atomicKeyArgMin(shared_t_d, &shared_arg_min, i);
#ifdef PRINT_VRB
    printf("i: %d\tshared_t_u: %d\tshared_arg_max: %d\n", i, shared_t_u[i],
	   shared_arg_max);
    printf("i: %d\tshared_t_d: %d\tshared_arg_min: %d\n", i, shared_t_u[i],
	   shared_arg_max);
#endif
  }
  __syncthreads();
  
  if (threadIdx.x==0) {
    *t_u = shared_t_u[shared_arg_max];
    *t_d = shared_t_d[shared_arg_min];
#ifdef PRINT_VRB
    printf("Kernel t_u: %d\tt_d: %d\n", *t_u, *t_d);
#endif
  }
}  

template <class KeyT, class ArrayT>
__global__ void eval_t_tilde_kernel(ArrayT *subarray,
				    position_t *m_u, position_t *m_d,
				    int *arg_max, KeyT *t_tilde)
{
  int i = *arg_max;
  int m_tilde = (m_u[i] + m_d[i])/2;
  m_tilde = max(m_tilde - 1, 0);
  *t_tilde = getKey(subarray[i], m_tilde);
  //printf("m_tilde: %d\t *t_tilde: %d\n", m_tilde, *t_tilde);
}

template <class KeyT, class ArrayT>
__global__ void case2_extra_elems_kernel(ArrayT *subarray,
					 uint k, position_t *m_d,
					 position_t *m_u,
					 KeyT *extra_elem,
					 int *extra_elem_idx,
					 int *n_extra_elems)
{
  int i = threadIdx.x;
  if (i == 0) {
    *n_extra_elems = 0;
  }
  __syncthreads();
  
  if (i >= k) return;
  int sub_size = (int)subarray[i].size;
  if (sub_size <= 0) return;
  
  if (m_u[i] > m_d[i]) {
    int i_elem = atomicAdd(n_extra_elems, 1);
    extra_elem[i_elem] = getKey(subarray[i], m_d[i]);
    extra_elem_idx[i_elem] = i;
  }
}


template <class ElementT, class ArrayT, class AuxArrayT>
__global__ void extract_partitions_kernel(ArrayT *subarray,
					  uint k, position_t *part_size,
					  position_t *part_size_cumul,
					  AuxArrayT aux_array)
{
  const int i_arr = blockIdx.x;
  position_t size_i_arr = part_size[i_arr];
  position_t i_aux_offset = part_size_cumul[i_arr];
  
  for (position_t i_elem = threadIdx.x; i_elem < size_i_arr;
       i_elem += blockDim.x) {
    position_t i_aux = i_aux_offset + i_elem;
    ElementT elem = getElem(subarray[i_arr], i_elem);
    setElem(aux_array, i_aux, elem);
  }
}

template <class ElementT, class TargetArray, class SourceArray> 
void __global__ CopyArray(TargetArray target_arr, SourceArray source_arr)
{
  position_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= source_arr.size) return;
  
  ElementT elem = getElem(source_arr, i);
  setElem(target_arr, i, elem);
}

template <class KeyT>
void contiguousTranslate(contiguous_array<KeyT> &arr,
			 position_t transl, char *d_buffer,
			 position_t buffer_size)
{
  position_t elem_num = arr.size;
  position_t s_pos0 = arr.offset;
  position_t t_pos0 = arr.offset + transl;
  
  if (transl>=elem_num) {
    gpuErrchk(cudaMemcpyAsync(&arr.data_pt[t_pos0],
			      &arr.data_pt[s_pos0],
			      elem_num*sizeof(KeyT), cudaMemcpyDeviceToDevice));
  }
  else {
    GPUMemCpyBuffered((char*)&arr.data_pt[t_pos0],
		      (char*)&arr.data_pt[s_pos0],
		      elem_num*sizeof(KeyT), d_buffer, buffer_size);
  }
  arr.offset += transl;
}

template <class KeyT, class ValueT>
void contiguousTranslate(contiguous_key_value<KeyT, ValueT> &arr,
			 position_t transl, char *d_buffer,
			 position_t buffer_size)
{
  position_t elem_num = arr.size;
  position_t s_pos0 = arr.offset;
  position_t t_pos0 = arr.offset + transl;
  
  if (transl>=elem_num) {
    gpuErrchk(cudaMemcpyAsync(&arr.key_pt[t_pos0],
			      &arr.key_pt[s_pos0],
			      elem_num*sizeof(KeyT), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpyAsync(&arr.value_pt[t_pos0],
			      &arr.value_pt[s_pos0],
			      elem_num*sizeof(ValueT),
			      cudaMemcpyDeviceToDevice));
  }
  else {
    GPUMemCpyBuffered((char*)&arr.key_pt[t_pos0],
		      (char*)&arr.key_pt[s_pos0],
		      elem_num*sizeof(KeyT), d_buffer, buffer_size);
    GPUMemCpyBuffered((char*)&arr.value_pt[t_pos0],
		      (char*)&arr.value_pt[s_pos0],
		      elem_num*sizeof(ValueT), d_buffer, buffer_size);
  }
  arr.offset += transl;
}

template <class KeyT, class ValueT>
void CopyRegion(regular_block_key_value<KeyT, ValueT> &arr,
		int t_ib, position_t t_j0, int s_ib, position_t s_j0,
		position_t elem_num, char *d_buffer,
		position_t buffer_size) {

  position_t transl = t_j0 - s_j0;
  if (t_ib != s_ib || transl>=elem_num) {
    gpuErrchk(cudaMemcpyAsync(&arr.h_key_pt[t_ib][t_j0],
			      &arr.h_key_pt[s_ib][s_j0],
			      elem_num*sizeof(KeyT), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpyAsync(&arr.h_value_pt[t_ib][t_j0],
			      &arr.h_value_pt[s_ib][s_j0],
			      elem_num*sizeof(ValueT),
			      cudaMemcpyDeviceToDevice));
  }
  else {
    GPUMemCpyBuffered((char*)&arr.h_key_pt[t_ib][t_j0],
		    (char*)&arr.h_key_pt[s_ib][s_j0],
		    elem_num*sizeof(KeyT), d_buffer, buffer_size);
    GPUMemCpyBuffered((char*)&arr.h_value_pt[t_ib][t_j0],
		    (char*)&arr.h_value_pt[s_ib][s_j0],
		    elem_num*sizeof(ValueT), d_buffer, buffer_size);
  }
}

template <class KeyT>
void CopyRegion(regular_block_array<KeyT> &arr,
		int t_ib, position_t t_j0, int s_ib, position_t s_j0,
		position_t elem_num, char *d_buffer,
		position_t buffer_size) {

  position_t transl = t_j0 - s_j0;
  if (t_ib != s_ib || transl>=elem_num) {
    gpuErrchk(cudaMemcpyAsync(&arr.h_data_pt[t_ib][t_j0],
			      &arr.h_data_pt[s_ib][s_j0],
			      elem_num*sizeof(KeyT), cudaMemcpyDeviceToDevice));
  }
  else {
    GPUMemCpyBuffered((char*)&arr.h_data_pt[t_ib][t_j0],
		    (char*)&arr.h_data_pt[s_ib][s_j0],
		    elem_num*sizeof(KeyT), d_buffer, buffer_size);
  }
}


template <class ArrayT>
void regularBlockTranslate(ArrayT &arr,
			   position_t transl, char *d_buffer,
			   position_t buffer_size)
{
  position_t elem_num = arr.size;
  position_t s_pos1 = arr.offset + elem_num - 1;
  int s_ib1 = (int)(s_pos1 / arr.block_size);
  position_t s_j1 = s_pos1 % arr.block_size;

  position_t t_pos1 = arr.offset + transl + elem_num - 1;
  int t_ib1 = (int)(t_pos1 / arr.block_size);
  position_t t_j1 = t_pos1 % arr.block_size;
  
  position_t s_num1 = s_j1 + 1;
  position_t t_num1 = t_j1 + 1;

  if (t_num1<elem_num && t_num1<s_num1) {
    CopyRegion(arr, t_ib1, 0, s_ib1, s_num1-t_num1, t_num1, d_buffer,
	       buffer_size);
    elem_num -= t_num1;
    s_num1 -= t_num1;
    t_num1 = arr.block_size;
    t_ib1--;
    
    if (s_num1<elem_num) {
      CopyRegion(arr, t_ib1, t_num1-s_num1, s_ib1, 0, s_num1, d_buffer,
		 buffer_size);
      elem_num -= s_num1;
      t_num1 -= s_num1;
      s_num1 = arr.block_size;
      s_ib1--;
    }
  }
  else if (s_num1<elem_num) { // && s_num1<t_num1) {
    CopyRegion(arr, t_ib1, t_num1-s_num1, s_ib1, 0, s_num1, d_buffer,
	       buffer_size);
    elem_num -= s_num1;
    t_num1 -= s_num1;
    s_num1 = arr.block_size;
    s_ib1--;
    
    if (t_num1<elem_num) {
      CopyRegion(arr, t_ib1, 0, s_ib1, s_num1-t_num1, t_num1, d_buffer,
		 buffer_size);
      elem_num -= t_num1;
      s_num1 -= t_num1;
      t_num1 = arr.block_size;
      t_ib1--;
    }
  }
  CopyRegion(arr, t_ib1, t_num1-elem_num, s_ib1, s_num1-elem_num, elem_num,
	     d_buffer, buffer_size);
  arr.offset += transl;
}

template <class KeyT>
void Translate(contiguous_array<KeyT> &arr,
	       position_t transl, char *d_buffer, position_t buffer_size)
{
  contiguousTranslate(arr, transl, d_buffer, buffer_size);
}

template <class KeyT, class ValueT>
void Translate(contiguous_key_value<KeyT, ValueT> &arr,
	       position_t transl, char *d_buffer, position_t buffer_size)
{
  contiguousTranslate(arr, transl, d_buffer, buffer_size);
}

template <class KeyT>
void Translate(regular_block_array<KeyT> &arr,
	       position_t transl, char *d_buffer, position_t buffer_size)
{
  regularBlockTranslate(arr, transl, d_buffer, buffer_size);
}

template <class KeyT, class ValueT>
void Translate(regular_block_key_value<KeyT, ValueT> &arr,
	       position_t transl, char *d_buffer, position_t buffer_size)
{
  regularBlockTranslate(arr, transl, d_buffer, buffer_size);
}

template <class ArrayT>
void repack(ArrayT *h_subarray,
	    uint k, position_t *part_size, char *d_buffer,
	    position_t buffer_size)
{
  position_t psize = part_size[k-1];
  h_subarray[k-1].offset += psize;
  h_subarray[k-1].size -= psize;

  position_t transl = psize; // translation of last subarray
                             // to be updated for each subarray

  // move blocks of memory to the right in reverse order
  for (int i_arr=k-2; i_arr>=0; i_arr--) {
    position_t sub_size = h_subarray[i_arr].size;
    if (sub_size <= 0) continue;
      
    psize = part_size[i_arr];
    h_subarray[i_arr].offset += psize;
    h_subarray[i_arr].size -= psize;
    Translate(h_subarray[i_arr], transl, d_buffer, buffer_size);
    transl += psize;
  }  
}


#endif
