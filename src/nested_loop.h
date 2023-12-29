/*
 *  nested_loop.h
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

#ifndef NESTED_LOOP_H
#define NESTED_LOOP_H

#include <cub/cub.cuh>
#include "cuda_error.h"
#include "prefix_scan.h"
#include "get_spike.h"
#include "rev_spike.h"


extern const int Ny_arr_size_;
extern int Ny_th_arr_[];

enum NestedLoopAlgo {
  BlockStepNestedLoopAlgo,
  CumulSumNestedLoopAlgo,
  SimpleNestedLoopAlgo,
  ParallelInnerNestedLoopAlgo,
  ParallelOuterNestedLoopAlgo,
  Frame1DNestedLoopAlgo,
  Frame2DNestedLoopAlgo,
  Smart1DNestedLoopAlgo,
  Smart2DNestedLoopAlgo
};

template<class ConnKeyT, class ConnStructT>
__device__ __forceinline__ void NestedLoopFunction(int i_func, int ix, int iy)
{
  switch (i_func) {
  case 0:
    NestedLoopFunction0<ConnKeyT, ConnStructT>(ix, iy);
    break;
  case 1:
    NestedLoopFunction1<ConnKeyT, ConnStructT>(ix, iy);
    break;
  }
}

//////////////////////////////////////////////////////////////////////
template<class ConnKeyT, class ConnStructT> 
__global__ void BlockStepNestedLoopKernel(int i_func, int Nx, int *Ny)
{
 const int ix = blockIdx.x;
  if (ix < Nx) {
    const int ny = Ny[ix];
    for (int iy = threadIdx.x; iy < ny; iy += blockDim.x){
      NestedLoopFunction<ConnKeyT, ConnStructT>(i_func, ix, iy);
    }
  }
}

template<class ConnKeyT, class ConnStructT> 
__global__ void CumulSumNestedLoopKernel(int i_func, int Nx, int *Ny_cumul_sum,
					 int Ny_sum)
{
  int blockId   = blockIdx.y * gridDim.x + blockIdx.x;
  int array_idx = blockId * blockDim.x + threadIdx.x;
  if (array_idx<Ny_sum) {
    int ix = locate(array_idx, Ny_cumul_sum, Nx + 1);
    int iy = (int)(array_idx - Ny_cumul_sum[ix]);
    NestedLoopFunction<ConnKeyT, ConnStructT>(i_func, ix, iy);
  }
}

//////////////////////////////////////////////////////////////////////
template<class ConnKeyT, class ConnStructT> 
__global__ void SimpleNestedLoopKernel(int i_func, int Nx, int *Ny)
{
  int ix = (blockIdx.x * blockDim.x) + threadIdx.x;
  int iy = (blockIdx.y * blockDim.y) + threadIdx.y;
  if (ix<Nx && iy<Ny[ix]) {
    NestedLoopFunction<ConnKeyT, ConnStructT>(i_func, ix, iy);
  }
}

//////////////////////////////////////////////////////////////////////
template<class ConnKeyT, class ConnStructT> 
__global__ void  ParallelInnerNestedLoopKernel(int i_func, int ix, int Ny)
{
  int iy = threadIdx.x + blockIdx.x * blockDim.x;
  if (iy<Ny) {
    NestedLoopFunction<ConnKeyT, ConnStructT>(i_func, ix, iy);
  }
}

//////////////////////////////////////////////////////////////////////
template<class ConnKeyT, class ConnStructT> 
__global__ void  ParallelOuterNestedLoopKernel(int i_func, int Nx, int *d_Ny)
{
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  if (ix<Nx) {
    for (int iy=0; iy<d_Ny[ix]; iy++) {
      NestedLoopFunction<ConnKeyT, ConnStructT>(i_func, ix, iy);
    }
  }
}


//////////////////////////////////////////////////////////////////////
template<class ConnKeyT, class ConnStructT> 
__global__ void Frame1DNestedLoopKernel(int i_func, int ix0,
					int dim_x, int dim_y,
					int *sorted_idx, int *sorted_Ny)
{
  int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (array_idx<dim_x*dim_y) {
    int ix = ix0 + array_idx % dim_x;
    int iy = array_idx / dim_x;
    if (iy<sorted_Ny[ix]) {
      // call here the function that should be called by the nested loop
      NestedLoopFunction<ConnKeyT, ConnStructT>(i_func, sorted_idx[ix], iy);
    }
  }
}

//////////////////////////////////////////////////////////////////////
template<class ConnKeyT, class ConnStructT> 
__global__ void Frame2DNestedLoopKernel(int i_func, int ix0,
					int dim_x, int dim_y,
					int *sorted_idx, int *sorted_Ny)
{
  int ix = (blockIdx.x * blockDim.x) + threadIdx.x;
  int iy = (blockIdx.y * blockDim.y) + threadIdx.y;
  if (ix<dim_x && iy<sorted_Ny[ix+ix0]) {
    // call here the function that should be called by the nested loop
    NestedLoopFunction<ConnKeyT, ConnStructT>(i_func, sorted_idx[ix+ix0], iy);
  }
}

//////////////////////////////////////////////////////////////////////
template<class ConnKeyT, class ConnStructT> 
__global__ void Smart1DNestedLoopKernel(int i_func, int ix0, int iy0,
					int dim_x, int dim_y,
					int *sorted_idx, int *sorted_Ny)
{
  int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (array_idx<dim_x*dim_y) {
    int ix = ix0 + array_idx % dim_x;
    int iy = iy0 + array_idx / dim_x;
    if (iy<sorted_Ny[ix]) {
      // call here the function that should be called by the nested loop
      NestedLoopFunction<ConnKeyT, ConnStructT>(i_func, sorted_idx[ix], iy);
    }
  }
}

//////////////////////////////////////////////////////////////////////
template<class ConnKeyT, class ConnStructT> 
__global__ void Smart2DNestedLoopKernel(int i_func, int ix0, int iy0,
					int dim_x, int dim_y, int *sorted_idx,
					int *sorted_Ny)
{
  int ix = (blockIdx.x * blockDim.x) + threadIdx.x;
  int iy = iy0 + (blockIdx.y * blockDim.y) + threadIdx.y;
  if (ix<dim_x && iy<sorted_Ny[ix+ix0]) {
    // call here the function that should be called by the nested loop
    NestedLoopFunction<ConnKeyT, ConnStructT>(i_func, sorted_idx[ix+ix0], iy);
  }
}





namespace NestedLoop
{
  extern void *d_sort_storage_;
  extern size_t sort_storage_bytes_;
  extern void *d_reduce_storage_;
  extern size_t reduce_storage_bytes_;

  extern int Nx_max_;
  extern int *d_max_Ny_;
  extern int *d_sorted_Ny_;

  extern int *d_idx_;
  extern int *d_sorted_idx_;

  extern int block_dim_x_;
  extern int block_dim_y_;
  extern int frame_area_;
  extern float x_lim_;
  
  extern int *d_Ny_cumul_sum_;

  extern PrefixScan prefix_scan_;
  
  int Init();

  int Init(int Nx_max);

  template<class ConnKeyT, class ConnStructT>
  int Run(int nested_loop_algo, int i_func, int Nx, int *d_Ny);

  template<class ConnKeyT, class ConnStructT>
  int BlockStepNestedLoop(int i_func, int Nx, int *d_Ny);
  
  template<class ConnKeyT, class ConnStructT>
  int CumulSumNestedLoop(int i_func, int Nx, int *d_Ny);  

  template<class ConnKeyT, class ConnStructT>
  int SimpleNestedLoop(int i_func, int Nx, int *d_Ny);

  template<class ConnKeyT, class ConnStructT>
  int SimpleNestedLoop(int i_func, int Nx, int *d_Ny, int max_Ny);

  template<class ConnKeyT, class ConnStructT>
  int ParallelInnerNestedLoop(int i_func, int Nx, int *d_Ny);

  template<class ConnKeyT, class ConnStructT>
  int ParallelOuterNestedLoop(int i_func, int Nx, int *d_Ny);

  template<class ConnKeyT, class ConnStructT>
  int Frame1DNestedLoop(int i_func, int Nx, int *d_Ny);

  template<class ConnKeyT, class ConnStructT>
  int Frame2DNestedLoop(int i_func, int Nx, int *d_Ny);

  template<class ConnKeyT, class ConnStructT>
  int Smart1DNestedLoop(int i_func, int Nx, int *d_Ny);

  template<class ConnKeyT, class ConnStructT>
  int Smart2DNestedLoop(int i_func, int Nx, int *d_Ny);

  int Free();
}

//////////////////////////////////////////////////////////////////////
template<class ConnKeyT, class ConnStructT>
int NestedLoop::Run(int nested_loop_algo, int i_func, int Nx, int *d_Ny)
{
  switch(nested_loop_algo) {
  case BlockStepNestedLoopAlgo:
    return BlockStepNestedLoop<ConnKeyT, ConnStructT>(i_func, Nx, d_Ny);
    break;
  case CumulSumNestedLoopAlgo:
    return CumulSumNestedLoop<ConnKeyT, ConnStructT>(i_func, Nx, d_Ny);
    break;
  case SimpleNestedLoopAlgo:
    return SimpleNestedLoop<ConnKeyT, ConnStructT>(i_func, Nx, d_Ny);
    break;
  case ParallelInnerNestedLoopAlgo:
    return ParallelInnerNestedLoop<ConnKeyT, ConnStructT>(i_func, Nx, d_Ny);
    break;
  case ParallelOuterNestedLoopAlgo:
    return ParallelOuterNestedLoop<ConnKeyT, ConnStructT>(i_func, Nx, d_Ny);
    break;
  case Frame1DNestedLoopAlgo:
    return Frame1DNestedLoop<ConnKeyT, ConnStructT>(i_func, Nx, d_Ny);
    break;
  case Frame2DNestedLoopAlgo:
    return Frame2DNestedLoop<ConnKeyT, ConnStructT>(i_func, Nx, d_Ny);
    break;
  case Smart1DNestedLoopAlgo:
    return Smart1DNestedLoop<ConnKeyT, ConnStructT>(i_func, Nx, d_Ny);
    break;
  case Smart2DNestedLoopAlgo:
    return Smart2DNestedLoop<ConnKeyT, ConnStructT>(i_func, Nx, d_Ny);
    break;
  default:
    return -1;
  }
}


//////////////////////////////////////////////////////////////////////
template<class ConnKeyT, class ConnStructT>
int NestedLoop::BlockStepNestedLoop(int i_func, int Nx, int *d_Ny)
{
  BlockStepNestedLoopKernel<ConnKeyT, ConnStructT><<<Nx, 1024>>>
    (i_func, Nx, d_Ny);
  gpuErrchk(cudaPeekAtLastError());
  //gpuErrchk(cudaDeviceSynchronize());
  
  return 0;
}

//////////////////////////////////////////////////////////////////////
template<class ConnKeyT, class ConnStructT>
int NestedLoop::SimpleNestedLoop(int i_func, int Nx, int *d_Ny)
{
  // Find max value of Ny
  cub::DeviceReduce::Max(d_reduce_storage_, reduce_storage_bytes_, d_Ny,
			 d_max_Ny_, Nx);
  int max_Ny;
  gpuErrchk(cudaMemcpy(&max_Ny, d_max_Ny_, sizeof(int),
			  cudaMemcpyDeviceToHost));
  return SimpleNestedLoop<ConnKeyT, ConnStructT>(i_func, Nx, d_Ny, max_Ny);
}

//////////////////////////////////////////////////////////////////////
template<class ConnKeyT, class ConnStructT>
int NestedLoop::SimpleNestedLoop(int i_func, int Nx, int *d_Ny, int max_Ny)
{
  if (max_Ny < 1) max_Ny = 1;
  dim3 threadsPerBlock(block_dim_x_, block_dim_y_);  // block size
  dim3 numBlocks((Nx - 1)/threadsPerBlock.x + 1,
		 (max_Ny - 1)/threadsPerBlock.y + 1);
  SimpleNestedLoopKernel<ConnKeyT, ConnStructT> <<<numBlocks,threadsPerBlock>>>
    (i_func, Nx, d_Ny);
  gpuErrchk(cudaPeekAtLastError());
  //gpuErrchk(cudaDeviceSynchronize());

  return 0;
}

//////////////////////////////////////////////////////////////////////
template<class ConnKeyT, class ConnStructT>
int NestedLoop::ParallelInnerNestedLoop(int i_func, int Nx, int *d_Ny)
{
  int h_Ny[Nx];
  gpuErrchk(cudaMemcpy(h_Ny, d_Ny, Nx*sizeof(int),
		       cudaMemcpyDeviceToHost));
  for (int ix=0; ix<Nx; ix++) {
    int Ny = h_Ny[ix];
    ParallelInnerNestedLoopKernel<ConnKeyT, ConnStructT>
      <<<(Ny+1023)/1024, 1024>>>(i_func, ix, Ny);
    // gpuErrchk(cudaPeekAtLastError()); // uncomment only for debugging
    // gpuErrchk(cudaDeviceSynchronize()); // uncomment only for debugging
  }
  gpuErrchk(cudaPeekAtLastError());
  //gpuErrchk(cudaDeviceSynchronize());

  return 0;
}

//////////////////////////////////////////////////////////////////////
template<class ConnKeyT, class ConnStructT>
int NestedLoop::ParallelOuterNestedLoop(int i_func, int Nx, int *d_Ny)
{
  ParallelOuterNestedLoopKernel<ConnKeyT, ConnStructT>
    <<<(Nx+1023)/1024, 1024>>>(i_func, Nx, d_Ny);
  gpuErrchk(cudaPeekAtLastError());
  //gpuErrchk(cudaDeviceSynchronize());
  
  return 0;
}

//////////////////////////////////////////////////////////////////////
template<class ConnKeyT, class ConnStructT>
int NestedLoop::Frame1DNestedLoop(int i_func, int Nx, int *d_Ny)
{
  if (Nx <= 0) return 0;
  int dim_x, dim_y;

  // Run sorting operation
  cub::DeviceRadixSort::SortPairs(d_sort_storage_, sort_storage_bytes_,
				  d_Ny, d_sorted_Ny_, d_idx_, d_sorted_idx_,
				  Nx);
  
  int ix0 = Nx;
  while(ix0>0) {
    gpuErrchk(cudaMemcpy(&dim_y, &d_sorted_Ny_[ix0-1], sizeof(int),
			    cudaMemcpyDeviceToHost));
    if (dim_y < 1) dim_y = 1;
    dim_x = (frame_area_ - 1) / dim_y + 1;
    ix0 -= dim_x;
    if (ix0<0) {
      dim_x += ix0;
      ix0 = 0;
    } 
    Frame1DNestedLoopKernel<ConnKeyT, ConnStructT>
      <<<(dim_x*dim_y+1023)/1024, 1024>>>
      (i_func, ix0, dim_x, dim_y, d_sorted_idx_, d_sorted_Ny_);
  }
  gpuErrchk(cudaPeekAtLastError());
  //gpuErrchk(cudaDeviceSynchronize());
  
  return 0;
}

//////////////////////////////////////////////////////////////////////
template<class ConnKeyT, class ConnStructT>
int NestedLoop::Frame2DNestedLoop(int i_func, int Nx, int *d_Ny)
{
  if (Nx <= 0) return 0;
  // Sort the pairs (ix, Ny) with ix=0,..,Nx-1 in ascending order of Ny.
  // After the sorting operation, d_sorted_idx_ are the reordered indexes ix
  // and d_sorted_Ny_ are the sorted values of Ny 
  cub::DeviceRadixSort::SortPairs(d_sort_storage_, sort_storage_bytes_,
				  d_Ny, d_sorted_Ny_, d_idx_, d_sorted_idx_,
				  Nx);  
  int ix0 = Nx;	      // proceeds from right to left
  while(ix0>0) {
    int dim_x, dim_y;  // width and height of the rectangular frame
    gpuErrchk(cudaMemcpy(&dim_y, &d_sorted_Ny_[ix0-1], sizeof(int),
			    cudaMemcpyDeviceToHost));
    if (dim_y < 1) dim_y = 1;
    // frame_area_ is the fixed value of the the rectangular frame area
    dim_x = (frame_area_ - 1) / dim_y + 1; // width of the rectangular frame
    ix0 -= dim_x; // update the index value
    if (ix0<0) {
      dim_x += ix0;  // adjust the width if ix0<0 
      ix0 = 0;
    }    
    dim3 threadsPerBlock(block_dim_x_, block_dim_y_);  // block size
    dim3 numBlocks((dim_x - 1)/threadsPerBlock.x + 1,
		   (dim_y - 1)/threadsPerBlock.y + 1);
    // run a nested loop kernel on the rectangular frame
    Frame2DNestedLoopKernel<ConnKeyT, ConnStructT>
      <<<numBlocks,threadsPerBlock>>>
      (i_func, ix0, dim_x, dim_y, d_sorted_idx_, d_sorted_Ny_);

  }
  gpuErrchk(cudaPeekAtLastError());
  //gpuErrchk(cudaDeviceSynchronize());

  return 0;
}

//////////////////////////////////////////////////////////////////////
template<class ConnKeyT, class ConnStructT>
int NestedLoop::Smart1DNestedLoop(int i_func, int Nx, int *d_Ny)
{
  // Find max value of Ny
  cub::DeviceReduce::Max(d_reduce_storage_, reduce_storage_bytes_, d_Ny,
			 d_max_Ny_, Nx);
  int max_Ny;
  gpuErrchk(cudaMemcpy(&max_Ny, d_max_Ny_, sizeof(int),
			  cudaMemcpyDeviceToHost));
  if (Nx <= 0) return 0;
  float f_Nx = 2.0*log((float)Nx)-5;
  int i_Nx = (int)floor(f_Nx);
  int Ny_th;
  if (i_Nx<0) {
    Ny_th = Ny_th_arr_[0];
  }
  else if (i_Nx>=Ny_arr_size_-1) {
    Ny_th = Ny_th_arr_[Ny_arr_size_-1];
  }
  else {
    float t = f_Nx - (float)i_Nx;
    Ny_th = Ny_th_arr_[i_Nx]*(1.0 - t) + Ny_th_arr_[i_Nx+1]*t;
  }
  if (max_Ny<Ny_th) {
    return SimpleNestedLoop<ConnKeyT, ConnStructT>(i_func, Nx, d_Ny, max_Ny);
  }

  if(max_Ny < 1) max_Ny = 1;
  
  int dim_x, dim_y;

  // Run sorting operation
  cub::DeviceRadixSort::SortPairs(d_sort_storage_, sort_storage_bytes_,
				  d_Ny, d_sorted_Ny_, d_idx_, d_sorted_idx_,
				  Nx);
  // CudaCheckError(); // uncomment only for debugging
  
  int ix1 = (int)round(x_lim_*Nx);
  if (ix1==Nx) ix1 = Nx - 1;
  int Ny1;
  gpuErrchk(cudaMemcpy(&Ny1, &d_sorted_Ny_[ix1], sizeof(int),
			  cudaMemcpyDeviceToHost));
  if(Ny1 < 1) Ny1 = 1;

  dim3 threadsPerBlock(block_dim_x_, block_dim_y_);  // block size
  int nbx = (Nx - 1)/threadsPerBlock.x + 1;
  int nby = (Ny1 - 1)/threadsPerBlock.y + 1;
  Ny1 = nby*threadsPerBlock.y;
  
  dim3 numBlocks(nbx, nby);
  SimpleNestedLoopKernel<ConnKeyT, ConnStructT>
    <<<numBlocks,threadsPerBlock>>>(i_func, Nx, d_Ny);
  //CudaCheckError(); // uncomment only for debugging
  
  int ix0 = Nx;
  while(ix0>ix1) {
    gpuErrchk(cudaMemcpy(&dim_y, &d_sorted_Ny_[ix0-1], sizeof(int),
			    cudaMemcpyDeviceToHost));
    dim_y -= Ny1;
    if (dim_y<=0) break;
    dim_x = (frame_area_ - 1) / dim_y + 1;
    ix0 -= dim_x;
    if (ix0<ix1) {
      dim_x += ix0 - ix1;
      ix0 = ix1;
    } 
    Smart1DNestedLoopKernel<ConnKeyT, ConnStructT>
      <<<(dim_x*dim_y+1023)/1024, 1024>>>
      (i_func, ix0, Ny1, dim_x, dim_y, d_sorted_idx_, d_sorted_Ny_);
    //CudaCheckError(); // uncomment only for debugging
  }
  gpuErrchk(cudaPeekAtLastError());
  //gpuErrchk(cudaDeviceSynchronize());

  return 0;
}

//////////////////////////////////////////////////////////////////////
template<class ConnKeyT, class ConnStructT>
int NestedLoop::Smart2DNestedLoop(int i_func, int Nx, int *d_Ny)
{
  // Find max value of Ny
  cub::DeviceReduce::Max(d_reduce_storage_, reduce_storage_bytes_, d_Ny,
			 d_max_Ny_, Nx);
  int max_Ny;
  gpuErrchk(cudaMemcpy(&max_Ny, d_max_Ny_, sizeof(int),
			  cudaMemcpyDeviceToHost));
  if (Nx <= 0) return 0;
  float f_Nx = 2.0*log((float)Nx)-5;
  int i_Nx = (int)floor(f_Nx);
  int Ny_th;
  if (i_Nx<0) {
    Ny_th = Ny_th_arr_[0];
  }
  else if (i_Nx>=Ny_arr_size_-1) {
    Ny_th = Ny_th_arr_[Ny_arr_size_-1];
  }
  else {
    float t = f_Nx - (float)i_Nx;
    Ny_th = Ny_th_arr_[i_Nx]*(1.0 - t) + Ny_th_arr_[i_Nx+1]*t;
  }
  if (max_Ny<Ny_th) {
    return SimpleNestedLoop<ConnKeyT, ConnStructT>(i_func, Nx, d_Ny, max_Ny);
  }

  if(max_Ny < 1) max_Ny = 1;

  int dim_x, dim_y;

  // Run sorting operation
  cub::DeviceRadixSort::SortPairs(d_sort_storage_, sort_storage_bytes_,
				  d_Ny, d_sorted_Ny_, d_idx_, d_sorted_idx_,
				  Nx);
  // CudaCheckError(); // uncomment only for debugging
  
  int ix1 = (int)round(x_lim_*Nx);
  if (ix1==Nx) ix1 = Nx - 1;
  int Ny1;
  gpuErrchk(cudaMemcpy(&Ny1, &d_sorted_Ny_[ix1], sizeof(int),
			  cudaMemcpyDeviceToHost));
  if(Ny1 < 1) Ny1 = 1;

  dim3 threadsPerBlock(block_dim_x_, block_dim_y_);  // block size
  int nbx = (Nx - 1)/threadsPerBlock.x + 1;
  int nby = (Ny1 - 1)/threadsPerBlock.y + 1;
  Ny1 = nby*threadsPerBlock.y;
  
  dim3 numBlocks(nbx, nby);
  SimpleNestedLoopKernel<ConnKeyT, ConnStructT>
    <<<numBlocks,threadsPerBlock>>>(i_func, Nx, d_Ny);
  //CudaCheckError(); // uncomment only for debugging
  
  int ix0 = Nx;
  while(ix0>ix1) {
    gpuErrchk(cudaMemcpy(&dim_y, &d_sorted_Ny_[ix0-1], sizeof(int),
			    cudaMemcpyDeviceToHost));
    dim_y -= Ny1;
    if (dim_y<=0) break;
    dim_x = (frame_area_ - 1) / dim_y + 1;
    ix0 -= dim_x;
    if (ix0<ix1) {
      dim_x += ix0 - ix1;
      ix0 = ix1;
    }

    dim3 threadsPerBlock(block_dim_x_, block_dim_y_);  // block size
    dim3 numBlocks((dim_x - 1)/threadsPerBlock.x + 1,
		   (dim_y - 1)/threadsPerBlock.y + 1);
    Smart2DNestedLoopKernel<ConnKeyT, ConnStructT>
      <<<numBlocks,threadsPerBlock>>>
      (i_func, ix0, Ny1, dim_x, dim_y, d_sorted_idx_, d_sorted_Ny_);
    //CudaCheckError(); // uncomment only for debugging      
  }
  gpuErrchk(cudaPeekAtLastError());
  //gpuErrchk(cudaDeviceSynchronize());
  
  return 0;
}

//////////////////////////////////////////////////////////////////////
template<class ConnKeyT, class ConnStructT>
int NestedLoop::CumulSumNestedLoop(int i_func, int Nx, int *d_Ny)
{
  //TMP
  //double time_mark=getRealTime();
  //
  prefix_scan_.Scan(d_Ny_cumul_sum_, d_Ny, Nx+1);
  //TMP
  //printf("pst: %lf\n", getRealTime()-time_mark);
  //	 
  int Ny_sum;
  gpuErrchk(cudaMemcpy(&Ny_sum, &d_Ny_cumul_sum_[Nx],
			  sizeof(int), cudaMemcpyDeviceToHost));

  //printf("CSNL: %d %d\n", Nx, Ny_sum);
  
  //printf("Ny_sum %u\n", Ny_sum);
  //temporary - remove
  /*
  if (Ny_sum==0) {
    printf("Nx %d\n", Nx);
    for (int i=0; i<Nx+1; i++) {
      int psum;
      gpuErrchk(cudaMemcpy(&psum, &d_Ny_cumul_sum_[i],
  			      sizeof(int), cudaMemcpyDeviceToHost));
      printf("%d %d\n", i, psum);
    }
  }
  */    
  ////
  if(Ny_sum>0) {
    int grid_dim_x, grid_dim_y;
    if (Ny_sum<65536*1024) { // max grid dim * max block dim
      grid_dim_x = (Ny_sum+1023)/1024;
      grid_dim_y = 1;
    }
    else {
      grid_dim_x = 32; // I think it's not necessary to increase it
      if (Ny_sum>grid_dim_x*1024*65535) {
	throw ngpu_exception(std::string("Ny sum ") + std::to_string(Ny_sum) +
			     " larger than threshold "
			     + std::to_string(grid_dim_x*1024*65535));
      }
      grid_dim_y = (Ny_sum + grid_dim_x*1024 -1) / (grid_dim_x*1024);
    }
    dim3 numBlocks(grid_dim_x, grid_dim_y);
    //TMP
    //double time_mark=getRealTime();
    //
    CumulSumNestedLoopKernel<ConnKeyT, ConnStructT><<<numBlocks, 1024>>>
    (i_func, Nx, d_Ny_cumul_sum_, Ny_sum);
    gpuErrchk(cudaPeekAtLastError());
    //gpuErrchk(cudaDeviceSynchronize());

    //TMP
    //printf("cst: %lf\n", getRealTime()-time_mark);
    //
  }
    
  return 0;
}


#endif
