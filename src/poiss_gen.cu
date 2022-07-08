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
#include <cmath>
#include <iostream>
//#include <stdio.h>
#include <stdint.h>
#include <curand.h>
#include <curand_kernel.h>

#include "utilities.h"
#include "nestgpu.h"
#include "neuron_models.h"
#include "poiss_gen.h"
#include "poiss_gen_variables.h"

extern __constant__ NodeGroupStruct NodeGroupArray[];
extern __device__ signed char *NodeGroupMap;

__global__ void SetupPoissKernel(curandState *curand_state, uint64_t n_dir_conn,
				 unsigned long long seed)
{
  uint64_t blockId   = (uint64_t)blockIdx.y * gridDim.x + blockIdx.x;
  uint64_t i_conn = blockId * blockDim.x + threadIdx.x;
  if (i_conn<n_dir_conn) {
    curand_init(seed, i_conn, 0, &curand_state[i_conn]);
  }
}

__global__ void PoissGenSendSpikeKernel(curandState *curand_state, double t,
					float time_step, float *param_arr,
					int n_param,
					DirectConnection *dir_conn_array,
					uint64_t n_dir_conn)
{
  uint64_t blockId   = (uint64_t)blockIdx.y * gridDim.x + blockIdx.x;
  uint64_t i_conn = blockId * blockDim.x + threadIdx.x;
  if (i_conn<n_dir_conn) {
    DirectConnection dir_conn = dir_conn_array[i_conn];
    int irel = dir_conn.irel_source_;
    int i_target = dir_conn.i_target_;
    int port = dir_conn.port_;
    float weight = dir_conn.weight_;
    float delay = dir_conn.delay_;
    float *param = param_arr + irel*n_param;
    double t_rel = t - origin - delay;

    if ((t_rel>=start) && (t_rel<=stop)){
      int n = curand_poisson(curand_state+i_conn, time_step*rate);
      if (n>0) { // //Send direct spike (i_target, port, weight*n);
	/////////////////////////////////////////////////////////////////
	int i_group=NodeGroupMap[i_target];
	int i = port*NodeGroupArray[i_group].n_node_ + i_target
	  - NodeGroupArray[i_group].i_node_0_;
	double d_val = (double)(weight*n);
	atomicAddDouble(&NodeGroupArray[i_group].get_spike_array_[i], d_val); 
	////////////////////////////////////////////////////////////////
      }
    }
  }
}


int poiss_gen::Init(int i_node_0, int n_node, int /*n_port*/,
		    int i_group, unsigned long long *seed)
{
  BaseNeuron::Init(i_node_0, n_node, 0 /*n_port*/, i_group, seed);
  node_type_ = i_poisson_generator_model;
  n_scal_param_ = N_POISS_GEN_SCAL_PARAM;
  n_param_ = n_scal_param_;
  scal_param_name_ = poiss_gen_scal_param_name;
  has_dir_conn_ = true;
  
  gpuErrchk(cudaMalloc(&param_arr_, n_node_*n_param_*sizeof(float)));

  SetScalParam(0, n_node, "rate", 0.0);
  SetScalParam(0, n_node, "origin", 0.0);
  SetScalParam(0, n_node, "start", 0.0);
  SetScalParam(0, n_node, "stop", 1.0e30);
  
  return 0;
}

int poiss_gen::Calibrate(double, float)
{
  gpuErrchk(cudaMalloc(&d_curand_state_, n_dir_conn_*sizeof(curandState)));

  unsigned int grid_dim_x, grid_dim_y;

  if (n_dir_conn_<65536*1024) { // max grid dim * max block dim
    grid_dim_x = (n_dir_conn_+1023)/1024;
    grid_dim_y = 1;
  }
  else {
    grid_dim_x = 64; // I think it's not necessary to increase it
    if (n_dir_conn_>grid_dim_x*1024*65535) {
      throw ngpu_exception(std::string("Number of direct connections ")
			   + std::to_string(n_dir_conn_) +
			   " larger than threshold "
			   + std::to_string(grid_dim_x*1024*65535));
    }
    grid_dim_y = (n_dir_conn_ + grid_dim_x*1024 -1) / (grid_dim_x*1024);
  }
  dim3 numBlocks(grid_dim_x, grid_dim_y);
  SetupPoissKernel<<<numBlocks, 1024>>>(d_curand_state_, n_dir_conn_, *seed_);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  
  return 0;
}


int poiss_gen::Update(long long it, double t1)
{
  return 0;
}

int poiss_gen::SendDirectSpikes(double t, float time_step)
{
  unsigned int grid_dim_x, grid_dim_y;
  
  if (n_dir_conn_<65536*1024) { // max grid dim * max block dim
    grid_dim_x = (n_dir_conn_+1023)/1024;
    grid_dim_y = 1;
  }
  else {
    grid_dim_x = 64; // I think it's not necessary to increase it
    if (n_dir_conn_>grid_dim_x*1024*65535) {
      throw ngpu_exception(std::string("Number of direct connections ")
			   + std::to_string(n_dir_conn_) +
			   " larger than threshold "
			   + std::to_string(grid_dim_x*1024*65535));
    }
    grid_dim_y = (n_dir_conn_ + grid_dim_x*1024 -1) / (grid_dim_x*1024);
  }
  dim3 numBlocks(grid_dim_x, grid_dim_y);
  PoissGenSendSpikeKernel<<<numBlocks, 1024>>>(d_curand_state_, t, time_step,
					       param_arr_, n_param_,
					       d_dir_conn_array_, n_dir_conn_);
  
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  return 0;
}
