/*
 *  poiss_gen.cu
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


#include <cmath>
#include <config.h>
#include <iostream>
// #include <stdio.h>
#include <cub/cub.cuh>
#include <curand.h>
#include <curand_kernel.h>
#include <stdint.h>

#include "connect.h"
#include "copass_kernels.h"
#include "nestgpu.h"
#include "neuron_models.h"
#include "poiss_gen.h"
#include "poiss_gen_variables.h"
#include "utilities.h"

extern __constant__ double NESTGPUTime;
extern __constant__ float NESTGPUTimeResolution;
extern __constant__ NodeGroupStruct NodeGroupArray[];
extern __device__ int16_t* NodeGroupMap;

namespace poiss_conn
{
Connection* conn_;
// typedef uint key_t;
// typedef regular_block_array<key_t> array_t;
// key_t **d_poiss_key_array_data_pt;
// array_t *d_poiss_subarray;
void* d_poiss_key_array_data_pt;
void* d_poiss_subarray;

int64_t* d_poiss_num;
int64_t* d_poiss_sum;
// key_t *d_poiss_thresh;
void* d_poiss_thresh;
int
organizeDirectConnections( Connection* conn )
{
  conn_ = conn;
  return conn->organizeDirectConnections(
    d_poiss_key_array_data_pt, d_poiss_subarray, d_poiss_num, d_poiss_sum, d_poiss_thresh );
}
};


__global__ void
SetupPoissKernel( curandState* curand_state, uint64_t n_conn, unsigned long long seed )
{
  uint64_t blockId = ( uint64_t ) blockIdx.y * gridDim.x + blockIdx.x;
  uint64_t i_conn = blockId * blockDim.x + threadIdx.x;
  if ( i_conn < n_conn )
  {
    curand_init( seed, i_conn, 0, &curand_state[ i_conn ] );
  }
}


__global__ void
PoissGenUpdateKernel( long long time_idx, int n_node, int max_delay, float* param_arr, int n_param, float* mu_arr )
{
  int i_node = threadIdx.x + blockIdx.x * blockDim.x;
  if ( i_node < n_node )
  {
    float* param = param_arr + i_node * n_param;
    double t_rel = NESTGPUTime - origin;
    if ( ( t_rel >= start ) && ( t_rel <= stop ) )
    {
      int it = ( int ) ( time_idx % max_delay );
      mu_arr[ it * n_node + i_node ] = NESTGPUTimeResolution * rate / 1000.0;
    }
  }
}


int
poiss_gen::Init( int i_node_0, int n_node, int /*n_port*/, int i_group )
{
  BaseNeuron::Init( i_node_0, n_node, 0 /*n_port*/, i_group );
  node_type_ = i_poisson_generator_model;
  n_scal_param_ = N_POISS_GEN_SCAL_PARAM;
  n_param_ = n_scal_param_;
  scal_param_name_ = poiss_gen_scal_param_name;
  has_dir_conn_ = true;

  CUDAMALLOCCTRL( "&param_arr_", &param_arr_, n_node_ * n_param_ * sizeof( float ) );

  SetScalParam( 0, n_node, "rate", 0.0 );
  SetScalParam( 0, n_node, "origin", 0.0 );
  SetScalParam( 0, n_node, "start", 0.0 );
  SetScalParam( 0, n_node, "stop", 1.0e30 );

  return 0;
}

int
poiss_gen::buildDirectConnections()
{
  // printf("i_node_0_ %d n_node_ %d i_conn0_ %ld n_dir_conn_ %ld
  //  max_delay_ %d\n",
  // i_node_0_, n_node_, i_conn0_, n_dir_conn_, max_delay_);
  return poiss_conn::conn_->buildDirectConnections(
    i_node_0_, n_node_, i_conn0_, n_dir_conn_, max_delay_, d_mu_arr_, d_poiss_key_array_ );
}

int
poiss_gen::SendDirectSpikes( long long time_idx )
{
  return poiss_conn::conn_->sendDirectSpikes(
    time_idx, i_conn0_, n_dir_conn_, n_node_, max_delay_, d_mu_arr_, d_poiss_key_array_, d_curand_state_ );
}

int
poiss_gen::Calibrate( double, float )
{
  CUDAMALLOCCTRL( "&d_curand_state_", &d_curand_state_, n_dir_conn_ * sizeof( curandState ) );

  unsigned int grid_dim_x, grid_dim_y;

  if ( n_dir_conn_ < 65536 * 1024 )
  { // max grid dim * max block dim
    grid_dim_x = ( n_dir_conn_ + 1023 ) / 1024;
    grid_dim_y = 1;
  }
  else
  {
    grid_dim_x = 64; // I think it's not necessary to increase it
    if ( n_dir_conn_ > grid_dim_x * 1024 * 65535 )
    {
      throw ngpu_exception( std::string( "Number of direct connections " ) + std::to_string( n_dir_conn_ )
        + " larger than threshold " + std::to_string( grid_dim_x * 1024 * 65535 ) );
    }
    grid_dim_y = ( n_dir_conn_ + grid_dim_x * 1024 - 1 ) / ( grid_dim_x * 1024 );
  }
  dim3 numBlocks( grid_dim_x, grid_dim_y );

  unsigned int* d_seed;
  unsigned int h_seed;

  CUDAMALLOCCTRL( "&d_seed", &d_seed, sizeof( unsigned int ) );
  CURAND_CALL( curandGenerate( *random_generator_, d_seed, 1 ) );
  // Copy seed from device memory to host
  gpuErrchk( cudaMemcpy( &h_seed, d_seed, sizeof( unsigned int ), cudaMemcpyDeviceToHost ) );
  // std::cout << "h_seed: " << h_seed << "\n";

  SetupPoissKernel<<< numBlocks, 1024 >>>( d_curand_state_, n_dir_conn_, h_seed );
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  return 0;
}


int
poiss_gen::Update( long long it, double )
{
  PoissGenUpdateKernel<<< ( n_node_ + 1023 ) / 1024, 1024 >>>(
    it, n_node_, max_delay_, param_arr_, n_param_, d_mu_arr_ );
  DBGCUDASYNC

  return 0;
}
