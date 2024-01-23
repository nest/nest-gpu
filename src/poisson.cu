/*
 *  poisson.cu
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


#include "cuda_error.h"
#include "poisson.h"
#include "spike_buffer.h"
#include <algorithm>
#include <config.h>
#include <cuda.h>
#include <curand.h>
#include <stdio.h>
#include <stdlib.h>

__device__ unsigned int* PoissonData;

__global__ void
PoissonUpdate( unsigned int* poisson_data )
{
  PoissonData = poisson_data;
}

__global__ void
PoissonSendSpikes( int i_node_0, int n_node )
{
  int i_node = threadIdx.x + blockIdx.x * blockDim.x;
  if ( i_node < n_node )
  {
    int i_node_abs = i_node_0 + i_node;
    unsigned int height = PoissonData[ i_node ];
    if ( height > 0 )
    {
      PushSpike( i_node_abs, ( float ) height );
    }
  }
}

__global__ void
FixPoissonGenerator( unsigned int* poisson_data, int n, float mean )
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if ( i < n )
  {
    unsigned int val = poisson_data[ i ];
    if ( val > mean * 5 )
    {
      poisson_data[ i ] = 0;
    }
  }
}

int
PoissonGenerator::Init( curandGenerator_t* random_generator, unsigned int n )
{
  poisson_data_size_ = n;
  // Allocate n integers on device
  CUDA_CALL( cudaMalloc( ( void** ) &dev_poisson_data_, n * sizeof( unsigned int ) ) );
  random_generator_ = random_generator;

  return 0;
}

int
PoissonGenerator::Generate()
{
  return Generate( n_steps_ );
}

int
PoissonGenerator::Generate( int max_n_steps )
{
  if ( max_n_steps <= 0 )
  {
    more_steps_ = n_steps_;
  }
  else
  {
    more_steps_ = std::min( n_steps_, max_n_steps );
  }
  // Generate N floats on device
  CURAND_CALL( curandGeneratePoisson( *random_generator_, dev_poisson_data_, n_node_ * more_steps_, lambda_ ) );
  FixPoissonGenerator<<< ( n_node_ + 1023 ) / 1024, 1024 >>>( dev_poisson_data_, n_node_ * more_steps_, lambda_ );

  return 0;
}

int
PoissonGenerator::Free()
{
  CUDA_CALL( cudaFree( dev_poisson_data_ ) );

  return 0;
}

PoissonGenerator::~PoissonGenerator()
{
  // Free();
}

PoissonGenerator::PoissonGenerator()
{
  buffer_size_ = 100000;
  n_node_ = 0;
}

int
PoissonGenerator::Create( curandGenerator_t* random_generator, int i_node_0, int n_node, float lambda )
{
  i_node_0_ = i_node_0;
  n_node_ = n_node;
  lambda_ = lambda;

  n_steps_ = ( buffer_size_ - 1 ) / n_node + 1;
  // with the above formula:
  // buffer_size <= n_node*n_steps <= buffer_size + n_node - 1
  Init( random_generator, n_node_ * n_steps_ );
  i_step_ = 0;

  return 0;
}

int
PoissonGenerator::Update( int max_n_steps )
{
  if ( i_step_ == 0 )
  {
    Generate( max_n_steps );
  }

  if ( i_step_ == more_steps_ )
  {
    throw ngpu_exception(
      "Step index larger than maximum number of steps "
      "in poisson generator" );
  }

  PoissonUpdate<<< 1, 1 >>>( &dev_poisson_data_[ i_step_ * n_node_ ] );
  gpuErrchk( cudaPeekAtLastError() );

  PoissonSendSpikes<<< ( n_node_ + 1023 ) / 1024, 1024 >>>( i_node_0_, n_node_ );
  gpuErrchk( cudaPeekAtLastError() );

  i_step_++;
  if ( i_step_ == n_steps_ )
  {
    i_step_ = 0;
  }

  return 0;
}
