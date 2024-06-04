/*
 *  cuda_error.h
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


#ifndef CUDAERROR_H
#define CUDAERROR_H
#include "ngpu_exception.h"
#include <stdio.h>

#define gpuErrchk( ans )                      \
  {                                           \
    gpuAssert( ( ans ), __FILE__, __LINE__ ); \
  }
inline void
gpuAssert( cudaError_t code, const char* file, int line, bool abort = true )
{
  if ( code != cudaSuccess )
  {
    fprintf( stderr, "GPUassert: %s %s %d\n", cudaGetErrorString( code ), file, line );
    if ( abort )
    {
      throw ngpu_exception( "CUDA error" );
    }
  }
}

#define CUDA_CALL( x )                                  \
  do                                                    \
  {                                                     \
    if ( ( x ) != cudaSuccess )                         \
    {                                                   \
      printf( "Error at %s:%d\n", __FILE__, __LINE__ ); \
      throw ngpu_exception( "CUDA error" );             \
    }                                                   \
  } while ( 0 )
#define CURAND_CALL( x )                                \
  do                                                    \
  {                                                     \
    if ( ( x ) != CURAND_STATUS_SUCCESS )               \
    {                                                   \
      printf( "Error at %s:%d\n", __FILE__, __LINE__ ); \
      throw ngpu_exception( "CUDA error" );             \
    }                                                   \
  } while ( 0 )

#endif
