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
#include <config.h>

#include "ngpu_exception.h"
#include <map>
#include <stdio.h>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

namespace cuda_error_ns
{
extern std::map< void*, size_t > alloc_map_;
extern size_t mem_used_;
extern size_t mem_max_;
extern int verbose_;
} // namespace cuda_error_ns

inline int
printMPIRank()
{

#ifdef HAVE_MPI
  int proc_num;
  MPI_Comm_size( MPI_COMM_WORLD, &proc_num );
  if ( proc_num > 1 )
  {
    int mpi_id;
    MPI_Comm_rank( MPI_COMM_WORLD, &mpi_id );
    printf( "MPI rank: %d\t", mpi_id );
  }
#endif

  return 0;
}

inline void
mapCUDAMemAlloc( void* dev_pt, size_t n_bytes )
{
  cuda_error_ns::alloc_map_.insert( { dev_pt, n_bytes } );
  cuda_error_ns::mem_used_ += n_bytes;

  if ( cuda_error_ns::mem_used_ > cuda_error_ns::mem_max_ )
  {
    cuda_error_ns::mem_max_ = cuda_error_ns::mem_used_;
  }
  if ( cuda_error_ns::verbose_ > 0 )
  {
    printMPIRank();
    printf( "GPU memory usage: used = %.3f, max used = %.3f\n",
      ( float ) cuda_error_ns::mem_used_ / 1024.0 / 1024.0,
      ( float ) cuda_error_ns::mem_max_ / 1024.0 / 1024.0 );
  }
}

inline void
mapCUDAMemFree( void* dev_pt )
{
  if ( cuda_error_ns::alloc_map_.find( dev_pt ) == cuda_error_ns::alloc_map_.end() )
  {
    throw ngpu_exception( "CUDA error: pointer not found in mapCUDAMemFree." );
  }
  size_t n_bytes = cuda_error_ns::alloc_map_.at( dev_pt );
  cuda_error_ns::alloc_map_.erase( dev_pt );
  cuda_error_ns::mem_used_ -= n_bytes;

  if ( cuda_error_ns::verbose_ > 0 )
  {
    printMPIRank();
    printf( "GPU memory usage: used = %.3f, max used = %.3f\n",
      ( float ) cuda_error_ns::mem_used_ / 1024.0 / 1024.0,
      ( float ) cuda_error_ns::mem_max_ / 1024.0 / 1024.0 );
  }
}

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

// #define DEBUG_CUDA_SYNC
#ifdef DEBUG_CUDA_SYNC
#define DBGCUDASYNC                   \
  gpuErrchk( cudaPeekAtLastError() ); \
  gpuErrchk( cudaDeviceSynchronize() );
#else
#define DBGCUDASYNC gpuErrchk( cudaPeekAtLastError() );
#endif
#define CUDASYNC                      \
  gpuErrchk( cudaPeekAtLastError() ); \
  gpuErrchk( cudaDeviceSynchronize() );

#define CUDAMALLOCCTRL( str, dev_pt, n_bytes )                                   \
  {                                                                              \
    if ( cuda_error_ns::verbose_ > 0 )                                           \
    {                                                                            \
      printMPIRank();                                                            \
      printf( "Allocating %lld bytes pointed by %s in device memory at %s:%d\n", \
        ( unsigned long long ) n_bytes,                                          \
        str,                                                                     \
        __FILE__,                                                                \
        __LINE__ );                                                              \
    }                                                                            \
    gpuAssert( cudaMalloc( dev_pt, n_bytes ), __FILE__, __LINE__ );              \
    mapCUDAMemAlloc( *dev_pt, n_bytes );                                         \
  }
#define CUDAFREECTRL( str, dev_pt )                                                                \
  {                                                                                                \
    if ( cuda_error_ns::verbose_ > 0 )                                                             \
    {                                                                                              \
      printMPIRank();                                                                              \
      printf( "Deallocating device memory pointed by %s in at %s:%d\n", str, __FILE__, __LINE__ ); \
    }                                                                                              \
    gpuAssert( cudaFree( dev_pt ), __FILE__, __LINE__ );                                           \
    mapCUDAMemFree( dev_pt );                                                                      \
  }
#endif

//#define ACTIVATE_PRINT_TIME
#ifdef ACTIVATE_PRINT_TIME
#define PRINT_TIME                              \
  gpuErrchk( cudaPeekAtLastError() );           \
  gpuErrchk( cudaDeviceSynchronize() );	        \
  std::cout << "Time from start at " << __FILE__ << ":" << __LINE__ << "\t" << getRealTime() - start_real_time_ << std::endl;
#else
#define PRINT_TIME
#endif
