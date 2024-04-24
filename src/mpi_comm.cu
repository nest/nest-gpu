/*
 *  mpi_comm.cu
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

#include <config.h>

#include <list>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_error.h"
#include "getRealTime.h"
#include "nestgpu.h"

#include "mpi_comm.h"
#include "remote_connect.h"
#include "remote_spike.h"

#ifdef HAVE_MPI
#include <mpi.h>
MPI_Request* recv_mpi_request;
#endif

// Send spikes to remote MPI processes
int
NESTGPU::SendSpikeToRemote( int n_ext_spikes )
{
#ifdef HAVE_MPI
  MPI_Request request;
  int mpi_id, tag = 1; // id is already in the class, can be removed
  MPI_Comm_rank( MPI_COMM_WORLD, &mpi_id );

  double time_mark = getRealTime();
  gpuErrchk( cudaMemcpy(
    h_ExternalTargetSpikeNum, d_ExternalTargetSpikeNum, n_hosts_ * sizeof( int ), cudaMemcpyDeviceToHost ) );
  SendSpikeToRemote_CUDAcp_time_ += ( getRealTime() - time_mark );

  time_mark = getRealTime();
  int n_spike_tot = 0;
  // copy spikes from GPU to CPU memory
  if ( n_ext_spikes > 0 )
  {
    gpuErrchk(
      cudaMemcpy( &n_spike_tot, d_ExternalTargetSpikeIdx0 + n_hosts_, sizeof( int ), cudaMemcpyDeviceToHost ) );
    if ( n_spike_tot >= max_remote_spike_num_ )
    {
      throw ngpu_exception( std::string( "Number of spikes to be sent remotely " ) + std::to_string( n_spike_tot )
        + " larger than limit " + std::to_string( max_remote_spike_num_ ) );
    }

    gpuErrchk( cudaMemcpy(
      h_ExternalTargetSpikeNodeId, d_ExternalTargetSpikeNodeId, n_spike_tot * sizeof( int ), cudaMemcpyDeviceToHost ) );
    gpuErrchk( cudaMemcpy( h_ExternalTargetSpikeIdx0,
      d_ExternalTargetSpikeIdx0,
      ( n_hosts_ + 1 ) * sizeof( int ),
      cudaMemcpyDeviceToHost ) );
  }
  else
  {
    for ( int i = 0; i < n_hosts_ + 1; i++ )
    {
      h_ExternalTargetSpikeIdx0[ i ] = 0;
    }
  }

  SendSpikeToRemote_CUDAcp_time_ += ( getRealTime() - time_mark );
  time_mark = getRealTime();

  // loop on remote MPI proc
  for ( int ih = 0; ih < n_hosts_; ih++ )
  {
    if ( ( int ) ih == mpi_id )
    { // skip self MPI proc
      continue;
    }
    // get index and size of spike packet that must be sent to MPI proc ih
    // array_idx is the first index of the packet for host ih
    int array_idx = h_ExternalTargetSpikeIdx0[ ih ];
    int n_spikes = h_ExternalTargetSpikeIdx0[ ih + 1 ] - array_idx;
    // printf("MPI_Send (src,tgt,nspike): %d %d %d\n", mpi_id, ih, n_spike);

    // nonblocking sent of spike packet to MPI proc ih
    MPI_Isend( &h_ExternalTargetSpikeNodeId[ array_idx ], n_spikes, MPI_INT, ih, tag, MPI_COMM_WORLD, &request );

    // printf("MPI_Send nspikes (src,tgt,nspike): "
    //	   "%d %d %d\n", mpi_id, ih, n_spikes);
    // printf("MPI_Send 1st-neuron-idx (src,tgt,idx): "
    //	   "%d %d %d\n", mpi_id, ih,
    //	   h_ExternalTargetSpikeNodeId[array_idx]);
  }
  SendSpikeToRemote_comm_time_ += ( getRealTime() - time_mark );

  return 0;
#else
  throw ngpu_exception( "MPI is not available in your build" );
#endif
}

// Receive spikes from remote MPI processes
int
NESTGPU::RecvSpikeFromRemote()

{
#ifdef HAVE_MPI
  int mpi_id, tag = 1; // id is already in the class, can be removed
  MPI_Comm_rank( MPI_COMM_WORLD, &mpi_id );

  double time_mark = getRealTime();

  // loop on remote MPI proc
  for ( int i_host = 0; i_host < n_hosts_; i_host++ )
  {
    if ( ( int ) i_host == mpi_id )
    {
      continue; // skip self MPI proc
    }
    // start nonblocking MPI receive from MPI proc i_host
    MPI_Irecv( &h_ExternalSourceSpikeNodeId[0][ i_host * max_spike_per_host_ ],
      max_spike_per_host_,
      MPI_INT,
      i_host,
      tag,
      MPI_COMM_WORLD,
      &recv_mpi_request[ i_host ] );
  }

  // xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  std::vector< std::vector< int > > &host_group = conn_->getHostGroup();
  uint nhg = host_group.size();

  //h_ExternalSourceSpikeNodeId = new uint*[ nhg + 1 ];
  //h_ExternalSourceSpikeNodeId[0] = new uint[ max_remote_spike_num_ ];
  for (uint ihg=0; ihg<nhg; ihg++) {
    int idx0 = h_ExternalHostGroupSpikeIdx0[ihg]; // position of subarray of spikes that must be sent to host group ihg
    uint* sendbuf = &h_ExternalHostGroupSpikeNodeId[idx0]; // send address
    int sendcount = h_ExternalHostGroupSpikeNum[ihg]; // send count
    void *recvbuf = &h_ExternalSourceSpikeNodeId[ihg+1]; //[ i_host * max_spike_per_host_ ] // receiving buffers
    int *recvcounts = h_ExternalSourceSpikeNum[ihg];
    int *displs = h_ExternalSourceSpikeDispl; // displacememnts of receiving buffers, all equal to max_spike_per_host_
    
    //const int recvcounts[], const int displs[] // specificare displacements tutti uguali, inizializzare in Init con dim=n_hosts_ totali

    // Rimpiazzare MPI_COMM_WORLD con comunicatore del gruppo ihg
    MPI_Iallgatherv(sendbuf, sendcount, MPI_INT, recvbuf, recvcounts, displs, MPI_INT,
		    MPI_COMM_WORLD, &recv_mpi_request[ n_hosts_ + ihg ]);
  }

      // xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

  
  MPI_Status statuses[ n_hosts_ + nhg ];
  recv_mpi_request[ mpi_id ] = MPI_REQUEST_NULL;
  MPI_Waitall( n_hosts_ + nhg, recv_mpi_request, statuses );

  for ( int i_host = 0; i_host < n_hosts_; i_host++ )
  {
    if ( ( int ) i_host == mpi_id )
    {
      h_ExternalSourceSpikeNum[ i_host ] = 0;
      continue;
    }
    int count;
    MPI_Get_count( &statuses[ i_host ], MPI_INT, &count );
    h_ExternalSourceSpikeNum[0][ i_host ] = count;
  }
  // Maybe the barrier is not necessary?
  MPI_Barrier( MPI_COMM_WORLD );
  RecvSpikeFromRemote_comm_time_ += ( getRealTime() - time_mark );

  return 0;
#else
  throw ngpu_exception( "MPI is not available in your build" );
#endif
}

int
NESTGPU::ConnectMpiInit( int argc, char* argv[] )
{
#ifdef HAVE_MPI
  CheckUncalibrated( "MPI connections cannot be initialized after calibration" );
  int initialized;
  MPI_Initialized( &initialized );
  if ( !initialized )
  {
    MPI_Init( &argc, &argv );
  }
  int n_hosts;
  int this_host;
  MPI_Comm_size( MPI_COMM_WORLD, &n_hosts );
  MPI_Comm_rank( MPI_COMM_WORLD, &this_host );
  mpi_flag_ = true;
  setNHosts( n_hosts );
  setThisHost( this_host );
  conn_->remoteConnectionMapInit();
  recv_mpi_request = new MPI_Request[ n_hosts_ ];

  return 0;
#else
  throw ngpu_exception( "MPI is not available in your build" );
#endif
}

int
NESTGPU::MpiFinalize()
{
#ifdef HAVE_MPI
  if ( mpi_flag_ )
  {
    int finalized;
    MPI_Finalized( &finalized );
    if ( !finalized )
    {
      MPI_Finalize();
    }
  }

  return 0;
#else
  throw ngpu_exception( "MPI is not available in your build" );
#endif
}
