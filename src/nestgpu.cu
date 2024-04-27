/*
 *  nestgpu.cu
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

#include <algorithm>
#include <cmath>
#include <config.h>
#include <curand.h>
#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <string>

#include "cuda_error.h"
#include "distribution.h"
#include "get_spike.h"
#include "send_spike.h"
#include "spike_buffer.h"
#include "syn_model.h"

#include "connect.h"
#include "getRealTime.h"
#include "multimeter.h"
#include "nested_loop.h"
#include "nestgpu.h"
#include "poiss_gen.h"
#include "random.h"
#include "remote_spike.h"
#include "rev_spike.h"
#include "spike_generator.h"

#include "conn12b.h"
#include "conn16b.h"
#include "input_spike_buffer.h"
#include "remote_connect.h"

////////////// TEMPORARY
#include "scan.h"
//////////////////////

// #define VERBOSE_TIME

__constant__ double NESTGPUTime;
__constant__ long long NESTGPUTimeIdx;
__constant__ float NESTGPUTimeResolution;

namespace cuda_error_ns
{
std::map< void*, size_t > alloc_map_;
size_t mem_used_;
size_t mem_max_;
int verbose_;
} // namespace cuda_error_ns

enum KernelFloatParamIndexes
{
  i_time_resolution = 0,
  i_max_spike_num_fact,
  i_max_spike_per_host_fact,
  i_max_remote_spike_num_fact,
  N_KERNEL_FLOAT_PARAM
};

enum KernelIntParamIndexes
{
  i_rnd_seed = 0,
  i_verbosity_level,
  i_max_spike_buffer_size,
  i_max_node_n_bits,
  i_max_syn_n_bits,
  i_max_delay_n_bits,
  i_conn_struct_type,
  i_spike_buffer_algo,
  N_KERNEL_INT_PARAM
};

enum KernelBoolParamIndexes
{
  i_print_time,
  i_remove_conn_key,
  i_remote_spike_mul,
  N_KERNEL_BOOL_PARAM
};

enum ConnStructType
{
  i_conn12b,
  i_conn16b,
  N_CONN_STRUCT_TYPE
};

const std::string kernel_float_param_name[ N_KERNEL_FLOAT_PARAM ] = { "time_resolution",
  "max_spike_num_fact",
  "max_spike_per_host_fact",
  "max_remote_spike_num_fact" };

const std::string kernel_int_param_name[ N_KERNEL_INT_PARAM ] = { "rnd_seed",
  "verbosity_level",
  "max_spike_buffer_size",
  "max_node_n_bits",
  "max_syn_n_bits",
  "max_delay_n_bits",
  "conn_struct_type",
  "spike_buffer_algo" };

const std::string kernel_bool_param_name[ N_KERNEL_BOOL_PARAM ] = { "print_time",
  "remove_conn_key",
  "remote_spike_mul" };

int
NESTGPU::setNHosts( int n_hosts )
{
  n_hosts_ = n_hosts;
  conn_->setNHosts( n_hosts );
  SetRandomSeed( kernel_seed_ );
  n_remote_nodes_.assign( n_hosts_, 0 );
  external_spike_flag_ = ( n_hosts > 1 ) ? true : false;
  gpuErrchk( cudaMemcpyToSymbolAsync( ExternalSpikeFlag, &external_spike_flag_, sizeof( bool ) ) );
      
  return 0;
}

int
NESTGPU::setThisHost( int i_host )
{
  this_host_ = i_host;
  conn_->setThisHost( i_host );
  SetRandomSeed( kernel_seed_ );

  return 0;
}

NESTGPU::NESTGPU()
{
  n_hosts_ = 1;
  this_host_ = 0;
  external_spike_flag_ = false;

  time_resolution_ = 0.1; // time resolution in ms

  random_generator_ = new curandGenerator_t;
  CURAND_CALL( curandCreateGenerator( random_generator_, CURAND_RNG_PSEUDO_DEFAULT ) );
  kernel_seed_ = 123456789ULL;
  CURAND_CALL( curandSetPseudoRandomGeneratorSeed( *random_generator_, kernel_seed_ + this_host_ ) );

  conn_ = nullptr;
  // by default, connection structure type used is the 12-byte type
  setConnStructType( i_conn12b );
  // setConnStructType( i_conn16b );

  distribution_ = new Distribution;
  multimeter_ = new Multimeter;

  calibrate_flag_ = false;
  create_flag_ = false;

  cuda_error_ns::mem_used_ = 0;
  cuda_error_ns::mem_max_ = 0;

  start_real_time_ = getRealTime();
  max_spike_buffer_size_ = 20;
  t_min_ = 0.0;
  sim_time_ = 1000.0; // Simulation time in ms
  // n_poiss_nodes_ = 0;
  n_remote_nodes_.assign( 1, 0 );

  max_spike_num_fact_ = 1.0;
  max_spike_per_host_fact_ = 1.0;
  max_remote_spike_num_fact_ = 1.0;

  error_flag_ = false;
  error_message_ = "";
  error_code_ = 0;

  on_exception_ = ON_EXCEPTION_EXIT;

  verbosity_level_ = 4;
  cuda_error_ns::verbose_ = 0;
  print_time_ = false;
  remove_conn_key_ = false;

  mpi_flag_ = false;
  remote_spike_mul_ = false;

  nested_loop_algo_ = CumulSumNestedLoopAlgo;

  SpikeBufferUpdate_time_ = 0;
  poisson_generator_time_ = 0;
  neuron_Update_time_ = 0;
  copy_ext_spike_time_ = 0;
  organizeExternalSpike_time_ = 0;
  SendSpikeToRemote_time_ = 0;
  RecvSpikeFromRemote_time_ = 0;
  NestedLoop_time_ = 0;
  GetSpike_time_ = 0;
  SpikeReset_time_ = 0;
  ExternalSpikeReset_time_ = 0;

  first_simulation_flag_ = true;
}

NESTGPU::~NESTGPU()
{
  multimeter_->CloseFiles();
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  if ( calibrate_flag_ )
  {
    FreeNodeGroupMap();
    FreeGetSpikeArrays();
  }

  for ( unsigned int i = 0; i < node_vect_.size(); i++ )
  {
    delete node_vect_[ i ];
  }

  delete multimeter_;
  curandDestroyGenerator( *random_generator_ );
  delete random_generator_;
}

int
NESTGPU::SetRandomSeed( unsigned long long seed )
{
  kernel_seed_ = seed;
  CURAND_CALL( curandSetPseudoRandomGeneratorSeed( *random_generator_, kernel_seed_ + this_host_ ) );
  conn_->setRandomSeed( seed );

  return 0;
}

int
NESTGPU::SetTimeResolution( float time_res )
{
  time_resolution_ = time_res;
  conn_->setTimeResolution( time_resolution_ );

  return 0;
}

int
NESTGPU::SetNestedLoopAlgo( int nested_loop_algo )
{
  nested_loop_algo_ = nested_loop_algo;

  return 0;
}

int
NESTGPU::SetMaxSpikeBufferSize( int max_size )
{
  max_spike_buffer_size_ = max_size;

  return 0;
}

int
NESTGPU::GetMaxSpikeBufferSize()
{
  return max_spike_buffer_size_;
}

uint
NESTGPU::GetNLocalNodes()
{
  return node_group_map_.size();
}

uint
NESTGPU::GetNTotalNodes()
{
  return GetNLocalNodes() + conn_->getNImageNodes();
}

int
NESTGPU::CheckImageNodes( int n_nodes )
{
  int i_node_0 = GetNLocalNodes();
  int max_n_nodes = ( int ) ( IntPow( 2, conn_->getMaxNodeNBits() ) - 1 );

  if ( ( i_node_0 + n_nodes ) > max_n_nodes )
  {
    throw ngpu_exception( std::string( "Local plus Image nodes exceed maximum"
                                       " number of nodes " )
      + std::to_string( max_n_nodes ) );
  }

  return i_node_0;
}

// method for changing connection structure type
int
NESTGPU::setConnStructType( int conn_struct_type )
{
  // std::cout << "In setConnStructType " << conn_struct_type << "\n";
  // Check if conn_ pointer has a nonzero value.
  // In this case connection object must be deallocated
  if ( conn_ != nullptr )
  {
    delete conn_;
  }
  // set new connection structure type
  conn_struct_type_ = conn_struct_type;
  // create connection object from the proper derived class
  // Note that conn_ is of the type pointer-to-the(abstract)-base class
  // while the object is in instance of a derived class
  // defined using templates
  switch ( conn_struct_type )
  {
  case i_conn12b:
    conn_ = new ConnectionTemplate< conn12b_key, conn12b_struct >;
    break;
  case i_conn16b:
    conn_ = new ConnectionTemplate< conn16b_key, conn16b_struct >;
    break;
  default:
    throw ngpu_exception( "Unrecognized connection structure type index" );
  }
  conn_->setRandomSeed( kernel_seed_ );

  // set time resolution in connection object
  conn_->setTimeResolution( time_resolution_ );

  return 0;
}

int
NESTGPU::CreateNodeGroup( int n_nodes, int n_ports )
{
  int i_node_0 = GetNLocalNodes();
  int max_node_nbits = conn_->getMaxNodeNBits();
  int max_n_nodes = ( int ) ( IntPow( 2, max_node_nbits ) - 1 );
  int max_n_ports = ( int ) ( IntPow( 2, conn_->getMaxPortNBits() ) - 1 );
  // std::cout << "max_node_nbits " << max_node_nbits << "\n";

  if ( ( i_node_0 + n_nodes ) > max_n_nodes )
  {
    throw ngpu_exception(
      std::string( "Maximum number of local nodes " ) + std::to_string( max_n_nodes ) + " exceeded" );
  }
  if ( n_ports > max_n_ports )
  {
    throw ngpu_exception( std::string( "Maximum number of ports " ) + std::to_string( max_n_ports ) + " exceeded" );
  }
  int i_group = node_vect_.size() - 1;
  node_group_map_.insert( node_group_map_.end(), n_nodes, i_group );

  node_vect_[ i_group ]->random_generator_ = random_generator_;
  node_vect_[ i_group ]->Init( i_node_0, n_nodes, n_ports, i_group );
  node_vect_[ i_group ]->get_spike_array_ = InitGetSpikeArray( n_nodes, n_ports );

  return i_node_0;
}

int
NESTGPU::CheckUncalibrated( std::string message )
{
  if ( calibrate_flag_ == true )
  {
    throw ngpu_exception( message );
  }

  return 0;
}

int
NESTGPU::Calibrate()
{
  CheckUncalibrated( "Calibration can be made only once" );

  if ( verbosity_level_ >= 1 )
  {
    std::cout << HostIdStr() << "Calibrating ...\n";
  }

  gpuErrchk( cudaMemcpyToSymbol( NESTGPUTimeResolution, &time_resolution_, sizeof( float ) ) );

  gpuErrchk( cudaMemcpyToSymbol( have_remote_spike_mul, &remote_spike_mul_, sizeof( bool ) ) );
  ///////////////////////////////////
  int n_nodes = GetNLocalNodes();
  gpuErrchk( cudaMemcpyToSymbol( n_local_nodes, &n_nodes, sizeof( int ) ) );

  int n_image_nodes = conn_->getNImageNodes();
  // std::cout << "n_local_nodes: " << n_nodes << " n_image_nodes: "
  //	    << n_image_nodes << "\n";
  if ( n_image_nodes > 0 )
  {
    CheckImageNodes( n_image_nodes );
    conn_->addOffsetToExternalNodeIds( GetNLocalNodes() );
  }

  calibrate_flag_ = true;

  conn_->organizeConnections( GetNTotalNodes() );

  conn_->calibrate();

  conn_->initInputSpikeBuffer( GetNLocalNodes(), GetNTotalNodes() );

  poiss_conn::organizeDirectConnections( conn_ );
  for ( unsigned int i = 0; i < node_vect_.size(); i++ )
  {
    if ( node_vect_[ i ]->has_dir_conn_ )
    {
      node_vect_[ i ]->buildDirectConnections();
    }
  }

  if ( remove_conn_key_ )
  {
    conn_->freeConnectionKey();
  }

  int max_delay_num = max_spike_buffer_size_;

  unsigned int n_spike_buffers = GetNTotalNodes();
  NestedLoop::Init( n_spike_buffers );

  // temporary
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  neur_t0_ = t_min_;
  neural_time_ = t_min_;

  NodeGroupArrayInit();

  max_spike_num_ = ( int ) round( max_spike_num_fact_ * GetNTotalNodes() * max_delay_num );
  max_spike_num_ = ( max_spike_num_ > 1 ) ? max_spike_num_ : 1;

  max_spike_per_host_ = ( int ) round( max_spike_per_host_fact_ * GetNLocalNodes() * max_delay_num );
  max_spike_per_host_ = ( max_spike_per_host_ > 1 ) ? max_spike_per_host_ : 1;

  max_remote_spike_num_ = max_spike_per_host_ * n_hosts_ * max_remote_spike_num_fact_;
  max_remote_spike_num_ = ( max_remote_spike_num_ > 1 ) ? max_remote_spike_num_ : 1;

  SpikeInit( max_spike_num_ );
  spikeBufferInit( GetNTotalNodes(), max_spike_buffer_size_, conn_->getSpikeBufferAlgo() );

  if ( n_hosts_ > 1 )
  {
    conn_->remoteConnectionMapCalibrate( GetNLocalNodes() );

    ExternalSpikeInit();
  }

  if ( conn_->getRevConnFlag() )
  {
    conn_->revSpikeInit( GetNLocalNodes() );
  }

  multimeter_->OpenFiles();

  for ( unsigned int i = 0; i < node_vect_.size(); i++ )
  {
    node_vect_[ i ]->Calibrate( t_min_, time_resolution_ );
  }

  SynGroupCalibrate();

  gpuErrchk( cudaMemcpyToSymbolAsync( NESTGPUTimeResolution, &time_resolution_, sizeof( float ) ) );

  return 0;
}

int
NESTGPU::Simulate( float sim_time )
{
  sim_time_ = sim_time;
  return Simulate();
}

int
NESTGPU::Simulate()
{
  StartSimulation();

  for ( long long it = 0; it < Nt_; it++ )
  {
    if ( it % 100 == 0 && verbosity_level_ >= 2 && print_time_ == true )
    {
      printf( "\r[%.2lf %%] Model time: %.3lf ms", 100.0 * ( neural_time_ - neur_t0_ ) / sim_time_, neural_time_ );
    }
    SimulationStep();
  }
  EndSimulation();

  return 0;
}

int
NESTGPU::StartSimulation()
{
  if ( !calibrate_flag_ )
  {
    Calibrate();
  }
  if ( first_simulation_flag_ )
  {
    gpuErrchk( cudaMemcpyToSymbolAsync( NESTGPUTime, &neur_t0_, sizeof( double ) ) );
    long long time_idx = ( int ) round( neur_t0_ / time_resolution_ );
    multimeter_->WriteRecords( neur_t0_, time_idx );
    build_real_time_ = getRealTime();
    first_simulation_flag_ = false;
  }
  else
  {
    neur_t0_ = neural_time_;
  }
  it_ = 0;
  Nt_ = ( long long ) round( sim_time_ / time_resolution_ );

  if ( verbosity_level_ >= 1 )
  {
    std::cout << HostIdStr() << "Simulating ...\n";
    printf( "Neural activity simulation time: %.3lf ms\n", sim_time_ );
  }

  return 0;
}

int
NESTGPU::EndSimulation()
{
  if ( verbosity_level_ >= 2 && print_time_ == true )
  {
    printf( "\r[%.2lf %%] Model time: %.3lf ms", 100.0 * ( neural_time_ - neur_t0_ ) / sim_time_, neural_time_ );
  }

  end_real_time_ = getRealTime();

  // multimeter_->CloseFiles();
  // neuron.rk5.Free();

  if ( verbosity_level_ >= 3 )
  {
    std::cout << "\n";
    std::cout << HostIdStr() << "  SpikeBufferUpdate_time: " << SpikeBufferUpdate_time_ << "\n";
    std::cout << HostIdStr() << "  poisson_generator_time: " << poisson_generator_time_ << "\n";
    std::cout << HostIdStr() << "  neuron_Update_time: " << neuron_Update_time_ << "\n";
    std::cout << HostIdStr() << "  copy_ext_spike_time: " << copy_ext_spike_time_ << "\n";
    std::cout << HostIdStr() << "  organizeExternalSpike_time: " << organizeExternalSpike_time_ << "\n";
    std::cout << HostIdStr() << "  SendSpikeToRemote_time: " << SendSpikeToRemote_time_ << "\n";
    std::cout << HostIdStr() << "  RecvSpikeFromRemote_time: " << RecvSpikeFromRemote_time_ << "\n";
    std::cout << HostIdStr() << "  NestedLoop_time: " << NestedLoop_time_ << "\n";
    std::cout << HostIdStr() << "  GetSpike_time: " << GetSpike_time_ << "\n";
    std::cout << HostIdStr() << "  SpikeReset_time: " << SpikeReset_time_ << "\n";
    std::cout << HostIdStr() << "  ExternalSpikeReset_time: " << ExternalSpikeReset_time_ << "\n";
  }

  if ( n_hosts_ > 1 && verbosity_level_ >= 4 )
  {
    std::cout << HostIdStr() << "  SendSpikeToRemote_comm_time: " << SendSpikeToRemote_comm_time_ << "\n";
    std::cout << HostIdStr() << "  RecvSpikeFromRemote_comm_time: " << RecvSpikeFromRemote_comm_time_ << "\n";
    std::cout << HostIdStr() << "  SendSpikeToRemote_CUDAcp_time: " << SendSpikeToRemote_CUDAcp_time_ << "\n";
    std::cout << HostIdStr() << "  RecvSpikeFromRemote_CUDAcp_time: " << RecvSpikeFromRemote_CUDAcp_time_ << "\n";
  }

  if ( verbosity_level_ >= 1 )
  {
    std::cout << HostIdStr() << "Building time: " << ( build_real_time_ - start_real_time_ ) << "\n";
    std::cout << HostIdStr() << "Simulation time: " << ( end_real_time_ - build_real_time_ ) << "\n";
  }

  return 0;
}

int
NESTGPU::SimulationStep()
{
  if ( first_simulation_flag_ )
  {
    StartSimulation();
  }
  double time_mark;

  if ( conn_->getSpikeBufferAlgo() != INPUT_SPIKE_BUFFER_ALGO )
  {
    time_mark = getRealTime();
    SpikeBufferUpdate<<< ( GetNTotalNodes() + 1023 ) / 1024, 1024 >>>();
    DBGCUDASYNC;

    SpikeBufferUpdate_time_ += ( getRealTime() - time_mark );
  }

  time_mark = getRealTime();
  neural_time_ = neur_t0_ + ( double ) time_resolution_ * ( it_ + 1 );
  // std::cout << "neural_time_: " << neural_time_ << "\n";
  gpuErrchk( cudaMemcpyToSymbolAsync( NESTGPUTime, &neural_time_, sizeof( double ) ) );
  long long time_idx = ( int ) round( neur_t0_ / time_resolution_ ) + it_ + 1;
  gpuErrchk( cudaMemcpyToSymbolAsync( NESTGPUTimeIdx, &time_idx, sizeof( long long ) ) );

  /*
  if (ConnectionSpikeTimeFlag) {
    if ( (time_idx & 0xffff) == 0x8000) {
      ResetConnectionSpikeTimeUp(net_connection_);
    }
    else if ( (time_idx & 0xffff) == 0) {
      ResetConnectionSpikeTimeDown(net_connection_);
    }
  }
  */

  for ( unsigned int i = 0; i < node_vect_.size(); i++ )
  {
    node_vect_[ i ]->Update( it_, neural_time_ );
  }
  gpuErrchk( cudaPeekAtLastError() );

  neuron_Update_time_ += ( getRealTime() - time_mark );
  multimeter_->WriteRecords( neural_time_, time_idx );

  if ( n_hosts_ > 1 )
  {
    int n_ext_spikes;
    time_mark = getRealTime();
    gpuErrchk( cudaMemcpy( &n_ext_spikes, d_ExternalSpikeNum, sizeof( int ), cudaMemcpyDeviceToHost ) );
    copy_ext_spike_time_ += ( getRealTime() - time_mark );

    if ( n_ext_spikes != 0 )
    {
      time_mark = getRealTime();
      organizeExternalSpikes( n_ext_spikes );
      organizeExternalSpike_time_ += ( getRealTime() - time_mark );
    }
    time_mark = getRealTime();
    SendSpikeToRemote( n_ext_spikes );

    SendSpikeToRemote_time_ += ( getRealTime() - time_mark );
    time_mark = getRealTime();
    RecvSpikeFromRemote();
    RecvSpikeFromRemote_time_ += ( getRealTime() - time_mark );
    CopySpikeFromRemote();
  }


  if ( conn_->getSpikeBufferAlgo() == INPUT_SPIKE_BUFFER_ALGO )
  {
    conn_->deliverSpikes();
  }
  else
  {
    int n_spikes;

    // Call will get delayed until ClearGetSpikesArrays()
    // afterwards the value of n_spikes will be available
    gpuErrchk( cudaMemcpyAsync( &n_spikes, d_SpikeNum, sizeof( int ), cudaMemcpyDeviceToHost ) );

    ClearGetSpikeArrays();
    gpuErrchk( cudaDeviceSynchronize() );
    if ( n_spikes > 0 )
    {
      time_mark = getRealTime();
      switch ( conn_struct_type_ )
      {
      case i_conn12b:
        NestedLoop::Run< 0 >( nested_loop_algo_, n_spikes, d_SpikeTargetNum );
        break;
      case i_conn16b:
        NestedLoop::Run< 2 >( nested_loop_algo_, n_spikes, d_SpikeTargetNum );
        break;
      default:
        throw ngpu_exception( "Unrecognized connection structure type index" );
      }
      NestedLoop_time_ += ( getRealTime() - time_mark );
    }
  }

  time_mark = getRealTime();
  for ( unsigned int i = 0; i < node_vect_.size(); i++ )
  {
    if ( node_vect_[ i ]->has_dir_conn_ )
    {
      node_vect_[ i ]->SendDirectSpikes( time_idx );
    }
  }
  poisson_generator_time_ += ( getRealTime() - time_mark );

  time_mark = getRealTime();
  for ( unsigned int i = 0; i < node_vect_.size(); i++ )
  {
    if ( node_vect_[ i ]->n_port_ > 0 )
    {

      int grid_dim_x = ( node_vect_[ i ]->n_node_ + 1023 ) / 1024;
      int grid_dim_y = node_vect_[ i ]->n_port_;
      dim3 grid_dim( grid_dim_x, grid_dim_y );
      // dim3 block_dim(1024,1);

      if ( conn_->getSpikeBufferAlgo() == INPUT_SPIKE_BUFFER_ALGO )
      {
        input_spike_buffer_ns::GetInputSpikes<<< grid_dim, 1024 >>> // block_dim>>>
          ( node_vect_[ i ]->i_node_0_,
            node_vect_[ i ]->n_node_,
            node_vect_[ i ]->n_port_,
            node_vect_[ i ]->n_var_,
            node_vect_[ i ]->port_weight_arr_,
            node_vect_[ i ]->port_weight_arr_step_,
            node_vect_[ i ]->port_weight_port_step_,
            node_vect_[ i ]->port_input_arr_,
            node_vect_[ i ]->port_input_arr_step_,
            node_vect_[ i ]->port_input_port_step_ );
        DBGCUDASYNC;
      }
      else
      {
        GetSpikes<<< grid_dim, 1024 >>> // block_dim>>>
          ( node_vect_[ i ]->get_spike_array_,
            node_vect_[ i ]->n_node_,
            node_vect_[ i ]->n_port_,
            node_vect_[ i ]->n_var_,
            node_vect_[ i ]->port_weight_arr_,
            node_vect_[ i ]->port_weight_arr_step_,
            node_vect_[ i ]->port_weight_port_step_,
            node_vect_[ i ]->port_input_arr_,
            node_vect_[ i ]->port_input_arr_step_,
            node_vect_[ i ]->port_input_port_step_ );
        DBGCUDASYNC;
      }
    }
  }

  GetSpike_time_ += ( getRealTime() - time_mark );

  time_mark = getRealTime();
  SpikeReset<<< 1, 1 >>>();
  DBGCUDASYNC;

  gpuErrchk( cudaPeekAtLastError() );
  SpikeReset_time_ += ( getRealTime() - time_mark );

  if ( n_hosts_ > 1 )
  {
    time_mark = getRealTime();
    ExternalSpikeReset();
    ExternalSpikeReset_time_ += ( getRealTime() - time_mark );
  }

  if ( conn_->getSpikeBufferAlgo() != INPUT_SPIKE_BUFFER_ALGO )
  {
    if ( conn_->getNRevConn() > 0 )
    {
      // time_mark = getRealTime();
      revSpikeReset<<< 1, 1 >>>();
      gpuErrchk( cudaPeekAtLastError() );
      revSpikeBufferUpdate<<< ( GetNLocalNodes() + 1023 ) / 1024, 1024 >>>( GetNLocalNodes() );
      gpuErrchk( cudaPeekAtLastError() );
      unsigned int n_rev_spikes;
      gpuErrchk(
        cudaMemcpy( &n_rev_spikes, conn_->getDevRevSpikeNumPt(), sizeof( unsigned int ), cudaMemcpyDeviceToHost ) );
      if ( n_rev_spikes > 0 )
      {
        switch ( conn_struct_type_ )
        {
        case i_conn12b:
          NestedLoop::Run< 1 >( nested_loop_algo_, n_rev_spikes, conn_->getDevRevSpikeNConnPt() );
          break;
        case i_conn16b:
          NestedLoop::Run< 3 >( nested_loop_algo_, n_rev_spikes, conn_->getDevRevSpikeNConnPt() );
          break;
        default:
          throw ngpu_exception( "Unrecognized connection structure type index" );
        }
      }
      // RevSpikeBufferUpdate_time_ += (getRealTime() - time_mark);
    }
  }

  for ( unsigned int i = 0; i < node_vect_.size(); i++ )
  {
    // if spike times recording is activated for node group...
    if ( node_vect_[ i ]->max_n_rec_spike_times_ > 0 )
    {
      // and if buffering is activated every n_step time steps...
      int n_step = node_vect_[ i ]->rec_spike_times_step_;
      if ( n_step > 0 && ( time_idx % n_step == n_step - 1 ) )
      {
        // extract recorded spike times and put them in buffers
        node_vect_[ i ]->BufferRecSpikeTimes();
      }
    }
  }

  it_++;

  return 0;
}

int
NESTGPU::CreateRecord( std::string file_name, std::string* var_name_arr, int* i_node_arr, int* port_arr, int n_node )
{
  std::vector< BaseNeuron* > neur_vect;
  std::vector< int > i_neur_vect;
  std::vector< int > port_vect;
  std::vector< std::string > var_name_vect;
  for ( int i = 0; i < n_node; i++ )
  {
    var_name_vect.push_back( var_name_arr[ i ] );
    int i_group = node_group_map_[ i_node_arr[ i ] ];
    i_neur_vect.push_back( i_node_arr[ i ] - node_vect_[ i_group ]->i_node_0_ );
    port_vect.push_back( port_arr[ i ] );
    neur_vect.push_back( node_vect_[ i_group ] );
  }

  return multimeter_->CreateRecord( neur_vect, file_name, var_name_vect, i_neur_vect, port_vect );
}

int
NESTGPU::CreateRecord( std::string file_name, std::string* var_name_arr, int* i_node_arr, int n_node )
{
  std::vector< int > port_vect( n_node, 0 );
  return CreateRecord( file_name, var_name_arr, i_node_arr, port_vect.data(), n_node );
}

std::vector< std::vector< float > >*
NESTGPU::GetRecordData( int i_record )
{
  return multimeter_->GetRecordData( i_record );
}

int
NESTGPU::GetNodeSequenceOffset( int i_node, int n_node, int& i_group )
{
  if ( i_node < 0 || ( i_node + n_node > ( int ) node_group_map_.size() ) )
  {
    throw ngpu_exception( "Unrecognized node in getting node sequence offset" );
  }
  i_group = node_group_map_[ i_node ];
  if ( node_group_map_[ i_node + n_node - 1 ] != i_group )
  {
    throw ngpu_exception(
      "Nodes belong to different node groups "
      "in setting parameter" );
  }
  return node_vect_[ i_group ]->i_node_0_;
}

std::vector< int >
NESTGPU::GetNodeArrayWithOffset( int* i_node, int n_node, int& i_group )
{
  int in0 = i_node[ 0 ];
  if ( in0 < 0 || in0 > ( int ) node_group_map_.size() )
  {
    throw ngpu_exception( "Unrecognized node in setting parameter" );
  }
  i_group = node_group_map_[ in0 ];
  int i0 = node_vect_[ i_group ]->i_node_0_;
  std::vector< int > nodes;
  nodes.assign( i_node, i_node + n_node );
  for ( int i = 0; i < n_node; i++ )
  {
    int in = nodes[ i ];
    if ( in < 0 || in >= ( int ) node_group_map_.size() )
    {
      throw ngpu_exception( "Unrecognized node in setting parameter" );
    }
    if ( node_group_map_[ in ] != i_group )
    {
      throw ngpu_exception(
        "Nodes belong to different node groups "
        "in setting parameter" );
    }
    nodes[ i ] -= i0;
  }
  return nodes;
}

int
NESTGPU::SetNeuronParam( int i_node, int n_node, std::string param_name, float val )
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset( i_node, n_node, i_group );

  return node_vect_[ i_group ]->SetScalParam( i_neuron, n_node, param_name, val );
}

int
NESTGPU::SetNeuronParam( int* i_node, int n_node, std::string param_name, float val )
{
  int i_group;
  std::vector< int > nodes = GetNodeArrayWithOffset( i_node, n_node, i_group );
  return node_vect_[ i_group ]->SetScalParam( nodes.data(), n_node, param_name, val );
}

int
NESTGPU::SetNeuronParam( int i_node, int n_node, std::string param_name, float* param, int array_size )
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset( i_node, n_node, i_group );
  if ( node_vect_[ i_group ]->IsPortParam( param_name ) )
  {
    return node_vect_[ i_group ]->SetPortParam( i_neuron, n_node, param_name, param, array_size );
  }
  else
  {
    return node_vect_[ i_group ]->SetArrayParam( i_neuron, n_node, param_name, param, array_size );
  }
}

int
NESTGPU::SetNeuronParam( int* i_node, int n_node, std::string param_name, float* param, int array_size )
{
  int i_group;
  std::vector< int > nodes = GetNodeArrayWithOffset( i_node, n_node, i_group );
  if ( node_vect_[ i_group ]->IsPortParam( param_name ) )
  {
    return node_vect_[ i_group ]->SetPortParam( nodes.data(), n_node, param_name, param, array_size );
  }
  else
  {
    return node_vect_[ i_group ]->SetArrayParam( nodes.data(), n_node, param_name, param, array_size );
  }
}

////////////////////////////////////////////////////////////////////////

int
NESTGPU::SetNeuronScalParamDistr( int i_node, int n_node, std::string param_name )
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset( i_node, n_node, i_group );

  return node_vect_[ i_group ]->SetScalParamDistr( i_neuron, n_node, param_name, distribution_ );
}

int
NESTGPU::SetNeuronScalVarDistr( int i_node, int n_node, std::string var_name )
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset( i_node, n_node, i_group );

  return node_vect_[ i_group ]->SetScalVarDistr( i_neuron, n_node, var_name, distribution_ );
}

int
NESTGPU::SetNeuronPortParamDistr( int i_node, int n_node, std::string param_name )
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset( i_node, n_node, i_group );

  return node_vect_[ i_group ]->SetPortParamDistr( i_neuron, n_node, param_name, distribution_ );
}

int
NESTGPU::SetNeuronPortVarDistr( int i_node, int n_node, std::string var_name )
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset( i_node, n_node, i_group );

  return node_vect_[ i_group ]->SetPortVarDistr( i_neuron, n_node, var_name, distribution_ );
}

int
NESTGPU::SetNeuronPtScalParamDistr( int* i_node, int n_node, std::string param_name )
{
  int i_group;
  std::vector< int > nodes = GetNodeArrayWithOffset( i_node, n_node, i_group );
  return node_vect_[ i_group ]->SetScalParamDistr( nodes.data(), n_node, param_name, distribution_ );
}

int
NESTGPU::SetNeuronPtScalVarDistr( int* i_node, int n_node, std::string var_name )
{
  int i_group;
  std::vector< int > nodes = GetNodeArrayWithOffset( i_node, n_node, i_group );
  return node_vect_[ i_group ]->SetScalVarDistr( nodes.data(), n_node, var_name, distribution_ );
}

int
NESTGPU::SetNeuronPtPortParamDistr( int* i_node, int n_node, std::string param_name )
{
  int i_group;
  std::vector< int > nodes = GetNodeArrayWithOffset( i_node, n_node, i_group );
  return node_vect_[ i_group ]->SetPortParamDistr( nodes.data(), n_node, param_name, distribution_ );
}

int
NESTGPU::SetNeuronPtPortVarDistr( int* i_node, int n_node, std::string var_name )
{
  int i_group;
  std::vector< int > nodes = GetNodeArrayWithOffset( i_node, n_node, i_group );
  return node_vect_[ i_group ]->SetPortVarDistr( nodes.data(), n_node, var_name, distribution_ );
}

int
NESTGPU::SetDistributionIntParam( std::string param_name, int val )
{
  return distribution_->SetIntParam( param_name, val );
}

int
NESTGPU::SetDistributionScalParam( std::string param_name, float val )
{
  return distribution_->SetScalParam( param_name, val );
}

int
NESTGPU::SetDistributionVectParam( std::string param_name, float val, int i )
{
  return distribution_->SetVectParam( param_name, val, i );
}

int
NESTGPU::SetDistributionFloatPtParam( std::string param_name, float* array_pt )
{
  return distribution_->SetFloatPtParam( param_name, array_pt );
}

int
NESTGPU::IsDistributionFloatParam( std::string param_name )
{
  return distribution_->IsFloatParam( param_name );
}

////////////////////////////////////////////////////////////////////////

int
NESTGPU::IsNeuronScalParam( int i_node, std::string param_name )
{
  int i_group;
  GetNodeSequenceOffset( i_node, 1, i_group );

  return node_vect_[ i_group ]->IsScalParam( param_name );
}

int
NESTGPU::IsNeuronPortParam( int i_node, std::string param_name )
{
  int i_group;
  GetNodeSequenceOffset( i_node, 1, i_group );

  return node_vect_[ i_group ]->IsPortParam( param_name );
}

int
NESTGPU::IsNeuronArrayParam( int i_node, std::string param_name )
{
  int i_group;
  GetNodeSequenceOffset( i_node, 1, i_group );

  return node_vect_[ i_group ]->IsArrayParam( param_name );
}

int
NESTGPU::SetNeuronIntVar( int i_node, int n_node, std::string var_name, int val )
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset( i_node, n_node, i_group );

  return node_vect_[ i_group ]->SetIntVar( i_neuron, n_node, var_name, val );
}

int
NESTGPU::SetNeuronIntVar( int* i_node, int n_node, std::string var_name, int val )
{
  int i_group;
  std::vector< int > nodes = GetNodeArrayWithOffset( i_node, n_node, i_group );
  return node_vect_[ i_group ]->SetIntVar( nodes.data(), n_node, var_name, val );
}

int
NESTGPU::SetNeuronVar( int i_node, int n_node, std::string var_name, float val )
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset( i_node, n_node, i_group );

  return node_vect_[ i_group ]->SetScalVar( i_neuron, n_node, var_name, val );
}

int
NESTGPU::SetNeuronVar( int* i_node, int n_node, std::string var_name, float val )
{
  int i_group;
  std::vector< int > nodes = GetNodeArrayWithOffset( i_node, n_node, i_group );
  return node_vect_[ i_group ]->SetScalVar( nodes.data(), n_node, var_name, val );
}

int
NESTGPU::SetNeuronVar( int i_node, int n_node, std::string var_name, float* var, int array_size )
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset( i_node, n_node, i_group );
  if ( node_vect_[ i_group ]->IsPortVar( var_name ) )
  {
    return node_vect_[ i_group ]->SetPortVar( i_neuron, n_node, var_name, var, array_size );
  }
  else
  {
    return node_vect_[ i_group ]->SetArrayVar( i_neuron, n_node, var_name, var, array_size );
  }
}

int
NESTGPU::SetNeuronVar( int* i_node, int n_node, std::string var_name, float* var, int array_size )
{
  int i_group;
  std::vector< int > nodes = GetNodeArrayWithOffset( i_node, n_node, i_group );
  if ( node_vect_[ i_group ]->IsPortVar( var_name ) )
  {
    return node_vect_[ i_group ]->SetPortVar( nodes.data(), n_node, var_name, var, array_size );
  }
  else
  {
    return node_vect_[ i_group ]->SetArrayVar( nodes.data(), n_node, var_name, var, array_size );
  }
}

int
NESTGPU::IsNeuronIntVar( int i_node, std::string var_name )
{
  int i_group;
  GetNodeSequenceOffset( i_node, 1, i_group );

  return node_vect_[ i_group ]->IsIntVar( var_name );
}

int
NESTGPU::IsNeuronScalVar( int i_node, std::string var_name )
{
  int i_group;
  GetNodeSequenceOffset( i_node, 1, i_group );

  return node_vect_[ i_group ]->IsScalVar( var_name );
}

int
NESTGPU::IsNeuronPortVar( int i_node, std::string var_name )
{
  int i_group;
  GetNodeSequenceOffset( i_node, 1, i_group );

  return node_vect_[ i_group ]->IsPortVar( var_name );
}

int
NESTGPU::IsNeuronArrayVar( int i_node, std::string var_name )
{
  int i_group;
  GetNodeSequenceOffset( i_node, 1, i_group );

  return node_vect_[ i_group ]->IsArrayVar( var_name );
}

int
NESTGPU::GetNeuronParamSize( int i_node, std::string param_name )
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset( i_node, 1, i_group );
  if ( node_vect_[ i_group ]->IsArrayParam( param_name ) != 0 )
  {
    return node_vect_[ i_group ]->GetArrayParamSize( i_neuron, param_name );
  }
  else
  {
    return node_vect_[ i_group ]->GetParamSize( param_name );
  }
}

int
NESTGPU::GetNeuronVarSize( int i_node, std::string var_name )
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset( i_node, 1, i_group );
  if ( node_vect_[ i_group ]->IsArrayVar( var_name ) != 0 )
  {
    return node_vect_[ i_group ]->GetArrayVarSize( i_neuron, var_name );
  }
  else
  {
    return node_vect_[ i_group ]->GetVarSize( var_name );
  }
}

float*
NESTGPU::GetNeuronParam( int i_node, int n_node, std::string param_name )
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset( i_node, n_node, i_group );
  if ( node_vect_[ i_group ]->IsScalParam( param_name ) )
  {
    return node_vect_[ i_group ]->GetScalParam( i_neuron, n_node, param_name );
  }
  else if ( node_vect_[ i_group ]->IsPortParam( param_name ) )
  {
    return node_vect_[ i_group ]->GetPortParam( i_neuron, n_node, param_name );
  }
  else if ( node_vect_[ i_group ]->IsArrayParam( param_name ) )
  {
    if ( n_node != 1 )
    {
      throw ngpu_exception(
        "Cannot get array parameters for more than one node"
        "at a time" );
    }
    return node_vect_[ i_group ]->GetArrayParam( i_neuron, param_name );
  }
  else
  {
    throw ngpu_exception( std::string( "Unrecognized parameter " ) + param_name );
  }
}

float*
NESTGPU::GetNeuronParam( int* i_node, int n_node, std::string param_name )
{
  int i_group;
  std::vector< int > nodes = GetNodeArrayWithOffset( i_node, n_node, i_group );
  if ( node_vect_[ i_group ]->IsScalParam( param_name ) )
  {
    return node_vect_[ i_group ]->GetScalParam( nodes.data(), n_node, param_name );
  }
  else if ( node_vect_[ i_group ]->IsPortParam( param_name ) )
  {
    return node_vect_[ i_group ]->GetPortParam( nodes.data(), n_node, param_name );
  }
  else if ( node_vect_[ i_group ]->IsArrayParam( param_name ) )
  {
    if ( n_node != 1 )
    {
      throw ngpu_exception(
        "Cannot get array parameters for more than one node"
        "at a time" );
    }
    return node_vect_[ i_group ]->GetArrayParam( nodes[ 0 ], param_name );
  }
  else
  {
    throw ngpu_exception( std::string( "Unrecognized parameter " ) + param_name );
  }
}

float*
NESTGPU::GetArrayParam( int i_node, std::string param_name )
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset( i_node, 1, i_group );

  return node_vect_[ i_group ]->GetArrayParam( i_neuron, param_name );
}

int*
NESTGPU::GetNeuronIntVar( int i_node, int n_node, std::string var_name )
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset( i_node, n_node, i_group );
  if ( node_vect_[ i_group ]->IsIntVar( var_name ) )
  {
    return node_vect_[ i_group ]->GetIntVar( i_neuron, n_node, var_name );
  }
  else
  {
    throw ngpu_exception( std::string( "Unrecognized integer variable " ) + var_name );
  }
}

int*
NESTGPU::GetNeuronIntVar( int* i_node, int n_node, std::string var_name )
{
  int i_group;
  std::vector< int > nodes = GetNodeArrayWithOffset( i_node, n_node, i_group );
  if ( node_vect_[ i_group ]->IsIntVar( var_name ) )
  {
    return node_vect_[ i_group ]->GetIntVar( nodes.data(), n_node, var_name );
  }
  else
  {
    throw ngpu_exception( std::string( "Unrecognized variable " ) + var_name );
  }
}

float*
NESTGPU::GetNeuronVar( int i_node, int n_node, std::string var_name )
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset( i_node, n_node, i_group );
  if ( node_vect_[ i_group ]->IsScalVar( var_name ) )
  {
    return node_vect_[ i_group ]->GetScalVar( i_neuron, n_node, var_name );
  }
  else if ( node_vect_[ i_group ]->IsPortVar( var_name ) )
  {
    return node_vect_[ i_group ]->GetPortVar( i_neuron, n_node, var_name );
  }
  else if ( node_vect_[ i_group ]->IsArrayVar( var_name ) )
  {
    if ( n_node != 1 )
    {
      throw ngpu_exception(
        "Cannot get array variables for more than one node"
        "at a time" );
    }
    return node_vect_[ i_group ]->GetArrayVar( i_neuron, var_name );
  }
  else
  {
    throw ngpu_exception( std::string( "Unrecognized variable " ) + var_name );
  }
}

float*
NESTGPU::GetNeuronVar( int* i_node, int n_node, std::string var_name )
{
  int i_group;
  std::vector< int > nodes = GetNodeArrayWithOffset( i_node, n_node, i_group );
  if ( node_vect_[ i_group ]->IsScalVar( var_name ) )
  {
    return node_vect_[ i_group ]->GetScalVar( nodes.data(), n_node, var_name );
  }
  else if ( node_vect_[ i_group ]->IsPortVar( var_name ) )
  {
    return node_vect_[ i_group ]->GetPortVar( nodes.data(), n_node, var_name );
  }
  else if ( node_vect_[ i_group ]->IsArrayVar( var_name ) )
  {
    if ( n_node != 1 )
    {
      throw ngpu_exception(
        "Cannot get array variables for more than one node"
        "at a time" );
    }
    return node_vect_[ i_group ]->GetArrayVar( nodes[ 0 ], var_name );
  }
  else
  {
    throw ngpu_exception( std::string( "Unrecognized variable " ) + var_name );
  }
}

float*
NESTGPU::GetArrayVar( int i_node, std::string var_name )
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset( i_node, 1, i_group );

  return node_vect_[ i_group ]->GetArrayVar( i_neuron, var_name );
}

std::string
NESTGPU::HostIdStr()
{
  if ( n_hosts_ > 1 )
  {
    return std::string( "Host " ) + std::to_string( this_host_ ) + " : ";
  }
  else
  {
    return "";
  }
}

size_t
NESTGPU::getCUDAMemHostUsed()
{
  return cuda_error_ns::mem_used_;
}

size_t
NESTGPU::getCUDAMemHostPeak()
{
  return cuda_error_ns::mem_max_;
}

size_t
NESTGPU::getCUDAMemTotal()
{
  size_t mem_free;
  size_t mem_total;
  cudaError_t cuda_status = cudaMemGetInfo( &mem_free, &mem_total );
  if ( cuda_status != cudaSuccess )
  {
    throw ngpu_exception( std::string( "CUDA error in getCUDAMemTotal: " ) + cudaGetErrorString( cuda_status ) );
  }

  return mem_total;
}

size_t
NESTGPU::getCUDAMemFree()
{
  size_t mem_free;
  size_t mem_total;
  cudaError_t cuda_status = cudaMemGetInfo( &mem_free, &mem_total );
  if ( cuda_status != cudaSuccess )
  {
    throw ngpu_exception( std::string( "CUDA error in getCUDAMemFree: " ) + cudaGetErrorString( cuda_status ) );
  }

  return mem_free;
}

unsigned int*
NESTGPU::RandomInt( size_t n )
{
  return curand_int( *random_generator_, n );
}

float*
NESTGPU::RandomUniform( size_t n )
{
  return curand_uniform( *random_generator_, n );
}

float*
NESTGPU::RandomNormal( size_t n, float mean, float stddev )
{
  return curand_normal( *random_generator_, n, mean, stddev );
}

float*
NESTGPU::RandomNormalClipped( size_t n, float mean, float stddev, float vmin, float vmax, float vstep )
{
  const float epsi = 1.0e-6;

  n = ( n / 4 + 1 ) * 4;
  int n_extra = n / 10;
  n_extra = ( n_extra / 4 + 1 ) * 4;
  if ( n_extra < 1024 )
  {
    n_extra = 1024;
  }
  int i_extra = 0;
  float* arr = curand_normal( *random_generator_, n, mean, stddev );
  float* arr_extra = nullptr;
  for ( size_t i = 0; i < n; i++ )
  {
    while ( arr[ i ] < vmin || arr[ i ] > vmax )
    {
      if ( i_extra == 0 )
      {
        arr_extra = curand_normal( *random_generator_, n_extra, mean, stddev );
      }
      arr[ i ] = arr_extra[ i_extra ];
      i_extra++;
      if ( i_extra == n_extra )
      {
        i_extra = 0;
        delete[] ( arr_extra );
        arr_extra = nullptr;
      }
    }
  }
  if ( arr_extra != nullptr )
  {
    delete[] ( arr_extra );
  }
  if ( vstep > stddev * epsi )
  {
    for ( size_t i = 0; i < n; i++ )
    {
      arr[ i ] = vmin + vstep * round( ( arr[ i ] - vmin ) / vstep );
    }
  }

  return arr;
}

std::vector< std::string >
NESTGPU::GetIntVarNames( int i_node )
{
  if ( i_node < 0 || i_node > ( int ) node_group_map_.size() )
  {
    throw ngpu_exception( "Unrecognized node in reading variable names" );
  }
  int i_group = node_group_map_[ i_node ];

  return node_vect_[ i_group ]->GetIntVarNames();
}

std::vector< std::string >
NESTGPU::GetScalVarNames( int i_node )
{
  if ( i_node < 0 || i_node > ( int ) node_group_map_.size() )
  {
    throw ngpu_exception( "Unrecognized node in reading variable names" );
  }
  int i_group = node_group_map_[ i_node ];

  return node_vect_[ i_group ]->GetScalVarNames();
}

int
NESTGPU::GetNIntVar( int i_node )
{
  if ( i_node < 0 || i_node > ( int ) node_group_map_.size() )
  {
    throw ngpu_exception(
      "Unrecognized node in reading number of "
      "variables" );
  }
  int i_group = node_group_map_[ i_node ];

  return node_vect_[ i_group ]->GetNIntVar();
}

int
NESTGPU::GetNScalVar( int i_node )
{
  if ( i_node < 0 || i_node > ( int ) node_group_map_.size() )
  {
    throw ngpu_exception(
      "Unrecognized node in reading number of "
      "variables" );
  }
  int i_group = node_group_map_[ i_node ];

  return node_vect_[ i_group ]->GetNScalVar();
}

std::vector< std::string >
NESTGPU::GetPortVarNames( int i_node )
{
  if ( i_node < 0 || i_node > ( int ) node_group_map_.size() )
  {
    throw ngpu_exception( "Unrecognized node in reading variable names" );
  }
  int i_group = node_group_map_[ i_node ];

  return node_vect_[ i_group ]->GetPortVarNames();
}

int
NESTGPU::GetNPortVar( int i_node )
{
  if ( i_node < 0 || i_node > ( int ) node_group_map_.size() )
  {
    throw ngpu_exception(
      "Unrecognized node in reading number of "
      "variables" );
  }
  int i_group = node_group_map_[ i_node ];

  return node_vect_[ i_group ]->GetNPortVar();
}

std::vector< std::string >
NESTGPU::GetScalParamNames( int i_node )
{
  if ( i_node < 0 || i_node > ( int ) node_group_map_.size() )
  {
    throw ngpu_exception( "Unrecognized node in reading parameter names" );
  }
  int i_group = node_group_map_[ i_node ];

  return node_vect_[ i_group ]->GetScalParamNames();
}

int
NESTGPU::GetNScalParam( int i_node )
{
  if ( i_node < 0 || i_node > ( int ) node_group_map_.size() )
  {
    throw ngpu_exception(
      "Unrecognized node in reading number of "
      "parameters" );
  }
  int i_group = node_group_map_[ i_node ];

  return node_vect_[ i_group ]->GetNScalParam();
}

std::vector< std::string >
NESTGPU::GetPortParamNames( int i_node )
{
  if ( i_node < 0 || i_node > ( int ) node_group_map_.size() )
  {
    throw ngpu_exception( "Unrecognized node in reading parameter names" );
  }
  int i_group = node_group_map_[ i_node ];

  return node_vect_[ i_group ]->GetPortParamNames();
}

int
NESTGPU::GetNPortParam( int i_node )
{
  if ( i_node < 0 || i_node > ( int ) node_group_map_.size() )
  {
    throw ngpu_exception(
      "Unrecognized node in reading number of "
      "parameters" );
  }
  int i_group = node_group_map_[ i_node ];

  return node_vect_[ i_group ]->GetNPortParam();
}

std::vector< std::string >
NESTGPU::GetArrayParamNames( int i_node )
{
  if ( i_node < 0 || i_node > ( int ) node_group_map_.size() )
  {
    throw ngpu_exception( "Unrecognized node in reading array parameter names" );
  }
  int i_group = node_group_map_[ i_node ];

  return node_vect_[ i_group ]->GetArrayParamNames();
}

int
NESTGPU::GetNArrayParam( int i_node )
{
  if ( i_node < 0 || i_node > ( int ) node_group_map_.size() )
  {
    throw ngpu_exception(
      "Unrecognized node in reading number of array "
      "parameters" );
  }
  int i_group = node_group_map_[ i_node ];

  return node_vect_[ i_group ]->GetNArrayParam();
}

std::vector< std::string >
NESTGPU::GetArrayVarNames( int i_node )
{
  if ( i_node < 0 || i_node > ( int ) node_group_map_.size() )
  {
    throw ngpu_exception( "Unrecognized node in reading array variable names" );
  }
  int i_group = node_group_map_[ i_node ];

  return node_vect_[ i_group ]->GetArrayVarNames();
}

int
NESTGPU::GetNArrayVar( int i_node )
{
  if ( i_node < 0 || i_node > ( int ) node_group_map_.size() )
  {
    throw ngpu_exception(
      "Unrecognized node in reading number of array "
      "variables" );
  }
  int i_group = node_group_map_[ i_node ];

  return node_vect_[ i_group ]->GetNArrayVar();
}

int
NESTGPU::GetConnectionFloatParamIndex( std::string param_name )
{
  return conn_->getConnectionFloatParamIndex( param_name );
}

int
NESTGPU::GetConnectionIntParamIndex( std::string param_name )
{
  return conn_->getConnectionIntParamIndex( param_name );
}

int
NESTGPU::IsConnectionFloatParam( std::string param_name )
{
  return conn_->isConnectionFloatParam( param_name );
}

int
NESTGPU::IsConnectionIntParam( std::string param_name )
{
  return conn_->isConnectionIntParam( param_name );
}

int
NESTGPU::GetConnectionFloatParam( int64_t* conn_ids, int64_t n_conn, float* h_param_arr, std::string param_name )
{
  return conn_->getConnectionFloatParam( conn_ids, n_conn, h_param_arr, param_name );
}

int
NESTGPU::GetConnectionIntParam( int64_t* conn_ids, int64_t n_conn, int* h_param_arr, std::string param_name )
{
  return conn_->getConnectionIntParam( conn_ids, n_conn, h_param_arr, param_name );
}

int
NESTGPU::SetConnectionFloatParamDistr( int64_t* conn_ids, int64_t n_conn, std::string param_name )
{
  return conn_->setConnectionFloatParamDistr( conn_ids, n_conn, param_name );
}

int
NESTGPU::SetConnectionFloatParam( int64_t* conn_ids, int64_t n_conn, float val, std::string param_name )
{
  return conn_->setConnectionFloatParam( conn_ids, n_conn, val, param_name );
}

int
NESTGPU::SetConnectionIntParamArr( int64_t* conn_ids, int64_t n_conn, int* h_param_arr, std::string param_name )
{
  return conn_->setConnectionIntParamArr( conn_ids, n_conn, h_param_arr, param_name );
}

int
NESTGPU::SetConnectionIntParam( int64_t* conn_ids, int64_t n_conn, int val, std::string param_name )
{
  return conn_->setConnectionIntParam( conn_ids, n_conn, val, param_name );
}

int
NESTGPU::GetConnectionStatus( int64_t* conn_ids,
  int64_t n_conn,
  inode_t* source,
  inode_t* target,
  int* port,
  int* syn_group,
  float* delay,
  float* weight )
{
  return conn_->getConnectionStatus( conn_ids, n_conn, source, target, port, syn_group, delay, weight );
}

int64_t*
NESTGPU::GetConnections( inode_t i_source,
  inode_t n_source,
  inode_t i_target,
  inode_t n_target,
  int syn_group,
  int64_t* n_conn )
{
  if ( n_source <= 0 )
  {
    i_source = 0;
    // gets also connections from image neurons
    n_source = GetNTotalNodes();
  }
  if ( n_target <= 0 )
  {
    i_target = 0;
    n_target = GetNLocalNodes();
  }
  inode_t* i_source_pt = new inode_t[ n_source ];
  for ( inode_t i = 0; i < n_source; i++ )
  {
    i_source_pt[ i ] = i_source + i;
  }
  inode_t* i_target_pt = new inode_t[ n_target ];
  for ( inode_t i = 0; i < n_target; i++ )
  {
    i_target_pt[ i ] = i_target + i;
  }

  int64_t* conn_ids = conn_->getConnections( i_source_pt, n_source, i_target_pt, n_target, syn_group, n_conn );

  delete[] i_source_pt;
  delete[] i_target_pt;

  return conn_ids;
}

int64_t*
NESTGPU::GetConnections( inode_t* i_source_pt,
  inode_t n_source,
  inode_t i_target,
  inode_t n_target,
  int syn_group,
  int64_t* n_conn )
{
  if ( n_target <= 0 )
  {
    i_target = 0;
    n_target = GetNLocalNodes();
  }
  inode_t* i_target_pt = new inode_t[ n_target ];
  for ( inode_t i = 0; i < n_target; i++ )
  {
    i_target_pt[ i ] = i_target + i;
  }

  int64_t* conn_ids = conn_->getConnections( i_source_pt, n_source, i_target_pt, n_target, syn_group, n_conn );

  delete[] i_target_pt;

  return conn_ids;
}

int64_t*
NESTGPU::GetConnections( inode_t i_source,
  inode_t n_source,
  inode_t* i_target_pt,
  inode_t n_target,
  int syn_group,
  int64_t* n_conn )
{
  if ( n_source <= 0 )
  {
    i_source = 0;
    //  gets also connections from image neurons
    n_source = GetNTotalNodes();
  }
  inode_t* i_source_pt = new inode_t[ n_source ];
  for ( inode_t i = 0; i < n_source; i++ )
  {
    i_source_pt[ i ] = i_source + i;
  }

  int64_t* conn_ids = conn_->getConnections( i_source_pt, n_source, i_target_pt, n_target, syn_group, n_conn );

  delete[] i_source_pt;

  return conn_ids;
}

int64_t*
NESTGPU::GetConnections( inode_t* i_source_pt,
  inode_t n_source,
  inode_t* i_target_pt,
  inode_t n_target,
  int syn_group,
  int64_t* n_conn )
{
  int64_t* conn_ids = conn_->getConnections( i_source_pt, n_source, i_target_pt, n_target, syn_group, n_conn );

  return conn_ids;
}

int64_t*
NESTGPU::GetConnections( NodeSeq source, NodeSeq target, int syn_group, int64_t* n_conn )
{
  return GetConnections( source.i0, source.n, target.i0, target.n, syn_group, n_conn );
}

int64_t*
NESTGPU::GetConnections( std::vector< inode_t > source, NodeSeq target, int syn_group, int64_t* n_conn )
{
  return GetConnections( source.data(), source.size(), target.i0, target.n, syn_group, n_conn );
}

int64_t*
NESTGPU::GetConnections( NodeSeq source, std::vector< inode_t > target, int syn_group, int64_t* n_conn )
{
  return GetConnections( source.i0, source.n, target.data(), target.size(), syn_group, n_conn );
}

int64_t*
NESTGPU::GetConnections( std::vector< inode_t > source, std::vector< inode_t > target, int syn_group, int64_t* n_conn )
{
  return conn_->getConnections( source.data(), source.size(), target.data(), target.size(), syn_group, n_conn );
}

int
NESTGPU::ActivateSpikeCount( int i_node, int n_node )
{
  CheckUncalibrated( "Spike count must be activated before calibration" );
  int i_group;
  int i_node_0 = GetNodeSequenceOffset( i_node, n_node, i_group );
  if ( i_node_0 != i_node || node_vect_[ i_group ]->n_node_ != n_node )
  {
    throw ngpu_exception(
      "Spike count must be activated for all and only "
      " the nodes of the same group" );
  }
  // std::cout << "Activating spike count for group " << i_group << "\n";
  node_vect_[ i_group ]->ActivateSpikeCount();

  return 0;
}

int
NESTGPU::ActivateRecSpikeTimes( int i_node, int n_node, int max_n_rec_spike_times )
{
  CheckUncalibrated(
    "Spike time recording must be activated "
    "before calibration" );
  int i_group;
  int i_node_0 = GetNodeSequenceOffset( i_node, n_node, i_group );
  if ( i_node_0 != i_node || node_vect_[ i_group ]->n_node_ != n_node )
  {
    throw ngpu_exception(
      "Spike count must be activated for all and only "
      " the nodes of the same group" );
  }
  // std::cout << "Activating spike time recording for group " << i_group << "\n";
  node_vect_[ i_group ]->ActivateRecSpikeTimes( max_n_rec_spike_times );

  return 0;
}

int
NESTGPU::SetRecSpikeTimesStep( int i_node, int n_node, int rec_spike_times_step )
{
  int i_group;
  int i_node_0 = GetNodeSequenceOffset( i_node, n_node, i_group );
  if ( i_node_0 != i_node || node_vect_[ i_group ]->n_node_ != n_node )
  {
    throw ngpu_exception(
      "Time step for buffering spike time recording "
      "must be set for all and only "
      "the nodes of the same group" );
  }
  node_vect_[ i_group ]->SetRecSpikeTimesStep( rec_spike_times_step );

  return 0;
}

// get number of recorded spike times for a node
int
NESTGPU::GetNRecSpikeTimes( int i_node )
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset( i_node, 1, i_group );
  return node_vect_[ i_group ]->GetNRecSpikeTimes( i_neuron );
}

// get recorded spike times for node group
int
NESTGPU::GetRecSpikeTimes( int i_node, int n_node, int** n_spike_times_pt, float*** spike_times_pt )
{
  int i_group;
  int i_node_0 = GetNodeSequenceOffset( i_node, n_node, i_group );
  if ( i_node_0 != i_node || node_vect_[ i_group ]->n_node_ != n_node )
  {
    throw ngpu_exception(
      "Spike times must be extracted for all and only "
      " the nodes of the same group" );
  }

  return node_vect_[ i_group ]->GetRecSpikeTimes( n_spike_times_pt, spike_times_pt );
}

int
NESTGPU::PushSpikesToNodes( int n_spikes, int* node_id, float* spike_mul )
{
  /*
  int *d_node_id;
  float *d_spike_mul;
  CUDAMALLOCCTRL("&d_node_id",&d_node_id, n_spikes*sizeof(int));
  CUDAMALLOCCTRL("&d_spike_mul",&d_spike_mul, n_spikes*sizeof(float));
  // Memcpy are synchronized by PushSpikeFromRemote kernel
  gpuErrchk(cudaMemcpyAsync(d_node_id, node_id, n_spikes*sizeof(int),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyAsync(d_spike_mul, spike_mul,
  n_spikes*sizeof(float), cudaMemcpyHostToDevice));
  PushSpikeFromRemote<<<(n_spikes+1023)/1024, 1024>>>(n_spikes, d_node_id,
                                                     d_spike_mul);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  CUDAFREECTRL("d_node_id",d_node_id);
  CUDAFREECTRL("d_spike_mul",d_spike_mul);
  */

  return 0;
}

int
NESTGPU::PushSpikesToNodes( int n_spikes, int* node_id )
{
  /*
  //std::cout << "n_spikes: " << n_spikes << "\n";
  //for (int i=0; i<n_spikes; i++) {
  //  std::cout << node_id[i] << " ";
  //}
  //std::cout << "\n";

  int *d_node_id;
  CUDAMALLOCCTRL("&d_node_id",&d_node_id, n_spikes*sizeof(int));
  // memcopy data transfer is overlapped with PushSpikeFromRemote kernel
  gpuErrchk(cudaMemcpyAsync(d_node_id, node_id, n_spikes*sizeof(int),
                       cudaMemcpyHostToDevice));
  PushSpikeFromRemote<<<(n_spikes+1023)/1024, 1024>>>(n_spikes, d_node_id);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  CUDAFREECTRL("d_node_id",d_node_id);
  */

  return 0;
}

int
NESTGPU::GetExtNeuronInputSpikes( int* n_spikes, int** node, int** port, float** spike_mul, bool include_zeros )
{
  ext_neuron_input_spike_node_.clear();
  ext_neuron_input_spike_port_.clear();
  ext_neuron_input_spike_mul_.clear();

  for ( unsigned int i = 0; i < node_vect_.size(); i++ )
  {
    if ( node_vect_[ i ]->IsExtNeuron() )
    {
      int n_node;
      int n_port;
      float* sh = node_vect_[ i ]->GetExtNeuronInputSpikes( &n_node, &n_port );
      for ( int i_neur = 0; i_neur < n_node; i_neur++ )
      {
        int i_node = i_neur + node_vect_[ i ]->i_node_0_;
        for ( int i_port = 0; i_port < n_port; i_port++ )
        {
          int j = i_neur * n_port + i_port;
          if ( sh[ j ] != 0.0 || include_zeros )
          {
            ext_neuron_input_spike_node_.push_back( i_node );
            ext_neuron_input_spike_port_.push_back( i_port );
            ext_neuron_input_spike_mul_.push_back( sh[ j ] );
          }
        }
      }
    }
  }
  *n_spikes = ext_neuron_input_spike_node_.size();
  *node = ext_neuron_input_spike_node_.data();
  *port = ext_neuron_input_spike_port_.data();
  *spike_mul = ext_neuron_input_spike_mul_.data();

  return 0;
}

int
NESTGPU::SetNeuronGroupParam( int i_node, int n_node, std::string param_name, float val )
{
  int i_group;
  int i_node_0 = GetNodeSequenceOffset( i_node, n_node, i_group );
  if ( i_node_0 != i_node || node_vect_[ i_group ]->n_node_ != n_node )
  {
    throw ngpu_exception(std::string("Group parameter ") + param_name +
                         " can only be set for all and only "
                         " the nodes of the same group");
  }
  return node_vect_[ i_group ]->SetGroupParam( param_name, val );
}

int
NESTGPU::IsNeuronGroupParam( int i_node, std::string param_name )
{
  int i_group;
  GetNodeSequenceOffset( i_node, 1, i_group );

  return node_vect_[ i_group ]->IsGroupParam( param_name );
}

float
NESTGPU::GetNeuronGroupParam( int i_node, std::string param_name )
{
  int i_group;
  GetNodeSequenceOffset( i_node, 1, i_group );

  return node_vect_[ i_group ]->GetGroupParam( param_name );
}

std::vector< std::string >
NESTGPU::GetGroupParamNames( int i_node )
{
  if ( i_node < 0 || i_node > ( int ) node_group_map_.size() )
  {
    throw ngpu_exception( "Unrecognized node in reading group parameter names" );
  }
  int i_group = node_group_map_[ i_node ];

  return node_vect_[ i_group ]->GetGroupParamNames();
}

int
NESTGPU::GetNGroupParam( int i_node )
{
  if ( i_node < 0 || i_node > ( int ) node_group_map_.size() )
  {
    throw ngpu_exception(
      "Unrecognized node in reading number of "
      "group parameters" );
  }
  int i_group = node_group_map_[ i_node ];

  return node_vect_[ i_group ]->GetNGroupParam();
}

int
NESTGPU::GetNBoolParam()
{
  return N_KERNEL_BOOL_PARAM;
}

std::vector< std::string >
NESTGPU::GetBoolParamNames()
{
  std::vector< std::string > param_name_vect;
  for ( int i = 0; i < N_KERNEL_BOOL_PARAM; i++ )
  {
    param_name_vect.push_back( kernel_bool_param_name[ i ] );
  }

  return param_name_vect;
}

bool
NESTGPU::IsBoolParam( std::string param_name )
{
  int i_param;
  for ( i_param = 0; i_param < N_KERNEL_BOOL_PARAM; i_param++ )
  {
    if ( param_name == kernel_bool_param_name[ i_param ] )
    {
      return true;
    }
  }
  return false;
}

int
NESTGPU::GetBoolParamIdx( std::string param_name )
{
  int i_param;
  for ( i_param = 0; i_param < N_KERNEL_BOOL_PARAM; i_param++ )
  {
    if ( param_name == kernel_bool_param_name[ i_param ] )
    {
      break;
    }
  }
  if ( i_param == N_KERNEL_BOOL_PARAM )
  {
    throw ngpu_exception( std::string( "Unrecognized kernel boolean parameter " ) + param_name );
  }

  return i_param;
}

bool
NESTGPU::GetBoolParam( std::string param_name )
{
  int i_param = GetBoolParamIdx( param_name );
  switch ( i_param )
  {
  case i_print_time:
    return print_time_;
  case i_remove_conn_key:
    return remove_conn_key_;
  case i_remote_spike_mul:
    return remote_spike_mul_;
  default:
    throw ngpu_exception( std::string( "Unrecognized kernel boolean parameter " ) + param_name );
  }
}

int
NESTGPU::SetBoolParam( std::string param_name, bool val )
{
  int i_param = GetBoolParamIdx( param_name );

  switch ( i_param )
  {
  case i_print_time:
    print_time_ = val;
    break;
  case i_remove_conn_key:
    remove_conn_key_ = val;
    break;
  case i_remote_spike_mul:
    remote_spike_mul_ = val;
    break;
  default:
    throw ngpu_exception( std::string( "Unrecognized kernel boolean parameter " ) + param_name );
  }

  return 0;
}

int
NESTGPU::GetNFloatParam()
{
  return N_KERNEL_FLOAT_PARAM;
}

std::vector< std::string >
NESTGPU::GetFloatParamNames()
{
  std::vector< std::string > param_name_vect;
  for ( int i = 0; i < N_KERNEL_FLOAT_PARAM; i++ )
  {
    param_name_vect.push_back( kernel_float_param_name[ i ] );
  }

  return param_name_vect;
}

bool
NESTGPU::IsFloatParam( std::string param_name )
{
  int i_param;
  for ( i_param = 0; i_param < N_KERNEL_FLOAT_PARAM; i_param++ )
  {
    if ( param_name == kernel_float_param_name[ i_param ] )
    {
      return true;
    }
  }
  return false;
}

int
NESTGPU::GetFloatParamIdx( std::string param_name )
{
  int i_param;
  for ( i_param = 0; i_param < N_KERNEL_FLOAT_PARAM; i_param++ )
  {
    if ( param_name == kernel_float_param_name[ i_param ] )
    {
      break;
    }
  }
  if ( i_param == N_KERNEL_FLOAT_PARAM )
  {
    throw ngpu_exception( std::string( "Unrecognized kernel float parameter " ) + param_name );
  }

  return i_param;
}

float
NESTGPU::GetFloatParam( std::string param_name )
{
  int i_param = GetFloatParamIdx( param_name );
  switch ( i_param )
  {
  case i_time_resolution:
    return time_resolution_;
  case i_max_spike_num_fact:
    return max_spike_num_fact_;
  case i_max_spike_per_host_fact:
    return max_spike_per_host_fact_;
  case i_max_remote_spike_num_fact:
    return max_remote_spike_num_fact_;
  default:
    throw ngpu_exception( std::string( "Unrecognized kernel float parameter " ) + param_name );
  }
}

int
NESTGPU::SetFloatParam( std::string param_name, float val )
{
  int i_param = GetFloatParamIdx( param_name );

  switch ( i_param )
  {
  case i_time_resolution:
    time_resolution_ = val;
    conn_->setTimeResolution( time_resolution_ );
    break;
  case i_max_spike_num_fact:
    max_spike_num_fact_ = val;
    break;
  case i_max_spike_per_host_fact:
    max_spike_per_host_fact_ = val;
    break;
  case i_max_remote_spike_num_fact:
    max_remote_spike_num_fact_ = val;
    break;
  default:
    throw ngpu_exception( std::string( "Unrecognized kernel float parameter " ) + param_name );
  }

  return 0;
}

int
NESTGPU::GetNIntParam()
{
  return N_KERNEL_INT_PARAM;
}

std::vector< std::string >
NESTGPU::GetIntParamNames()
{
  std::vector< std::string > param_name_vect;
  for ( int i = 0; i < N_KERNEL_INT_PARAM; i++ )
  {
    param_name_vect.push_back( kernel_int_param_name[ i ] );
  }

  return param_name_vect;
}

bool
NESTGPU::IsIntParam( std::string param_name )
{
  int i_param;
  for ( i_param = 0; i_param < N_KERNEL_INT_PARAM; i_param++ )
  {
    if ( param_name == kernel_int_param_name[ i_param ] )
    {
      return true;
    }
  }

  return false;
}

int
NESTGPU::GetIntParamIdx( std::string param_name )
{
  int i_param;
  for ( i_param = 0; i_param < N_KERNEL_INT_PARAM; i_param++ )
  {
    if ( param_name == kernel_int_param_name[ i_param ] )
    {
      break;
    }
  }
  if ( i_param == N_KERNEL_INT_PARAM )
  {
    throw ngpu_exception( std::string( "Unrecognized kernel int parameter " ) + param_name );
  }

  return i_param;
}

int
NESTGPU::GetIntParam( std::string param_name )
{
  int i_param = GetIntParamIdx( param_name );
  switch ( i_param )
  {
  case i_rnd_seed:
    return kernel_seed_;
  case i_verbosity_level:
    return verbosity_level_;
  case i_max_spike_buffer_size:
    return max_spike_buffer_size_;
  case i_max_node_n_bits:
    return conn_->getMaxNodeNBits();
  case i_max_syn_n_bits:
    return conn_->getMaxSynNBits();
  case i_max_delay_n_bits:
    return conn_->getMaxDelayNBits();
  case i_conn_struct_type:
    return conn_struct_type_;
  case i_spike_buffer_algo:
    return conn_->getSpikeBufferAlgo();
  default:
    throw ngpu_exception( std::string( "Unrecognized kernel int parameter " ) + param_name );
  }
}

int
NESTGPU::SetIntParam( std::string param_name, int val )
{
  int i_param = GetIntParamIdx( param_name );
  switch ( i_param )
  {
  case i_rnd_seed:
    SetRandomSeed( val );
    break;
  case i_verbosity_level:
    SetVerbosityLevel( val );
    if ( val >= 5 )
    {
      cuda_error_ns::verbose_ = 1;
    }
    else
    {
      cuda_error_ns::verbose_ = 0;
    }
    break;
  case i_max_spike_buffer_size:
    SetMaxSpikeBufferSize( val );
    break;
  case i_max_node_n_bits:
    conn_->setMaxNodeNBits( val );
    break;
  case i_max_syn_n_bits:
    conn_->setMaxSynNBits( val );
    break;
  case i_max_delay_n_bits:
    conn_->setMaxDelayNBits( val );
    break;
  case i_conn_struct_type:
    if ( conn_struct_type_ != val )
    {
      setConnStructType( val );
    }
    break;
  case i_spike_buffer_algo:
    conn_->setSpikeBufferAlgo( val );
    break;
  default:
    throw ngpu_exception( std::string( "Unrecognized kernel int parameter " ) + param_name );
  }

  return 0;
}

RemoteNodeSeq
NESTGPU::RemoteCreate( int i_host, std::string model_name, inode_t n_nodes /*=1*/, int n_ports /*=1*/ )
{
  if ( !create_flag_ )
  {
    create_flag_ = true;
    start_real_time_ = getRealTime();
  }
  if ( n_hosts_ > 1 )
  {
    if ( i_host >= n_hosts_ )
    {
      throw ngpu_exception( "Invalid host index in RemoteCreate" );
    }
    NodeSeq node_seq( n_remote_nodes_[ i_host ], n_nodes );
    n_remote_nodes_[ i_host ] += n_nodes;
    if ( i_host == this_host_ )
    {
      NodeSeq check_node_seq = _Create( model_name, n_nodes, n_ports );
      if ( check_node_seq.i0 != node_seq.i0 )
      {
        throw ngpu_exception(
          "Inconsistency in number of nodes in local"
          " and remote representation of the host." );
      }
    }
    return RemoteNodeSeq( i_host, node_seq );
  }
  else
  {
    throw ngpu_exception( "RemoteCreate requires at least two hosts" );
  }
}

// Method that creates a group of hosts for remote spike communication (i.e. a group of MPI processes)
// host_arr: array of host inexes, n_hosts: nomber of hosts in the group
int NESTGPU::CreateHostGroup(int *host_arr, int n_hosts) {
  return conn_->CreateHostGroup(host_arr, n_hosts);
}
