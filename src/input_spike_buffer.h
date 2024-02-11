#ifndef INPUT_SPIKE_BUFFER_H
#define INPUT_SPIKE_BUFFER_H

#include <iostream>
#include <stdio.h>
// The following line must be skipped by clang-tidy to avoid errors
// which are not related to our code but to the CUB CUDA library
//<BEGIN-CLANG-TIDY-SKIP>//
#include <cub/cub.cuh>
//<END-CLANG-TIDY-SKIP>//

#include "connect.h"
#include "cuda_error.h"

extern __constant__ long long NESTGPUTimeIdx;

namespace input_spike_buffer_ns
{
// algorithm for spike buffering and delivery
extern __device__ int algo_;

// number of input ports of each (local) node
extern __device__ int* n_input_ports_; // [n_local_nodes]

// two-dimensional array of maximum delay among the incoming connections of each input port
// of each (local) target node
extern __device__ int** max_input_delay_; // [n_local_nodes][n_input_ports[i_node]]

// three-dimensional input spike buffer array
// [n_local_nodes][n_input_ports[i_node]][n_slots[i_target][i_port]]
extern __device__ double*** input_spike_buffer_;

// index of the first connection outgoing from each local node (-1 for no connections)
// [n_local_nodes]
extern __device__ int64_t* first_out_connection_;

// array of the first connection of each spike emitted at current time step
// [n_all_nodes*time_resolution*avg_max_firing_rate]
extern __device__ int64_t* spike_first_connection_;

// number of spikes emitted at current time step
extern __device__ int* n_spikes_;

// Initialize input spike buffer pointers in device memory
__global__ void initInputSpikeBufferPointersKernel( int* n_input_ports,
  int** max_input_delay,
  double*** input_spike_buffer,
  int64_t* first_out_connection,
  int64_t* spike_first_connection,
  int* n_spikes );

// Evaluates the number of input ports of each (local) target node
template < class ConnKeyT, class ConnStructT >
__global__ void
getNInputPortsKernel( int64_t n_conn, int* n_input_ports )
{
  int64_t i_conn = ( int64_t ) blockIdx.x * blockDim.x + threadIdx.x;
  if ( i_conn >= n_conn )
  {
    return;
  }
  // get connection block and relative index within the block
  uint i_block = ( uint ) ( i_conn / ConnBlockSize );
  int64_t i_block_conn = i_conn % ConnBlockSize;

  // get references to key-structure pair representing the connection
  ConnKeyT& conn_key = ( ( ConnKeyT** ) ConnKeyArray )[ i_block ][ i_block_conn ];
  ConnStructT& conn_struct = ( ( ConnStructT** ) ConnStructArray )[ i_block ][ i_block_conn ];

  // MAYBE CAN BE IMPROVED BY USING A BIT TO SPECIFY IF A CONNECTION IS DIRECT
  // get target node index and delay
  inode_t i_target = getConnTarget< ConnStructT >( conn_struct );
  int i_port = getConnPort< ConnKeyT, ConnStructT >( conn_key, conn_struct );

  // printf("ic: %ld\tit: %d\tip: %d\n", i_conn, i_target, i_port);

  // atomic operation to avoid conflicts in memory access
  // if i_port + 1 is larger than current number of ports evaluated for target node, update it
  atomicMax( &n_input_ports[ i_target ], i_port + 1 );
  // printf("ic1: %ld\tit: %d\tip: %d\n", i_conn, i_target, n_input_ports[i_target]);
}

__global__ void initMaxInputDelayArrayKernel( uint n_local_nodes,
  int** max_input_delay,
  int* max_input_delay_1d,
  int64_t* n_input_ports_cumul );


// Evaluates the maximum delay among the incoming connections of each (local) target node and receptor port
template < class ConnKeyT, class ConnStructT >
__global__ void
getMaxInputDelayKernel( int64_t n_conn, int** max_input_delay )
{
  int64_t i_conn = ( int64_t ) blockIdx.x * blockDim.x + threadIdx.x;
  if ( i_conn >= n_conn )
  {
    return;
  }
  // get connection block and relative index within the block
  uint i_block = ( uint ) ( i_conn / ConnBlockSize );
  int64_t i_block_conn = i_conn % ConnBlockSize;

  // get references to key-structure pair representing the connection
  ConnKeyT& conn_key = ( ( ConnKeyT** ) ConnKeyArray )[ i_block ][ i_block_conn ];
  ConnStructT& conn_struct = ( ( ConnStructT** ) ConnStructArray )[ i_block ][ i_block_conn ];

  // MAYBE CAN BE IMPROVED BY USING A BIT TO SPECIFY IF A CONNECTION IS DIRECT
  // get target node index and delay
  inode_t i_target = getConnTarget< ConnStructT >( conn_struct );
  int i_port = getConnPort< ConnKeyT, ConnStructT >( conn_key, conn_struct );
  int i_delay = getConnDelay< ConnKeyT >( conn_key );

  // atomic operation to avoid conflicts in memory access
  // if delay is larger than current maximum evaluated for target node, update the maximum
  atomicMax( &max_input_delay[ i_target ][ i_port ], i_delay + 1 );
}


// Evaluates the index of the first outgoing connection of each (local) target node
template < class ConnKeyT >
__global__ void
getFirstOutConnectionKernel( int64_t n_conn, int64_t* first_out_connection )
{
  int64_t i_conn = ( int64_t ) blockIdx.x * blockDim.x + threadIdx.x;
  if ( i_conn >= n_conn )
  {
    return;
  }
  // get connection block and relative index within the block
  uint i_block = ( uint ) ( i_conn / ConnBlockSize );
  int64_t i_block_conn = i_conn % ConnBlockSize;

  // get references to key-structure pair representing the connection
  ConnKeyT& conn_key = ( ( ConnKeyT** ) ConnKeyArray )[ i_block ][ i_block_conn ];

  int i_source = getConnSource< ConnKeyT >( conn_key );
  if ( i_conn >= 1 )
  {
    int64_t i_conn_prev = i_conn - 1; // previous connection
    // get connection block and relative index within the block
    uint i_block_prev = ( uint ) ( i_conn_prev / ConnBlockSize );
    int64_t i_block_conn_prev = i_conn_prev % ConnBlockSize;

    // get references to key-structure pair representing the connection
    ConnKeyT& conn_key_prev = ( ( ConnKeyT** ) ConnKeyArray )[ i_block_prev ][ i_block_conn_prev ];

    int i_source_prev = getConnSource< ConnKeyT >( conn_key_prev );
    if ( i_source_prev == i_source )
    {
      return;
    }
  }
  // if i_conn is 0 or its source node index differs from the previous one
  // then i_conn is the first connection if its source node
  first_out_connection[ i_source ] = i_conn;
}


template < class ConnKeyT, class ConnStructT >
__global__ void
deliverSpikesKernel( int64_t n_conn )
{
  __shared__ inode_t i_source0;
  uint i_spike = blockIdx.x;
  int64_t i_conn0 = input_spike_buffer_ns::spike_first_connection_[ i_spike ];
  if ( i_conn0 < 0 )
  {
    return;
  }
  int64_t i_conn1 = i_conn0 + threadIdx.x;
  if ( i_conn1 > n_conn )
  {
    return;
  }

  int i_block = ( int ) ( i_conn1 / ConnBlockSize );
  int64_t i_block_conn = i_conn1 % ConnBlockSize;
  ConnKeyT conn_key = ( ( ConnKeyT** ) ConnKeyArray )[ i_block ][ i_block_conn ];
  inode_t i_source = getConnSource< ConnKeyT >( conn_key );
  if ( threadIdx.x == 0 )
  {
    i_source0 = i_source;
    // printf("i_source0: %d\n", i_source0);
  }
  __syncthreads();
  for ( int64_t i_conn = i_conn1; i_conn < n_conn; i_conn += blockDim.x )
  {
    if ( i_conn != i_conn1 )
    {
      i_block = ( int ) ( i_conn / ConnBlockSize );
      i_block_conn = i_conn % ConnBlockSize;
      conn_key = ( ( ConnKeyT** ) ConnKeyArray )[ i_block ][ i_block_conn ];
      i_source = getConnSource< ConnKeyT >( conn_key );
    }
    if ( i_source != i_source0 )
    {
      return;
    }


    ConnStructT& conn_struct = ( ( ConnStructT** ) ConnStructArray )[ i_block ][ i_block_conn ];
    inode_t i_target = getConnTarget< ConnStructT >( conn_struct );
    int i_delay = getConnDelay< ConnKeyT >( conn_key );
    int i_port = getConnPort< ConnKeyT, ConnStructT >( conn_key, conn_struct );
    float weight = conn_struct.weight;
    int n_slots = max_input_delay_[ i_target ][ i_port ];
    int i_slot = ( NESTGPUTimeIdx + i_delay ) % n_slots;
    // printf("i_conn: %ld\ti_source: %d\ti_target: %d\ti_port: %d\ti_delay: %d\tw: %f\tn_slots: %d\n", i_conn,
    // i_source, i_target, i_port, i_delay, weight, n_slots); printf("n_slots: %d\ti_delay: %d\ttime_idx: %lld\ti_slot:
    // %d\n", n_slots, i_delay, NESTGPUTimeIdx, i_slot);

    // if (i_target==37) {
    //   printf("deliver i_port: %d\ti_slot: %d\n", i_port, i_slot);
    //   printf("deliver input_spike_buffer: %lf\n",
    //	     input_spike_buffer_[i_target][i_port][i_slot]);
    //   printf("deliver w: %f\n", weight);
    // }
    atomicAdd( &input_spike_buffer_[ i_target ][ i_port ][ i_slot ], weight );
  }
}

template < class ConnKeyT, class ConnStructT >
__global__ void
sendDirectSpikeKernel( curandState* curand_state,
  long long time_idx,
  float* mu_arr,
  ConnKeyT* poiss_key_array,
  int64_t n_conn,
  int64_t i_conn_0,
  int64_t block_size,
  int n_node,
  int max_delay )
{
  int64_t blockId = ( int64_t ) blockIdx.y * gridDim.x + blockIdx.x;
  int64_t i_conn_rel = blockId * blockDim.x + threadIdx.x;
  if ( i_conn_rel >= n_conn )
  {
    return;
  }
  ConnKeyT& conn_key = poiss_key_array[ i_conn_rel ];
  int i_source = getConnSource< ConnKeyT >( conn_key );
  int i_delay = getConnDelay< ConnKeyT >( conn_key );
  int id = ( int ) ( ( time_idx - i_delay + 1 ) % max_delay );

  if ( id < 0 )
  {
    return;
  }

  float mu = mu_arr[ id * n_node + i_source ];
  int n = curand_poisson( curand_state + i_conn_rel, mu );
  if ( n > 0 )
  {
    int64_t i_conn = i_conn_0 + i_conn_rel;
    int i_block = ( int ) ( i_conn / block_size );
    int64_t i_block_conn = i_conn % block_size;
    ConnStructT& conn_struct = ( ( ConnStructT** ) ConnStructArray )[ i_block ][ i_block_conn ];

    int i_target = getConnTarget< ConnStructT >( conn_struct );
    int i_port = getConnPort< ConnKeyT, ConnStructT >( conn_key, conn_struct );
    float weight = conn_struct.weight;
    double d_val = ( double ) ( weight * n );

    if ( input_spike_buffer_ns::algo_ == INPUT_SPIKE_BUFFER_ALGO )
    {
      int n_slots = max_input_delay_[ i_target ][ i_port ];
      int i_slot = NESTGPUTimeIdx % n_slots;
      // if (i_target==37) {
      //   printf("poiss signal i_port: %d\ti_slot: %d\n", i_port, i_slot);
      //   printf("poiss signal input_spike_buffer: %lf\n",
      //	     input_spike_buffer_[i_target][i_port][i_slot]);
      //   printf("poiss signal w: %f\tn: %d\td_val: %lf\n", weight, n , d_val);
      // }

      atomicAdd( &input_spike_buffer_[ i_target ][ i_port ][ i_slot ], d_val );
    }
    else
    {
      int i_group = NodeGroupMap[ i_target ];
      int i = i_port * NodeGroupArray[ i_group ].n_node_ + i_target - NodeGroupArray[ i_group ].i_node_0_;
      atomicAddDouble( &NodeGroupArray[ i_group ].get_spike_array_[ i ], d_val );
    }
  }
}

__global__ void initInputSpikeBuffer2DKernel( int64_t n_input_ports_tot,
  double* input_spike_buffer_1d,
  double** input_spike_buffer_2d,
  int64_t* max_input_delay_cumul );

__global__ void initInputSpikeBufferKernel( uint n_local_nodes,
  double*** input_spike_buffer,
  double** input_spike_buffer_2d,
  int64_t* n_input_ports_cumul );

// Initialize array of first outgoing connection index of each node to default value of -1 (no outgoing connections)
__global__ void initFirstOutConnectionKernel( uint n_local_nodes, int64_t* first_out_connection );

__global__ void GetInputSpikes( inode_t i_node0,
  inode_t n_nodes,
  int n_port,
  int n_var,
  float* port_weight_arr,
  int port_weight_arr_step,
  int port_weight_port_step,
  float* port_input_arr,
  int port_input_arr_step,
  int port_input_port_step );


} // namespace input_spike_buffer_ns

using namespace input_spike_buffer_ns;

template < class ConnKeyT, class ConnStructT >
int
ConnectionTemplate< ConnKeyT, ConnStructT >::initInputSpikeBuffer( uint n_local_nodes )
{
  // allocate array of number of input ports of each (local) target node
  CUDAMALLOCCTRL( "&d_n_input_ports_", &d_n_input_ports_, n_local_nodes * sizeof( int ) );
  gpuErrchk( cudaMemset( d_n_input_ports_, 0, n_local_nodes * sizeof( int ) ) );

  getNInputPortsKernel< ConnKeyT, ConnStructT > <<< ( n_conn_ + 1023 ) / 1024, 1024 >>>( n_conn_, d_n_input_ports_ );
  DBGCUDASYNC;

  CUDAMALLOCCTRL( "&d_n_input_ports_cumul_", &d_n_input_ports_cumul_, ( n_local_nodes + 1 ) * sizeof( int64_t ) );
  // Determine temporary device storage requirements
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  //<BEGIN-CLANG-TIDY-SKIP>//
  cub::DeviceScan::ExclusiveSum(
    d_temp_storage, temp_storage_bytes, d_n_input_ports_, d_n_input_ports_cumul_, n_local_nodes + 1 );
  //<END-CLANG-TIDY-SKIP>//

  // Allocate temporary storage
  CUDAMALLOCCTRL( "&d_temp_storage", &d_temp_storage, temp_storage_bytes );
  // Run exclusive prefix sum
  //<BEGIN-CLANG-TIDY-SKIP>//
  cub::DeviceScan::ExclusiveSum(
    d_temp_storage, temp_storage_bytes, d_n_input_ports_, d_n_input_ports_cumul_, n_local_nodes + 1 );
  //<END-CLANG-TIDY-SKIP>//

  // The last element is the total number of input ports over all local nodes
  gpuErrchk( cudaMemcpy(
    &n_input_ports_tot_, &d_n_input_ports_cumul_[ n_local_nodes ], sizeof( int64_t ), cudaMemcpyDeviceToHost ) );

  // allocate one-dimensional array of maximum delay among the incoming connections of each input port of each (local)
  // target node
  CUDAMALLOCCTRL( "&d_max_input_delay_1d_", &d_max_input_delay_1d_, n_input_ports_tot_ * sizeof( int ) );
  gpuErrchk( cudaMemset( d_max_input_delay_1d_, 0, n_input_ports_tot_ * sizeof( int ) ) );

  // allocate (two-dimensional) array of maximum delay among the incoming connections of each input port of each (local)
  // target node
  CUDAMALLOCCTRL( "&d_max_input_delay_", &d_max_input_delay_, n_local_nodes * sizeof( int* ) );

  initMaxInputDelayArrayKernel<<< ( n_local_nodes + 1023 ) / 1024, 1024 >>>(
    n_local_nodes, d_max_input_delay_, d_max_input_delay_1d_, d_n_input_ports_cumul_ );
  DBGCUDASYNC;

  getMaxInputDelayKernel< ConnKeyT, ConnStructT > <<< ( n_conn_ + 1023 ) / 1024, 1024 >>>(
    n_conn_, d_max_input_delay_ );
  DBGCUDASYNC;

  // Evaluates the cumulative sum of of maximum delay among the incoming connections of each input port of each (local)
  // target node
  CUDAMALLOCCTRL(
    "&d_max_input_delay_cumul_", &d_max_input_delay_cumul_, ( n_input_ports_tot_ + 1 ) * sizeof( int64_t ) );
  // Determine temporary device storage requirements
  d_temp_storage = NULL;
  temp_storage_bytes = 0;
  //<BEGIN-CLANG-TIDY-SKIP>//
  cub::DeviceScan::ExclusiveSum(
    d_temp_storage, temp_storage_bytes, d_max_input_delay_1d_, d_max_input_delay_cumul_, n_input_ports_tot_ + 1 );
  //<END-CLANG-TIDY-SKIP>//

  // Allocate temporary storage
  CUDAMALLOCCTRL( "&d_temp_storage", &d_temp_storage, temp_storage_bytes );
  // Run exclusive prefix sum
  //<BEGIN-CLANG-TIDY-SKIP>//
  cub::DeviceScan::ExclusiveSum(
    d_temp_storage, temp_storage_bytes, d_max_input_delay_1d_, d_max_input_delay_cumul_, n_input_ports_tot_ + 1 );
  //<END-CLANG-TIDY-SKIP>//

  // The last element is the total number of slots in the input spike buffers
  gpuErrchk( cudaMemcpy( &n_input_spike_buffer_tot_,
    &d_max_input_delay_cumul_[ n_input_ports_tot_ ],
    sizeof( int64_t ),
    cudaMemcpyDeviceToHost ) );

  // check that the buffer size is not zero
  if ( n_input_spike_buffer_tot_ > 0 )
  {
    // allocate memory for input spike buffer flattened on one dimension
    CUDAMALLOCCTRL(
      "&d_input_spike_buffer_1d_", &d_input_spike_buffer_1d_, n_input_spike_buffer_tot_ * sizeof( double ) );
    gpuErrchk( cudaMemset( d_input_spike_buffer_1d_, 0, n_input_spike_buffer_tot_ * sizeof( double ) ) );

    CUDAMALLOCCTRL( "&d_input_spike_buffer_2d_", &d_input_spike_buffer_2d_, n_input_ports_tot_ * sizeof( double* ) );
    initInputSpikeBuffer2DKernel<<< ( n_input_ports_tot_ + 1023 ) / 1024, 1024 >>>(
      n_input_ports_tot_, d_input_spike_buffer_1d_, d_input_spike_buffer_2d_, d_max_input_delay_cumul_ );
    DBGCUDASYNC;

    CUDAMALLOCCTRL( "&d_input_spike_buffer_", &d_input_spike_buffer_, n_local_nodes * sizeof( double** ) );

    initInputSpikeBufferKernel<<< ( n_local_nodes + 1023 ) / 1024, 1024 >>>(
      n_local_nodes, d_input_spike_buffer_, d_input_spike_buffer_2d_, d_n_input_ports_cumul_ );
    DBGCUDASYNC;

    // allocate array of indexed of first outgoing connection from each node
    CUDAMALLOCCTRL( "&d_first_out_connection_", &d_first_out_connection_, n_local_nodes * sizeof( int64_t ) );
    initFirstOutConnectionKernel<<< ( n_local_nodes + 1023 ) / 1024, 1024 >>>( n_local_nodes, d_first_out_connection_ );

    // Evaluates the index of the first outgoing connection of each (local) target node
    getFirstOutConnectionKernel< ConnKeyT > <<< ( n_conn_ + 1023 ) / 1024, 1024 >>>( n_conn_, d_first_out_connection_ );

    // allocate array of the first connection of each spike emitted at current time step
    // temporary size is n_local_nodes, in the future evaluate [n_all_nodes*time_resolution*avg_max_firing_rate]
    CUDAMALLOCCTRL( "&d_spike_first_connection_", &d_spike_first_connection_, n_local_nodes * sizeof( int64_t ) );

    // number of spikes emitted at current time step
    CUDAMALLOCCTRL( "&d_n_spikes_", &d_n_spikes_, sizeof( int ) );
    gpuErrchk( cudaMemset( d_n_spikes_, 0, sizeof( int ) ) );

    initInputSpikeBufferPointersKernel<<< 1, 1 >>>( d_n_input_ports_,
      d_max_input_delay_,
      d_input_spike_buffer_,
      d_first_out_connection_,
      d_spike_first_connection_,
      d_n_spikes_ );
    DBGCUDASYNC;

    /*
    //////////////////////////////////////////////////////////////
    ////////////////////// TEMPORARY, FOR TEST
    double ***h_input_spike_buffer = new double**[n_local_nodes];
    gpuErrchk( cudaMemcpy(h_input_spike_buffer, d_input_spike_buffer_, n_local_nodes*sizeof( double** ),
    cudaMemcpyDeviceToHost ) );

    int **h_max_input_delay = new int*[n_local_nodes];
    gpuErrchk( cudaMemcpy(h_max_input_delay, d_max_input_delay_, n_local_nodes*sizeof( int* ), cudaMemcpyDeviceToHost )
    );

    int *h_n_input_ports = new int[n_local_nodes];
    gpuErrchk( cudaMemcpy(h_n_input_ports, d_n_input_ports_, n_local_nodes*sizeof( int ), cudaMemcpyDeviceToHost ) );

    for (uint i_target=0; i_target<n_local_nodes; i_target++) {
      int n_ports = h_n_input_ports[i_target];
      printf("target: %d\tn_ports: %d\n", i_target, n_ports);
      double **h_input_spike_buffer_1 = new double*[n_ports];
      gpuErrchk( cudaMemcpy(h_input_spike_buffer_1, h_input_spike_buffer[i_target], n_ports*sizeof( double* ),
    cudaMemcpyDeviceToHost ) );

      int *h_max_input_delay_1 = new int[n_ports];
      gpuErrchk( cudaMemcpy(h_max_input_delay_1, h_max_input_delay[i_target], n_ports*sizeof( int ),
    cudaMemcpyDeviceToHost ) ); for (int i_port=0; i_port<n_ports; i_port++) { int n_delays =
    h_max_input_delay_1[i_port]; printf("\t\tport: %d\tn_delays: %d\t&input_spike_buffer[][][]: %lld\n", i_port,
    n_delays, (long long)&h_input_spike_buffer_1[i_port]);
      }
      delete[] h_input_spike_buffer_1;
      delete[] h_max_input_delay_1;
    }
    delete[] h_input_spike_buffer;
    delete[] h_max_input_delay;
    delete[] h_n_input_ports;

    int64_t *h_first_out_connection = new int64_t[n_local_nodes];
    gpuErrchk( cudaMemcpy(h_first_out_connection, d_first_out_connection_, n_local_nodes*sizeof( int64_t ),
    cudaMemcpyDeviceToHost ) ); for (uint i=0; i<n_local_nodes; i++) { printf("i_node: %d\tfirst conn: %ld\n", i,
    h_first_out_connection[i]);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    */
  }
  // CUDAFREECTRL( "d_n_input_ports_cumul", d_n_input_ports_cumul );

  return 0;
}

template < class ConnKeyT, class ConnStructT >
int
ConnectionTemplate< ConnKeyT, ConnStructT >::deliverSpikes()
{
  int n_spikes;
  gpuErrchk( cudaMemcpy( &n_spikes, d_n_spikes_, sizeof( int ), cudaMemcpyDeviceToHost ) );
  // printf("n_spikes: %d\n", n_spikes);
  if ( n_spikes > 0 )
  {
    deliverSpikesKernel< ConnKeyT, ConnStructT > <<< n_spikes, 1024 >>>( n_conn_ );
    DBGCUDASYNC;
  }

  return 0;
}

template < class ConnKeyT, class ConnStructT >
int
ConnectionTemplate< ConnKeyT, ConnStructT >::sendDirectSpikes( long long time_idx,
  int64_t i_conn0,
  int64_t n_dir_conn,
  inode_t n_node,
  int max_delay,
  float* d_mu_arr,
  void* d_poiss_key_array,
  curandState* d_curand_state )
{
  unsigned int grid_dim_x, grid_dim_y;

  if ( n_dir_conn < 65536 * 1024 )
  { // max grid dim * max block dim
    grid_dim_x = ( n_dir_conn + 1023 ) / 1024;
    grid_dim_y = 1;
  }
  else
  {
    grid_dim_x = 64; // I think it's not necessary to increase it
    if ( n_dir_conn > grid_dim_x * 1024 * 65535 )
    {
      throw ngpu_exception( std::string( "Number of direct connections " ) + std::to_string( n_dir_conn )
        + " larger than threshold " + std::to_string( grid_dim_x * 1024 * 65535 ) );
    }
    grid_dim_y = ( n_dir_conn + grid_dim_x * 1024 - 1 ) / ( grid_dim_x * 1024 );
  }
  dim3 numBlocks( grid_dim_x, grid_dim_y );
  sendDirectSpikeKernel< ConnKeyT, ConnStructT > <<< numBlocks, 1024 >>>( d_curand_state,
    time_idx,
    d_mu_arr,
    ( ConnKeyT* ) d_poiss_key_array,
    n_dir_conn,
    i_conn0,
    conn_block_size_,
    n_node,
    max_delay );

  DBGCUDASYNC

  return 0;
}

template < class ConnKeyT, class ConnStructT >
int
ConnectionTemplate< ConnKeyT, ConnStructT >::_setSpikeBufferAlgo( int spike_buffer_algo )
{
  spike_buffer_algo_ = spike_buffer_algo;
  gpuErrchk( cudaMemcpyToSymbol( algo_, &spike_buffer_algo_, sizeof( int ) ) );

  return 0;
}

template < class ConnKeyT, class ConnStructT >
int
ConnectionTemplate< ConnKeyT, ConnStructT >::getSpikeBufferAlgo()
{
  return spike_buffer_algo_;
}

#endif // INPUT_SPIKE_BUFFER_H
