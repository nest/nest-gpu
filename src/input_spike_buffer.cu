#include "connect.h"

extern __constant__ long long NESTGPUTimeIdx;

namespace input_spike_buffer_ns
{
// algorithm for spike buffering and delivery
__device__ int algo_;

// number of input ports of each (local) node
__device__ int* n_input_ports_; // [n_local_nodes]

// two-dimensional array of maximum delay among the incoming connections of each input port
// of each (local) target node
__device__ int** max_input_delay_; // [n_local_nodes][n_input_ports[i_node]]

// three-dimensional input spike buffer array
// [n_local_nodes][n_input_ports[i_node]][n_slots[i_target][i_port]]
__device__ double*** input_spike_buffer_;

// index of the first connection outgoing from each local node (-1 for no connections)
// [n_local_nodes]
__device__ int64_t* first_out_connection_;

// array of the first connection of each spike emitted at current time step
// [n_all_nodes*time_resolution*avg_max_firing_rate]
__device__ int64_t* spike_first_connection_;

// number of spikes emitted at current time step
__device__ int* n_spikes_;

// Initialize input spike buffer pointers in device memory
__global__ void
initInputSpikeBufferPointersKernel( int* n_input_ports,
  int** max_input_delay,
  double*** input_spike_buffer,
  int64_t* first_out_connection,
  int64_t* spike_first_connection,
  int* n_spikes )
{
  n_input_ports_ = n_input_ports;
  max_input_delay_ = max_input_delay;
  input_spike_buffer_ = input_spike_buffer;
  first_out_connection_ = first_out_connection;
  spike_first_connection_ = spike_first_connection;
  n_spikes_ = n_spikes;
}


__global__ void
initInputSpikeBuffer2DKernel( int64_t n_input_ports_tot,
  double* input_spike_buffer_1d,
  double** input_spike_buffer_2d,
  int64_t* max_input_delay_cumul )
{
  int64_t i_port_abs = ( int64_t ) blockIdx.x * blockDim.x + threadIdx.x;
  if ( i_port_abs >= n_input_ports_tot )
  {
    return;
  }
  input_spike_buffer_2d[ i_port_abs ] = input_spike_buffer_1d + max_input_delay_cumul[ i_port_abs ];
}

__global__ void
initInputSpikeBufferKernel( uint n_local_nodes,
  double*** input_spike_buffer,
  double** input_spike_buffer_2d,
  int64_t* n_input_ports_cumul )
{
  inode_t i_target = blockIdx.x * blockDim.x + threadIdx.x;
  if ( i_target >= n_local_nodes )
  {
    return;
  }
  input_spike_buffer[ i_target ] = input_spike_buffer_2d + n_input_ports_cumul[ i_target ];
}

__global__ void
initMaxInputDelayArrayKernel( uint n_local_nodes,
  int** max_input_delay,
  int* max_input_delay_1d,
  int64_t* n_input_ports_cumul )
{
  inode_t i_node = blockIdx.x * blockDim.x + threadIdx.x;
  if ( i_node >= n_local_nodes )
  {
    return;
  }
  max_input_delay[ i_node ] = max_input_delay_1d + n_input_ports_cumul[ i_node ];
}

// Initialize array of first outgoing connection index of each node to default value of -1 (no outgoing connections)
__global__ void
initFirstOutConnectionKernel( uint n_local_nodes, int64_t* first_out_connection )
{
  inode_t i_node = blockIdx.x * blockDim.x + threadIdx.x;
  if ( i_node >= n_local_nodes )
  {
    return;
  }
  first_out_connection[ i_node ] = -1;
}


__global__ void
GetInputSpikes( inode_t i_node0,
  inode_t n_nodes,
  int n_port,
  int n_var,
  float* port_weight_arr,
  int port_weight_arr_step,
  int port_weight_port_step,
  float* port_input_arr,
  int port_input_arr_step,
  int port_input_port_step )
{
  inode_t i_target_rel = blockIdx.x * blockDim.x + threadIdx.x;
  int i_port = blockIdx.y * blockDim.y + threadIdx.y;

  if ( i_target_rel < n_nodes && i_port < n_port )
  {
    inode_t i_target = i_node0 + i_target_rel;
    if ( i_port >= n_input_ports_[ i_target ] )
    { // number of ports actually reached by connections
      return;
    }
    int n_slots = max_input_delay_[ i_target ][ i_port ];
    int i_slot = NESTGPUTimeIdx % n_slots;

    double spike_input = input_spike_buffer_[ i_target ][ i_port ][ i_slot ];
    input_spike_buffer_[ i_target ][ i_port ][ i_slot ] = 0.0;
    int port_input = i_target_rel * port_input_arr_step + port_input_port_step * i_port;
    int port_weight = i_target_rel * port_weight_arr_step + port_weight_port_step * i_port;
    double d_val = ( double ) port_input_arr[ port_input ] + spike_input * port_weight_arr[ port_weight ];
    // if (i_target==37 && d_val > 700) {
    //   printf("port_input_arr: %lf\tspike_input: %lf\tport_weight_arr: %f\td_val: %lf\n",
    //      port_input_arr[ port_input ], spike_input, port_weight_arr[ port_weight ], d_val);
    //   printf("i_port: %d\ti_slot: %d\n", i_port, i_slot);
    // }

    port_input_arr[ port_input ] = ( float ) d_val;
  }
}


} // namespace input_spike_buffer_ns
