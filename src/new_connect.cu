#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <climits>
#include <vector>
#include <utility>
#include <cuda.h>
#include <curand.h>
#include <cub/cub.cuh>
#include "cuda_error.h"
#include "copass_kernels.h"
#include "copass_sort.h"
#include "distribution.h"
#include "new_connect.h"
#include "nestgpu.h"
#include "utilities.h"

uint h_MaxNodeNBits;
__device__ uint MaxNodeNBits;
// maximum number of bits used to represent node index 

uint h_MaxPortNBits;
__device__ uint MaxPortNBits;
// maximum number of bits used to represent receptor port index and delays 

uint h_PortMask;
__device__ uint PortMask;
// bit mask used to extract port index

uint *d_ConnGroupNum;
__device__ uint *ConnGroupNum;
// ConnGroupNum[i_spike_buffer]
// Number of connection groups outgoing from node i_spike_buffer
// where i_spike_buffer is the source node index
// Output connections from the source nodes are organized in groups
// All connection of a group have the same delay

uint *d_ConnGroupIdx0;
__device__ uint *ConnGroupIdx0;
// ig0 = ConnGroupIdx0[i_spike_buffer] is the index in the whole
// connection-group array of the first connection group outgoing
// from the node i_spike_buffer

int64_t *d_ConnGroupIConn0;
__device__ int64_t *ConnGroupIConn0;
// i_conn0 = ConnGroupIConn0[ig] with ig = 0, ..., Ng
//  is the index in the whole connection array of the first connection
// belonging to the connection group ig

int64_t *d_ConnGroupNConn;
__device__ int64_t *ConnGroupNConn;
// ConnGroupNConn[ig] with ig = 0, ..., Ng
// Ng: total number of connection groups for the whole network
// number of output connections in the connection group ig
// of the node i_spike_buffer

uint *d_ConnGroupDelay;
__device__ uint *ConnGroupDelay;
// ConnGroupDelay[ig]
// delay associated to all connections of the connection group ig
// with ig = 0, ..., Ng

int64_t NConn; // total number of connections in the whole network

int64_t h_ConnBlockSize = 20000000; //50000000;
__device__ int64_t ConnBlockSize;
// size (i.e. number of connections) of connection blocks 

uint h_MaxDelayNum;

std::vector<uint*> KeySubarray;
__device__ uint** SourceDelayArray;
// Array of source node indexes and delays of all connections
// Source node indexes and delays are merged in a single integer variable
// The most significant MaxNodeNBits are used for the node index 
// the others (less significant) bits are used to represent the delay
// This array is used as a key array for sorting the connections
// in ascending order according to the source node index
// Connections from the same source node are sorted according to
// the delay

std::vector<connection_struct*> ConnectionSubarray;
__device__ connection_struct** ConnectionArray;
// array of target node indexes, receptor port index, synapse type,
// weight of all connections
// used as a value for key-value sorting of the connections (see above)



__global__ void OrganizeConnectionGroups(uint *key_subarray,
					 uint *key_subarray_prev,
					 int64_t n_block_conn,
					 uint *conn_group_num_tmp,
					 int64_t block_conn_idx0,
					 uint *conn_group_idx0,
					 int64_t *conn_group_iconn0,
					 uint *conn_group_key)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_block_conn) return;
  uint val = key_subarray[i_conn];
  uint i_neuron = val >> MaxPortNBits;
  int64_t prev_val;
  if (i_conn==0) {
    if (key_subarray_prev != NULL) {
      prev_val = *key_subarray_prev;
    }
    else {
      prev_val = -1;      // just to ensure it is different from val
    }
  }
  else {
    prev_val = key_subarray[i_conn-1];
  }
  if (val != prev_val) {
    uint i_source_conn_group = atomicAdd(&conn_group_num_tmp[i_neuron], 1);
    uint ig0 = conn_group_idx0[i_neuron];
    uint conn_group_idx = ig0 + i_source_conn_group;
    conn_group_iconn0[conn_group_idx] = block_conn_idx0 + i_conn;
    conn_group_key[conn_group_idx] = val;
  }
}


__global__ void checkConnGroups(uint n_neuron, int64_t *source_conn_idx0,
				int64_t *source_conn_num, uint **key_subarray,
				int64_t block_size, int64_t *conn_group_iconn0,
				uint *conn_group_nconn, uint *conn_group_num,
				uint *conn_group_idx0)
{
  const uint i_neuron = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_neuron>=n_neuron) return;
  
  const int64_t nc =source_conn_num[i_neuron];
  const int64_t ic0 = source_conn_idx0[i_neuron];
  const uint ig0 = conn_group_idx0[i_neuron];
  
  int i_source_conn_group = 0;
  for (int64_t ic=ic0; ic<ic0+nc; ic++) {
    uint ib =(uint)(ic / block_size);
    int64_t jc = ic % block_size;
    uint val = key_subarray[ib][jc];
    
    uint prev_val = 0;
    if (jc==0 && ib!=0) {
      prev_val = key_subarray[ib-1][block_size-1];
    }
    else if (jc>0) {
      prev_val = key_subarray[ib][jc-1];
    }
    if (i_source_conn_group==0 || val!=prev_val) {
      uint conn_group_idx = ig0 + i_source_conn_group;
      conn_group_iconn0[conn_group_idx] = ic;
      if (ic > ic0) {
	conn_group_nconn[conn_group_idx - 1] = ic
	  - conn_group_iconn0[conn_group_idx - 1];
      }
      i_source_conn_group++;
    }
  }
  uint conn_group_idx = ig0 + i_source_conn_group;
  conn_group_nconn[conn_group_idx - 1] = ic0 + nc
    - conn_group_iconn0[conn_group_idx - 1];

}

__global__ void getSourceConnNum(uint n_neuron, int64_t *source_conn_idx0,
				 int64_t *source_conn_num, int64_t n_conn)
{
  uint i_neuron = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_neuron>=n_neuron) {
    return;
  }
  if ( i_neuron==(n_neuron-1) ) {
    source_conn_num[i_neuron] = n_conn - source_conn_idx0[i_neuron];
  }
  else {
    source_conn_num[i_neuron] = source_conn_idx0[i_neuron + 1]
      - source_conn_idx0[i_neuron];
  }
}
  

__global__ void countConnectionGroups(uint *key_subarray,
					uint *key_subarray_prev,
					int64_t n_block_conn,
					uint *conn_group_num,
					int64_t block_conn_idx0)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_block_conn) return;
  uint val = key_subarray[i_conn];
  uint i_neuron = val >> MaxPortNBits;
  int64_t prev_val;
  if (i_conn==0) {
    if (key_subarray_prev != NULL) {
      prev_val = *key_subarray_prev;
    }
    else {
      prev_val = -1;      // just to ensure it is different from val
    }
  }
  else {
    prev_val = key_subarray[i_conn-1];
  }
  if (val != prev_val) {
    atomicAdd(&conn_group_num[i_neuron], 1);
  }
}


bool print_sort_err = true;
bool print_sort_cfr = false;
bool compare_with_serial = false;
uint last_i_sub = 0;

__global__ void setWeights(connection_struct *conn_subarray, float weight,
			   int64_t n_conn)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) return;
  conn_subarray[i_conn].weight = weight;
}

__global__ void setWeights(connection_struct *conn_subarray, float *arr_val,
			   int64_t n_conn)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) return;
  conn_subarray[i_conn].weight = arr_val[i_conn];
}

__global__ void setDelays(uint *key_subarray, float *arr_val,
			  int64_t n_conn, float time_resolution)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) return;
  int delay = (int)round(arr_val[i_conn]/time_resolution);
  delay = max(delay,1);
  key_subarray[i_conn] = (key_subarray[i_conn] << MaxPortNBits) | delay;
}

__global__ void setDelays(uint *key_subarray, float fdelay,
			  int64_t n_conn, float time_resolution)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) return;
  int delay = (int)round(fdelay/time_resolution);
  delay = max(delay,1);
  key_subarray[i_conn] = (key_subarray[i_conn] << MaxPortNBits) | delay;
}

__global__ void setPort(connection_struct *conn_subarray, uint port,
			   int64_t n_conn)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) return;
  conn_subarray[i_conn].target_port =
    (conn_subarray[i_conn].target_port << MaxPortNBits) | port; 
}

__global__ void setSynGroup(connection_struct *conn_subarray,
			    unsigned char syn_group,
			    int64_t n_conn)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) return;
  conn_subarray[i_conn].syn_group = syn_group; 
}

__global__ void getConnGroupNConn(int64_t *conn_group_iconn0,
				  int64_t *conn_group_nconn,
				  uint conn_group_num, int64_t n_conn)
{
  uint conn_group_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (conn_group_idx >= conn_group_num) return;
  else if (conn_group_idx == (conn_group_num - 1)) {
    conn_group_nconn[conn_group_num - 1] = n_conn
      - conn_group_iconn0[conn_group_num - 1];
  }
  else {
    conn_group_nconn[conn_group_idx] = conn_group_iconn0[conn_group_idx + 1]
      - conn_group_iconn0[conn_group_idx];
  }
}

__global__ void getConnGroupDelay(uint *conn_group_key,
				  uint *conn_group_delay,
				  uint conn_group_num)
{
  uint conn_group_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (conn_group_idx >= conn_group_num) return;
  conn_group_delay[conn_group_idx] = conn_group_key[conn_group_idx]
    & PortMask;
}

int allocateNewBlocks(std::vector<uint*> &key_subarray,
		      std::vector<connection_struct*> &conn_subarray,
		      int64_t block_size, uint new_n_block)
{
  // Allocating GPU memory for new connection blocks
  // allocate new blocks if needed
  for (uint ib=key_subarray.size(); ib<new_n_block; ib++) {
    uint *d_key_pt;
    connection_struct *d_connection_pt;
    // allocate GPU memory for new blocks 
    gpuErrchk(cudaMalloc(&d_key_pt, block_size*sizeof(uint)));
    gpuErrchk(cudaMalloc(&d_connection_pt,
			 block_size*sizeof(connection_struct)));
    key_subarray.push_back(d_key_pt);
    conn_subarray.push_back(d_connection_pt);
  }

  return 0;
}


int setConnectionWeights(curandGenerator_t &gen, void *d_storage,
			 connection_struct *conn_subarray, int64_t n_conn,
			 SynSpec &syn_spec)
{
  if (syn_spec.weight_distr_ >= DISTR_TYPE_ARRAY   // probability distribution
      && syn_spec.weight_distr_ < N_DISTR_TYPE) {  // or array
    if (syn_spec.weight_distr_ == DISTR_TYPE_ARRAY) {
      gpuErrchk(cudaMemcpy(d_storage, syn_spec.weight_h_array_pt_,
			   n_conn*sizeof(float), cudaMemcpyHostToDevice));    
    }
    else if (syn_spec.weight_distr_ == DISTR_TYPE_NORMAL_CLIPPED) {
      CURAND_CALL(curandGenerateUniform(gen, (float*)d_storage, n_conn));
      randomNormalClipped((float*)d_storage, n_conn, syn_spec.weight_mu_,
			  syn_spec.weight_sigma_, syn_spec.weight_low_,
			  syn_spec.weight_high_);
    }
    else if (syn_spec.weight_distr_==DISTR_TYPE_NORMAL) {
      float low = syn_spec.weight_mu_ - 5.0*syn_spec.weight_sigma_;
      float high = syn_spec.weight_mu_ + 5.0*syn_spec.weight_sigma_;
      CURAND_CALL(curandGenerateUniform(gen, (float*)d_storage, n_conn));
      randomNormalClipped((float*)d_storage, n_conn, syn_spec.weight_mu_,
			  syn_spec.weight_sigma_, low, high);
    }
    else {
      throw ngpu_exception("Invalid connection weight distribution type");
    }
    setWeights<<<(n_conn+1023)/1024, 1024>>>
      (conn_subarray, (float*)d_storage, n_conn);
    DBGCUDASYNC
  }
  else {
    setWeights<<<(n_conn+1023)/1024, 1024>>>
      (conn_subarray, syn_spec.weight_, n_conn);
    DBGCUDASYNC
  }
    
  return 0;
}


int setConnectionDelays(curandGenerator_t &gen, void *d_storage,
			uint *key_subarray, int64_t n_conn,
			SynSpec &syn_spec, float time_resolution)
{
  if (syn_spec.delay_distr_ >= DISTR_TYPE_ARRAY   // probability distribution
      && syn_spec.delay_distr_ < N_DISTR_TYPE) {  // or array
    if (syn_spec.delay_distr_ == DISTR_TYPE_ARRAY) {
      gpuErrchk(cudaMemcpy(d_storage, syn_spec.delay_h_array_pt_,
			   n_conn*sizeof(float), cudaMemcpyHostToDevice));
    }
    else if (syn_spec.delay_distr_ == DISTR_TYPE_NORMAL_CLIPPED) {
      CURAND_CALL(curandGenerateUniform(gen, (float*)d_storage, n_conn));
      randomNormalClipped((float*)d_storage, n_conn, syn_spec.delay_mu_,
			  syn_spec.delay_sigma_, syn_spec.delay_low_,
			  syn_spec.delay_high_);
    }
    else if (syn_spec.delay_distr_ == DISTR_TYPE_NORMAL) {
      float low = syn_spec.delay_mu_ - 5.0*syn_spec.delay_sigma_;
      float high = syn_spec.delay_mu_ + 5.0*syn_spec.delay_sigma_;
      CURAND_CALL(curandGenerateUniform(gen, (float*)d_storage, n_conn));
      randomNormalClipped((float*)d_storage, n_conn, syn_spec.delay_mu_,
			  syn_spec.delay_sigma_, syn_spec.delay_low_,
			  syn_spec.delay_high_);
    }
    else {
      throw ngpu_exception("Invalid connection delay distribution type");
    }

    setDelays<<<(n_conn+1023)/1024, 1024>>>
      (key_subarray, (float*)d_storage, n_conn, time_resolution);
    DBGCUDASYNC

  }
  else {
    setDelays<<<(n_conn+1023)/1024, 1024>>>
      (key_subarray, syn_spec.delay_, n_conn, time_resolution);
    DBGCUDASYNC
  }
  return 0;
}


int organizeConnections(float time_resolution, uint n_node, int64_t n_conn,
			int64_t block_size,
			std::vector<uint*> &key_subarray,
			std::vector<connection_struct*> &conn_subarray)
{
  typedef uint key_t;
  timeval startTV;
  timeval endTV;
  cudaDeviceSynchronize();
  gettimeofday(&startTV, NULL);
  
  printf("Allocating auxiliary GPU memory...\n");
  int64_t storage_bytes = 0;
  void *d_storage = NULL;
  copass_sort::sort<uint, connection_struct>(key_subarray.data(),
					conn_subarray.data(), n_conn,
					block_size, d_storage, storage_bytes);
  printf("storage bytes: %ld\n", storage_bytes);
  gpuErrchk(cudaMalloc(&d_storage, storage_bytes));
  
  printf("Sorting...\n");
  copass_sort::sort<uint, connection_struct>(key_subarray.data(),
					conn_subarray.data(), n_conn,
					block_size, d_storage, storage_bytes);
  
  // free temporarily allocated storage
  gpuErrchk(cudaFree(d_storage));
  storage_bytes = 0; 
  
  printf("Indexing connection groups...\n");
  uint k = key_subarray.size();

  gpuErrchk(cudaMalloc(&d_ConnGroupNum, n_node*sizeof(uint)));
  gpuErrchk(cudaMemset(d_ConnGroupNum, 0, n_node*sizeof(uint)));
  
  uint *key_subarray_prev = NULL;
  for (uint i=0; i<k; i++) {
    uint n_block_conn = i<(k-1) ? block_size : n_conn - block_size*(k-1);
    countConnectionGroups<<<(n_block_conn+1023)/1024, 1024>>>
      (key_subarray[i], key_subarray_prev, n_block_conn, d_ConnGroupNum,
       block_size*i);
    DBGCUDASYNC
      
    key_subarray_prev = key_subarray[i] + block_size - 1;
  }
  
  gpuErrchk(cudaMalloc(&d_ConnGroupIdx0, (n_node+1)*sizeof(uint)));  
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;

  // Determine temporary device storage requirements for prefix sum
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
				d_ConnGroupNum, d_ConnGroupIdx0,
				n_node+1);
  // Allocate temporary storage for prefix sum
  gpuErrchk(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  // Run exclusive prefix sum
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
				d_ConnGroupNum, d_ConnGroupIdx0,
				n_node+1);
  gpuErrchk(cudaFree(d_temp_storage));  // free temporary allocated storage

  uint tot_conn_group_num;
  gpuErrchk(cudaMemcpy(&tot_conn_group_num, &d_ConnGroupIdx0[n_node],
		       sizeof(uint), cudaMemcpyDeviceToHost));
  printf("Total number of connection groups: %d\n", tot_conn_group_num);
  
  
  //////////////////////////////////////////////////////////////////////
  
  int64_t *d_conn_group_iconn0_unsorted;
  gpuErrchk(cudaMalloc(&d_conn_group_iconn0_unsorted,
		       tot_conn_group_num*sizeof(int64_t)));
  
  uint *d_conn_group_key_unsorted;
  gpuErrchk(cudaMalloc(&d_conn_group_key_unsorted,
		       tot_conn_group_num*sizeof(uint)));
  
  gpuErrchk(cudaMemset(d_ConnGroupNum, 0, n_node*sizeof(uint)));
  key_subarray_prev = NULL;
  for (uint i=0; i<k; i++) {
    uint n_block_conn = i<(k-1) ? block_size : n_conn - block_size*(k-1);
    OrganizeConnectionGroups<<<(n_block_conn+1023)/1024, 1024>>>
      (key_subarray[i], key_subarray_prev, n_block_conn,
       d_ConnGroupNum, block_size*i, d_ConnGroupIdx0,
       d_conn_group_iconn0_unsorted, d_conn_group_key_unsorted);
    DBGCUDASYNC
    key_subarray_prev = key_subarray[i] + block_size - 1;
  }

  gpuErrchk(cudaMalloc(&d_ConnGroupIConn0,
		       tot_conn_group_num*sizeof(int64_t)));
  uint *d_conn_group_key;
  gpuErrchk(cudaMalloc(&d_conn_group_key,
		       tot_conn_group_num*sizeof(uint)));
  void *d_conn_group_storage = NULL;
  size_t conn_group_storage_bytes = 0;

  // Determine temporary storage requirements for sorting connection groups
  cub::DeviceRadixSort::SortPairs(d_conn_group_storage,
				  conn_group_storage_bytes,
				  d_conn_group_key_unsorted,
				  d_conn_group_key,
				  d_conn_group_iconn0_unsorted,
				  d_ConnGroupIConn0,
				  tot_conn_group_num);
  // Allocate temporary storage for sorting
  gpuErrchk(cudaMalloc(&d_conn_group_storage, conn_group_storage_bytes));
  // Run radix sort
  cub::DeviceRadixSort::SortPairs(d_conn_group_storage,
				  conn_group_storage_bytes,
				  d_conn_group_key_unsorted,
				  d_conn_group_key,
				  d_conn_group_iconn0_unsorted,
				  d_ConnGroupIConn0,
				  tot_conn_group_num);
  gpuErrchk(cudaFree(d_conn_group_storage));
  gpuErrchk(cudaFree(d_conn_group_iconn0_unsorted));
  gpuErrchk(cudaFree(d_conn_group_key_unsorted));
  
  gpuErrchk(cudaMalloc(&d_ConnGroupNConn,
		       tot_conn_group_num*sizeof(int64_t)));
  
  getConnGroupNConn<<<(tot_conn_group_num+1023)/1024, 1024>>>
    (d_ConnGroupIConn0, d_ConnGroupNConn, tot_conn_group_num, n_conn);
  DBGCUDASYNC
  gpuErrchk(cudaMalloc(&d_ConnGroupDelay,
		       tot_conn_group_num*sizeof(uint)));
  
  getConnGroupDelay<<<(tot_conn_group_num+1023)/1024, 1024>>>
    (d_conn_group_key, d_ConnGroupDelay, tot_conn_group_num);
  DBGCUDASYNC
    
  gpuErrchk(cudaFree(d_conn_group_key));

  // find maxumum number of connection groups (delays) over all neurons
  uint *d_max_delay_num = NULL;  
  d_temp_storage = NULL;
  temp_storage_bytes = 0;  
  // Determine temporary device storage requirements
  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes,
			 d_ConnGroupNum, d_max_delay_num, n_node);
  // Allocate temporary storage
  gpuErrchk(cudaMalloc(&d_temp_storage, temp_storage_bytes));
  gpuErrchk(cudaMalloc(&d_max_delay_num, sizeof(uint)));
  
  // Run maximum search
  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes,
			 d_ConnGroupNum, d_max_delay_num, n_node);
	    
  CUDASYNC
  gpuErrchk(cudaFree(d_temp_storage)); // free temporary allocated storage  

  gpuErrchk(cudaMemcpy(&h_MaxDelayNum, d_max_delay_num,
		       sizeof(uint), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaFree(d_max_delay_num));

  printf("Maximum number of connection groups (delays) over all nodes: %d\n",
	 h_MaxDelayNum);

  gettimeofday(&endTV, NULL);
  long time = (long)((endTV.tv_sec * 1000000.0 + endTV.tv_usec)
		     - (startTV.tv_sec * 1000000.0 + startTV.tv_usec));
  printf("%-40s%.2f ms\n", "Time: ", (double)time / 1000.);
  printf("Done\n");
  
  
  return 0;
}



__global__ void NewConnectInitKernel(uint *conn_group_idx0,
				     uint *conn_group_num,
				     int64_t *conn_group_iconn0,
				     int64_t *conn_group_nconn,
				     uint *conn_group_delay,
				     int64_t block_size,
				     uint **source_delay_array,
				     connection_struct **connection_array)
{
  
  ConnGroupIdx0 = conn_group_idx0;
  ConnGroupNum = conn_group_num;
  ConnGroupIConn0 = conn_group_iconn0;
  ConnGroupNConn = conn_group_nconn;
  ConnGroupDelay = conn_group_delay;
  ConnBlockSize = block_size;
  SourceDelayArray = source_delay_array;
  ConnectionArray = connection_array;
}

int NewConnectInit()
{
  uint k = ConnectionSubarray.size();
  uint **d_source_delay_array;
  gpuErrchk(cudaMalloc(&d_source_delay_array, k*sizeof(uint*)));
  gpuErrchk(cudaMemcpy(d_source_delay_array, KeySubarray.data(),
		       k*sizeof(uint*), cudaMemcpyHostToDevice));
  
  connection_struct **d_connection_array;
  gpuErrchk(cudaMalloc(&d_connection_array, k*sizeof(connection_struct*)));
  gpuErrchk(cudaMemcpy(d_connection_array, ConnectionSubarray.data(),
		       k*sizeof(connection_struct*), cudaMemcpyHostToDevice));

  NewConnectInitKernel<<<1,1>>>(d_ConnGroupIdx0, d_ConnGroupNum,
				d_ConnGroupIConn0, d_ConnGroupNConn,
				d_ConnGroupDelay, h_ConnBlockSize,
				d_source_delay_array,
				d_connection_array);
  DBGCUDASYNC

  return 0;
}

__global__ void setMaxNodeNBitsKernel(int max_node_nbits, int max_port_nbits,
				      int port_mask)
{
  MaxNodeNBits = max_node_nbits;
  MaxPortNBits = max_port_nbits;
  PortMask = port_mask;
}

int setMaxNodeNBits(int max_node_nbits)
{
  h_MaxNodeNBits = max_node_nbits;
  h_MaxPortNBits = 32 - h_MaxNodeNBits;
  h_PortMask = (1 << h_MaxPortNBits) - 1;
  setMaxNodeNBitsKernel<<<1,1>>>(h_MaxNodeNBits, h_MaxPortNBits, h_PortMask);
  DBGCUDASYNC

  return 0;
}  

int *sortArray(int *h_arr, int n_elem)
{
  // allocate unsorted and sorted array in device memory
  int *d_arr_unsorted;
  int *d_arr_sorted;
  gpuErrchk(cudaMalloc(&d_arr_unsorted, n_elem*sizeof(int)));
  gpuErrchk(cudaMalloc(&d_arr_sorted, n_elem*sizeof(int)));
  gpuErrchk(cudaMemcpy(d_arr_unsorted, h_arr, n_elem*sizeof(int),
		       cudaMemcpyHostToDevice));
  void *d_storage = NULL;
  size_t storage_bytes = 0;
  // Determine temporary storage requirements for sorting source indexes
  cub::DeviceRadixSort::SortKeys(d_storage, storage_bytes, d_arr_unsorted,
				 d_arr_sorted, n_elem);
  // Allocate temporary storage for sorting
  gpuErrchk(cudaMalloc(&d_storage, storage_bytes));
  // Run radix sort
  cub::DeviceRadixSort::SortKeys(d_storage, storage_bytes, d_arr_unsorted,
				 d_arr_sorted, n_elem);
  gpuErrchk(cudaFree(d_storage));
  gpuErrchk(cudaFree(d_arr_unsorted));

  return d_arr_sorted;
}

__global__ void setSourceTargetIndexKernel(int64_t n_src_tgt, int  n_source,
					   int n_target, int64_t *d_src_tgt_arr,
					   int *d_src_arr, int *d_tgt_arr)
{
  int64_t i_src_tgt = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; 
  if (i_src_tgt >= n_src_tgt) return;
  int i_src =(int)(i_src_tgt / n_target);
  int i_tgt =(int)(i_src_tgt % n_target);
  int src_id = d_src_arr[i_src];
  int tgt_id = d_tgt_arr[i_tgt];
  int64_t src_tgt_id = ((int64_t)src_id << 32) + tgt_id;
  d_src_tgt_arr[i_src_tgt] = src_tgt_id;
}


// Count number of connections per source-target couple
__global__ void CountConnectionsKernel(int64_t n_conn, int n_source,
				       int n_target, int64_t *src_tgt_arr,
				       int64_t *src_tgt_conn_num,
				       int syn_group)
{
  int64_t i_conn = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; 
  if (i_conn >= n_conn) return;

  uint i_block = (uint)(i_conn / ConnBlockSize);
  int64_t i_block_conn = i_conn % ConnBlockSize;
  connection_struct conn = ConnectionArray[i_block][i_block_conn];
  if (syn_group==-1 || conn.syn_group == syn_group) {
    // First get target node index
    uint target_port = conn.target_port;
    int i_target = target_port >> MaxPortNBits;
    uint source_delay = SourceDelayArray[i_block][i_block_conn];
    int i_source = source_delay >> MaxPortNBits;
    int64_t i_src_tgt = ((int64_t)i_source << 32) + i_target;
    int64_t i_arr = locate(i_src_tgt, src_tgt_arr, n_source*n_target);
    if (src_tgt_arr[i_arr] == i_src_tgt) {
      // (atomic)increase the number of connections for source-target couple
      atomicAdd((unsigned long long *)&src_tgt_conn_num[i_arr], 1);
    }
  }
}


// Fill array of connection indexes
__global__ void SetConnectionsIndexKernel(int64_t n_conn, int n_source,
					  int n_target, int64_t *src_tgt_arr,
					  int64_t *src_tgt_conn_num,
					  int64_t *src_tgt_conn_cumul,
					  int syn_group, int64_t *conn_ids)
{
  int64_t i_conn = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; 
  if (i_conn >= n_conn) return;

  uint i_block = (uint)(i_conn / ConnBlockSize);
  int64_t i_block_conn = i_conn % ConnBlockSize;
  connection_struct conn = ConnectionArray[i_block][i_block_conn];
  if (syn_group==-1 || conn.syn_group == syn_group) {
    // First get target node index
    uint target_port = conn.target_port;
    int i_target = target_port >> MaxPortNBits;
    uint source_delay = SourceDelayArray[i_block][i_block_conn];
    int i_source = source_delay >> MaxPortNBits;
    int64_t i_src_tgt = ((int64_t)i_source << 32) & i_target;
    int64_t i_arr = locate(i_src_tgt, src_tgt_arr, n_source*n_target);
    if (src_tgt_arr[i_arr] == i_src_tgt) {
      // (atomic)increase the number of connections for source-target couple
      int64_t pos =
	atomicAdd((unsigned long long *)&src_tgt_conn_num[i_arr], 1);
      conn_ids[src_tgt_conn_cumul[i_arr] + pos] = i_conn;
    }
  }
}


int64_t *NESTGPU::GetConnections(int *i_source_pt, int n_source,
				 int *i_target_pt, int n_target,
				 int syn_group, int64_t *n_conn)
{  
  int64_t *h_conn_ids = NULL;
  int64_t *d_conn_ids = NULL;
  int64_t n_src_tgt = (int64_t)n_source * n_target;
  int64_t n_conn_ids = 0;

  if (n_src_tgt > 0) {
    // sort source node index array in GPU memory
    int *d_src_arr = sortArray(i_source_pt, n_source);
    // sort target node index array in GPU memory
    int *d_tgt_arr = sortArray(i_target_pt, n_target);
    // Allocate array of combined source-target indexes (src_arr x tgt_arr)
    int64_t *d_src_tgt_arr;
    gpuErrchk(cudaMalloc(&d_src_tgt_arr, n_src_tgt*sizeof(int64_t)));
    // Fill it with combined source-target indexes
    setSourceTargetIndexKernel<<<(n_src_tgt+1023)/1024, 1024>>>
      (n_src_tgt, n_source, n_target, d_src_tgt_arr, d_src_arr, d_tgt_arr);
    // Allocate array of number of connections per source-target couple
    // and initialize it to 0
    int64_t *d_src_tgt_conn_num;
    gpuErrchk(cudaMalloc(&d_src_tgt_conn_num, (n_src_tgt + 1)*sizeof(int64_t)));
    gpuErrchk(cudaMemset(d_src_tgt_conn_num, 0,
			 (n_src_tgt + 1)*sizeof(int64_t)));

    // Count number of connections per source-target couple
    CountConnectionsKernel<<<(NConn+1023)/1024, 1024>>>
      (NConn, n_source, n_target, d_src_tgt_arr, d_src_tgt_conn_num, syn_group);
    // Evaluate exclusive sum of connections per source-target couple
    // Allocate array for cumulative sum
    int64_t *d_src_tgt_conn_cumul;
    gpuErrchk(cudaMalloc(&d_src_tgt_conn_cumul,
			 (n_src_tgt + 1)*sizeof(int64_t)));
    // Determine temporary device storage requirements
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
				  d_src_tgt_conn_num,
				  d_src_tgt_conn_cumul,
				  n_src_tgt + 1);
    // Allocate temporary storage
    gpuErrchk(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
				  d_src_tgt_conn_num,
				  d_src_tgt_conn_cumul,
				  n_src_tgt + 1);
    gpuErrchk(cudaFree(d_temp_storage));
    
    // The last element is the total number of required connection Ids
    cudaMemcpy(&n_conn_ids, &d_src_tgt_conn_cumul[n_src_tgt],
	       sizeof(int64_t), cudaMemcpyDeviceToHost);
    
    if (n_conn_ids > 0) {
      // Allocate array of connection indexes
      gpuErrchk(cudaMalloc(&d_conn_ids, n_conn_ids*sizeof(int64_t)));  
      // Set number of connections per source-target couple to 0 again
      gpuErrchk(cudaMemset(d_src_tgt_conn_num, 0,
			   (n_src_tgt + 1)*sizeof(int64_t)));
      // Fill array of connection indexes
      SetConnectionsIndexKernel<<<(NConn+1023)/1024, 1024>>>
	(NConn, n_source, n_target, d_src_tgt_arr, d_src_tgt_conn_num,
	 d_src_tgt_conn_cumul, syn_group, d_conn_ids);

      /// check if allocating with new is more appropriate
      h_conn_ids = (int64_t*)malloc(n_conn_ids*sizeof(int64_t));
      gpuErrchk(cudaMemcpy(h_conn_ids, d_conn_ids,
			   n_conn_ids*sizeof(int64_t),
			   cudaMemcpyDeviceToHost));
	
      gpuErrchk(cudaFree(d_src_tgt_arr));
      gpuErrchk(cudaFree(d_src_tgt_conn_num));
      gpuErrchk(cudaFree(d_src_tgt_conn_cumul));
      gpuErrchk(cudaFree(d_conn_ids));
    }
  }
  *n_conn = n_conn_ids;
  
  return h_conn_ids;
}
