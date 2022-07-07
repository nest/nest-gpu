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

uint h_MaxNodeNBits;
__device__ uint MaxNodeNBits;

uint h_MaxPortNBits;
__device__ uint MaxPortNBits;

uint h_PortMask;
__device__ uint PortMask;

uint *d_ConnGroupIdx0;
__device__ uint *ConnGroupIdx0;

uint *d_ConnGroupNum;
__device__ uint *ConnGroupNum;

int64_t *d_ConnGroupIConn0;
__device__ int64_t *ConnGroupIConn0;

int64_t *d_ConnGroupNConn;
__device__ int64_t *ConnGroupNConn;

uint *d_ConnGroupDelay;
__device__ uint *ConnGroupDelay;

int64_t NConn;

int64_t h_ConnBlockSize = 50000000;
__device__ int64_t ConnBlockSize;

std::vector<uint*> KeySubarray;
std::vector<connection_struct*> ConnectionSubarray;

__device__ connection_struct** ConnectionArray;

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
    //printf("i_neuron: %d\tib: %d\tjc:%ld\n", i_neuron, ib, jc);
    uint val = key_subarray[ib][jc];
    //printf("i_neuron: %d\tib: %d\tjc:%ld\tval: %d\n", i_neuron, ib, jc, val);
    
    uint prev_val = 0;
    if (jc==0 && ib!=0) {
      prev_val = key_subarray[ib-1][block_size-1];
    }
    else if (jc>0) {
      prev_val = key_subarray[ib][jc-1];
    }
    //printf("i_neuron: %d\tib: %d\tjc:%ld\tprev_val: %d\n", i_neuron, ib, jc,
    //	   prev_val);
    if (i_source_conn_group==0 || val!=prev_val) {
      uint conn_group_idx = ig0 + i_source_conn_group;
      //printf("i_neuron: %d ok0\tig0: %d\ti_source_conn_group: %d\t"
      //     "conn_group_idx: %d\tic: %ld\n",
      //     i_neuron, ig0, i_source_conn_group, conn_group_idx, ic);
      conn_group_iconn0[conn_group_idx] = ic;
      //printf("i_neuron: %d ok1\n", i_neuron);
      if (ic > ic0) {
	//printf("i_neuron: %d ok2\n", i_neuron);
	conn_group_nconn[conn_group_idx - 1] = ic
	  - conn_group_iconn0[conn_group_idx - 1];
	//printf("i_neuron: %d ok3\n", i_neuron);
	//conn_group_delay[conn_group_idx] = val % 1024;
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
					int64_t block_conn_idx0,
					int64_t *source_conn_idx0)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_block_conn) return;
  uint val = key_subarray[i_conn];
  uint i_neuron = val >> MaxPortNBits;
  int64_t prev_val;
  int64_t prev_neuron;
  if (i_conn==0) {
    if (key_subarray_prev != NULL) {
      prev_val = *key_subarray_prev;
      prev_neuron = prev_val >> MaxPortNBits;
    }
    else {
      prev_val = -1;      // just to ensure it is different from val
      prev_neuron = -1;   // just to ensure it is different from i_neuron
    }
  }
  else {
    prev_val = key_subarray[i_conn-1];
    prev_neuron = prev_val >> MaxPortNBits;
  }
  if (val != prev_val) {
    atomicAdd(&conn_group_num[i_neuron], 1);
  }
  if (prev_neuron != i_neuron) {
    source_conn_idx0[i_neuron] = block_conn_idx0 + i_conn;
  }
}


bool print_sort_err = true;
bool print_sort_cfr = false;
bool compare_with_serial = false;
uint last_i_sub = 0;


__global__ void setSource(uint *key_subarray, uint *rand_val,
			  int64_t n_conn, uint idx_min, uint n_idx)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) return;
  key_subarray[i_conn] = idx_min + rand_val[i_conn]%n_idx;
}

__global__ void setTarget(connection_struct *conn_subarray, uint *rand_val,
			  int64_t n_conn, uint idx_min, uint n_idx)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) return;
  conn_subarray[i_conn].target_port = idx_min + rand_val[i_conn]%n_idx;
}

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
  int delay = (int)(arr_val[i_conn]/time_resolution);
  delay = max(delay,1);
  key_subarray[i_conn] = (key_subarray[i_conn] << MaxPortNBits) | delay;
}

__global__ void setDelays(uint *key_subarray, float fdelay,
			  int64_t n_conn, float time_resolution)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) return;
  int delay = (int)(fdelay/time_resolution);
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

__global__ void setAllToAllSourceTarget(uint *key_subarray,
					connection_struct *conn_subarray,
					int64_t n_block_conn,
					int64_t n_prev_conn,
					uint i_source0, uint n_source,
					uint i_target0, uint n_target)
{
  int64_t i_block_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_block_conn>=n_block_conn) return;
  int64_t i_conn = n_prev_conn + i_block_conn;
  uint i_source = i_source0 + i_conn / n_target;
  uint i_target = i_target0 + i_conn % n_target;
  key_subarray[i_block_conn] = i_source;
  conn_subarray[i_block_conn].target_port = i_target;
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
  //printf("Allocating GPU memory for new connection blocks...\n");
  // allocate new blocks if needed
  for (uint ib=key_subarray.size(); ib<new_n_block; ib++) {
    //printf("new_n_block; %d\tib %d\tblock_size %ld\n", new_n_block, ib,
    //	   block_size);
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
  if (syn_spec.weight_distr_ > DISTR_TYPE_ARRAY // probability distribution
      && syn_spec.weight_distr_ < N_DISTR_TYPE) { 
    //n_conn = 10000; // temporary, remove!!!!!!!!!!!!!!!!
    CURAND_CALL(curandGenerateUniform(gen, (float*)d_storage, n_conn));
    if (syn_spec.weight_distr_ == DISTR_TYPE_NORMAL_CLIPPED) {
      randomNormalClipped((float*)d_storage, n_conn, syn_spec.weight_mu_,
			  syn_spec.weight_sigma_, syn_spec.weight_low_,
			  syn_spec.weight_high_);
    }
    else if (syn_spec.weight_distr_==DISTR_TYPE_NORMAL) {
      float low = syn_spec.weight_mu_ - 5.0*syn_spec.weight_sigma_;
      float high = syn_spec.weight_mu_ + 5.0*syn_spec.weight_sigma_;
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
  if (syn_spec.delay_distr_ > DISTR_TYPE_ARRAY // probability distribution
      && syn_spec.delay_distr_ < N_DISTR_TYPE) { 
    CURAND_CALL(curandGenerateUniform(gen, (float*)d_storage, n_conn));
    if (syn_spec.delay_distr_ == DISTR_TYPE_NORMAL_CLIPPED) {
      randomNormalClipped((float*)d_storage, n_conn, syn_spec.delay_mu_,
			  syn_spec.delay_sigma_, syn_spec.delay_low_,
			  syn_spec.delay_high_);
    }
    else if (syn_spec.delay_distr_ == DISTR_TYPE_NORMAL) {
      float low = syn_spec.delay_mu_ - 5.0*syn_spec.delay_sigma_;
      float high = syn_spec.delay_mu_ + 5.0*syn_spec.delay_sigma_;
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

int connect_fixed_total_number(curandGenerator_t &gen,
			       void *d_storage, float time_resolution,
			       std::vector<uint*> &key_subarray,
			       std::vector<connection_struct*> &conn_subarray,
			       int64_t &n_conn, int64_t block_size,
			       int64_t total_num, int i_source0, int n_source,
			       int i_target0, int n_target,
			       SynSpec &syn_spec)
{
  if (total_num==0) return 0;
  uint64_t old_n_conn = n_conn;
  uint64_t n_new_conn = total_num;
  n_conn += n_new_conn; // new number of connections
  uint new_n_block = (uint)((n_conn + block_size - 1) / block_size);

  allocateNewBlocks(key_subarray, conn_subarray, block_size, new_n_block);
      
  //printf("Generating connections with fixed_total_number rule...\n");
  uint ib0 = (uint)(old_n_conn / block_size);
  for (uint ib=ib0; ib<new_n_block; ib++) {
    uint64_t n_block_conn; // number of connections in a block
    uint64_t i_conn0; // index of first connection in a block
    if (new_n_block == ib0 + 1) {  // all connections are in the same block
        i_conn0 = old_n_conn % block_size;
	n_block_conn =   n_new_conn;
    }
    else if (ib == ib0) { // first block
      i_conn0 = old_n_conn % block_size;
      n_block_conn = block_size - i_conn0;
    }
    else if (ib == new_n_block-1) { // last block
      i_conn0 = 0;
      n_block_conn = (n_conn - 1) % block_size + 1;
    }
    else {
      i_conn0 = 0;
      n_block_conn = block_size;
    }
    // generate random source index in range 0 - n_neuron
    CURAND_CALL(curandGenerate(gen, (uint*)d_storage, n_block_conn));
    //printf("old_n_conn: %ld\n", old_n_conn);
    //printf("n_new_conn: %ld\n", n_new_conn);
    //printf("new_n_block: %d\n", new_n_block);
    //printf("ib: %d\n", ib);
    //printf("n_block_conn: %ld\n", n_block_conn);
    setSource<<<(n_block_conn+1023)/1024, 1024>>>
      (key_subarray[ib] + i_conn0, (uint*)d_storage, n_block_conn,
       i_source0, n_source);
    DBGCUDASYNC

    // generate random target index in range 0 - n_neuron
    CURAND_CALL(curandGenerate(gen, (uint*)d_storage, n_block_conn));
    setTarget<<<(n_block_conn+1023)/1024, 1024>>>
      (conn_subarray[ib] + i_conn0, (uint*)d_storage, n_block_conn,
       i_target0, n_target);
    DBGCUDASYNC

    setConnectionWeights(gen, d_storage, conn_subarray[ib] + i_conn0,
			 n_block_conn, syn_spec);

    setConnectionDelays(gen, d_storage, key_subarray[ib] + i_conn0,
			n_block_conn, syn_spec, time_resolution);

    setPort<<<(n_block_conn+1023)/1024, 1024>>>
      (conn_subarray[ib] + i_conn0, syn_spec.port_, n_block_conn);
    DBGCUDASYNC
  }
  
  return 0;
}



int connect_all_to_all(curandGenerator_t &gen,
		       void *d_storage, float time_resolution,
		       std::vector<uint*> &key_subarray,
		       std::vector<connection_struct*> &conn_subarray,
		       int64_t &n_conn, int64_t block_size,
		       int i_source0, int n_source,
		       int i_target0, int n_target,
		       SynSpec &syn_spec)
{
  uint64_t old_n_conn = n_conn;
  uint64_t n_new_conn = n_source*n_target;
  n_conn += n_new_conn; // new number of connections
  uint new_n_block = (uint)((n_conn + block_size - 1) / block_size);

  allocateNewBlocks(key_subarray, conn_subarray, block_size, new_n_block);
    
  //printf("Generating connections with all-to-all rule...\n");
  int64_t n_prev_conn = 0;
  uint ib0 = (uint)(old_n_conn / block_size);
  for (uint ib=ib0; ib<new_n_block; ib++) {
    uint64_t n_block_conn; // number of connections in a block
    uint64_t i_conn0; // index of first connection in a block
    if (new_n_block == ib0 + 1) {  // all connections are in the same block
      i_conn0 = old_n_conn % block_size;
      n_block_conn = n_new_conn;
    }
    else if (ib == ib0) { // first block
      i_conn0 = old_n_conn % block_size;
      n_block_conn = block_size - i_conn0;
    }
    else if (ib == new_n_block-1) { // last block
      i_conn0 = 0;
      n_block_conn = (n_conn - 1) % block_size + 1;
    }
    else {
      i_conn0 = 0;
      n_block_conn = block_size;
    }
    
    setAllToAllSourceTarget<<<(n_block_conn+1023)/1024, 1024>>>
      (key_subarray[ib] + i_conn0, conn_subarray[ib] + i_conn0,
       n_block_conn, n_prev_conn, i_source0, n_source, i_target0, n_target);
    DBGCUDASYNC
    setConnectionWeights(gen, d_storage, conn_subarray[ib] + i_conn0,
			 n_block_conn, syn_spec);

    setConnectionDelays(gen, d_storage, key_subarray[ib] + i_conn0,
			n_block_conn, syn_spec, time_resolution);
    
    setPort<<<(n_block_conn+1023)/1024, 1024>>>
      (conn_subarray[ib] + i_conn0, syn_spec.port_, n_block_conn);
    DBGCUDASYNC  
    
    n_prev_conn += n_block_conn;
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
  
  //printf("ok0 block_size %ld\tn_node %d\tn_conn %ld\n", block_size,
  //	 n_node, n_conn);
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
  printf("Indexing connection groups...\n");
  uint k = key_subarray.size();
  storage_bytes = 0; // free temporarily allocated storage
  
  gpuErrchk(cudaMalloc(&d_ConnGroupNum, n_node*sizeof(uint)));
  gpuErrchk(cudaMalloc(&d_ConnGroupIdx0, (n_node+1)*sizeof(uint)));
  gpuErrchk(cudaMemset(d_ConnGroupNum, 0, n_node*sizeof(uint)));
	    
  int64_t *d_source_conn_idx0;
  int64_t *d_source_conn_num;
  cudaReusableAlloc(d_storage, storage_bytes, &d_source_conn_idx0,
		    n_node, sizeof(int64_t));
  cudaReusableAlloc(d_storage, storage_bytes, &d_source_conn_num,
		    n_node, sizeof(int64_t));

  uint *key_subarray_prev = NULL;
  for (uint i=0; i<k; i++) {
    uint n_block_conn = i<(k-1) ? block_size : n_conn - block_size*(k-1);
    countConnectionGroups<<<(n_block_conn+1023)/1024, 1024>>>
      (key_subarray[i], key_subarray_prev, n_block_conn, d_ConnGroupNum,
       block_size*i, d_source_conn_idx0);
    DBGCUDASYNC
    key_subarray_prev = key_subarray[i] + block_size - 1;
  }
  
  
  void *d_cumul_storage = NULL;
  size_t cumul_storage_bytes = 0;
  
  // Determine temporary device storage requirements for prefix sum
  cub::DeviceScan::ExclusiveSum(d_cumul_storage, cumul_storage_bytes,
				d_ConnGroupNum, d_ConnGroupIdx0,
				n_node+1);
  size_t storage_bytes_bk = storage_bytes; // backup storage bytes
  // Allocate temporary storage for prefix sum
  cudaReusableAlloc(d_storage, storage_bytes, &d_cumul_storage,
		    cumul_storage_bytes, sizeof(char));
  // Run exclusive prefix sum
  cub::DeviceScan::ExclusiveSum(d_cumul_storage, cumul_storage_bytes,
				d_ConnGroupNum, d_ConnGroupIdx0,
				n_node+1);
  storage_bytes = storage_bytes_bk; // free temporary allocated storage
  
  uint tot_conn_group_num;
  gpuErrchk(cudaMemcpy(&tot_conn_group_num, &d_ConnGroupIdx0[n_node],
		       sizeof(uint), cudaMemcpyDeviceToHost));
  printf("Total number of connection groups: %d\n", tot_conn_group_num);
  
  
  //////////////////////////////////////////////////////////////////////
  
  int64_t *d_conn_group_iconn0_unsorted;
  cudaReusableAlloc(d_storage, storage_bytes, &d_conn_group_iconn0_unsorted,
		    tot_conn_group_num, sizeof(int64_t));
  
  gpuErrchk(cudaMalloc(&d_ConnGroupIConn0,
		       tot_conn_group_num*sizeof(int64_t)));
  uint *d_conn_group_key_unsorted;
  cudaReusableAlloc(d_storage, storage_bytes, &d_conn_group_key_unsorted,
		    tot_conn_group_num, sizeof(uint));
  uint *d_conn_group_key;
  cudaReusableAlloc(d_storage, storage_bytes, &d_conn_group_key,
		    tot_conn_group_num, sizeof(uint));
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
  cudaReusableAlloc(d_storage, storage_bytes, &d_conn_group_storage,
		    conn_group_storage_bytes, sizeof(char));
  // Run radix sort
  cub::DeviceRadixSort::SortPairs(d_conn_group_storage,
				  conn_group_storage_bytes,
				  d_conn_group_key_unsorted,
				  d_conn_group_key,
				  d_conn_group_iconn0_unsorted,
				  d_ConnGroupIConn0,
				  tot_conn_group_num);
  
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
  cudaDeviceSynchronize();
  gettimeofday(&endTV, NULL);
  long time = (long)((endTV.tv_sec * 1000000.0 + endTV.tv_usec)
		     - (startTV.tv_sec * 1000000.0 + startTV.tv_usec));
  printf("%-40s%.2f ms\n", "Time: ", (double)time / 1000.);
  printf("Done\n");
  
  
  return 0;
}

template <>
int NESTGPU::_ConnectAllToAll<int, int>
(int source, int n_source, int target, int n_target, SynSpec &syn_spec)
{
  //printf("In new specialized connection all-to-all\n");
  //float weight_mean = syn_spec.weight_;
  //float weight_std = syn_spec.weight_ / 10.0;
  //float delay_mean = syn_spec.delay_;
  //float delay_std = syn_spec.delay_ / 4.0;
  //int port = syn_spec.port_;
  
  void *d_storage;
  gpuErrchk(cudaMalloc(&d_storage, h_ConnBlockSize*sizeof(int)));
  
  connect_all_to_all(*random_generator_, d_storage, time_resolution_,
		     KeySubarray, ConnectionSubarray, NConn,
		     h_ConnBlockSize, source, n_source,
		     target, n_target, syn_spec);
  gpuErrchk(cudaFree(d_storage));
  
  return 0;
}

template <>
int NESTGPU::_ConnectFixedTotalNumber<int, int>
(int source, int n_source, int target, int n_target, int total_num,
 SynSpec &syn_spec)
{
  //printf("In new specialized connection fixed-total-number\n");
  //float weight_mean = syn_spec.weight_;
  //float weight_std = syn_spec.weight_ / 10.0;
  //float delay_mean = syn_spec.delay_;
  //float delay_std = syn_spec.delay_ / 4.0;
  //int port = syn_spec.port_;
  
  void *d_storage;
  gpuErrchk(cudaMalloc(&d_storage, h_ConnBlockSize*sizeof(int)));
  
  connect_fixed_total_number(*random_generator_, d_storage, time_resolution_,
			     KeySubarray, ConnectionSubarray, NConn,
			     h_ConnBlockSize, total_num, source, n_source,
			     target, n_target, syn_spec);
  gpuErrchk(cudaFree(d_storage));
  
  return 0;
}

__global__ void NewConnectInitKernel(uint *conn_group_idx0,
				     uint *conn_group_num,
				     int64_t *conn_group_iconn0,
				     int64_t *conn_group_nconn,
				     uint *conn_group_delay,
				     int64_t block_size,
				     connection_struct **connection_array)
{
  
  ConnGroupIdx0 = conn_group_idx0;
  ConnGroupNum = conn_group_num;
  ConnGroupIConn0 = conn_group_iconn0;
  ConnGroupNConn = conn_group_nconn;
  ConnGroupDelay = conn_group_delay;
  ConnBlockSize = block_size;
  ConnectionArray = connection_array;
}

int NewConnectInit()
{
  uint k = ConnectionSubarray.size();
  connection_struct **d_connection_array;
  gpuErrchk(cudaMalloc(&d_connection_array, k*sizeof(connection_struct*)));
  
  gpuErrchk(cudaMemcpy(d_connection_array, ConnectionSubarray.data(),
		       k*sizeof(connection_struct*), cudaMemcpyHostToDevice));

  NewConnectInitKernel<<<1,1>>>(d_ConnGroupIdx0, d_ConnGroupNum,
				d_ConnGroupIConn0, d_ConnGroupNConn,
				d_ConnGroupDelay, h_ConnBlockSize,
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
