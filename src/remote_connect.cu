#include <iostream>
#include <vector>

#include "connect.h"
#include "remote_connect.h"

// INITIALIZATION
//
// Define two arrays that map remote source nodes to local spike buffers
// There is one element for each remote host (MPI process),
// so the array size is mpi_proc_num
// Each of the two arrays contain n_remote_source_node_map elements
// that represent a map, with n_remote_source_node_map pairs
// (remote node index, local spike buffer index)
// where n_remote_source_node_map is the number of nodes in the source host
// (MPI pocess) that have outgoing connections to local nodes.
// All elements are initially empty:
// n_remote_source_nodes[i_source_host] = 0 for each i_source_host
// The map is organized in blocks each with remote_node_map_block_size
// elements, which are allocated dynamically

__constant__ uint remote_node_map_block_size; // = 100000;

// number of elements in the map for each source host
// n_remote_source_node_map[i_source_host]
// with i_source_host = 0, ..., mpi_proc_num-1 excluding this host itself
__device__ uint *n_remote_source_node_map; // [mpi_proc_num];
uint *d_n_remote_source_node_map;

// remote_source_node_map_index[i_source_host][i_block][i]
std::vector< std::vector<uint*> > h_remote_source_node_map_index;
__device__ uint ***remote_source_node_map_index;

// local_spike_buffer_map_index[i_source_host][i_block][i]
std::vector< std::vector<uint*> > h_local_spike_buffer_map_index;
__device__ uint ***local_spike_buffer_map_index;

// Define two arrays that map local source nodes to remote spike buffers.
// The structure is the same as for remote source nodes

// number of elements in the map for each target host
// n_local_source_node_map[i_target_host]
// with i_target_host = 0, ..., mpi_proc_num-1 excluding this host itself
__device__ uint *n_local_source_node_map; // [mpi_proc_num]; 
uint *d_n_local_source_node_map;
  
// local_source_node_map_index[i_target_host][i_block][i]
std::vector< std::vector<uint*> > h_local_source_node_map_index;
__device__ uint ***local_source_node_map_index;
uint ***d_local_source_node_map_index;

// Define a boolean array with one boolean value for each connection rule
// - true if the rule always creates at least one outgoing connection
// from each source node (one_to_one, all_to_all, fixed_outdegree)
// - false otherwise (fixed_indegree, fixed_total_number, pairwise_bernoulli)
bool *use_all_source_nodes; // [n_connection_rules]:

// Allocate GPU memory for new remote-source-node-map blocks
int allocRemoteSourceNodeMapBlocks(std::vector<uint*> &i_remote_src_node_map,
				   std::vector<uint*> &i_local_spike_buf_map,
				   int64_t block_size, uint new_n_block)
{
  // allocate new blocks if needed
  for (uint ib=i_remote_src_node_map.size(); ib<new_n_block; ib++) {
    uint *d_remote_src_node_blk_pt;
    uint *d_local_spike_buf_blk_pt;
    // allocate GPU memory for new blocks 
    gpuErrchk(cudaMalloc(&d_remote_src_node_blk_pt, block_size*sizeof(uint)));
    gpuErrchk(cudaMalloc(&d_local_spike_buf_blk_pt, block_size*sizeof(uint)));
      
    i_remote_src_node_map.push_back(d_remote_src_node_blk_pt);
    i_local_spike_buf_map.push_back(d_local_spike_buf_blk_pt);
  }
    
  return 0;
}

// Allocate GPU memory for new local-source-node-map blocks
int allocLocalSourceNodeMapBlocks(std::vector<uint*> &i_local_src_node_map,
				  int64_t block_size, uint new_n_block)
{
  // allocate new blocks if needed
  for (uint ib=i_local_src_node_map.size(); ib<new_n_block; ib++) {
    uint *d_local_src_node_blk_pt;
    // allocate GPU memory for new blocks 
    gpuErrchk(cudaMalloc(&d_local_src_node_blk_pt, block_size*sizeof(uint)));
      
    i_local_src_node_map.push_back(d_local_src_node_blk_pt);
  }
    
  return 0;
}


// Initialize the maps for n_hosts hosts (i.e. number of MPI processes)
int RemoteConnectionMapInit(uint n_hosts)
{
  int bs = 10000; // initialize node map block size
  cudaMemcpyToSymbol(remote_node_map_block_size, &bs, sizeof(int));

  // allocate and init to 0 n. of elements in the map for each source host
  gpuErrchk(cudaMalloc(&d_n_remote_source_node_map, n_hosts*sizeof(int)));
  gpuErrchk(cudaMemset(d_n_remote_source_node_map, 0, n_hosts*sizeof(int)));

  // allocate and init to 0 n. of elements in the map for each source host
  gpuErrchk(cudaMalloc(&d_n_local_source_node_map, n_hosts*sizeof(int)));
  gpuErrchk(cudaMemset(d_n_local_source_node_map, 0, n_hosts*sizeof(int)));

  // initialize maps
  for (uint i_host=0; i_host<n_hosts; i_host++) {
    std::vector<uint*> rsn_map_index;
    h_remote_source_node_map_index.push_back(rsn_map_index);
      
    std::vector<uint*> lsb_map_index;
    h_local_spike_buffer_map_index.push_back(lsb_map_index);

    std::vector<uint*> lsn_map_index;
    h_local_source_node_map_index.push_back(lsn_map_index);
  }
    

  // launch kernel to copy pointers to CUDA variables ?? maybe in calibration?
  // .....
  //RemoteConnectionMapInitKernel // <<< , >>>
  //  (d_n_remote_source_node_map,
  //   d_remote_source_node_map_index,
  //   d_local_spike_buffer_map_index,
  //   d_n_local_source_node_map,
  //   d_local_source_node_map_index);
    
  return 0;
}

// Calibrate the maps
int RemoteConnectionMapCalibrate(int n_nodes)
{
  //....
  return 0;
}

// kernel that flags source nodes used in at least one new connection
// of a given block
__global__ void setUsedSourceNodeKernel(uint *key_subarray,
					int64_t n_conn,
					int *source_node_flag)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) return;
  int i_source = key_subarray[i_conn] >> MaxPortNBits;
  // it is not necessary to use atomic operation. See:
  // https://stackoverflow.com/questions/8416374/several-threads-writing-the-same-value-in-the-same-global-memory-location
  printf("i_conn: %ld\t i_source: %d\n", i_conn, i_source); 
  source_node_flag[i_source] = 1;
}
      
// Loop on all new connections and set source_node_flag[i_source]=true
int setUsedSourceNodes(std::vector<uint*> &key_subarray,
		       int64_t old_n_conn, int64_t n_conn,
		       int64_t block_size, int *d_source_node_flag)
{
  uint64_t n_new_conn = n_conn - old_n_conn; // number of new connections
  std::cout << "n_new_conn: " << n_new_conn
	    << "\tn_conn: " << n_conn
	    << "\told_n_conn: " << old_n_conn << "\n";
  
  uint ib0 = (uint)(old_n_conn / block_size); // first block index
  uint ib1 = (uint)((n_conn - 1) / block_size); // last block
  for (uint ib=ib0; ib<=ib1; ib++) { // loop on blocks
    uint64_t n_block_conn; // number of connections in a block
    uint64_t i_conn0; // index of first connection in a block
    if (ib1 == ib0) {  // all connections are in the same block
      i_conn0 = old_n_conn % block_size;
      n_block_conn = n_new_conn;
    }
    else if (ib == ib0) { // first block
      i_conn0 = old_n_conn % block_size;
      n_block_conn = block_size - i_conn0;
    }
    else if (ib == ib1) { // last block
      i_conn0 = 0;
      n_block_conn = (n_conn - 1) % block_size + 1;
    }
    else {
      i_conn0 = 0;
      n_block_conn = block_size;
    }
    std::cout << "n_new_conn: " << n_new_conn
	      << "\ti_conn0: " << i_conn0
	      << "\tn_block_conn: " << n_block_conn << "\n";
	  
    setUsedSourceNodeKernel<<<(n_block_conn+1023)/1024, 1024>>>
      (key_subarray[ib] + i_conn0, n_block_conn, d_source_node_flag);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }
  return 0;
}
      

// kernel that counts source nodes actually used in new connections
__global__ void countUsedSourceNodeKernel(uint n_source,
					  int *n_used_source_nodes,
					  int *source_node_flag)
{
  uint i_source = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_source>=n_source) return;
  // Count how many source_node_flag are true using atomic increase
  // on n_used_source_nodes
  if (source_node_flag[i_source] != 0) {
    atomicAdd(n_used_source_nodes, 1);
  }
}

// kernel that searches source node indexes in remote-connection map
__global__ void searchSourceNodeIndexInMapKernel
(
 int *source_node_map_index,
 int *spike_buffer_map_index,
 int *sorted_source_node_index,
 bool *source_node_index_to_be_mapped,
 int *n_new_source_node_map,
 int n_source)
{
  uint i_source = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_source>=n_source) return;
  // Check for sorted_source_node_index unique values:
  // - either if it is the first of the array (i_source = 0)
  // - or it is different from previous
  int node_index = sorted_source_node_index[i_source];
  if (i_source==0 || node_index!=sorted_source_node_index[i_source-1]) {
    bool mapped = false;
    // If the map is not empty search node index in the map
    if (n_source_node_map>0) {
      // determine number of blocks in node map
      int n_blocks = (n_source_node_map - 1) / node_map_block_size + 1;
      // determine number of elements in last block
      int n_node_last = (n_source_node_map - 1) % node_map_block_size + 1;
      // check if node_index is between the minimu and the maximum in the map
      if (node_index>=source_node_map_index[0][0] &&
	  node_index<=source_node_map_index[n_blocks-1][n_node_last-1]) {
	for (int ib=0; ib<n_blocks; ib++) {
	  int n = node_map_block_size;
	  if (ib==n_blocks-1) {
	    n = n_node_last;
	  }
	  if (node_index>=source_node_map_index[ib][0] &&
	      node_index<=source_node_map_index[ib][n-1]) {
	    int pos = locate(node_index, source_node_map_index[ib], n);    
	    if (source_node_map_index[ib][pos] == node_index) {
	      // If it is in the map then flag it as already mapped
	      mapped = true;
	    }
	  }
	  else if (node_index>source_node_map_index[ib][n-1]) {
	    break;
	  }
	}
      }
    }
    // If it is not in the map then flag it to be mapped
    // and atomic increase n_new_source_node_map
    if (!mapped) {
      source_node_index_to_be_mapped[i_source] = true;
      atomicInc(n_new_source_nodes_map);
    }
  }
}


