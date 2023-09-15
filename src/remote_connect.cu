#include <iostream>
#include <vector>

#include "connect.h"
#include "remote_connect.h"
#include "utilities.h"

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
// The map is organized in blocks each with node_map_block_size
// elements, which are allocated dynamically

__constant__ uint node_map_block_size; // = 100000;
uint h_node_map_block_size; // = 100000;

// number of elements in the map for each source host
// n_remote_source_node_map[i_source_host]
// with i_source_host = 0, ..., mpi_proc_num-1 excluding this host itself
__device__ uint *n_remote_source_node_map; // [mpi_proc_num];
uint *d_n_remote_source_node_map;
std::vector<uint> h_n_remote_source_node_map;

// remote_source_node_map[i_source_host][i_block][i]
std::vector< std::vector<int*> > h_remote_source_node_map;
__device__ int ***remote_source_node_map;

// local_spike_buffer_map[i_source_host][i_block][i]
std::vector< std::vector<int*> > h_local_spike_buffer_map;
__device__ int ***local_spike_buffer_map;
int ***d_local_spike_buffer_map;
// hd_local_spike_buffer_map[i_source_host] vector of pointers to gpu memory
std::vector<int**> hd_local_spike_buffer_map;

// Define two arrays that map local source nodes to remote spike buffers.
// The structure is the same as for remote source nodes

// number of elements in the map for each target host
// n_local_source_node_map[i_target_host]
// with i_target_host = 0, ..., mpi_proc_num-1 excluding this host itself
__device__ uint *n_local_source_node_map; // [mpi_proc_num]; 
uint *d_n_local_source_node_map;
std::vector<uint> h_n_local_source_node_map;

// local_source_node_map[i_target_host][i_block][i]
std::vector< std::vector<int*> > h_local_source_node_map;
__device__ int ***local_source_node_map;
int ***d_local_source_node_map;
// hd_local_source_node_map[i_target_host] vector of pointers to gpu memory
std::vector<int**> hd_local_source_node_map;


// number of remote target hosts (i.e. MPI processes) on which each local node
// has outgoing connections. Must be initially set to 0
int *d_n_target_hosts; // [n_nodes] 


// Define a boolean array with one boolean value for each connection rule
// - true if the rule always creates at least one outgoing connection
// from each source node (one_to_one, all_to_all, fixed_outdegree)
// - false otherwise (fixed_indegree, fixed_total_number, pairwise_bernoulli)
bool *use_all_source_nodes; // [n_connection_rules]:

// Allocate GPU memory for new remote-source-node-map blocks
int allocRemoteSourceNodeMapBlocks(std::vector<int*> &i_remote_src_node_map,
				   std::vector<int*> &i_local_spike_buf_map,
				   int64_t block_size, uint new_n_block)
{
  // allocate new blocks if needed
  for (uint ib=i_remote_src_node_map.size(); ib<new_n_block; ib++) {
    int *d_remote_src_node_blk_pt;
    int *d_local_spike_buf_blk_pt;
    // allocate GPU memory for new blocks 
    gpuErrchk(cudaMalloc(&d_remote_src_node_blk_pt, block_size*sizeof(int)));
    gpuErrchk(cudaMalloc(&d_local_spike_buf_blk_pt, block_size*sizeof(int)));
      
    i_remote_src_node_map.push_back(d_remote_src_node_blk_pt);
    i_local_spike_buf_map.push_back(d_local_spike_buf_blk_pt);
  }
    
  return 0;
}

// Allocate GPU memory for new local-source-node-map blocks
int allocLocalSourceNodeMapBlocks(std::vector<int*> &i_local_src_node_map,
				  int64_t block_size, uint new_n_block)
{
  // allocate new blocks if needed
  for (uint ib=i_local_src_node_map.size(); ib<new_n_block; ib++) {
    int *d_local_src_node_blk_pt;
    // allocate GPU memory for new blocks 
    gpuErrchk(cudaMalloc(&d_local_src_node_blk_pt, block_size*sizeof(int)));
      
    i_local_src_node_map.push_back(d_local_src_node_blk_pt);
  }
    
  return 0;
}


// Initialize the maps for n_hosts hosts (i.e. number of MPI processes)
int RemoteConnectionMapInit(uint n_hosts)
{
  h_node_map_block_size = 3; //10000; // initialize node map block size
  cudaMemcpyToSymbol(node_map_block_size, &h_node_map_block_size, sizeof(int));

  // allocate and init to 0 n. of elements in the map for each source host
  gpuErrchk(cudaMalloc(&d_n_remote_source_node_map, n_hosts*sizeof(int)));
  gpuErrchk(cudaMemset(d_n_remote_source_node_map, 0, n_hosts*sizeof(int)));

  // allocate and init to 0 n. of elements in the map for each source host
  gpuErrchk(cudaMalloc(&d_n_local_source_node_map, n_hosts*sizeof(int)));
  gpuErrchk(cudaMemset(d_n_local_source_node_map, 0, n_hosts*sizeof(int)));

  // initialize maps
  for (uint i_host=0; i_host<n_hosts; i_host++) {
    std::vector<int*> rsn_map;
    h_remote_source_node_map.push_back(rsn_map);
      
    std::vector<int*> lsb_map;
    h_local_spike_buffer_map.push_back(lsb_map);

    std::vector<int*> lsn_map;
    h_local_source_node_map.push_back(lsn_map);
  }
    

  // launch kernel to copy pointers to CUDA variables ?? maybe in calibration?
  // .....
  //RemoteConnectionMapInitKernel // <<< , >>>
  //  (d_n_remote_source_node_map,
  //   d_remote_source_node_map,
  //   d_local_spike_buffer_map,
  //   d_n_local_source_node_map,
  //   d_local_source_node_map);
    
  return 0;
}

// Calibrate the maps
int  NESTGPU::RemoteConnectionMapCalibrate(int i_host, int n_hosts)
{
  // vector of pointers to local source node maps in device memory
  // per target host hd_local_source_node_map[target_host]
  // type std::vector<int*>
  // set its size and initialize to NULL
  hd_local_source_node_map.resize(n_hosts, NULL);
  // number of elements in each local source node map
  // h_n_local_source_node_map[target_host]
  // set its size and initialize to 0
  h_n_local_source_node_map.resize(n_hosts, 0);
  // vector of pointers to local spike buffer maps in device memory
  // per source host hd_local_spike_buffer_map[source_host]
  // type std::vector<int*>
  // set its size and initialize to NULL
  hd_local_spike_buffer_map.resize(n_hosts, NULL);
  // number of elements in each remote-source-node->local-spike-buffer map
  // h_n_remote_source_node_map[source_host]
  // set its size and initialize to 0
  h_n_remote_source_node_map.resize(n_hosts, 0);
  // loop on target hosts, skip self host
  for (int tg_host=0; tg_host<n_hosts; tg_host++) {
    if (tg_host != i_host) {
      // get number of elements in each map from device memory
      int n_node_map;
      gpuErrchk(cudaMemcpy(&n_node_map,
			   &d_n_local_source_node_map[tg_host], sizeof(int),
			   cudaMemcpyDeviceToHost));
      // put it in h_n_local_source_node_map[tg_host]
      h_n_local_source_node_map[tg_host] = n_node_map;
      // Allocate array of local source node map blocks
      // and copy their address from host to device
      int n_blocks = h_local_source_node_map[tg_host].size();
      if (n_blocks>0) {
	gpuErrchk(cudaMalloc(&hd_local_source_node_map[tg_host],
			     n_blocks*sizeof(int*)));
	gpuErrchk(cudaMemcpy(hd_local_source_node_map[tg_host],
			     &h_local_source_node_map[tg_host][0],
			     n_blocks*sizeof(int*),
			     cudaMemcpyHostToDevice));
      }
    }
  }
  // allocate d_local_source_node_map and copy it from host to device
  gpuErrchk(cudaMalloc(&d_local_source_node_map, n_hosts*sizeof(int**)));
  gpuErrchk(cudaMemcpy(d_local_source_node_map, &hd_local_source_node_map[0],
		       n_hosts*sizeof(int**), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyToSymbol(local_source_node_map,
			       &d_local_source_node_map, sizeof(int***)));

  // loop on source hosts, skip self host
  for (int src_host=0; src_host<n_hosts; src_host++) {
    if (src_host != i_host) {
      // get number of elements in each map from device memory
      int n_node_map;
      gpuErrchk(cudaMemcpy(&n_node_map,
			   &d_n_remote_source_node_map[src_host], sizeof(int),
			   cudaMemcpyDeviceToHost));
      // put it in h_n_remote_source_node_map[src_host]
      h_n_remote_source_node_map[src_host] = n_node_map;
      // Allocate array of local spike buffer map blocks
      // and copy their address from host to device
      int n_blocks = h_local_spike_buffer_map[src_host].size();
      if (n_blocks>0) {
	gpuErrchk(cudaMalloc(&hd_local_spike_buffer_map[src_host],
			     n_blocks*sizeof(int*)));
	gpuErrchk(cudaMemcpy(hd_local_spike_buffer_map[src_host],
			     &h_local_spike_buffer_map[src_host][0],
			     n_blocks*sizeof(int*),
			     cudaMemcpyHostToDevice));
      }
    }
  }
  // allocate d_local_spike_buffer_map and copy it from host to device
  gpuErrchk(cudaMalloc(&d_local_spike_buffer_map, n_hosts*sizeof(int**)));
  gpuErrchk(cudaMemcpy(d_local_spike_buffer_map, &hd_local_spike_buffer_map[0],
		       n_hosts*sizeof(int**), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyToSymbol(local_spike_buffer_map,
			       &d_local_spike_buffer_map, sizeof(int***)));

  //// TEMPORARY, FOR CHECK
  std::cout << "////////////////////////////////////////\n";
  std::cout << "IN MAP CALIBRATION\n";
  
  int tmp_n_hosts = 2;
  int tmp_tg_host = 0;
  int tmp_src_host = 1;
  
  int **tmp_pt2[tmp_n_hosts];
  int tmp_n[tmp_n_hosts];
  int tmp_map[h_node_map_block_size];
  int n_map;
  int n_blocks;

  gpuErrchk(cudaMemcpy(tmp_n, d_n_local_source_node_map,
		       tmp_n_hosts*sizeof(int), cudaMemcpyDeviceToHost));
  n_map = tmp_n[tmp_tg_host];
  if (n_map>0) {
    std::cout << "////////////////////////////////////////\n";
    std::cout << "Local Source Node Map\n";
    std::cout << "target host: " << tmp_tg_host << "\n";
    std::cout << "n_local_source_node_map: " << n_map << "\n";
    gpuErrchk(cudaMemcpy(tmp_pt2, d_local_source_node_map,
			 tmp_n_hosts*sizeof(int**), cudaMemcpyDeviceToHost));
  
    n_blocks = (n_map - 1) / h_node_map_block_size + 1;
    std::cout << "n_blocks: " << n_blocks << "\n";
    int *tmp_pt1[n_blocks];
    gpuErrchk(cudaMemcpy(tmp_pt1, tmp_pt2[tmp_tg_host],
			 n_blocks*sizeof(int*), cudaMemcpyDeviceToHost));
    
    for (int ib=0; ib<n_blocks; ib++) {
      std::cout << "block " << ib << "\n";
      int n = h_node_map_block_size;
      if (ib==n_blocks-1) {
	n = (n_map - 1) % h_node_map_block_size + 1;
      }
      gpuErrchk(cudaMemcpy(tmp_map, tmp_pt1[ib],
			   n*sizeof(int), cudaMemcpyDeviceToHost));
      std::cout << "local source node index\n";
      for (int i=0; i<n; i++) {
	std::cout << tmp_map[i] << "\n";
      }
    }
  }

  //gpuErrchk(cudaMemcpy(tmp_n, d_n_local_spike_buffer_map,
  gpuErrchk(cudaMemcpy(tmp_n, d_n_remote_source_node_map,
		       tmp_n_hosts*sizeof(int), cudaMemcpyDeviceToHost));
  n_map = tmp_n[tmp_src_host];
  if (n_map>0) {
    std::cout << "////////////////////////////////////////\n";
    std::cout << "Local Spike Buffer Map\n";
    std::cout << "source host: " << tmp_src_host << "\n";
    std::cout << "n_local_spike_buffer_map: " << n_map << "\n";
    gpuErrchk(cudaMemcpy(tmp_pt2, d_local_spike_buffer_map,
			 tmp_n_hosts*sizeof(int**), cudaMemcpyDeviceToHost));
  
    n_blocks = (n_map - 1) / h_node_map_block_size + 1;
    std::cout << "n_blocks: " << n_blocks << "\n";
    int *tmp_pt1[n_blocks];
    gpuErrchk(cudaMemcpy(tmp_pt1, tmp_pt2[tmp_src_host],
			 n_blocks*sizeof(int*), cudaMemcpyDeviceToHost));
    
    for (int ib=0; ib<n_blocks; ib++) {
      std::cout << "block " << ib << "\n";
      int n = h_node_map_block_size;
      if (ib==n_blocks-1) {
	n = (n_map - 1) % h_node_map_block_size + 1;
      }
      gpuErrchk(cudaMemcpy(tmp_map, tmp_pt1[ib],
			   n*sizeof(int), cudaMemcpyDeviceToHost));
      std::cout << "local spike buffer index\n";
      for (int i=0; i<n; i++) {
	std::cout << tmp_map[i] << "\n";
      }
    }
  }

  ////////////////////////////////////////
  
  int n_nodes = GetNNode(); // number of nodes
  // n_target_hosts[i_node] is the number of remote target hosts
  // (i.e. MPI processes) on which each local node
  // has outgoing connections
  // allocate d_n_target_hosts[n_nodes] and init to 0
  gpuErrchk(cudaMalloc(&d_n_target_hosts, n_nodes*sizeof(int)));
  gpuErrchk(cudaMemset(d_n_target_hosts, 0, n_nodes*sizeof(int)));

  // For each local node, count the number of remote target hosts
  // on which it has outgoing connections, i.e. n_target_hosts[i_node] 
  // Loop on target hosts (i.e. on MPI processes)
  for (int tg_host=0; tg_host<n_hosts; tg_host++) {
    if (tg_host != i_host) {
      int **d_node_map = hd_local_source_node_map[tg_host];
      int n_node_map = h_n_local_source_node_map[tg_host];
      // Launch kernel that searches each node in the map
      // of local source nodes having outgoing connections to target host
      // if found, increase n_target_hosts[i_node]
      searchNodeIndexInMapKernel<<<(n_nodes+1023)/1024, 1024>>>
	(d_node_map, n_node_map, d_n_target_hosts, n_nodes);
      gpuErrchk( cudaPeekAtLastError() );
      gpuErrchk( cudaDeviceSynchronize() );
    }
  }

  // TEMPORARY, FOR TESTING
  int h_n_target_hosts[n_nodes];
  gpuErrchk(cudaMemcpy(h_n_target_hosts, d_n_target_hosts,
  		       n_nodes*sizeof(int), cudaMemcpyDeviceToHost));
  std::cout << "////////////////////////////////////////\n";
  std::cout << "i_node, n_target_hosts\n";
  for (int i_node=0; i_node<n_nodes; i_node++) {
    std::cout << i_node << "\t" << h_n_target_hosts[i_node] << "\n";
  }
  //////////////////////////////////////////////////////////////////////
  
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


// device function that checks if an int value is in a sorted 2d-array 
// assuming that the entries in the 2d-array are sorted.
// The 2d-array is divided in noncontiguous blocks of size block_size
__device__ bool checkIfValueIsIn2DArr(int value, int **arr, int n_elem,
				      int block_size, int *i_block,
				      int *i_in_block)
{
  // If the array is empty surely the value is not contained in it
  if (n_elem<=0) {
    return false;
  }
  // determine number of blocks in array
  int n_blocks = (n_elem - 1) / block_size + 1;
  // determine number of elements in last block
  int n_last = (n_elem - 1) % block_size + 1;
  // check if value is between the minimum and the maximum in the map
  if (value<arr[0][0] ||
      value>arr[n_blocks-1][n_last-1]) {
    return false;
  }
  for (int ib=0; ib<n_blocks; ib++) {
    if (arr[ib][0] > value) { // the array is sorted, so in this case
      return false;           // value cannot be in the following elements
    }
    int n = block_size;
    if (ib==n_blocks-1) { // the last block can be not completely full
      n = n_last;
    }
    // search value in the block
    int pos = locate<int, int>(value, arr[ib], n);
    // if value is in the block return true
    if (pos>=0 && pos<n && arr[ib][pos]==value) {
      *i_block = ib;
      *i_in_block = pos;
      return true;
    }
  }
  return false; // value not found
}  


// kernel that searches node indexes in map
// increase counter of mapped nodes
__global__ void searchNodeIndexInMapKernel
(
 int **node_map,
 int n_node_map,
 int *count_mapped, // i.e. *n_target_hosts for our application
 int n_node)
{
  uint i_node = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_node>=n_node) return;
  int i_block;
  int i_in_block;
  // check if node index is in map
  bool mapped = checkIfValueIsIn2DArr(i_node, node_map,
				      n_node_map, node_map_block_size,
				      &i_block, &i_in_block);
  // If it is mapped
  if (mapped) {
    // i_node_map = i_block*node_map_block_size + i_in_block;
    count_mapped[i_node]++;
  }
}

// kernel that searches node indexes not in map
// flags nodes not yet mapped and counts them
__global__ void searchNodeIndexNotInMapKernel
(
 int **node_map,
 int n_node_map,
 int *sorted_node_index,
 bool *node_to_map,
 int *n_node_to_map,
 int n_node)
{
  uint i_node = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_node>=n_node) return;
  // Check for sorted_node_index unique values:
  // - either if it is the first of the array (i_node = 0)
  // - or it is different from previous
  int node_index = sorted_node_index[i_node];
  if (i_node==0 || node_index!=sorted_node_index[i_node-1]) {
    int i_block;
    int i_in_block;
    bool mapped = checkIfValueIsIn2DArr(node_index, node_map,
					n_node_map, node_map_block_size,
					&i_block, &i_in_block);
    // If it is not in the map then flag it to be mapped
    // and atomic increase n_new_source_node_map
    if (!mapped) {
      node_to_map[i_node] = true;
      atomicAdd(n_node_to_map, 1);
    }
  }
}


// kernel that checks if nodes are already in map
// if not insert them in the map
// In the target host unmapped remote source nodes must be mapped
// to local nodes from n_nodes to n_nodes + n_node_to_map
__global__ void insertNodesInMapKernel
(
 int **node_map,
 int **spike_buffer_map,
 int spike_buffer_map_i0,
 int old_n_node_map,
 int *sorted_node_index,
 bool *node_to_map,
 int *i_node_to_map,
 int n_node)
{
  uint i_node = threadIdx.x + blockIdx.x * blockDim.x;
  // if thread is out of range or node is already mapped, return
  if (i_node>=n_node || !node_to_map[i_node]) return;
  // node has to be inserted in the map
  // get and atomically increase index of node to be mapped
  int pos = atomicAdd(i_node_to_map, 1);
  int i_node_map = old_n_node_map + pos;
  int i_block = i_node_map / node_map_block_size;
  int i = i_node_map % node_map_block_size;
  node_map[i_block][i] = sorted_node_index[i_node];
  if (spike_buffer_map != NULL) {
    spike_buffer_map[i_block][i] = spike_buffer_map_i0 + pos;
  }
}

// kernel that replaces the source node index
// in a new remote connection of a given block
// source_node[i_conn] with the value of the element pointed by the
// index itself in the array local_node_index
__global__ void fixConnectionSourceNodeIndexesKernel(uint *key_subarray,
						     int64_t n_conn,
						     int *local_node_index)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) return;
  int i_source = key_subarray[i_conn] >> MaxPortNBits;
  int i_delay = key_subarray[i_conn] & PortMask;
  int new_i_source = local_node_index[i_source];

  key_subarray[i_conn] = (new_i_source << MaxPortNBits) | i_delay;
  printf("i_conn: %ld\t new_i_source: %d\n", i_conn, new_i_source); 
}

// Loops on all new connections and replaces the source node index
// source_node[i_conn] with the value of the element pointed by the
// index itself in the array local_node_index
int fixConnectionSourceNodeIndexes(std::vector<uint*> &key_subarray,
				   int64_t old_n_conn, int64_t n_conn,
				   int64_t block_size,
				   int *d_local_node_index)
{
  uint64_t n_new_conn = n_conn - old_n_conn; // number of new connections
  std::cout << "Fixing source node indexes in new remote connections\n";
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
	  
    fixConnectionSourceNodeIndexesKernel<<<(n_block_conn+1023)/1024, 1024>>>
      (key_subarray[ib] + i_conn0, n_block_conn, d_local_node_index);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }
  return 0;
}
