
#include <iostream>
#include <vector>

#include "connect.h"
#include "remote_connect.h"
#include "utilities.h"

// INITIALIZATION
//
// Define two arrays that map remote source nodes to local spike buffers
// There is one element for each remote host,
// so the array size is n_hosts
// Each of the two arrays contain n_remote_source_node_map elements
// that represent a map, with n_remote_source_node_map pairs
// (remote node index, local spike buffer index)
// where n_remote_source_node_map is the number of nodes in the source host
// that have outgoing connections to local nodes.
// All elements are initially empty:
// n_remote_source_nodes[i_source_host] = 0 for each i_source_host
// The map is organized in blocks each with node_map_block_size
// elements, which are allocated dynamically

__constant__ uint node_map_block_size; // = 100000;

// number of elements in the map for each source host
// n_remote_source_node_map[i_source_host]
// with i_source_host = 0, ..., n_hosts-1 excluding this host itself
__device__ uint *n_remote_source_node_map; // [n_hosts];

// remote_source_node_map[i_source_host][i_block][i]
__device__ uint ***remote_source_node_map;

// local_spike_buffer_map[i_source_host][i_block][i]
__device__ uint ***local_spike_buffer_map;

// Define two arrays that map local source nodes to remote spike buffers.
// The structure is the same as for remote source nodes

// number of elements in the map for each target host
// n_local_source_node_map[i_target_host]
// with i_target_host = 0, ..., n_hosts-1 excluding this host itself
__device__ uint *n_local_source_node_map; // [n_hosts]; 

// local_source_node_map[i_target_host][i_block][i]
__device__ uint ***local_source_node_map;

__constant__ uint n_local_nodes; // number of local nodes


// kernel that flags source nodes used in at least one new connection
// of a given block
__global__ void setUsedSourceNodeOnSourceHostKernel(inode_t *conn_source_ids,
						    int64_t n_conn,
						    uint *source_node_flag)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) return;
  inode_t i_source = conn_source_ids[i_conn];
  // it is not necessary to use atomic operation. See:
  // https://stackoverflow.com/questions/8416374/several-threads-writing-the-same-value-in-the-same-global-memory-location
  //printf("i_conn: %ld\t i_source: %d\n", i_conn, i_source);

  source_node_flag[i_source] = 1;
}

__global__ void setTargetHostArrayNodePointersKernel
(uint *target_host_array, uint *target_host_i_map, uint *n_target_hosts_cumul,
 uint **node_target_hosts, uint **node_target_host_i_map, uint n_nodes)
{
  uint i_node = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_node>=n_nodes) return;
  node_target_hosts[i_node] = target_host_array + n_target_hosts_cumul[i_node];
  node_target_host_i_map[i_node] = target_host_i_map
    + n_target_hosts_cumul[i_node];
}


// kernel that fills the arrays target_host_array
// and target_host_i_map using the node map
__global__ void fillTargetHostArrayFromMapKernel
(uint **node_map, uint n_node_map, uint *count_mapped, uint **node_target_hosts,
 uint **node_target_host_i_map, uint n_nodes, uint i_target_host)
{
  uint i_node = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_node>=n_nodes) return;
  uint i_block;
  uint i_in_block;
  // check if node index is in map
  bool mapped = checkIfValueIsIn2DArr(i_node, node_map,
				      n_node_map, node_map_block_size,
				      &i_block, &i_in_block);
  // If it is mapped
  if (mapped) {
    uint i_node_map = i_block*node_map_block_size + i_in_block;
    uint pos = count_mapped[i_node]++;
    node_target_host_i_map[i_node][pos] = i_node_map;
    node_target_hosts[i_node][pos] = i_target_host;  
  }
}



      

// kernel that counts source nodes actually used in new connections
__global__ void countUsedSourceNodeKernel(uint n_source,
					  uint *n_used_source_nodes,
					  uint *source_node_flag)
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
__device__ bool checkIfValueIsIn2DArr(uint value, uint **arr, uint n_elem,
				      uint block_size, uint *i_block,
				      uint *i_in_block)
{
  // If the array is empty surely the value is not contained in it
  if (n_elem<=0) {
    return false;
  }
  // determine number of blocks in array
  uint n_blocks = (n_elem - 1) / block_size + 1;
  // determine number of elements in last block
  uint n_last = (n_elem - 1) % block_size + 1;
  // check if value is between the minimum and the maximum in the map
  if (value<arr[0][0] ||
      value>arr[n_blocks-1][n_last-1]) {
    return false;
  }
  for (uint ib=0; ib<n_blocks; ib++) {
    if (arr[ib][0] > value) { // the array is sorted, so in this case
      return false;           // value cannot be in the following elements
    }
    uint n = block_size;
    if (ib==n_blocks-1) { // the last block can be not completely full
      n = n_last;
    }
    // search value in the block
    int pos = locate<uint, int>(value, arr[ib], (int)n);
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
 uint **node_map,
 uint n_node_map,
 uint *count_mapped, // i.e. *n_target_hosts for our application
 uint n_node)
{
  uint i_node = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_node>=n_node) return;
  uint i_block;
  uint i_in_block;
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
 uint **node_map,
 uint n_node_map,
 uint *sorted_node_index,
 bool *node_to_map,
 uint *n_node_to_map,
 uint n_node)
{
  uint i_node = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_node>=n_node) return;
  // Check for sorted_node_index unique values:
  // - either if it is the first of the array (i_node = 0)
  // - or it is different from previous
  uint node_index = sorted_node_index[i_node];
  if (i_node==0 || node_index!=sorted_node_index[i_node-1]) {
    uint i_block;
    uint i_in_block;
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
 uint **node_map,
 uint **spike_buffer_map,
 uint spike_buffer_map_i0,
 uint old_n_node_map,
 uint *sorted_node_index,
 bool *node_to_map,
 uint *i_node_to_map,
 uint n_node)
{
  uint i_node = threadIdx.x + blockIdx.x * blockDim.x;
  // if thread is out of range or node is already mapped, return
  if (i_node>=n_node || !node_to_map[i_node]) return;
  // node has to be inserted in the map
  // get and atomically increase index of node to be mapped
  uint pos = atomicAdd(i_node_to_map, 1);
  uint i_node_map = old_n_node_map + pos;
  uint i_block = i_node_map / node_map_block_size;
  uint i = i_node_map % node_map_block_size;
  node_map[i_block][i] = sorted_node_index[i_node];
  if (spike_buffer_map != NULL) {
    spike_buffer_map[i_block][i] = spike_buffer_map_i0 + pos;
  }
}



__global__ void MapIndexToSpikeBufferKernel(uint n_hosts, uint *host_offset,
					    uint *node_index)
{
  const uint i_host = blockIdx.x;
  if (i_host < n_hosts) {    
    const uint pos = host_offset[i_host];
    const uint num = host_offset[i_host+1] - pos;
    for (uint i_elem = threadIdx.x; i_elem < num; i_elem += blockDim.x) {
      const uint i_node_map = node_index[pos + i_elem];
      const uint i_block = i_node_map / node_map_block_size;
      const uint i = i_node_map % node_map_block_size;
      const uint i_spike_buffer = local_spike_buffer_map[i_host][i_block][i];
      node_index[pos + i_elem] = i_spike_buffer; 
    }
  }
}


__global__ void addOffsetToSpikeBufferMapKernel(uint i_host, uint n_node_map,
						uint i_image_node_0)
{
  uint i_node_map = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_node_map>=n_node_map) return;
  
  const uint i_block = i_node_map / node_map_block_size;
  const uint i = i_node_map % node_map_block_size;
  local_spike_buffer_map[i_host][i_block][i] += i_image_node_0;
}

