
#ifndef REMOTECONNECTH
#define REMOTECONNECTH

// #include <cub/cub.cuh>
#include <fstream>
#include <vector>

#include "getRealTime.h"
// #include "nestgpu.h"
#include "connect.h"
#include "copass_sort.h"
#include "cuda_error.h"
// Arrays that map remote source nodes to local image nodes

// The map is organized in blocks having block size:
extern __constant__ uint node_map_block_size; // = 100000;

// number of elements in the map for each host group and for each source host in
// the group n_remote_source_node_map[group_local_id][i_host] with i_host = 0,
// ..., host_group_[group_local_id].size()-1 excluding this host itself
extern __device__ uint **n_remote_source_node_map;

// remote_source_node_map[group_local_id][i_host][i_block][i]
extern __device__ uint ****remote_source_node_map;

// image_node_map[group_local_id][i_host][i_block][i]
extern __device__ uint ****local_image_node_map;

// Arrays that map local source nodes to remote image nodes

// number of elements in the map for each target host
// n_local_source_node_map[i_target_host]
// with i_target_host = 0, ..., n_hosts-1 excluding this host itself
extern __device__ uint *n_local_source_node_map; // [n_hosts];

// local_source_node_map[i_target_host][i_block][i]
extern __device__ uint ***local_source_node_map;

extern __constant__ uint n_local_nodes; // number of local nodes

inline uint msb(uint val) {
  uint r = 0;
  while (val >>= 1) {
    r++;
  }
  return r;
}

// device function that checks if an int value is in a sorted 2d-array
// assuming that the entries in the 2d-array are sorted.
// The 2d-array is divided in noncontiguous blocks of size block_size
__device__ bool checkIfValueIsIn2DArr(uint value, uint **arr, uint n_elem,
                                      uint block_size, uint *i_block,
                                      uint *i_in_block);

// kernel that copies part of a block array to an array
__global__ void copyBlockArrayToArrayKernel(uint *array, uint **block_array,
                                            uint block_size, uint i0,
                                            uint n_elem);

// kernel that copies an array to part of a block array
__global__ void copyArrayToBlockArrayKernel(uint **block_array, uint *array,
                                            uint block_size, uint i0,
                                            uint n_elem);

// Kernel that extracts local image index from already mapped remote source
// nodes and match it to the nodes in a sequence
__global__ void extractLocalImageIndexOfMappedSourceNodes(
    uint **node_map, uint i_node_map_0, uint n_elem, uint i_node_0,
    bool *node_mapped, bool use_image_node_map = false,
    uint **image_node_map = nullptr, uint *local_node_index = nullptr);

// Kernel that maps remote source nodes in a sequence to local images
// checking if nodes are already in map
// Unmapped remote source nodes must be mapped
// to local nodes from n_nodes to n_nodes + n_node_to_map
__global__ void mapRemoteSourceNodesToLocalImagesKernel(
    uint **node_map, bool *node_mapped, uint n_node, uint i_node_map_0,
    uint i_node_0, bool use_image_node_map = false,
    uint **image_node_map = nullptr, uint image_node_map_i0 = 0,
    uint *i_node_to_map = nullptr, uint *local_node_index = nullptr);

template <class ConnKeyT>
__global__ void checkUsedSourceNodeKernel(ConnKeyT *conn_key_subarray,
                                          int64_t n_conn,
                                          uint *source_node_flag, int this_host,
                                          uint n_source) {
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn >= n_conn) {
    return;
  }
  inode_t i_source = getConnSource<ConnKeyT>(conn_key_subarray[i_conn]);
  if (i_source >= n_source) {
    printf("this_host: %d\t CUSNK i_source: %d\tn_source: %d, i_conn: %ld\n",
           this_host, i_source, n_source, i_conn);
  }
  // printf("i_conn: %ld\t i_source: %d\n", i_conn, i_source);
}

template <class ConnKeyT>
// kernel that flags source nodes used in at least one new connection
// of a given block
__global__ void setUsedSourceNodeKernel(ConnKeyT *conn_key_subarray,
                                        int64_t n_conn,
                                        uint *source_node_flag) {
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn >= n_conn) {
    return;
  }
  inode_t i_source = getConnSource<ConnKeyT>(conn_key_subarray[i_conn]);
  // it is not necessary to use atomic operation. See:
  // https://stackoverflow.com/questions/8416374/several-threads-writing-the-same-value-in-the-same-global-memory-location
  // printf("i_conn: %ld\t i_source: %d\n", i_conn, i_source);

  source_node_flag[i_source] = 1;
}

// kernel that flags source nodes used in at least one new connection
// of a given block
__global__ void setUsedSourceNodeOnSourceHostKernel(inode_t *conn_source_ids,
                                                    int64_t n_conn,
                                                    uint *source_node_flag);

// kernel that fills the arrays of nodes actually used by new connections
template <class T>
__global__ void
getUsedSourceNodeIndexKernel(T source, uint n_source, uint *n_used_source_nodes,
                             bool use_source_node_flag, bool use_i_source_arr,
                             uint *source_node_flag, uint *u_source_node_idx,
                             uint *i_source_arr = nullptr) {
  uint i_source = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_source >= n_source) {
    return;
  }
  if (use_source_node_flag) {
    // Count how many source_node_flag are true using atomic increase
    // on n_used_source_nodes
    if (source_node_flag[i_source] != 0) {
      uint pos = atomicAdd(n_used_source_nodes, 1);
      u_source_node_idx[pos] = getNodeIndex(source, i_source);
      // printf("i_source: %d\tpos: %d\tu_source_node_idx: %d\n", i_source, pos,
      // u_source_node_idx[ pos ]);
      if (use_i_source_arr) {
        i_source_arr[pos] = i_source;
      }
    }
  } else {
    u_source_node_idx[i_source] = getNodeIndex(source, i_source);
    // printf("i_source: %d\t u_source_node_idx: %d\n", i_source,
    // u_source_node_idx[ i_source ]);
    if (use_i_source_arr) {
      i_source_arr[i_source] = i_source;
    }
  }
}

// kernel that counts source nodes actually used in new connections
__global__ void countUsedSourceNodeKernel(uint n_source,
                                          uint *n_used_source_nodes,
                                          uint *source_node_flag);

// kernel that searches source node indexes in the map,
// and set local_node_index
template <class T>
__global__ void setLocalNodeIndexKernel(T source, uint n_source,
                                        bool use_source_node_flag,
                                        uint *source_node_flag, uint **node_map,
                                        uint **image_node_map, uint n_node_map,
                                        uint *local_node_index) {
  uint i_source = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_source >= n_source) {
    return;
  }
  // Count how many source_node_flag are true using atomic increase
  // on n_used_source_nodes
  if (!use_source_node_flag || source_node_flag[i_source] != 0) {
    uint node_index = getNodeIndex(source, i_source);
    uint i_block;
    uint i_in_block;
    bool mapped =
        checkIfValueIsIn2DArr(node_index, node_map, n_node_map,
                              node_map_block_size, &i_block, &i_in_block);
    if (!mapped) {
      printf("Error in setLocalNodeIndexKernel: node index not mapped\n");
      return;
    }
    uint i_image_node = image_node_map[i_block][i_in_block];
    local_node_index[i_source] = i_image_node;
  }
}

// kernel that replaces the source node index
// in a new remote connection of a given block
// source_node[i_conn] with the value of the element pointed by the
// index itself in the array local_node_index
template <class ConnKeyT>
__global__ void
fixConnectionSourceNodeIndexesKernel(ConnKeyT *conn_key_subarray,
                                     int64_t n_conn, uint *local_node_index) {
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn >= n_conn) {
    return;
  }
  uint i_source = getConnSource<ConnKeyT>(conn_key_subarray[i_conn]);
  uint i_delay = getConnDelay<ConnKeyT>(conn_key_subarray[i_conn]);
  uint new_i_source = local_node_index[i_source];

  setConnSource<ConnKeyT>(conn_key_subarray[i_conn], new_i_source);

  // printf("i_conn: %ld\t new_i_source: %d\n", i_conn, new_i_source);
}

// kernel that searches node indexes in map
// increase counter of mapped nodes
__global__ void searchNodeIndexInMapKernel(
    uint **node_map, uint n_node_map,
    uint *count_mapped, // i.e. *n_target_hosts for our application
    uint n_node);

// kernel that searches node indexes in map
// flags nodes not yet mapped and counts them
__global__ void searchNodeIndexNotInMapKernel(
    uint **node_map, uint n_node_map, uint *sorted_node_index,
    bool *node_to_map, uint *n_node_to_map, uint n_node,
    bool use_image_node_map = false, uint **image_node_map = nullptr,
    uint *mapped_local_node_index = nullptr);

// kernel that checks if nodes are already in map
// if not insert them in the map
// In the target host unmapped remote source nodes must be mapped
// to local nodes from n_nodes to n_nodes + n_node_to_map
__global__ void insertNodesInMapKernel(
    uint **node_map, uint old_n_node_map, uint *sorted_node_index,
    bool *node_to_map, uint *i_node_to_map, uint n_node,
    bool use_image_node_map = false, uint **image_node_map = nullptr,
    uint image_node_map_i0 = 0, uint *i_sorted_arr = nullptr,
    uint *local_node_index = nullptr, uint *mapped_local_node_index = nullptr);

template <class ConnKeyT, class ConnStructT>
__global__ void
addOffsetToExternalNodeIdsKernel(int64_t n_conn, ConnKeyT *conn_key_subarray,
                                 ConnStructT *conn_struct_subarray,
                                 uint i_image_node_0) {
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn >= n_conn) {
    return;
  }
  // uint target_port_syn = conn_subarray[i_conn].target_port_syn;
  // if (target_port_syn & (1 << (MaxPortSynNBits - 1))) {
  // target_port_syn = target_port_syn ^ (1 << (MaxPortSynNBits - 1));
  // conn_subarray[i_conn].target_port_syn = target_port_syn;
  // key_subarray[i_conn] += (i_image_node_0 << MaxPortSynNBits);
  uint remote_flag = getConnRemoteFlag<ConnKeyT, ConnStructT>(
      conn_key_subarray[i_conn], conn_struct_subarray[i_conn]);
  if (remote_flag == 1) {
    // IN THE FUTURE KEEP IT!!!!!!!!!!!!!!!!!!!!!!!!!!
    clearConnRemoteFlag<ConnKeyT, ConnStructT>(conn_key_subarray[i_conn],
                                               conn_struct_subarray[i_conn]);
    uint i_source = getConnSource<ConnKeyT>(conn_key_subarray[i_conn]);
    i_source += i_image_node_0;
    setConnSource<ConnKeyT>(conn_key_subarray[i_conn], i_source);
  }
}

__global__ void MapIndexToImageNodeKernel(uint n_hosts, uint *host_offset,
                                          uint *node_index);

// only for debugging
__global__ void checkMapIndexToImageNodeKernel(uint n_hosts, uint *host_offset,
                                               uint *node_index, uint *n_map,
                                               uint node_index_size,
                                               int this_host);

// Allocate GPU memory for new remote-source-node-map blocks
template <class ConnKeyT, class ConnStructT>
int ConnectionTemplate<ConnKeyT, ConnStructT>::allocRemoteSourceNodeMapBlocks(
    std::vector<uint *> &i_remote_src_node_map,
    std::vector<uint *> &i_local_img_node_map, uint new_n_block) {
  // allocate new blocks if needed
  for (uint ib = i_remote_src_node_map.size(); ib < new_n_block; ib++) {
    uint *d_remote_src_node_blk_pt;
    uint *d_local_img_node_blk_pt;
    // allocate GPU memory for new blocks
    CUDAMALLOCCTRL("&d_remote_src_node_blk_pt", &d_remote_src_node_blk_pt,
                   node_map_block_size_ * sizeof(uint));
    CUDAMALLOCCTRL("&d_local_img_node_blk_pt", &d_local_img_node_blk_pt,
                   node_map_block_size_ * sizeof(uint));

    i_remote_src_node_map.push_back(d_remote_src_node_blk_pt);
    i_local_img_node_map.push_back(d_local_img_node_blk_pt);
  }

  return 0;
}

// Allocate GPU memory for new local-source-node-map blocks
template <class ConnKeyT, class ConnStructT>
int ConnectionTemplate<ConnKeyT, ConnStructT>::allocLocalSourceNodeMapBlocks(
    std::vector<uint *> &i_local_src_node_map, uint new_n_block) {
  // allocate new blocks if needed
  for (uint ib = i_local_src_node_map.size(); ib < new_n_block; ib++) {
    uint *d_local_src_node_blk_pt;
    // allocate GPU memory for new blocks
    CUDAMALLOCCTRL("&d_local_src_node_blk_pt", &d_local_src_node_blk_pt,
                   node_map_block_size_ * sizeof(uint));

    i_local_src_node_map.push_back(d_local_src_node_blk_pt);
  }

  return 0;
}

// Loop on all new connections and set source_node_flag[i_source]=true
template <class ConnKeyT, class ConnStructT>
int ConnectionTemplate<ConnKeyT, ConnStructT>::setUsedSourceNodes(
    int64_t old_n_conn, uint *d_source_node_flag)
// , int n_source ) // uncomment only for debugging
{
  int64_t n_new_conn = n_conn_ - old_n_conn; // number of new connections

  uint ib0 = (uint)(old_n_conn / conn_block_size_);    // first block index
  uint ib1 = (uint)((n_conn_ - 1) / conn_block_size_); // last block
  for (uint ib = ib0; ib <= ib1; ib++) {               // loop on blocks
    int64_t n_block_conn; // number of connections in a block
    int64_t i_conn0;      // index of first connection in a block
    if (ib1 == ib0) {     // all connections are in the same block
      i_conn0 = old_n_conn % conn_block_size_;
      n_block_conn = n_new_conn;
    } else if (ib == ib0) { // first block
      i_conn0 = old_n_conn % conn_block_size_;
      n_block_conn = conn_block_size_ - i_conn0;
    } else if (ib == ib1) { // last block
      i_conn0 = 0;
      n_block_conn = (n_conn_ - 1) % conn_block_size_ + 1;
    } else {
      i_conn0 = 0;
      n_block_conn = conn_block_size_;
    }

    // uncomment only for debugging
    // checkUsedSourceNodeKernel< ConnKeyT > <<< ( n_block_conn + 1023 ) / 1024,
    // 1024 >>>
    //  (conn_key_vect_[ ib ] + i_conn0, n_block_conn, d_source_node_flag,
    //  this_host_, n_source);
    // CUDASYNC;
    if (n_block_conn > 0) {
      setUsedSourceNodeKernel<ConnKeyT><<<(n_block_conn + 1023) / 1024, 1024>>>(
          conn_key_vect_[ib] + i_conn0, n_block_conn, d_source_node_flag);
      CUDASYNC;
    }
  }
  return 0;
}

// Loop on all new connections and set source_node_flag[i_source]=true
template <class ConnKeyT, class ConnStructT>
int ConnectionTemplate<ConnKeyT, ConnStructT>::setUsedSourceNodesOnSourceHost(
    int64_t old_n_conn, uint *d_source_node_flag) {
  int64_t n_new_conn = n_conn_ - old_n_conn; // number of new connections

  if (n_new_conn > 0) {
    setUsedSourceNodeOnSourceHostKernel<<<(n_new_conn + 1023) / 1024, 1024>>>(
        d_conn_source_ids_, n_new_conn, d_source_node_flag);
    CUDASYNC;
  }

  return 0;
}

__global__ void setTargetHostArrayNodePointersKernel(
    uint *target_host_array, uint *target_host_i_map,
    uint *n_target_hosts_cumul, uint **node_target_hosts,
    uint **node_target_host_i_map, uint n_nodes);

// kernel that fills the arrays target_host_array
// and target_host_i_map using the node map
__global__ void
fillTargetHostArrayFromMapKernel(uint **node_map, uint n_node_map,
                                 uint *count_mapped, uint **node_target_hosts,
                                 uint **node_target_host_i_map, uint n_nodes,
                                 uint i_target_host);

__global__ void addOffsetToImageNodeMapKernel(uint group_local_id, uint gi_host,
                                              uint n_node_map,
                                              uint i_image_node_0);

// Initialize the maps
template <class ConnKeyT, class ConnStructT>
int ConnectionTemplate<ConnKeyT, ConnStructT>::remoteConnectionMapInit() {
  node_map_block_size_ = 128 * 1024; // 10000; // initialize node map block size

  cudaMemcpyToSymbol(node_map_block_size, &node_map_block_size_, sizeof(uint));

  // number of host groups
  uint nhg = host_group_.size();
  // first evaluate how many elements must be allocated
  uint elem_to_alloc = 0;
  for (uint group_local_id = 0; group_local_id < nhg; group_local_id++) {
    uint nh =
        host_group_[group_local_id].size(); // number of hosts in the group
    elem_to_alloc += nh;
  }
  uint *d_n_node_map;
  // allocate the whole array and initialize each element to 0
  CUDAMALLOCCTRL("&d_n_node_map", &d_n_node_map, elem_to_alloc * sizeof(uint));
  gpuErrchk(cudaMemset(d_n_node_map, 0, elem_to_alloc * sizeof(uint)));
  // now assign the proper address to each element of the array
  d_n_remote_source_node_map_.push_back(d_n_node_map);
  for (uint group_local_id = 1; group_local_id < nhg; group_local_id++) {
    uint nh =
        host_group_[group_local_id - 1].size(); // number of hosts in the group
    d_n_remote_source_node_map_.push_back(
        d_n_remote_source_node_map_[group_local_id - 1] + nh);
  }

  // allocate and init to 0 n. of elements in the map for each source host
  CUDAMALLOCCTRL("&d_n_local_source_node_map_", &d_n_local_source_node_map_,
                 n_hosts_ * sizeof(uint));
  gpuErrchk(cudaMemset(d_n_local_source_node_map_, 0, n_hosts_ * sizeof(uint)));

  // initialize maps
  for (uint group_local_id = 0; group_local_id < nhg; group_local_id++) {
    uint nh =
        host_group_[group_local_id].size(); // number of hosts in the group
    std::vector<std::vector<uint *>> rsn_map;
    std::vector<std::vector<uint *>> lin_map;
    for (uint ih = 0; ih < nh; ih++) {
      std::vector<uint *> rsn_ih_map;
      std::vector<uint *> lin_ih_map;
      rsn_map.push_back(rsn_ih_map);
      lin_map.push_back(lin_ih_map);
    }
    h_remote_source_node_map_.push_back(rsn_map);
    h_image_node_map_.push_back(lin_map);

    hc_remote_source_node_map_.push_back(
        std::vector<std::vector<uint>>(nh, std::vector<uint>()));
    hc_image_node_map_.push_back(
        std::vector<std::vector<uint>>(nh, std::vector<uint>()));
  }

  for (int i_host = 0; i_host < n_hosts_; i_host++) {
    std::vector<uint *> lsn_map;
    h_local_source_node_map_.push_back(lsn_map);
  }

  // launch kernel to copy pointers to CUDA variables ?? maybe in calibration?
  // .....
  // RemoteConnectionMapInitKernel // <<< , >>>
  //  (d_n_remote_source_node_map_,
  //   d_remote_source_node_map,
  //   d_image_node_map,
  //   d_n_local_source_node_map_,
  //   d_local_source_node_map);

  return 0;
}

// Loops on all new connections and replaces the source node index
// source_node[i_conn] with the value of the element pointed by the
// index itself in the array local_node_index
template <class ConnKeyT, class ConnStructT>
int ConnectionTemplate<ConnKeyT, ConnStructT>::fixConnectionSourceNodeIndexes(
    int64_t old_n_conn, uint *d_local_node_index) {
  int64_t n_new_conn = n_conn_ - old_n_conn; // number of new connections

  uint ib0 = (uint)(old_n_conn / conn_block_size_);    // first block index
  uint ib1 = (uint)((n_conn_ - 1) / conn_block_size_); // last block
  for (uint ib = ib0; ib <= ib1; ib++) {               // loop on blocks
    int64_t n_block_conn; // number of connections in a block
    int64_t i_conn0;      // index of first connection in a block
    if (ib1 == ib0) {     // all connections are in the same block
      i_conn0 = old_n_conn % conn_block_size_;
      n_block_conn = n_new_conn;
    } else if (ib == ib0) { // first block
      i_conn0 = old_n_conn % conn_block_size_;
      n_block_conn = conn_block_size_ - i_conn0;
    } else if (ib == ib1) { // last block
      i_conn0 = 0;
      n_block_conn = (n_conn_ - 1) % conn_block_size_ + 1;
    } else {
      i_conn0 = 0;
      n_block_conn = conn_block_size_;
    }

    if (n_block_conn > 0) {
      fixConnectionSourceNodeIndexesKernel<ConnKeyT>
          <<<(n_block_conn + 1023) / 1024, 1024>>>(
              conn_key_vect_[ib] + i_conn0, n_block_conn, d_local_node_index);
      CUDASYNC;
    }
  }
  return 0;
}

// Calibrate the maps
template <class ConnKeyT, class ConnStructT>
int ConnectionTemplate<ConnKeyT, ConnStructT>::remoteConnectionMapCalibrate(
    inode_t n_nodes) {
  PRINT_TIME;

  //  vector of pointers to local source node maps in device memory
  //  per target host hd_local_source_node_map[target_host]
  //  type std::vector<uint*>
  //  set its size and initialize to NULL
  hd_local_source_node_map_.resize(n_hosts_, NULL);
  // number of elements in each local source node map
  // h_n_local_source_node_map[target_host]
  // set its size and initialize to 0
  h_n_local_source_node_map_.resize(n_hosts_, 0);

  uint nhg = host_group_.size(); // number of local host groups
  for (uint group_local_id = 0; group_local_id < nhg; group_local_id++) {
    uint nh =
        host_group_[group_local_id].size(); // number of hosts in the group
    // vector of vectors of pointers to local image node maps in device memory
    // per local host group and per source host
    // hd_image_node_map[group_local_id][i_host] type std::vector< std::vector <
    // uint** > > set its size and initialize to NULL
    std::vector<uint **> inm(nh, NULL);
    hd_image_node_map_.push_back(inm);

    // number of elements in each remote-source-node->local-image-node map
    // h_n_remote_source_node_map[group_local_id][i_host]
    // set its size and initialize to 0
    std::vector<uint> n_nm(nh, 0);
    h_n_remote_source_node_map_.push_back(n_nm);
  }
  // loop on target hosts, skip self host
  for (int tg_host = 0; tg_host < n_hosts_; tg_host++) {
    if (tg_host != this_host_) {
      // get number of elements in each map from device memory
      uint n_node_map;
      gpuErrchk(cudaMemcpy(&n_node_map, &d_n_local_source_node_map_[tg_host],
                           sizeof(uint), cudaMemcpyDeviceToHost));
      // put it in h_n_local_source_node_map[tg_host]
      h_n_local_source_node_map_[tg_host] = n_node_map;
      // Allocate array of local source node map blocks
      // and copy their address from host to device
      hd_local_source_node_map_[tg_host] = NULL;
      uint n_blocks = h_local_source_node_map_[tg_host].size();
      if (n_blocks > 0) {
        CUDAMALLOCCTRL("&hd_local_source_node_map[tg_host]",
                       &hd_local_source_node_map_[tg_host],
                       n_blocks * sizeof(uint *));
        gpuErrchk(cudaMemcpy(hd_local_source_node_map_[tg_host],
                             &h_local_source_node_map_[tg_host][0],
                             n_blocks * sizeof(uint *),
                             cudaMemcpyHostToDevice));
      }
    }
  }

  PRINT_TIME;

  // allocate d_local_source_node_map and copy it from host to device
  CUDAMALLOCCTRL("&d_local_source_node_map", &d_local_source_node_map_,
                 n_hosts_ * sizeof(uint **));
  gpuErrchk(cudaMemcpy(d_local_source_node_map_, &hd_local_source_node_map_[0],
                       n_hosts_ * sizeof(uint **), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyToSymbol(local_source_node_map, &d_local_source_node_map_,
                               sizeof(uint ***)));

  hdd_image_node_map_.resize(nhg, NULL);

  bit_pack_nbits_.resize(nhg);
  bit_pack_nbits_this_host_.resize(nhg);

  for (uint group_local_id = 0; group_local_id < nhg;
       group_local_id++) { // loop on local host groups
    uint nh =
        host_group_[group_local_id].size(); // number of hosts in the group
    bit_pack_nbits_[group_local_id].resize(nh);
    for (uint gi_host = 0; gi_host < nh; gi_host++) { // loop on hosts
      int nbits = 0;
      if (group_local_id > 0) {
        uint n_src;
        if (host_group_source_node_sequence_flag_) {
          n_src =
              (host_group_source_node_max_[group_local_id][gi_host] >=
               host_group_source_node_min_[group_local_id][gi_host])
                  ? (host_group_source_node_max_[group_local_id][gi_host] -
                     host_group_source_node_min_[group_local_id][gi_host] + 1)
                  : 0;
        } else {
          n_src = host_group_source_node_[group_local_id][gi_host].size();
        }
        nbits = msb(n_src) + 1;
        bit_pack_nbits_[group_local_id][gi_host] = nbits;
      }
      int src_host = host_group_[group_local_id][gi_host];
      if (src_host == this_host_) {
        bit_pack_nbits_this_host_[group_local_id] = nbits;
      } else { // skip self host
        // get number of elements in each map from device memory
        uint n_node_map;
        gpuErrchk(cudaMemcpy(
            &n_node_map, &d_n_remote_source_node_map_[group_local_id][gi_host],
            sizeof(uint), cudaMemcpyDeviceToHost));
        // put it in h_n_remote_source_node_map[src_host]
        h_n_remote_source_node_map_[group_local_id][gi_host] = n_node_map;
        // Allocate array of local image node map blocks
        // and copy their address from host to device
        uint n_blocks = h_image_node_map_[group_local_id][gi_host].size();
        hd_image_node_map_[group_local_id][gi_host] = NULL;
        if (n_blocks > 0) {
          CUDAMALLOCCTRL("&hd_image_node_map_[group_local_id][gi_host]",
                         &hd_image_node_map_[group_local_id][gi_host],
                         n_blocks * sizeof(uint *));
          gpuErrchk(cudaMemcpy(hd_image_node_map_[group_local_id][gi_host],
                               &h_image_node_map_[group_local_id][gi_host][0],
                               n_blocks * sizeof(uint *),
                               cudaMemcpyHostToDevice));
        }
      }
    }
    PRINT_TIME;

    if (nh > 0) {
      CUDAMALLOCCTRL("&hdd_image_node_map_[group_local_id]",
                     &hdd_image_node_map_[group_local_id],
                     nh * sizeof(uint **));
      gpuErrchk(cudaMemcpy(hdd_image_node_map_[group_local_id],
                           &hd_image_node_map_[group_local_id][0],
                           nh * sizeof(uint **), cudaMemcpyHostToDevice));
    }
  }

  // allocate d_image_node_map and copy it from host to device
  CUDAMALLOCCTRL("&d_image_node_map_", &d_image_node_map_,
                 nhg * sizeof(uint ***));
  gpuErrchk(cudaMemcpy(d_image_node_map_, &hdd_image_node_map_[0],
                       nhg * sizeof(uint ***), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyToSymbol(local_image_node_map, &d_image_node_map_,
                               sizeof(uint ****)));

  PRINT_TIME;

  // uint n_nodes = GetNLocalNodes(); // number of nodes
  //  n_target_hosts[i_node] is the number of remote target hosts
  //  on which each local node
  //  has outgoing connections
  //  allocate d_n_target_hosts[n_nodes] and init to 0
  //  verbosePrint( std::string("allocate d_n_target_hosts n_nodes: ") +
  //  std::to_string(n_nodes) + "\n" );
  CUDAMALLOCCTRL("&d_n_target_hosts_", &d_n_target_hosts_,
                 n_nodes * sizeof(uint));
  // verbosePrint( std::string("d_n_target_hosts: ") +
  // std::to_string(d_n_target_hosts_) + "\n" );
  gpuErrchk(cudaMemset(d_n_target_hosts_, 0, n_nodes * sizeof(uint)));
  // allocate d_n_target_hosts_cumul[n_nodes+1]
  // representing the prefix scan (cumulative sum) of d_n_target_hosts
  CUDAMALLOCCTRL("&d_n_target_hosts_cumul_", &d_n_target_hosts_cumul_,
                 (n_nodes + 1) * sizeof(uint));

  PRINT_TIME;

  // For each local node, count the number of remote target hosts
  // on which it has outgoing connections, i.e. n_target_hosts[i_node]
  // Loop on target hosts
  for (int tg_host = 0; tg_host < n_hosts_; tg_host++) {
    if (tg_host != this_host_) {
      uint **d_node_map = hd_local_source_node_map_[tg_host];
      uint n_node_map = h_n_local_source_node_map_[tg_host];
      // Launch kernel that searches each node in the map
      // of local source nodes having outgoing connections to target host
      // if found, increase n_target_hosts[i_node]
      if (n_nodes > 0) {
        searchNodeIndexInMapKernel<<<(n_nodes + 1023) / 1024, 1024>>>(
            d_node_map, n_node_map, d_n_target_hosts_, n_nodes);
        CUDASYNC;
      }
    }
  }

  PRINT_TIME;

  //////////////////////////////////////////////////////////////////////
  // Evaluate exclusive sum of reverse connections per target node
  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  //<BEGIN-CLANG-TIDY-SKIP>//
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                d_n_target_hosts_, d_n_target_hosts_cumul_,
                                n_nodes + 1);
  //<END-CLANG-TIDY-SKIP>//

  // Allocate temporary storage
  CUDAMALLOCCTRL("&d_temp_storage", &d_temp_storage, temp_storage_bytes);
  // Run exclusive prefix sum
  //<BEGIN-CLANG-TIDY-SKIP>//
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                d_n_target_hosts_, d_n_target_hosts_cumul_,
                                n_nodes + 1);
  //<END-CLANG-TIDY-SKIP>//

  CUDAFREECTRL("d_temp_storage", d_temp_storage);
  // The last element is the sum of all elements of n_target_hosts
  uint n_target_hosts_sum;
  gpuErrchk(cudaMemcpy(&n_target_hosts_sum, &d_n_target_hosts_cumul_[n_nodes],
                       sizeof(uint), cudaMemcpyDeviceToHost));

  PRINT_TIME;

  if (n_target_hosts_sum > 0) {
    //////////////////////////////////////////////////////////////////////
    // allocate global array with remote target hosts of all nodes
    CUDAMALLOCCTRL("&d_target_host_array_", &d_target_host_array_,
                   n_target_hosts_sum * sizeof(uint));
    // allocate global array with remote target hosts map index
    CUDAMALLOCCTRL("&d_target_host_i_map_", &d_target_host_i_map_,
                   n_target_hosts_sum * sizeof(uint));
    // allocate array of pointers to the starting position in target_host array
    // of the target hosts for each node
  } else {
    d_target_host_array_ = nullptr;
    d_target_host_i_map_ = nullptr;
  }
  CUDAMALLOCCTRL("&d_node_target_hosts_", &d_node_target_hosts_,
                 n_nodes * sizeof(uint *));
  // allocate array of pointers to the starting position in target_host_i_map
  // of the target hosts map indexes for each node
  CUDAMALLOCCTRL("&d_node_target_host_i_map_", &d_node_target_host_i_map_,
                 n_nodes * sizeof(uint *));
  // Launch kernel to evaluate the pointers d_node_target_hosts
  // and d_node_target_host_i_map from the positions in target_host_array
  // given by  n_target_hosts_cumul

  PRINT_TIME;

  if (n_nodes > 0) {
    setTargetHostArrayNodePointersKernel<<<(n_nodes + 1023) / 1024, 1024>>>(
        d_target_host_array_, d_target_host_i_map_, d_n_target_hosts_cumul_,
        d_node_target_hosts_, d_node_target_host_i_map_, n_nodes);
    CUDASYNC;
  }

  PRINT_TIME;

  // reset to 0 d_n_target_hosts[n_nodes] to reuse it in the next kernel
  gpuErrchk(cudaMemset(d_n_target_hosts_, 0, n_nodes * sizeof(uint)));

  PRINT_TIME;

  // Loop on target hosts
  for (int tg_host = 0; tg_host < n_hosts_; tg_host++) {
    if (tg_host != this_host_) {
      uint **d_node_map = hd_local_source_node_map_[tg_host];
      uint n_node_map = h_n_local_source_node_map_[tg_host];
      // Launch kernel to fill the arrays target_host_array
      // and target_host_i_map using the node map
      if (n_nodes > 0) {
        fillTargetHostArrayFromMapKernel<<<(n_nodes + 1023) / 1024, 1024>>>(
            d_node_map, n_node_map, d_n_target_hosts_, d_node_target_hosts_,
            d_node_target_host_i_map_, n_nodes, tg_host);
        CUDASYNC;
      }
    }
  }

  PRINT_TIME;

  addOffsetToImageNodeMap(n_nodes);

  PRINT_TIME;

  uint src_node_max = 0;

  std::vector<std::unordered_set<int>> node_target_host_group_us;
  node_target_host_group_.resize(n_nodes);
  node_target_host_group_us.resize(n_nodes);

  for (uint group_local_id = 1; group_local_id < nhg; group_local_id++) {
    host_group_local_source_node_map_[group_local_id].resize(n_nodes);
    uint nh =
        host_group_[group_local_id].size(); // number of hosts in the group
    for (uint gi_host = 0; gi_host < nh; gi_host++) { // loop on hosts
      uint n_src;
      if (host_group_source_node_sequence_flag_) {
        n_src = (host_group_source_node_max_[group_local_id][gi_host] >=
                 host_group_source_node_min_[group_local_id][gi_host])
                    ? (host_group_source_node_max_[group_local_id][gi_host] -
                       host_group_source_node_min_[group_local_id][gi_host] + 1)
                    : 0;
      } else {
        n_src = host_group_source_node_[group_local_id][gi_host].size();

        // n_src_max = max(n_src_max, n_src);
        host_group_source_node_vect_[group_local_id][gi_host].resize(n_src);
        std::copy(
            host_group_source_node_[group_local_id][gi_host].begin(),
            host_group_source_node_[group_local_id][gi_host].end(),
            host_group_source_node_vect_[group_local_id][gi_host].begin());
        host_group_source_node_[group_local_id][gi_host].clear();
      }

      host_group_local_node_index_[group_local_id][gi_host].resize(n_src);
    }
  }

  PRINT_TIME;

  // copy remote-source-node-to-local-image-index maps from GPU to CPU memory
  if (!first_out_conn_in_device_) {
    for (int i_host = 0; i_host < n_hosts_; i_host++) { // loop on hosts
      // get number of elements in the map
      uint n_node_map;
      gpuErrchk(cudaMemcpy(&n_node_map, &d_n_remote_source_node_map_[0][i_host],
                           sizeof(uint), cudaMemcpyDeviceToHost));
      if (n_node_map > 0) {
        hc_remote_source_node_map_[0][i_host].resize(n_node_map);
        hc_image_node_map_[0][i_host].resize(n_node_map);
        // loop on remote-source-node-to-local-image-node map blocks
        uint n_map_blocks = h_remote_source_node_map_[0][i_host].size();
        for (uint ib = 0; ib < n_map_blocks; ib++) {
          uint n_elem;
          if (ib < n_map_blocks - 1) {
            n_elem = node_map_block_size_;
          } else {
            n_elem = (n_node_map - 1) % node_map_block_size_ + 1;
          }
          gpuErrchk(cudaMemcpy(
              &hc_remote_source_node_map_[0][i_host][ib * node_map_block_size_],
              h_remote_source_node_map_[0][i_host][ib], n_elem * sizeof(uint),
              cudaMemcpyDeviceToHost));
          gpuErrchk(cudaMemcpy(
              &hc_image_node_map_[0][i_host][ib * node_map_block_size_],
              h_image_node_map_[0][i_host][ib], n_elem * sizeof(uint),
              cudaMemcpyDeviceToHost));
        }
      }
    }
  }

  PRINT_TIME;

  std::vector<int64_t> tmp_node_map;
  // tmp_node_map.resize(src_node_max);

  for (uint group_local_id = 1; group_local_id < nhg; group_local_id++) {
    uint nh =
        host_group_[group_local_id].size(); // number of hosts in the group
    for (uint gi_host = 0; gi_host < nh; gi_host++) { // loop on hosts
      int src_host = host_group_[group_local_id][gi_host];
      ///*
      if (src_host != this_host_) { // skip self host
        // get number of elements in the map
        uint n_node_map;
        gpuErrchk(cudaMemcpy(
            &n_node_map, &d_n_remote_source_node_map_[group_local_id][gi_host],
            sizeof(uint), cudaMemcpyDeviceToHost));
        if (n_node_map > 0) {
          hc_remote_source_node_map_[group_local_id][gi_host].resize(
              n_node_map);
          hc_image_node_map_[group_local_id][gi_host].resize(n_node_map);
          // loop on remote-source-node-to-local-image-node map blocks
          uint n_map_blocks =
              h_remote_source_node_map_[group_local_id][gi_host].size();
          for (uint ib = 0; ib < n_map_blocks; ib++) {
            uint n_elem;
            if (ib < n_map_blocks - 1) {
              n_elem = node_map_block_size_;
            } else {
              n_elem = (n_node_map - 1) % node_map_block_size_ + 1;
            }
            gpuErrchk(cudaMemcpy(
                &hc_remote_source_node_map_[group_local_id][gi_host]
                                           [ib * node_map_block_size_],
                h_remote_source_node_map_[group_local_id][gi_host][ib],
                n_elem * sizeof(uint), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(&hc_image_node_map_[group_local_id][gi_host]
                                                    [ib * node_map_block_size_],
                                 h_image_node_map_[group_local_id][gi_host][ib],
                                 n_elem * sizeof(uint),
                                 cudaMemcpyDeviceToHost));
          }
          ///*
          bool resize_flag = false;
          for (uint i = 0; i < n_node_map; i++) {
            inode_t src_node =
                hc_remote_source_node_map_[group_local_id][gi_host][i];
            if (src_node > src_node_max) {
              resize_flag = true;
              src_node_max = src_node;
              // std::cerr << "Error. src_node: " << src_node << " greater than
              // n_src_max: " << n_src_max << "\n"; exit(0);
            }
          }
          if (resize_flag) {
            tmp_node_map.resize(src_node_max + 1);
          }
          std::fill(tmp_node_map.begin(), tmp_node_map.end(), -1);
          for (uint i = 0; i < n_node_map; i++) {
            inode_t src_node =
                hc_remote_source_node_map_[group_local_id][gi_host][i];
            tmp_node_map[src_node] =
                hc_image_node_map_[group_local_id][gi_host][i];
          }

          uint n_src;
          if (host_group_source_node_sequence_flag_) {
            n_src =
                (host_group_source_node_max_[group_local_id][gi_host] >=
                 host_group_source_node_min_[group_local_id][gi_host])
                    ? (host_group_source_node_max_[group_local_id][gi_host] -
                       host_group_source_node_min_[group_local_id][gi_host] + 1)
                    : 0;
          } else {
            n_src =
                host_group_source_node_vect_[group_local_id][gi_host].size();
          }
          for (uint i = 0; i < n_src; i++) {
            inode_t src_node;
            if (host_group_source_node_sequence_flag_) {
              src_node =
                  host_group_source_node_min_[group_local_id][gi_host] + i;
            } else {
              src_node =
                  host_group_source_node_vect_[group_local_id][gi_host][i];
            }
            int64_t pos;
            if (src_node > src_node_max) {
              pos = -1;
            } else {
              pos = tmp_node_map[src_node];
            }
            // if (pos<0) {
            //   throw ngpu_exception( "source node not found in host map" );
            // }
            host_group_local_node_index_[group_local_id][gi_host][i] = pos;
          }

        } else {
          std::fill(
              host_group_local_node_index_[group_local_id][gi_host].begin(),
              host_group_local_node_index_[group_local_id][gi_host].end(), -1);
        }
      } else { // only in the source, i.e. if src_host == this_host_
        uint n_src;
        if (host_group_source_node_sequence_flag_) {
          n_src =
              (host_group_source_node_max_[group_local_id][gi_host] >=
               host_group_source_node_min_[group_local_id][gi_host])
                  ? (host_group_source_node_max_[group_local_id][gi_host] -
                     host_group_source_node_min_[group_local_id][gi_host] + 1)
                  : 0;
        } else {
          n_src = host_group_source_node_vect_[group_local_id][gi_host].size();
        }
        for (uint i = 0; i < n_src; i++) {
          inode_t src_node;
          if (host_group_source_node_sequence_flag_) {
            src_node = host_group_source_node_min_[group_local_id][gi_host] + i;
          } else {
            src_node = host_group_source_node_vect_[group_local_id][gi_host][i];
          }
          host_group_local_source_node_map_[group_local_id][src_node] = i;
          std::pair<std::unordered_set<int>::iterator, bool> insert_it =
              node_target_host_group_us[src_node].insert(group_local_id);
          if (insert_it.second) {
            node_target_host_group_[src_node].push_back(group_local_id);
          }
        }
      }
    }
  }
  PRINT_TIME;

  if (delete_remote_node_map_ || delete_image_node_map_) {
    for (int i_host = 0; i_host < n_hosts_; i_host++) { // loop on hosts
      // loop on remote-source-node-to-local-image-node map blocks
      uint n_map_blocks = h_remote_source_node_map_[0][i_host].size();
      for (uint ib = 0; ib < n_map_blocks; ib++) {
        if (delete_remote_node_map_) {
          CUDAFREECTRL("d_remote_src_node_blk_pt",
                       h_remote_source_node_map_[0][i_host][ib]);
        }
        if (delete_image_node_map_) {
          CUDAFREECTRL("d_local_src_node_blk_pt",
                       h_image_node_map_[0][i_host][ib]);
        }
      }
    }

    for (uint group_local_id = 1; group_local_id < nhg; group_local_id++) {
      uint nh =
          host_group_[group_local_id].size(); // number of hosts in the group
      for (uint gi_host = 0; gi_host < nh; gi_host++) { // loop on hosts
        int src_host = host_group_[group_local_id][gi_host];
        if (src_host != this_host_) { // skip self host
          // loop on remote-source-node-to-local-image-node map blocks
          uint n_map_blocks =
              h_remote_source_node_map_[group_local_id][gi_host].size();
          for (uint ib = 0; ib < n_map_blocks; ib++) {
            if (delete_remote_node_map_) {
              CUDAFREECTRL(
                  "d_remote_src_node_blk_pt",
                  h_remote_source_node_map_[group_local_id][gi_host][ib]);
            }
            if (delete_image_node_map_) {
              CUDAFREECTRL("d_local_src_node_blk_pt",
                           h_image_node_map_[group_local_id][gi_host][ib]);
            }
          }
        }
      }
    }
  }

  return 0;
}

template <class ConnKeyT, class ConnStructT>
int ConnectionTemplate<ConnKeyT, ConnStructT>::remoteConnectionMapSave() {
  std::vector<std::vector<uint>> hc_remote_source_node_map;
  std::vector<std::vector<uint>> hc_image_node_map;

  hc_remote_source_node_map.resize(n_hosts_);
  hc_image_node_map.resize(n_hosts_);

  for (int src_host = 0; src_host < n_hosts_; src_host++) { // loop on hosts
    if (src_host != this_host_) {                           // skip self host
      // get number of elements in the map
      uint n_node_map;
      gpuErrchk(cudaMemcpy(&n_node_map,
                           &d_n_remote_source_node_map_[0][src_host],
                           sizeof(uint), cudaMemcpyDeviceToHost));

      if (n_node_map > 0) {
        hc_remote_source_node_map[src_host].resize(n_node_map);
        hc_image_node_map[src_host].resize(n_node_map);
        // loop on remote-source-node-to-local-image-node map blocks
        uint n_map_blocks = h_remote_source_node_map_[0][src_host].size();

        for (uint ib = 0; ib < n_map_blocks; ib++) {
          uint n_elem;
          if (ib < n_map_blocks - 1) {
            n_elem = node_map_block_size_;
          } else {
            n_elem = (n_node_map - 1) % node_map_block_size_ + 1;
          }
          gpuErrchk(cudaMemcpy(
              &hc_remote_source_node_map[src_host][ib * node_map_block_size_],
              h_remote_source_node_map_[0][src_host][ib], n_elem * sizeof(uint),
              cudaMemcpyDeviceToHost));
          gpuErrchk(cudaMemcpy(
              &hc_image_node_map[src_host][ib * node_map_block_size_],
              h_image_node_map_[0][src_host][ib], n_elem * sizeof(uint),
              cudaMemcpyDeviceToHost));
        }
      }

      std::string filename = std::string("map_remote_src_") +
                             std::to_string(this_host_) + "_" +
                             std::to_string(src_host) + ".dat";
      std::ofstream ofs;
      ofs.open(filename, std::ios::out);
      if (ofs.fail()) {
        std::cerr << "Cannot open output file\n";
        exit(-1);
      }
      ofs << n_node_map << "\n";
      for (uint i = 0; i < n_node_map; i++) {
        ofs << hc_remote_source_node_map[src_host][i] << "\t"
            << hc_image_node_map[src_host][i] << "\n";
      }
      ofs.close();
    }
  }

  std::vector<std::vector<uint>> hc_local_source_node_map;

  hc_local_source_node_map.resize(n_hosts_);

  for (int tg_host = 0; tg_host < n_hosts_; tg_host++) { // loop on hosts
    if (tg_host != this_host_) {                         // skip self host
      // get number of elements in the map
      uint n_node_map;
      gpuErrchk(cudaMemcpy(&n_node_map, &d_n_local_source_node_map_[tg_host],
                           sizeof(uint), cudaMemcpyDeviceToHost));

      if (n_node_map > 0) {
        hc_local_source_node_map[tg_host].resize(n_node_map);
        // loop on remote-source-node-to-local-image-node map blocks
        uint n_map_blocks = h_local_source_node_map_[tg_host].size();

        for (uint ib = 0; ib < n_map_blocks; ib++) {
          uint n_elem;
          if (ib < n_map_blocks - 1) {
            n_elem = node_map_block_size_;
          } else {
            n_elem = (n_node_map - 1) % node_map_block_size_ + 1;
          }
          gpuErrchk(cudaMemcpy(
              &hc_local_source_node_map[tg_host][ib * node_map_block_size_],
              h_local_source_node_map_[tg_host][ib], n_elem * sizeof(uint),
              cudaMemcpyDeviceToHost));
        }
      }

      std::string filename = std::string("map_local_src_") +
                             std::to_string(this_host_) + "_" +
                             std::to_string(tg_host) + ".dat";
      std::ofstream ofs;
      ofs.open(filename, std::ios::out);
      if (ofs.fail()) {
        std::cerr << "Cannot open output file\n";
        exit(-1);
      }
      ofs << n_node_map << "\n";
      for (uint i = 0; i < n_node_map; i++) {
        ofs << hc_local_source_node_map[tg_host][i] << "\n";
      }
      ofs.close();
    }
  }

  return 0;
}

template <class ConnKeyT, class ConnStructT>
int ConnectionTemplate<ConnKeyT, ConnStructT>::addOffsetToImageNodeMap(
    inode_t n_nodes) {
  uint i_image_node_0 = n_nodes;

  uint nhg = host_group_.size(); // number of local host groups
  for (uint group_local_id = 0; group_local_id < nhg; group_local_id++) {
    uint nh =
        host_group_[group_local_id].size(); // number of hosts in the group
    for (uint gi_host = 0; gi_host < nh; gi_host++) { // loop on hosts
      int src_host = host_group_[group_local_id][gi_host];
      if (src_host != this_host_) { // skip self host
        uint n_node_map = h_n_remote_source_node_map_[group_local_id][gi_host];
        if (n_node_map > 0) {
          addOffsetToImageNodeMapKernel<<<(n_node_map + 1023) / 1024, 1024>>>(
              group_local_id, gi_host, n_node_map, i_image_node_0);
          CUDASYNC;
        }
      }
    }
  }

  return 0;
}

// REMOTE CONNECT FUNCTION
template <class ConnKeyT, class ConnStructT>
template <class T1, class T2>
int ConnectionTemplate<ConnKeyT, ConnStructT>::_RemoteConnect(
    int source_host, T1 h_source, inode_t n_source, int target_host,
    T2 h_target, inode_t n_target, int i_host_group, ConnSpec &conn_spec,
    SynSpec &syn_spec) {
  double time_mark;
  if (first_connection_flag_ == true) {
    remoteConnectionMapInit();
    first_connection_flag_ = false;
  }

  if (source_host >= n_hosts_) {
    throw ngpu_exception("Source host index out of range in _RemoteConnect");
  }
  if (target_host >= n_hosts_) {
    throw ngpu_exception("Target host index out of range in _RemoteConnect");
  }
  if (this_host_ >= n_hosts_) {
    throw ngpu_exception("this_host index out of range in _RemoteConnect");
  }
  // semplificare i free. Forse si può spostare tutto in Connect?
  T1 d_source = copyNodeArrayToDevice(h_source, n_source);
  T2 d_target = copyNodeArrayToDevice(h_target, n_target);

  // Check if it is a local connection
  if (this_host_ == source_host && source_host == target_host) {
    int ret = _Connect<T1, T2>(d_source, n_source, d_target, n_target,
                               conn_spec, syn_spec);
    freeNodeArrayFromDevice(d_source);
    freeNodeArrayFromDevice(d_target);
    return ret;
  }

  // i_host_group is the global host-group index, given as optional argument to
  // the RemoteConnect command. Default value is either -1 or 0 (initialized as
  // kernel parameter). i_host_group = -1 for point-to-point MPI communication
  //              > 0 for other host groups
  //             ??? DESIGN DECISION = 0 for the world group, which includes all
  //             hosts (i.e. all MPI processes)
  int group_local_id = 0;
  int i_host = 0;
  if (i_host_group >= 0) { // not a point-to-point MPI communication
    group_local_id = host_group_local_id_[i_host_group];
    if (group_local_id >= 0) { // this host is in group
      // find the source host index in the host group
      auto it = std::find(host_group_[group_local_id].begin(),
                          host_group_[group_local_id].end(), source_host);
      if (it == host_group_[group_local_id].end()) {
        throw ngpu_exception("source host not found in host group");
      }
      i_host = it - host_group_[group_local_id].begin();

      time_mark = getRealTime();
      if (host_group_source_node_sequence_flag_) {
        inode_t inode_min;
        inode_t inode_max;
        getNodeIndexRange(h_source, n_source, inode_min, inode_max);
        host_group_source_node_min_[group_local_id][i_host] =
            min(host_group_source_node_min_[group_local_id][i_host], inode_min);
        host_group_source_node_max_[group_local_id][i_host] =
            max(host_group_source_node_max_[group_local_id][i_host], inode_max);
      } else {
        for (inode_t i = 0; i < n_source; i++) {
          inode_t i_source = hGetNodeIndex(h_source, i);
          host_group_source_node_[group_local_id][i_host].insert(i_source);
        }
      }
      InsertHostGroupSourceNode_time_ += (getRealTime() - time_mark);
    }
  }
  // if i_host_group<0, i.e. a point-to-point MPI communication is required
  // and this host is the source (but it is not a local connection) call
  // RemoteConnectTarget Check if source_host matches this_host
  else {
    // p2p_flag_ = true;
    p2p_host_conn_matrix_[source_host][target_host] = true;
    if (this_host_ == source_host) {
      time_mark = getRealTime();
      int ret =
          remoteConnectTarget<T1, T2>(target_host, d_source, n_source, d_target,
                                      n_target, conn_spec, syn_spec);
      RemoteConnectTarget_time_ += (getRealTime() - time_mark);
      freeNodeArrayFromDevice(d_source);
      freeNodeArrayFromDevice(d_target);

      return ret;
    }
  }
  // Check if target_host matches this_host
  if (this_host_ == target_host) {
    if (group_local_id < 0) {
      throw ngpu_exception("target host is not in host group");
    }

    time_mark = getRealTime();
    int ret = remoteConnectSource<T1, T2>(source_host, d_source, n_source,
                                          d_target, n_target, group_local_id,
                                          conn_spec, syn_spec);
    RemoteConnectSource_time_ += (getRealTime() - time_mark);

    freeNodeArrayFromDevice(d_source);
    freeNodeArrayFromDevice(d_target);
    return ret;
  }

  freeNodeArrayFromDevice(d_source);
  freeNodeArrayFromDevice(d_target);

  return 0;
}

template <class ConnKeyT, class ConnStructT>
int ConnectionTemplate<ConnKeyT, ConnStructT>::addOffsetToExternalNodeIds(
    uint n_local_nodes) {
  uint n_blocks = (n_conn_ - 1) / conn_block_size_ + 1;
  // uint i_image_node_0 = getNLocalNodes();
  uint i_image_node_0 = n_local_nodes;

  for (uint ib = 0; ib < n_blocks; ib++) {
    int64_t n_block_conn =
        conn_block_size_;     // number of connections in the block
    if (ib == n_blocks - 1) { // last block
      n_block_conn = (n_conn_ - 1) % conn_block_size_ + 1;
    }
    if (n_block_conn > 0) {
      addOffsetToExternalNodeIdsKernel<ConnKeyT, ConnStructT>
          <<<(n_block_conn + 1023) / 1024, 1024>>>(
              n_block_conn, conn_key_vect_[ib],
              (ConnStructT *)conn_struct_vect_[ib], i_image_node_0);
      CUDASYNC;
    }
  }

  return 0;
}

// REMOTE CONNECT FUNCTION for target_host matching this_host
template <class ConnKeyT, class ConnStructT>
template <class T1, class T2>
int ConnectionTemplate<ConnKeyT, ConnStructT>::remoteConnectSource(
    int source_host, T1 source, inode_t n_source, T2 target, inode_t n_target,
    int group_local_id, ConnSpec &conn_spec, SynSpec &syn_spec) {
  if (n_source <= 0 || n_target <= 0) {
    return 0;
  }

  // number of nodes actually used in new connections
  uint n_used_source_nodes;

  // only for testing
  uint *d_check_local_node_index =
      nullptr; // [n_source]; // only on target host

  // map clones used only in special runs for testing
  uint **h_check_node_map = nullptr;
  uint **h_check_image_node_map = nullptr;
  uint **d_check_node_map = nullptr;
  uint **d_check_image_node_map = nullptr;

  double time_mark;

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////

  // n_nodes will be the first index for new mapping of remote source nodes
  // to local image nodes
  // int image_node_map_i0 = GetNNode();
  uint image_node_map_i0 = n_image_nodes_;
  // syn_spec.port_ = syn_spec.port_ |
  //   (1 << (h_MaxPortSynNBits - max_syn_nbits_ - 1));
  syn_spec.syn_group_ = syn_spec.syn_group_ | (1 << max_syn_nbits_);

  time_mark = getRealTime();
  int64_t old_n_conn = n_conn_;
  // The connect command is performed on both source and target host using
  // the same initial seed and using as source node indexes the integers
  // from 0 to n_source_nodes - 1
  _Connect<inode_t, T2>(conn_random_generator_[source_host][this_host_], 0,
                        n_source, target, n_target, conn_spec, syn_spec, false);
  ConnectRemoteConnectSource_time_ += (getRealTime() - time_mark);
  if (n_conn_ == old_n_conn) {
    return 0;
  }

  bool use_source_node_flag = false;
  // check if the connection rule or number of connections is such that all
  // remote source nodes should actually be used
  if (!conn_spec.use_all_remote_source_nodes_) {
    use_source_node_flag = true;
    // on both the source and target hosts create a temporary array
    // of booleans having size equal to the number of source nodes
    CUDAREALLOCIFSMALLER("&d_source_node_flag_", &d_source_node_flag_,
                         n_source * sizeof(uint), n_source * sizeof(uint));
    gpuErrchk(cudaMemset(d_source_node_flag_, 0, n_source * sizeof(uint)));

    // printf("this_host: %d\tn_source: %d, n_conn_; %ld\told_n_conn: %ld\n",
    // this_host_, n_source, n_conn_, old_n_conn); flag source nodes used in at
    // least one new connection

    // Loop on all new connections and set source_node_flag[i_source]=true
    time_mark = getRealTime();
    setUsedSourceNodes(old_n_conn, d_source_node_flag_);
    SetUsedSourceNodes_time_ += (getRealTime() - time_mark);
    //, n_source ); // uncomment only for debugging

    // Count source nodes actually used in new connections
    // Allocate n_used_source_nodes and initialize it to 0
    CUDAREALLOCIFSMALLER("&d_n_used_source_nodes_", &d_n_used_source_nodes_,
                         sizeof(uint), 0);
    gpuErrchk(cudaMemset(d_n_used_source_nodes_, 0, sizeof(uint)));
    // Launch kernel to count used nodes
    if (n_source > 0) {
      time_mark = getRealTime();
      countUsedSourceNodeKernel<<<(n_source + 1023) / 1024, 1024>>>(
          n_source, d_n_used_source_nodes_, d_source_node_flag_);
      CUDASYNC;
      CountUsedSourceNodes_time_ += (getRealTime() - time_mark);
    }

    // copy result from GPU to CPU memory
    gpuErrchk(cudaMemcpy(&n_used_source_nodes, d_n_used_source_nodes_,
                         sizeof(uint), cudaMemcpyDeviceToHost));
    // Reset n_used_source_nodes to 0
    gpuErrchk(cudaMemset(d_n_used_source_nodes_, 0, sizeof(uint)));
  } else {
    n_used_source_nodes = n_source;
  }

  int gi_host;
  // if ( group_local_id <= 1 ) { // point-to-point communication (0)  NOT FOR
  // NOW and world group (1) include all hosts
  if (group_local_id ==
      0) { // point-to-point communication (0) include all hosts
    gi_host = source_host;
  } else {
    // find the source host index in the host group
    auto it = std::find(host_group_[group_local_id].begin(),
                        host_group_[group_local_id].end(), source_host);
    if (it == host_group_[group_local_id].end()) {
      throw ngpu_exception("source host not found in host group");
    }
    gi_host = it - host_group_[group_local_id].begin();
  }
  uint n_blocks = h_remote_source_node_map_[group_local_id][gi_host].size();
  // get current number of elements in the map
  uint n_node_map;
  gpuErrchk(cudaMemcpy(&n_node_map,
                       &d_n_remote_source_node_map_[group_local_id][gi_host],
                       sizeof(uint), cudaMemcpyDeviceToHost));

  if (n_blocks > 0) {
    // check for consistency between number of elements
    // and number of blocks in the map
    uint tmp_n_blocks = (n_node_map - 1) / node_map_block_size_ + 1;
    if (tmp_n_blocks != n_blocks) {
      throw ngpu_exception(
          std::string("Inconsistent number of elements ") +
          std::to_string(n_node_map) + " and number of blocks " +
          std::to_string(n_blocks) + " in remote_source_node_map\n" +
          std::string("group_local_id: ") + std::to_string(group_local_id) +
          "\n" + std::string("gi_host: ") + std::to_string(gi_host) + "\n");
    }
  }

  uint i0;
  uint i1;
  uint transl = 0;
  uint n_down_0 = 0;
  uint n_down_1 = 0;
  uint new_n_node_map = 0;
  uint h_n_node_to_map = 0;

  // The following section is used only if the source nodes are defined by a
  // sequence and if the connection rule prescribe that they are all used
  if (!check_node_maps_ && isSequence(source) &&
      conn_spec.use_all_remote_source_nodes_) {
    verbosePrint("Updating remote node maps\n");
    // if source nodes are defined by a sequence, find the first and last index
    // of the sequence
    inode_t i_source_0 = firstNodeIndex(source);
    inode_t i_source_1 = i_source_0 + n_source - 1;

    uint n_mapped = 0;

    if (n_node_map > 0) {
      time_mark = getRealTime();
      // Find number of elements < i_source_0 in the map
      n_down_0 = search_block_array_down<inode_t>(
          &h_remote_source_node_map_[group_local_id][gi_host][0], n_node_map,
          node_map_block_size_, i_source_0);
      // Find number of elements < i_source_1 + 1 in the map
      n_down_1 = search_block_array_down<inode_t>(
          &h_remote_source_node_map_[group_local_id][gi_host][0], n_node_map,
          node_map_block_size_, i_source_1 + 1, n_down_0);
      // Compute number of source nodes already mapped
      n_mapped = n_down_1 - n_down_0;
      SearchSourceNodesRangeInMap_time_ += (getRealTime() - time_mark);
    }
    // Compute number of source nodes that must be mapped
    h_n_node_to_map = n_used_source_nodes - n_mapped;

    new_n_node_map = n_node_map + h_n_node_to_map;

    if (new_n_node_map > 0) {
      uint new_n_blocks = (new_n_node_map - 1) / node_map_block_size_ + 1;

      // if new blocks are required for the map, allocate them
      if (new_n_blocks != n_blocks) {
        // Allocate GPU memory for new remote-source-node-map blocks
        time_mark = getRealTime();
        allocRemoteSourceNodeMapBlocks(
            h_remote_source_node_map_[group_local_id][gi_host],
            h_image_node_map_[group_local_id][gi_host], new_n_blocks);
        AllocRemoteSourceNodeMapBlocks_time_ += (getRealTime() - time_mark);
      }

      // allocate d_node_map and get it from host
      CUDAREALLOCIFSMALLER("&d_node_map_", &d_node_map_,
                           new_n_blocks * sizeof(uint *),
                           new_n_blocks * sizeof(uint *));
      gpuErrchk(cudaMemcpy(
          d_node_map_, &h_remote_source_node_map_[group_local_id][gi_host][0],
          new_n_blocks * sizeof(uint *), cudaMemcpyHostToDevice));
      // allocate d_check_image_node_map and get it from host

      CUDAREALLOCIFSMALLER("&d_image_node_map_tmp_", &d_image_node_map_tmp_,
                           new_n_blocks * sizeof(uint *),
                           new_n_blocks * sizeof(uint *));
      gpuErrchk(cudaMemcpy(
          d_image_node_map_tmp_, &h_image_node_map_[group_local_id][gi_host][0],
          new_n_blocks * sizeof(uint *), cudaMemcpyHostToDevice));
    }

    if (n_node_map > 0 && h_n_node_to_map > 0) {
      uint aux_size = node_map_block_size_;
      i1 = n_node_map - 1;
      i0 = n_down_1;
      transl = h_n_node_to_map;

      time_mark = getRealTime();
      CUDAREALLOCIFSMALLER("&d_ru_storage_", &d_ru_storage_,
                           aux_size * sizeof(uint), aux_size * sizeof(uint));

      // printf("n_node_map: %d, new_n_node_map: %d, transl: %d,
      // node_map_block_size: %d, aux_size: %d, i0: %d, i1: %d\n",
      //      n_node_map, new_n_node_map, transl, node_map_block_size_,
      //      aux_size, i0, i1);

      int64_t i_right = i1;
      while (i_right >= i0) {
        // check that i_right + 1 - aux_size > i0 (considering that i_right is
        // unsigned); if yes, set i_left = i_right + 1 - aux_size otherwise set
        // i_left = i0
        int64_t i_left = max(i_right + 1 - aux_size, (int64_t)i0);
        int64_t n_elem = i_right - i_left + 1;
        // printf("i_left: %ld, i_right: %ld, n_elem: %ld\n", i_left, i_right,
        // n_elem);
        if (n_elem > 0) {
          copyBlockArrayToArrayKernel<<<(n_elem + 1023) / 1024, 1024>>>(
              (uint *)d_ru_storage_, d_node_map_, node_map_block_size_, i_left,
              n_elem);
          CUDASYNC;
          copyArrayToBlockArrayKernel<<<(n_elem + 1023) / 1024, 1024>>>(
              d_node_map_, (uint *)d_ru_storage_, node_map_block_size_,
              i_left + transl, n_elem);
          CUDASYNC;
          copyBlockArrayToArrayKernel<<<(n_elem + 1023) / 1024, 1024>>>(
              (uint *)d_ru_storage_, d_image_node_map_tmp_,
              node_map_block_size_, i_left, n_elem);
          CUDASYNC;
          copyArrayToBlockArrayKernel<<<(n_elem + 1023) / 1024, 1024>>>(
              d_image_node_map_tmp_, (uint *)d_ru_storage_,
              node_map_block_size_, i_left + transl, n_elem);
          CUDASYNC;
        }
        i_right -= aux_size;
      }
      TranslateSourceNodeMap_time_ += (getRealTime() - time_mark);
    }

    if (new_n_node_map > 0) {
      time_mark = getRealTime();
      // Allocate the index of the nodes to be mapped and initialize it to 0
      CUDAREALLOCIFSMALLER("&d_i_node_to_map_", &d_i_node_to_map_, sizeof(uint),
                           0);
      gpuErrchk(cudaMemset(d_i_node_to_map_, 0, sizeof(uint)));

      // on the target hosts create a temporary array of integers having size
      // equal to the number of source nodes
      CUDAREALLOCIFSMALLER("&d_local_node_index_", &d_local_node_index_,
                           n_source * sizeof(uint), n_source * sizeof(uint));

      // Allocate boolean array for flagging remote source nodes already mapped
      // and initialize all elements to 0 (false)
      CUDAREALLOCIFSMALLER("&d_node_mapped_", &d_node_mapped_,
                           n_source * sizeof(bool), n_source * sizeof(bool));
      gpuErrchk(cudaMemset(d_node_mapped_, 0, n_source * sizeof(bool)));

      // Launch Kernel that extracts local image index from already mapped
      // remote source nodes and match it to the nodes in a sequence
      if (n_mapped > 0) {
        extractLocalImageIndexOfMappedSourceNodes<<<(n_mapped + 1023) / 1024,
                                                    1024>>>(
            d_node_map_, n_down_0, n_mapped, i_source_0, d_node_mapped_, true,
            d_image_node_map_tmp_, d_local_node_index_);

        CUDASYNC;
      }

      if (n_source > 0) {
        mapRemoteSourceNodesToLocalImagesKernel<<<(n_source + 1023) / 1024,
                                                  1024>>>(
            d_node_map_, d_node_mapped_, n_source, n_down_0, i_source_0, true,
            d_image_node_map_tmp_, image_node_map_i0, d_i_node_to_map_,
            d_local_node_index_);
        CUDASYNC;
      }

      // CUDAFREECTRL( "d_i_node_to_map", d_i_node_to_map );
      // d_i_node_to_map = nullptr;
      // CUDAFREECTRL( "d_local_node_index", d_local_node_index );

      MapSourceNodeSequence_time_ += (getRealTime() - time_mark);

      // update number of elements in remote source node map
      n_node_map = new_n_node_map;
      gpuErrchk(
          cudaMemcpy(&d_n_remote_source_node_map_[group_local_id][gi_host],
                     &n_node_map, sizeof(uint), cudaMemcpyHostToDevice));

      // check for consistency between number of elements
      // and number of blocks in the map
      n_blocks = h_remote_source_node_map_[group_local_id][gi_host].size();
      uint tmp_n_blocks = (n_node_map - 1) / node_map_block_size_ + 1;
      if (tmp_n_blocks != n_blocks) {
        throw ngpu_exception(
            std::string("Inconsistent number of elements ") +
            std::to_string(n_node_map) + " and number of blocks " +
            std::to_string(n_blocks) + " in remote_source_node_map");
      }
    }
    verbosePrint("Finished udating remote node maps\n");
  } else {
    // Allocate arrays of size n_used_source_nodes
    time_mark = getRealTime();

    CUDAREALLOCIFSMALLER(
        "&d_unsorted_source_node_index_", &d_unsorted_source_node_index_,
        n_used_source_nodes * sizeof(uint), n_used_source_nodes * sizeof(uint));
    CUDAREALLOCIFSMALLER(
        "&d_sorted_source_node_index_", &d_sorted_source_node_index_,
        n_used_source_nodes * sizeof(uint), n_used_source_nodes * sizeof(uint));

    CUDAREALLOCIFSMALLER("&d_i_unsorted_source_arr_", &d_i_unsorted_source_arr_,
                         n_used_source_nodes * sizeof(uint),
                         n_used_source_nodes * sizeof(uint));
    CUDAREALLOCIFSMALLER("&d_i_sorted_source_arr_", &d_i_sorted_source_arr_,
                         n_used_source_nodes * sizeof(uint),
                         n_used_source_nodes * sizeof(uint));

    AllocUsedSourceNodes_time_ += (getRealTime() - time_mark);

    // Fill the arrays of nodes actually used by new connections
    // Launch kernel to fill the arrays
    if (n_source > 0) {
      time_mark = getRealTime();
      getUsedSourceNodeIndexKernel<<<(n_source + 1023) / 1024, 1024>>>(
          source, n_source, d_n_used_source_nodes_, use_source_node_flag, true,
          d_source_node_flag_, d_unsorted_source_node_index_,
          d_i_unsorted_source_arr_);
      CUDASYNC;
      GetUsedSourceNodeIndex_time_ += (getRealTime() - time_mark);
    }

    // Sort the arrays using unsorted_source_node_index as key
    // and i_source as value -> sorted_source_node_index
    time_mark = getRealTime();
    // Determine temporary storage requirements for RadixSort
    size_t sort_storage_bytes = 0;
    //<BEGIN-CLANG-TIDY-SKIP>//
    cub::DeviceRadixSort::SortPairs(
        nullptr, // d_ru_storage_,
        sort_storage_bytes, d_unsorted_source_node_index_,
        d_sorted_source_node_index_, d_i_unsorted_source_arr_,
        d_i_sorted_source_arr_, n_used_source_nodes);
    //<END-CLANG-TIDY-SKIP>//

    // Allocate temporary storage
    CUDAREALLOCIFSMALLER("&d_ru_storage_", &d_ru_storage_, sort_storage_bytes,
                         sort_storage_bytes);

    // Run sorting operation
    //<BEGIN-CLANG-TIDY-SKIP>//
    cub::DeviceRadixSort::SortPairs(
        d_ru_storage_, sort_storage_bytes, d_unsorted_source_node_index_,
        d_sorted_source_node_index_, d_i_unsorted_source_arr_,
        d_i_sorted_source_arr_, n_used_source_nodes);
    //<END-CLANG-TIDY-SKIP>//
    SortUsedSourceNodeIndex_time_ += (getRealTime() - time_mark);
    //

    DBGCUDASYNC;

    if (n_blocks > 0) {
      CUDAREALLOCIFSMALLER("&d_node_map_", &d_node_map_,
                           n_blocks * sizeof(uint *),
                           n_blocks * sizeof(uint *));
      gpuErrchk(cudaMemcpy(
          d_node_map_, &h_remote_source_node_map_[group_local_id][gi_host][0],
          n_blocks * sizeof(uint *), cudaMemcpyHostToDevice));

      // allocate d_image_node_map and get it from host
      CUDAREALLOCIFSMALLER("&d_image_node_map_tmp_", &d_image_node_map_tmp_,
                           n_blocks * sizeof(uint *),
                           n_blocks * sizeof(uint *));
      gpuErrchk(cudaMemcpy(d_image_node_map_tmp_,
                           &h_image_node_map_[group_local_id][gi_host][0],
                           n_blocks * sizeof(uint *), cudaMemcpyHostToDevice));
    }

    time_mark = getRealTime();
    // Allocate boolean array for flagging remote source nodes not yet mapped
    // and initialize all elements to 0 (false)
    CUDAREALLOCIFSMALLER("&d_node_to_map_", &d_node_to_map_,
                         n_used_source_nodes * sizeof(bool),
                         n_used_source_nodes * sizeof(bool));
    gpuErrchk(
        cudaMemset(d_node_to_map_, 0, n_used_source_nodes * sizeof(bool)));

    // Allocate number of nodes to be mapped and initialize it to 0
    CUDAREALLOCIFSMALLER("&d_n_node_to_map_", &d_n_node_to_map_, sizeof(uint),
                         0);
    gpuErrchk(cudaMemset(d_n_node_to_map_, 0, sizeof(uint)));
    // Allocate array of indexes of already mapped nodes local image indexes
    CUDAREALLOCIFSMALLER(
        "&d_mapped_local_node_index_", &d_mapped_local_node_index_,
        n_used_source_nodes * sizeof(uint), n_used_source_nodes * sizeof(uint));
    AllocNodeToMap_time_ += (getRealTime() - time_mark);

    // launch kernel that searches remote source nodes indexes not in the map,
    // flags the nodes not yet mapped and counts them
    if (n_used_source_nodes > 0) {
      time_mark = getRealTime();
      searchNodeIndexNotInMapKernel<<<(n_used_source_nodes + 1023) / 1024,
                                      1024>>>(
          d_node_map_, n_node_map, d_sorted_source_node_index_, d_node_to_map_,
          d_n_node_to_map_, n_used_source_nodes, true, d_image_node_map_tmp_,
          d_mapped_local_node_index_);
      CUDASYNC;
      SearchNodeIndexNotInMap_time_ += (getRealTime() - time_mark);
    }

    gpuErrchk(cudaMemcpy(&h_n_node_to_map, d_n_node_to_map_, sizeof(uint),
                         cudaMemcpyDeviceToHost));

    // The following section is used only in special runs for checking maps
    if (check_node_maps_ && isSequence(source) &&
        conn_spec.use_all_remote_source_nodes_) {
      verbosePrint("Preparing check of remote node maps\n");
      // if source nodes are defined by a sequence, find the first and last
      // index of the sequence
      inode_t i_source_0 = firstNodeIndex(source);
      verbosePrint(std::string(" i_source_0: ") + std::to_string(i_source_0) +
                   "\n");
      inode_t i_source_1 = i_source_0 + n_source - 1;
      verbosePrint(std::string(" i_source_1: ") + std::to_string(i_source_1) +
                   "\n");

      verbosePrint(std::string(" n_node_map: ") + std::to_string(n_node_map) +
                   "\n");

      uint check_n_mapped = 0;

      if (n_node_map > 0) {
        // Find number of elements < i_source_0 in the map
        n_down_0 = search_block_array_down<inode_t>(
            &h_remote_source_node_map_[group_local_id][gi_host][0], n_node_map,
            node_map_block_size_, i_source_0);
        verbosePrint(std::string(" n_down_0: ") + std::to_string(n_down_0) +
                     "\n");

        // Find number of elements < i_source_1 + 1 in the map
        n_down_1 = search_block_array_down<inode_t>(
            &h_remote_source_node_map_[group_local_id][gi_host][0], n_node_map,
            node_map_block_size_, i_source_1 + 1, n_down_0);
        verbosePrint(std::string(" n_down_1: ") + std::to_string(n_down_1) +
                     "\n");
        // Compute number of source nodes already mapped
        check_n_mapped = n_down_1 - n_down_0;
      }
      verbosePrint(std::string(" check n_mapped: ") +
                   std::to_string(check_n_mapped) + "\n");
      // Compute number of source nodes that must be mapped
      uint check_to_be_mapped = n_used_source_nodes - check_n_mapped;
      verbosePrint(std::string(" check to_be_mapped: ") +
                   std::to_string(check_to_be_mapped) + "\n");
      verbosePrint(std::string(" to_be_mapped: ") +
                   std::to_string(h_n_node_to_map) + "\n");
      if (check_to_be_mapped != h_n_node_to_map) {
        throw ngpu_exception("aaa Error in computing n. of nodes to be mapped");
      }

      // if (n_node_map > 0 && check_to_be_mapped > 0) {
      // uint aux_size = node_map_block_size_;
      // i1 = n_node_map - 1;
      // i0 = n_down_1;
      // transl = check_to_be_mapped;

      new_n_node_map = n_node_map + transl;
      // printf("%d new_n_node_map: %d, n_node_map: %d, transl: %d\n",
      // this_host_, new_n_node_map, n_node_map, transl);
      if (new_n_node_map > 0) {
        int new_n_blocks =
            (int)(((int64_t)new_n_node_map - 1) / node_map_block_size_ + 1);
        // printf("%d new_n_blocks: %d, n_blocks: %d\n", this_host_,
        // new_n_blocks, n_blocks);

        h_check_node_map = new uint *[new_n_blocks];
        h_check_image_node_map = new uint *[new_n_blocks];
        for (uint ib = 0; ib < (uint)new_n_blocks; ib++) {
          CUDAMALLOCCTRL("h_check_node_map", &h_check_node_map[ib],
                         node_map_block_size_ * sizeof(uint));
          CUDAMALLOCCTRL("h_check_image_node_map", &h_check_image_node_map[ib],
                         node_map_block_size_ * sizeof(uint));
          if (ib < n_blocks) {
            gpuErrchk(cudaMemcpy(
                h_check_node_map[ib],
                h_remote_source_node_map_[group_local_id][gi_host][ib],
                node_map_block_size_ * sizeof(uint), cudaMemcpyDeviceToDevice));
            gpuErrchk(cudaMemcpy(h_check_image_node_map[ib],
                                 h_image_node_map_[group_local_id][gi_host][ib],
                                 node_map_block_size_ * sizeof(uint),
                                 cudaMemcpyDeviceToDevice));
          }
        }
        // allocate d_check_node_map and get it from host
        CUDAMALLOCCTRL("&d_check_node_map", &d_check_node_map,
                       new_n_blocks * sizeof(uint *));
        gpuErrchk(cudaMemcpy(d_check_node_map, h_check_node_map,
                             new_n_blocks * sizeof(uint *),
                             cudaMemcpyHostToDevice));
        // allocate d_check_image_node_map and get it from host

        CUDAMALLOCCTRL("&d_check_image_node_map", &d_check_image_node_map,
                       new_n_blocks * sizeof(uint *));
        gpuErrchk(cudaMemcpy(d_check_image_node_map, h_check_image_node_map,
                             new_n_blocks * sizeof(uint *),
                             cudaMemcpyHostToDevice));
      }

      if (n_node_map > 0 && check_to_be_mapped > 0) {
        uint aux_size = node_map_block_size_;
        i1 = n_node_map - 1;
        i0 = n_down_1;
        transl = check_to_be_mapped;

        CUDAREALLOCIFSMALLER("&d_ru_storage_", &d_ru_storage_,
                             aux_size * sizeof(uint), aux_size * sizeof(uint));

        // printf("n_node_map: %d, new_n_node_map: %d, transl: %d,
        // node_map_block_size: %d, aux_size: %d, i0: %d, i1: %d\n",
        //      n_node_map, new_n_node_map, transl, node_map_block_size_,
        //      aux_size, i0, i1);

        int64_t i_right = i1;
        while (i_right >= i0) {
          // check that i_right + 1 - aux_size > i0 (considering that i_right is
          // unsigned); if yes, set i_left = i_right + 1 - aux_size otherwise
          // set i_left = i0
          int64_t i_left = max(i_right + 1 - aux_size, (int64_t)i0);
          int64_t n_elem = i_right - i_left + 1;
          // printf("i_left: %ld, i_right: %ld, n_elem: %ld\n", i_left, i_right,
          // n_elem);
          if (n_elem > 0) {
            copyBlockArrayToArrayKernel<<<(n_elem + 1023) / 1024, 1024>>>(
                (uint *)d_ru_storage_, d_check_node_map, node_map_block_size_,
                i_left, n_elem);
            CUDASYNC;
            copyArrayToBlockArrayKernel<<<(n_elem + 1023) / 1024, 1024>>>(
                d_check_node_map, (uint *)d_ru_storage_, node_map_block_size_,
                i_left + transl, n_elem);
            CUDASYNC;
            copyBlockArrayToArrayKernel<<<(n_elem + 1023) / 1024, 1024>>>(
                (uint *)d_ru_storage_, d_check_image_node_map,
                node_map_block_size_, i_left, n_elem);
            CUDASYNC;
            copyArrayToBlockArrayKernel<<<(n_elem + 1023) / 1024, 1024>>>(
                d_check_image_node_map, (uint *)d_ru_storage_,
                node_map_block_size_, i_left + transl, n_elem);
            CUDASYNC;
          }
          i_right -= aux_size;
        }
      }

      if (new_n_node_map > 0) {
        // Allocate the index of the nodes to be mapped and initialize it to 0
        CUDAREALLOCIFSMALLER("&d_i_node_to_map_", &d_i_node_to_map_,
                             sizeof(uint), 0);
        gpuErrchk(cudaMemset(d_i_node_to_map_, 0, sizeof(uint)));

        // on the target hosts create a temporary array of integers having size
        // equal to the number of source nodes
        CUDAREALLOCIFSMALLER("&d_local_node_index_", &d_local_node_index_,
                             n_source * sizeof(uint), n_source * sizeof(uint));

        // Allocate boolean array for flagging remote source nodes already
        // mapped and initialize all elements to 0 (false)
        CUDAREALLOCIFSMALLER("&d_node_mapped_", &d_node_mapped_,
                             n_source * sizeof(bool), n_source * sizeof(bool));
        gpuErrchk(cudaMemset(d_node_mapped_, 0, n_source * sizeof(bool)));

        // Launch Kernel that extracts local image index from already mapped
        // remote source nodes and match it to the nodes in a sequence
        if (check_n_mapped > 0) {
          extractLocalImageIndexOfMappedSourceNodes<<<
              (check_n_mapped + 1023) / 1024, 1024>>>(
              d_check_node_map, n_down_0, check_n_mapped, i_source_0,
              d_node_mapped_, true, d_check_image_node_map,
              d_local_node_index_);

          CUDASYNC;
        }

        if (n_source > 0) {
          mapRemoteSourceNodesToLocalImagesKernel<<<(n_source + 1023) / 1024,
                                                    1024>>>(
              d_check_node_map, d_node_mapped_, n_source, n_down_0, i_source_0,
              true, d_check_image_node_map, image_node_map_i0, d_i_node_to_map_,
              d_local_node_index_);
          CUDASYNC;
        }

        // CUDAFREECTRL( "d_i_node_to_map_", d_i_node_to_map_ );
        // d_i_node_to_map_ = nullptr;
        // CUDAFREECTRL( "d_local_node_index_", d_local_node_index_ );
        // d_local_node_index__ = nullptr;
      }
      verbosePrint("Finished preparing check of remote node maps\n");
    }

    ////////////////////////////////////////////

    uint new_n_blocks = 0;
    if (h_n_node_to_map > 0) {
      // Check if new blocks are required for the map
      new_n_blocks =
          (n_node_map + h_n_node_to_map - 1) / node_map_block_size_ + 1;

      // if new blocks are required for the map, allocate them
      if (new_n_blocks != n_blocks) {
        // Allocate GPU memory for new remote-source-node-map blocks
        time_mark = getRealTime();
        allocRemoteSourceNodeMapBlocks(
            h_remote_source_node_map_[group_local_id][gi_host],
            h_image_node_map_[group_local_id][gi_host], new_n_blocks);
        AllocRemoteSourceNodeMapBlocks_time_ += (getRealTime() - time_mark);
        // free d_node_map
        // if ( d_node_map != nullptr )
        //  {
        //    CUDAFREECTRL( "d_node_map", d_node_map );
        //    d_node_map = nullptr;
        //  }
        // update number of blocks in the map
        n_blocks = new_n_blocks;

        // reallocate d_node_map and get it from host
        CUDAREALLOCIFSMALLER("&d_node_map_", &d_node_map_,
                             n_blocks * sizeof(uint *),
                             n_blocks * sizeof(uint *));
        gpuErrchk(cudaMemcpy(
            d_node_map_, &h_remote_source_node_map_[group_local_id][gi_host][0],
            n_blocks * sizeof(uint *), cudaMemcpyHostToDevice));
      }
    }
    if (n_blocks > 0) {
      // if (d_image_node_map != nullptr) {
      // CUDAFREECTRL( "d_image_node_map", d_image_node_map );
      // d_image_node_map = nullptr;
      // }
      //  allocate d_image_node_map and get it from host
      CUDAREALLOCIFSMALLER("&d_image_node_map_tmp_", &d_image_node_map_tmp_,
                           n_blocks * sizeof(uint *),
                           n_blocks * sizeof(uint *));
      gpuErrchk(cudaMemcpy(d_image_node_map_tmp_,
                           &h_image_node_map_[group_local_id][gi_host][0],
                           n_blocks * sizeof(uint *), cudaMemcpyHostToDevice));

      // Map the not-yet-mapped source nodes using a kernel
      // similar to the one used for counting
      // In the target host unmapped remote source nodes must be mapped
      // to local nodes from n_nodes to n_nodes + n_node_to_map

      // Allocate the index of the nodes to be mapped and initialize it to 0
      CUDAREALLOCIFSMALLER("&d_i_node_to_map_", &d_i_node_to_map_, sizeof(uint),
                           0);
      gpuErrchk(cudaMemset(d_i_node_to_map_, 0, sizeof(uint)));

      // on the target hosts create a temporary array of integers having size
      // equal to the number of source nodes
      CUDAREALLOCIFSMALLER("&d_local_node_index_", &d_local_node_index_,
                           n_source * sizeof(uint), n_source * sizeof(uint));

      // launch kernel that checks if nodes are already in map
      // if not insert them in the map
      // In the target host, put in the map the pair:
      // (source_node_index, image_node_map_i0 + i_node_to_map)

      if (n_used_source_nodes > 0) {
        time_mark = getRealTime();
        insertNodesInMapKernel<<<(n_used_source_nodes + 1023) / 1024, 1024>>>(
            d_node_map_, n_node_map, d_sorted_source_node_index_,
            d_node_to_map_, d_i_node_to_map_, n_used_source_nodes, true,
            d_image_node_map_tmp_, image_node_map_i0, d_i_sorted_source_arr_,
            d_local_node_index_, d_mapped_local_node_index_);
        CUDASYNC;
        InsertNodesInMap_time_ += (getRealTime() - time_mark);
      }
    }

    // update number of elements in remote source node map
    n_node_map += h_n_node_to_map;
    gpuErrchk(cudaMemcpy(&d_n_remote_source_node_map_[group_local_id][gi_host],
                         &n_node_map, sizeof(uint), cudaMemcpyHostToDevice));

    // check for consistency between number of elements
    // and number of blocks in the map
    uint tmp_n_blocks = (n_node_map - 1) / node_map_block_size_ + 1;
    if (tmp_n_blocks != n_blocks) {
      throw ngpu_exception(std::string("Inconsistent number of elements ") +
                           std::to_string(n_node_map) +
                           " and number of blocks " + std::to_string(n_blocks) +
                           " in remote_source_node_map");
    }

    if (h_n_node_to_map > 0) {
      // Sort the WHOLE key-value pair map source_node_map, image_node_map
      // using block sort algorithm copass_sort
      // typical usage:
      // copass_sort::sort<uint, value_struct>(key_subarray, value_subarray, n,
      //				       aux_size, d_storage,
      // storage_bytes);
      // Determine temporary storage requirements for copass_sort
      time_mark = getRealTime();
      int64_t storage_bytes = 0;
      // if (d_storage != nullptr) {
      // CUDAFREECTRL( "d_storage", d_storage );
      // d_storage = nullptr;
      // }

      copass_sort::sort<uint, uint>(
          &h_remote_source_node_map_[group_local_id][gi_host][0],
          &h_image_node_map_[group_local_id][gi_host][0], n_node_map,
          node_map_block_size_,
          nullptr, // d_ru_storage_,
          storage_bytes, 0);

      // Allocate temporary storage
      CUDAREALLOCIFSMALLER("&d_ru_storage_", &d_ru_storage_, storage_bytes,
                           storage_bytes);

      // Run sorting operation
      copass_sort::sort<uint, uint>(
          &h_remote_source_node_map_[group_local_id][gi_host][0],
          &h_image_node_map_[group_local_id][gi_host][0], n_node_map,
          node_map_block_size_, d_ru_storage_, storage_bytes, 0);
      // CUDAFREECTRL( "d_storage", d_storage );
      // d_storage = nullptr;

      SortSourceImageNodeMap_time_ += (getRealTime() - time_mark);
    }
  }

  // The following block is used only for testing
  if (check_node_maps_) {
    verbosePrint("Started checking extraction of local node indexees from "
                 "remote node maps\n");
    CUDAMALLOCCTRL("&d_check_local_node_index", &d_check_local_node_index,
                   n_source * sizeof(uint));
    gpuErrchk(cudaMemset(d_check_local_node_index, 0, n_source * sizeof(uint)));

    // Launch kernel that searches source node indexes in the map
    // and set corresponding values of local_node_index
    if (n_source > 0) {
      time_mark = getRealTime();
      setLocalNodeIndexKernel<<<(n_source + 1023) / 1024, 1024>>>(
          source, n_source, use_source_node_flag, d_source_node_flag_,
          d_node_map_, d_image_node_map_tmp_, n_node_map,
          d_check_local_node_index);
      CUDASYNC;
      SetLocalNodeIndex_time_ += (getRealTime() - time_mark);
    }

    uint *h_check_local_node_index = new uint[n_source];
    uint *h_local_node_index = new uint[n_source];
    gpuErrchk(cudaMemcpy(h_check_local_node_index, d_check_local_node_index,
                         n_source * sizeof(uint), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_local_node_index, d_local_node_index_,
                         n_source * sizeof(uint), cudaMemcpyDeviceToHost));

    uint *h_source_node_flag = nullptr;
    if (!conn_spec.use_all_remote_source_nodes_) {
      h_source_node_flag = new uint[n_source];
      gpuErrchk(cudaMemcpy(h_source_node_flag, d_source_node_flag_,
                           n_source * sizeof(uint), cudaMemcpyDeviceToHost));
    }

    bool err = false;
    for (uint i = 0; i < n_source; i++) {
      if (conn_spec.use_all_remote_source_nodes_ ||
          h_source_node_flag[i] != 0) {
        // verbosePrint( std::string("Compare local_node_index. i: " +
        // std::to_string(i) + " local_node_index: "
        //+ std::to_string(h_local_node_index[i]) + " check: " +
        // std::to_string(h_check_local_node_index[i] + "\n" );
        if (h_local_node_index[i] != h_check_local_node_index[i]) {
          err = true;
        }
      }
    }
    if (err) {
      throw ngpu_exception("Error in local node index map");
    }
    verbosePrint("Finished checking extraction of local node indexees from "
                 "remote node maps\n");
  }

  ////////////////////////////
  // On target host. Loop on all new connections and replace
  // the source node index source_node[i_conn] with the value of the element
  // pointed by the index itself in the array local_node_index
  // source_node[i_conn] = local_node_index[source_node[i_conn]];

  // similar to setUsedSourceNodes
  // replace source_node_flag[i_source] with local_node_index[i_source]
  // clearly read it instead of writing on it!
  // setUsedSourceNodes(old_n_conn, d_source_node_flag);
  // becomes something like
  time_mark = getRealTime();
  fixConnectionSourceNodeIndexes(old_n_conn, d_local_node_index_);
  FixConnectionSourceNodeIndexes_time_ += (getRealTime() - time_mark);

  // On target host. Create n_nodes_to_map nodes of type image_node
  // verbosePrint( std::string("h_n_node_to_map ") +
  // std::to_string(h_n_node_to_map) + "\n" );
  if (h_n_node_to_map > 0) {
    //_Create("image_node", h_n_node_to_map);
    n_image_nodes_ += h_n_node_to_map;
    // verbosePrint( std::string("n_image_nodes_ ") +
    // std::to_string(n_image_nodes_) + "\n" );
  }

  // The following section is used only in special runs for checking maps
  if (check_node_maps_ && new_n_node_map > 0 && isSequence(source) &&
      conn_spec.use_all_remote_source_nodes_) {
    verbosePrint("Started checking remote node maps\n");
    for (uint ib = 0; ib < n_blocks; ib++) {
      uint *hh_node_map = new uint[node_map_block_size_];
      uint *hh_check_node_map = new uint[node_map_block_size_];

      gpuErrchk(cudaMemcpy(
          hh_node_map, h_remote_source_node_map_[group_local_id][gi_host][ib],
          node_map_block_size_ * sizeof(uint), cudaMemcpyDeviceToHost));
      gpuErrchk(cudaMemcpy(hh_check_node_map, h_check_node_map[ib],
                           node_map_block_size_ * sizeof(uint),
                           cudaMemcpyDeviceToHost));
      uint n_in_block = (ib < n_blocks - 1)
                            ? node_map_block_size_
                            : ((n_node_map - 1) % node_map_block_size_ + 1);
      bool err = false;

      for (uint i = 0; i < n_in_block; i++) {
        if (hh_node_map[i] != hh_check_node_map[i]) {
          err = true;
        }
      }
      if (err) {
        throw ngpu_exception("Error in sorted source node map");
      }
    }
    verbosePrint("\tchecking remote node indexes: OK\n");
    for (uint ib = 0; ib < n_blocks; ib++) {
      uint *hh_image_node_map = new uint[node_map_block_size_];
      uint *hh_check_image_node_map = new uint[node_map_block_size_];

      gpuErrchk(cudaMemcpy(
          hh_image_node_map, h_image_node_map_[group_local_id][gi_host][ib],
          node_map_block_size_ * sizeof(uint), cudaMemcpyDeviceToHost));
      gpuErrchk(cudaMemcpy(hh_check_image_node_map, h_check_image_node_map[ib],
                           node_map_block_size_ * sizeof(uint),
                           cudaMemcpyDeviceToHost));
      uint n_in_block = (ib < n_blocks - 1)
                            ? node_map_block_size_
                            : ((n_node_map - 1) % node_map_block_size_ + 1);
      std::vector<uint> i_mapped_vector;
      std::vector<uint> mapped_vector;
      std::vector<uint> check_mapped_vector;
      bool err = false;
      for (uint i = 0; i < n_in_block; i++) {
        uint i_map = ib * node_map_block_size_ + i;
        // printf("sorted map i: %d image_node_map: %d, ", i,
        // hh_image_node_map[i]);
        if (i_map >= n_down_0 && i_map < n_down_1 + transl) {
          // printf("xxxxxxxxxxxxxxxxxxxx\n");
          i_mapped_vector.push_back(i);
          mapped_vector.push_back(hh_image_node_map[i]);
          check_mapped_vector.push_back(hh_check_image_node_map[i]);
        } else {
          // printf("check_image_node_map: %d\n", hh_check_image_node_map[i]);
          if (hh_image_node_map[i] != hh_check_image_node_map[i]) {
            err = true;
          }
        }
      }
      if (err) {
        throw ngpu_exception("Error in sorted image node map");
      }
      verbosePrint("\tchecking image node indexes: OK\n");
      std::sort(mapped_vector.begin(), mapped_vector.end());
      std::sort(check_mapped_vector.begin(), check_mapped_vector.end());
      for (uint i = 0; i < mapped_vector.size(); i++) {
        // printf("i: %d, i_mapped_vector[i]: %d, mapped_vector[i]: %d,
        // check_mapped_vector[i]: %d\n", i, i_mapped_vector[i],
        //      mapped_vector[i], check_mapped_vector[i]);
        if (mapped_vector[i] != check_mapped_vector[i]) {
          err = true;
        }
      }
      if (err) {
        throw ngpu_exception(
            "Error in newly mapped elements in sorted image node map");
      }
    }
    verbosePrint("Finished checking remote node maps\n");
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////////////
  // If necessary , free pointers to memory allocated dynamically in GPU memory
  // if (d_storage != nullptr) {
  // CUDAFREECTRL( "d_storage", d_storage );
  //}

  // if (d_source_node_flag != nullptr) {
  // CUDAFREECTRL( "d_source_node_flag", d_source_node_flag );
  // }

  // if (d_n_used_source_nodes != nullptr) {
  // CUDAFREECTRL( "d_n_used_source_nodes", d_n_used_source_nodes );
  // }

  // if (d_unsorted_source_node_index != nullptr) {
  //   CUDAFREECTRL( "d_unsorted_source_node_index",
  //   d_unsorted_source_node_index );
  // }

  // if (d_sorted_source_node_index != nullptr) {
  //   CUDAFREECTRL( "d_sorted_source_node_index", d_sorted_source_node_index );
  // }

  // if (d_i_unsorted_source_arr != nullptr) {
  //   CUDAFREECTRL( "d_i_unsorted_source_arr", d_i_unsorted_source_arr );
  // }

  // if (d_i_sorted_source_arr != nullptr) {
  //   CUDAFREECTRL( "d_i_sorted_source_arr", d_i_sorted_source_arr );
  // }

  // if (d_node_map != nullptr) {
  //   CUDAFREECTRL( "d_node_map", d_node_map );
  // }

  // if (d_image_node_map != nullptr) {
  //   CUDAFREECTRL( "d_image_node_map", d_image_node_map );
  // }

  // if (d_node_to_map != nullptr) {
  //   CUDAFREECTRL( "d_node_to_map", d_node_to_map );
  // }

  // if (d_node_mapped != nullptr) {
  //   CUDAFREECTRL( "d_node_mapped", d_node_mapped );
  // }

  // if (d_n_node_to_map != nullptr) {
  //   CUDAFREECTRL( "d_n_node_to_map", d_n_node_to_map );
  // }

  // if (d_mapped_local_node_index != nullptr) {
  //   CUDAFREECTRL( "d_mapped_local_node_index", d_mapped_local_node_index );
  // }

  // if (d_i_node_to_map != nullptr) {
  //   CUDAFREECTRL( "d_i_node_to_map", d_i_node_to_map );
  // }

  // if (d_local_node_index != nullptr) {
  //   CUDAFREECTRL( "d_local_node_index", d_local_node_index );
  // }

  if (d_check_local_node_index != nullptr) {
    CUDAFREECTRL("d_check_local_node_index", d_check_local_node_index);
  }

  if (d_check_node_map != nullptr) {
    CUDAFREECTRL("d_check_node_map", d_check_node_map);
    for (uint ib = 0; ib < n_blocks; ib++) {
      CUDAFREECTRL("h_check_node_map[]", h_check_node_map[ib]);
      CUDAFREECTRL("h_check_image_node_map[]", h_check_image_node_map[ib]);
    }
    delete[] h_check_node_map;
    delete[] h_check_image_node_map;
  }

  return 0;
}

// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

// REMOTE CONNECT FUNCTION for target_host matching this_host
template <class ConnKeyT, class ConnStructT>
template <class T1, class T2>
int ConnectionTemplate<ConnKeyT, ConnStructT>::remoteConnectTarget(
    int target_host, T1 source, inode_t n_source, T2 target, inode_t n_target,
    ConnSpec &conn_spec, SynSpec &syn_spec) {
  if (n_source <= 0 || n_target <= 0) {
    return 0;
  }

  // number of nodes actually used in new connections
  uint n_used_source_nodes;

  // map clones used only in special runs for testing
  uint **h_check_node_map = nullptr;
  uint **d_check_node_map = nullptr;

  double time_mark;

  time_mark = getRealTime();
  int64_t old_n_conn = n_conn_;
  // The connect command is performed on both source and target host using
  // the same initial seed and using as source node indexes the integers
  // from 0 to n_source_nodes - 1
  _Connect<inode_t, T2>(conn_random_generator_[this_host_][target_host], 0,
                        n_source, target, n_target, conn_spec, syn_spec, true);
  ConnectRemoteConnectTarget_time_ += (getRealTime() - time_mark);
  if (n_conn_ == old_n_conn) {
    return 0;
  }

  bool use_source_node_flag = false;
  // check if the connection rule or number of connections is such that all
  // remote source nodes should actually be used
  if (!conn_spec.use_all_remote_source_nodes_) {
    use_source_node_flag = true;
    // on both the source and target hosts create a temporary array
    // of booleans having size equal to the number of source nodes
    CUDAREALLOCIFSMALLER("&d_source_node_flag_", &d_source_node_flag_,
                         n_source * sizeof(uint), n_source * sizeof(uint));
    gpuErrchk(cudaMemset(d_source_node_flag_, 0, n_source * sizeof(uint)));

    // printf("this_host: %d\tn_source: %d, n_conn_; %ld\told_n_conn: %ld\n",
    // this_host_, n_source, n_conn_, old_n_conn);

    // flag source nodes used in at least one new connection
    // Loop on all new connections and set source_node_flag[i_source]=true
    time_mark = getRealTime();
    setUsedSourceNodesOnSourceHost(old_n_conn, d_source_node_flag_);
    SetUsedSourceNodes_time_ += (getRealTime() - time_mark);
    //, n_source ); // uncomment only for debugging

    // Count source nodes actually used in new connections
    // Allocate n_used_source_nodes and initialize it to 0
    CUDAREALLOCIFSMALLER("&d_n_used_source_nodes_", &d_n_used_source_nodes_,
                         sizeof(uint), 0);
    gpuErrchk(cudaMemset(d_n_used_source_nodes_, 0, sizeof(uint)));
    // Launch kernel to count used nodes
    if (n_source > 0) {
      time_mark = getRealTime();
      countUsedSourceNodeKernel<<<(n_source + 1023) / 1024, 1024>>>(
          n_source, d_n_used_source_nodes_, d_source_node_flag_);
      CUDASYNC;
      CountUsedSourceNodes_time_ += (getRealTime() - time_mark);
    }

    // copy result from GPU to CPU memory
    gpuErrchk(cudaMemcpy(&n_used_source_nodes, d_n_used_source_nodes_,
                         sizeof(uint), cudaMemcpyDeviceToHost));
    // Reset n_used_source_nodes to 0
    gpuErrchk(cudaMemset(d_n_used_source_nodes_, 0, sizeof(uint)));
  } else {
    n_used_source_nodes = n_source;
  }

  uint n_blocks = h_local_source_node_map_[target_host].size();
  // get current number of elements in the map
  uint n_node_map;
  gpuErrchk(cudaMemcpy(&n_node_map, &d_n_local_source_node_map_[target_host],
                       sizeof(uint), cudaMemcpyDeviceToHost));

  if (n_blocks > 0) {
    // check for consistency between number of elements
    // and number of blocks in the map
    uint tmp_n_blocks = (n_node_map - 1) / node_map_block_size_ + 1;
    if (tmp_n_blocks != n_blocks) {
      throw ngpu_exception(std::string("Inconsistent number of elements ") +
                           std::to_string(n_node_map) +
                           " and number of blocks " + std::to_string(n_blocks) +
                           " in local_source_node_map\n");
    }
  }

  uint i0;
  uint i1;
  uint transl = 0;
  uint n_down_0 = 0;
  uint n_down_1 = 0;
  uint new_n_node_map = 0;
  uint h_n_node_to_map = 0;

  // The following section is used only if the source nodes are defined by a
  // sequence and if the connection rule prescribe that they are all used
  if (!check_node_maps_ && isSequence(source) &&
      conn_spec.use_all_remote_source_nodes_) {
    verbosePrint("Updating local source node maps\n");
    // if source nodes are defined by a sequence, find the first and last index
    // of the sequence
    inode_t i_source_0 = firstNodeIndex(source);
    inode_t i_source_1 = i_source_0 + n_source - 1;

    uint n_mapped = 0;

    if (n_node_map > 0) {
      time_mark = getRealTime();
      // Find number of elements < i_source_0 in the map
      n_down_0 = search_block_array_down<inode_t>(
          &h_local_source_node_map_[target_host][0], n_node_map,
          node_map_block_size_, i_source_0);

      // Find number of elements < i_source_1 + 1 in the map
      n_down_1 = search_block_array_down<inode_t>(
          &h_local_source_node_map_[target_host][0], n_node_map,
          node_map_block_size_, i_source_1 + 1, n_down_0);

      // Compute number of source nodes already mapped
      n_mapped = n_down_1 - n_down_0;
      SearchSourceNodesRangeInMap_time_ += (getRealTime() - time_mark);
    }
    // Compute number of source nodes that must be mapped
    h_n_node_to_map = n_used_source_nodes - n_mapped;

    new_n_node_map = n_node_map + h_n_node_to_map;

    if (new_n_node_map > 0) {
      uint new_n_blocks = (new_n_node_map - 1) / node_map_block_size_ + 1;

      // if new blocks are required for the map, allocate them
      if (new_n_blocks != n_blocks) {
        // Allocate GPU memory for new local-source-node-map blocks
        time_mark = getRealTime();
        allocLocalSourceNodeMapBlocks(h_local_source_node_map_[target_host],
                                      new_n_blocks);
        AllocLocalSourceNodeMapBlocks_time_ += (getRealTime() - time_mark);
      }

      // allocate d_node_map and get it from host
      CUDAREALLOCIFSMALLER("&d_node_map_", &d_node_map_,
                           new_n_blocks * sizeof(uint *),
                           new_n_blocks * sizeof(uint *));
      gpuErrchk(
          cudaMemcpy(d_node_map_, &h_local_source_node_map_[target_host][0],
                     new_n_blocks * sizeof(uint *), cudaMemcpyHostToDevice));
    }

    if (n_node_map > 0 && h_n_node_to_map > 0) {
      uint aux_size = node_map_block_size_;
      i1 = n_node_map - 1;
      i0 = n_down_1;
      transl = h_n_node_to_map;

      time_mark = getRealTime();
      CUDAREALLOCIFSMALLER("&d_ru_storage_", &d_ru_storage_,
                           aux_size * sizeof(uint), aux_size * sizeof(uint));

      // printf("n_node_map: %d, new_n_node_map: %d, transl: %d,
      // node_map_block_size: %d, aux_size: %d, i0: %d, i1: %d\n",
      //      n_node_map, new_n_node_map, transl, node_map_block_size_,
      //      aux_size, i0, i1);

      int64_t i_right = i1;
      while (i_right >= i0) {
        // check that i_right + 1 - aux_size > i0 (considering that i_right is
        // unsigned); if yes, set i_left = i_right + 1 - aux_size otherwise set
        // i_left = i0
        int64_t i_left = max(i_right + 1 - aux_size, (int64_t)i0);
        int64_t n_elem = i_right - i_left + 1;
        // printf("i_left: %ld, i_right: %ld, n_elem: %ld\n", i_left, i_right,
        // n_elem);
        if (n_elem > 0) {
          copyBlockArrayToArrayKernel<<<(n_elem + 1023) / 1024, 1024>>>(
              (uint *)d_ru_storage_, d_node_map_, node_map_block_size_, i_left,
              n_elem);
          CUDASYNC;
          copyArrayToBlockArrayKernel<<<(n_elem + 1023) / 1024, 1024>>>(
              d_node_map_, (uint *)d_ru_storage_, node_map_block_size_,
              i_left + transl, n_elem);
          CUDASYNC;
        }
        i_right -= aux_size;
      }
      TranslateSourceNodeMap_time_ += (getRealTime() - time_mark);
    }

    if (new_n_node_map > 0) {
      time_mark = getRealTime();
      // Allocate the index of the nodes to be mapped and initialize it to 0
      CUDAREALLOCIFSMALLER("&d_i_node_to_map_", &d_i_node_to_map_, sizeof(uint),
                           0);
      gpuErrchk(cudaMemset(d_i_node_to_map_, 0, sizeof(uint)));

      // Allocate boolean array for flagging remote source nodes already mapped
      // and initialize all elements to 0 (false)
      CUDAREALLOCIFSMALLER("&d_node_mapped_", &d_node_mapped_,
                           n_source * sizeof(bool), n_source * sizeof(bool));
      gpuErrchk(cudaMemset(d_node_mapped_, 0, n_source * sizeof(bool)));

      // Launch Kernel that extracts local image index from already mapped
      // remote source nodes and match it to the nodes in a sequence
      if (n_mapped > 0) {
        extractLocalImageIndexOfMappedSourceNodes<<<(n_mapped + 1023) / 1024,
                                                    1024>>>(
            d_node_map_, n_down_0, n_mapped, i_source_0, d_node_mapped_);

        CUDASYNC;
      }

      if (n_source > 0) {
        mapRemoteSourceNodesToLocalImagesKernel<<<(n_source + 1023) / 1024,
                                                  1024>>>(
            d_node_map_, d_node_mapped_, n_source, n_down_0, i_source_0);
        CUDASYNC;
      }

      // CUDAFREECTRL( "d_i_node_to_map", d_i_node_to_map );
      // d_i_node_to_map = nullptr;
      // CUDAFREECTRL( "d_local_node_index", d_local_node_index );

      MapSourceNodeSequence_time_ += (getRealTime() - time_mark);

      // update number of elements in remote source node map
      n_node_map = new_n_node_map;
      gpuErrchk(cudaMemcpy(&d_n_local_source_node_map_[target_host],
                           &n_node_map, sizeof(uint), cudaMemcpyHostToDevice));

      // check for consistency between number of elements
      // and number of blocks in the map
      n_blocks = h_local_source_node_map_[target_host].size();
      uint tmp_n_blocks = (n_node_map - 1) / node_map_block_size_ + 1;
      if (tmp_n_blocks != n_blocks) {
        throw ngpu_exception(
            std::string("Inconsistent number of elements ") +
            std::to_string(n_node_map) + " and number of blocks " +
            std::to_string(n_blocks) + " in local_source_node_map\n");
      }
    }
    verbosePrint("Finished udating local source node maps\n");
  } else {
    // Allocate arrays of size n_used_source_nodes
    time_mark = getRealTime();

    CUDAREALLOCIFSMALLER(
        "&d_unsorted_source_node_index_", &d_unsorted_source_node_index_,
        n_used_source_nodes * sizeof(uint), n_used_source_nodes * sizeof(uint));
    CUDAREALLOCIFSMALLER(
        "&d_sorted_source_node_index_", &d_sorted_source_node_index_,
        n_used_source_nodes * sizeof(uint), n_used_source_nodes * sizeof(uint));

    AllocUsedSourceNodes_time_ += (getRealTime() - time_mark);

    // Fill the arrays of nodes actually used by new connections
    // Launch kernel to fill the arrays
    if (n_source > 0) {
      time_mark = getRealTime();
      getUsedSourceNodeIndexKernel<<<(n_source + 1023) / 1024, 1024>>>(
          source, n_source, d_n_used_source_nodes_, use_source_node_flag, false,
          d_source_node_flag_, d_unsorted_source_node_index_);
      CUDASYNC;
      GetUsedSourceNodeIndex_time_ += (getRealTime() - time_mark);
    }

    // Sort the arrays using unsorted_source_node_index as key ->
    // sorted_source_node_index
    time_mark = getRealTime();

    // Determine temporary storage requirements for RadixSort
    size_t sort_storage_bytes = 0;
    //<BEGIN-CLANG-TIDY-SKIP>//
    cub::DeviceRadixSort::SortKeys(
        nullptr, // d_ru_storage_,
        sort_storage_bytes, d_unsorted_source_node_index_,
        d_sorted_source_node_index_, n_used_source_nodes);
    //<END-CLANG-TIDY-SKIP>//

    // Allocate temporary storage
    CUDAREALLOCIFSMALLER("&d_ru_storage_", &d_ru_storage_, sort_storage_bytes,
                         sort_storage_bytes);

    // Run sorting operation
    //<BEGIN-CLANG-TIDY-SKIP>//
    cub::DeviceRadixSort::SortKeys(
        d_ru_storage_, sort_storage_bytes, d_unsorted_source_node_index_,
        d_sorted_source_node_index_, n_used_source_nodes);
    //<END-CLANG-TIDY-SKIP>//
    SortUsedSourceNodeIndex_time_ += (getRealTime() - time_mark);

    DBGCUDASYNC;

    if (n_blocks > 0) {
      CUDAREALLOCIFSMALLER("&d_node_map_", &d_node_map_,
                           n_blocks * sizeof(uint *),
                           n_blocks * sizeof(uint *));
      gpuErrchk(cudaMemcpy(d_node_map_,
                           &h_local_source_node_map_[target_host][0],
                           n_blocks * sizeof(uint *), cudaMemcpyHostToDevice));
    }

    time_mark = getRealTime();
    // Allocate boolean array for flagging remote source nodes not yet mapped
    // and initialize all elements to 0 (false)
    CUDAREALLOCIFSMALLER("&d_node_to_map_", &d_node_to_map_,
                         n_used_source_nodes * sizeof(bool),
                         n_used_source_nodes * sizeof(bool));
    gpuErrchk(
        cudaMemset(d_node_to_map_, 0, n_used_source_nodes * sizeof(bool)));

    // Allocate number of nodes to be mapped and initialize it to 0
    CUDAREALLOCIFSMALLER("&d_n_node_to_map_", &d_n_node_to_map_, sizeof(uint),
                         0);
    gpuErrchk(cudaMemset(d_n_node_to_map_, 0, sizeof(uint)));
    AllocNodeToMap_time_ += (getRealTime() - time_mark);

    // launch kernel that searches remote source nodes indexes not in the map,
    // flags the nodes not yet mapped and counts them
    if (n_used_source_nodes > 0) {
      time_mark = getRealTime();
      searchNodeIndexNotInMapKernel<<<(n_used_source_nodes + 1023) / 1024,
                                      1024>>>(
          d_node_map_, n_node_map, d_sorted_source_node_index_, d_node_to_map_,
          d_n_node_to_map_, n_used_source_nodes);
      CUDASYNC;
      SearchNodeIndexNotInMap_time_ += (getRealTime() - time_mark);
    }

    gpuErrchk(cudaMemcpy(&h_n_node_to_map, d_n_node_to_map_, sizeof(uint),
                         cudaMemcpyDeviceToHost));

    // The following section is used only in special runs for checking maps
    if (check_node_maps_ && isSequence(source) &&
        conn_spec.use_all_remote_source_nodes_) {
      verbosePrint("Preparing check of remote node maps\n");
      // if source nodes are defined by a sequence, find the first and last
      // index of the sequence
      inode_t i_source_0 = firstNodeIndex(source);
      // verbosePrint( std::string(" i_source_0: " + std::to_string(i_source_0)
      // + "\n" );
      inode_t i_source_1 = i_source_0 + n_source - 1;
      // verbosePrint( std::string(" i_source_1: " + std::to_string(i_source_1)
      // + "\n" );

      // verbosePrint( std::string(" n_node_map: " + std::to_string(n_node_map)
      // + "\n" );

      uint check_n_mapped = 0;

      if (n_node_map > 0) {
        // Find number of elements < i_source_0 in the map
        n_down_0 = search_block_array_down<inode_t>(
            &h_local_source_node_map_[target_host][0], n_node_map,
            node_map_block_size_, i_source_0);
        // verbosePrint( std::string(" n_down_0: ") + std::to_string(n_down_0) +
        // "\n" );

        // Find number of elements < i_source_1 + 1 in the map
        n_down_1 = search_block_array_down<inode_t>(
            &h_local_source_node_map_[target_host][0], n_node_map,
            node_map_block_size_, i_source_1 + 1, n_down_0);
        // verbosePrint( std::string(" n_down_1: ") + std::to_string(n_down_1) +
        // "\n" );
        //  Compute number of source nodes already mapped
        check_n_mapped = n_down_1 - n_down_0;
      }
      // verbosePrint( std::string(" check n_mapped: ") +
      // std::to_string(check_n_mapped) + "\n" );
      //  Compute number of source nodes that must be mapped
      uint check_to_be_mapped = n_used_source_nodes - check_n_mapped;
      // verbosePrint( std::string(" check to_be_mapped: ") +
      // std::to_string(check_to_be_mapped) + "\n" ); verbosePrint(
      // std::string(" to_be_mapped: ") + std::to_string(h_n_node_to_map) + "\n"
      // );
      if (check_to_be_mapped != h_n_node_to_map) {
        throw ngpu_exception("bbb Error in computing n. of nodes to be mapped");
      }

      // if (n_node_map > 0 && check_to_be_mapped > 0) {
      // uint aux_size = node_map_block_size_;
      // i1 = n_node_map - 1;
      // i0 = n_down_1;
      // transl = check_to_be_mapped;

      new_n_node_map = n_node_map + transl;
      // printf("%d new_n_node_map: %d, n_node_map: %d, transl: %d\n",
      // this_host_, new_n_node_map, n_node_map, transl);
      if (new_n_node_map > 0) {
        int new_n_blocks =
            (int)(((int64_t)new_n_node_map - 1) / node_map_block_size_ + 1);
        // printf("%d new_n_blocks: %d, n_blocks: %d\n", this_host_,
        // new_n_blocks, n_blocks);

        h_check_node_map = new uint *[new_n_blocks];
        for (uint ib = 0; ib < (uint)new_n_blocks; ib++) {
          CUDAMALLOCCTRL("h_check_node_map", &h_check_node_map[ib],
                         node_map_block_size_ * sizeof(uint));
          if (ib < n_blocks) {
            gpuErrchk(cudaMemcpy(
                h_check_node_map[ib], h_local_source_node_map_[target_host][ib],
                node_map_block_size_ * sizeof(uint), cudaMemcpyDeviceToDevice));
          }
        }
        // allocate d_check_node_map and get it from host
        CUDAMALLOCCTRL("&d_check_node_map", &d_check_node_map,
                       new_n_blocks * sizeof(uint *));
        gpuErrchk(cudaMemcpy(d_check_node_map, h_check_node_map,
                             new_n_blocks * sizeof(uint *),
                             cudaMemcpyHostToDevice));
        // allocate d_check_image_node_map and get it from host
      }

      if (n_node_map > 0 && check_to_be_mapped > 0) {
        uint aux_size = node_map_block_size_;
        i1 = n_node_map - 1;
        i0 = n_down_1;
        transl = check_to_be_mapped;

        CUDAREALLOCIFSMALLER("&d_ru_storage_", &d_ru_storage_,
                             aux_size * sizeof(uint), aux_size * sizeof(uint));

        // printf("n_node_map: %d, new_n_node_map: %d, transl: %d,
        // node_map_block_size: %d, aux_size: %d, i0: %d, i1: %d\n",
        //      n_node_map, new_n_node_map, transl, node_map_block_size_,
        //      aux_size, i0, i1);

        int64_t i_right = i1;
        while (i_right >= i0) {
          // check that i_right + 1 - aux_size > i0 (considering that i_right is
          // unsigned); if yes, set i_left = i_right + 1 - aux_size otherwise
          // set i_left = i0
          int64_t i_left = max(i_right + 1 - aux_size, (int64_t)i0);
          int64_t n_elem = i_right - i_left + 1;
          // printf("i_left: %ld, i_right: %ld, n_elem: %ld\n", i_left, i_right,
          // n_elem);
          if (n_elem > 0) {
            copyBlockArrayToArrayKernel<<<(n_elem + 1023) / 1024, 1024>>>(
                (uint *)d_ru_storage_, d_check_node_map, node_map_block_size_,
                i_left, n_elem);
            CUDASYNC;
            copyArrayToBlockArrayKernel<<<(n_elem + 1023) / 1024, 1024>>>(
                d_check_node_map, (uint *)d_ru_storage_, node_map_block_size_,
                i_left + transl, n_elem);
            CUDASYNC;
          }
          i_right -= aux_size;
        }
      }

      if (new_n_node_map > 0) {
        // Allocate the index of the nodes to be mapped and initialize it to 0
        CUDAREALLOCIFSMALLER("&d_i_node_to_map_", &d_i_node_to_map_,
                             sizeof(uint), 0);
        gpuErrchk(cudaMemset(d_i_node_to_map_, 0, sizeof(uint)));

        // Allocate boolean array for flagging remote source nodes already
        // mapped and initialize all elements to 0 (false)
        CUDAREALLOCIFSMALLER("&d_node_mapped_", &d_node_mapped_,
                             n_source * sizeof(bool), n_source * sizeof(bool));
        gpuErrchk(cudaMemset(d_node_mapped_, 0, n_source * sizeof(bool)));

        // Launch Kernel that extracts local image index from already mapped
        // remote source nodes and match it to the nodes in a sequence
        if (check_n_mapped > 0) {
          extractLocalImageIndexOfMappedSourceNodes<<<
              (check_n_mapped + 1023) / 1024, 1024>>>(
              d_check_node_map, n_down_0, check_n_mapped, i_source_0,
              d_node_mapped_);

          CUDASYNC;
        }

        if (n_source > 0) {
          mapRemoteSourceNodesToLocalImagesKernel<<<(n_source + 1023) / 1024,
                                                    1024>>>(
              d_check_node_map, d_node_mapped_, n_source, n_down_0, i_source_0);
          CUDASYNC;
        }

        // CUDAFREECTRL( "d_i_node_to_map", d_i_node_to_map );
        // d_i_node_to_map = nullptr;
      }
      verbosePrint("Finished preparing check of remote node maps\n");
    }

    ////////////////////////////////////////////

    uint new_n_blocks = 0;
    if (h_n_node_to_map > 0) {
      // Check if new blocks are required for the map
      new_n_blocks =
          (n_node_map + h_n_node_to_map - 1) / node_map_block_size_ + 1;

      // if new blocks are required for the map, allocate them
      if (new_n_blocks != n_blocks) {
        // Allocate GPU memory for new local-source-node-map blocks
        time_mark = getRealTime();
        allocLocalSourceNodeMapBlocks(h_local_source_node_map_[target_host],
                                      new_n_blocks);
        AllocLocalSourceNodeMapBlocks_time_ += (getRealTime() - time_mark);

        // free d_node_map
        // if ( d_node_map != nullptr )
        //  {
        //    CUDAFREECTRL( "d_node_map", d_node_map );
        //    d_node_map = nullptr;
        //  }
        // update number of blocks in the map
        n_blocks = new_n_blocks;

        // reallocate d_node_map and get it from host
        CUDAREALLOCIFSMALLER("&d_node_map_", &d_node_map_,
                             n_blocks * sizeof(uint *),
                             n_blocks * sizeof(uint *));
        gpuErrchk(
            cudaMemcpy(d_node_map_, &h_local_source_node_map_[target_host][0],
                       n_blocks * sizeof(uint *), cudaMemcpyHostToDevice));
      }
    }
    if (n_blocks > 0) {

      // Map the not-yet-mapped source nodes using a kernel
      // similar to the one used for counting
      // In the target host unmapped remote source nodes must be mapped
      // to local nodes from n_nodes to n_nodes + n_node_to_map

      // Allocate the index of the nodes to be mapped and initialize it to 0
      CUDAREALLOCIFSMALLER("&d_i_node_to_map_", &d_i_node_to_map_, sizeof(uint),
                           0);
      gpuErrchk(cudaMemset(d_i_node_to_map_, 0, sizeof(uint)));

      // launch kernel that checks if nodes are already in map
      // if not insert them in the map
      // In the target host, put in the map the pair:
      // (source_node_index, image_node_map_i0 + i_node_to_map)

      if (n_used_source_nodes > 0) {
        time_mark = getRealTime();
        insertNodesInMapKernel<<<(n_used_source_nodes + 1023) / 1024, 1024>>>(
            d_node_map_, n_node_map, d_sorted_source_node_index_,
            d_node_to_map_, d_i_node_to_map_, n_used_source_nodes);
        CUDASYNC;
        InsertNodesInMap_time_ += (getRealTime() - time_mark);
      }
    }

    // update number of elements in remote source node map
    n_node_map += h_n_node_to_map;
    gpuErrchk(cudaMemcpy(&d_n_local_source_node_map_[target_host], &n_node_map,
                         sizeof(uint), cudaMemcpyHostToDevice));

    // check for consistency between number of elements
    // and number of blocks in the map
    uint tmp_n_blocks = (n_node_map - 1) / node_map_block_size_ + 1;
    if (tmp_n_blocks != n_blocks) {
      throw ngpu_exception(std::string("Inconsistent number of elements ") +
                           std::to_string(n_node_map) +
                           " and number of blocks " + std::to_string(n_blocks) +
                           " in local_source_node_map\n");
    }

    if (h_n_node_to_map > 0) {

      // Sort the WHOLE map source_node_map
      // using block sort algorithm copass_sort
      // typical usage:
      // copass_sort::sort<uint>(key_subarray, n,
      //				       aux_size, d_storage,
      // storage_bytes);
      // Determine temporary storage requirements for copass_sort
      time_mark = getRealTime();
      int64_t storage_bytes = 0;
      // if (d_storage != nullptr) {
      //	CUDAFREECTRL( "d_storage", d_storage );
      //	d_storage = nullptr;
      // }
      copass_sort::sort<uint>(&h_local_source_node_map_[target_host][0],
                              n_node_map, node_map_block_size_,
                              nullptr, // d_ru_storage_,
                              storage_bytes, 0);

      // Allocate temporary storage
      CUDAREALLOCIFSMALLER("&d_ru_storage_", &d_ru_storage_, storage_bytes,
                           storage_bytes);

      // Run sorting operation
      copass_sort::sort<uint>(&h_local_source_node_map_[target_host][0],
                              n_node_map, node_map_block_size_, d_ru_storage_,
                              storage_bytes, 0);
      // CUDAFREECTRL( "d_storage", d_storage );
      // d_storage = nullptr;

      SortSourceImageNodeMap_time_ += (getRealTime() - time_mark);
    }
  }

  // Remove temporary new connections in source host !!!!!!!!!!!
  // potential problem: check that number of blocks is an independent variable
  // not calculated from n_conn_
  // connect.cu riga 462. Corrected but better keep an eye
  // also, hopefully the is no global device variable for n_conn_
  n_conn_ = old_n_conn;

  // The following section is used only in special runs for checking maps
  if (check_node_maps_ && new_n_node_map > 0 && isSequence(source) &&
      conn_spec.use_all_remote_source_nodes_) {
    verbosePrint("Started checking local source node maps\n");
    for (uint ib = 0; ib < n_blocks; ib++) {
      uint *hh_node_map = new uint[node_map_block_size_];
      uint *hh_check_node_map = new uint[node_map_block_size_];

      gpuErrchk(cudaMemcpy(
          hh_node_map, h_local_source_node_map_[target_host][ib],
          node_map_block_size_ * sizeof(uint), cudaMemcpyDeviceToHost));
      gpuErrchk(cudaMemcpy(hh_check_node_map, h_check_node_map[ib],
                           node_map_block_size_ * sizeof(uint),
                           cudaMemcpyDeviceToHost));
      uint n_in_block = (ib < n_blocks - 1)
                            ? node_map_block_size_
                            : ((n_node_map - 1) % node_map_block_size_ + 1);
      bool err = false;

      for (uint i = 0; i < n_in_block; i++) {
        if (hh_node_map[i] != hh_check_node_map[i]) {
          err = true;
        }
      }
      if (err) {
        throw ngpu_exception("Error in sorted local source node map");
      }
    }
    verbosePrint("Finished checking local source node maps\n");
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////////////
  // If necessary , free pointers to memory allocated dynamically in GPU memory
  // if (d_storage != nullptr) {
  //  CUDAFREECTRL( "d_storage", d_storage );
  //}

  // if (d_source_node_flag != nullptr) {
  //   CUDAFREECTRL( "d_source_node_flag", d_source_node_flag );
  // }

  // if (d_n_used_source_nodes != nullptr) {
  //   CUDAFREECTRL( "d_n_used_source_nodes", d_n_used_source_nodes );
  // }

  // if (d_unsorted_source_node_index != nullptr) {
  //   CUDAFREECTRL( "d_unsorted_source_node_index",
  //   d_unsorted_source_node_index );
  // }

  // if (d_sorted_source_node_index != nullptr) {
  //   CUDAFREECTRL( "d_sorted_source_node_index", d_sorted_source_node_index );
  // }

  // if (d_node_map != nullptr) {
  //   CUDAFREECTRL( "d_node_map", d_node_map );
  // }

  // if (d_node_to_map != nullptr) {
  //   CUDAFREECTRL( "d_node_to_map", d_node_to_map );
  // }

  // if (d_node_mapped != nullptr) {
  //   CUDAFREECTRL( "d_node_mapped", d_node_mapped );
  // }

  // if (d_n_node_to_map != nullptr) {
  //   CUDAFREECTRL( "d_n_node_to_map", d_n_node_to_map );
  // }

  // if (d_i_node_to_map != nullptr) {
  //   CUDAFREECTRL( "d_i_node_to_map", d_i_node_to_map );
  // }

  if (d_check_node_map != nullptr) {
    CUDAFREECTRL("d_check_node_map", d_check_node_map);
    for (uint ib = 0; ib < n_blocks; ib++) {
      CUDAFREECTRL("h_check_node_map[]", h_check_node_map[ib]);
    }
    delete[] h_check_node_map;
  }

  return 0;
}

// Method that creates a group of hosts for remote spike communication (i.e. a
// group of MPI processes) host_arr: array of host inexes, n_hosts: nomber of
// hosts in the group
template <class ConnKeyT, class ConnStructT>
int ConnectionTemplate<ConnKeyT, ConnStructT>::CreateHostGroup(int *host_arr,
                                                               int n_hosts,
                                                               bool mpi_flag) {
  if (first_connection_flag_ == false) {
    throw ngpu_exception("Host groups must be defined before creating "
                         "connections");
  }
  // pushes all the host indexes in a vector, hg, and check whether this host is
  // in the group TO IMPROVE BY USING AN UNORDERED SET, TO AVOID POSSIBLE
  // REPETITIONS IN host_arr OR CHECK THAT THERE ARE NO REPETITIONS
  std::vector<int> hg;
  bool this_host_is_in_group = false;
  for (int ih = 0; ih < n_hosts; ih++) {
    int i_host = host_arr[ih];
    // check whether this host is in the group
    if (i_host == this_host_) {
      this_host_is_in_group = true;
    }
    hg.push_back(i_host);
  }

  // if this host is not in the group, set the entry of host_group_local_id_ to
  // -1
  if (!this_host_is_in_group) {
    host_group_local_id_.push_back(-1);
  } else { // the code in the block is executed only if this host is in the
           // group
    // set the local id of the group to be the current size of the local host
    // group array
    int group_local_id = host_group_.size();
    // push the local id in the array of local indexes  of all host groups
    host_group_local_id_.push_back(group_local_id);
    // push the new group into the host_group_ vector
    host_group_.push_back(hg);
    if (host_group_source_node_sequence_flag_) {
      // push a vector of node indexes in host_group_source_node_min
      std::vector<inode_t> i_node_min_vect(n_hosts, UINT_MAX);
      host_group_source_node_min_.push_back(i_node_min_vect);
      // push a vector of node indexes in host_group_source_node_max
      std::vector<inode_t> i_node_max_vect(n_hosts, 0);
      host_group_source_node_max_.push_back(i_node_max_vect);
    } else {
      // push a vector of empty unordered sets into host_group_source_node_
      std::vector<std::unordered_set<inode_t>> empty_node_us(n_hosts);
      host_group_source_node_.push_back(empty_node_us);
      // push a vector of empty vectors into host_group_source_node_vect
      std::vector<std::vector<inode_t>> empty_node_vect(n_hosts);
      host_group_source_node_vect_.push_back(empty_node_vect);
    }
    host_group_local_source_node_map_.push_back(std::vector<uint>());
    std::vector<std::vector<int64_t>> hg_lni(hg.size(), std::vector<int64_t>());
    host_group_local_node_index_.push_back(hg_lni);
  }
#ifdef HAVE_MPI
  if (mpi_flag) {
    // It is mandatory that the collective creation of MPI_Group and MPI_Comm
    // are executed on all MPI processess no matter if they are actually used or
    // not Get the group from the world communicator
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    // create new MPI group from host group hg
    MPI_Group newgroup;
    MPI_Group_incl(world_group, hg.size(), &hg[0], &newgroup);
    // create new MPI communicator
    MPI_Comm newcomm;
    MPI_Comm_create(MPI_COMM_WORLD, newgroup, &newcomm);
    // They are inserted in the vectors only if they are actually needed
    // Note that the two vectors will be indexed by the local group id, in the
    // same way as host_group_
    if (this_host_is_in_group) {
      // insert them in MPI groups and comm vectors
      mpi_group_vect_.push_back(newgroup);
      mpi_comm_vect_.push_back(newcomm);
    }
  }
#endif
  // return as output the index of the last entry in host_group_local_id_
  // which correspond to the newly created group
  return host_group_local_id_.size() - 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// replace each element of an array arr[i_elem] with its modulo with modulus,
// arr[i_elem] % modulus
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
__global__ void moduloKernel(T *array, int64_t n_elem, int64_t modulus) {
  int64_t i_elem = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_elem >= n_elem) {
    return;
  }
  array[i_elem] = array[i_elem] % modulus;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// CUDA kernel that subtracts an offset from the source node indexes
// in a range of connections in a block
template <class ConnKeyT>
__global__ void subtractSourceOffsetKernel(ConnKeyT *conn_key_subarray,
                                           int64_t n_conn, int64_t offset) {
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn >= n_conn) {
    return;
  }
  inode_t index = (inode_t)((int64_t)conn_key_subarray[i_conn] - offset);
  setConnSource<ConnKeyT>(conn_key_subarray[i_conn], index);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Print connections in a block for distributed rule with source host+node
// combined index
template <class T, class ConnStructT>
__global__ void printCombinedSourceHostAndNodeConnections(
    T *conn_key_subarray, ConnStructT *conn_struct_subarray, int64_t n_conn) {
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn >= n_conn) {
    return;
  }
  T source = conn_key_subarray[i_conn];
  inode_t target = getConnTarget<ConnStructT>(conn_struct_subarray[i_conn]);

  printf("printCombinedSourceHostAndNodeConnections i_conn: %lld, i_source: "
         "%lld, i_target: %d\n",
         i_conn, (uint64_t)source, target);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Build connections with fixed indegree rule for source neurons and target
// neurons distributed across MPI processes (hosts) Template
///////////////////////////////////////////////////////////////////////////////////////////////////////
template <class ConnKeyT, class ConnStructT>
template <class T1, class T2>
int ConnectionTemplate<ConnKeyT, ConnStructT>::_ConnectDistributedFixedIndegree(
    int *source_host_arr, int n_source_host, T1 *h_source_arr,
    inode_t *n_source_arr, int *target_host_arr, int n_target_host,
    T2 *h_target_arr, inode_t *n_target_arr, int indegree, int i_host_group,
    SynSpec &syn_spec) {
  // check that connection key part has the right size in bytes
  if (sizeof(ConnKeyT) != 4 && sizeof(ConnKeyT) != 8) {
    throw ngpu_exception("sizeof(ConnKeyT) must be either 4 or 8 bytes"
                         " in  _ConnectDistrubutedFixedIndegree");
  }

  if (first_connection_flag_ == true) {
    remoteConnectionMapInit();
    first_connection_flag_ = false;
  }

  if (i_host_group < 0) { // point-to-point MPI communication
    throw ngpu_exception("_ConnectDistrubutedFixedIndegree is not available "
                         "for point-to-point MPI communication");
  }
  int group_local_id = host_group_local_id_[i_host_group];
  if (group_local_id < 0) { // this host is not in group
    return 0;
  }

  // Connections must be created only for target_host == this_host
  // so first of all check if this_host is in target_host_arr, and in this case
  // find the index in the array and check that it is not present more than once
  int i_target_host = -1;
  for (int ith = 0; ith < n_target_host; ith++) {
    int target_host = target_host_arr[ith];
    if (target_host >= n_hosts_) {
      throw ngpu_exception(
          "Target host index out of range in _ConnectDistributedFixedIndegree");
    }
    if (target_host == this_host_) {
      if (i_target_host >= 0) {
        throw ngpu_exception("Same target host repeated more than once in "
                             "_ConnectDistributedFixedIndegree");
      }
      i_target_host = ith;
    }
  }

  // arrays of source and target nodes of each source and target host in GPU
  // memory They can be either the index of the first node of each source/target
  // group or pointers to arrays of indexes
  T1 d_source_arr[n_source_host];
  T2 d_target_arr[n_target_host];

  // loop on source hosts
  for (int ish = 0; ish < n_source_host; ish++) {
    // find the position i_host of the source host in the host group
    int source_host = source_host_arr[ish];
    if (source_host >= n_hosts_) {
      throw ngpu_exception(
          "Source host index out of range in _ConnectDistributedFixedIndegree");
    }
    // find the source host index in the host group
    auto it = std::find(host_group_[group_local_id].begin(),
                        host_group_[group_local_id].end(), source_host);
    if (it == host_group_[group_local_id].end()) {
      throw ngpu_exception("source host not found in host group in "
                           "_ConnectDistrubutedFixedIndegree");
    }
    int i_host = it - host_group_[group_local_id].begin();

    // insert the source node indexes in the host-group array of source nodes of
    // the host i_host
    double time_mark = getRealTime();
    for (inode_t i = 0; i < n_source_arr[ish]; i++) {
      if (host_group_source_node_sequence_flag_) {
        inode_t inode_min;
        inode_t inode_max;
        getNodeIndexRange(h_source_arr[ish], n_source_arr[ish], inode_min,
                          inode_max);
        host_group_source_node_min_[group_local_id][i_host] =
            min(host_group_source_node_min_[group_local_id][i_host], inode_min);
        host_group_source_node_max_[group_local_id][i_host] =
            max(host_group_source_node_max_[group_local_id][i_host], inode_max);
      } else {
        inode_t i_source = hGetNodeIndex(h_source_arr[ish], i);
        host_group_source_node_[group_local_id][i_host].insert(i_source);
      }
    }
    InsertHostGroupSourceNode_time_ += (getRealTime() - time_mark);
  }

  if (i_target_host < 0) {
    return 0; // this_host_ is not among target hosts
  }

  for (int ish = 0; ish < n_source_host; ish++) {
    // if T1 is a pointer copy source index array from CPU to GPU memory
    // otherwise d_source_arr[ish] will be the first index of the source node
    // group
    d_source_arr[ish] =
        copyNodeArrayToDevice(h_source_arr[ish], n_source_arr[ish]);
  }
  inode_t n_target = n_target_arr[i_target_host];
  // if T2 is a pointer copy target index array from CPU to GPU memory
  // otherwise d_target will be the first index of the target node group
  T2 d_target = copyNodeArrayToDevice(h_target_arr[i_target_host], n_target);

  // compute number of new connections that must be created
  int64_t n_new_conn_tot = n_target * indegree;
  // printf("this_host: %d\tn_new_conn_tot: %ld\tconn_block_size_:
  // %ld\tindegree: %d\n", this_host_, n_new_conn_tot, conn_block_size_,
  // indegree);

  // Create new connection blocks as needed
  int new_n_block = (int)((n_conn_ + n_new_conn_tot + conn_block_size_ - 1) /
                          conn_block_size_);
  allocateNewBlocks(new_n_block);

  // Cumulative sum of n_source: n_source_cumul
  int64_t n_source_cumul[n_source_host + 1];
  n_source_cumul[0] = 0;
  for (int ish = 0; ish < n_source_host; ish++) {
    n_source_cumul[ish + 1] = n_source_cumul[ish] + n_source_arr[ish];
  }

  // total number of source nodes:
  unsigned long long n_source_tot = n_source_cumul[n_source_host];
  // printf("n_source_tot: %lld\n", n_source_tot);

  if (n_source_tot > (ConnKeyT)(-1)) { // for an unsigned integer type, -1 is
                                       // the max value it can represent
    throw ngpu_exception(
        "Total number of source neuron too large for the connection key type"
        " used in _ConnectDistrubutedFixedIndegree");
  }

  // printf("Generating connections with fixed_indegree rule...\n");

  // Loop on connection blocks where new connections are stored
  // int64_t conn_source_ids_offset = 0; uncomment only if needed
  int64_t n_prev_conn = 0;
  int ib0 = (int)(n_conn_ / conn_block_size_);
  for (int ib = ib0; ib < new_n_block; ib++) {
    int64_t n_block_conn; // number of new connections in the current block of
                          // the loop
    int64_t i_conn0;      // index of first new connection in this block
    if (new_n_block == ib0 + 1) // no new blocks were needed/allocated
    {                           // all connections are in the same block
      i_conn0 = n_conn_ % conn_block_size_;
      n_block_conn = n_new_conn_tot;
    } else if (ib ==
               ib0) // first block of the loop, cannot be the last (see above)
    {               // first block
      i_conn0 = n_conn_ % conn_block_size_;
      n_block_conn = conn_block_size_ - i_conn0;
    } else if (ib ==
               new_n_block -
                   1) // last block of the loop, cannot be the first (see above)
    {                 // last block
      i_conn0 = 0;
      n_block_conn = (n_conn_ + n_new_conn_tot - 1) % conn_block_size_ + 1;
    } else // block is neither the first nor the last of the loop
    {
      i_conn0 = 0;
      n_block_conn = conn_block_size_;
    }
    // Connection source host&node indexes are extracted as 32 bit or 64 bit
    // random integers and then replaced with their modulo in the range from 0
    // to n_source_tot - 1 . Each target node index is repeated indegree times
    // in the connections.
    if (sizeof(ConnKeyT) == 8) { // 64 bit
      // generate array of 64 bit random unsigned integers
      curandGenerateLongLong(conn_random_generator_[this_host_][this_host_],
                             (unsigned long long *)conn_key_vect_[ib] + i_conn0,
                             n_block_conn);
      // Replace each 64 bit random integer with its modulo in the range from 0
      // to n_source_tot - 1
      moduloKernel<unsigned long long><<<(n_block_conn + 1023) / 1024, 1024>>>(
          (unsigned long long *)conn_key_vect_[ib] + i_conn0, n_block_conn,
          n_source_tot);
      DBGCUDASYNC;
    } else {
      // generate array of 32 bit random unsigned integers
      curandGenerate(conn_random_generator_[this_host_][this_host_],
                     (unsigned int *)conn_key_vect_[ib] + i_conn0,
                     n_block_conn);

      // Replace each 32 bit random integer with its modulo in the range from 0
      // to n_source_tot - 1
      moduloKernel<unsigned int><<<(n_block_conn + 1023) / 1024, 1024>>>(
          (unsigned int *)conn_key_vect_[ib] + i_conn0, n_block_conn,
          n_source_tot);
      DBGCUDASYNC;
    }

    // set the target indexes in the new connections using the fixed-indegree
    // rule
    if (n_block_conn > 0) {
      setIndegreeTarget<T2, ConnStructT>
          <<<(n_block_conn + 1023) / 1024, 1024>>>(
              conn_struct_vect_[ib] + i_conn0, n_block_conn, n_prev_conn,
              d_target, indegree);
      DBGCUDASYNC;
    }

    n_prev_conn += n_block_conn;
  } // end of loop on connection blocks

  int64_t i_conn0 =
      n_conn_ % conn_block_size_; // index of the first new connection

  // Sort the connections in the blocks with the COPASS algorithm
  // Sorting should start from block ib0 and it should be performed on
  // n_new_conn_tot + i_conn0 elements skipping the first i_conn0 connections

  // Allocating auxiliary GPU memory
  int64_t sort_storage_bytes = 0;

  copass_sort::sort<ConnKeyT, ConnStructT>(
      &conn_key_vect_[ib0], &conn_struct_vect_[ib0], n_new_conn_tot,
      conn_block_size_, nullptr, sort_storage_bytes, i_conn0);
  // printf( "xxx storage bytes: %ld\n", sort_storage_bytes );
  CUDAREALLOCIFSMALLER("&d_ru_storage_", &d_ru_storage_, sort_storage_bytes,
                       sort_storage_bytes / 4);

  // printf( "Sorting...\n" );
  copass_sort::sort<ConnKeyT, ConnStructT>(
      &conn_key_vect_[ib0], &conn_struct_vect_[ib0], n_new_conn_tot,
      conn_block_size_, d_ru_storage_, sort_storage_bytes, i_conn0);

  ///////////////////////////////////////////////////////////////////
  // Partition the new connections according to the source host
  // using the cumulative sum array n_source_cumul[n_source_host + 1] created
  // previously Each element of this array, n_source_cumul[i_host], represents
  // the first index of the source nodes belonging to the host i_host in the
  // host&node index representation as integers from 0 to n_source_tot Since the
  // connection are sorted by the source host&node index in this representation,
  // this index can be used to partition them according to the source host
  // through a search (locate)

  int ib1 = ib0;
  int64_t part_conn0 =
      n_conn_; // index of the first connection of the first partition
  int64_t old_n_conn_ = n_conn_;
  int ret;

  // Loop on source hosts (it's fine to do it with a loop, no need for
  // parallelizing)
  for (int ish = 0; ish < n_source_host; ish++) {
    // Locate (search) the current element of the n_source_cumul array in the
    // conn_key_vect_ block
    int64_t value = n_source_cumul[ish + 1];
    // printf("ish: %d\tvalue: %ld\n", ish, value);

    int64_t n_block_conn; // number of new connections in the current block of
                          // the loop
    int64_t i_conn0;      // index of first new connection in this block
    int64_t h_position = -1; // position of first connection of current
                             // partition in the connection block
    int ib;                  // index of loop on connection blocks
    // Loop on connection blocks where new connections are stored
    for (ib = ib1; ib < new_n_block; ib++) {
      if (new_n_block == ib1 + 1) // all connections are in the same block
      {                           // all connections are in the same block
        i_conn0 = part_conn0 % conn_block_size_;
        n_block_conn =
            (old_n_conn_ + n_new_conn_tot - 1) % conn_block_size_ + 1 - i_conn0;
      } else if (ib ==
                 ib1) // first block of the loop, cannot be the last (see above)
      {               // first block
        i_conn0 = part_conn0 % conn_block_size_;
        n_block_conn = conn_block_size_ - i_conn0;
      } else if (ib == new_n_block - 1) // last block of the loop, cannot be the
                                        // first (see above)
      {                                 // last block
        i_conn0 = 0;
        n_block_conn =
            (old_n_conn_ + n_new_conn_tot - 1) % conn_block_size_ + 1;
      } else // block is neither the first nor the last of the loop
      {
        i_conn0 = 0;
        n_block_conn = conn_block_size_;
      }

      // allocate a 64 bit integer to store the search result
      CUDAREALLOCIFSMALLER("&d_position_", &d_position_, sizeof(int64_t), 0);

      if (sizeof(ConnKeyT) == 8) { // 64 bit
        // perform search in an array of 64 bit unsigned integers
        // Find number of elements < val in a sorted array array[i+1]>=array[i]
        search_down<unsigned long long, 1024>
            <<<1, 1024>>>((unsigned long long *)conn_key_vect_[ib] + i_conn0,
                          n_block_conn, value, d_position_);
        DBGCUDASYNC;
      } else {
        // perform search in an array of 32 bit unsigned integers
        // Find number of elements < val in a sorted array array[i+1]>=array[i]
        search_down<unsigned int, 1024>
            <<<1, 1024>>>((unsigned int *)conn_key_vect_[ib] + i_conn0,
                          n_block_conn, value, d_position_);
        DBGCUDASYNC;
      }
      // Copy position from GPU to CPU memory
      gpuErrchk(cudaMemcpy(&h_position, d_position_, sizeof(int64_t),
                           cudaMemcpyDeviceToHost));
      // check if found
      if (h_position < n_block_conn) {
        ib1 = ib; // next partition search should start from current block,
                  // where current partition ends
        h_position += i_conn0;
        break;
      }
    }
    if (h_position < 0) {
      throw ngpu_exception("Search error in partitioning new connections in "
                           "_ConnectDistrubutedFixedIndegree");
    }
    int64_t part_conn1;
    int64_t n_new_conn;
    if (ib == new_n_block) { // last block passed
      // compute index of the last connection of the current partition + 1
      part_conn1 =
          (new_n_block - 1) * conn_block_size_ + i_conn0 + n_block_conn;
    } else { // not over the last block
      // compute index of the last connection of the current partition + 1
      part_conn1 = ib1 * conn_block_size_ + h_position;
    }
    // compute number of connections in current partition
    n_new_conn = part_conn1 - part_conn0;
    // Now we must subtract n_source_cumul[ish] and convert the source node
    // indexes to the ConnKeyT representation in all connections of all blocks
    // of the current partition To do this, we need to loop on all blocks of the
    // current partition

    // Loop on connection blocks where connections of the current partition are
    // stored int64_t conn_source_ids_offset = 0; uncomment only if needed
    // int64_t n_prev_conn = 0; uncomment only if needed
    int ib0 = (int)(part_conn0 / conn_block_size_);
    int ib01 = (int)((part_conn1 - 1) / conn_block_size_);

    for (int ib = ib0; ib <= ib01; ib++) {
      int64_t n_block_conn; // number of new connections in the current block of
                            // the loop
      int64_t i_conn0;      // index of first new connection in this block
      if (ib01 == ib0)      // all connections are in the same block
      {                     // all connections are in the same block
        i_conn0 = part_conn0 % conn_block_size_;
        n_block_conn = n_new_conn;
      } else if (ib ==
                 ib0) // first block of the loop, cannot be the last (see above)
      {               // first block
        i_conn0 = part_conn0 % conn_block_size_;
        n_block_conn = conn_block_size_ - i_conn0;
      } else if (ib == ib01) // last block of the loop, cannot be the first (see
                             // above)
      {                      // last block
        i_conn0 = 0;
        n_block_conn = (part_conn1 - 1) % conn_block_size_ + 1;
      } else // block is neither the first nor the last of the loop
      {
        i_conn0 = 0;
        n_block_conn = conn_block_size_;
      }
      if (n_block_conn > 0) {
        // Launch CUDA kernel that subtracts n_source_cumul[ish] from the source
        // node indexes in all connections of the current block
        subtractSourceOffsetKernel<ConnKeyT>
            <<<(n_block_conn + 1023) / 1024, 1024>>>(
                conn_key_vect_[ib] + i_conn0, n_block_conn,
                n_source_cumul[ish]);
        DBGCUDASYNC;
      }
    }
    // To run the test test_connect_distributed_fixed_indegree.sh uncomment the
    // following line: printf("TDFID %d, n_new_conn: %ld\n", this_host_,
    // n_new_conn);

    // Do a regular RemoteConnectSource with source host n. ish
    // using a new connection rule that uses already created connections
    // filled only with source node relative indexes and target node
    // indexes and fills them with weights, delays, syn_geoups, ports
    ConnSpec conn_spec(ASSIGNED_NODES, n_new_conn);
    // printf("this_host: %d\tish: %d\tn_source_arr[ish]: %d\n",  this_host_,
    // ish, n_source_arr[ish]);
    if ((double)n_new_conn >= (use_all_source_node_fact_ * n_source_arr[ish])) {
      conn_spec.use_all_remote_source_nodes_ = true;
    }
    double time_mark = getRealTime();
    ret = remoteConnectSource<T1, T2>(source_host_arr[ish], d_source_arr[ish],
                                      n_source_arr[ish], d_target, n_target,
                                      group_local_id, conn_spec, syn_spec);
    RemoteConnectSource_time_ += (getRealTime() - time_mark);
    if (ib == new_n_block || ret != 0) { // last block passed, exit loop
      break;
    }
    part_conn0 = part_conn1; // update index of the first connection of the next
                             // partition
  }

  freeNodeArrayFromDevice(d_target);
  for (int ish = 0; ish < n_source_host; ish++) {
    freeNodeArrayFromDevice(d_source_arr[ish]);
  }

  return ret;
}

#endif // REMOTECONNECTH
