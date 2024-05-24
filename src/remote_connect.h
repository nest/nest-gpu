
#ifndef REMOTECONNECTH
#define REMOTECONNECTH
// #include <cub/cub.cuh>
#include <vector>
// #include "nestgpu.h"
#include "connect.h"
#include "copass_sort.h"
#include "cuda_error.h"
// Arrays that map remote source nodes to local image nodes

// The map is organized in blocks having block size:
extern __constant__ uint node_map_block_size; // = 100000;

// number of elements in the map for each host group and for each source host in the group
// n_remote_source_node_map[group_local_id][i_host]
// with i_host = 0, ..., host_group_[group_local_id].size()-1 excluding this host itself
extern __device__ uint** n_remote_source_node_map;

// remote_source_node_map[group_local_id][i_host][i_block][i]
extern __device__ uint**** remote_source_node_map;

// image_node_map[group_local_id][i_host][i_block][i]
extern __device__ uint**** local_image_node_map;

// Arrays that map local source nodes to remote image nodes

// number of elements in the map for each target host
// n_local_source_node_map[i_target_host]
// with i_target_host = 0, ..., n_hosts-1 excluding this host itself
extern __device__ uint* n_local_source_node_map; // [n_hosts];

// local_source_node_map[i_target_host][i_block][i]
extern __device__ uint*** local_source_node_map;

extern __constant__ uint n_local_nodes; // number of local nodes

// device function that checks if an int value is in a sorted 2d-array
// assuming that the entries in the 2d-array are sorted.
// The 2d-array is divided in noncontiguous blocks of size block_size
__device__ bool
checkIfValueIsIn2DArr( uint value, uint** arr, uint n_elem, uint block_size, uint* i_block, uint* i_in_block );

template < class ConnKeyT >
// kernel that flags source nodes used in at least one new connection
// of a given block
__global__ void
setUsedSourceNodeKernel( ConnKeyT* conn_key_subarray, int64_t n_conn, uint* source_node_flag )
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if ( i_conn >= n_conn )
  {
    return;
  }
  inode_t i_source = getConnSource< ConnKeyT >( conn_key_subarray[ i_conn ] );
  // it is not necessary to use atomic operation. See:
  // https://stackoverflow.com/questions/8416374/several-threads-writing-the-same-value-in-the-same-global-memory-location
  // printf("i_conn: %ld\t i_source: %d\n", i_conn, i_source);

  source_node_flag[ i_source ] = 1;
}

// kernel that flags source nodes used in at least one new connection
// of a given block
__global__ void setUsedSourceNodeOnSourceHostKernel( inode_t* conn_source_ids, int64_t n_conn, uint* source_node_flag );

// kernel that fills the arrays of nodes actually used by new connections
template < class T >
__global__ void
getUsedSourceNodeIndexKernel( T source,
  uint n_source,
  uint* n_used_source_nodes,
  uint* source_node_flag,
  uint* u_source_node_idx,
  uint* i_source_arr )
{
  uint i_source = threadIdx.x + blockIdx.x * blockDim.x;
  if ( i_source >= n_source )
  {
    return;
  }
  // Count how many source_node_flag are true using atomic increase
  // on n_used_source_nodes
  if ( source_node_flag[ i_source ] != 0 )
  {
    uint pos = atomicAdd( n_used_source_nodes, 1 );
    u_source_node_idx[ pos ] = getNodeIndex( source, i_source );
    i_source_arr[ pos ] = i_source;
  }
}

// kernel that counts source nodes actually used in new connections
__global__ void countUsedSourceNodeKernel( uint n_source, uint* n_used_source_nodes, uint* source_node_flag );

// kernel that searches source node indexes in the map,
// and set local_node_index
template < class T >
__global__ void
setLocalNodeIndexKernel( T source,
  uint n_source,
  uint* source_node_flag,
  uint** node_map,
  uint** image_node_map,
  uint n_node_map,
  uint* local_node_index )
{
  uint i_source = threadIdx.x + blockIdx.x * blockDim.x;
  if ( i_source >= n_source )
  {
    return;
  }
  // Count how many source_node_flag are true using atomic increase
  // on n_used_source_nodes
  if ( source_node_flag[ i_source ] != 0 )
  {
    uint node_index = getNodeIndex( source, i_source );
    uint i_block;
    uint i_in_block;
    bool mapped = checkIfValueIsIn2DArr( node_index, node_map, n_node_map, node_map_block_size, &i_block, &i_in_block );
    if ( !mapped )
    {
      printf( "Error in setLocalNodeIndexKernel: node index not mapped\n" );
      return;
    }
    uint i_image_node = image_node_map[ i_block ][ i_in_block ];
    local_node_index[ i_source ] = i_image_node;
  }
}

// kernel that replaces the source node index
// in a new remote connection of a given block
// source_node[i_conn] with the value of the element pointed by the
// index itself in the array local_node_index
template < class ConnKeyT >
__global__ void
fixConnectionSourceNodeIndexesKernel( ConnKeyT* conn_key_subarray, int64_t n_conn, uint* local_node_index )
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if ( i_conn >= n_conn )
  {
    return;
  }
  uint i_source = getConnSource< ConnKeyT >( conn_key_subarray[ i_conn ] );
  uint i_delay = getConnDelay< ConnKeyT >( conn_key_subarray[ i_conn ] );
  uint new_i_source = local_node_index[ i_source ];

  setConnSource< ConnKeyT >( conn_key_subarray[ i_conn ], new_i_source );

  // printf("i_conn: %ld\t new_i_source: %d\n", i_conn, new_i_source);
}

// kernel that searches node indexes in map
// increase counter of mapped nodes
__global__ void searchNodeIndexInMapKernel( uint** node_map,
  uint n_node_map,
  uint* count_mapped, // i.e. *n_target_hosts for our application
  uint n_node );

// kernel that searches node indexes in map
// flags nodes not yet mapped and counts them
__global__ void searchNodeIndexNotInMapKernel( uint** node_map,
  uint n_node_map,
  uint* sorted_node_index,
  bool* node_to_map,
  uint* n_node_to_map,
  uint n_node );

// kernel that checks if nodes are already in map
// if not insert them in the map
// In the target host unmapped remote source nodes must be mapped
// to local nodes from n_nodes to n_nodes + n_node_to_map
__global__ void insertNodesInMapKernel( uint** node_map,
  uint** image_node_map,
  uint image_node_map_i0,
  uint old_n_node_map,
  uint* sorted_node_index,
  bool* node_to_map,
  uint* i_node_to_map,
  uint n_node );

template < class ConnKeyT, class ConnStructT >
__global__ void
addOffsetToExternalNodeIdsKernel( int64_t n_conn,
  ConnKeyT* conn_key_subarray,
  ConnStructT* conn_struct_subarray,
  uint i_image_node_0 )
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if ( i_conn >= n_conn )
  {
    return;
  }
  // uint target_port_syn = conn_subarray[i_conn].target_port_syn;
  // if (target_port_syn & (1 << (MaxPortSynNBits - 1))) {
  // target_port_syn = target_port_syn ^ (1 << (MaxPortSynNBits - 1));
  // conn_subarray[i_conn].target_port_syn = target_port_syn;
  // key_subarray[i_conn] += (i_image_node_0 << MaxPortSynNBits);
  uint remote_flag =
    getConnRemoteFlag< ConnKeyT, ConnStructT >( conn_key_subarray[ i_conn ], conn_struct_subarray[ i_conn ] );
  if ( remote_flag == 1 )
  {
    // IN THE FUTURE KEEP IT!!!!!!!!!!!!!!!!!!!!!!!!!!
    clearConnRemoteFlag< ConnKeyT, ConnStructT >( conn_key_subarray[ i_conn ], conn_struct_subarray[ i_conn ] );
    uint i_source = getConnSource< ConnKeyT >( conn_key_subarray[ i_conn ] );
    i_source += i_image_node_0;
    setConnSource< ConnKeyT >( conn_key_subarray[ i_conn ], i_source );
  }
}

__global__ void MapIndexToImageNodeKernel( uint n_hosts, uint* host_offset, uint* node_index );



// Allocate GPU memory for new remote-source-node-map blocks
template < class ConnKeyT, class ConnStructT >
int
ConnectionTemplate< ConnKeyT, ConnStructT >::allocRemoteSourceNodeMapBlocks(
  std::vector< uint* >& i_remote_src_node_map,
  std::vector< uint* >& i_local_img_node_map,
  uint new_n_block )
{
  // allocate new blocks if needed
  for ( uint ib = i_remote_src_node_map.size(); ib < new_n_block; ib++ )
  {
    uint* d_remote_src_node_blk_pt;
    uint* d_local_img_node_blk_pt;
    // allocate GPU memory for new blocks
    CUDAMALLOCCTRL( "&d_remote_src_node_blk_pt", &d_remote_src_node_blk_pt, node_map_block_size_ * sizeof( uint ) );
    CUDAMALLOCCTRL( "&d_local_img_node_blk_pt", &d_local_img_node_blk_pt, node_map_block_size_ * sizeof( uint ) );

    i_remote_src_node_map.push_back( d_remote_src_node_blk_pt );
    i_local_img_node_map.push_back( d_local_img_node_blk_pt );
  }

  return 0;
}

// Allocate GPU memory for new local-source-node-map blocks
template < class ConnKeyT, class ConnStructT >
int
ConnectionTemplate< ConnKeyT, ConnStructT >::allocLocalSourceNodeMapBlocks( std::vector< uint* >& i_local_src_node_map,
  uint new_n_block )
{
  // allocate new blocks if needed
  for ( uint ib = i_local_src_node_map.size(); ib < new_n_block; ib++ )
  {
    uint* d_local_src_node_blk_pt;
    // allocate GPU memory for new blocks
    CUDAMALLOCCTRL( "&d_local_src_node_blk_pt", &d_local_src_node_blk_pt, node_map_block_size_ * sizeof( uint ) );

    i_local_src_node_map.push_back( d_local_src_node_blk_pt );
  }

  return 0;
}

// Loop on all new connections and set source_node_flag[i_source]=true
template < class ConnKeyT, class ConnStructT >
int
ConnectionTemplate< ConnKeyT, ConnStructT >::setUsedSourceNodes( int64_t old_n_conn, uint* d_source_node_flag )
{
  int64_t n_new_conn = n_conn_ - old_n_conn; // number of new connections

  uint ib0 = ( uint ) ( old_n_conn / conn_block_size_ );      // first block index
  uint ib1 = ( uint ) ( ( n_conn_ - 1 ) / conn_block_size_ ); // last block
  for ( uint ib = ib0; ib <= ib1; ib++ )
  {                       // loop on blocks
    int64_t n_block_conn; // number of connections in a block
    int64_t i_conn0;      // index of first connection in a block
    if ( ib1 == ib0 )
    { // all connections are in the same block
      i_conn0 = old_n_conn % conn_block_size_;
      n_block_conn = n_new_conn;
    }
    else if ( ib == ib0 )
    { // first block
      i_conn0 = old_n_conn % conn_block_size_;
      n_block_conn = conn_block_size_ - i_conn0;
    }
    else if ( ib == ib1 )
    { // last block
      i_conn0 = 0;
      n_block_conn = ( n_conn_ - 1 ) % conn_block_size_ + 1;
    }
    else
    {
      i_conn0 = 0;
      n_block_conn = conn_block_size_;
    }

    setUsedSourceNodeKernel< ConnKeyT > <<< ( n_block_conn + 1023 ) / 1024, 1024 >>>(
      conn_key_vect_[ ib ] + i_conn0, n_block_conn, d_source_node_flag );
    CUDASYNC;
  }
  return 0;
}

// Loop on all new connections and set source_node_flag[i_source]=true
template < class ConnKeyT, class ConnStructT >
int
ConnectionTemplate< ConnKeyT, ConnStructT >::setUsedSourceNodesOnSourceHost( int64_t old_n_conn,
  uint* d_source_node_flag )
{
  int64_t n_new_conn = n_conn_ - old_n_conn; // number of new connections

  setUsedSourceNodeOnSourceHostKernel<<< ( n_new_conn + 1023 ) / 1024, 1024 >>>(
    d_conn_source_ids_, n_new_conn, d_source_node_flag );
  CUDASYNC;

  return 0;
}

__global__ void setTargetHostArrayNodePointersKernel( uint* target_host_array,
  uint* target_host_i_map,
  uint* n_target_hosts_cumul,
  uint** node_target_hosts,
  uint** node_target_host_i_map,
  uint n_nodes );

// kernel that fills the arrays target_host_array
// and target_host_i_map using the node map
__global__ void fillTargetHostArrayFromMapKernel( uint** node_map,
  uint n_node_map,
  uint* count_mapped,
  uint** node_target_hosts,
  uint** node_target_host_i_map,
  uint n_nodes,
  uint i_target_host );

__global__ void addOffsetToImageNodeMapKernel( uint group_local_id, uint gi_host, uint n_node_map, uint i_image_node_0 );




// Initialize the maps
template < class ConnKeyT, class ConnStructT >
int
ConnectionTemplate< ConnKeyT, ConnStructT >::remoteConnectionMapInit()
{
  node_map_block_size_ = 10000; // initialize node map block size

  cudaMemcpyToSymbol( node_map_block_size, &node_map_block_size_, sizeof( uint ) );

  // number of host groups
  uint nhg = host_group_.size();
  // first evaluate how many elements must be allocated
  uint elem_to_alloc = 0;
  for (uint group_local_id=0; group_local_id<nhg; group_local_id++) {
    uint nh = host_group_[group_local_id].size(); // number of hosts in the group
    elem_to_alloc += nh;
  }
  uint *d_n_node_map;
  // allocate the whole array and initialize each element to 0
  CUDAMALLOCCTRL( "&d_n_node_map", &d_n_node_map, elem_to_alloc * sizeof( uint ) );
  gpuErrchk( cudaMemset( d_n_node_map, 0, elem_to_alloc * sizeof( uint ) ) );
  // now assign the proper address to each element of the array
  d_n_remote_source_node_map_.push_back(d_n_node_map);
  for (uint group_local_id=1; group_local_id<nhg; group_local_id++) {
    uint nh = host_group_[group_local_id - 1].size(); // number of hosts in the group
    d_n_remote_source_node_map_.push_back(d_n_remote_source_node_map_[group_local_id - 1] + nh);
  }

  // allocate and init to 0 n. of elements in the map for each source host  
  CUDAMALLOCCTRL( "&d_n_local_source_node_map_", &d_n_local_source_node_map_, n_hosts_ * sizeof( uint ) );
  gpuErrchk( cudaMemset( d_n_local_source_node_map_, 0, n_hosts_ * sizeof( uint ) ) );

  // initialize maps
  for (uint group_local_id=0; group_local_id<nhg; group_local_id++) {
    uint nh = host_group_[group_local_id].size(); // number of hosts in the group
    std::vector< std::vector< uint* > > rsn_map;
    std::vector< std::vector< uint* > > lin_map;
    for ( uint ih = 0; ih < nh; ih++ ) {
      std::vector< uint* > rsn_ih_map;
      std::vector< uint* > lin_ih_map;
      rsn_map.push_back( rsn_ih_map );
      lin_map.push_back( lin_ih_map );
    }
    h_remote_source_node_map_.push_back( rsn_map );
    h_image_node_map_.push_back( lin_map );
    
    hc_remote_source_node_map_.push_back( std::vector< std::vector< uint > > (nh, std::vector< uint >() ) );
    hc_image_node_map_.push_back( std::vector< std::vector< uint > > (nh, std::vector< uint >() ) );

  }

  for ( int i_host = 0; i_host < n_hosts_; i_host++ )
  {
    std::vector< uint* > lsn_map;
    h_local_source_node_map_.push_back( lsn_map );
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
template < class ConnKeyT, class ConnStructT >
int
ConnectionTemplate< ConnKeyT, ConnStructT >::fixConnectionSourceNodeIndexes( int64_t old_n_conn,
  uint* d_local_node_index )
{
  int64_t n_new_conn = n_conn_ - old_n_conn; // number of new connections

  uint ib0 = ( uint ) ( old_n_conn / conn_block_size_ );      // first block index
  uint ib1 = ( uint ) ( ( n_conn_ - 1 ) / conn_block_size_ ); // last block
  for ( uint ib = ib0; ib <= ib1; ib++ )
  {                       // loop on blocks
    int64_t n_block_conn; // number of connections in a block
    int64_t i_conn0;      // index of first connection in a block
    if ( ib1 == ib0 )
    { // all connections are in the same block
      i_conn0 = old_n_conn % conn_block_size_;
      n_block_conn = n_new_conn;
    }
    else if ( ib == ib0 )
    { // first block
      i_conn0 = old_n_conn % conn_block_size_;
      n_block_conn = conn_block_size_ - i_conn0;
    }
    else if ( ib == ib1 )
    { // last block
      i_conn0 = 0;
      n_block_conn = ( n_conn_ - 1 ) % conn_block_size_ + 1;
    }
    else
    {
      i_conn0 = 0;
      n_block_conn = conn_block_size_;
    }

    fixConnectionSourceNodeIndexesKernel< ConnKeyT > <<< ( n_block_conn + 1023 ) / 1024, 1024 >>>(
      conn_key_vect_[ ib ] + i_conn0, n_block_conn, d_local_node_index );
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }
  return 0;
}

// Calibrate the maps
template < class ConnKeyT, class ConnStructT >
int
ConnectionTemplate< ConnKeyT, ConnStructT >::remoteConnectionMapCalibrate( inode_t n_nodes )
{
  //  vector of pointers to local source node maps in device memory
  //  per target host hd_local_source_node_map[target_host]
  //  type std::vector<uint*>
  //  set its size and initialize to NULL
  hd_local_source_node_map_.resize( n_hosts_, NULL );
  // number of elements in each local source node map
  // h_n_local_source_node_map[target_host]
  // set its size and initialize to 0
  h_n_local_source_node_map_.resize( n_hosts_, 0 );

  uint nhg = host_group_.size(); // number of local host groups
  for (uint group_local_id=0; group_local_id<nhg; group_local_id++) {
    uint nh = host_group_[group_local_id].size(); // number of hosts in the group
    // vector of vectors of pointers to local image node maps in device memory
    // per local host group and per source host hd_image_node_map[group_local_id][i_host]
    // type std::vector< std::vector < uint** > > 
    // set its size and initialize to NULL
    std::vector < uint** > inm( nh, NULL );
    hd_image_node_map_.push_back(inm);
    
    // number of elements in each remote-source-node->local-image-node map
    // h_n_remote_source_node_map[group_local_id][i_host]
    // set its size and initialize to 0
    std::vector < uint > n_nm( nh, 0 );
    h_n_remote_source_node_map_.push_back(n_nm);
  }
  // loop on target hosts, skip self host
  for ( int tg_host = 0; tg_host < n_hosts_; tg_host++ )
  {
    if ( tg_host != this_host_ )
    {
      // get number of elements in each map from device memory
      uint n_node_map;
      gpuErrchk(
        cudaMemcpy( &n_node_map, &d_n_local_source_node_map_[ tg_host ], sizeof( uint ), cudaMemcpyDeviceToHost ) );
      // put it in h_n_local_source_node_map[tg_host]
      h_n_local_source_node_map_[ tg_host ] = n_node_map;
      // Allocate array of local source node map blocks
      // and copy their address from host to device
      hd_local_source_node_map_[ tg_host ] = NULL;
      uint n_blocks = h_local_source_node_map_[ tg_host ].size();
      if ( n_blocks > 0 )
      {
        CUDAMALLOCCTRL(
          "&hd_local_source_node_map[tg_host]", &hd_local_source_node_map_[ tg_host ], n_blocks * sizeof( uint* ) );
        gpuErrchk( cudaMemcpy( hd_local_source_node_map_[ tg_host ],
          &h_local_source_node_map_[ tg_host ][ 0 ],
          n_blocks * sizeof( uint* ),
          cudaMemcpyHostToDevice ) );
      }
    }
  }
  // allocate d_local_source_node_map and copy it from host to device
  CUDAMALLOCCTRL( "&d_local_source_node_map", &d_local_source_node_map_, n_hosts_ * sizeof( uint** ) );
  gpuErrchk( cudaMemcpy(
    d_local_source_node_map_, &hd_local_source_node_map_[ 0 ], n_hosts_ * sizeof( uint** ), cudaMemcpyHostToDevice ) );
  gpuErrchk( cudaMemcpyToSymbol( local_source_node_map, &d_local_source_node_map_, sizeof( uint*** ) ) );
  
  hdd_image_node_map_.resize(nhg, NULL);
  
  for (uint group_local_id=0; group_local_id<nhg; group_local_id++) { // loop on local host groups
    uint nh = host_group_[group_local_id].size(); // number of hosts in the group
    for ( uint gi_host = 0; gi_host < nh; gi_host++ ) {// loop on hosts
      int src_host = host_group_[group_local_id][gi_host];      
      if ( src_host != this_host_ ) { // skip self host
	// get number of elements in each map from device memory
	uint n_node_map;
	gpuErrchk(cudaMemcpy( &n_node_map, &d_n_remote_source_node_map_[group_local_id][gi_host],
			      sizeof( uint ), cudaMemcpyDeviceToHost ) );
	// put it in h_n_remote_source_node_map[src_host]
	h_n_remote_source_node_map_[group_local_id][gi_host] = n_node_map;
	// Allocate array of local image node map blocks
	// and copy their address from host to device
	uint n_blocks = h_image_node_map_[group_local_id][gi_host].size();
	hd_image_node_map_[group_local_id][gi_host] = NULL;
	if ( n_blocks > 0 ) {
	  CUDAMALLOCCTRL( "&hd_image_node_map_[group_local_id][gi_host]",
			  &hd_image_node_map_[group_local_id][gi_host],
			  n_blocks * sizeof( uint* ) );
	  gpuErrchk( cudaMemcpy( hd_image_node_map_[group_local_id][gi_host],
				 &h_image_node_map_[group_local_id][gi_host][ 0 ],
				 n_blocks * sizeof( uint* ),
				 cudaMemcpyHostToDevice ) );
	}
      }
    }

    if ( nh > 0 ) {
      CUDAMALLOCCTRL( "&hdd_image_node_map_[group_local_id]", &hdd_image_node_map_[group_local_id],
		      nh * sizeof( uint** ) );
      gpuErrchk( cudaMemcpy( hdd_image_node_map_[group_local_id],
			     &hd_image_node_map_[group_local_id][ 0 ],
			     nh * sizeof( uint** ),
			     cudaMemcpyHostToDevice ) );
    }
  }

  // allocate d_image_node_map and copy it from host to device
  CUDAMALLOCCTRL( "&d_image_node_map_", &d_image_node_map_, nhg * sizeof( uint*** ) );
  gpuErrchk( cudaMemcpy( d_image_node_map_,
    &hdd_image_node_map_[ 0 ],
    nhg * sizeof( uint*** ),
    cudaMemcpyHostToDevice ) );
  gpuErrchk( cudaMemcpyToSymbol( local_image_node_map, &d_image_node_map_, sizeof( uint**** ) ) );

  // uint n_nodes = GetNLocalNodes(); // number of nodes
  //  n_target_hosts[i_node] is the number of remote target hosts
  //  on which each local node
  //  has outgoing connections
  //  allocate d_n_target_hosts[n_nodes] and init to 0
  //  std::cout << "allocate d_n_target_hosts n_nodes: " << n_nodes << "\n";
  CUDAMALLOCCTRL( "&d_n_target_hosts_", &d_n_target_hosts_, n_nodes * sizeof( uint ) );
  // std::cout << "d_n_target_hosts: " << d_n_target_hosts_ << "\n";
  gpuErrchk( cudaMemset( d_n_target_hosts_, 0, n_nodes * sizeof( uint ) ) );
  // allocate d_n_target_hosts_cumul[n_nodes+1]
  // representing the prefix scan (cumulative sum) of d_n_target_hosts
  CUDAMALLOCCTRL( "&d_n_target_hosts_cumul_", &d_n_target_hosts_cumul_, ( n_nodes + 1 ) * sizeof( uint ) );

  // For each local node, count the number of remote target hosts
  // on which it has outgoing connections, i.e. n_target_hosts[i_node]
  // Loop on target hosts
  for ( int tg_host = 0; tg_host < n_hosts_; tg_host++ )
  {
    if ( tg_host != this_host_ )
    {
      uint** d_node_map = hd_local_source_node_map_[ tg_host ];
      uint n_node_map = h_n_local_source_node_map_[ tg_host ];
      // Launch kernel that searches each node in the map
      // of local source nodes having outgoing connections to target host
      // if found, increase n_target_hosts[i_node]
      searchNodeIndexInMapKernel<<< ( n_nodes + 1023 ) / 1024, 1024 >>>(
        d_node_map, n_node_map, d_n_target_hosts_, n_nodes );
      gpuErrchk( cudaPeekAtLastError() );
      gpuErrchk( cudaDeviceSynchronize() );
    }
  }

  //////////////////////////////////////////////////////////////////////
  // Evaluate exclusive sum of reverse connections per target node
  // Determine temporary device storage requirements
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  //<BEGIN-CLANG-TIDY-SKIP>//
  cub::DeviceScan::ExclusiveSum(
    d_temp_storage, temp_storage_bytes, d_n_target_hosts_, d_n_target_hosts_cumul_, n_nodes + 1 );
  //<END-CLANG-TIDY-SKIP>//

  // Allocate temporary storage
  CUDAMALLOCCTRL( "&d_temp_storage", &d_temp_storage, temp_storage_bytes );
  // Run exclusive prefix sum
  //<BEGIN-CLANG-TIDY-SKIP>//
  cub::DeviceScan::ExclusiveSum(
    d_temp_storage, temp_storage_bytes, d_n_target_hosts_, d_n_target_hosts_cumul_, n_nodes + 1 );
  //<END-CLANG-TIDY-SKIP>//

  CUDAFREECTRL( "d_temp_storage", d_temp_storage );
  // The last element is the sum of all elements of n_target_hosts
  uint n_target_hosts_sum;
  gpuErrchk(
    cudaMemcpy( &n_target_hosts_sum, &d_n_target_hosts_cumul_[ n_nodes ], sizeof( uint ), cudaMemcpyDeviceToHost ) );

  //////////////////////////////////////////////////////////////////////
  // allocate global array with remote target hosts of all nodes
  CUDAMALLOCCTRL( "&d_target_host_array_", &d_target_host_array_, n_target_hosts_sum * sizeof( uint ) );
  // allocate global array with remote target hosts map index
  CUDAMALLOCCTRL( "&d_target_host_i_map_", &d_target_host_i_map_, n_target_hosts_sum * sizeof( uint ) );
  // allocate array of pointers to the starting position in target_host array
  // of the target hosts for each node
  CUDAMALLOCCTRL( "&d_node_target_hosts_", &d_node_target_hosts_, n_nodes * sizeof( uint* ) );
  // allocate array of pointers to the starting position in target_host_i_map
  // of the target hosts map indexes for each node
  CUDAMALLOCCTRL( "&d_node_target_host_i_map_", &d_node_target_host_i_map_, n_nodes * sizeof( uint* ) );
  // Launch kernel to evaluate the pointers d_node_target_hosts
  // and d_node_target_host_i_map from the positions in target_host_array
  // given by  n_target_hosts_cumul
  setTargetHostArrayNodePointersKernel<<< ( n_nodes + 1023 ) / 1024, 1024 >>>( d_target_host_array_,
    d_target_host_i_map_,
    d_n_target_hosts_cumul_,
    d_node_target_hosts_,
    d_node_target_host_i_map_,
    n_nodes );
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  // reset to 0 d_n_target_hosts[n_nodes] to reuse it in the next kernel
  gpuErrchk( cudaMemset( d_n_target_hosts_, 0, n_nodes * sizeof( uint ) ) );

  // Loop on target hosts
  for ( int tg_host = 0; tg_host < n_hosts_; tg_host++ )
  {
    if ( tg_host != this_host_ )
    {
      uint** d_node_map = hd_local_source_node_map_[ tg_host ];
      uint n_node_map = h_n_local_source_node_map_[ tg_host ];
      // Launch kernel to fill the arrays target_host_array
      // and target_host_i_map using the node map
      fillTargetHostArrayFromMapKernel<<< ( n_nodes + 1023 ) / 1024, 1024 >>>(
        d_node_map, n_node_map, d_n_target_hosts_, d_node_target_hosts_, d_node_target_host_i_map_, n_nodes, tg_host );
      gpuErrchk( cudaPeekAtLastError() );
      gpuErrchk( cudaDeviceSynchronize() );
    }
  }

  addOffsetToImageNodeMap( n_nodes );

  node_target_host_group_.resize(n_nodes);
  for (uint group_local_id=1; group_local_id<nhg; group_local_id++) {
    host_group_local_source_node_map_[group_local_id].resize(n_nodes);
    uint nh = host_group_[group_local_id].size(); // number of hosts in the group
    for ( uint gi_host = 0; gi_host < nh; gi_host++ ) {// loop on hosts
      uint n_src = host_group_source_node_[group_local_id][gi_host].size();
      host_group_source_node_vect_[group_local_id][gi_host].resize(n_src);
      std::copy(host_group_source_node_[group_local_id][gi_host].begin(), host_group_source_node_[group_local_id][gi_host].end(),
		host_group_source_node_vect_[group_local_id][gi_host].begin());
      
      host_group_local_node_index_[group_local_id][gi_host].resize(n_src);
      std::fill(host_group_local_node_index_[group_local_id][gi_host].begin(), host_group_local_node_index_[group_local_id][gi_host].end(), -1);
      int src_host = host_group_[group_local_id][gi_host];
      if ( src_host != this_host_ ) { // skip self host
	// get number of elements in the map
	uint n_node_map;
	gpuErrchk(
		  cudaMemcpy( &n_node_map, &d_n_remote_source_node_map_[group_local_id][ gi_host ], sizeof( uint ), cudaMemcpyDeviceToHost ) );

	if (n_node_map > 0) {
	  hc_remote_source_node_map_[group_local_id][gi_host].resize(n_node_map);
	  hc_image_node_map_[group_local_id][gi_host].resize(n_node_map);
	  // loop on remote-source-node-to-local-image-node map blocks
	  uint n_map_blocks =  h_remote_source_node_map_[group_local_id][gi_host].size();
	  for (uint ib=0; ib<n_map_blocks; ib++) {
	    uint n_elem;
	    if (ib<n_map_blocks-1) {
	      n_elem = node_map_block_size_;
	    }
	    else {
	      n_elem = (n_node_map - 1) % node_map_block_size_ + 1;
	    }
	    gpuErrchk(cudaMemcpy(&hc_remote_source_node_map_[group_local_id][gi_host][ib*node_map_block_size_],
				 h_remote_source_node_map_[group_local_id][gi_host][ib], n_elem*sizeof(uint), cudaMemcpyDeviceToHost ));
	    gpuErrchk(cudaMemcpy(&hc_image_node_map_[group_local_id][gi_host][ib*node_map_block_size_],
				 h_image_node_map_[group_local_id][gi_host][ib], n_elem*sizeof(uint), cudaMemcpyDeviceToHost ));
	  }
       
	  for (uint i=0; i<n_node_map; i++) {
	    inode_t src_node = hc_remote_source_node_map_[group_local_id][gi_host][i];
	    auto it = std::find(host_group_source_node_vect_[group_local_id][gi_host].begin(),
				host_group_source_node_vect_[group_local_id][gi_host].end(), src_node);
	    if (it == host_group_source_node_vect_[group_local_id][gi_host].end()) {
	      throw ngpu_exception( "source node not found in host map" );
	    }
	    inode_t pos = it - host_group_source_node_vect_[group_local_id][gi_host].begin();
	    host_group_local_node_index_[group_local_id][gi_host][pos] = hc_image_node_map_[group_local_id][gi_host][i];
	  }
	}

	gpuErrchk(
		  cudaMemcpy( &n_node_map, &d_n_remote_source_node_map_[group_local_id][ gi_host ], sizeof( uint ), cudaMemcpyDeviceToHost ) );
	
      }
      else { // only in the source, i.e. if src_host == this_host_       
	for (uint i=0; i<n_src; i++) {
	  inode_t i_source = host_group_source_node_vect_[group_local_id][gi_host][i];
	  host_group_local_source_node_map_[group_local_id][i_source] = i;
	  node_target_host_group_[i_source].insert(group_local_id);	  
	}
      }
    }
  }

  return 0;
}

template < class ConnKeyT, class ConnStructT >
int
ConnectionTemplate< ConnKeyT, ConnStructT >::addOffsetToImageNodeMap( inode_t n_nodes )
{
  uint i_image_node_0 = n_nodes;

  uint nhg = host_group_.size(); // number of local host groups
  for (uint group_local_id=0; group_local_id<nhg; group_local_id++) {
    uint nh = host_group_[group_local_id].size(); // number of hosts in the group
    for ( uint gi_host = 0; gi_host < nh; gi_host++ ) {// loop on hosts
      int src_host = host_group_[group_local_id][gi_host];      
      if ( src_host != this_host_ ) { // skip self host
	uint n_node_map = h_n_remote_source_node_map_[group_local_id][gi_host];
	if ( n_node_map > 0 ) {
	  addOffsetToImageNodeMapKernel<<< ( n_node_map + 1023 ) / 1024, 1024 >>>( group_local_id, gi_host, n_node_map, i_image_node_0 );
	  gpuErrchk( cudaPeekAtLastError() );
	  gpuErrchk( cudaDeviceSynchronize() );
	}
      }
    }
  }

  return 0;
}

// REMOTE CONNECT FUNCTION
template < class ConnKeyT, class ConnStructT >
template < class T1, class T2 >
int
ConnectionTemplate< ConnKeyT, ConnStructT >::_RemoteConnect( int source_host,
  T1 h_source,
  inode_t n_source,
  int target_host,
  T2 h_target,
  inode_t n_target,
  int i_host_group,
  ConnSpec& conn_spec,
  SynSpec& syn_spec )
{
  if (first_connection_flag_ == true) {
    remoteConnectionMapInit();
    first_connection_flag_ = false;
  }

  if ( source_host >= n_hosts_ )
  {
    throw ngpu_exception( "Source host index out of range in _RemoteConnect" );
  }
  if ( target_host >= n_hosts_ )
  {
    throw ngpu_exception( "Target host index out of range in _RemoteConnect" );
  }
  if ( this_host_ >= n_hosts_ )
  {
    throw ngpu_exception( "this_host index out of range in _RemoteConnect" );
  }
  // semplificare i free. Forse si può spostare tutto in Connect? 
  T1 d_source = copyNodeArrayToDevice( h_source, n_source );
  T2 d_target = copyNodeArrayToDevice( h_target, n_target );

  // Check if it is a local connection
  if ( this_host_ == source_host && source_host == target_host )
  {
    int ret = _Connect< T1, T2 >( d_source, n_source, d_target, n_target, conn_spec, syn_spec );
    freeNodeArrayFromDevice(d_source);
    freeNodeArrayFromDevice(d_target);
    return ret;
  }

  // i_host_group is the global host-group index, given as optional argument to the
  // RemoteConnect command. Default value is either -1 or 0 (initialized as kernel parameter).
  // i_host_group = -1 for point-to-point MPI communication
  //              > 0 for other host groups
  //             ??? DESIGN DECISION = 0 for the world group, which includes all hosts (i.e. all MPI processes)
  int group_local_id = 0;
  int i_host = 0;
  if (i_host_group>=0) { // not a point-to-point MPI communication
    group_local_id = host_group_local_id_[i_host_group];
    if (group_local_id >= 0) { // this host is in group
      // find the source host index in the host group
      auto it = std::find(host_group_[group_local_id].begin(), host_group_[group_local_id].end(), source_host);
      if (it == host_group_[group_local_id].end()) {
	throw ngpu_exception( "source host not found in host group" );
      }
      i_host = it - host_group_[group_local_id].begin();

      for (inode_t i=0; i<n_source; i++) {
	inode_t i_source = hGetNodeIndex(h_source, i);
	host_group_source_node_[group_local_id][i_host].insert(i_source);
      }
    }
  }
  // if i_host_group<0, i.e. a point-to-point MPI communication is required
  // and this host is the source (but it is not a local connection) call RemoteConnectTarget
  // Check if source_host matches this_host
  else {
    //p2p_flag_ = true;
    p2p_host_conn_matrix_[source_host][target_host] = true;
    if (this_host_ == source_host) {
      int ret = remoteConnectTarget< T1, T2 >( target_host, d_source, n_source, d_target, n_target, conn_spec, syn_spec );
      freeNodeArrayFromDevice(d_source);
      freeNodeArrayFromDevice(d_target);

      return ret;
    }
  }
  // Check if target_host matches this_host
  if (this_host_ == target_host) {
    if (group_local_id < 0) {
      throw ngpu_exception( "target host is not in host group" );
    }

    int ret = remoteConnectSource< T1, T2 >( source_host, d_source, n_source, d_target, n_target, group_local_id, conn_spec, syn_spec );

    freeNodeArrayFromDevice(d_source);
    freeNodeArrayFromDevice(d_target);
    return ret;
  }

  freeNodeArrayFromDevice(d_source);
  freeNodeArrayFromDevice(d_target);

  return 0;
}

template < class ConnKeyT, class ConnStructT >
int
ConnectionTemplate< ConnKeyT, ConnStructT >::addOffsetToExternalNodeIds( uint n_local_nodes )
{
  uint n_blocks = ( n_conn_ - 1 ) / conn_block_size_ + 1;
  // uint i_image_node_0 = getNLocalNodes();
  uint i_image_node_0 = n_local_nodes;

  for ( uint ib = 0; ib < n_blocks; ib++ )
  {
    int64_t n_block_conn = conn_block_size_; // number of connections in the block
    if ( ib == n_blocks - 1 )
    { // last block
      n_block_conn = ( n_conn_ - 1 ) % conn_block_size_ + 1;
    }
    addOffsetToExternalNodeIdsKernel< ConnKeyT, ConnStructT > <<< ( n_block_conn + 1023 ) / 1024, 1024 >>>(
      n_block_conn, conn_key_vect_[ ib ], ( ConnStructT* ) conn_struct_vect_[ ib ], i_image_node_0 );
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }

  return 0;
}

// REMOTE CONNECT FUNCTION for target_host matching this_host
template < class ConnKeyT, class ConnStructT >
template < class T1, class T2 >
int
ConnectionTemplate< ConnKeyT, ConnStructT >::remoteConnectSource( int source_host,
  T1 source,
  inode_t n_source,
  T2 target,
  inode_t n_target,
  int group_local_id,
  ConnSpec& conn_spec,
  SynSpec& syn_spec )
{
  // n_nodes will be the first index for new mapping of remote source nodes
  // to local image nodes
  // int image_node_map_i0 = GetNNode();
  uint image_node_map_i0 = n_image_nodes_;
  // syn_spec.port_ = syn_spec.port_ |
  //   (1 << (h_MaxPortSynNBits - max_syn_nbits_ - 1));
  syn_spec.syn_group_ = syn_spec.syn_group_ | ( 1 << max_syn_nbits_ );

  // check if the flag UseAllSourceNodes[conn_rule] is false
  // if (!use_all_source_nodes_flag) {

  // on both the source and target hosts create a temporary array
  // of booleans having size equal to the number of source nodes

  uint* d_source_node_flag; // [n_source] // each element is initially false
  CUDAMALLOCCTRL( "&d_source_node_flag", &d_source_node_flag, n_source * sizeof( uint ) );
  gpuErrchk( cudaMemset( d_source_node_flag, 0, n_source * sizeof( uint ) ) );

  // on the target hosts create a temporary array of integers having size
  // equal to the number of source nodes

  uint* d_local_node_index; // [n_source]; // only on target host
  CUDAMALLOCCTRL( "&d_local_node_index", &d_local_node_index, n_source * sizeof( uint ) );

  int64_t old_n_conn = n_conn_;
  // The connect command is performed on both source and target host using
  // the same initial seed and using as source node indexes the integers
  // from 0 to n_source_nodes - 1
  _Connect< inode_t, T2 >(
    conn_random_generator_[ source_host ][ this_host_ ], 0, n_source, target, n_target, conn_spec, syn_spec, false );
  if ( n_conn_ == old_n_conn )
  {
    return 0;
  }

  // flag source nodes used in at least one new connection
  // Loop on all new connections and set source_node_flag[i_source]=true
  setUsedSourceNodes( old_n_conn, d_source_node_flag );

  // Count source nodes actually used in new connections
  // Allocate n_used_source_nodes and initialize it to 0
  uint* d_n_used_source_nodes;
  CUDAMALLOCCTRL( "&d_n_used_source_nodes", &d_n_used_source_nodes, sizeof( uint ) );
  gpuErrchk( cudaMemset( d_n_used_source_nodes, 0, sizeof( uint ) ) );
  // Launch kernel to count used nodes
  countUsedSourceNodeKernel<<< ( n_source + 1023 ) / 1024, 1024 >>>(
    n_source, d_n_used_source_nodes, d_source_node_flag );
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  // copy result from GPU to CPU memory
  uint n_used_source_nodes;
  gpuErrchk( cudaMemcpy( &n_used_source_nodes, d_n_used_source_nodes, sizeof( uint ), cudaMemcpyDeviceToHost ) );

  // Define and allocate arrays of size n_used_source_nodes
  uint* d_unsorted_source_node_index; // [n_used_source_nodes];
  uint* d_sorted_source_node_index;   // [n_used_source_nodes];
  // i_source_arr are the positions in the arrays source_node_flag
  // and local_node_index
  uint* d_i_unsorted_source_arr;          // [n_used_source_nodes];
  uint* d_i_sorted_source_arr;            // [n_used_source_nodes];
  bool* d_source_node_index_to_be_mapped; //[n_used_source_nodes]; // initially
                                          // false
  CUDAMALLOCCTRL(
    "&d_unsorted_source_node_index", &d_unsorted_source_node_index, n_used_source_nodes * sizeof( uint ) );
  CUDAMALLOCCTRL( "&d_sorted_source_node_index", &d_sorted_source_node_index, n_used_source_nodes * sizeof( uint ) );
  CUDAMALLOCCTRL( "&d_i_unsorted_source_arr", &d_i_unsorted_source_arr, n_used_source_nodes * sizeof( uint ) );
  CUDAMALLOCCTRL( "&d_i_sorted_source_arr", &d_i_sorted_source_arr, n_used_source_nodes * sizeof( uint ) );
  CUDAMALLOCCTRL(
    "&d_source_node_index_to_be_mapped", &d_source_node_index_to_be_mapped, n_used_source_nodes * sizeof( int8_t ) );
  // source_node_index_to_be_mapped is initially false
  gpuErrchk( cudaMemset( d_source_node_index_to_be_mapped, 0, n_used_source_nodes * sizeof( int8_t ) ) );

  // Fill the arrays of nodes actually used by new connections
  // Reset n_used_source_nodes to 0
  gpuErrchk( cudaMemset( d_n_used_source_nodes, 0, sizeof( uint ) ) );
  // Launch kernel to fill the arrays
  getUsedSourceNodeIndexKernel<<< ( n_source + 1023 ) / 1024, 1024 >>>( source,
    n_source,
    d_n_used_source_nodes,
    d_source_node_flag,
    d_unsorted_source_node_index,
    d_i_unsorted_source_arr );
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  // Sort the arrays using unsorted_source_node_index as key
  // and i_source as value -> sorted_source_node_index

  // Determine temporary storage requirements for RadixSort
  void* d_sort_storage = NULL;
  size_t sort_storage_bytes = 0;
  //<BEGIN-CLANG-TIDY-SKIP>//
  cub::DeviceRadixSort::SortPairs( d_sort_storage,
    sort_storage_bytes,
    d_unsorted_source_node_index,
    d_sorted_source_node_index,
    d_i_unsorted_source_arr,
    d_i_sorted_source_arr,
    n_used_source_nodes );
  //<END-CLANG-TIDY-SKIP>//

  // Allocate temporary storage
  CUDAMALLOCCTRL( "&d_sort_storage", &d_sort_storage, sort_storage_bytes );

  // Run sorting operation
  //<BEGIN-CLANG-TIDY-SKIP>//
  cub::DeviceRadixSort::SortPairs( d_sort_storage,
    sort_storage_bytes,
    d_unsorted_source_node_index,
    d_sorted_source_node_index,
    d_i_unsorted_source_arr,
    d_i_sorted_source_arr,
    n_used_source_nodes );
  //<END-CLANG-TIDY-SKIP>//

  //////////////////////////////
  // Allocate array of remote source node map blocks
  // and copy their address from host to device
  //uint n_blocks = h_remote_source_node_map_[ source_host ].size();
  uint** d_node_map = NULL;
  uint** d_image_node_map = NULL;

  DBGCUDASYNC;
  int gi_host;
  //if ( group_local_id <= 1 ) { // point-to-point communication (0)  NOT FOR NOW and world group (1) include all hosts
  if ( group_local_id == 0 ) { // point-to-point communication (0) include all hosts
    gi_host = source_host;
  }
  else {
    // find the source host index in the host group
    auto it = std::find(host_group_[group_local_id].begin(), host_group_[group_local_id].end(), source_host);
    if (it == host_group_[group_local_id].end()) {
      throw ngpu_exception( "source host not found in host group" );
    }
    gi_host = it - host_group_[group_local_id].begin();
  }
  uint n_blocks = h_remote_source_node_map_[group_local_id][ gi_host ].size();
  // get current number of elements in the map
  uint n_node_map;
  gpuErrchk(
    cudaMemcpy( &n_node_map, &d_n_remote_source_node_map_[group_local_id][ gi_host ], sizeof( uint ), cudaMemcpyDeviceToHost ) );

  if ( n_blocks > 0 )
  {
    // check for consistency between number of elements
    // and number of blocks in the map
    uint tmp_n_blocks = ( n_node_map - 1 ) / node_map_block_size_ + 1;
    if ( tmp_n_blocks != n_blocks )
    {
      std::cerr << "Inconsistent number of elements " << n_node_map << " and number of blocks " << n_blocks
                << " in remote_source_node_map\n";
      std::cout << "group_local_id" << group_local_id << "\n";
      std::cout << "gi_host" << gi_host << "\n";
      
      //exit( -1 );
      throw ngpu_exception( "Error" );
    }
    CUDAMALLOCCTRL( "&d_node_map", &d_node_map, n_blocks * sizeof( uint* ) );
    gpuErrchk( cudaMemcpy( d_node_map,
      &h_remote_source_node_map_[group_local_id][gi_host][ 0 ],
      n_blocks * sizeof( uint* ),
      cudaMemcpyHostToDevice ) );
  }
  
  // Allocate boolean array for flagging remote source nodes not yet mapped
  // and initialize all elements to 0 (false)
  bool* d_node_to_map;
  CUDAMALLOCCTRL( "&d_node_to_map", &d_node_to_map, n_used_source_nodes * sizeof( bool ) );
  gpuErrchk( cudaMemset( d_node_to_map, 0, n_used_source_nodes * sizeof( bool ) ) );
  // Allocate number of nodes to be mapped and initialize it to 0
  uint* d_n_node_to_map;
  CUDAMALLOCCTRL( "&d_n_node_to_map", &d_n_node_to_map, sizeof( uint ) );
  gpuErrchk( cudaMemset( d_n_node_to_map, 0, sizeof( uint ) ) );

  // launch kernel that searches remote source nodes indexes not in the map,
  // flags the nodes not yet mapped and counts them
  searchNodeIndexNotInMapKernel<<< ( n_used_source_nodes + 1023 ) / 1024, 1024 >>>(
    d_node_map, n_node_map, d_sorted_source_node_index, d_node_to_map, d_n_node_to_map, n_used_source_nodes );
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  uint h_n_node_to_map;

  gpuErrchk( cudaMemcpy( &h_n_node_to_map, d_n_node_to_map, sizeof( uint ), cudaMemcpyDeviceToHost ) );

  // Check if new blocks are required for the map
  uint new_n_blocks = ( n_node_map + h_n_node_to_map - 1 ) / node_map_block_size_ + 1;

  // if new blocks are required for the map, allocate them
  if ( new_n_blocks != n_blocks )
  {
    // Allocate GPU memory for new remote-source-node-map blocks
    allocRemoteSourceNodeMapBlocks(
      h_remote_source_node_map_[group_local_id][gi_host], h_image_node_map_[group_local_id][gi_host], new_n_blocks );
    // free d_node_map
    if ( n_blocks > 0 )
    {
      CUDAFREECTRL( "d_node_map", d_node_map );
    }
    // update number of blocks in the map
    n_blocks = new_n_blocks;

    // reallocate d_node_map and get it from host
    CUDAMALLOCCTRL( "&d_node_map", &d_node_map, n_blocks * sizeof( uint* ) );
    gpuErrchk( cudaMemcpy( d_node_map,
      &h_remote_source_node_map_[group_local_id][gi_host][ 0 ],
      n_blocks * sizeof( uint* ),
      cudaMemcpyHostToDevice ) );
  }
  if ( n_blocks > 0 )
  {
    // allocate d_image_node_map and get it from host
    CUDAMALLOCCTRL( "&d_image_node_map", &d_image_node_map, n_blocks * sizeof( uint* ) );
    gpuErrchk( cudaMemcpy( d_image_node_map,
      &h_image_node_map_[group_local_id][gi_host][ 0 ],
      n_blocks * sizeof( uint* ),
      cudaMemcpyHostToDevice ) );
  }
  
  // Map the not-yet-mapped source nodes using a kernel
  // similar to the one used for counting
  // In the target host unmapped remote source nodes must be mapped
  // to local nodes from n_nodes to n_nodes + n_node_to_map

  // Allocate the index of the nodes to be mapped and initialize it to 0
  uint* d_i_node_to_map;
  CUDAMALLOCCTRL( "&d_i_node_to_map", &d_i_node_to_map, sizeof( uint ) );
  gpuErrchk( cudaMemset( d_i_node_to_map, 0, sizeof( uint ) ) );

  // launch kernel that checks if nodes are already in map
  // if not insert them in the map
  // In the target host, put in the map the pair:
  // (source_node_index, image_node_map_i0 + i_node_to_map)
  insertNodesInMapKernel<<< ( n_used_source_nodes + 1023 ) / 1024, 1024 >>>( d_node_map,
    d_image_node_map,
    image_node_map_i0,
    n_node_map,
    d_sorted_source_node_index,
    d_node_to_map,
    d_i_node_to_map,
    n_used_source_nodes );
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  
  // update number of elements in remote source node map
  n_node_map += h_n_node_to_map;
  gpuErrchk(
    cudaMemcpy( &d_n_remote_source_node_map_[group_local_id][gi_host], &n_node_map, sizeof( uint ), cudaMemcpyHostToDevice ) );

  // check for consistency between number of elements
  // and number of blocks in the map
  uint tmp_n_blocks = ( n_node_map - 1 ) / node_map_block_size_ + 1;
  if ( tmp_n_blocks != n_blocks )
  {
    std::cerr << "Inconsistent number of elements " << n_node_map << " and number of blocks " << n_blocks
              << " in remote_source_node_map\n";
    //exit( -1 );
    throw ngpu_exception( "Error" );
  }
  
  // Sort the WHOLE key-pair map source_node_map, image_node_map
  // using block sort algorithm copass_sort
  // typical usage:
  // copass_sort::sort<uint, value_struct>(key_subarray, value_subarray, n,
  //				       aux_size, d_storage, storage_bytes);
  // Determine temporary storage requirements for copass_sort
  int64_t storage_bytes = 0;
  void* d_storage = NULL;
  copass_sort::sort< uint, uint >( &h_remote_source_node_map_[group_local_id][gi_host][ 0 ],
    &h_image_node_map_[group_local_id][gi_host][ 0 ],
    n_node_map,
    node_map_block_size_,
    d_storage,
    storage_bytes );

  // Allocate temporary storage
  CUDAMALLOCCTRL( "&d_storage", &d_storage, storage_bytes );

  // Run sorting operation
  copass_sort::sort< uint, uint >( &h_remote_source_node_map_[group_local_id][gi_host][ 0 ],
    &h_image_node_map_[group_local_id][gi_host][ 0 ],
    n_node_map,
    node_map_block_size_,
    d_storage,
    storage_bytes );
  CUDAFREECTRL( "d_storage", d_storage );

  // Launch kernel that searches source node indexes in the map
  // and set corresponding values of local_node_index
  setLocalNodeIndexKernel<<< ( n_source + 1023 ) / 1024, 1024 >>>(
    source, n_source, d_source_node_flag, d_node_map, d_image_node_map, n_node_map, d_local_node_index );
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  // On target host. Loop on all new connections and replace
  // the source node index source_node[i_conn] with the value of the element
  // pointed by the index itself in the array local_node_index
  // source_node[i_conn] = local_node_index[source_node[i_conn]];

  // similar to setUsedSourceNodes
  // replace source_node_flag[i_source] with local_node_index[i_source]
  // clearly read it instead of writing on it!
  // setUsedSourceNodes(old_n_conn, d_source_node_flag);
  // becomes something like
  fixConnectionSourceNodeIndexes( old_n_conn, d_local_node_index );

  // On target host. Create n_nodes_to_map nodes of type image_node
  // std::cout << "h_n_node_to_map " << h_n_node_to_map <<"\n";
  if ( h_n_node_to_map > 0 )
  {
    //_Create("image_node", h_n_node_to_map);
    n_image_nodes_ += h_n_node_to_map;
    // std::cout << "n_image_nodes_ " << n_image_nodes_ <<"\n";
  }

  return 0;
}

// REMOTE CONNECT FUNCTION for source_host matching this_host
template < class ConnKeyT, class ConnStructT >
template < class T1, class T2 >
int
ConnectionTemplate< ConnKeyT, ConnStructT >::remoteConnectTarget( int target_host,
  T1 source,
  inode_t n_source,
  T2 target,
  inode_t n_target,
  ConnSpec& conn_spec,
  SynSpec& syn_spec )
{
  // check if the flag UseAllSourceNodes[conn_rule] is false
  // if (!use_all_source_nodes_flag) {

  // on both the source and target hosts create a temporary array
  // of booleans having size equal to the number of source nodes

  uint* d_source_node_flag; // [n_source] // each element is initially false
  CUDAMALLOCCTRL( "&d_source_node_flag", &d_source_node_flag, n_source * sizeof( uint ) );
  gpuErrchk( cudaMemset( d_source_node_flag, 0, n_source * sizeof( uint ) ) );

  int64_t old_n_conn = n_conn_;
  // The connect command is performed on both source and target host using
  // the same initial seed and using as source node indexes the integers
  // from 0 to n_source_nodes - 1
  _Connect< inode_t, T2 >(
    conn_random_generator_[ this_host_ ][ target_host ], 0, n_source, target, n_target, conn_spec, syn_spec, true );

  if ( n_conn_ == old_n_conn )
  {
    return 0;
  }

  // flag source nodes used in at least one new connection
  // Loop on all new connections and set source_node_flag[i_source]=true
  setUsedSourceNodesOnSourceHost( old_n_conn, d_source_node_flag );

  // Count source nodes actually used in new connections
  // Allocate n_used_source_nodes and initialize it to 0
  uint* d_n_used_source_nodes;
  CUDAMALLOCCTRL( "&d_n_used_source_nodes", &d_n_used_source_nodes, sizeof( uint ) );
  gpuErrchk( cudaMemset( d_n_used_source_nodes, 0, sizeof( uint ) ) );
  // Launch kernel to count used nodes
  countUsedSourceNodeKernel<<< ( n_source + 1023 ) / 1024, 1024 >>>(
    n_source, d_n_used_source_nodes, d_source_node_flag );
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  // copy result from GPU to CPU memory
  uint n_used_source_nodes;
  gpuErrchk( cudaMemcpy( &n_used_source_nodes, d_n_used_source_nodes, sizeof( uint ), cudaMemcpyDeviceToHost ) );

  // Define and allocate arrays of size n_used_source_nodes
  uint* d_unsorted_source_node_index; // [n_used_source_nodes];
  uint* d_sorted_source_node_index;   // [n_used_source_nodes];
  // i_source_arr are the positions in the arrays source_node_flag
  // and local_node_index
  uint* d_i_unsorted_source_arr;          // [n_used_source_nodes];
  uint* d_i_sorted_source_arr;            // [n_used_source_nodes];
  bool* d_source_node_index_to_be_mapped; //[n_used_source_nodes]; // initially
                                          // false
  CUDAMALLOCCTRL(
    "&d_unsorted_source_node_index", &d_unsorted_source_node_index, n_used_source_nodes * sizeof( uint ) );
  CUDAMALLOCCTRL( "&d_sorted_source_node_index", &d_sorted_source_node_index, n_used_source_nodes * sizeof( uint ) );
  CUDAMALLOCCTRL( "&d_i_unsorted_source_arr", &d_i_unsorted_source_arr, n_used_source_nodes * sizeof( uint ) );
  CUDAMALLOCCTRL( "&d_i_sorted_source_arr", &d_i_sorted_source_arr, n_used_source_nodes * sizeof( uint ) );
  CUDAMALLOCCTRL(
    "&d_source_node_index_to_be_mapped", &d_source_node_index_to_be_mapped, n_used_source_nodes * sizeof( int8_t ) );
  // source_node_index_to_be_mapped is initially false
  gpuErrchk( cudaMemset( d_source_node_index_to_be_mapped, 0, n_used_source_nodes * sizeof( int8_t ) ) );

  // Fill the arrays of nodes actually used by new connections
  // Reset n_used_source_nodes to 0
  gpuErrchk( cudaMemset( d_n_used_source_nodes, 0, sizeof( uint ) ) );
  // Launch kernel to fill the arrays
  getUsedSourceNodeIndexKernel<<< ( n_source + 1023 ) / 1024, 1024 >>>( source,
    n_source,
    d_n_used_source_nodes,
    d_source_node_flag,
    d_unsorted_source_node_index,
    d_i_unsorted_source_arr );
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  // Sort the arrays using unsorted_source_node_index as key
  // and i_source as value -> sorted_source_node_index

  // Determine temporary storage requirements for RadixSort
  void* d_sort_storage = NULL;
  size_t sort_storage_bytes = 0;
  //<BEGIN-CLANG-TIDY-SKIP>//
  cub::DeviceRadixSort::SortPairs( d_sort_storage,
    sort_storage_bytes,
    d_unsorted_source_node_index,
    d_sorted_source_node_index,
    d_i_unsorted_source_arr,
    d_i_sorted_source_arr,
    n_used_source_nodes );
  //<END-CLANG-TIDY-SKIP>//

  // Allocate temporary storage
  CUDAMALLOCCTRL( "&d_sort_storage", &d_sort_storage, sort_storage_bytes );

  // Run sorting operation
  //<BEGIN-CLANG-TIDY-SKIP>//
  cub::DeviceRadixSort::SortPairs( d_sort_storage,
    sort_storage_bytes,
    d_unsorted_source_node_index,
    d_sorted_source_node_index,
    d_i_unsorted_source_arr,
    d_i_sorted_source_arr,
    n_used_source_nodes );
  //<END-CLANG-TIDY-SKIP>//

  // !!!!!!!!!!!!!!!!
  // Allocate array of local source node map blocks
  // and copy their address from host to device
  uint n_blocks = h_local_source_node_map_[ target_host ].size();
  uint** d_node_map = NULL;
  // get current number of elements in the map
  uint n_node_map;
  gpuErrchk(
    cudaMemcpy( &n_node_map, &d_n_local_source_node_map_[ target_host ], sizeof( uint ), cudaMemcpyDeviceToHost ) );

  if ( n_blocks > 0 )
  {
    // check for consistency between number of elements
    // and number of blocks in the map
    uint tmp_n_blocks = ( n_node_map - 1 ) / node_map_block_size_ + 1;
    if ( tmp_n_blocks != n_blocks )
    {
      std::cerr << "Inconsistent number of elements " << n_node_map << " and number of blocks " << n_blocks
                << " in local_source_node_map\n";
      //exit( -1 );
      throw ngpu_exception( "Error" );
    }
    CUDAMALLOCCTRL( "&d_node_map", &d_node_map, n_blocks * sizeof( uint* ) );
    gpuErrchk( cudaMemcpy(
      d_node_map, &h_local_source_node_map_[ target_host ][ 0 ], n_blocks * sizeof( uint* ), cudaMemcpyHostToDevice ) );
  }

  // Allocate boolean array for flagging remote source nodes not yet mapped
  // and initialize all elements to 0 (false)
  bool* d_node_to_map;
  CUDAMALLOCCTRL( "&d_node_to_map", &d_node_to_map, n_used_source_nodes * sizeof( bool ) );
  gpuErrchk( cudaMemset( d_node_to_map, 0, n_used_source_nodes * sizeof( bool ) ) );
  // Allocate number of nodes to be mapped and initialize it to 0
  uint* d_n_node_to_map;
  CUDAMALLOCCTRL( "&d_n_node_to_map", &d_n_node_to_map, sizeof( uint ) );
  gpuErrchk( cudaMemset( d_n_node_to_map, 0, sizeof( uint ) ) );

  // launch kernel that searches remote source nodes indexes in the map,
  // flags the nodes not yet mapped and counts them
  searchNodeIndexNotInMapKernel<<< ( n_used_source_nodes + 1023 ) / 1024, 1024 >>>(
    d_node_map, n_node_map, d_sorted_source_node_index, d_node_to_map, d_n_node_to_map, n_used_source_nodes );
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  uint h_n_node_to_map;

  gpuErrchk( cudaMemcpy( &h_n_node_to_map, d_n_node_to_map, sizeof( uint ), cudaMemcpyDeviceToHost ) );

  // Check if new blocks are required for the map
  uint new_n_blocks = ( n_node_map + h_n_node_to_map - 1 ) / node_map_block_size_ + 1;

  // if new blocks are required for the map, allocate them
  if ( new_n_blocks != n_blocks )
  {
    // Allocate GPU memory for new remote-source-node-map blocks
    allocLocalSourceNodeMapBlocks( h_local_source_node_map_[ target_host ], new_n_blocks );
    // free d_node_map
    if ( n_blocks > 0 )
    {
      CUDAFREECTRL( "d_node_map", d_node_map );
    }
    // update number of blocks in the map
    n_blocks = new_n_blocks;

    // reallocate d_node_map and get it from host
    CUDAMALLOCCTRL( "&d_node_map", &d_node_map, n_blocks * sizeof( uint* ) );
    gpuErrchk( cudaMemcpy(
      d_node_map, &h_local_source_node_map_[ target_host ][ 0 ], n_blocks * sizeof( uint* ), cudaMemcpyHostToDevice ) );
  }

  // Map the not-yet-mapped source nodes using a kernel
  // similar to the one used for counting
  // In the target host unmapped remote source nodes must be mapped
  // to local nodes from n_nodes to n_nodes + n_node_to_map

  // Allocate the index of the nodes to be mapped and initialize it to 0
  uint* d_i_node_to_map;
  CUDAMALLOCCTRL( "&d_i_node_to_map", &d_i_node_to_map, sizeof( uint ) );
  gpuErrchk( cudaMemset( d_i_node_to_map, 0, sizeof( uint ) ) );

  // launch kernel that checks if nodes are already in map
  // if not insert them in the map
  // In the source host, put in the mapsource_node_index
  insertNodesInMapKernel<<< ( n_used_source_nodes + 1023 ) / 1024, 1024 >>>(
    d_node_map, NULL, 0, n_node_map, d_sorted_source_node_index, d_node_to_map, d_i_node_to_map, n_used_source_nodes );
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  // update number of elements in remote source node map
  n_node_map += h_n_node_to_map;
  gpuErrchk(
    cudaMemcpy( &d_n_local_source_node_map_[ target_host ], &n_node_map, sizeof( uint ), cudaMemcpyHostToDevice ) );

  // check for consistency between number of elements
  // and number of blocks in the map
  uint tmp_n_blocks = ( n_node_map - 1 ) / node_map_block_size_ + 1;
  if ( tmp_n_blocks != n_blocks )
  {
    std::cerr << "Inconsistent number of elements " << n_node_map << " and number of blocks " << n_blocks
              << " in local_source_node_map\n";
    //exit( -1 );
    throw ngpu_exception( "Error" );
  }

  // Sort the WHOLE map source_node_map
  // using block sort algorithm copass_sort
  // typical usage:
  // copass_sort::sort<uint>(key_subarray, n,
  //				       aux_size, d_storage, storage_bytes);
  // Determine temporary storage requirements for copass_sort
  int64_t storage_bytes = 0;
  void* d_storage = NULL;
  copass_sort::sort< uint >(
    &h_local_source_node_map_[ target_host ][ 0 ], n_node_map, node_map_block_size_, d_storage, storage_bytes );

  // Allocate temporary storage
  CUDAMALLOCCTRL( "&d_storage", &d_storage, storage_bytes );

  // Run sorting operation
  copass_sort::sort< uint >(
    &h_local_source_node_map_[ target_host ][ 0 ], n_node_map, node_map_block_size_, d_storage, storage_bytes );
  CUDAFREECTRL( "d_storage", d_storage );

  // Remove temporary new connections in source host !!!!!!!!!!!
  // potential problem: check that number of blocks is an independent variable
  // not calculated from n_conn_
  // connect.cu riga 462. Corrected but better keep an eye
  // also, hopefully the is no global device variable for n_conn_
  n_conn_ = old_n_conn;

  return 0;
}

// Method that creates a group of hosts for remote spike communication (i.e. a group of MPI processes)
// host_arr: array of host inexes, n_hosts: nomber of hosts in the group
template < class ConnKeyT, class ConnStructT >
int
ConnectionTemplate< ConnKeyT, ConnStructT >::CreateHostGroup(int *host_arr, int n_hosts, bool mpi_flag)
{
  if (first_connection_flag_ == false) {
    throw ngpu_exception("Host groups must be defined before creating "
			 "connections");
  }
  // pushes all the host indexes in a vector, hg, and check whether this host is in the group
  // TO IMPROVE BY USING AN UNORDERED SET, TO AVOID POSSIBLE REPETITIONS IN host_arr
  // OR CHECK THAT THERE ARE NO REPETITIONS
  std::vector<int> hg;
  bool this_host_is_in_group = false;
  for (int ih=0; ih<n_hosts; ih++) {
    int i_host = host_arr[ih];
    // check whether this host is in the group 
    if (i_host == this_host_) {
      this_host_is_in_group = true;
    }
    hg.push_back(i_host);
  }
  
  // if this host is not in the group, set the entry of host_group_local_id_ to -1
  if (!this_host_is_in_group) {
    host_group_local_id_.push_back(-1);
  }
  else { // the code in the block is executed only if this host is in the group
    // set the local id of the group to be the current size of the local host group array
    int group_local_id = host_group_.size();
    // push the local id in the array of local indexes  of all host groups
    host_group_local_id_.push_back(group_local_id);
    // push the new group into the host_group_ vector
    host_group_.push_back(hg);
    // push a vector of empty unordered sets into host_group_source_node_
    std::vector< std::unordered_set< inode_t > > empty_node_us(n_hosts);
    host_group_source_node_.push_back(empty_node_us);
    // push a vector of empty vectors into host_group_source_node_vect
    std::vector< std::vector< inode_t > > empty_node_vect(n_hosts);
    host_group_source_node_vect_.push_back(empty_node_vect);
    
    host_group_local_source_node_map_.push_back(std::vector< uint >());
    std::vector< std::vector< int > > hg_lni(hg.size(), std::vector< int >());
    host_group_local_node_index_.push_back(hg_lni);
  }
#ifdef HAVE_MPI
  if (mpi_flag) {
    // It is mandatory that the collective creation of MPI_Group and MPI_Comm are executed on all MPI processess
    // no matter if they are actually used or not
    // Get the group from the world communicator
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    // create new MPI group from host group hg
    MPI_Group newgroup;
    MPI_Group_incl(world_group, hg.size(), &hg[0], &newgroup);
    // create new MPI communicator
    MPI_Comm newcomm;
    MPI_Comm_create(MPI_COMM_WORLD, newgroup, &newcomm);
    // They are inserted in the vectors only if they are actually needed
    // Note that the two vectors will be indexed by the local group id, in the same way as host_group_
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


#endif // REMOTECONNECTH
