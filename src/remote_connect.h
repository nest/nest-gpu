//#define CHECKRC

#ifndef REMOTECONNECTH
#define REMOTECONNECTH
#include <vector>
#include <cub/cub.cuh>
#include "nestgpu.h"
#include "connect.h"
#include "copass_sort.h"
// Arrays that map remote source nodes to local spike buffers
  
// The map is organized in blocks having block size:
extern  __constant__ uint node_map_block_size; // = 100000;
extern uint h_node_map_block_size; // = 100000;

// number of elements in the map for each source host
// n_remote_source_node_map[i_source_host]
// with i_source_host = 0, ..., n_hosts-1 excluding this host itself
extern __device__ uint *n_remote_source_node_map; // [n_hosts];
extern uint *d_n_remote_source_node_map;

// remote_source_node_map[i_source_host][i_block][i]
extern std::vector< std::vector<uint*> > h_remote_source_node_map;
extern __device__ uint ***remote_source_node_map;

// local_spike_buffer_map[i_source_host][i_block][i]
extern std::vector< std::vector<uint*> > h_local_spike_buffer_map;
extern __device__ uint ***local_spike_buffer_map;
extern uint ***d_local_spike_buffer_map;

// Arrays that map local source nodes to remote spike buffers

// number of elements in the map for each target host
// n_local_source_node_map[i_target_host]
// with i_target_host = 0, ..., n_hosts-1 excluding this host itself
extern __device__ uint *n_local_source_node_map; // [n_hosts]; 
extern uint *d_n_local_source_node_map;

// local_source_node_map[i_target_host][i_block][i]
extern std::vector< std::vector<uint*> > h_local_source_node_map;
extern __device__ uint ***local_source_node_map;
extern uint ***d_local_source_node_map;

// number of remote target hosts on which each local node
//has outgoing connections
extern uint *d_n_target_hosts; // [n_nodes] 
// target hosts for the node i_node
extern uint **d_node_target_hosts; // [i_node]
// target host map indexes for the node i_node
extern uint **d_node_target_host_i_map; // [i_node]

// Boolean array with one boolean value for each connection rule
// - true if the rule always creates at least one outgoing connection
// from each source node (one_to_one, all_to_all, fixed_outdegree)
// - false otherwise (fixed_indegree, fixed_total_number, pairwise_bernoulli)
extern bool *use_all_source_nodes; // [n_connection_rules]:

extern __constant__ uint n_local_nodes; // number of local nodes

// device function that checks if an int value is in a sorted 2d-array 
// assuming that the entries in the 2d-array are sorted.
// The 2d-array is divided in noncontiguous blocks of size block_size
__device__ bool checkIfValueIsIn2DArr(uint value, uint **arr, uint n_elem,
				      uint block_size, uint *i_block,
				      uint *i_in_block);

// Initialize the maps
int RemoteConnectionMapInit(int n_hosts);

// Allocate GPU memory for new remote-source-node-map blocks
int allocRemoteSourceNodeMapBlocks(std::vector<uint*> &i_remote_src_node_map,
				   std::vector<uint*> &i_local_spike_buf_map,
				   int64_t block_size, uint new_n_block);

// Allocate GPU memory for new local-source-node-map blocks
int allocLocalSourceNodeMapBlocks(std::vector<uint*> &i_local_src_node_map,
				  int64_t block_size, uint new_n_block);

template <class ConnKeyT>
// kernel that flags source nodes used in at least one new connection
// of a given block
__global__ void setUsedSourceNodeKernel(ConnKeyT *conn_key_subarray,
					int64_t n_conn,
					uint *source_node_flag)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) return;
  uint i_source = getConnSource<ConnKeyT>(conn_key_subarray[i_conn]);
  // it is not necessary to use atomic operation. See:
  // https://stackoverflow.com/questions/8416374/several-threads-writing-the-same-value-in-the-same-global-memory-location
#ifdef CHECKRC
  printf("i_conn: %ld\t i_source: %d\n", i_conn, i_source);
#endif
  source_node_flag[i_source] = 1;
}

template <class ConnKeyT>
// Loop on all new connections and set source_node_flag[i_source]=true
int setUsedSourceNodes(std::vector<void*> &conn_key_vect,
		       int64_t old_n_conn, int64_t n_conn,
		       int64_t block_size, uint *d_source_node_flag)
{
  int64_t n_new_conn = n_conn - old_n_conn; // number of new connections

#ifdef CHECKRC
  //////////////////////////////////////////////////////////////////////
  std::cout << "n_new_conn: " << n_new_conn
	    << "\tn_conn: " << n_conn
	    << "\told_n_conn: " << old_n_conn << "\n";
//////////////////////////////////////////////////////////////////////
#endif
  
  uint ib0 = (uint)(old_n_conn / block_size); // first block index
  uint ib1 = (uint)((n_conn - 1) / block_size); // last block
  for (uint ib=ib0; ib<=ib1; ib++) { // loop on blocks
    int64_t n_block_conn; // number of connections in a block
    int64_t i_conn0; // index of first connection in a block
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

    //////////////////////////////////////////////////////////////////////
#ifdef CHECKRC
    std::cout << "n_new_conn: " << n_new_conn
	      << "\ti_conn0: " << i_conn0
	      << "\tn_block_conn: " << n_block_conn << "\n";
#endif
    //////////////////////////////////////////////////////////////////////
    
    setUsedSourceNodeKernel<ConnKeyT><<<(n_block_conn+1023)/1024, 1024>>>
    ((ConnKeyT*)conn_key_vect[ib] + i_conn0, n_block_conn, d_source_node_flag);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }
  return 0;
}


// kernel that fills the arrays of nodes actually used by new connections 
template <class T>
__global__ void getUsedSourceNodeIndexKernel(T source, uint n_source,
					     uint *n_used_source_nodes,
					     uint *source_node_flag,
					     uint *u_source_node_idx,
					     uint *i_source_arr)
{
  uint i_source = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_source>=n_source) return;
  // Count how many source_node_flag are true using atomic increase
  // on n_used_source_nodes
  if (source_node_flag[i_source] != 0) {
    uint pos = atomicAdd(n_used_source_nodes, 1);
    u_source_node_idx[pos] = GetNodeIndex(source, i_source);
    i_source_arr[pos] = i_source;
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
					uint *source_node_flag,
					uint **node_map,
					uint **spike_buffer_map,
					uint n_node_map,
					uint *local_node_index
					)
{
  uint i_source = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_source>=n_source) return;
  // Count how many source_node_flag are true using atomic increase
  // on n_used_source_nodes
  if (source_node_flag[i_source] != 0) {
    uint node_index = GetNodeIndex(source, i_source);
    uint i_block;
    uint i_in_block;
    bool mapped = checkIfValueIsIn2DArr(node_index, node_map,
					n_node_map, node_map_block_size,
					&i_block, &i_in_block);
    if (!mapped) {
      printf("Error in setLocalNodeIndexKernel: node index not mapped\n");
      return;
    }
    uint i_spike_buffer = spike_buffer_map[i_block][i_in_block];
    local_node_index[i_source] = i_spike_buffer;
  }
}


// kernel that replaces the source node index
// in a new remote connection of a given block
// source_node[i_conn] with the value of the element pointed by the
// index itself in the array local_node_index
template <class ConnKeyT>
__global__ void fixConnectionSourceNodeIndexesKernel
(ConnKeyT *conn_key_subarray, int64_t n_conn, uint *local_node_index)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) return;
  uint i_source = getConnSource<ConnKeyT>(conn_key_subarray[i_conn]);
  uint i_delay = getConnDelay<ConnKeyT>(conn_key_subarray[i_conn]);
  uint new_i_source = local_node_index[i_source];

  setConnSource<ConnKeyT>(conn_key_subarray[i_conn], new_i_source);
#ifdef CHECKRC
  printf("i_conn: %ld\t new_i_source: %d\n", i_conn, new_i_source);
#endif

}

// Loops on all new connections and replaces the source node index
// source_node[i_conn] with the value of the element pointed by the
// index itself in the array local_node_index
template <class ConnKeyT>
int fixConnectionSourceNodeIndexes(std::vector<void*> &conn_key_vect,
				   int64_t old_n_conn, int64_t n_conn,
				   int64_t block_size,
				   uint *d_local_node_index)
{
  int64_t n_new_conn = n_conn - old_n_conn; // number of new connections

#ifdef CHECKRC
  //////////////////////////////////////////////////////////////////////
  std::cout << "Fixing source node indexes in new remote connections\n";
  std::cout << "n_new_conn: " << n_new_conn
	    << "\tn_conn: " << n_conn
	    << "\told_n_conn: " << old_n_conn << "\n";
  //////////////////////////////////////////////////////////////////////
#endif
  
  uint ib0 = (uint)(old_n_conn / block_size); // first block index
  uint ib1 = (uint)((n_conn - 1) / block_size); // last block
  for (uint ib=ib0; ib<=ib1; ib++) { // loop on blocks
    int64_t n_block_conn; // number of connections in a block
    int64_t i_conn0; // index of first connection in a block
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

#ifdef CHECKRC
    //////////////////////////////////////////////////////////////////////
    std::cout << "n_new_conn: " << n_new_conn
	      << "\ti_conn0: " << i_conn0
	      << "\tn_block_conn: " << n_block_conn << "\n";
    //////////////////////////////////////////////////////////////////////
#endif
    
    
    fixConnectionSourceNodeIndexesKernel<ConnKeyT>
      <<<(n_block_conn+1023)/1024, 1024>>>
      ((ConnKeyT*)conn_key_vect[ib] + i_conn0, n_block_conn,
       d_local_node_index);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }
  return 0;
}


// REMOTE CONNECT FUNCTION
template <class T1, class T2, class ConnKeyT, class ConnStructT>
int NESTGPU::_RemoteConnect(int this_host,
			    int source_host, T1 source, inode_t n_source,
			    int target_host, T2 target, inode_t n_target,
			    ConnSpec &conn_spec, SynSpec &syn_spec)
{
  if (source_host>=n_hosts_) {
    throw ngpu_exception("Source host index out of range in _RemoteConnect");
  }
  if (target_host>=n_hosts_) {
    throw ngpu_exception("Target host index out of range in _RemoteConnect");
  }
  if (this_host>=n_hosts_) {
    throw ngpu_exception("this_host index out of range in _RemoteConnect");
  }

  // Check if it is a local connection
  if (this_host==source_host && source_host==target_host) {
    _Connect<T1, T2, ConnKeyT, ConnStructT>
      (source, n_source, target, n_target, conn_spec, syn_spec);
  }
  // Check if target_host matches this_host
  else if (this_host==target_host) {
    _RemoteConnectSource<T1, T2, ConnKeyT, ConnStructT>
      (source_host, source, n_source, target, n_target, conn_spec, syn_spec);
  }
  // Check if source_host matches this_host
  else if (this_host==source_host) {
    _RemoteConnectTarget<T1, T2, ConnKeyT, ConnStructT>
      (target_host, source, n_source, target, n_target, conn_spec, syn_spec);
  }
  
  return 0;
}

/*
template
int NESTGPU::_RemoteConnect<uint, uint>
(uint this_host, uint source_host, uint source, uint n_source,
 uint target_host, uint target, uint n_target,
 ConnSpec &conn_spec, SynSpec &syn_spec);

template
int NESTGPU::_RemoteConnect<uint, uint*>
(uint this_host, uint source_host, uint source, uint n_source,
 uint target_host, uint *target, uint n_target,
 ConnSpec &conn_spec, SynSpec &syn_spec);

template
int NESTGPU::_RemoteConnect<uint*, uint>
(uint this_host, uint source_host, uint *source, uint n_source,
 uint target_host, uint target, uint n_target,
 ConnSpec &conn_spec, SynSpec &syn_spec);

template
int NESTGPU::_RemoteConnect<uint*, uint*>
(uint this_host, uint source_host, uint *source, uint n_source,
 uint target_host, uint *target, uint n_target,
 ConnSpec &conn_spec, SynSpec &syn_spec);
*/

template <class T1, class T2, class ConnKeyT, class ConnStructT>
int NESTGPU::_RemoteConnect(int source_host, T1 source, inode_t n_source,
			    int target_host, T2 target, inode_t n_target,
			    ConnSpec &conn_spec, SynSpec &syn_spec)
{
  return _RemoteConnect<T1, T2, ConnKeyT, ConnStructT>
    (this_host_, source_host, source, n_source,
			 target_host, target, n_target,
			 conn_spec, syn_spec);
}

/*
template
int NESTGPU::_RemoteConnect<uint, uint>
(uint source_host, uint source, uint n_source,
 uint target_host, uint target, uint n_target,
 ConnSpec &conn_spec, SynSpec &syn_spec);

template
int NESTGPU::_RemoteConnect<uint, uint*>
(uint source_host, uint source, uint n_source,
 uint target_host, uint *target, uint n_target,
 ConnSpec &conn_spec, SynSpec &syn_spec);

template
int NESTGPU::_RemoteConnect<uint*, uint>
(uint source_host, uint *source, uint n_source,
 uint target_host, uint target, uint n_target,
 ConnSpec &conn_spec, SynSpec &syn_spec);

template
int NESTGPU::_RemoteConnect<uint*, uint*>
(uint source_host, uint *source, uint n_source,
 uint target_host, uint *target, uint n_target,
 ConnSpec &conn_spec, SynSpec &syn_spec);
*/



// kernel that searches node indexes in map
// increase counter of mapped nodes
__global__ void searchNodeIndexInMapKernel
(
 uint **node_map,
 uint n_node_map,
 uint *count_mapped, // i.e. *n_target_hosts for our application
 uint n_node);

// kernel that searches node indexes in map
// flags nodes not yet mapped and counts them
__global__ void searchNodeIndexNotInMapKernel
(
 uint **node_map,
 uint n_node_map,
 uint *sorted_node_index,
 bool *node_to_map,
 uint *n_node_to_map,
 uint n_node);


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
 uint n_node);


template <class ConnKeyT, class ConnStructT>
__global__ void addOffsetToExternalNodeIdsKernel
(int64_t n_conn, ConnKeyT *conn_key_subarray,
 ConnStructT *conn_struct_subarray, uint i_image_node_0)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) return;
  //uint target_port_syn = conn_subarray[i_conn].target_port_syn;
  //if (target_port_syn & (1 << (MaxPortSynNBits - 1))) {
  //target_port_syn = target_port_syn ^ (1 << (MaxPortSynNBits - 1));
  //conn_subarray[i_conn].target_port_syn = target_port_syn; 
  //key_subarray[i_conn] += (i_image_node_0 << MaxPortSynNBits);
  uint remote_flag = getConnRemoteFlag<ConnKeyT, ConnStructT>
    (conn_key_subarray[i_conn], conn_struct_subarray[i_conn]);
  if (remote_flag==1) {
    // IN THE FUTURE KEEP IT!!!!!!!!!!!!!!!!!!!!!!!!!!
    clearConnRemoteFlag<ConnKeyT, ConnStructT>
      (conn_key_subarray[i_conn], conn_struct_subarray[i_conn]);
    uint i_source = getConnSource<ConnKeyT>(conn_key_subarray[i_conn]);
    i_source += i_image_node_0;
    setConnSource<ConnKeyT>(conn_key_subarray[i_conn], i_source);
  }
}

template <class ConnKeyT, class ConnStructT>
int NESTGPU::addOffsetToExternalNodeIds()
{
  int64_t block_size = h_ConnBlockSize;
  uint n_blocks = (NConn - 1) / block_size + 1;
  uint i_image_node_0 = GetNLocalNodes();

  for (uint ib=0; ib<n_blocks; ib++) {
    int64_t n_block_conn = block_size; // number of connections in the block
    if (ib == n_blocks-1) { // last block
      n_block_conn = (NConn - 1) % block_size + 1;
    }
    addOffsetToExternalNodeIdsKernel<ConnKeyT, ConnStructT>
      <<<(n_block_conn+1023)/1024, 1024>>>
      (n_block_conn, (ConnKeyT*)ConnKeyVect[ib],
       (ConnStructT*)ConnStructVect[ib], i_image_node_0);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }

  /////////////////////////////////////////////////////////////////////
#ifdef CHECKRC
  /// TEMPORARY for check
  std::cout << "After addOffsetToExternalNodeIds\n";
  uint h_source_delay[NConn];
  uint h_source[NConn];
  uint h_delay[NConn];
  gpuErrchk(cudaMemcpy(h_source_delay, ConnKeyVect[0],
		       NConn*sizeof(uint), cudaMemcpyDeviceToHost));
  for (uint i=0; i<NConn; i++) {
    h_source[i] = h_source_delay[i] >> h_MaxPortNBits;
    h_delay[i] = h_source_delay[i] & h_PortMask;
    std::cout << "i_conn: " << i << " source: " << h_source[i];
    std::cout << " delay: " << h_delay[i] << "\n";
  }
#endif  
  //////////////////////////////
  
  return 0;
}





// REMOTE CONNECT FUNCTION for target_host matching this_host
template <class T1, class T2, class ConnKeyT, class ConnStructT>
int NESTGPU::_RemoteConnectSource(int source_host, T1 source, inode_t n_source,
				  T2 target, inode_t n_target,
				  ConnSpec &conn_spec, SynSpec &syn_spec)
{
  // n_nodes will be the first index for new mapping of remote source nodes
  // to local spike buffers
  //int spike_buffer_map_i0 = GetNNode();
  uint spike_buffer_map_i0 = n_image_nodes_;
  //syn_spec.port_ = syn_spec.port_ |
  //  (1 << (h_MaxPortSynNBits - h_MaxSynNBits - 1));
  syn_spec.syn_group_ = syn_spec.syn_group_ | (1 << h_MaxSynNBits);
    
  // check if the flag UseAllSourceNodes[conn_rule] is false
  // if (!use_all_source_nodes_flag) {
    
  // on both the source and target hosts create a temporary array
  // of booleans having size equal to the number of source nodes
    
  uint *d_source_node_flag; // [n_source] // each element is initially false
  CUDAMALLOCCTRL("&d_source_node_flag",&d_source_node_flag,
		 n_source*sizeof(uint));
  //std::cout << "d_source_node_flag: " << d_source_node_flag << "\n";
  gpuErrchk(cudaMemset(d_source_node_flag, 0, n_source*sizeof(uint)));  
    
  // on the target hosts create a temporary array of integers having size
  // equal to the number of source nodes
    
  uint *d_local_node_index; // [n_source]; // only on target host
  CUDAMALLOCCTRL("&d_local_node_index",&d_local_node_index,
		 n_source*sizeof(uint));
    
  int64_t old_n_conn = NConn;
  // The connect command is performed on both source and target host using
  // the same initial seed and using as source node indexes the integers
  // from 0 to n_source_nodes - 1
  _Connect<inode_t, T2, ConnKeyT, ConnStructT>
    (conn_random_generator_[source_host][this_host_],
     0, n_source, target, n_target,
     conn_spec, syn_spec);
  if (NConn == old_n_conn) {
    return 0;
  }
  /////////////////////////////////////////////////////////////////////
#ifdef CHECKRC
  /// TEMPORARY for check
  uint h_source_delay[NConn];
  uint h_source[NConn];
  uint h_delay[NConn];
  gpuErrchk(cudaMemcpy(h_source_delay, ConnKeyVect[0],
		       NConn*sizeof(uint), cudaMemcpyDeviceToHost));
  for (uint i=0; i<NConn; i++) {
    h_source[i] = h_source_delay[i] >> h_MaxPortNBits;
    h_delay[i] = h_source_delay[i] & h_PortMask;
  }
#endif  
  //////////////////////////////
    

  // flag source nodes used in at least one new connection
  // Loop on all new connections and set source_node_flag[i_source]=true
  setUsedSourceNodes<ConnKeyT>(ConnKeyVect, old_n_conn, NConn, h_ConnBlockSize,
		     d_source_node_flag);

  //////////////////////////////////////////////////////////////////////
#ifdef CHECKRC
  /// TEMPORARY for check
  std::cout << "n_source: " << n_source << "\n";
  uint h_source_node_flag[n_source];
  //std::cout << "d_source_node_flag: " << d_source_node_flag << "\n";
    
  gpuErrchk(cudaMemcpy(h_source_node_flag, d_source_node_flag,
		       n_source*sizeof(uint), cudaMemcpyDeviceToHost));

  for (uint i=0; i<n_source; i++) {
    std::cout << "i_source: " << i << " source_node_flag: "
	      << h_source_node_flag[i] << "\n";
  }
#endif
  //////////////////////////////
    
  // Count source nodes actually used in new connections
  // Allocate n_used_source_nodes and initialize it to 0
  uint *d_n_used_source_nodes;
  CUDAMALLOCCTRL("&d_n_used_source_nodes",&d_n_used_source_nodes, sizeof(uint));
  gpuErrchk(cudaMemset(d_n_used_source_nodes, 0, sizeof(uint)));  
  // Launch kernel to count used nodes
  countUsedSourceNodeKernel<<<(n_source+1023)/1024, 1024>>>
    (n_source, d_n_used_source_nodes, d_source_node_flag);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  // copy result from GPU to CPU memory
  uint n_used_source_nodes;
  gpuErrchk(cudaMemcpy(&n_used_source_nodes, d_n_used_source_nodes,
		       sizeof(uint), cudaMemcpyDeviceToHost));

#ifdef CHECKRC
  // TEMPORARY
  std::cout << "n_used_source_nodes: " << n_used_source_nodes << "\n";
  //
#endif
    
  // Define and allocate arrays of size n_used_source_nodes
  uint *d_unsorted_source_node_index; // [n_used_source_nodes];
  uint *d_sorted_source_node_index; // [n_used_source_nodes];
  // i_source_arr are the positions in the arrays source_node_flag
  // and local_node_index 
  uint *d_i_unsorted_source_arr; // [n_used_source_nodes];
  uint *d_i_sorted_source_arr; // [n_used_source_nodes];
  bool *d_source_node_index_to_be_mapped; //[n_used_source_nodes]; // initially false
  CUDAMALLOCCTRL("&d_unsorted_source_node_index",&d_unsorted_source_node_index,
		       n_used_source_nodes*sizeof(uint));
  CUDAMALLOCCTRL("&d_sorted_source_node_index",&d_sorted_source_node_index,
		       n_used_source_nodes*sizeof(uint));
  CUDAMALLOCCTRL("&d_i_unsorted_source_arr",&d_i_unsorted_source_arr,
		       n_used_source_nodes*sizeof(uint));
  CUDAMALLOCCTRL("&d_i_sorted_source_arr",&d_i_sorted_source_arr,
		       n_used_source_nodes*sizeof(uint));
  CUDAMALLOCCTRL("&d_source_node_index_to_be_mapped",&d_source_node_index_to_be_mapped,
		       n_used_source_nodes*sizeof(int8_t));
  // source_node_index_to_be_mapped is initially false
  gpuErrchk(cudaMemset(d_source_node_index_to_be_mapped, 0,
		       n_used_source_nodes*sizeof(int8_t)));
    
  // Fill the arrays of nodes actually used by new connections 
  // Reset n_used_source_nodes to 0
  gpuErrchk(cudaMemset(d_n_used_source_nodes, 0, sizeof(uint)));  
  // Launch kernel to fill the arrays
  getUsedSourceNodeIndexKernel<<<(n_source+1023)/1024, 1024>>>
    (source, n_source, d_n_used_source_nodes, d_source_node_flag,
     d_unsorted_source_node_index, d_i_unsorted_source_arr);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  //////////////////////////////////////////////////////////////////////
#ifdef CHECKRC
  /// TEMPORARY for check
  std::cout << "n_used_source_nodes: " << n_used_source_nodes << "\n";
  uint h_unsorted_source_node_index[n_used_source_nodes];
  uint h_i_unsorted_source_arr[n_used_source_nodes];
    
  gpuErrchk(cudaMemcpy(h_unsorted_source_node_index,
		       d_unsorted_source_node_index,
		       n_used_source_nodes*sizeof(uint),
		       cudaMemcpyDeviceToHost));

  gpuErrchk(cudaMemcpy(h_i_unsorted_source_arr,
		       d_i_unsorted_source_arr,
		       n_used_source_nodes*sizeof(uint),
		       cudaMemcpyDeviceToHost));

  for (uint i=0; i<n_used_source_nodes; i++) {
    std::cout << "i_used_source: " << i << " unsorted_source_node_index: "
	      << h_unsorted_source_node_index[i]
	      << " i_unsorted_source_arr: "
	      << h_i_unsorted_source_arr[i] << "\n";
  }
#endif
  //////////////////////////////

  // Sort the arrays using unsorted_source_node_index as key
  // and i_source as value -> sorted_source_node_index


  // Determine temporary storage requirements for RadixSort
  void *d_sort_storage = NULL;
  size_t sort_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(d_sort_storage, sort_storage_bytes,
				  d_unsorted_source_node_index,
				  d_sorted_source_node_index,
				  d_i_unsorted_source_arr,
				  d_i_sorted_source_arr,
				  n_used_source_nodes);
  // Allocate temporary storage
  CUDAMALLOCCTRL("&d_sort_storage",&d_sort_storage, sort_storage_bytes);

  // Run sorting operation
  cub::DeviceRadixSort::SortPairs(d_sort_storage, sort_storage_bytes,
				  d_unsorted_source_node_index,
				  d_sorted_source_node_index,
				  d_i_unsorted_source_arr,
				  d_i_sorted_source_arr,
				  n_used_source_nodes);

  //////////////////////////////////////////////////////////////////////
#ifdef CHECKRC
  /// TEMPORARY for check
  uint h_sorted_source_node_index[n_used_source_nodes];
  uint h_i_sorted_source_arr[n_used_source_nodes];
  
  gpuErrchk(cudaMemcpy(h_sorted_source_node_index,
		       d_sorted_source_node_index,
		       n_used_source_nodes*sizeof(uint),
		       cudaMemcpyDeviceToHost));

  gpuErrchk(cudaMemcpy(h_i_sorted_source_arr,
		       d_i_sorted_source_arr,
		       n_used_source_nodes*sizeof(uint),
		       cudaMemcpyDeviceToHost));

  for (uint i=0; i<n_used_source_nodes; i++) {
    std::cout << "i_used_source: " << i << " sorted_source_node_index: "
	      << h_sorted_source_node_index[i]
	      << " i_sorted_source_arr: "
	      << h_i_sorted_source_arr[i] << "\n";
  }
#endif
  //////////////////////////////////////////////////////////////////////


  //////////////////////////////
  // Allocate array of remote source node map blocks
  // and copy their address from host to device
  uint n_blocks = h_remote_source_node_map[source_host].size();
  uint **d_node_map = NULL;
  uint **d_spike_buffer_map = NULL;
  // get current number of elements in the map
  uint n_node_map;
  gpuErrchk(cudaMemcpy(&n_node_map,
		       &d_n_remote_source_node_map[source_host], sizeof(uint),
		       cudaMemcpyDeviceToHost));
  
    
  if (n_blocks>0) {
    // check for consistency between number of elements
    // and number of blocks in the map
    uint tmp_n_blocks = (n_node_map - 1) / h_node_map_block_size + 1;
    if (tmp_n_blocks != n_blocks) {
      std::cerr << "Inconsistent number of elements "
		<< n_node_map << " and number of blocks "
		<< n_blocks << " in remote_source_node_map\n";
      exit(-1);
    }
    CUDAMALLOCCTRL("&d_node_map",&d_node_map, n_blocks*sizeof(uint*));
    gpuErrchk(cudaMemcpy(d_node_map,
			 &h_remote_source_node_map[source_host][0],
			 n_blocks*sizeof(uint*),
			 cudaMemcpyHostToDevice));
  }

  // Allocate boolean array for flagging remote source nodes not yet mapped
  // and initialize all elements to 0 (false)
  bool *d_node_to_map;
  CUDAMALLOCCTRL("&d_node_to_map",&d_node_to_map, n_used_source_nodes*sizeof(bool));
  gpuErrchk(cudaMemset(d_node_to_map, 0, n_used_source_nodes*sizeof(bool)));
  // Allocate number of nodes to be mapped and initialize it to 0 
  uint *d_n_node_to_map;
  CUDAMALLOCCTRL("&d_n_node_to_map",&d_n_node_to_map, sizeof(uint));
  gpuErrchk(cudaMemset(d_n_node_to_map, 0, sizeof(uint)));

  // launch kernel that searches remote source nodes indexes not in the map,
  // flags the nodes not yet mapped and counts them
  searchNodeIndexNotInMapKernel<<<(n_used_source_nodes+1023)/1024, 1024>>>
    (d_node_map, n_node_map, d_sorted_source_node_index, d_node_to_map,
     d_n_node_to_map, n_used_source_nodes);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  uint h_n_node_to_map;
    
  gpuErrchk(cudaMemcpy(&h_n_node_to_map, d_n_node_to_map, sizeof(uint),
		       cudaMemcpyDeviceToHost));

  //////////////////////////////////////////////////////////////////////
#ifdef CHECKRC
  /// TEMPORARY for check
  std::cout << "n_node_to_map: " << h_n_node_to_map << "\n";
  
  bool h_node_to_map[n_used_source_nodes];
    
  gpuErrchk(cudaMemcpy(h_node_to_map, d_node_to_map,
		       n_used_source_nodes*sizeof(bool),
		       cudaMemcpyDeviceToHost));

  for (uint i=0; i<n_used_source_nodes; i++) {
    std::cout << "i_used_source: " << i << " sorted_source_node_index: "
	      << h_sorted_source_node_index[i]
	      << " node_to_map: " << h_node_to_map[i] << "\n";
  }
#endif
  //////////////////////////////

  // Check if new blocks are required for the map
  uint new_n_blocks = (n_node_map + h_n_node_to_map - 1)
    / h_node_map_block_size + 1;

#ifdef CHECKRC
  std::cout << "new_n_blocks: " << new_n_blocks << "\n";
#endif
  
  // if new blocks are required for the map, allocate them
  if (new_n_blocks != n_blocks) {
    // Allocate GPU memory for new remote-source-node-map blocks
    allocRemoteSourceNodeMapBlocks(h_remote_source_node_map[source_host],
				   h_local_spike_buffer_map[source_host],
				   h_node_map_block_size, new_n_blocks);
    // free d_node_map
    if (n_blocks>0) {
      CUDAFREECTRL("d_node_map",d_node_map);
    }
    // update number of blocks in the map 
    n_blocks = new_n_blocks;

    // reallocate d_node_map and get it from host
    CUDAMALLOCCTRL("&d_node_map",&d_node_map, n_blocks*sizeof(uint*));
    gpuErrchk(cudaMemcpy(d_node_map,
			 &h_remote_source_node_map[source_host][0],
			 n_blocks*sizeof(uint*),
			 cudaMemcpyHostToDevice));
  }
  if (n_blocks > 0) {
    // allocate d_spike_buffer_map and get it from host
    CUDAMALLOCCTRL("&d_spike_buffer_map",&d_spike_buffer_map, n_blocks*sizeof(uint*));
    gpuErrchk(cudaMemcpy(d_spike_buffer_map,
			 &h_local_spike_buffer_map[source_host][0],
			 n_blocks*sizeof(uint*),
			 cudaMemcpyHostToDevice));
  }
  
  // Map the not-yet-mapped source nodes using a kernel
  // similar to the one used for counting
  // In the target host unmapped remote source nodes must be mapped
  // to local nodes from n_nodes to n_nodes + n_node_to_map
  
  // Allocate the index of the nodes to be mapped and initialize it to 0 
  uint *d_i_node_to_map;
  CUDAMALLOCCTRL("&d_i_node_to_map",&d_i_node_to_map, sizeof(uint));
  gpuErrchk(cudaMemset(d_i_node_to_map, 0, sizeof(uint)));

  // launch kernel that checks if nodes are already in map
  // if not insert them in the map
  // In the target host, put in the map the pair:
  // (source_node_index, spike_buffer_map_i0 + i_node_to_map)
  insertNodesInMapKernel<<<(n_used_source_nodes+1023)/1024, 1024>>>
    (d_node_map, d_spike_buffer_map, spike_buffer_map_i0,
     n_node_map, d_sorted_source_node_index, d_node_to_map,
     d_i_node_to_map, n_used_source_nodes);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  // update number of elements in remote source node map
  n_node_map += h_n_node_to_map;
  gpuErrchk(cudaMemcpy(&d_n_remote_source_node_map[source_host],
		       &n_node_map, sizeof(uint), cudaMemcpyHostToDevice));
  
  // check for consistency between number of elements
  // and number of blocks in the map
  uint tmp_n_blocks = (n_node_map - 1) / h_node_map_block_size + 1;
  if (tmp_n_blocks != n_blocks) {
    std::cerr << "Inconsistent number of elements "
	      << n_node_map << " and number of blocks "
	      << n_blocks << " in remote_source_node_map\n";
    exit(-1);
  }

  //////////////////////////////////////////////////////////////////////
#ifdef CHECKRC
  /// TEMPORARY for check
  std::cout << "//////////////////////////////////////////////\n";
  std::cout << "UPDATED UNSORTED MAP\n";
  std::cout << "OF REMOTE-SOURCE_NODES TO LOCAL-SPIKE-BUFFERS\n";
  std::cout << "n_node_map: " << n_node_map << "\n";
  std::cout << "n_blocks: " << n_blocks << "\n";
  std::cout << "block_size: " << h_node_map_block_size << "\n";

  uint block_size = h_node_map_block_size;
  uint h_node_map_block[block_size];
  uint h_spike_buffer_map_block[block_size];
  for (uint ib=0; ib<n_blocks; ib++) {
    gpuErrchk(cudaMemcpy(h_node_map_block,
			 h_remote_source_node_map[source_host][ib],
			 block_size*sizeof(uint),
			 cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_spike_buffer_map_block,
			 h_local_spike_buffer_map[source_host][ib],
			 block_size*sizeof(uint),
			 cudaMemcpyDeviceToHost));
    std::cout << "\n";
    std::cout << "block " << ib << "\n";
    std::cout << "remote source node index, local spike buffer index\n";
    uint n = block_size;
    if (ib==n_blocks-1) {
      n = (n_node_map - 1) % block_size + 1;
    }
    for (uint i=0; i<n; i++) {
      std::cout << h_node_map_block[i] << "\t" << h_spike_buffer_map_block[i]
		<< "\n"; 
    }
    std::cout << "\n";
  }
#endif
  //////////////////////////////////////////////////////////////////////
  
  // Sort the WHOLE key-pair map source_node_map, spike_buffer_map
  // using block sort algorithm copass_sort
  // typical usage:
  // copass_sort::sort<uint, value_struct>(key_subarray, value_subarray, n,
  //				       aux_size, d_storage, storage_bytes);
  // Determine temporary storage requirements for copass_sort
  int64_t storage_bytes = 0;
  void *d_storage = NULL;
  copass_sort::sort<uint, uint>
    (&h_remote_source_node_map[source_host][0],
     &h_local_spike_buffer_map[source_host][0],
     n_node_map, h_node_map_block_size, d_storage, storage_bytes);

#ifdef CHECKRC
  printf("storage bytes for copass sort: %ld\n", storage_bytes);
#endif
  
  // Allocate temporary storage
  CUDAMALLOCCTRL("&d_storage",&d_storage, storage_bytes);

  // Run sorting operation
  copass_sort::sort<uint, uint>
    (&h_remote_source_node_map[source_host][0],
     &h_local_spike_buffer_map[source_host][0],
     n_node_map, h_node_map_block_size, d_storage, storage_bytes);
  CUDAFREECTRL("d_storage",d_storage);

  //////////////////////////////////////////////////////////////////////
#ifdef CHECKRC
  /// TEMPORARY for check
  std::cout << "//////////////////////////////////////////////\n";
  std::cout << "UPDATED SORTED MAP\n";
  std::cout << "OF REMOTE-SOURCE_NODES TO LOCAL-SPIKE-BUFFERS\n";
  std::cout << "n_node_map: " << n_node_map << "\n";
  std::cout << "n_blocks: " << n_blocks << "\n";
  std::cout << "block_size: " << block_size << "\n";

  for (uint ib=0; ib<n_blocks; ib++) {
    gpuErrchk(cudaMemcpy(h_node_map_block,
			 h_remote_source_node_map[source_host][ib],
			 block_size*sizeof(uint),
			 cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_spike_buffer_map_block,
			 h_local_spike_buffer_map[source_host][ib],
			 block_size*sizeof(uint),
			 cudaMemcpyDeviceToHost));
    std::cout << "\n";
    std::cout << "block " << ib << "\n";
    std::cout << "remote source node index, local spike buffer index\n";
    uint n = block_size;
    if (ib==n_blocks-1) {
      n = (n_node_map - 1) % block_size + 1;
    }
    for (uint i=0; i<n; i++) {
      std::cout << h_node_map_block[i] << "\t" << h_spike_buffer_map_block[i]
		<< "\n"; 
    }
    std::cout << "\n";
  }
#endif
  //////////////////////////////////////////////////////////////////////

  // Launch kernel that searches source node indexes in the map
  // and set corresponding values of local_node_index
  setLocalNodeIndexKernel<<<(n_source+1023)/1024, 1024>>>
    (source, n_source, d_source_node_flag,
     d_node_map, d_spike_buffer_map, n_node_map, d_local_node_index);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  //////////////////////////////////////////////////////////////////////
#ifdef CHECKRC
  /// TEMPORARY for check
  std::cout << "n_source: " << n_source << "\n";
  uint h_local_node_index[n_source];
  gpuErrchk(cudaMemcpy(h_local_node_index, d_local_node_index,
		       n_source*sizeof(uint), cudaMemcpyDeviceToHost));

  for (uint i=0; i<n_source; i++) {
    std::cout << "i_source: " << i << " source_node_flag: "
	      << h_source_node_flag[i] << " local_node_index: ";
    if (h_source_node_flag[i]) {
      std::cout << h_local_node_index[i];
    }
    else {
      std::cout << "---";
    }
    std::cout << "\n";
  }
#endif
  //////////////////////////////


  // On target host. Loop on all new connections and replace
  // the source node index source_node[i_conn] with the value of the element
  // pointed by the index itself in the array local_node_index
  // source_node[i_conn] = local_node_index[source_node[i_conn]];

  // similar to setUsedSourceNodes
  // replace source_node_flag[i_source] with local_node_index[i_source]
  // clearly read it instead of writing on it!
  //setUsedSourceNodes(ConnKeyVect, old_n_conn, NConn, h_ConnBlockSize,
  //		     d_source_node_flag);
  // becomes something like
  fixConnectionSourceNodeIndexes<ConnKeyT>(ConnKeyVect, old_n_conn, NConn,
					   h_ConnBlockSize, d_local_node_index);

  // On target host. Create n_nodes_to_map nodes of type image_node
  //std::cout << "h_n_node_to_map " << h_n_node_to_map <<"\n";
  if (h_n_node_to_map > 0) {
    //_Create("image_node", h_n_node_to_map);
    n_image_nodes_ += h_n_node_to_map;
    //std::cout << "n_image_nodes_ " << n_image_nodes_ <<"\n";
  }
  
  return 0;
}



// REMOTE CONNECT FUNCTION for source_host matching this_host
template <class T1, class T2, class ConnKeyT, class ConnStructT>
int NESTGPU::_RemoteConnectTarget(int target_host, T1 source, inode_t n_source,
				  T2 target, inode_t n_target,
				  ConnSpec &conn_spec, SynSpec &syn_spec)
{
  // check if the flag UseAllSourceNodes[conn_rule] is false
  // if (!use_all_source_nodes_flag) {
    
  // on both the source and target hosts create a temporary array
  // of booleans having size equal to the number of source nodes
    
  uint *d_source_node_flag; // [n_source] // each element is initially false
  CUDAMALLOCCTRL("&d_source_node_flag",&d_source_node_flag,
		 n_source*sizeof(uint));
  //std::cout << "d_source_node_flag: " << d_source_node_flag << "\n";
  gpuErrchk(cudaMemset(d_source_node_flag, 0, n_source*sizeof(uint)));  
    
  int64_t old_n_conn = NConn;
  // The connect command is performed on both source and target host using
  // the same initial seed and using as source node indexes the integers
  // from 0 to n_source_nodes - 1
  _Connect<inode_t, T2, ConnKeyT, ConnStructT>
    (conn_random_generator_[this_host_][target_host],
     0, n_source, target, n_target,
     conn_spec, syn_spec);
  
  if (NConn == old_n_conn) {
    return 0;
  }
  //////////////////////////////////////////////////////////////////////
#ifdef CHECKRC
  /// TEMPORARY for check
  uint h_source_delay[NConn];
  uint h_source[NConn];
  uint h_delay[NConn];
  gpuErrchk(cudaMemcpy(h_source_delay, ConnKeyVect[0],
		       NConn*sizeof(uint), cudaMemcpyDeviceToHost));
  for (uint i=0; i<NConn; i++) {
    h_source[i] = h_source_delay[i] >> h_MaxPortNBits;
    h_delay[i] = h_source_delay[i] & h_PortMask;
    std::cout << "i_conn: " << i << " source: " << h_source[i];
    std::cout << " delay: " << h_delay[i] << "\n";
  }
#endif
  //////////////////////////////
    

  // flag source nodes used in at least one new connection
  // Loop on all new connections and set source_node_flag[i_source]=true
  setUsedSourceNodes<ConnKeyT>(ConnKeyVect, old_n_conn, NConn, h_ConnBlockSize,
			       d_source_node_flag);

  //////////////////////////////////////////////////////////////////////
#ifdef CHECKRC
  /// TEMPORARY for check
  std::cout << "n_source: " << n_source << "\n";
  uint h_source_node_flag[n_source];
  //  std::cout << "d_source_node_flag: " << d_source_node_flag << "\n";
    
  gpuErrchk(cudaMemcpy(h_source_node_flag, d_source_node_flag,
		       n_source*sizeof(uint), cudaMemcpyDeviceToHost));

  for (uint i=0; i<n_source; i++) {
    std::cout << "i_source: " << i << " source_node_flag: "
  	      << h_source_node_flag[i] << "\n";
  }
#endif
  //////////////////////////////
    
  // Count source nodes actually used in new connections
  // Allocate n_used_source_nodes and initialize it to 0
  uint *d_n_used_source_nodes;
  CUDAMALLOCCTRL("&d_n_used_source_nodes",&d_n_used_source_nodes, sizeof(uint));
  gpuErrchk(cudaMemset(d_n_used_source_nodes, 0, sizeof(uint)));  
  // Launch kernel to count used nodes
  countUsedSourceNodeKernel<<<(n_source+1023)/1024, 1024>>>
    (n_source, d_n_used_source_nodes, d_source_node_flag);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  // copy result from GPU to CPU memory
  uint n_used_source_nodes;
  gpuErrchk(cudaMemcpy(&n_used_source_nodes, d_n_used_source_nodes,
		       sizeof(uint), cudaMemcpyDeviceToHost));

#ifdef CHECKRC
  // TEMPORARY
  std::cout << "n_used_source_nodes: " << n_used_source_nodes << "\n";
  //
#endif
    
  // Define and allocate arrays of size n_used_source_nodes
  uint *d_unsorted_source_node_index; // [n_used_source_nodes];
  uint *d_sorted_source_node_index; // [n_used_source_nodes];
  // i_source_arr are the positions in the arrays source_node_flag
  // and local_node_index 
  uint *d_i_unsorted_source_arr; // [n_used_source_nodes];
  uint *d_i_sorted_source_arr; // [n_used_source_nodes];
  bool *d_source_node_index_to_be_mapped; //[n_used_source_nodes]; // initially false
  CUDAMALLOCCTRL("&d_unsorted_source_node_index",&d_unsorted_source_node_index,
		       n_used_source_nodes*sizeof(uint));
  CUDAMALLOCCTRL("&d_sorted_source_node_index",&d_sorted_source_node_index,
		       n_used_source_nodes*sizeof(uint));
  CUDAMALLOCCTRL("&d_i_unsorted_source_arr",&d_i_unsorted_source_arr,
		       n_used_source_nodes*sizeof(uint));
  CUDAMALLOCCTRL("&d_i_sorted_source_arr",&d_i_sorted_source_arr,
		       n_used_source_nodes*sizeof(uint));
  CUDAMALLOCCTRL("&d_source_node_index_to_be_mapped",
		 &d_source_node_index_to_be_mapped,
		 n_used_source_nodes*sizeof(int8_t));
  // source_node_index_to_be_mapped is initially false
  gpuErrchk(cudaMemset(d_source_node_index_to_be_mapped, 0,
		       n_used_source_nodes*sizeof(int8_t)));
    
  // Fill the arrays of nodes actually used by new connections 
  // Reset n_used_source_nodes to 0
  gpuErrchk(cudaMemset(d_n_used_source_nodes, 0, sizeof(uint)));  
  // Launch kernel to fill the arrays
  getUsedSourceNodeIndexKernel<<<(n_source+1023)/1024, 1024>>>
    (source, n_source, d_n_used_source_nodes, d_source_node_flag,
     d_unsorted_source_node_index, d_i_unsorted_source_arr);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  //////////////////////////////////////////////////////////////////////
#ifdef CHECKRC
  /// TEMPORARY for check
  std::cout << "n_used_source_nodes: " << n_used_source_nodes << "\n";
  uint h_unsorted_source_node_index[n_used_source_nodes];
  uint h_i_unsorted_source_arr[n_used_source_nodes];
    
  gpuErrchk(cudaMemcpy(h_unsorted_source_node_index,
		       d_unsorted_source_node_index,
		       n_used_source_nodes*sizeof(uint),
		       cudaMemcpyDeviceToHost));

  gpuErrchk(cudaMemcpy(h_i_unsorted_source_arr,
		       d_i_unsorted_source_arr,
		       n_used_source_nodes*sizeof(uint),
		       cudaMemcpyDeviceToHost));

  for (uint i=0; i<n_used_source_nodes; i++) {
    std::cout << "i_used_source: " << i << " unsorted_source_node_index: "
	      << h_unsorted_source_node_index[i]
	      << " i_unsorted_source_arr: "
	      << h_i_unsorted_source_arr[i] << "\n";
  }
#endif
  //////////////////////////////

  // Sort the arrays using unsorted_source_node_index as key
  // and i_source as value -> sorted_source_node_index


  // Determine temporary storage requirements for RadixSort
  void *d_sort_storage = NULL;
  size_t sort_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(d_sort_storage, sort_storage_bytes,
				  d_unsorted_source_node_index,
				  d_sorted_source_node_index,
				  d_i_unsorted_source_arr,
				  d_i_sorted_source_arr,
				  n_used_source_nodes);
  // Allocate temporary storage
  CUDAMALLOCCTRL("&d_sort_storage",&d_sort_storage, sort_storage_bytes);

  // Run sorting operation
  cub::DeviceRadixSort::SortPairs(d_sort_storage, sort_storage_bytes,
				  d_unsorted_source_node_index,
				  d_sorted_source_node_index,
				  d_i_unsorted_source_arr,
				  d_i_sorted_source_arr,
				  n_used_source_nodes);

  //////////////////////////////////////////////////////////////////////
#ifdef CHECKRC
  /// TEMPORARY for check
  uint h_sorted_source_node_index[n_used_source_nodes];
  uint h_i_sorted_source_arr[n_used_source_nodes];
    
  gpuErrchk(cudaMemcpy(h_sorted_source_node_index,
		       d_sorted_source_node_index,
		       n_used_source_nodes*sizeof(uint),
		       cudaMemcpyDeviceToHost));

  gpuErrchk(cudaMemcpy(h_i_sorted_source_arr,
		       d_i_sorted_source_arr,
		       n_used_source_nodes*sizeof(uint),
		       cudaMemcpyDeviceToHost));

  for (uint i=0; i<n_used_source_nodes; i++) {
    std::cout << "i_used_source: " << i << " sorted_source_node_index: "
	      << h_sorted_source_node_index[i]
	      << " i_sorted_source_arr: "
	      << h_i_sorted_source_arr[i] << "\n";
  }
#endif
  //////////////////////////////


  // !!!!!!!!!!!!!!!!  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  // Allocate array of local source node map blocks
  // and copy their address from host to device
  uint n_blocks = h_local_source_node_map[target_host].size();
  uint **d_node_map = NULL;
  // get current number of elements in the map
  uint n_node_map;
  //std::cout << "ok2 th " << target_host << "\n"; 
  gpuErrchk(cudaMemcpy(&n_node_map,
		       &d_n_local_source_node_map[target_host], sizeof(uint),
		       cudaMemcpyDeviceToHost));
  
    
  if (n_blocks>0) {
    // check for consistency between number of elements
    // and number of blocks in the map
    uint tmp_n_blocks = (n_node_map - 1) / h_node_map_block_size + 1;
    if (tmp_n_blocks != n_blocks) {
      std::cerr << "Inconsistent number of elements "
		<< n_node_map << " and number of blocks "
		<< n_blocks << " in local_source_node_map\n";
      exit(-1);
    }
    CUDAMALLOCCTRL("&d_node_map",&d_node_map, n_blocks*sizeof(uint*));
    gpuErrchk(cudaMemcpy(d_node_map,
			 &h_local_source_node_map[target_host][0],
			 n_blocks*sizeof(uint*),
			 cudaMemcpyHostToDevice));
  }

  // Allocate boolean array for flagging remote source nodes not yet mapped
  // and initialize all elements to 0 (false)
  bool *d_node_to_map;
  CUDAMALLOCCTRL("&d_node_to_map",&d_node_to_map,
		 n_used_source_nodes*sizeof(bool));
  gpuErrchk(cudaMemset(d_node_to_map, 0, n_used_source_nodes*sizeof(bool)));
  // Allocate number of nodes to be mapped and initialize it to 0 
  uint *d_n_node_to_map;
  CUDAMALLOCCTRL("&d_n_node_to_map",&d_n_node_to_map, sizeof(uint));
  gpuErrchk(cudaMemset(d_n_node_to_map, 0, sizeof(uint)));

  // launch kernel that searches remote source nodes indexes in the map,
  // flags the nodes not yet mapped and counts them
  searchNodeIndexNotInMapKernel<<<(n_used_source_nodes+1023)/1024, 1024>>>
    (d_node_map, n_node_map, d_sorted_source_node_index, d_node_to_map,
     d_n_node_to_map, n_used_source_nodes);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  uint h_n_node_to_map;
    
  gpuErrchk(cudaMemcpy(&h_n_node_to_map, d_n_node_to_map, sizeof(uint),
		       cudaMemcpyDeviceToHost));

  //////////////////////////////////////////////////////////////////////
#ifdef CHECKRC
  /// TEMPORARY for check
  std::cout << "n_node_to_map: " << h_n_node_to_map << "\n";
  
  bool h_node_to_map[n_used_source_nodes];
    
  gpuErrchk(cudaMemcpy(h_node_to_map, d_node_to_map,
		       n_used_source_nodes*sizeof(bool),
		       cudaMemcpyDeviceToHost));

  for (uint i=0; i<n_used_source_nodes; i++) {
    std::cout << "i_used_source: " << i << " sorted_source_node_index: "
	      << h_sorted_source_node_index[i]
	      << " node_to_map: " << h_node_to_map[i] << "\n";
  }
#endif
  //////////////////////////////

  // Check if new blocks are required for the map
  uint new_n_blocks = (n_node_map + h_n_node_to_map - 1)
    / h_node_map_block_size + 1;

#ifdef CHECKRC
  std::cout << "new_n_blocks: " << new_n_blocks << "\n";
#endif
  
  // if new blocks are required for the map, allocate them
  if (new_n_blocks != n_blocks) {
    // Allocate GPU memory for new remote-source-node-map blocks
    allocLocalSourceNodeMapBlocks(h_local_source_node_map[target_host],
				   h_node_map_block_size, new_n_blocks);
    // free d_node_map
    if (n_blocks>0) {
      CUDAFREECTRL("d_node_map",d_node_map);
    }
    // update number of blocks in the map 
    n_blocks = new_n_blocks;

    // reallocate d_node_map and get it from host
    CUDAMALLOCCTRL("&d_node_map",&d_node_map, n_blocks*sizeof(uint*));
    gpuErrchk(cudaMemcpy(d_node_map,
			 &h_local_source_node_map[target_host][0],
			 n_blocks*sizeof(uint*),
			 cudaMemcpyHostToDevice));
  }
  
  // Map the not-yet-mapped source nodes using a kernel
  // similar to the one used for counting
  // In the target host unmapped remote source nodes must be mapped
  // to local nodes from n_nodes to n_nodes + n_node_to_map
  
  // Allocate the index of the nodes to be mapped and initialize it to 0 
  uint *d_i_node_to_map;
  CUDAMALLOCCTRL("&d_i_node_to_map",&d_i_node_to_map, sizeof(uint));
  gpuErrchk(cudaMemset(d_i_node_to_map, 0, sizeof(uint)));

  // launch kernel that checks if nodes are already in map
  // if not insert them in the map
  // In the source host, put in the mapsource_node_index
  insertNodesInMapKernel<<<(n_used_source_nodes+1023)/1024, 1024>>>
    (d_node_map, NULL, 0,
     n_node_map, d_sorted_source_node_index, d_node_to_map,
     d_i_node_to_map, n_used_source_nodes);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  // update number of elements in remote source node map
  n_node_map += h_n_node_to_map;
  //std::cout << "ok1 nnm " << n_node_map << " th " << target_host << "\n";
  gpuErrchk(cudaMemcpy(&d_n_local_source_node_map[target_host],
		       &n_node_map, sizeof(uint), cudaMemcpyHostToDevice));
  
  // check for consistency between number of elements
  // and number of blocks in the map
  uint tmp_n_blocks = (n_node_map - 1) / h_node_map_block_size + 1;
  if (tmp_n_blocks != n_blocks) {
    std::cerr << "Inconsistent number of elements "
	      << n_node_map << " and number of blocks "
	      << n_blocks << " in local_source_node_map\n";
    exit(-1);
  }

  //////////////////////////////////////////////////////////////////////
#ifdef CHECKRC
  /// TEMPORARY for check
  std::cout << "//////////////////////////////////////////////\n";
  std::cout << "UPDATED UNSORTED MAP\n";
  std::cout << "OF LOCAL-SOURCE_NODES\n";
  std::cout << "n_node_map: " << n_node_map << "\n";
  std::cout << "n_blocks: " << n_blocks << "\n";
  std::cout << "block_size: " << h_node_map_block_size << "\n";

  uint block_size = h_node_map_block_size;
  uint h_node_map_block[block_size];
  for (uint ib=0; ib<n_blocks; ib++) {
    gpuErrchk(cudaMemcpy(h_node_map_block,
			 h_local_source_node_map[target_host][ib],
			 block_size*sizeof(uint),
			 cudaMemcpyDeviceToHost));
    std::cout << "\n";
    std::cout << "block " << ib << "\n";
    std::cout << "local source node index\n";
    uint n = block_size;
    if (ib==n_blocks-1) {
      n = (n_node_map - 1) % block_size + 1;
    }
    for (uint i=0; i<n; i++) {
      std::cout << h_node_map_block[i] << "\n"; 
    }
    std::cout << "\n";
  }
#endif
  //////////////////////////////////////////////////////////////////////

  // Sort the WHOLE map source_node_map
  // using block sort algorithm copass_sort
  // typical usage:
  // copass_sort::sort<uint>(key_subarray, n,
  //				       aux_size, d_storage, storage_bytes);
  // Determine temporary storage requirements for copass_sort
  int64_t storage_bytes = 0;
  void *d_storage = NULL;
  copass_sort::sort<uint>
    (&h_local_source_node_map[target_host][0],
     n_node_map, h_node_map_block_size, d_storage, storage_bytes);

#ifdef CHECKRC
  printf("storage bytes for copass sort: %ld\n", storage_bytes);
#endif
  
  // Allocate temporary storage
  CUDAMALLOCCTRL("&d_storage",&d_storage, storage_bytes);

  // Run sorting operation
  copass_sort::sort<uint>
    (&h_local_source_node_map[target_host][0],
     n_node_map, h_node_map_block_size, d_storage, storage_bytes);
  CUDAFREECTRL("d_storage",d_storage);

  //////////////////////////////////////////////////////////////////////
#ifdef CHECKRC
  /// TEMPORARY for check
  std::cout << "//////////////////////////////////////////////\n";
  std::cout << "UPDATED SORTED MAP\n";
  std::cout << "OF LOCAL-SOURCE_NODES\n";
  std::cout << "n_node_map: " << n_node_map << "\n";
  std::cout << "n_blocks: " << n_blocks << "\n";
  std::cout << "block_size: " << block_size << "\n";

  for (uint ib=0; ib<n_blocks; ib++) {
    gpuErrchk(cudaMemcpy(h_node_map_block,
			 h_local_source_node_map[target_host][ib],
			 block_size*sizeof(uint),
			 cudaMemcpyDeviceToHost));
    std::cout << "\n";
    std::cout << "block " << ib << "\n";
    std::cout << "local source node index\n";
    uint n = block_size;
    if (ib==n_blocks-1) {
      n = (n_node_map - 1) % block_size + 1;
    }
    for (uint i=0; i<n; i++) {
      std::cout << h_node_map_block[i] << "\n"; 
    }
    std::cout << "\n";
  }
#endif
  //////////////////////////////////////////////////////////////////////

  // Remove temporary new connections in source host !!!!!!!!!!!
  // potential problem: check that number of blocks is an independent variable
  // not calculated from NConn
  // connect.cu riga 462. Corrected but better keep an eye
  // also, hopefully the is no global device variable for NConn
  NConn = old_n_conn; 

  return 0;
}

__global__ void MapIndexToSpikeBufferKernel(uint n_hosts, uint *host_offset,
					    uint *node_index);


#endif // REMOTECONNECTH

