#ifndef REMOTECONNECTH
#define REMOTECONNECTH
#include <vector>
#include <cub/cub.cuh>
#include "nestgpu.h"

// Arrays that map remote source nodes to local spike buffers
  
// The map is organized in blocks having block size:
extern  __constant__ uint remote_node_map_block_size; // = 100000;

// number of elements in the map for each source host
// n_remote_source_node_map[i_source_host]
// with i_source_host = 0, ..., mpi_proc_num-1 excluding this host itself
extern __device__ uint *n_remote_source_node_map; // [mpi_proc_num];

// remote_source_node_map_index[i_source_host][i_block][i]
extern std::vector< std::vector<uint*> > h_remote_source_node_map_index;
extern __device__ uint ***remote_source_node_map_index;

// local_spike_buffer_map_index[i_source_host][i_block][i]
extern std::vector< std::vector<uint*> > h_local_spike_buffer_map_index;
extern __device__ uint ***local_spike_buffer_map_index;

// Arrays that map local source nodes to remote spike buffers

// number of elements in the map for each target host
// n_local_source_node_map[i_target_host]
// with i_target_host = 0, ..., mpi_proc_num-1 excluding this host itself
extern __device__ uint *n_local_source_node_map; // [mpi_proc_num]; 

// local_source_node_map_index[i_target_host][i_block][i]
extern std::vector< std::vector<uint*> > h_local_source_node_map_index;
extern __device__ uint ***local_source_node_map_index;
extern uint ***d_local_source_node_map_index;

// Boolean array with one boolean value for each connection rule
// - true if the rule always creates at least one outgoing connection
// from each source node (one_to_one, all_to_all, fixed_outdegree)
// - false otherwise (fixed_indegree, fixed_total_number, pairwise_bernoulli)
extern bool *use_all_source_nodes; // [n_connection_rules]:

// Initialize the maps
int RemoteConnectionMapInit(uint n_hosts);

int setUsedSourceNodes(std::vector<uint*> &key_subarray,
		       int64_t old_n_conn, int64_t n_conn,
		       int64_t block_size, int *d_source_node_flag);

// kernel that fills the arrays of nodes actually used by new connections 
template <class T>
__global__ void getUsedSourceNodeIndexKernel(T source, uint n_source,
					     int *n_used_source_nodes,
					     int *source_node_flag,
					     int *u_source_node_idx,
					     int *i_source_arr)
{
  uint i_source = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_source>=n_source) return;
  // Count how many source_node_flag are true using atomic increase
  // on n_used_source_nodes
  if (source_node_flag[i_source] != 0) {
    int pos = atomicAdd(n_used_source_nodes, 1);
    u_source_node_idx[pos] = GetNodeIndex(source, i_source);
    i_source_arr[pos] = i_source;
  }
}

// kernel that counts source nodes actually used in new connections
__global__ void countUsedSourceNodeKernel(uint n_source,
					  int *n_used_source_nodes,
					  int *source_node_flag);


// REMOTE CONNECT FUNCTION
template <class T1, class T2>
int NESTGPU::_RemoteConnect(int source_host, T1 source, int n_source,
			    int target_host, T2 target, int n_target,
			    ConnSpec &conn_spec, SynSpec &syn_spec)
{
  /*
      int RemoteConnect(int source_host, int n_source, int target_host,
		    int n_target,
		    curandGenerator_t &gen,
		    void *d_storage, float time_resolution,
		    std::vector<uint*> &key_subarray,
		    std::vector<connection_struct*> &conn_subarray,
		    int64_t &n_conn, int64_t block_size,
		    int64_t total_num, T1 source, int n_source,
		    T2 target, int n_target,
		    SynSpec &syn_spec
		    ) { //.........
    
    int connect_rnd_seed = master_seed + mpi_proc_num*target_host_index + 1;
    + source_host_index;
    update(master_seed);
  */
  // Check if it is a local connection
  if (MpiId()==source_host && source_host==target_host) {
    return _Connect(source, n_source, target, n_target,
		    conn_spec, syn_spec);
  }
  // Check if target_host matches the MPI ID.
  else if (MpiId()==target_host) {
    return _RemoteConnectSource(source_host, source, n_source,
				target, n_target, conn_spec, syn_spec);
  }
  // Check if target_host matches the MPI ID.
  else if (MpiId()==source_host) {
    return _RemoteConnectTarget(target_host, source, n_source,
				target, n_target, conn_spec, syn_spec);
  }
  
  return 0;
}

// REMOTE CONNECT FUNCTION for target_host matching the MPI ID.
template <class T1, class T2>
int NESTGPU::_RemoteConnectSource(int source_host, T1 source, int n_source,
				  T2 target, int n_target,
				  ConnSpec &conn_spec, SynSpec &syn_spec)
{
  // n_nodes will be the first index for new mapping of remote source nodes
  // to local spike buffers
  int local_spike_buffer_map_index0 = GetNNode();
    
  // check if the flag UseAllSourceNodes[conn_rule] is false
  // if (!use_all_source_nodes_flag) {
    
  // on both the source and target hosts create a temporary array
  // of booleans having size equal to the number of source nodes
    
  int *d_source_node_flag; // [n_source] // each element is initially false
  gpuErrchk(cudaMalloc(&d_source_node_flag, n_source*sizeof(int)));
  std::cout << "d_source_node_flag: " << d_source_node_flag << "\n";
  gpuErrchk(cudaMemset(d_source_node_flag, 0, n_source*sizeof(int)));  
    
  // on the target hosts create a temporary array of integers having size
  // equal to the number of source nodes
    
  int *d_local_node_index; // [n_source]; // only on target host
  gpuErrchk(cudaMalloc(&d_local_node_index, n_source*sizeof(int)));
    
  uint64_t old_n_conn = NConn;
  // The connect command is performed on both source and target host using
  // the same initial seed and using as source node indexes the integers
  // from 0 to n_source_nodes - 1
  //_Connect<int, T2>(0, n_source, target, n_target,
  //		      conn_spec, syn_spec);
  Connect(0, n_source, target, n_target,
	  conn_spec, syn_spec);

  /// TEMPORARY for check
  uint h_source_delay[NConn];
  int h_source[NConn];
  int h_delay[NConn];
  gpuErrchk(cudaMemcpy(h_source_delay, KeySubarray[0],
		       NConn*sizeof(uint), cudaMemcpyDeviceToHost));
  for (int i=0; i<NConn; i++) {
    h_source[i] = h_source_delay[i] >> h_MaxPortNBits;
    h_delay[i] = h_source_delay[i] & h_PortMask;
    std::cout << "i_conn: " << i << " source: " << h_source[i];
    std::cout << " delay: " << h_delay[i] << "\n";
  }
  //////////////////////////////
    

  // flag source nodes used in at least one new connection
  // Loop on all new connections and set source_node_flag[i_source]=true
  setUsedSourceNodes(KeySubarray, old_n_conn, NConn, h_ConnBlockSize,
		     d_source_node_flag);

  /// TEMPORARY for check
  std::cout << "n_source: " << n_source << "\n";
  int h_source_node_flag[n_source];
  std::cout << "d_source_node_flag: " << d_source_node_flag << "\n";
    
  gpuErrchk(cudaMemcpy(h_source_node_flag, d_source_node_flag,
		       n_source*sizeof(int), cudaMemcpyDeviceToHost));

  for (int i=0; i<n_source; i++) {
    std::cout << "i_source: " << i << " source_node_flag: "
	      << h_source_node_flag[i] << "\n";
  }
  //////////////////////////////
    
  // Count source nodes actually used in new connections
  // Allocate n_used_source_nodes and initialize it to 0
  int *d_n_used_source_nodes;
  gpuErrchk(cudaMalloc(&d_n_used_source_nodes, sizeof(int)));
  gpuErrchk(cudaMemset(d_n_used_source_nodes, 0, sizeof(int)));  
  // Launch kernel to count used nodes
  countUsedSourceNodeKernel<<<(n_source+1023)/1024, 1024>>>
    (n_source, d_n_used_source_nodes, d_source_node_flag);
  // copy result from GPU to CPU memory
  int n_used_source_nodes;
  gpuErrchk(cudaMemcpy(&n_used_source_nodes, d_n_used_source_nodes,
		       sizeof(int), cudaMemcpyDeviceToHost));

  // TEMPORARY
  std::cout << "n_used_source_nodes: " << n_used_source_nodes << "\n";
  //
    
  // Define and allocate arrays of size n_used_source_nodes
  int *d_unsorted_source_node_index; // [n_used_source_nodes];
  int *d_sorted_source_node_index; // [n_used_source_nodes];
  // i_source_arr are the positions in the arrays source_node_flag
  // and local_node_index 
  int *d_i_unsorted_source_arr; // [n_used_source_nodes];
  int *d_i_sorted_source_arr; // [n_used_source_nodes];
  bool *d_source_node_index_to_be_mapped; //[n_used_source_nodes]; // initially false
  gpuErrchk(cudaMalloc(&d_unsorted_source_node_index,
		       n_used_source_nodes*sizeof(int)));
  gpuErrchk(cudaMalloc(&d_sorted_source_node_index,
		       n_used_source_nodes*sizeof(int)));
  gpuErrchk(cudaMalloc(&d_i_unsorted_source_arr,
		       n_used_source_nodes*sizeof(int)));
  gpuErrchk(cudaMalloc(&d_i_sorted_source_arr,
		       n_used_source_nodes*sizeof(int)));
  gpuErrchk(cudaMalloc(&d_source_node_index_to_be_mapped,
		       n_used_source_nodes*sizeof(int8_t)));
  // source_node_index_to_be_mapped is initially false
  gpuErrchk(cudaMemset(d_source_node_index_to_be_mapped, 0,
		       n_used_source_nodes*sizeof(int8_t)));
    
  // Fill the arrays of nodes actually used by new connections 
  // Reset n_used_source_nodes to 0
  gpuErrchk(cudaMemset(d_n_used_source_nodes, 0, sizeof(int)));  
  // Launch kernel to fill the arrays
  getUsedSourceNodeIndexKernel<<<(n_source+1023)/1024, 1024>>>
    (source, n_source, d_n_used_source_nodes, d_source_node_flag,
     d_unsorted_source_node_index, d_i_unsorted_source_arr);
      
  /// TEMPORARY for check
  std::cout << "n_used_source_nodes: " << n_used_source_nodes << "\n";
  int h_unsorted_source_node_index[n_used_source_nodes];
  int h_i_unsorted_source_arr[n_used_source_nodes];
    
  gpuErrchk(cudaMemcpy(h_unsorted_source_node_index,
		       d_unsorted_source_node_index,
		       n_used_source_nodes*sizeof(int),
		       cudaMemcpyDeviceToHost));

  gpuErrchk(cudaMemcpy(h_i_unsorted_source_arr,
		       d_i_unsorted_source_arr,
		       n_used_source_nodes*sizeof(int),
		       cudaMemcpyDeviceToHost));

  for (int i=0; i<n_used_source_nodes; i++) {
    std::cout << "i_used_source: " << i << " unsorted_source_node_index: "
	      << h_unsorted_source_node_index[i]
	      << " i_unsorted_source_arr: "
	      << h_i_unsorted_source_arr[i] << "\n";
  }
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
  gpuErrchk(cudaMalloc(&d_sort_storage, sort_storage_bytes));

  // Run sorting operation
  cub::DeviceRadixSort::SortPairs(d_sort_storage, sort_storage_bytes,
				  d_unsorted_source_node_index,
				  d_sorted_source_node_index,
				  d_i_unsorted_source_arr,
				  d_i_sorted_source_arr,
				  n_used_source_nodes);

  /// TEMPORARY for check
  int h_sorted_source_node_index[n_used_source_nodes];
  int h_i_sorted_source_arr[n_used_source_nodes];
    
  gpuErrchk(cudaMemcpy(h_sorted_source_node_index,
		       d_sorted_source_node_index,
		       n_used_source_nodes*sizeof(int),
		       cudaMemcpyDeviceToHost));

  gpuErrchk(cudaMemcpy(h_i_sorted_source_arr,
		       d_i_sorted_source_arr,
		       n_used_source_nodes*sizeof(int),
		       cudaMemcpyDeviceToHost));

  for (int i=0; i<n_used_source_nodes; i++) {
    std::cout << "i_used_source: " << i << " sorted_source_node_index: "
	      << h_sorted_source_node_index[i]
	      << " i_sorted_source_arr: "
	      << h_i_sorted_source_arr[i] << "\n";
  }
  //////////////////////////////


  // Initialize n_new_source_node_map to 0

  int n_new_source_node_map = 0;

  // Check for sorted_source_node_index unique values:
  // - either if it is the first of the array (i_thread = 0)
  // - or it is different from previous
  // CUDA KERNEL input for target host: remote_source_node_map_index[i_source_host]->source_node_map_index, local_spike_buffer_map_index[i_source_host]->spike_buffer_map_index
// CUDA KERNEL input for source host: local_source_node_map_index[i_target_host]->source_node_map_index, remote_spike_buffer_map_index[i_target_host]->spike_buffer_map_index

if (i_thread == 0 || sorted_source_node_index[i_thread] != sorted_source_node_index[i_thread-1]) {

// b12) In such case search sorted_source_node_index in the map (locate)
// If it is not in the map then flag it to be mapped and atomic increase n_new_source_node_map

search(sorted_source_node_index[i_thread], source_node_map_index,...)
if (not_found) {
  source_node_index_to_be_mapped[i_thread] = true;
  atomicInc(n_new_source_nodes_map);
}







  
  return 0;
}

#endif // REMOTECONNECTH

