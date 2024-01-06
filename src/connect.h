/*
 *  connect.h
 *
 *  This file is part of NEST GPU.
 *
 *  Copyright (C) 2021 The NEST Initiative
 *
 *  NEST GPU is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST GPU is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST GPU.  If not, see <http://www.gnu.org/licenses/>.
 *
 */


#ifndef CONNECT_H
#define CONNECT_H

#include <time.h>
#include <sys/time.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>
#include <vector>

#include "cuda_error.h"
#include "copass_kernels.h"
#include "copass_sort.h"
#include "connect_spec.h"
//#include "nestgpu.h"
#include "distribution.h"
#include "utilities.h"
#include "node_group.h"

typedef uint inode_t;
typedef uint iconngroup_t;

class Connection
{
public:
  virtual int calibrate() = 0;
  
  virtual int setMaxNodeNBits(int max_node_nbits) = 0;

  virtual int setMaxSynNBits(int max_syn_nbits) = 0;

  virtual int getMaxNodeNBits() = 0;

  virtual int getMaxPortNBits() = 0;

  virtual int getMaxSynNBits() = 0;

  virtual int getMaxDelayNum() = 0;

  virtual int getNImageNodes() = 0;

  virtual bool getRevConnFlag() = 0;

  virtual int getNRevConn() = 0;

  virtual uint* getDevRevSpikeNumPt() = 0;

  virtual int* getDevRevSpikeNConnPt() = 0;

  virtual uint* getDevNTargetHosts() = 0;
  
  virtual uint** getDevNodeTargetHosts() = 0;

  virtual uint** getDevNodeTargetHostIMap() = 0;
  
  virtual int organizeConnections(inode_t n_node) = 0;

  virtual int connect(inode_t source, inode_t n_source,
		      inode_t target, inode_t n_target,
		      ConnSpec &conn_spec, SynSpec &syn_spec) = 0;

  virtual int connect(inode_t source, inode_t n_source,
		      inode_t *target, inode_t n_target,
		      ConnSpec &conn_spec, SynSpec &syn_spec) = 0;

  virtual int connect(inode_t *source, inode_t n_source,
		      inode_t target, inode_t n_target,
		      ConnSpec &conn_spec, SynSpec &syn_spec) = 0;

  virtual int connect(inode_t *source, inode_t n_source,
		      inode_t *target, inode_t n_target,
		      ConnSpec &conn_spec, SynSpec &syn_spec) = 0;

  int isConnectionIntParam(std::string param_name);

  int isConnectionFloatParam(std::string param_name);
  
  int getConnectionIntParamIndex(std::string param_name);
  
  int getConnectionFloatParamIndex(std::string param_name);

  virtual int getConnectionFloatParam(int64_t *conn_ids, int64_t n_conn,
			      float *h_param_arr, std::string param_name) = 0;
			      
  virtual int getConnectionIntParam(int64_t *conn_ids, int64_t n_conn,
			    int *h_param_arr, std::string param_name) = 0;

  virtual int setConnectionFloatParam(int64_t *conn_ids, int64_t n_conn,
			      float val, std::string param_name) = 0;
  
  virtual int setConnectionFloatParamDistr(int64_t *conn_ids, int64_t n_conn,
				   std::string param_name) = 0;
				 
  virtual int setConnectionIntParamArr(int64_t *conn_ids, int64_t n_conn,
			       int *h_param_arr, std::string param_name) = 0;

  virtual int setConnectionIntParam(int64_t *conn_ids, int64_t n_conn,
			    int val, std::string param_name) = 0;
			  
  virtual int64_t *getConnections(inode_t *i_source_pt, inode_t n_source,
			  inode_t *i_target_pt, inode_t n_target,
			  int syn_group, int64_t *n_conn) = 0;
  
  virtual int getConnectionStatus(int64_t *conn_ids, int64_t n_conn,
			  inode_t *source, inode_t *target,
			  int *port, int *syn_group, float *delay,
			  float *weight) = 0;

  virtual int buildDirectConnections(inode_t i_node_0, inode_t n_node,
				     int64_t &i_conn0, int64_t &n_dir_conn,
				     int &max_delay, float* &d_mu_arr,
				     void* &d_poiss_key_array) = 0;

  virtual int sendDirectSpikes(long long time_idx,
			       int64_t i_conn0, int64_t n_dir_conn,
			       inode_t n_node, int max_delay,
			       float *d_mu_arr,
			       void *d_poiss_key_array,
			       curandState *d_curand_state) = 0;

  virtual int organizeDirectConnections(void* &d_poiss_key_array_data_pt,
					void* &d_poiss_subarray,
					int64_t* &d_poiss_num,
					int64_t* &d_poiss_sum,
					void* &d_poiss_thresh) = 0;

  virtual int addOffsetToExternalNodeIds(uint n_local_nodes) = 0;

  virtual int freeConnectionKey() = 0;

  virtual int revSpikeInit(uint n_spike_buffers) = 0;
  
  virtual int resetConnectionSpikeTimeUp() = 0;

  virtual int resetConnectionSpikeTimeDown() = 0;

  virtual int setRandomSeed(unsigned long long seed) = 0;
  
  // set number of hosts
  virtual int setNHosts(int n_hosts) = 0;
  
  // set index of this host
  virtual int setThisHost(int this_host) = 0;
  

  virtual int remoteConnectionMapInit() = 0;

  virtual int remoteConnectionMapCalibrate(inode_t n_nodes) = 0;

  virtual int remoteConnect(int source_host, inode_t source, inode_t n_source,
			    int target_host, inode_t target, inode_t n_target,
			    ConnSpec &conn_spec, SynSpec &syn_spec) = 0;

  virtual int remoteConnect(int source_host, inode_t *source, inode_t n_source,
			    int target_host, inode_t target, inode_t n_target,
			    ConnSpec &conn_spec, SynSpec &syn_spec) = 0;

  virtual int remoteConnect(int source_host, inode_t source, inode_t n_source,
			    int target_host, inode_t *target, inode_t n_target,
			    ConnSpec &conn_spec, SynSpec &syn_spec) = 0;

  virtual int remoteConnect(int source_host, inode_t *source, inode_t n_source,
			    int target_host, inode_t *target, inode_t n_target,
			    ConnSpec &conn_spec, SynSpec &syn_spec) = 0;
  
  virtual int addOffsetToSpikeBufferMap(inode_t n_nodes) = 0;

};


template <class ConnKeyT, class ConnStructT>
class ConnectionTemplate : public Connection
{
  //////////////////////////////////////////////////
  // Member variables
  //////////////////////////////////////////////////
  static const int conn_seed_offset_ = 12345;
  
  int64_t conn_block_size_;
  
  int64_t n_conn_;
  
  std::vector<ConnKeyT*> conn_key_vect_;
  
  std::vector<ConnStructT*> conn_struct_vect_;
  
  float time_resolution_;
  
  std::vector < std::vector <curandGenerator_t > > conn_random_generator_;
  
  Distribution *distribution_;

  // pointer to temporary storage in device memory
  void *d_conn_storage_;
  
  // maximum number of bits used to represent node index
  int max_node_nbits_;

  // maximum number of bits used to represent delays
  int max_delay_nbits_;

  // maximum number of bits used to represent synapse group index
  int max_syn_nbits_;

  // maximum number of bits used to represent receptor port index
  int max_port_nbits_;

  // maximum number of bits used to represent receptor port index
  // and synapse group index
  int max_port_syn_nbits_;

  // bit mask used to extract source node index
  uint source_mask_;

  // bit mask used to extract delay
  uint delay_mask_;
  
  // bit mask used to extract target node index
  uint target_mask_;

  // bit mask used to extract synapse group index
  uint syn_mask_;

  // bit mask used to extract port index
  uint port_mask_;

  // bit mask used to extract port and synapse group index
  uint port_syn_mask_;

  iconngroup_t *d_conn_group_idx0_;

  int64_t *d_conn_group_iconn0_;

  int *d_conn_group_delay_;

  iconngroup_t tot_conn_group_num_;

  int max_delay_num_;

  ConnKeyT** d_conn_key_array_;

  ConnStructT** d_conn_struct_array_;
  
  //////////////////////////////////////////////////
  // Remote-connection-related member variables
  //////////////////////////////////////////////////
  int this_host_;
  
  int n_hosts_;
  
  int n_image_nodes_;
  
  // The arrays that map remote source nodes to local spike buffers
  // are organized in blocks having block size:
  uint node_map_block_size_; // = 100000;

  // number of elements in the map for each source host
  // n_remote_source_node_map[i_source_host]
  // with i_source_host = 0, ..., n_hosts-1 excluding this host itself
  std::vector<uint> h_n_remote_source_node_map_;
  uint *d_n_remote_source_node_map_;

  // remote_source_node_map_[i_source_host][i_block][i]
  std::vector< std::vector<uint*> > h_remote_source_node_map_;

  // local_spike_buffer_map[i_source_host][i_block][i]
  std::vector< std::vector<uint*> > h_local_spike_buffer_map_;
  uint ***d_local_spike_buffer_map_;
  
  // hd_local_spike_buffer_map_[i_source_host] vector of pointers to gpu memory
  std::vector<uint**> hd_local_spike_buffer_map_;

  // Arrays that map local source nodes to remote spike buffers
  // number of elements in the map for each target host
  // n_local_source_node_map[i_target_host]
  // with i_target_host = 0, ..., n_hosts-1 excluding this host itself
  uint *d_n_local_source_node_map_;
  std::vector<uint> h_n_local_source_node_map_;
  
  // local_source_node_map[i_target_host][i_block][i]
  std::vector< std::vector<uint*> > h_local_source_node_map_;
  uint ***d_local_source_node_map_;

  // hd_local_source_node_map_[i_target_host] vector of pointers to gpu memory
  std::vector<uint**> hd_local_source_node_map_;

  // number of remote target hosts on which each local node
  //has outgoing connections
  uint *d_n_target_hosts_; // [n_nodes]
  // cumulative sum of d_n_target_hosts
  uint *d_n_target_hosts_cumul_; // [n_nodes+1]

  // Global array with remote target hosts indexes of all nodes
  // target_host_array[total_num] where total_num is the sum
  // of n_target_hosts[i_node] on all nodes
  uint *d_target_host_array_;
  // pointer to the starting position in target_host_array
  // of the target hosts for the node i_node
  uint **d_node_target_hosts_; // [i_node]
  
  // Global array with remote target hosts map indexes of all nodes
  // target_host_i_map[total_num] where total_num is the sum
  // of n_target_hosts[i_node] on all nodes
  uint *d_target_host_i_map_;
  // pointer to the starting position in target_host_i_map array
  // of the target host map indexes for the node i_node
  uint **d_node_target_host_i_map_; // [i_node]

  // node map index
  uint **d_node_map_index_; // [i_node]

  // Boolean array with one boolean value for each connection rule
  // - true if the rule always creates at least one outgoing connection
  // from each source node (one_to_one, all_to_all, fixed_outdegree)
  // - false otherwise (fixed_indegree, fixed_total_number, pairwise_bernoulli)
  bool *use_all_source_nodes_; // [n_connection_rules]:

  //////////////////////////////////////////////////
  // reverse-connection-related member variables
  //////////////////////////////////////////////////
  bool rev_conn_flag_;
  
  bool spike_time_flag_;
  
  unsigned short *d_conn_spike_time_; // [n_conn_];
  
  int64_t n_rev_conn_;
  
  uint *d_rev_spike_num_;
  
  uint *d_rev_spike_target_;
  
  int *d_rev_spike_n_conn_;
  
  int64_t *d_rev_conn_; //[i] i=0,..., n_rev_conn_ - 1;
  
  int *d_target_rev_conn_size_; //[i] i=0,..., n_neuron-1;
  
  int64_t **d_target_rev_conn_; //[i][j] j=0,...,rev_conn_size_[i]-1

  //////////////////////////////////////////////////
  // class ConnectionTemplate methods
  //////////////////////////////////////////////////
public:
  ConnectionTemplate();

  int init();
  
  int calibrate();

  int initConnRandomGenerator();

  int freeConnRandomGenerator();

  int setRandomSeed(unsigned long long seed);
  
  int setMaxNodeNBits(int max_node_nbits);

  int setMaxSynNBits(int max_syn_nbits);

  int getMaxNodeNBits() {return max_node_nbits_;}

  int getMaxPortNBits() {return max_port_nbits_;}

  int getMaxSynNBits() {return max_syn_nbits_;}

  int getMaxDelayNum() {return max_delay_num_;}

  int getNImageNodes() {return n_image_nodes_;}

  bool getRevConnFlag() {return rev_conn_flag_;}
  
  int getNRevConn() {return n_rev_conn_;}

  uint* getDevRevSpikeNumPt() {return d_rev_spike_num_;}
  
  int* getDevRevSpikeNConnPt() {return d_rev_spike_n_conn_;}

  uint* getDevNTargetHosts() {return d_n_target_hosts_;}
  
  uint** getDevNodeTargetHosts() {return d_node_target_hosts_;}

  uint** getDevNodeTargetHostIMap() {return d_node_target_host_i_map_;}
  
  int allocateNewBlocks(int new_n_block);
  
  int freeConnectionKey();
  
  int setConnectionWeights(curandGenerator_t &gen, void *d_storage,
			   ConnStructT *conn_struct_subarray,
			   int64_t n_conn, SynSpec &syn_spec);
  
  int setConnectionDelays(curandGenerator_t &gen, void *d_storage,
			  ConnKeyT *conn_key_subarray,
			  int64_t n_conn, SynSpec &syn_spec);

  void setConnSource(ConnKeyT &conn_key, inode_t source);

  int connect(inode_t source, inode_t n_source,
	      inode_t target, inode_t n_target,
	      ConnSpec &conn_spec, SynSpec &syn_spec)
  {
    return _Connect(source, n_source, target, n_target, conn_spec, syn_spec);
  }

  int connect(inode_t *source, inode_t n_source,
	      inode_t target, inode_t n_target,
	      ConnSpec &conn_spec, SynSpec &syn_spec)
  {
    return _Connect(source, n_source, target, n_target, conn_spec, syn_spec);
  }

  int connect(inode_t source, inode_t n_source,
	      inode_t *target, inode_t n_target,
	      ConnSpec &conn_spec, SynSpec &syn_spec)
  {
    return _Connect(source, n_source, target, n_target, conn_spec, syn_spec);
  }

  int connect(inode_t *source, inode_t n_source,
	      inode_t *target, inode_t n_target,
	      ConnSpec &conn_spec, SynSpec &syn_spec)
  {
    return _Connect(source, n_source, target, n_target, conn_spec, syn_spec);
  }

  template <class T1, class T2>
  int _Connect(T1 source, inode_t n_source, T2 target, inode_t n_target,
		 ConnSpec &conn_spec, SynSpec &syn_spec);
  
  template <class T1, class T2>
  int _Connect(curandGenerator_t &gen, T1 source, inode_t n_source,
	       T2 target, inode_t n_target,
	       ConnSpec &conn_spec, SynSpec &syn_spec);

  template <class T1, class T2>
  int connectOneToOne(curandGenerator_t &gen, T1 source, T2 target,
		      inode_t n_node, SynSpec &syn_spec);

  template <class T1, class T2>
  int connectAllToAll(curandGenerator_t &gen, T1 source, inode_t n_source,
		      T2 target, inode_t n_target, SynSpec &syn_spec);

  template <class T1, class T2>
  int connectFixedTotalNumber(curandGenerator_t &gen,
			      T1 source, inode_t n_source,
			      T2 target, inode_t n_target,
			      int64_t total_num, SynSpec &syn_spec);
  template <class T1, class T2>
  int connectFixedIndegree(curandGenerator_t &gen,
			   T1 source, inode_t n_source,
			   T2 target, inode_t n_target,
			   int indegree, SynSpec &syn_spec);

  template <class T1, class T2>
  int connectFixedOutdegree(curandGenerator_t &gen,
			    T1 source, inode_t n_source,
			    T2 target, inode_t n_target,
			    int outdegree, SynSpec &syn_spec);

public:
  int organizeConnections(inode_t n_node);
  
  int getConnectionFloatParam(int64_t *conn_ids, int64_t n_conn,
			      float *h_param_arr, std::string param_name);
			      
  int getConnectionIntParam(int64_t *conn_ids, int64_t n_conn,
			    int *h_param_arr, std::string param_name);

  int setConnectionFloatParam(int64_t *conn_ids, int64_t n_conn,
			      float val, std::string param_name);
  
  int setConnectionFloatParamDistr(int64_t *conn_ids, int64_t n_conn,
				   std::string param_name);
				 
  int setConnectionIntParamArr(int64_t *conn_ids, int64_t n_conn,
			       int *h_param_arr, std::string param_name);

  int setConnectionIntParam(int64_t *conn_ids, int64_t n_conn,
			    int val, std::string param_name);
			  
  int64_t *getConnections(inode_t *i_source_pt, inode_t n_source,
			  inode_t *i_target_pt, inode_t n_target,
			  int syn_group, int64_t *n_conn);
  
  int getConnectionStatus(int64_t *conn_ids, int64_t n_conn,
			  inode_t *source, inode_t *target,
			  int *port, int *syn_group, float *delay,
			  float *weight);

  //////////////////////////////////////////////////
  // class ConnectionTemplate remote-connection-related methods
  //////////////////////////////////////////////////

  // set number of hosts
  int setNHosts(int n_hosts);
  
  // set index of this host
  int setThisHost(int this_host);
  
  // Initialize the maps
  int remoteConnectionMapInit();

  // Calibrate the maps
  int remoteConnectionMapCalibrate(inode_t n_nodes);

  // Allocate GPU memory for new remote-source-node-map blocks
  int allocRemoteSourceNodeMapBlocks(std::vector<uint*> &i_remote_src_node_map,
				     std::vector<uint*> &i_local_spike_buf_map,
				     uint new_n_block);

  // Allocate GPU memory for new local-source-node-map blocks
  int allocLocalSourceNodeMapBlocks(std::vector<uint*> &i_local_src_node_map,
				    uint new_n_block);

  // Loop on all new connections and set source_node_flag[i_source]=true
  int setUsedSourceNodes(int64_t old_n_conn, uint *d_source_node_flag);

  // Loops on all new connections and replaces the source node index
  // source_node[i_conn] with the value of the element pointed by the
  // index itself in the array local_node_index
  int fixConnectionSourceNodeIndexes(int64_t old_n_conn,
				     uint *d_local_node_index);


  // remote connect functions
  int remoteConnect(int source_host, inode_t source, inode_t n_source,
		    int target_host, inode_t target, inode_t n_target,
		    ConnSpec &conn_spec, SynSpec &syn_spec)
  {
    return _RemoteConnect<inode_t, inode_t>(source_host, source, n_source,
					    target_host, target, n_target,
					    conn_spec, syn_spec);
  }

  int remoteConnect(int source_host, inode_t *source, inode_t n_source,
		    int target_host, inode_t target, inode_t n_target,
		    ConnSpec &conn_spec, SynSpec &syn_spec)
  {
    return _RemoteConnect<inode_t*, inode_t>(source_host, source, n_source,
					     target_host, target, n_target,
					     conn_spec, syn_spec);
  }

  int remoteConnect(int source_host, inode_t source, inode_t n_source,
		    int target_host, inode_t *target, inode_t n_target,
		    ConnSpec &conn_spec, SynSpec &syn_spec)
  {
    return _RemoteConnect<inode_t, inode_t*>(source_host, source, n_source,
					     target_host, target, n_target,
					     conn_spec, syn_spec);
  }

  int remoteConnect(int source_host, inode_t *source, inode_t n_source,
		    int target_host, inode_t *target, inode_t n_target,
		    ConnSpec &conn_spec, SynSpec &syn_spec)
  {
    return _RemoteConnect<inode_t*, inode_t*>(source_host, source, n_source,
					      target_host, target, n_target,
					      conn_spec, syn_spec);
  }
  
  template <class T1, class T2>
  int _RemoteConnect(int source_host, T1 source, inode_t n_source,
		     int target_host, T2 target, inode_t n_target,
		     ConnSpec &conn_spec, SynSpec &syn_spec);

  int addOffsetToExternalNodeIds(uint n_local_nodes);

  // REMOTE CONNECT FUNCTION for target_host matching this_host
  template <class T1, class T2>
  int remoteConnectSource(int source_host, T1 source, inode_t n_source,
			  T2 target, inode_t n_target,
			  ConnSpec &conn_spec, SynSpec &syn_spec);

  // REMOTE CONNECT FUNCTION for source_host matching this_host
  template <class T1, class T2>
  int remoteConnectTarget(int target_host, T1 source, inode_t n_source,
			  T2 target, inode_t n_target,
			  ConnSpec &conn_spec, SynSpec &syn_spec);

  int addOffsetToSpikeBufferMap(inode_t n_nodes);
  
  //////////////////////////////////////////////////
  // class ConnectionTemplate reverse-connection-related methods
  //////////////////////////////////////////////////
  int revSpikeFree();
  
  int revSpikeInit(uint n_spike_buffers);
  
  int resetConnectionSpikeTimeUp();

  int resetConnectionSpikeTimeDown();

  //////////////////////////////////////////////////
  // class ConnectionTemplate direct-connection-related methods
  //////////////////////////////////////////////////
  int buildDirectConnections(inode_t i_node_0, inode_t n_node,
			     int64_t &i_conn0, int64_t &n_dir_conn,
			     int &max_delay, float* &d_mu_arr,
			     void* &d_poiss_key_array);
  
  int organizeDirectConnections(void* &d_poiss_key_array_data_pt,
				void* &d_poiss_subarray,
				int64_t* &d_poiss_num,
				int64_t* &d_poiss_sum,
				void* &d_poiss_thresh);


  int sendDirectSpikes(long long time_idx,
		       int64_t i_conn0, int64_t n_dir_conn,
		       inode_t n_node, int max_delay,
		       float *d_mu_arr,
		       void *d_poiss_key_array,
		       curandState *d_curand_state);

};

namespace poiss_conn
{
  extern void *d_poiss_key_array_data_pt;
  extern void *d_poiss_subarray;  
  extern int64_t *d_poiss_num;
  extern int64_t *d_poiss_sum;
  extern void *d_poiss_thresh;
  int organizeDirectConnections(Connection *conn);
};


enum ConnectionFloatParamIndexes {
  i_weight_param = 0,
  i_delay_param,
  N_CONN_FLOAT_PARAM
};

enum ConnectionIntParamIndexes {
  i_source_param = 0,
  i_target_param,
  i_port_param,
  i_syn_group_param,
  N_CONN_INT_PARAM
};

extern __constant__ float NESTGPUTimeResolution;

extern __device__ int16_t *NodeGroupMap;

extern __constant__ NodeGroupStruct NodeGroupArray[];


// maximum number of bits used to represent node index 
extern __device__ int MaxNodeNBits;

// maximum number of bits used to represent delays
extern __device__ int MaxDelayNBits;

// maximum number of bits used to represent synapse group index
extern __device__ int MaxSynNBits;

// maximum number of bits used to represent receptor port index
extern __device__ int MaxPortNBits;

// maximum number of bits used to represent receptor port index
// and synapse group index
extern __device__ int MaxPortSynNBits;

// bit mask used to extract source node index
extern __device__ uint SourceMask;

// bit mask used to extract delay
extern __device__ uint DelayMask;

// bit mask used to extract target node index
extern __device__ uint TargetMask;

// bit mask used to extract synapse group index
extern __device__ uint SynMask;

// bit mask used to extract port index
extern __device__ uint PortMask;

// bit mask used to extract port and synapse group index
extern __device__ uint PortSynMask;

extern __device__ iconngroup_t *ConnGroupIdx0;

extern __device__ int64_t *ConnGroupIConn0;

extern __device__ int *ConnGroupDelay;

extern __device__ int64_t ConnBlockSize;

// it seems that there is no relevant advantage in using a constant array
// however better to keep this option ready and commented
//extern __constant__ uint* ConnKeyArray[];
extern __device__ void* ConnKeyArray;

//extern __constant__ connection_struct* ConnStructArray[];
extern __device__ void* ConnStructArray;

extern __device__ unsigned short *ConnectionSpikeTime;

template <class ConnKeyT>
__device__ __forceinline__ void setConnDelay
(ConnKeyT &conn_key, int delay);

template <class ConnKeyT>
__device__ __forceinline__ void setConnSource
(ConnKeyT &conn_key, inode_t source);

template <class ConnStructT>
__device__ __forceinline__ void setConnTarget
(ConnStructT &conn_struct, inode_t target);

template <class ConnKeyT, class ConnStructT>
__device__ __forceinline__ void setConnPort
(ConnKeyT &conn_key, ConnStructT &conn_struct, int port);

template <class ConnKeyT, class ConnStructT>
__device__ __forceinline__ void setConnSyn
(ConnKeyT &conn_key, ConnStructT &conn_struct, int syn);


template <class ConnKeyT>
__device__ __forceinline__ int getConnDelay(const ConnKeyT &conn_key);

template <class ConnKeyT>
__device__ __forceinline__ inode_t getConnSource(ConnKeyT &conn_key);

template <class ConnStructT>
__device__ __forceinline__ inode_t getConnTarget(ConnStructT &conn_struct);

template <class ConnKeyT, class ConnStructT>
__device__ __forceinline__ int getConnPort
(ConnKeyT &conn_key, ConnStructT &conn_struct);

template <class ConnKeyT, class ConnStructT>
__device__ __forceinline__ int getConnSyn
(ConnKeyT &conn_key, ConnStructT &conn_struct);

template <class ConnKeyT, class ConnStructT>
__device__ __forceinline__ bool getConnRemoteFlag
(ConnKeyT &conn_key, ConnStructT &conn_struct);

template <class ConnKeyT, class ConnStructT>
__device__ __forceinline__ void clearConnRemoteFlag
(ConnKeyT &conn_key, ConnStructT &conn_struct);


template<class ConnStructT>
__global__ void setWeights(ConnStructT *conn_struct_subarray, float weight,
			   int64_t n_conn)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) return;
  conn_struct_subarray[i_conn].weight = weight;
}

template<class ConnStructT>
__global__ void setWeights(ConnStructT *conn_struct_subarray, float *arr_val,
			   int64_t n_conn)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) return;
  conn_struct_subarray[i_conn].weight = arr_val[i_conn];
}


template<class ConnKeyT>
__global__ void setDelays(ConnKeyT *conn_key_subarray, float *arr_val,
			  int64_t n_conn, float time_resolution)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) return;
  int delay = (int)round(arr_val[i_conn]/time_resolution);
  delay = max(delay,1);
  setConnDelay<ConnKeyT>(conn_key_subarray[i_conn], delay);
}

template<class ConnKeyT>
__global__ void setDelays(ConnKeyT *conn_key_subarray, float fdelay,
			  int64_t n_conn, float time_resolution)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) return;
  int delay = (int)round(fdelay/time_resolution);
  delay = max(delay,1);
  setConnDelay<ConnKeyT>(conn_key_subarray[i_conn], delay);
}


template <class ConnKeyT, class ConnStructT>
__global__ void setPort(ConnKeyT *conn_key_subarray,
			ConnStructT *conn_struct_subarray, int port,
			int64_t n_conn)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) return;
  setConnPort<ConnKeyT, ConnStructT>(conn_key_subarray[i_conn],
				     conn_struct_subarray[i_conn],
				     port);
}


template <class ConnKeyT, class ConnStructT>
__global__ void setSynGroup(ConnKeyT *conn_key_subarray,
			    ConnStructT *conn_struct_subarray,
			    int syn_group, int64_t n_conn)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) return;
  setConnSyn<ConnKeyT, ConnStructT>(conn_key_subarray[i_conn],
				    conn_struct_subarray[i_conn],
				    syn_group);
}


template <class ConnKeyT, class ConnStructT>
__global__ void setPortSynGroup(ConnKeyT *conn_key_subarray,
				ConnStructT *conn_struct_subarray,
				int port,
				int syn_group,
				int64_t n_conn)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) return;
  setConnPort<ConnKeyT, ConnStructT>(conn_key_subarray[i_conn],
				     conn_struct_subarray[i_conn],
				     port);
  setConnSyn<ConnKeyT, ConnStructT>(conn_key_subarray[i_conn],
				    conn_struct_subarray[i_conn],
				    syn_group);  
}

__global__ void setSourceTargetIndexKernel(uint64_t n_src_tgt, inode_t n_source,
					   inode_t n_target,
					   uint64_t *d_src_tgt_arr,
					   inode_t *d_src_arr,
					   inode_t *d_tgt_arr);

__global__ void setConnGroupNum(inode_t n_compact,
				iconngroup_t *conn_group_num,
				iconngroup_t *conn_group_idx0_compact,
				inode_t *conn_group_source_compact);


__global__ void setConnGroupIConn0(int64_t n_block_conn,
				   int *conn_group_iconn0_mask,
				   iconngroup_t *conn_group_iconn0_mask_cumul,
				   int64_t *conn_group_iconn0, int64_t i_conn0,
				   iconngroup_t *offset);


template <class T>
__global__ void setConnGroupNewOffset(T *offset, T *add_offset)
{
  *offset = *offset + *add_offset;
}


template <class ConnKeyT>
__global__ void buildConnGroupIConn0Mask(ConnKeyT *conn_key_subarray,
					 ConnKeyT *conn_key_subarray_prev,
					 int64_t n_block_conn,
					 int *conn_group_iconn0_mask)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_block_conn) return;
  ConnKeyT val = conn_key_subarray[i_conn];
  ConnKeyT prev_val;
  inode_t prev_source;
  int prev_delay;
  if (i_conn==0) {
    if (conn_key_subarray_prev != NULL) {
      prev_val = *conn_key_subarray_prev;
      prev_source = getConnSource<ConnKeyT>(prev_val);
      prev_delay = getConnDelay<ConnKeyT>(prev_val);
    }
    else {
      prev_source = 0;
      prev_delay = -1;      // just to ensure it is different
    }
  }
  else {
    prev_val = conn_key_subarray[i_conn-1];
    prev_source = getConnSource<ConnKeyT>(prev_val);
    prev_delay = getConnDelay<ConnKeyT>(prev_val);
  }
  inode_t source = getConnSource<ConnKeyT>(val);
  int delay = getConnDelay<ConnKeyT>(val);
  if (source != prev_source || delay != prev_delay) {
    conn_group_iconn0_mask[i_conn] = 1;
  }
}


template <class ConnKeyT>
__global__ void setConnGroupIdx0Compact
(ConnKeyT *conn_key_subarray, int64_t n_block_conn, int *conn_group_idx0_mask,
 iconngroup_t *conn_group_iconn0_mask_cumul,
 inode_t *conn_group_idx0_mask_cumul,
 iconngroup_t *conn_group_idx0_compact, inode_t *conn_group_source_compact,
 iconngroup_t *iconn0_offset, inode_t *idx0_offset)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>n_block_conn) return;
  if (i_conn<n_block_conn && conn_group_idx0_mask[i_conn]==0) return;
  iconngroup_t i_group = conn_group_iconn0_mask_cumul[i_conn] + *iconn0_offset;
  inode_t i_source_compact = conn_group_idx0_mask_cumul[i_conn]
    + *idx0_offset;
  conn_group_idx0_compact[i_source_compact] = i_group;
  if (i_conn<n_block_conn) {
    //int source = conn_key_subarray[i_conn] >> MaxPortSynNBits;
    inode_t source = getConnSource<ConnKeyT>(conn_key_subarray[i_conn]);
    conn_group_source_compact[i_source_compact] = source;
  }
}

template <class ConnKeyT>
__global__ void getConnGroupDelay(int64_t block_size,
				  ConnKeyT **conn_key_array,
				  int64_t *conn_group_iconn0,
				  int *conn_group_delay,
				  iconngroup_t conn_group_num)
{
  iconngroup_t conn_group_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (conn_group_idx >= conn_group_num) return;
  int64_t i_conn = conn_group_iconn0[conn_group_idx];
  int i_block = (int)(i_conn / block_size);
  int64_t i_block_conn = i_conn % block_size;
  ConnKeyT &conn_key = conn_key_array[i_block][i_block_conn];
  conn_group_delay[conn_group_idx] = getConnDelay(conn_key);
}

template <class ConnKeyT>
__global__ void buildConnGroupMask(ConnKeyT *conn_key_subarray,
				   ConnKeyT *conn_key_subarray_prev,
				   int64_t n_block_conn,
				   int *conn_group_iconn0_mask,
				   int *conn_group_idx0_mask)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_block_conn) return;
  ConnKeyT val = conn_key_subarray[i_conn];
  ConnKeyT prev_val;
  inode_t prev_source;
  int prev_delay;
  if (i_conn==0) {
    if (conn_key_subarray_prev != NULL) {
      prev_val = *conn_key_subarray_prev;
      //prev_source = prev_val >> MaxPortSynNBits; 
      prev_source = getConnSource<ConnKeyT>(prev_val);
      prev_delay = getConnDelay<ConnKeyT>(prev_val);
    }
    else {
      prev_source = 0;
      prev_delay = -1;      // just to ensure it is different
    }
  }
  else {
    prev_val = conn_key_subarray[i_conn-1];
    //prev_source = prev_val >> MaxPortSynNBits;
    prev_source = getConnSource<ConnKeyT>(prev_val);
    prev_delay = getConnDelay<ConnKeyT>(prev_val);
  }
  //int source = val >> MaxPortSynNBits;
  inode_t source = getConnSource<ConnKeyT>(val);
  if (source != prev_source || prev_delay<0) {
    conn_group_iconn0_mask[i_conn] = 1;
    conn_group_idx0_mask[i_conn] = 1;
  }
  else {
    int delay = getConnDelay<ConnKeyT>(val);
    if (delay != prev_delay) {
      conn_group_iconn0_mask[i_conn] = 1;
    }
  }
}

__device__ __forceinline__
inode_t getNodeIndex(inode_t i_node_0, inode_t i_node_rel)
{
  return i_node_0 + i_node_rel;
}

__device__ __forceinline__
inode_t getNodeIndex(inode_t *i_node_0, inode_t i_node_rel)
{
  return *(i_node_0 + i_node_rel);
}

template <class T, class ConnKeyT>
__global__ void setSource(ConnKeyT *conn_key_subarray, uint *rand_val,
			  int64_t n_conn, T source, inode_t n_source)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) return;
  inode_t i_source = getNodeIndex(source, rand_val[i_conn]%n_source);
  setConnSource<ConnKeyT>(conn_key_subarray[i_conn], i_source);    
}

template <class T, class ConnStructT>
__global__ void setTarget(ConnStructT *conn_struct_subarray, uint *rand_val,
			  int64_t n_conn, T target, inode_t n_target)
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn>=n_conn) return;
  inode_t i_target = getNodeIndex(target, rand_val[i_conn]%n_target);
  setConnTarget<ConnStructT>(conn_struct_subarray[i_conn], i_target);    
}

template <class T1, class T2, class ConnKeyT, class ConnStructT>
__global__ void setOneToOneSourceTarget(ConnKeyT *conn_key_subarray,
					ConnStructT *conn_struct_subarray,
					int64_t n_block_conn,
					int64_t n_prev_conn,
					T1 source, T2 target)
{
  int64_t i_block_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_block_conn>=n_block_conn) return;
  int64_t i_conn = n_prev_conn + i_block_conn;
  inode_t i_source = getNodeIndex(source, (int)(i_conn));
  inode_t i_target = getNodeIndex(target, (int)(i_conn));
  setConnSource<ConnKeyT>(conn_key_subarray[i_block_conn], i_source);
  setConnTarget<ConnStructT>(conn_struct_subarray[i_block_conn], i_target);
}

template <class T1, class T2, class ConnKeyT, class ConnStructT>
__global__ void setAllToAllSourceTarget(ConnKeyT *conn_key_subarray,
					ConnStructT *conn_struct_subarray,
					int64_t n_block_conn,
					int64_t n_prev_conn,
					T1 source, inode_t n_source,
					T2 target, inode_t n_target)
{
  int64_t i_block_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_block_conn>=n_block_conn) return;
  int64_t i_conn = n_prev_conn + i_block_conn;
  inode_t i_source = getNodeIndex(source, (int)(i_conn / n_target));
  inode_t i_target = getNodeIndex(target, (int)(i_conn % n_target));
  setConnSource<ConnKeyT>(conn_key_subarray[i_block_conn], i_source);    
  setConnTarget<ConnStructT>(conn_struct_subarray[i_block_conn], i_target);
}

template <class T, class ConnStructT>
__global__ void setIndegreeTarget(ConnStructT *conn_struct_subarray,
				  int64_t n_block_conn,
				  int64_t n_prev_conn,
				  T target, int indegree)
{
  int64_t i_block_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_block_conn>=n_block_conn) return;
  int64_t i_conn = n_prev_conn + i_block_conn;
  inode_t i_target = getNodeIndex(target, (int)(i_conn / indegree));
  setConnTarget<ConnStructT>(conn_struct_subarray[i_block_conn], i_target);
}

template <class T, class ConnKeyT>
__global__ void setOutdegreeSource(ConnKeyT *conn_key_subarray,
				   int64_t n_block_conn,
				   int64_t n_prev_conn,
				   T source, int outdegree)
{
  int64_t i_block_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_block_conn>=n_block_conn) return;
  int64_t i_conn = n_prev_conn + i_block_conn;
  inode_t i_source = getNodeIndex(source, (int)(i_conn / outdegree));
  setConnSource<ConnKeyT>(conn_key_subarray[i_block_conn], i_source);    
}

// Count number of connections per source-target couple
template <class ConnKeyT, class ConnStructT>
__global__ void countConnectionsKernel(int64_t n_conn, inode_t n_source,
				       inode_t n_target, uint64_t *src_tgt_arr,
				       uint64_t *src_tgt_conn_num,
				       int syn_group)
{
  int64_t i_conn = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; 
  if (i_conn >= n_conn) return;

  int i_block = (int)(i_conn / ConnBlockSize);
  int64_t i_block_conn = i_conn % ConnBlockSize;
  ConnStructT &conn_struct =
    ((ConnStructT **)ConnStructArray)[i_block][i_block_conn];
  ConnKeyT &conn_key =
    ((ConnKeyT **)ConnKeyArray)[i_block][i_block_conn];
  // if (syn_group==-1 || conn.syn_group == syn_group) {
  int syn_group1 = getConnSyn<ConnKeyT, ConnStructT>(conn_key, conn_struct);
  if (syn_group==-1 || (syn_group1 == syn_group)) {
    // First get source and target node index
    inode_t i_target = getConnTarget<ConnStructT>(conn_struct);
    inode_t i_source = getConnSource<ConnKeyT>(conn_key);
    uint64_t i_src_tgt = ((int64_t)i_source << 32) | i_target;
    uint64_t i_arr = locate(i_src_tgt, src_tgt_arr, n_source*n_target);
    if (src_tgt_arr[i_arr] == i_src_tgt) {
      //printf("i_conn %lld i_source %d i_target %d i_src_tgt %lld "
      //     "i_arr %lld\n", i_conn, i_source, i_target, i_src_tgt, i_arr);
      // (atomic)increase the number of connections for source-target couple
      atomicAdd((unsigned long long *)&src_tgt_conn_num[i_arr], 1);
    }
  }
}



// Fill array of connection indexes
template <class ConnKeyT, class ConnStructT>
__global__ void setConnectionsIndexKernel(int64_t n_conn, inode_t n_source,
					  inode_t n_target,
					  uint64_t *src_tgt_arr,
					  uint64_t *src_tgt_conn_num,
					  uint64_t *src_tgt_conn_cumul,
					  int syn_group, int64_t *conn_ids)
{
  int64_t i_conn = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; 
  if (i_conn >= n_conn) return;

  int i_block = (int)(i_conn / ConnBlockSize);
  int64_t i_block_conn = i_conn % ConnBlockSize;
  ConnStructT &conn_struct =
    ((ConnStructT **)ConnStructArray)[i_block][i_block_conn];
  ConnKeyT &conn_key =
    ((ConnKeyT **)ConnKeyArray)[i_block][i_block_conn];
  // if (syn_group==-1 || conn.syn_group == syn_group) {
  int syn_group1 = getConnSyn<ConnKeyT, ConnStructT>(conn_key, conn_struct);
  if (syn_group==-1 || (syn_group1 == syn_group)) {
    // First get source and target node index
    inode_t i_target = getConnTarget<ConnStructT>(conn_struct);
    inode_t i_source = getConnSource<ConnKeyT>(conn_key);
    uint64_t i_src_tgt = ((int64_t)i_source << 32) | i_target;
    uint64_t i_arr = locate(i_src_tgt, src_tgt_arr, n_source*n_target);
    if (src_tgt_arr[i_arr] == i_src_tgt) {
      //printf("i_conn %lld i_source %d i_target %d i_src_tgt %lld "
      //     "i_arr %lld\n", i_conn, i_source, i_target, i_src_tgt, i_arr);
      // (atomic)increase the number of connections for source-target couple
      uint64_t pos =
	atomicAdd((unsigned long long *)&src_tgt_conn_num[i_arr], 1);
      //printf("pos %lld src_tgt_conn_cumul[i_arr] %lld\n",
      //     pos, src_tgt_conn_cumul[i_arr]);
      conn_ids[src_tgt_conn_cumul[i_arr] + pos] = i_conn;
    }
  }
}

//////////////////////////////////////////////////////////////////////
// CUDA Kernel that gets all parameters of an array of n_conn connections,
// identified by the indexes conn_ids[i], and puts them in the arrays
// i_source, i_target, port, syn_group, delay, weight
//////////////////////////////////////////////////////////////////////
template <class ConnKeyT, class ConnStructT>
__global__ void getConnectionStatusKernel
(int64_t *conn_ids, int64_t n_conn, inode_t *source, inode_t *target,
 int *port, int *syn_group, float *delay, float *weight)
{
  int64_t i_arr = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; 
  if (i_arr >= n_conn) return;

   // get connection index, connection block index and index within block
  int64_t i_conn = conn_ids[i_arr];
  int i_block = (int)(i_conn / ConnBlockSize);
  int64_t i_block_conn = i_conn % ConnBlockSize;
  // get connection structure
  ConnStructT &conn_struct =
    ((ConnStructT **)ConnStructArray)[i_block][i_block_conn];
  ConnKeyT &conn_key =
    ((ConnKeyT **)ConnKeyArray)[i_block][i_block_conn];
  // Get source, target, port, synaptic group and delay
  inode_t i_source = getConnSource<ConnKeyT>(conn_key);
  inode_t i_target = getConnTarget<ConnStructT>(conn_struct);
  int i_port = getConnPort<ConnKeyT, ConnStructT>(conn_key, conn_struct);
  int i_syn_group = getConnSyn<ConnKeyT, ConnStructT>(conn_key, conn_struct);
  int i_delay = getConnDelay<ConnKeyT>(conn_key);
  source[i_arr] = i_source;
  target[i_arr] = i_target;
  port[i_arr] = i_port;
  // Get weight and synapse group
  weight[i_arr] = conn_struct.weight;
  syn_group[i_arr] = i_syn_group;
  delay[i_arr] = NESTGPUTimeResolution * i_delay;
}


//////////////////////////////////////////////////////////////////////
// CUDA Kernel that gets a float parameter of an array of n_conn connections,
// identified by the indexes conn_ids[i], and puts it in the array
// param_arr
//////////////////////////////////////////////////////////////////////
template <class ConnKeyT, class ConnStructT>
__global__ void getConnectionFloatParamKernel
(int64_t *conn_ids, int64_t n_conn, float *param_arr, int i_param)
{
  int64_t i_arr = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; 
  if (i_arr >= n_conn) return;

   // get connection index, connection block index and index within block
  int64_t i_conn = conn_ids[i_arr];
  int i_block = (int)(i_conn / ConnBlockSize);
  int64_t i_block_conn = i_conn % ConnBlockSize;
  // get connection structure
  ConnStructT &conn_struct =
    ((ConnStructT **)ConnStructArray)[i_block][i_block_conn];
  ConnKeyT &conn_key =
    ((ConnKeyT **)ConnKeyArray)[i_block][i_block_conn];
  switch (i_param) {
  case i_weight_param: {
    param_arr[i_arr] = conn_struct.weight;
    break;
  }
  case i_delay_param: {
    // Get joined source-delay parameter, then delay
    int i_delay = getConnDelay<ConnKeyT>(conn_key);
    param_arr[i_arr] = NESTGPUTimeResolution * i_delay;
    break;
  }
  }
}

template <class ConnKeyT, class ConnStructT>
//////////////////////////////////////////////////////////////////////
// CUDA Kernel that gets an integer parameter of an array of n_conn connections,
// identified by the indexes conn_ids[i], and puts it in the array
// param_arr
//////////////////////////////////////////////////////////////////////
__global__ void getConnectionIntParamKernel
(int64_t *conn_ids, int64_t n_conn, int *param_arr, int i_param)
{
  int64_t i_arr = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; 
  if (i_arr >= n_conn) return;

   // get connection index, connection block index and index within block
  int64_t i_conn = conn_ids[i_arr];
  int i_block = (int)(i_conn / ConnBlockSize);
  int64_t i_block_conn = i_conn % ConnBlockSize;
  // get connection structure
  ConnStructT &conn_struct =
    ((ConnStructT **)ConnStructArray)[i_block][i_block_conn];
  ConnKeyT &conn_key =
    ((ConnKeyT **)ConnKeyArray)[i_block][i_block_conn];
  switch (i_param) {
  case i_source_param: {
    inode_t i_source = getConnSource<ConnKeyT>(conn_key);
    param_arr[i_arr] = i_source;
    break;
  }
  case i_target_param: {
    inode_t i_target = getConnTarget<ConnStructT>(conn_struct);
    param_arr[i_arr] = i_target;
    break;
  }
  case i_port_param: {
    int i_port = getConnPort<ConnKeyT, ConnStructT>(conn_key, conn_struct);
    param_arr[i_arr] = i_port;
    break;
  }
  case i_syn_group_param: {
    // Get synapse group
    int i_syn_group = getConnSyn<ConnKeyT, ConnStructT>(conn_key, conn_struct);
    param_arr[i_arr] = i_syn_group;
    break;
  }
  }
}

template <class ConnStructT>
//////////////////////////////////////////////////////////////////////
// CUDA Kernel that sets a float parameter of an array of n_conn connections,
// identified by the indexes conn_ids[i], using values from the array
// param_arr
//////////////////////////////////////////////////////////////////////
__global__ void setConnectionFloatParamKernel
(int64_t *conn_ids, int64_t n_conn, float *param_arr, int i_param)
{
  int64_t i_arr = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; 
  if (i_arr >= n_conn) return;

   // get connection index, connection block index and index within block
  int64_t i_conn = conn_ids[i_arr];
  int i_block = (int)(i_conn / ConnBlockSize);
  int64_t i_block_conn = i_conn % ConnBlockSize;
  // get connection structure
  ConnStructT &conn_struct =
    ((ConnStructT **)ConnStructArray)[i_block][i_block_conn];
  switch (i_param) {
  case i_weight_param: {
    conn_struct.weight = param_arr[i_arr]; 
    break;
  }
  }
}

template <class ConnStructT>
//////////////////////////////////////////////////////////////////////
// CUDA Kernel that sets a float parameter of an array of n_conn connections,
// identified by the indexes conn_ids[i], to the value val
//////////////////////////////////////////////////////////////////////
__global__ void setConnectionFloatParamKernel
(int64_t *conn_ids, int64_t n_conn, float val, int i_param)
{
  int64_t i_arr = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; 
  if (i_arr >= n_conn) return;

   // get connection index, connection block index and index within block
  int64_t i_conn = conn_ids[i_arr];
  int i_block = (int)(i_conn / ConnBlockSize);
  int64_t i_block_conn = i_conn % ConnBlockSize;
  // get connection structure
  ConnStructT &conn_struct =
    ((ConnStructT **)ConnStructArray)[i_block][i_block_conn];
  switch (i_param) {
  case i_weight_param: {
    conn_struct.weight = val; 
    break;
  }
  }
}

template <class ConnKeyT, class ConnStructT>
//////////////////////////////////////////////////////////////////////
// CUDA Kernel that sets an integer parameter of an array of n_conn connections,
// identified by the indexes conn_ids[i], using values from the array
// param_arr
//////////////////////////////////////////////////////////////////////
__global__ void setConnectionIntParamKernel
(int64_t *conn_ids, int64_t n_conn, int *param_arr, int i_param)
{
  int64_t i_arr = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; 
  if (i_arr >= n_conn) return;

   // get connection index, connection block index and index within block
  int64_t i_conn = conn_ids[i_arr];
  int i_block = (int)(i_conn / ConnBlockSize);
  int64_t i_block_conn = i_conn % ConnBlockSize;
  // get connection structure
  ConnStructT &conn_struct =
    ((ConnStructT **)ConnStructArray)[i_block][i_block_conn];
  ConnKeyT &conn_key =
    ((ConnKeyT **)ConnKeyArray)[i_block][i_block_conn];
  switch (i_param) {
  case i_target_param: {
    setConnTarget<ConnStructT>(conn_struct, param_arr[i_arr]);
    break;
  }
  case i_port_param: {
    setConnPort<ConnKeyT, ConnStructT>(conn_key, conn_struct, param_arr[i_arr]);
    break;
  }
  case i_syn_group_param: {
    setConnSyn<ConnKeyT, ConnStructT>(conn_key, conn_struct, param_arr[i_arr]);
    break;
  }
  }
}


//////////////////////////////////////////////////////////////////////
// CUDA Kernel that sets an integer parameter of an array of n_conn connections,
// identified by the indexes conn_ids[i], to the value val
//////////////////////////////////////////////////////////////////////
template <class ConnKeyT, class ConnStructT>
__global__ void setConnectionIntParamKernel
(int64_t *conn_ids, int64_t n_conn, int val, int i_param)
{
  int64_t i_arr = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; 
  if (i_arr >= n_conn) return;

   // get connection index, connection block index and index within block
  int64_t i_conn = conn_ids[i_arr];
  int i_block = (int)(i_conn / ConnBlockSize);
  int64_t i_block_conn = i_conn % ConnBlockSize;
  // get connection structure
  ConnStructT &conn_struct =
    ((ConnStructT **)ConnStructArray)[i_block][i_block_conn];
  ConnKeyT &conn_key =
    ((ConnKeyT **)ConnKeyArray)[i_block][i_block_conn];
  switch (i_param) {
      case i_target_param: {
    setConnTarget<ConnStructT>(conn_struct, val);
    break;
  }
  case i_port_param: {
    setConnPort<ConnKeyT, ConnStructT>(conn_key, conn_struct, val);
    break;
  }
  case i_syn_group_param: {
    setConnSyn<ConnKeyT, ConnStructT>(conn_key, conn_struct, val);
    break;
  }
  }
}

/*
// max delay functor
struct MaxDelay
{
  template <class ConnKeyT>
  __device__ __forceinline__
  //uint operator()(const uint &source_delay_a, const uint &source_delay_b)
  //const {
  ConnKeyT operator()(const ConnKeyT &conn_key_a,
		      const ConnKeyT &conn_key_b) const {
    int i_delay_a = getConnDelay<ConnKeyT>(conn_key_a);
    int i_delay_b = getConnDelay<ConnKeyT>(conn_key_b);
    return (i_delay_b > i_delay_a) ? i_delay_b : i_delay_a;
  }
};
*/

// max delay functor
template <class ConnKeyT>
struct MaxDelay
{
  __device__ __forceinline__
  //uint operator()(const uint &source_delay_a, const uint &source_delay_b)
  //const {
  ConnKeyT operator()(const ConnKeyT &conn_key_a,
		      const ConnKeyT &conn_key_b) const {
    int i_delay_a = getConnDelay<ConnKeyT>(conn_key_a);
    int i_delay_b = getConnDelay<ConnKeyT>(conn_key_b);
    return (i_delay_b > i_delay_a) ? i_delay_b : i_delay_a;
  }
};

template <class ConnKeyT>
__global__ void poissGenSubstractFirstNodeIndexKernel(int64_t n_conn,
						      ConnKeyT *poiss_key_array,
						      int i_node_0)
{
  int64_t blockId   = (int64_t)blockIdx.y * gridDim.x + blockIdx.x;
  int64_t i_conn_rel = blockId * blockDim.x + threadIdx.x;
  if (i_conn_rel >= n_conn) {
    return;
  }
  ConnKeyT &conn_key = poiss_key_array[i_conn_rel];
  int i_source_rel = getConnSource<ConnKeyT>(conn_key) - i_node_0;
  setConnSource<ConnKeyT>(conn_key, i_source_rel);
}

template <class ConnKeyT, class ConnStructT>
__global__ void sendDirectSpikeKernel(curandState *curand_state,
				      long long time_idx,
				      float *mu_arr,
				      ConnKeyT *poiss_key_array,
				      int64_t n_conn, int64_t i_conn_0,
				      int64_t block_size, int n_node,
				      int max_delay)
{
  int64_t blockId   = (int64_t)blockIdx.y * gridDim.x + blockIdx.x;
  int64_t i_conn_rel = blockId * blockDim.x + threadIdx.x;
  if (i_conn_rel >= n_conn) {
    return;
  }
  ConnKeyT &conn_key = poiss_key_array[i_conn_rel];
  int i_source = getConnSource<ConnKeyT>(conn_key);
  int i_delay = getConnDelay<ConnKeyT>(conn_key);
  int id = (int)((time_idx - i_delay + 1) % max_delay);
  float mu = mu_arr[id*n_node + i_source];
  int n = curand_poisson(curand_state+i_conn_rel, mu);
  if (n>0) {
    int64_t i_conn = i_conn_0 + i_conn_rel;
    int i_block = (int)(i_conn / block_size);
    int64_t i_block_conn = i_conn % block_size;
    ConnStructT &conn_struct =
      ((ConnStructT**)ConnStructArray)[i_block][i_block_conn];

    int i_target = getConnTarget<ConnStructT>(conn_struct);
    int port = getConnPort<ConnKeyT, ConnStructT>(conn_key, conn_struct); 
    float weight = conn_struct.weight;

    int i_group=NodeGroupMap[i_target];
    int i = port*NodeGroupArray[i_group].n_node_ + i_target
      - NodeGroupArray[i_group].i_node_0_;
    double d_val = (double)(weight*n);
    atomicAddDouble(&NodeGroupArray[i_group].get_spike_array_[i], d_val);
  }
}
// Count number of reverse connections per target node
template <class ConnKeyT, class ConnStructT>
__global__ void countRevConnectionsKernel
(int64_t n_conn, int64_t *target_rev_connection_size_64)
{
  int64_t i_conn = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; 
  if (i_conn >= n_conn) return;

  uint i_block = (uint)(i_conn / ConnBlockSize);
  int64_t i_block_conn = i_conn % ConnBlockSize;
  ConnKeyT &conn_key =
    ((ConnKeyT**)ConnKeyArray)[i_block][i_block_conn];
  ConnStructT &conn_struct =
    ((ConnStructT**)ConnStructArray)[i_block][i_block_conn];

  // TO BE IMPROVED BY CHECKING IF THE SYNAPSE TYPE OF THE GROUP
  // REQUIRES REVERSE CONNECTION
  // - Check syn_group of all connections.
  // - If syn_group>0 must create a reverse connection:
  uint syn_group = getConnSyn<ConnKeyT, ConnStructT>(conn_key, conn_struct);
  if (syn_group > 0) {
    // First get target node index
    uint i_target = getConnTarget<ConnStructT>(conn_struct);
    // (atomic)increase the number of reverse connections for target
    atomicAdd((unsigned long long *)&target_rev_connection_size_64[i_target],
	      1);
  }
}

// Fill array of reverse connection indexes
template <class ConnKeyT, class ConnStructT>
__global__ void setRevConnectionsIndexKernel
(int64_t n_conn, int *target_rev_connection_size,
 int64_t **target_rev_connection)
{
  int64_t i_conn = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; 
  if (i_conn >= n_conn) return;

  uint i_block = (uint)(i_conn / ConnBlockSize);
  int64_t i_block_conn = i_conn % ConnBlockSize;
  ConnKeyT &conn_key =
    ((ConnKeyT**)ConnKeyArray)[i_block][i_block_conn];
  ConnStructT &conn_struct =
    ((ConnStructT**)ConnStructArray)[i_block][i_block_conn];
  
  // TO BE IMPROVED BY CHECKING IF THE SYNAPSE TYPE OF THE GROUP
  // REQUIRES REVERSE CONNECTION  
  // - Check syn_group of all connections.
  // - If syn_group>0 must create a reverse connection:
  uint syn_group = getConnSyn<ConnKeyT, ConnStructT>(conn_key, conn_struct);
  if (syn_group > 0) {
  // First get target node index
  uint i_target = getConnTarget<ConnStructT>(conn_struct);
    // (atomic)increase the number of reverse connections for target
    int pos = atomicAdd(&target_rev_connection_size[i_target], 1);
    // Evaluate the pointer to the rev connection position in the
    // array of reverse connection indexes
    int64_t *rev_conn_pt = target_rev_connection[i_target] + pos;
    // Fill it with the connection index
    *rev_conn_pt = i_conn;
  }
}

__global__ void revConnectionInitKernel(int64_t *rev_conn,
					int *target_rev_conn_size,
					int64_t **target_rev_conn);

__global__ void setConnectionSpikeTime(unsigned int n_conn,
				       unsigned short time_idx);

__global__ void deviceRevSpikeInit(unsigned int *rev_spike_num,
				   unsigned int *rev_spike_target,
				   int *rev_spike_n_conn);

__global__ void setTargetRevConnectionsPtKernel
(int n_spike_buffer, int64_t *target_rev_connection_cumul,
 int64_t **target_rev_connection, int64_t *rev_connections);

__global__ void resetConnectionSpikeTimeUpKernel(unsigned int n_conn);

__global__ void resetConnectionSpikeTimeDownKernel(unsigned int n_conn);


__global__ void connectCalibrateKernel(iconngroup_t *conn_group_idx0,
				       int64_t *conn_group_iconn0,
				       int *conn_group_delay,
				       int64_t block_size,
				       void *conn_key_array,
				       void *conn_struct_array,
				       unsigned short *conn_spike_time);

//template <class ConnKeyT, class ConnStructT>
//ConnectionTemplate<ConnKeyT, ConnStructT>::ConnectionTemplate()
//{
//  init();
//}

template <class ConnKeyT, class ConnStructT>
int ConnectionTemplate<ConnKeyT, ConnStructT>::init()
{
  /////////////////////////////////////////////////
  // member variables initialization
  distribution_ = NULL;
  
  conn_block_size_ = 10000000;

  n_conn_ = 0;

  d_conn_storage_ = NULL;
  
  time_resolution_ = 0.1;

  d_conn_group_idx0_ = NULL;

  d_conn_group_iconn0_  = NULL;

  d_conn_group_delay_  = NULL;

  tot_conn_group_num_ = 0;

  max_delay_num_ = 0;

  d_conn_key_array_  = NULL;

  d_conn_struct_array_ = NULL;
  
  //////////////////////////////////////////////////
  // Remote-connection-related member variables
  //////////////////////////////////////////////////
  this_host_ = 0;
  
  n_hosts_ = 1;
  
  n_image_nodes_ = 0;

  // The arrays that map remote source nodes to local spike buffers
  // are organized in blocks having block size:
  node_map_block_size_ = 100000;

  // number of elements in the map for each source host
  // n_remote_source_node_map[i_source_host]
  // with i_source_host = 0, ..., n_hosts-1 excluding this host itself
  d_n_remote_source_node_map_ = NULL;

  d_local_spike_buffer_map_ = NULL;
  
  // Arrays that map local source nodes to remote spike buffers
  // number of elements in the map for each target host
  // n_local_source_node_map[i_target_host]
  // with i_target_host = 0, ..., n_hosts-1 excluding this host itself
  d_n_local_source_node_map_ = NULL;

  // local_source_node_map[i_target_host][i_block][i]
  d_local_source_node_map_ = NULL;

  // number of remote target hosts on which each local node
  //has outgoing connections
  d_n_target_hosts_ = NULL; // [n_nodes] 
  // target hosts for the node i_node
  d_node_target_hosts_ = NULL; // [i_node]
  // target host map indexes for the node i_node
  d_node_target_host_i_map_ = NULL; // [i_node]

  // Boolean array with one boolean value for each connection rule
  // - true if the rule always creates at least one outgoing connection
  // from each source node (one_to_one, all_to_all, fixed_outdegree)
  // - false otherwise (fixed_indegree, fixed_total_number, pairwise_bernoulli)
  use_all_source_nodes_ = NULL; // [n_connection_rules]:

  //////////////////////////////////////////////////
  // reverse-connection-related member variables
  //////////////////////////////////////////////////
  rev_conn_flag_ = false;
  spike_time_flag_ = false;
  d_conn_spike_time_ = NULL;
  
  n_rev_conn_ = 0;
  d_rev_spike_num_ = NULL;
  d_rev_spike_target_ = NULL;
  d_rev_spike_n_conn_ = NULL;
  d_rev_conn_ = NULL; //[i] i=0,..., n_rev_conn_ - 1;
  d_target_rev_conn_size_ = NULL; //[i] i=0,..., n_neuron-1;
  d_target_rev_conn_ = NULL; //[i][j] j=0,...,rev_conn_size_[i]-1

  initConnRandomGenerator();
  return 0;
}

template<class ConnKeyT, class ConnStructT>
int ConnectionTemplate<ConnKeyT, ConnStructT>::calibrate()
{
  if (spike_time_flag_){
    CUDAMALLOCCTRL("&d_conn_spike_time_",&d_conn_spike_time_,
		   n_conn_*sizeof(unsigned short));
  }
  
  connectCalibrateKernel<<<1,1>>>(d_conn_group_idx0_, d_conn_group_iconn0_,
				  d_conn_group_delay_, conn_block_size_,
				  d_conn_key_array_, d_conn_struct_array_,
				  d_conn_spike_time_);
  DBGCUDASYNC;
  
  return 0;
}


template<class ConnKeyT, class ConnStructT>
int ConnectionTemplate<ConnKeyT, ConnStructT>::allocateNewBlocks
(int new_n_block)
{
  // Allocating GPU memory for new connection blocks
  // allocate new blocks if needed
  for (int ib=conn_key_vect_.size(); ib<new_n_block; ib++) {
    ConnKeyT *d_key_pt;
    ConnStructT *d_connection_pt;
    // allocate GPU memory for new blocks 
    CUDAMALLOCCTRL("&d_key_pt",&d_key_pt, conn_block_size_*sizeof(ConnKeyT));
    CUDAMALLOCCTRL("&d_connection_pt",&d_connection_pt,
		   conn_block_size_*sizeof(ConnStructT));
    conn_key_vect_.push_back(d_key_pt);
    conn_struct_vect_.push_back(d_connection_pt);
  }
  
  return 0;
}

template<class ConnKeyT, class ConnStructT>
int ConnectionTemplate<ConnKeyT, ConnStructT>::freeConnectionKey()
{
  for (uint ib=0; ib<conn_key_vect_.size(); ib++) {
    ConnKeyT *d_key_pt = conn_key_vect_[ib];
    if (d_key_pt != NULL) {
      CUDAFREECTRL("d_key_pt", d_key_pt);
    }
  }
  return 0;
}

template<class ConnKeyT, class ConnStructT>
int ConnectionTemplate<ConnKeyT, ConnStructT>::setConnectionWeights
(curandGenerator_t &gen,
 void *d_storage,
 ConnStructT *conn_struct_subarray,
 int64_t n_conn,
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
    setWeights<ConnStructT><<<(n_conn+1023)/1024, 1024>>>
      (conn_struct_subarray, (float*)d_storage, n_conn);
    DBGCUDASYNC;
  }
  else {
    setWeights<ConnStructT><<<(n_conn+1023)/1024, 1024>>>
      (conn_struct_subarray, syn_spec.weight_, n_conn);
    DBGCUDASYNC;
  }
    
  return 0;
}


template<class ConnKeyT, class ConnStructT>
int ConnectionTemplate<ConnKeyT, ConnStructT>::setConnectionDelays
(curandGenerator_t &gen,
 void *d_storage,
 ConnKeyT *conn_key_subarray,
 int64_t n_conn,
 SynSpec &syn_spec)
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

    setDelays<ConnKeyT><<<(n_conn+1023)/1024, 1024>>>
      (conn_key_subarray, (float*)d_storage, n_conn, time_resolution_);
    DBGCUDASYNC;

  }
  else {
    setDelays<ConnKeyT><<<(n_conn+1023)/1024, 1024>>>
      (conn_key_subarray, syn_spec.delay_, n_conn, time_resolution_);
    DBGCUDASYNC;
  }
  return 0;
}


template <class ConnKeyT, class ConnStructT>
int ConnectionTemplate<ConnKeyT, ConnStructT>::organizeConnections
(inode_t n_node)
{
  timeval startTV;
  timeval endTV;
  CUDASYNC;
  gettimeofday(&startTV, NULL);

  if (d_conn_storage_ != NULL) {
    CUDAFREECTRL("d_conn_storage_", d_conn_storage_);
  }
  
  if (n_conn_ > 0) {
    printf("Allocating auxiliary GPU memory...\n");
    int64_t sort_storage_bytes = 0;
    void *d_sort_storage = NULL;
    copass_sort::sort<ConnKeyT, ConnStructT>
      (conn_key_vect_.data(), conn_struct_vect_.data(),
       n_conn_, conn_block_size_, d_sort_storage, sort_storage_bytes);
    printf("storage bytes: %ld\n", sort_storage_bytes);
    CUDAMALLOCCTRL("&d_sort_storage",&d_sort_storage, sort_storage_bytes);
    
    printf("Sorting...\n");
    copass_sort::sort<ConnKeyT, ConnStructT>
      (conn_key_vect_.data(), conn_struct_vect_.data(),
       n_conn_, conn_block_size_, d_sort_storage, sort_storage_bytes);
    CUDAFREECTRL("d_sort_storage",d_sort_storage);

    size_t storage_bytes = 0;
    size_t storage_bytes1 = 0;
    void *d_storage = NULL;
    printf("Indexing connection groups...\n");
    // It is important to separate number of allocated blocks
    // (determined by conn_key_vect_.size()) from number of blocks
    // on which there are connections, which is determined by n_conn_
    // number of used connection blocks
    int k = (n_conn_ - 1)  / conn_block_size_ + 1;
    
    // it seems that there is no relevant advantage in using a constant array
    // however better to keep this option ready and commented
    //gpuErrchk(cudaMemcpyToSymbol(ConnKeyArray, conn_key_vect_.data(),
    //				 k*sizeof(ConnKeyT*)));
    //, cudaMemcpyHostToDevice));
    //gpuErrchk(cudaMemcpyToSymbol(ConnStructArray, conn_struct_vect_.data(),
    //				 k*sizeof(ConnStructT*)));
    //, cudaMemcpyHostToDevice));

    CUDAMALLOCCTRL("&d_conn_key_array_",&d_conn_key_array_,
		   k*sizeof(ConnKeyT*));
    gpuErrchk(cudaMemcpy(d_conn_key_array_, conn_key_vect_.data(),
			 k*sizeof(ConnKeyT*), cudaMemcpyHostToDevice));
  
    CUDAMALLOCCTRL("&d_conn_struct_array_",&d_conn_struct_array_,
		   k*sizeof(ConnStructT*));
    gpuErrchk(cudaMemcpy(d_conn_struct_array_,
			 conn_struct_vect_.data(),
			 k*sizeof(ConnStructT*), cudaMemcpyHostToDevice));

    //////////////////////////////////////////////////////////////////////
    
    int *d_conn_group_iconn0_mask;
    CUDAMALLOCCTRL("&d_conn_group_iconn0_mask",
		   &d_conn_group_iconn0_mask,
		   conn_block_size_*sizeof(int));

    iconngroup_t *d_conn_group_iconn0_mask_cumul;
    CUDAMALLOCCTRL("&d_conn_group_iconn0_mask_cumul",
		   &d_conn_group_iconn0_mask_cumul,
		   (conn_block_size_+1)*sizeof(iconngroup_t));
    
    int *d_conn_group_idx0_mask;
    CUDAMALLOCCTRL("&d_conn_group_idx0_mask",
		   &d_conn_group_idx0_mask,
		   conn_block_size_*sizeof(int));

    inode_t *d_conn_group_idx0_mask_cumul;
    CUDAMALLOCCTRL("&d_conn_group_idx0_mask_cumul",
		   &d_conn_group_idx0_mask_cumul,
		   (conn_block_size_+1)*sizeof(inode_t));

    iconngroup_t *d_conn_group_idx0_compact;
    int64_t reserve_size = n_node<conn_block_size_ ? n_node : conn_block_size_;
    CUDAMALLOCCTRL("&d_conn_group_idx0_compact",
		   &d_conn_group_idx0_compact,
		   (reserve_size+1)*sizeof(iconngroup_t));
  
    inode_t *d_conn_group_source_compact;
    CUDAMALLOCCTRL("&d_conn_group_source_compact",
		   &d_conn_group_source_compact,
		   reserve_size*sizeof(inode_t));
  
    iconngroup_t *d_iconn0_offset;
    CUDAMALLOCCTRL("&d_iconn0_offset", &d_iconn0_offset, sizeof(iconngroup_t));
    gpuErrchk(cudaMemset(d_iconn0_offset, 0, sizeof(iconngroup_t)));
    inode_t *d_idx0_offset;
    CUDAMALLOCCTRL("&d_idx0_offset", &d_idx0_offset, sizeof(inode_t));
    gpuErrchk(cudaMemset(d_idx0_offset, 0, sizeof(inode_t)));

    ConnKeyT *conn_key_subarray_prev = NULL;
    for (int ib=0; ib<k; ib++) {
      int64_t n_block_conn = ib<(k-1) ? conn_block_size_ : n_conn_ - conn_block_size_*(k-1);
      gpuErrchk(cudaMemset(d_conn_group_iconn0_mask, 0,
			   n_block_conn*sizeof(int)));
      buildConnGroupIConn0Mask<ConnKeyT><<<(n_block_conn+1023)/1024, 1024>>>
	(conn_key_vect_[ib], conn_key_subarray_prev,
       n_block_conn, d_conn_group_iconn0_mask);
      CUDASYNC;
      
      conn_key_subarray_prev = conn_key_vect_[ib] + conn_block_size_ - 1;
    
      if (ib==0) {
	// Determine temporary device storage requirements for prefix sum
	cub::DeviceScan::ExclusiveSum(NULL, storage_bytes,
				      d_conn_group_iconn0_mask,
				      d_conn_group_iconn0_mask_cumul,
				      n_block_conn+1);
	// Allocate temporary storage for prefix sum
	CUDAMALLOCCTRL("&d_storage",&d_storage, storage_bytes);
      }
      // Run exclusive prefix sum
      cub::DeviceScan::ExclusiveSum(d_storage, storage_bytes,
				    d_conn_group_iconn0_mask,
				    d_conn_group_iconn0_mask_cumul,
				    n_block_conn+1);

      setConnGroupNewOffset<<<1, 1>>>(d_iconn0_offset,
				      d_conn_group_iconn0_mask_cumul
				      + n_block_conn);

      CUDASYNC;
      
    }
    gpuErrchk(cudaMemcpy(&tot_conn_group_num_, d_iconn0_offset,
			 sizeof(iconngroup_t), cudaMemcpyDeviceToHost));
    printf("Total number of connection groups: %d\n", tot_conn_group_num_);

    if (tot_conn_group_num_ > 0) {
      iconngroup_t *d_conn_group_num;
      CUDAMALLOCCTRL("&d_conn_group_num", &d_conn_group_num,
		     n_node*sizeof(iconngroup_t));
      gpuErrchk(cudaMemset(d_conn_group_num, 0, sizeof(iconngroup_t)));
    
      ConnKeyT *conn_key_subarray_prev = NULL;
      gpuErrchk(cudaMemset(d_iconn0_offset, 0, sizeof(iconngroup_t)));

      CUDAMALLOCCTRL("&d_conn_group_iconn0_",&d_conn_group_iconn0_,
		     (tot_conn_group_num_+1)*sizeof(int64_t));

      inode_t n_compact = 0; 
      for (int ib=0; ib<k; ib++) {
	int64_t n_block_conn = ib<(k-1) ? conn_block_size_ :
	  n_conn_ - conn_block_size_*(k-1);
	gpuErrchk(cudaMemset(d_conn_group_iconn0_mask, 0,
			     n_block_conn*sizeof(int)));
	gpuErrchk(cudaMemset(d_conn_group_idx0_mask, 0,
			     n_block_conn*sizeof(int)));
	buildConnGroupMask<ConnKeyT><<<(n_block_conn+1023)/1024, 1024>>>
	  (conn_key_vect_[ib], conn_key_subarray_prev,
	   n_block_conn, d_conn_group_iconn0_mask, d_conn_group_idx0_mask);
	CUDASYNC;
      
	conn_key_subarray_prev = conn_key_vect_[ib] + conn_block_size_ - 1;
    
	// Run exclusive prefix sum
	cub::DeviceScan::ExclusiveSum(d_storage, storage_bytes,
				      d_conn_group_iconn0_mask,
				      d_conn_group_iconn0_mask_cumul,
				      n_block_conn+1);
	DBGCUDASYNC;
	cub::DeviceScan::ExclusiveSum(d_storage, storage_bytes,
				      d_conn_group_idx0_mask,
				      d_conn_group_idx0_mask_cumul,
				      n_block_conn+1);

	DBGCUDASYNC;
	int64_t i_conn0 = conn_block_size_*ib;
	setConnGroupIConn0<<<(n_block_conn+1023)/1024, 1024>>>
	  (n_block_conn, d_conn_group_iconn0_mask,
	   d_conn_group_iconn0_mask_cumul, d_conn_group_iconn0_,
	   i_conn0, d_iconn0_offset);
	CUDASYNC;

	setConnGroupIdx0Compact<ConnKeyT><<<(n_block_conn+1023)/1024, 1024>>>
	  (conn_key_vect_[ib], n_block_conn, d_conn_group_idx0_mask,
	   d_conn_group_iconn0_mask_cumul, d_conn_group_idx0_mask_cumul,
	   d_conn_group_idx0_compact, d_conn_group_source_compact,
	   d_iconn0_offset, d_idx0_offset);
	CUDASYNC;

	inode_t n_block_compact; 
	gpuErrchk(cudaMemcpy(&n_block_compact, d_conn_group_idx0_mask_cumul
			     + n_block_conn,
			     sizeof(inode_t), cudaMemcpyDeviceToHost));
	//std::cout << "number of nodes with outgoing connections "
	//"in block " << ib << ": " << n_block_compact << "\n";
	n_compact += n_block_compact;
            
	setConnGroupNewOffset<<<1, 1>>>(d_iconn0_offset,
					d_conn_group_iconn0_mask_cumul
					+ n_block_conn);
	setConnGroupNewOffset<<<1, 1>>>(d_idx0_offset,
					d_conn_group_idx0_mask_cumul
					+ n_block_conn);
	CUDASYNC;
      }
      gpuErrchk(cudaMemcpy(d_conn_group_iconn0_+tot_conn_group_num_, &n_conn_,
			   sizeof(int64_t), cudaMemcpyHostToDevice));

      setConnGroupNum<<<(n_compact+1023)/1024, 1024>>>
	(n_compact, d_conn_group_num, d_conn_group_idx0_compact,
	 d_conn_group_source_compact);
      CUDASYNC;

      CUDAMALLOCCTRL("&d_conn_group_idx0_", &d_conn_group_idx0_,
		     (n_node+1)*sizeof(iconngroup_t));
      storage_bytes1 = 0;
      
      // Determine temporary device storage requirements for prefix sum
      cub::DeviceScan::ExclusiveSum(NULL, storage_bytes1,
				    d_conn_group_num,
				    d_conn_group_idx0_,
				    n_node+1);
      if (storage_bytes1 > storage_bytes) {
	storage_bytes = storage_bytes1;
	CUDAFREECTRL("d_storage",d_storage);
	// Allocate temporary storage for prefix sum
	CUDAMALLOCCTRL("&d_storage",&d_storage, storage_bytes);
      }
      // Run exclusive prefix sum
      cub::DeviceScan::ExclusiveSum(d_storage, storage_bytes,
				    d_conn_group_num,
				    d_conn_group_idx0_,
				    n_node+1);

      // find maxumum number of connection groups (delays) over all neurons
      int *d_max_delay_num;
      CUDAMALLOCCTRL("&d_max_delay_num",&d_max_delay_num, sizeof(int));
    
      storage_bytes1 = 0; 
      // Determine temporary device storage requirements
      cub::DeviceReduce::Max(NULL, storage_bytes1,
			     d_conn_group_num, d_max_delay_num, n_node);
      if (storage_bytes1 > storage_bytes) {
	storage_bytes = storage_bytes1;
	CUDAFREECTRL("d_storage",d_storage);
	// Allocate temporary storage for prefix sum
	CUDAMALLOCCTRL("&d_storage",&d_storage, storage_bytes);
      }
    
      // Run maximum search
      cub::DeviceReduce::Max(d_storage, storage_bytes,
			     d_conn_group_num, d_max_delay_num, n_node);
    
      CUDASYNC;
      gpuErrchk(cudaMemcpy(&max_delay_num_, d_max_delay_num,
			   sizeof(int), cudaMemcpyDeviceToHost));
      CUDAFREECTRL("d_max_delay_num",d_max_delay_num);

      printf("Maximum number of connection groups (delays)"
	     " over all nodes: %d\n", max_delay_num_);
    
      ///////////////////////////////////////////////////////////////////
      ///////////////////////////////////////////////////////////////////
      CUDAFREECTRL("d_storage",d_storage); // free temporary allocated storage
      CUDAFREECTRL("d_conn_group_iconn0_mask",d_conn_group_iconn0_mask);
      CUDAFREECTRL("d_conn_group_iconn0_mask_cumul",
		   d_conn_group_iconn0_mask_cumul);
      CUDAFREECTRL("d_iconn0_offset",d_iconn0_offset);
      CUDAFREECTRL("d_conn_group_idx0_mask",d_conn_group_idx0_mask);
      CUDAFREECTRL("d_conn_group_idx0_mask_cumul",d_conn_group_idx0_mask_cumul);
      CUDAFREECTRL("d_idx0_offset",d_idx0_offset);
      CUDAFREECTRL("d_conn_group_idx0_compact",d_conn_group_idx0_compact);
      CUDAFREECTRL("d_conn_group_num",d_conn_group_num);
      
#ifndef OPTIMIZE_FOR_MEMORY
      CUDAMALLOCCTRL("&d_conn_group_delay_",&d_conn_group_delay_,
		     tot_conn_group_num_*sizeof(int));

      getConnGroupDelay<ConnKeyT><<<(tot_conn_group_num_+1023)/1024, 1024>>>
	(conn_block_size_, d_conn_key_array_, d_conn_group_iconn0_,
	 d_conn_group_delay_, tot_conn_group_num_);
      DBGCUDASYNC;
#endif
	
    }
    else {
      throw ngpu_exception("Number of connections groups must be positive "
			   "for number of connections > 0");   
    }
  }
  else {
    gpuErrchk(cudaMemset(d_conn_group_idx0_, 0,
			 (n_node+1)*sizeof(iconngroup_t)));
    max_delay_num_ = 0;
  }
  
  gettimeofday(&endTV, NULL);
  long time = (long)((endTV.tv_sec * 1000000.0 + endTV.tv_usec)
		     - (startTV.tv_sec * 1000000.0 + startTV.tv_usec));
  printf("%-40s%.2f ms\n", "Time: ", (double)time / 1000.);
  printf("Done\n");
  
  
  return 0;
}

template <class ConnKeyT, class ConnStructT>
template <class T1, class T2>
int ConnectionTemplate<ConnKeyT, ConnStructT>::_Connect
(T1 source, inode_t n_source, T2 target, inode_t n_target,
 ConnSpec &conn_spec, SynSpec &syn_spec)
{
  return _Connect(conn_random_generator_[this_host_][this_host_],
		  source, n_source, target, n_target, conn_spec, syn_spec);
}



template <class ConnKeyT, class ConnStructT>
template <class T1, class T2>
int ConnectionTemplate<ConnKeyT, ConnStructT>::_Connect
(curandGenerator_t &gen, T1 source, inode_t n_source,
 T2 target, inode_t n_target,
 ConnSpec &conn_spec, SynSpec &syn_spec)
{
  if (d_conn_storage_ == NULL) {
    CUDAMALLOCCTRL("&d_conn_storage_", &d_conn_storage_,
		   conn_block_size_*sizeof(uint));
  }
  
  ////////////////////////
    //TEMPORARY, TO BE IMPROVED
  if (syn_spec.syn_group_>=1) {
    spike_time_flag_ = true;
    rev_conn_flag_ = true;
  }

  switch (conn_spec.rule_) {
  case ONE_TO_ONE:
    if (n_source != n_target) {
      throw ngpu_exception("Number of source and target nodes must be equal "
			   "for the one-to-one connection rule");
    }
    return connectOneToOne<T1, T2>
      (gen, source, target, n_source, syn_spec);
    break;

  case ALL_TO_ALL:
    return connectAllToAll<T1, T2>
      (gen, source, n_source, target, n_target, syn_spec);
    break;
  case FIXED_TOTAL_NUMBER:
    return connectFixedTotalNumber<T1, T2>
      (gen, source, n_source, target, n_target,
       conn_spec.total_num_, syn_spec);
    break;
  case FIXED_INDEGREE:
    return connectFixedIndegree<T1, T2>
      (gen, source, n_source, target, n_target,
       conn_spec.indegree_, syn_spec);
    break;
  case FIXED_OUTDEGREE:
    return connectFixedOutdegree<T1, T2>
      (gen, source, n_source, target, n_target,
       conn_spec.outdegree_, syn_spec);
    break;
  default:
    throw ngpu_exception("Unknown connection rule");
  }
  return 0;
}


template <class ConnKeyT, class ConnStructT>
template <class T1, class T2>
int ConnectionTemplate<ConnKeyT, ConnStructT>::connectOneToOne
(curandGenerator_t &gen, T1 source, T2 target, inode_t n_node,
 SynSpec &syn_spec)
{
  int64_t old_n_conn = n_conn_;
  int64_t n_new_conn = n_node;
  n_conn_ += n_new_conn; // new number of connections
  int new_n_block = (int)((n_conn_ + conn_block_size_ - 1) / conn_block_size_);
  allocateNewBlocks(new_n_block);

  //printf("Generating connections with one-to-one rule...\n");
  int64_t n_prev_conn = 0;
  int ib0 = (int)(old_n_conn / conn_block_size_);
  for (int ib=ib0; ib<new_n_block; ib++) {
    int64_t n_block_conn; // number of connections in a block
    int64_t i_conn0; // index of first connection in a block
    if (new_n_block == ib0 + 1) {  // all connections are in the same block
      i_conn0 = old_n_conn % conn_block_size_;
      n_block_conn = n_new_conn;
    }
    else if (ib == ib0) { // first block
      i_conn0 = old_n_conn % conn_block_size_;
      n_block_conn = conn_block_size_ - i_conn0;
    }
    else if (ib == new_n_block-1) { // last block
      i_conn0 = 0;
      n_block_conn = (n_conn_ - 1) % conn_block_size_ + 1;
    }
    else {
      i_conn0 = 0;
      n_block_conn = conn_block_size_;
    }
    setOneToOneSourceTarget<T1, T2, ConnKeyT, ConnStructT>
      <<<(n_block_conn+1023)/1024, 1024>>>
      (conn_key_vect_[ib] + i_conn0, conn_struct_vect_[ib] + i_conn0,
       n_block_conn, n_prev_conn, source, target);
    DBGCUDASYNC;
    setConnectionWeights(gen, d_conn_storage_, conn_struct_vect_[ib] + i_conn0,
       n_block_conn, syn_spec);
    setConnectionDelays(gen, d_conn_storage_, conn_key_vect_[ib] + i_conn0,
       n_block_conn, syn_spec);
    setPort<ConnKeyT, ConnStructT><<<(n_block_conn+1023)/1024, 1024>>>
    (conn_key_vect_[ib] + i_conn0, conn_struct_vect_[ib] + i_conn0,
     syn_spec.port_, n_block_conn);
    DBGCUDASYNC;
    setSynGroup<ConnKeyT, ConnStructT><<<(n_block_conn+1023)/1024, 1024>>>
      (conn_key_vect_[ib] + i_conn0, conn_struct_vect_[ib] + i_conn0,
       syn_spec.syn_group_, n_block_conn);
    DBGCUDASYNC;
    CUDASYNC;
    n_prev_conn += n_block_conn;
  }

  return 0;
}


template <class ConnKeyT, class ConnStructT>
template <class T1, class T2>
int ConnectionTemplate<ConnKeyT, ConnStructT>::connectAllToAll
(curandGenerator_t &gen, T1 source, inode_t n_source,
 T2 target, inode_t n_target, SynSpec &syn_spec)
{
  int64_t old_n_conn = n_conn_;
  int64_t n_new_conn = n_source*n_target;
  n_conn_ += n_new_conn; // new number of connections
  int new_n_block = (int)((n_conn_ + conn_block_size_ - 1) / conn_block_size_);

  allocateNewBlocks(new_n_block);

  //printf("Generating connections with all-to-all rule...\n");
  int64_t n_prev_conn = 0;
  int ib0 = (int)(old_n_conn / conn_block_size_);
  for (int ib=ib0; ib<new_n_block; ib++) {
    int64_t n_block_conn; // number of connections in a block
    int64_t i_conn0; // index of first connection in a block
    if (new_n_block == ib0 + 1) {  // all connections are in the same block
      i_conn0 = old_n_conn % conn_block_size_;
      n_block_conn = n_new_conn;
    }
    else if (ib == ib0) { // first block
      i_conn0 = old_n_conn % conn_block_size_;
      n_block_conn = conn_block_size_ - i_conn0;
    }
    else if (ib == new_n_block-1) { // last block
      i_conn0 = 0;
      n_block_conn = (n_conn_ - 1) % conn_block_size_ + 1;
    }
    else {
      i_conn0 = 0;
      n_block_conn = conn_block_size_;
    }

    setAllToAllSourceTarget<T1, T2, ConnKeyT, ConnStructT>
      <<<(n_block_conn+1023)/1024, 1024>>>
      (conn_key_vect_[ib] + i_conn0, conn_struct_vect_[ib] + i_conn0,
       n_block_conn, n_prev_conn, source, n_source, target, n_target);
    DBGCUDASYNC;
    setConnectionWeights(gen, d_conn_storage_, conn_struct_vect_[ib] + i_conn0,
       n_block_conn, syn_spec);

    setConnectionDelays(gen, d_conn_storage_, conn_key_vect_[ib] + i_conn0,
       n_block_conn, syn_spec);

    setPort<ConnKeyT, ConnStructT><<<(n_block_conn+1023)/1024, 1024>>>
      (conn_key_vect_[ib] + i_conn0, conn_struct_vect_[ib] + i_conn0,
       syn_spec.port_, n_block_conn);
    DBGCUDASYNC;
    setSynGroup<ConnKeyT, ConnStructT><<<(n_block_conn+1023)/1024, 1024>>>
      (conn_key_vect_[ib] + i_conn0, conn_struct_vect_[ib] + i_conn0,
       syn_spec.syn_group_, n_block_conn);
    DBGCUDASYNC;

    n_prev_conn += n_block_conn;
  }

  return 0;
}


template <class ConnKeyT, class ConnStructT>
template <class T1, class T2>
int ConnectionTemplate<ConnKeyT, ConnStructT>::connectFixedTotalNumber
(curandGenerator_t &gen, T1 source, inode_t n_source,
 T2 target, inode_t n_target, int64_t total_num, SynSpec &syn_spec)
{
  if (total_num==0) return 0;
  int64_t old_n_conn = n_conn_;
  int64_t n_new_conn = total_num;
  n_conn_ += n_new_conn; // new number of connections
  int new_n_block = (int)((n_conn_ + conn_block_size_ - 1) / conn_block_size_);

  allocateNewBlocks(new_n_block);

  //printf("Generating connections with fixed_total_number rule...\n");
  int ib0 = (int)(old_n_conn / conn_block_size_);
  for (int ib=ib0; ib<new_n_block; ib++) {
    int64_t n_block_conn; // number of connections in a block
    int64_t i_conn0; // index of first connection in a block
    if (new_n_block == ib0 + 1) {  // all connections are in the same block
        i_conn0 = old_n_conn % conn_block_size_;
	n_block_conn =   n_new_conn;
    }
    else if (ib == ib0) { // first block
      i_conn0 = old_n_conn % conn_block_size_;
      n_block_conn = conn_block_size_ - i_conn0;
    }
    else if (ib == new_n_block-1) { // last block
      i_conn0 = 0;
      n_block_conn = (n_conn_ - 1) % conn_block_size_ + 1;
    }
    else {
      i_conn0 = 0;
      n_block_conn = conn_block_size_;
    }
    // generate random source index in range 0 - n_neuron
    CURAND_CALL(curandGenerate(gen, (uint*)d_conn_storage_, n_block_conn));
    setSource<T1, ConnKeyT><<<(n_block_conn+1023)/1024, 1024>>>
      (conn_key_vect_[ib] + i_conn0, (uint*)d_conn_storage_, n_block_conn,
       source, n_source);
    DBGCUDASYNC;

    // generate random target index in range 0 - n_neuron
    CURAND_CALL(curandGenerate(gen, (uint*)d_conn_storage_, n_block_conn));
    setTarget<T2, ConnStructT><<<(n_block_conn+1023)/1024, 1024>>>
      (conn_struct_vect_[ib] + i_conn0, (uint*)d_conn_storage_,
       n_block_conn, target, n_target);
    DBGCUDASYNC;

    setConnectionWeights(gen, d_conn_storage_, conn_struct_vect_[ib] + i_conn0,
			 n_block_conn, syn_spec);

    setConnectionDelays(gen, d_conn_storage_, conn_key_vect_[ib] + i_conn0,
			n_block_conn, syn_spec);

    setPort<ConnKeyT, ConnStructT><<<(n_block_conn+1023)/1024, 1024>>>
      (conn_key_vect_[ib] + i_conn0,
       conn_struct_vect_[ib] + i_conn0, syn_spec.port_,
       n_block_conn);
    DBGCUDASYNC;
    setSynGroup<ConnKeyT, ConnStructT><<<(n_block_conn+1023)/1024, 1024>>>
      (conn_key_vect_[ib] + i_conn0, conn_struct_vect_[ib] + i_conn0, syn_spec.syn_group_,
       n_block_conn);
    DBGCUDASYNC;

  }

  return 0;
}

template <class ConnKeyT, class ConnStructT>
template <class T1, class T2>
int ConnectionTemplate<ConnKeyT, ConnStructT>::connectFixedIndegree
(curandGenerator_t &gen, T1 source, inode_t n_source,
 T2 target, inode_t n_target, int indegree, SynSpec &syn_spec)
{
  if (indegree<=0) return 0;
  int64_t old_n_conn = n_conn_;
  int64_t n_new_conn = n_target*indegree;
  n_conn_ += n_new_conn; // new number of connections
  int new_n_block = (int)((n_conn_ + conn_block_size_ - 1) / conn_block_size_);

  allocateNewBlocks(new_n_block);

  //printf("Generating connections with fixed_indegree rule...\n");
  int64_t n_prev_conn = 0;
  int ib0 = (int)(old_n_conn / conn_block_size_);
  for (int ib=ib0; ib<new_n_block; ib++) {
    int64_t n_block_conn; // number of connections in a block
    int64_t i_conn0; // index of first connection in a block
    if (new_n_block == ib0 + 1) {  // all connections are in the same block
        i_conn0 = old_n_conn % conn_block_size_;
	n_block_conn =   n_new_conn;
    }
    else if (ib == ib0) { // first block
      i_conn0 = old_n_conn % conn_block_size_;
      n_block_conn = conn_block_size_ - i_conn0;
    }
    else if (ib == new_n_block-1) { // last block
      i_conn0 = 0;
      n_block_conn = (n_conn_ - 1) % conn_block_size_ + 1;
    }
    else {
      i_conn0 = 0;
      n_block_conn = conn_block_size_;
    }
    // generate random source index in range 0 - n_neuron
    CURAND_CALL(curandGenerate(gen, (uint*)d_conn_storage_, n_block_conn));
    setSource<T1, ConnKeyT><<<(n_block_conn+1023)/1024, 1024>>>
      (conn_key_vect_[ib] + i_conn0, (uint*)d_conn_storage_, n_block_conn,
       source, n_source);
    DBGCUDASYNC;

    setIndegreeTarget<T2, ConnStructT><<<(n_block_conn+1023)/1024, 1024>>>
      (conn_struct_vect_[ib] + i_conn0, n_block_conn, n_prev_conn,
       target, indegree);
    DBGCUDASYNC;

    setConnectionWeights(gen, d_conn_storage_, conn_struct_vect_[ib] + i_conn0,
			 n_block_conn, syn_spec);

    setConnectionDelays(gen, d_conn_storage_, conn_key_vect_[ib] + i_conn0,
			n_block_conn, syn_spec);

    setPort<ConnKeyT, ConnStructT><<<(n_block_conn+1023)/1024, 1024>>>
      (conn_key_vect_[ib] + i_conn0, conn_struct_vect_[ib] + i_conn0,
       syn_spec.port_, n_block_conn);
    DBGCUDASYNC;
    setSynGroup<ConnKeyT, ConnStructT><<<(n_block_conn+1023)/1024, 1024>>>
      (conn_key_vect_[ib] + i_conn0, conn_struct_vect_[ib] + i_conn0,
       syn_spec.syn_group_, n_block_conn);
    DBGCUDASYNC;

    n_prev_conn += n_block_conn;
  }

  return 0;
}


template <class ConnKeyT, class ConnStructT>
template <class T1, class T2>
int ConnectionTemplate<ConnKeyT, ConnStructT>::connectFixedOutdegree
(curandGenerator_t &gen, T1 source, inode_t n_source,
 T2 target, inode_t n_target, int outdegree, SynSpec &syn_spec)
{
  if (outdegree<=0) return 0;
  int64_t old_n_conn = n_conn_;
  int64_t n_new_conn = n_source*outdegree;
  n_conn_ += n_new_conn; // new number of connections
  int new_n_block = (int)((n_conn_ + conn_block_size_ - 1) / conn_block_size_);

  allocateNewBlocks(new_n_block);

  //printf("Generating connections with fixed_outdegree rule...\n");
  int64_t n_prev_conn = 0;
  int ib0 = (int)(old_n_conn / conn_block_size_);
  for (int ib=ib0; ib<new_n_block; ib++) {
    int64_t n_block_conn; // number of connections in a block
    int64_t i_conn0; // index of first connection in a block
    if (new_n_block == ib0 + 1) {  // all connections are in the same block
        i_conn0 = old_n_conn % conn_block_size_;
	n_block_conn =   n_new_conn;
    }
    else if (ib == ib0) { // first block
      i_conn0 = old_n_conn % conn_block_size_;
      n_block_conn = conn_block_size_ - i_conn0;
    }
    else if (ib == new_n_block-1) { // last block
      i_conn0 = 0;
      n_block_conn = (n_conn_ - 1) % conn_block_size_ + 1;
    }
    else {
      i_conn0 = 0;
      n_block_conn = conn_block_size_;
    }

    setOutdegreeSource<T1, ConnKeyT><<<(n_block_conn+1023)/1024, 1024>>>
      (conn_key_vect_[ib] + i_conn0, n_block_conn, n_prev_conn,
       source, outdegree);
    DBGCUDASYNC;

    // generate random target index in range 0 - n_neuron
    CURAND_CALL(curandGenerate(gen, (uint*)d_conn_storage_, n_block_conn));
    setTarget<T2, ConnStructT><<<(n_block_conn+1023)/1024, 1024>>>
      (conn_struct_vect_[ib] + i_conn0, (uint*)d_conn_storage_, n_block_conn,
       target, n_target);
    DBGCUDASYNC;

    setConnectionWeights(gen, d_conn_storage_, conn_struct_vect_[ib] + i_conn0,
			 n_block_conn, syn_spec);

    setConnectionDelays(gen, d_conn_storage_, conn_key_vect_[ib] + i_conn0,
			n_block_conn, syn_spec);

    setPort<ConnKeyT, ConnStructT><<<(n_block_conn+1023)/1024, 1024>>>
      (conn_key_vect_[ib] + i_conn0, conn_struct_vect_[ib] + i_conn0,
       syn_spec.port_, n_block_conn);
    DBGCUDASYNC;
    setSynGroup<ConnKeyT, ConnStructT><<<(n_block_conn+1023)/1024, 1024>>>
      (conn_key_vect_[ib] + i_conn0, conn_struct_vect_[ib] + i_conn0,
       syn_spec.syn_group_,
       n_block_conn);
    DBGCUDASYNC;

    n_prev_conn += n_block_conn;
  }

  return 0;
}


//////////////////////////////////////////////////////////////////////
// Get the float parameter param_name of an array of n_conn connections,
// identified by the indexes conn_ids[i], and put it in the array
// h_param_arr
// NOTE: host array should be pre-allocated to store n_conn elements
//////////////////////////////////////////////////////////////////////
template <class ConnKeyT, class ConnStructT>
int ConnectionTemplate<ConnKeyT, ConnStructT>::getConnectionFloatParam
(int64_t *conn_ids,
 int64_t n_conn,
 float *h_param_arr,
 std::string param_name)
{
  // Check if param_name is a connection float parameter
  int i_param = getConnectionFloatParamIndex(param_name);
  if (i_param < 0) {
    throw ngpu_exception(std::string("Unrecognized connection float parameter ")
			 + param_name);
  }
  if (n_conn > 0) {
    // declare pointers to arrays in device memory
    int64_t *d_conn_ids;
    float *d_arr;
    // allocate array of connection ids in device memory
    // and copy the ids from host to device array
    CUDAMALLOCCTRL("&d_conn_ids",&d_conn_ids, n_conn*sizeof(int64_t));
    gpuErrchk(cudaMemcpy(d_conn_ids, conn_ids, n_conn*sizeof(int64_t),
			 cudaMemcpyHostToDevice));
    
    // allocate connection parameter array in device memory
    CUDAMALLOCCTRL("&d_arr",&d_arr, n_conn*sizeof(float));
    
    // launch kernel to get connection parameters
    getConnectionFloatParamKernel<ConnKeyT, ConnStructT>
      <<<(n_conn+1023)/1024, 1024 >>>
      (d_conn_ids, n_conn, d_arr, i_param);
    
    // copy connection parameter array from device to host memory
    gpuErrchk(cudaMemcpy(h_param_arr, d_arr, n_conn*sizeof(float),
			 cudaMemcpyDeviceToHost));
    // free allocated device memory
    CUDAFREECTRL("d_conn_ids",d_conn_ids);
    CUDAFREECTRL("d_arr",d_arr);
  }
  
  return 0;
}


//////////////////////////////////////////////////////////////////////
// Get the integer parameter param_name of an array of n_conn connections,
// identified by the indexes conn_ids[i], and put it in the array
// h_param_arr
// NOTE: host array should be pre-allocated to store n_conn elements
//////////////////////////////////////////////////////////////////////
template <class ConnKeyT, class ConnStructT>
int ConnectionTemplate<ConnKeyT, ConnStructT>::getConnectionIntParam
(int64_t *conn_ids, int64_t n_conn,
 int *h_param_arr,
 std::string param_name)
{
  // Check if param_name is a connection integer parameter
  int i_param = getConnectionIntParamIndex(param_name);
  if (i_param < 0) {
    throw ngpu_exception(std::string("Unrecognized connection "
				     "integer parameter ") + param_name);
  }
  if (n_conn > 0) {
    // declare pointers to arrays in device memory
    int64_t *d_conn_ids;
    int *d_arr;
    // allocate array of connection ids in device memory
    // and copy the ids from host to device array
    CUDAMALLOCCTRL("&d_conn_ids",&d_conn_ids, n_conn*sizeof(int64_t));
    gpuErrchk(cudaMemcpy(d_conn_ids, conn_ids, n_conn*sizeof(int64_t),
			 cudaMemcpyHostToDevice));
    
    // allocate connection parameter array in device memory
    CUDAMALLOCCTRL("&d_arr",&d_arr, n_conn*sizeof(int));
    
    // launch kernel to get connection parameters
    getConnectionIntParamKernel<ConnKeyT, ConnStructT>
      <<<(n_conn+1023)/1024, 1024 >>>
      (d_conn_ids, n_conn, d_arr, i_param);
    
    // copy connection parameter array from device to host memory
    gpuErrchk(cudaMemcpy(h_param_arr, d_arr, n_conn*sizeof(int),
			 cudaMemcpyDeviceToHost));
    // free allocated device memory
    CUDAFREECTRL("d_conn_ids",d_conn_ids);
    CUDAFREECTRL("d_arr",d_arr);
  }
  
  return 0;
}


//////////////////////////////////////////////////////////////////////
// Set the float parameter param_name of an array of n_conn connections,
// identified by the indexes conn_ids[i], to the value val
//////////////////////////////////////////////////////////////////////
template <class ConnKeyT, class ConnStructT>
int ConnectionTemplate<ConnKeyT, ConnStructT>::setConnectionFloatParam
(int64_t *conn_ids,
 int64_t n_conn,
 float val,
 std::string param_name)
{
  // Check if param_name is a connection float parameter
  int i_param = getConnectionFloatParamIndex(param_name);
  if (i_param < 0) {
    throw ngpu_exception(std::string("Unrecognized connection float parameter ")
			 + param_name);
  }
  if (i_param == i_delay_param) {
        throw ngpu_exception("Connection delay cannot be modified");
  }

  if (n_conn > 0) {
    // declare pointers to arrays in device memory
    int64_t *d_conn_ids;
    // allocate array of connection ids in device memory
    // and copy the ids from host to device array
    CUDAMALLOCCTRL("&d_conn_ids",&d_conn_ids, n_conn*sizeof(int64_t));
    gpuErrchk(cudaMemcpy(d_conn_ids, conn_ids, n_conn*sizeof(int64_t),
			 cudaMemcpyHostToDevice));
        
    // launch kernel to set connection parameters
    setConnectionFloatParamKernel<ConnStructT>
      <<<(n_conn+1023)/1024, 1024 >>>
      (d_conn_ids, n_conn, val, i_param);
    // free allocated device memory
    CUDAFREECTRL("d_conn_ids",d_conn_ids);    
  }
  
  return 0;
}


//////////////////////////////////////////////////////////////////////
// Set the float parameter param_name of an array of n_conn connections,
// identified by the indexes conn_ids[i], using values from a distribution
// or from an array
//////////////////////////////////////////////////////////////////////
template <class ConnKeyT, class ConnStructT>
int ConnectionTemplate<ConnKeyT, ConnStructT>::setConnectionFloatParamDistr
(int64_t *conn_ids,
 int64_t n_conn,
 std::string param_name)
{
  // Check if param_name is a connection float parameter
  int i_param = getConnectionFloatParamIndex(param_name);
  if (i_param < 0) {
    throw ngpu_exception(std::string("Unrecognized connection float parameter ")
			 + param_name);
  }
  if (i_param == i_delay_param) {
    throw ngpu_exception("Connection delay cannot be modified");
  }

  if (n_conn > 0) {
    // declare pointers to arrays in device memory
    int64_t *d_conn_ids;
    // allocate array of connection ids in device memory
    // and copy the ids from host to device array
    CUDAMALLOCCTRL("&d_conn_ids",&d_conn_ids, n_conn*sizeof(int64_t));
    gpuErrchk(cudaMemcpy(d_conn_ids, conn_ids, n_conn*sizeof(int64_t),
			 cudaMemcpyHostToDevice));
    
    // get values from array or distribution
    float *d_arr = distribution_->getArray
      (conn_random_generator_[this_host_][this_host_], n_conn);
    // launch kernel to set connection parameters
    setConnectionFloatParamKernel<ConnStructT>
      <<<(n_conn+1023)/1024, 1024 >>>
      (d_conn_ids, n_conn, d_arr, i_param);
    // free allocated device memory
    CUDAFREECTRL("d_conn_ids",d_conn_ids);
    CUDAFREECTRL("d_arr",d_arr);
  }
  
  return 0;
}


//////////////////////////////////////////////////////////////////////
// Set the integer parameter param_name of an array of n_conn connections,
// identified by the indexes conn_ids[i], using the values from the array
// h_param_arr
//////////////////////////////////////////////////////////////////////
template <class ConnKeyT, class ConnStructT>
int ConnectionTemplate<ConnKeyT, ConnStructT>::setConnectionIntParamArr
(int64_t *conn_ids,
 int64_t n_conn,
 int *h_param_arr,
 std::string param_name)
{
  // Check if param_name is a connection int parameter
  int i_param = getConnectionIntParamIndex(param_name);
  if (i_param < 0) {
    throw ngpu_exception(std::string("Unrecognized connection int parameter ")
			 + param_name);
  }
  if (i_param == i_source_param) {
    throw ngpu_exception("Connection source node cannot be modified");
  }

  if (n_conn > 0) {
    // declare pointers to arrays in device memory
    int64_t *d_conn_ids;
    int *d_arr;
    // allocate array of connection ids in device memory
    // and copy the ids from host to device array
    CUDAMALLOCCTRL("&d_conn_ids",&d_conn_ids, n_conn*sizeof(int64_t));
    gpuErrchk(cudaMemcpy(d_conn_ids, conn_ids, n_conn*sizeof(int64_t),
			 cudaMemcpyHostToDevice));
    
    // allocate connection parameter array in device memory
    CUDAMALLOCCTRL("&d_arr",&d_arr, n_conn*sizeof(int));

    // copy connection parameter array from host to device memory
    gpuErrchk(cudaMemcpy(d_arr, h_param_arr, n_conn*sizeof(int),
			 cudaMemcpyHostToDevice));
    
    // launch kernel to set connection parameters
    setConnectionIntParamKernel<ConnKeyT, ConnStructT>
      <<<(n_conn+1023)/1024, 1024 >>>
      (d_conn_ids, n_conn, d_arr, i_param);
    // free allocated device memory
    CUDAFREECTRL("d_conn_ids",d_conn_ids);
    CUDAFREECTRL("d_arr",d_arr);
  }
  
  return 0;
}


//////////////////////////////////////////////////////////////////////
// Set the int parameter param_name of an array of n_conn connections,
// identified by the indexes conn_ids[i], to the value val
//////////////////////////////////////////////////////////////////////
template <class ConnKeyT, class ConnStructT>
int ConnectionTemplate<ConnKeyT, ConnStructT>::setConnectionIntParam
(int64_t *conn_ids, int64_t n_conn,
 int val, std::string param_name)
{
  // Check if param_name is a connection float parameter
  int i_param = getConnectionIntParamIndex(param_name);
  if (i_param < 0) {
    throw ngpu_exception(std::string("Unrecognized connection int parameter ")
			 + param_name);
  }
  if (i_param == i_source_param) {
    throw ngpu_exception("Connection source node cannot be modified");
  }

  if (n_conn > 0) {
    // declare pointers to arrays in device memory
    int64_t *d_conn_ids;
    // allocate array of connection ids in device memory
    // and copy the ids from host to device array
    CUDAMALLOCCTRL("&d_conn_ids",&d_conn_ids, n_conn*sizeof(int64_t));
    gpuErrchk(cudaMemcpy(d_conn_ids, conn_ids, n_conn*sizeof(int64_t),
			 cudaMemcpyHostToDevice));
        
    // launch kernel to set connection parameters
    setConnectionIntParamKernel<ConnKeyT, ConnStructT>
      <<<(n_conn+1023)/1024, 1024 >>>
      (d_conn_ids, n_conn, val, i_param);
    // free allocated device memory
    CUDAFREECTRL("d_conn_ids",d_conn_ids);
  }
  
  return 0;
}


template <class ConnKeyT, class ConnStructT>
int64_t *ConnectionTemplate<ConnKeyT, ConnStructT>::getConnections
(inode_t *i_source_pt,
 inode_t n_source,
 inode_t *i_target_pt,
 inode_t n_target,
 int syn_group, int64_t *n_conn)
{
  int64_t *h_conn_ids = NULL;
  int64_t *d_conn_ids = NULL;
  uint64_t n_src_tgt = (uint64_t)n_source * n_target;
  int64_t n_conn_ids = 0;
  
  if (n_src_tgt > 0) {
    //std::cout << "n_src_tgt " << n_src_tgt << "n_source " << n_source
    //	      << "n_target " << n_target << "\n";
    // sort source node index array in GPU memory
    inode_t *d_src_arr = sortArray(i_source_pt, n_source);
    // sort target node index array in GPU memory
    inode_t *d_tgt_arr = sortArray(i_target_pt, n_target);
    // Allocate array of combined source-target indexes (src_arr x tgt_arr)
    uint64_t *d_src_tgt_arr;
    CUDAMALLOCCTRL("&d_src_tgt_arr",&d_src_tgt_arr, n_src_tgt*sizeof(uint64_t));
    // Fill it with combined source-target indexes
    setSourceTargetIndexKernel<<<(n_src_tgt+1023)/1024, 1024>>>
      (n_src_tgt, n_source, n_target, d_src_tgt_arr, d_src_arr, d_tgt_arr);
    // Allocate array of number of connections per source-target couple
    // and initialize it to 0
    uint64_t *d_src_tgt_conn_num;
    CUDAMALLOCCTRL("&d_src_tgt_conn_num",&d_src_tgt_conn_num,
		   (n_src_tgt + 1)*sizeof(uint64_t));
    gpuErrchk(cudaMemset(d_src_tgt_conn_num, 0,
			 (n_src_tgt + 1)*sizeof(uint64_t)));

    // Count number of connections per source-target couple
    countConnectionsKernel<ConnKeyT, ConnStructT><<<(n_conn_+1023)/1024, 1024>>>
      (n_conn_, n_source, n_target, d_src_tgt_arr, d_src_tgt_conn_num,
       syn_group);
    // Evaluate exclusive sum of connections per source-target couple
    // Allocate array for cumulative sum
    uint64_t *d_src_tgt_conn_cumul;
    CUDAMALLOCCTRL("&d_src_tgt_conn_cumul",&d_src_tgt_conn_cumul,
			 (n_src_tgt + 1)*sizeof(uint64_t));
    // Determine temporary device storage requirements
    void *d_storage = NULL;
    size_t storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_storage, storage_bytes,
				  d_src_tgt_conn_num,
				  d_src_tgt_conn_cumul,
				  n_src_tgt + 1);
    // Allocate temporary storage
    CUDAMALLOCCTRL("&d_storage",&d_storage, storage_bytes);
    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(d_storage, storage_bytes,
				  d_src_tgt_conn_num,
				  d_src_tgt_conn_cumul,
				  n_src_tgt + 1);
    CUDAFREECTRL("d_storage",d_storage);
    
    // The last element is the total number of required connection Ids
    cudaMemcpy(&n_conn_ids, &d_src_tgt_conn_cumul[n_src_tgt],
	       sizeof(int64_t), cudaMemcpyDeviceToHost);
    
    if (n_conn_ids > 0) {
      // Allocate array of connection indexes
      CUDAMALLOCCTRL("&d_conn_ids",&d_conn_ids, n_conn_ids*sizeof(int64_t));  
      // Set number of connections per source-target couple to 0 again
      gpuErrchk(cudaMemset(d_src_tgt_conn_num, 0,
			   (n_src_tgt + 1)*sizeof(uint64_t)));
      // Fill array of connection indexes
      setConnectionsIndexKernel<ConnKeyT, ConnStructT>
	<<<(n_conn_+1023)/1024, 1024>>>
	(n_conn_, n_source, n_target, d_src_tgt_arr, d_src_tgt_conn_num,
	 d_src_tgt_conn_cumul, syn_group, d_conn_ids);

      /// check if allocating with new is more appropriate
      h_conn_ids = (int64_t*)malloc(n_conn_ids*sizeof(int64_t));
      gpuErrchk(cudaMemcpy(h_conn_ids, d_conn_ids,
			   n_conn_ids*sizeof(int64_t),
			   cudaMemcpyDeviceToHost));
	
      CUDAFREECTRL("d_src_tgt_arr",d_src_tgt_arr);
      CUDAFREECTRL("d_src_tgt_conn_num",d_src_tgt_conn_num);
      CUDAFREECTRL("d_src_tgt_conn_cumul",d_src_tgt_conn_cumul);
      CUDAFREECTRL("d_conn_ids",d_conn_ids);
    }
  }
  *n_conn = n_conn_ids;
  
  return h_conn_ids;
}



//////////////////////////////////////////////////////////////////////
// Get all parameters of an array of n_conn connections, identified by
// the indexes conn_ids[i], and put them in the arrays
// i_source, i_target, port, syn_group, delay, weight
// NOTE: host arrays should be pre-allocated to store n_conn elements
//////////////////////////////////////////////////////////////////////
template <class ConnKeyT, class ConnStructT>
int ConnectionTemplate<ConnKeyT, ConnStructT>::getConnectionStatus
(int64_t *conn_ids, int64_t n_conn,
 inode_t *source, inode_t *target,
 int *port,
 int *syn_group, float *delay,
 float *weight)
{
  if (n_conn > 0) {
    // declare pointers to arrays in device memory
    int64_t *d_conn_ids;
    inode_t *d_source;
    inode_t *d_target;
    int *d_port;
    int *d_syn_group;
    float *d_delay;
    float *d_weight;

    // allocate array of connection ids in device memory
    // and copy the ids from host to device array
    CUDAMALLOCCTRL("&d_conn_ids",&d_conn_ids, n_conn*sizeof(int64_t));
    gpuErrchk(cudaMemcpy(d_conn_ids, conn_ids, n_conn*sizeof(int64_t),
			 cudaMemcpyHostToDevice));

    // allocate arrays of connection parameters in device memory
    CUDAMALLOCCTRL("&d_source",&d_source, n_conn*sizeof(inode_t));
    CUDAMALLOCCTRL("&d_target",&d_target, n_conn*sizeof(inode_t));
    CUDAMALLOCCTRL("&d_port",&d_port, n_conn*sizeof(int));
    CUDAMALLOCCTRL("&d_syn_group",&d_syn_group, n_conn*sizeof(int));
    CUDAMALLOCCTRL("&d_delay",&d_delay, n_conn*sizeof(float));
    CUDAMALLOCCTRL("&d_weight",&d_weight, n_conn*sizeof(float));
    // host arrays
    
    // launch kernel to get connection parameters
    getConnectionStatusKernel<ConnKeyT, ConnStructT>
      <<<(n_conn+1023)/1024, 1024 >>>
      (d_conn_ids, n_conn, d_source, d_target, d_port, d_syn_group,
       d_delay, d_weight);

    // copy connection parameters from device to host memory
    gpuErrchk(cudaMemcpy(source, d_source, n_conn*sizeof(inode_t),
			 cudaMemcpyDeviceToHost));
    
    gpuErrchk(cudaMemcpy(target, d_target, n_conn*sizeof(inode_t),
			 cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(port, d_port, n_conn*sizeof(int),
			 cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(syn_group, d_syn_group,
			 n_conn*sizeof(int),
			 cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(delay, d_delay, n_conn*sizeof(float),
			 cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(weight, d_weight, n_conn*sizeof(float),
			 cudaMemcpyDeviceToHost));
  }
  
  return 0;
}


template <class ConnKeyT, class ConnStructT>
int ConnectionTemplate<ConnKeyT, ConnStructT>::freeConnRandomGenerator()
{
  for (int i_host=0; i_host<n_hosts_; i_host++) {
    for (int j_host=0; j_host<n_hosts_; j_host++) {
      CURAND_CALL(curandDestroyGenerator
		  (conn_random_generator_[i_host][j_host]));
    }
  }
  
  return 0;
}

template <class ConnKeyT, class ConnStructT>
int ConnectionTemplate<ConnKeyT, ConnStructT>::initConnRandomGenerator()
{
  conn_random_generator_.resize(n_hosts_);
  for (int i_host=0; i_host<n_hosts_; i_host++) {
    conn_random_generator_[i_host].resize(n_hosts_);
    for (int j_host=0; j_host<n_hosts_; j_host++) {
      CURAND_CALL(curandCreateGenerator
		  (&conn_random_generator_[i_host][j_host],
		   CURAND_RNG_PSEUDO_DEFAULT));
    }
  }
  
  return 0;
}

template <class ConnKeyT, class ConnStructT>
int ConnectionTemplate<ConnKeyT, ConnStructT>::setRandomSeed
(unsigned long long seed)
{
  for (int i_host=0; i_host<n_hosts_; i_host++) {
    for (int j_host=0; j_host<n_hosts_; j_host++) {
      CURAND_CALL(curandSetPseudoRandomGeneratorSeed
		  (conn_random_generator_[i_host][j_host],
		   seed + conn_seed_offset_ + i_host*n_hosts_ + j_host));
    }
  }
  
  return 0;
}


template <class ConnKeyT, class ConnStructT>
int ConnectionTemplate<ConnKeyT, ConnStructT>::setNHosts(int n_hosts)
{
  // free previous instances before creating new
  freeConnRandomGenerator();
  n_hosts_ = n_hosts;
  initConnRandomGenerator();

  return 0;
}

template <class ConnKeyT, class ConnStructT>
int ConnectionTemplate<ConnKeyT, ConnStructT>::setThisHost(int this_host)
{
  this_host_ = this_host;

  return 0;
}

template<class ConnKeyT, class ConnStructT>
int ConnectionTemplate<ConnKeyT, ConnStructT>::organizeDirectConnections
(void* &d_poiss_key_array_data_pt,
 void* &d_poiss_subarray,
 int64_t* &d_poiss_num,
 int64_t* &d_poiss_sum,
 void* &d_poiss_thresh
)
{
  int k = conn_key_vect_.size();    
  ConnKeyT **conn_key_array = conn_key_vect_.data();
  
  CUDAMALLOCCTRL("&d_poiss_key_array_data_pt",&d_poiss_key_array_data_pt,
		 k*sizeof(ConnKeyT*));
  gpuErrchk(cudaMemcpy(d_poiss_key_array_data_pt, conn_key_array,
		       k*sizeof(ConnKeyT*), cudaMemcpyHostToDevice));
  
  regular_block_array<ConnKeyT> h_poiss_subarray[k];
  for (int i=0; i<k; i++) {
    h_poiss_subarray[i].h_data_pt = conn_key_array;
    h_poiss_subarray[i].data_pt = (ConnKeyT**)d_poiss_key_array_data_pt;
    h_poiss_subarray[i].block_size = conn_block_size_;
    h_poiss_subarray[i].offset = i * conn_block_size_;
    h_poiss_subarray[i].size = i<k-1 ? conn_block_size_ :
      n_conn_ - (k-1)*conn_block_size_;
  }
  
  CUDAMALLOCCTRL("&d_poiss_subarray",&d_poiss_subarray,
		 k*sizeof(regular_block_array<ConnKeyT>));
  gpuErrchk(cudaMemcpyAsync(d_poiss_subarray, h_poiss_subarray,
			    k*sizeof(regular_block_array<ConnKeyT>),
			    cudaMemcpyHostToDevice));
  
  CUDAMALLOCCTRL("&d_poiss_num",&d_poiss_num, 2*k*sizeof(int64_t));
  CUDAMALLOCCTRL("&d_poiss_sum",&d_poiss_sum, 2*sizeof(int64_t));
  
  
  CUDAMALLOCCTRL("&d_poiss_thresh",&d_poiss_thresh, 2*sizeof(ConnKeyT));
  
  return 0;
}





template<class ConnKeyT, class ConnStructT>
int ConnectionTemplate<ConnKeyT, ConnStructT>::buildDirectConnections
(inode_t i_node_0, inode_t n_node, int64_t &i_conn0, int64_t &n_dir_conn,
 int &max_delay, float* &d_mu_arr, void* &d_poiss_key_array)
{
  int k = conn_key_vect_.size();
  
  ConnKeyT **conn_key_array = (ConnKeyT**)conn_key_vect_.data();  
  ConnKeyT h_poiss_thresh[2];
  h_poiss_thresh[0] = 0;
  setConnSource(h_poiss_thresh[0], i_node_0);
    
  h_poiss_thresh[1] = 0;
  setConnSource(h_poiss_thresh[1], i_node_0 + n_node);

  gpuErrchk(cudaMemcpy(poiss_conn::d_poiss_thresh, h_poiss_thresh,
		       2*sizeof(ConnKeyT),
		       cudaMemcpyHostToDevice));
  
  int64_t h_poiss_num[2*k];
  int64_t *d_num0 = &poiss_conn::d_poiss_num[0];
  int64_t *d_num1 = &poiss_conn::d_poiss_num[k];
  int64_t *h_num0 = &h_poiss_num[0];
  int64_t *h_num1 = &h_poiss_num[k];

  search_multi_down<ConnKeyT, regular_block_array<ConnKeyT>, 1024>
    ( (regular_block_array<ConnKeyT>*) poiss_conn::d_poiss_subarray,
      k, &(((ConnKeyT*) poiss_conn::d_poiss_thresh)[0]), d_num0,
     &poiss_conn::d_poiss_sum[0]);
  CUDASYNC;
    
  search_multi_down<ConnKeyT, regular_block_array<ConnKeyT>, 1024>
    ( (regular_block_array<ConnKeyT>*) poiss_conn::d_poiss_subarray,
      k, &(((ConnKeyT*) poiss_conn::d_poiss_thresh)[1]), d_num1,
     &poiss_conn::d_poiss_sum[1]);
  CUDASYNC;

  gpuErrchk(cudaMemcpy(h_poiss_num, poiss_conn::d_poiss_num,
		       2*k*sizeof(int64_t), cudaMemcpyDeviceToHost));

  i_conn0 = 0;
  int64_t i_conn1 = 0;
  int ib0 = 0;
  int ib1 = 0;
  for (int i=0; i<k; i++) {
    if (h_num0[i] < conn_block_size_) {
      i_conn0 = conn_block_size_*i + h_num0[i];
      ib0 = i;
      break;
    }
  }
  
  for (int i=0; i<k; i++) {
    if (h_num1[i] < conn_block_size_) {
      i_conn1 = conn_block_size_*i + h_num1[i];
      ib1 = i;
      break;
    }
  }

  n_dir_conn = i_conn1 - i_conn0;
  
  if (n_dir_conn>0) {
    CUDAMALLOCCTRL("&d_poiss_key_array",&d_poiss_key_array,
		   n_dir_conn*sizeof(ConnKeyT));
    
    int64_t offset = 0;
    for (int ib=ib0; ib<=ib1; ib++) {
      if (ib==ib0 && ib==ib1) {
	gpuErrchk(cudaMemcpy(d_poiss_key_array,
			     conn_key_array[ib] + h_num0[ib],
			     n_dir_conn*sizeof(ConnKeyT),
			     cudaMemcpyDeviceToDevice));
	break;
      }
      else if (ib==ib0) {
	offset = conn_block_size_ - h_num0[ib];
	gpuErrchk(cudaMemcpy(d_poiss_key_array,
			     conn_key_array[ib] + h_num0[ib],
			     offset*sizeof(ConnKeyT),
			     cudaMemcpyDeviceToDevice));
      }
      else if (ib==ib1) {
	gpuErrchk(cudaMemcpy((ConnKeyT*)d_poiss_key_array + offset,
			     conn_key_array[ib],
			     h_num1[ib]*sizeof(ConnKeyT),
			     cudaMemcpyDeviceToDevice));
	break;
      }
      else {
	gpuErrchk(cudaMemcpy((ConnKeyT*)d_poiss_key_array + offset,
			     conn_key_array[ib],
			     conn_block_size_*sizeof(ConnKeyT),
			     cudaMemcpyDeviceToDevice));
	offset += conn_block_size_;
      }
    }

    unsigned int grid_dim_x, grid_dim_y;
  
    if (n_dir_conn<65536*1024) { // max grid dim * max block dim
      grid_dim_x = (n_dir_conn+1023)/1024;
      grid_dim_y = 1;
    }
    else {
      grid_dim_x = 64; // I think it's not necessary to increase it
      if (n_dir_conn>grid_dim_x*1024*65535) {
	throw ngpu_exception(std::string("Number of direct connections ")
			     + std::to_string(n_dir_conn) +
			     " larger than threshold "
			     + std::to_string(grid_dim_x*1024*65535));
      }
      grid_dim_y = (n_dir_conn + grid_dim_x*1024 -1) / (grid_dim_x*1024);
    }
    dim3 numBlocks(grid_dim_x, grid_dim_y);
    poissGenSubstractFirstNodeIndexKernel<ConnKeyT><<<numBlocks, 1024>>>
      (n_dir_conn, (ConnKeyT*)d_poiss_key_array, i_node_0);
    DBGCUDASYNC

  }

  // Find maximum delay of poisson direct connections
  int *d_max_delay; // maximum delay pointer in device memory
  CUDAMALLOCCTRL("&d_max_delay",&d_max_delay, sizeof(int));
  MaxDelay<ConnKeyT> max_op; // comparison operator used by Reduce function 
  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes,
			    (ConnKeyT*)d_poiss_key_array, d_max_delay,
			    n_dir_conn,
			    max_op, INT_MIN);
  // Allocate temporary storage
  CUDAMALLOCCTRL("&d_temp_storage",&d_temp_storage, temp_storage_bytes);
  // Run reduction
  cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes,
			    (ConnKeyT*)d_poiss_key_array, d_max_delay,
			    n_dir_conn,
			    max_op, INT_MIN);
  gpuErrchk(cudaMemcpy(&max_delay, d_max_delay, sizeof(int),
		       cudaMemcpyDeviceToHost));

  // max_delay = 200;
  printf("Max delay of direct (poisson generator) connections: %d\n",
	 max_delay);
  CUDAMALLOCCTRL("&d_mu_arr",&d_mu_arr, n_node*max_delay*sizeof(float));
  gpuErrchk(cudaMemset(d_mu_arr, 0, n_node*max_delay*sizeof(float)));
  
  /*
  CUDAFREECTRL("d_key_array_data_pt",d_key_array_data_pt);
  CUDAFREECTRL("d_subarray",d_subarray);
  CUDAFREECTRL("d_num",d_num);
  CUDAFREECTRL("d_sum",d_sum);
  CUDAFREECTRL("d_thresh",d_thresh);
  */
  
  return 0;
}


template <class ConnKeyT, class ConnStructT>
int ConnectionTemplate<ConnKeyT, ConnStructT>::sendDirectSpikes
(long long time_idx,
 int64_t i_conn0, int64_t n_dir_conn,
 inode_t n_node, int max_delay,
 float *d_mu_arr,
 void *d_poiss_key_array,
 curandState *d_curand_state)
{
  unsigned int grid_dim_x, grid_dim_y;
  
  if (n_dir_conn<65536*1024) { // max grid dim * max block dim
    grid_dim_x = (n_dir_conn+1023)/1024;
    grid_dim_y = 1;
  }
  else {
    grid_dim_x = 64; // I think it's not necessary to increase it
    if (n_dir_conn>grid_dim_x*1024*65535) {
      throw ngpu_exception(std::string("Number of direct connections ")
			   + std::to_string(n_dir_conn) +
			   " larger than threshold "
			   + std::to_string(grid_dim_x*1024*65535));
    }
    grid_dim_y = (n_dir_conn + grid_dim_x*1024 -1) / (grid_dim_x*1024);
  }
  dim3 numBlocks(grid_dim_x, grid_dim_y);
  sendDirectSpikeKernel<ConnKeyT, ConnStructT><<<numBlocks, 1024>>>
    (d_curand_state,
     time_idx, d_mu_arr, (ConnKeyT*)d_poiss_key_array,
     n_dir_conn, i_conn0,
     conn_block_size_, n_node, max_delay);

  DBGCUDASYNC

  return 0;
}

template <class ConnKeyT, class ConnStructT>
int ConnectionTemplate<ConnKeyT, ConnStructT>::revSpikeInit
(uint n_spike_buffers)
{
  //printf("n_spike_buffers: %d\n", n_spike_buffers);

  //////////////////////////////////////////////////////////////////////
  /////// Organize reverse connections (new version)
  // CHECK THE GLOBAL VARIABLES THAT MUST BE CONVERTED TO 64 bit ARRAYS
  //////////////////////////////////////////////////////////////////////  
  // Alloc 64 bit array of number of reverse connections per target node
  // and initialize it to 0
  int64_t *d_target_rev_conn_size_64;
  int64_t *d_target_rev_conn_cumul;
  CUDAMALLOCCTRL("&d_target_rev_conn_size_64",&d_target_rev_conn_size_64,
		       (n_spike_buffers+1)*sizeof(int64_t));
  gpuErrchk(cudaMemset(d_target_rev_conn_size_64, 0,
		       (n_spike_buffers+1)*sizeof(int64_t)));
  // Count number of reverse connections per target node
  countRevConnectionsKernel<ConnKeyT, ConnStructT>
    <<<(n_conn_+1023)/1024, 1024>>>
    (n_conn_, d_target_rev_conn_size_64);
  // Evaluate exclusive sum of reverse connections per target node
  // Allocate array for cumulative sum
  CUDAMALLOCCTRL("&d_target_rev_conn_cumul",&d_target_rev_conn_cumul,
		       (n_spike_buffers+1)*sizeof(int64_t));
  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
				d_target_rev_conn_size_64,
				d_target_rev_conn_cumul,
				n_spike_buffers+1);
  // Allocate temporary storage
  CUDAMALLOCCTRL("&d_temp_storage",&d_temp_storage, temp_storage_bytes);
  // Run exclusive prefix sum
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
				d_target_rev_conn_size_64,
				d_target_rev_conn_cumul,
				n_spike_buffers+1);
  // The last element is the total number of reverse connections
  gpuErrchk(cudaMemcpy(&n_rev_conn_, &d_target_rev_conn_cumul[n_spike_buffers],
		       sizeof(int64_t), cudaMemcpyDeviceToHost));
  if (n_rev_conn_ > 0) {
    // Allocate array of reverse connection indexes
    // CHECK THAT d_RevConnections is of type int64_t array
    CUDAMALLOCCTRL("&d_rev_conn_",&d_rev_conn_, n_rev_conn_*sizeof(int64_t));  
    // For each target node evaluate the pointer
    // to its first reverse connection using the exclusive sum
    // CHECK THAT d_target_rev_conn_ is of type int64_t* pointer
    CUDAMALLOCCTRL("&d_target_rev_conn_",&d_target_rev_conn_, n_spike_buffers
			 *sizeof(int64_t*));
    setTargetRevConnectionsPtKernel<<<(n_spike_buffers+1023)/1024, 1024>>>
      (n_spike_buffers, d_target_rev_conn_cumul,
       d_target_rev_conn_, d_rev_conn_);
  
    // alloc 32 bit array of number of reverse connections per target node
    CUDAMALLOCCTRL("&d_target_rev_conn_size_",&d_target_rev_conn_size_,
			 n_spike_buffers*sizeof(int));
    // and initialize it to 0
    gpuErrchk(cudaMemset(d_target_rev_conn_size_, 0,
			 n_spike_buffers*sizeof(int)));
    // Fill array of reverse connection indexes
    setRevConnectionsIndexKernel<ConnKeyT, ConnStructT>
      <<<(n_conn_+1023)/1024, 1024>>>
      (n_conn_, d_target_rev_conn_size_, d_target_rev_conn_);

    revConnectionInitKernel<<<1,1>>>
      (d_rev_conn_, d_target_rev_conn_size_, d_target_rev_conn_);

    setConnectionSpikeTime
      <<<(n_conn_+1023)/1024, 1024>>>
      (n_conn_, 0x8000);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    CUDAMALLOCCTRL("&d_rev_spike_num_",&d_rev_spike_num_, sizeof(uint));
  
    CUDAMALLOCCTRL("&d_rev_spike_target_",&d_rev_spike_target_,
		   n_spike_buffers*sizeof(uint));
    
    CUDAMALLOCCTRL("&d_rev_spike_n_conn",&d_rev_spike_n_conn_,
		   n_spike_buffers*sizeof(int));

    deviceRevSpikeInit<<<1,1>>>(d_rev_spike_num_, d_rev_spike_target_,
				d_rev_spike_n_conn_);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }
  
  CUDAFREECTRL("d_temp_storage",d_temp_storage);
  CUDAFREECTRL("d_target_rev_conn_size_64",d_target_rev_conn_size_64);
  CUDAFREECTRL("d_target_rev_conn_cumul",d_target_rev_conn_cumul);

  return 0;
}

template <class ConnKeyT, class ConnStructT>
int ConnectionTemplate<ConnKeyT, ConnStructT>::revSpikeFree()
{
  CUDAFREECTRL("&d_rev_spike_num_",&d_rev_spike_num_);
  CUDAFREECTRL("&d_rev_spike_target_",&d_rev_spike_target_);
  CUDAFREECTRL("&d_rev_spike_n_conn_",&d_rev_spike_n_conn_);

  return 0;
}


template <class ConnKeyT, class ConnStructT>
int ConnectionTemplate<ConnKeyT, ConnStructT>::resetConnectionSpikeTimeUp()
{  
  resetConnectionSpikeTimeUpKernel
    <<<(n_conn_+1023)/1024, 1024>>>
    (n_conn_);
  gpuErrchk( cudaPeekAtLastError() );

  return 0;
}

template <class ConnKeyT, class ConnStructT>
int ConnectionTemplate<ConnKeyT, ConnStructT>::resetConnectionSpikeTimeDown()
{  
  resetConnectionSpikeTimeDownKernel
    <<<(n_conn_+1023)/1024, 1024>>>
    (n_conn_);
  gpuErrchk( cudaPeekAtLastError() );

  return 0;
}




#endif
