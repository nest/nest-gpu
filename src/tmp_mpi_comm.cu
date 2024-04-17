#include <iostream>
#include <vector>

// Maybe the world group can be initialized by a specialized command in the Init that calls
// the CreateHostGroup method giving the array of all hosts' indexes (0,1,2,3, ...) as parameter

// array of local indexes  of all host groups.
// Index: global host group index
// Each element is the local index of the group, -1 if this host is not in the group
// n_host_groups_  = host_group_local_id_.size()
// the first element corresponds to the world group and must be equal to zero
std::vector<int> host_group_local_id_;
// local group of hosts (i.e. group of MPI processes)
// two-dimensional array, indexes: [group_local_id][i_host]
// The first element corresponds to the world group and should contain all hosts
// however maybe in this case it is not necessary to explicitly store their indexes
// n_hosts[group_local_id] = host_group_[group_local_id].size();
std::vector<std::vector< int >> host_group_;

// Indexes of source nodes of each host that communicate spikes to a specific host group
// three-dimensional array, indexes: [group_local_id][i_host][i_node]
// besides the index group_local_id, the content should be the same across all hosts of the group
std::vector<std::vector< std::unordered_set< int > > > host_group_source_node_;

// Method that creates a group of hosts for remote spike communication (i.e. a group of MPI processes)
// host_arr: array of host inexes, n_hosts: nomber of hosts in the group
int CreateHostGroup(int *host_arr, int n_hosts)
{
  if (first_connection_flag_ == false) {
    throw ngpu_exception("Host groups must be defined before creating "
			 "connections");
  }
  // pushes all the host indexes in a vector, hg, and check whether this host is in the group 
  std::vector<int> hg;
  bool this_host_is_in_group = false;
  for (int ih=0; ih<n_hosts; ih++) {
    int i_host = host_arr[ih];
    // check whether this host is in the group 
    if (i_host == this_host) {
      this_host_is_in_group = true;
    }
    hg.push_back(i_host);
  }
  // the code in the block is executed only if this host is in the group
  if (this_host_is_in_group) {
    // set the local id of the group to be the current size of the local host group array
    int group_local_id = host_group_.size();
    // push the local id in the array of local indexes  of all host groups
    host_group_local_id_.push_back(group_local_id);
    // push the new group into the host_group_ vector 
    host_group_.push_back(hg);
    // push a vector of empty unordered sets into host_group_source_node_
    std::vector< std::unordered_set< int > > empty_i_node(n_hosts);
    host_group_source_node_.push_back(empty_i_node);
  }
  else {
    // if this host is not in the group, set the entry of host_group_local_id_ to -1 
    host_group_local_id_.push_back(-1);
  }
  // return as output the index of the last entry in host_group_local_id_
  // which correspond to the newly created group
  return host_group_local_id_.size() - 1;
}

/////////////////////////////////////////////
// integrate/merge the following function in the RemoteConnect method.
// i_host_group is the global host-group index, given as optional argument to the
// RemoteConnect command. Default value is either -1 or 0 (initialized as kernel parameter).
// i_host_group = -1 for point-to-point MPI communication
//              = 0 for the world group, which includes all hosts (i.e. all MPI processes)
//              > 0 for all the other host groups
template <class T>
int RemoteConnect(int source_host, T source, int n_source, int i_host_group)
{
  if (i_host_group>=0) { // not a point-to-point MPI communication
    int group_local_id = host_group_local_id_[i_host_group];
    if (group_local_id >= 0) { // this host is in group
      // find the source host index in the host group
      i_host = find(host_group_[group_local_id], source_host);
      for (int i=0; i<n_source; i++) {
	int i_source = GetSource(source, i);
	host_group_source_node_[group_local_id][i_host].insert(i_source);
      }
    }
  }
  // if i_host_group<0, i.e. a point-to-point MPI communication is required
  // and this host is the source (but it is not a local connection) call RemoteConnectTarget
  else if (source_host!=target_host && this_host==source_host) {
    return RemoteConnectTarget();
  }
  // the following are the usual calls of RemoteConnect
  if (source_host == target_host && this_host==source_host) {
    return Connect();
  } 
  if (this_host == target_host) {
    return RemoteConnectSource();
  }
 
  return 0;
}

/////////////////////////////////////////////
// integrate/merge the following function in the Calibrate method.
// in the target
int Calibrate()
{
  // loop on source host indexes
  for (int ish=0; ish<n_hosts_; ish++) {
    if (ish==this_host_) {
      continue;
    }
    // loop on remote-source-node-to-local-image-node map blocks
    for (int ib=0; ib<n_map_blocks; ib++) {
      if (ib<n_map_blocks-1) {
	n_elem = map_block_size;
      }
      else {
	n_elem = map_size%map_block_size;
      }
      /////////// !!!!!!!!!!!! non ho definito ig !!!!!!!!!!!!!!!!!!!!!!
      CudaMemCpy(&h_remote_source_node_[ig][ish][ib*map_block_size],
		 remote_source_node_[ig*n_hosts_ + ish][i_block], n_elem);
      CudaMemCpy(&h_local_node_index_[ig][ish][ib*map_block_size],
		 local_node_index_[ig*n_hosts_ + ish][i_block], n_elem);
    }
    for (int i=0; i<map_size; i++) {
      i_node = h_remote_source_node_[ig][ish].size();
      if (pos=find(i_node, host_group_source_node_vect_
		   [group_local_id][source_host]>0)) {
	host_group_local_node_index_vect_[pos] =
	  h_local_node_index_[ig][ish][i];
      }
    }
  }

  // only in the source
  for (int i=0; i<n; i++) {
    i_source = host_group_source_node_vect_[group_local_id][source_host][i];
    my_map[i_source] = i;
  }
}
