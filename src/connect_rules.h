/*
 *  connect_rules.h
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





#ifndef CONNECTRULES_H
#define CONNECTRULES_H

#include <iostream>
#include <numeric>
#include <stdio.h>
#include "nestgpu.h"

#ifdef HAVE_MPI
#include "connect_mpi.h"
#endif

#define THREAD_MAXNUM 1
#define THREAD_IDX 0

template<>
int RemoteNode<int>::GetINode(int in)
{
  return i_node_ + in;
}

template<>
int RemoteNode<int*>::GetINode(int in)
{
  return *(i_node_ + in);
}

template <class T1, class T2>
int NESTGPU::_Connect(T1 source, int n_source, T2 target, int n_target,
			ConnSpec &conn_spec, SynSpec &syn_spec)
{
  CheckUncalibrated("Connections cannot be created after calibration");
  ////////////////////////
  // TO DO:
  // if (syn_spec.weight_distr_ != NULL) {
  //   syn_spec.weight_array_ = Distribution(syn_spec.weight_distr, n);
  // }
  // if (syn_spec.delay_distr_ != NULL) {
  //   syn_spec.delay_array_ = Distribution(syn_spec.delay_distr, n);
  // }
  
  switch (conn_spec.rule_) {
  case ONE_TO_ONE:
    if (n_source != n_target) {
      throw ngpu_exception("Number of source and target nodes must be equal "
			   "for the one-to-one connection rule");
    }
    return _ConnectOneToOne<T1, T2>(source, target, n_source, syn_spec);
    break;
  case ALL_TO_ALL:
    return _ConnectAllToAll<T1, T2>(source, n_source, target, n_target,
				    syn_spec);
    break;
  case FIXED_TOTAL_NUMBER:
    return _ConnectFixedTotalNumber<T1, T2>(source, n_source, target, n_target,
				       conn_spec.total_num_, syn_spec);
    break;
  case FIXED_INDEGREE:
    return _ConnectFixedIndegree<T1, T2>(source, n_source, target, n_target,
				    conn_spec.indegree_, syn_spec);
    break;
  case FIXED_OUTDEGREE:
    return _ConnectFixedOutdegree<T1, T2>(source, n_source, target, n_target,
					  conn_spec.outdegree_, syn_spec);
    break;
  default:
    throw ngpu_exception("Unknown connection rule");
  }
  return 0;
}

template<class T1, class T2>
int NESTGPU::_SingleConnect(T1 source, int i_source, T2 target, int i_target,
			      int i_array, SynSpec &syn_spec)
{
  float weight;
  if (syn_spec.weight_array_ != NULL) {
    weight = syn_spec.weight_array_[i_array];
  }
  else {
    weight = syn_spec.weight_;
  }
  float delay;
  if (syn_spec.delay_array_ != NULL) {
    delay = syn_spec.delay_array_[i_array];
  }
  else {
    delay = syn_spec.delay_;
  }
  return _SingleConnect<T1, T2>(source, i_source, target, i_target,
				weight, delay, i_array, syn_spec);
}

template<class T1, class T2>
int NESTGPU::_SingleConnect(T1 source, int i_source, T2 target, int i_target,
			      float weight, float delay, int i_array,
			      SynSpec &syn_spec)
{
  throw ngpu_exception("Unknown type for _SingleConnect template");
}

template<class T>
int NESTGPU::_RemoteSingleConnect(int i_source, T target, int i_target,
				    int i_array, SynSpec &syn_spec)
{
  float weight;
  if (syn_spec.weight_array_ != NULL) {
    weight = syn_spec.weight_array_[i_array];
  }
  else {
    weight = syn_spec.weight_;
  }
  float delay;
  if (syn_spec.delay_array_ != NULL) {
    delay = syn_spec.delay_array_[i_array];
  }
  else {
    delay = syn_spec.delay_;
  }
  return _RemoteSingleConnect<T>(i_source, target, i_target,
				 weight, delay, i_array, syn_spec);
}

template<class T>
int NESTGPU::_RemoteSingleConnect(int i_source, T target, int i_target,
				    float weight, float delay, int i_array,
				    SynSpec &syn_spec)
{
  throw ngpu_exception("Unknown type for _RemoteSingleConnect template");
}




template <class T1, class T2>
int NESTGPU::_ConnectOneToOne(T1 source, T2 target, int n_node,
				SynSpec &syn_spec)	       
{
  for (int in=0; in<n_node; in++) {
    _SingleConnect<T1, T2>(source, in, target, in, in, syn_spec);
  }

  return 0;
}

template <class T1, class T2>
int NESTGPU::_ConnectAllToAll
(T1 source, int n_source, T2 target, int n_target, SynSpec &syn_spec)
{
  for (int itn=0; itn<n_target; itn++) {
    for (int isn=0; isn<n_source; isn++) {
      size_t i_array = (size_t)itn*n_source + isn;
      _SingleConnect<T1, T2>(source, isn, target, itn, i_array, syn_spec);
    }
  }

  return 0;
}

template <class T1, class T2>
int NESTGPU::_ConnectFixedTotalNumber
(T1 source, int n_source, T2 target, int n_target, int n_conn,
 SynSpec &syn_spec)
{
  unsigned int *rnd = RandomInt(2*n_conn);
  for (int i_conn=0; i_conn<n_conn; i_conn++) {
    int isn = rnd[2*i_conn] % n_source;
    int itn = rnd[2*i_conn+1] % n_target;
    _SingleConnect<T1, T2>(source, isn, target, itn, i_conn, syn_spec);
  }
  delete[] rnd;
  
  return 0;
}


template <class T1, class T2>
int NESTGPU::_ConnectFixedIndegree
(
 T1 source, int n_source, T2 target, int n_target, int indegree,
 SynSpec &syn_spec
 )
{
  const int method_thresh = 5;
  if (indegree>n_source) {
    throw ngpu_exception("Indegree larger than number of source nodes");
  }
  int n_rnd = indegree*THREAD_MAXNUM;
  if (n_source>=method_thresh*indegree) { // nuovo metodo
    n_rnd *= 5; 
  } 
  unsigned int *rnd = RandomInt(n_rnd);

  for (int k=0; k<n_target; k+=THREAD_MAXNUM) {
    for (int ith=0; ith<THREAD_MAXNUM; ith++) {
      int itn = k + ith;
      if (itn < n_target) {
	std::vector<int> int_vect;
      	//int_vect.clear();
	if (n_source<method_thresh*indegree) { // vecchio metodo
	  //https://stackoverflow.com/questions/18625223
	  // v = sequence(0, n_source-1)
	  int_vect.resize(n_source);
	  int start = 0;
	  std::iota(int_vect.begin(), int_vect.end(), start);
	  for (int i=0; i<indegree; i++) {
	    int j = i + rnd[i*THREAD_MAXNUM + ith] % (n_source - i);
	    if (j != i) {
	      std::swap(int_vect[i], int_vect[j]);
	    }
	  }
	}
	else { // nuovo metodo
	  std::vector<int> sorted_vect;
	  for (int i=0; i<indegree; i++) {
	    int i1 = 0;
	    std::vector<int>::iterator iter;
	    int j;
	    do {
	      j = rnd[(i1*indegree + i)*THREAD_MAXNUM + ith]
		% n_source;
	      // https://riptutorial.com/cplusplus/example/7270/using-a-sorted-vector-for-fast-element-lookup
	      // check if j is in target_vect
	      iter = std::lower_bound(sorted_vect.begin(),
				      sorted_vect.end(), j);
	      i1++;
	    } while (iter != sorted_vect.end() && *iter == j); // we found j 
	    sorted_vect.insert(iter, j);
	    int_vect.push_back(j);
	  }
	}
	for (int i=0; i<indegree; i++) {
	  int isn = int_vect[i];
	  size_t i_array = (size_t)itn*indegree + i;
	  _SingleConnect<T1, T2>(source, isn, target, itn, i_array, syn_spec);
	}
      }
    }
  }
  delete[] rnd;
  
  return 0;
}


template <class T1, class T2>
int NESTGPU::_ConnectFixedOutdegree
(
 T1 source, int n_source, T2 target, int n_target, int outdegree,
 SynSpec &syn_spec
 )
{
  const int method_thresh = 5;
  if (outdegree>n_target) {
    throw ngpu_exception("Outdegree larger than number of target nodes");
  }
  int n_rnd = outdegree*THREAD_MAXNUM;
  if (n_target>=method_thresh*outdegree) { // choose method
    n_rnd *= 5; 
  } 
  unsigned int *rnd = RandomInt(n_rnd);


  for (int is0=0; is0<n_source; is0+=THREAD_MAXNUM) {
    for (int ith=0; ith<THREAD_MAXNUM; ith++) {
      int isn = is0 + ith;
      if (isn < n_source) {
	std::vector<int> int_vect;
 	if (n_target<method_thresh*outdegree) { // choose method
	  //https://stackoverflow.com/questions/18625223
	  // v = sequence(0, n_target-1)
	  int_vect.resize(n_target);
	  int start = 0;
	  std::iota(int_vect.begin(), int_vect.end(), start);
	  for (int i=0; i<outdegree; i++) {
	    int j = i + rnd[i*THREAD_MAXNUM + ith] % (n_target - i);
	    if (j != i) {
	      std::swap(int_vect[i], int_vect[j]);
	    }
	  }
	}
	else { // other method
	  std::vector<int> sorted_vect;
	  for (int i=0; i<outdegree; i++) {
	    int i1 = 0;
	    std::vector<int>::iterator iter;
	    int j;
	    do {
	      j = rnd[(i1*outdegree + i)*THREAD_MAXNUM + ith]
		% n_target;
	      // https://riptutorial.com/cplusplus/example/7270/using-a-sorted-vector-for-fast-element-lookup
	      // check if j is in target_vect
	      iter = std::lower_bound(sorted_vect.begin(),
				      sorted_vect.end(), j);
	      i1++;
	    } while (iter != sorted_vect.end() && *iter == j); // we found j 
	    sorted_vect.insert(iter, j);
	    int_vect.push_back(j);
	  }
	}	
	for (int k=0; k<outdegree; k++) {
	  int itn = int_vect[k];
	  size_t i_array = (size_t)isn*outdegree + k;
      _SingleConnect<T1, T2>(source, isn, target, itn, i_array, syn_spec);
	}
      }
    }
  }
  delete[] rnd;
  
  return 0;
}


#ifdef HAVE_MPI

template <class T1, class T2>
int NESTGPU::_RemoteConnect(RemoteNode<T1> source, int n_source,
			      RemoteNode<T2> target, int n_target,
			      ConnSpec &conn_spec, SynSpec &syn_spec)
{
  CheckUncalibrated("Connections cannot be created after calibration");
  switch (conn_spec.rule_) {
  case ONE_TO_ONE:
    if (n_source != n_target) {
      throw ngpu_exception("Number of source and target nodes must be equal "
			   "for the one-to-one connection rule");
    }
    return _RemoteConnectOneToOne<T1, T2>(source, target, n_source, syn_spec);
    break;
  case ALL_TO_ALL:
    return _RemoteConnectAllToAll<T1, T2>(source, n_source, target, n_target,
				     syn_spec);
    break;
  case FIXED_TOTAL_NUMBER:
    return _RemoteConnectFixedTotalNumber<T1, T2>(source, n_source, target,
						  n_target,
						  conn_spec.total_num_,
						  syn_spec);
    break;
  case FIXED_INDEGREE:
    return _RemoteConnectFixedIndegree<T1, T2>(source, n_source, target,
					       n_target, conn_spec.indegree_,
					       syn_spec);
    break;
  case FIXED_OUTDEGREE:
    return _RemoteConnectFixedOutdegree<T1, T2>(source, n_source, target,
						n_target, conn_spec.outdegree_,
						syn_spec);
    break;
  default:
    throw ngpu_exception("Unknown connection rule");
  }
  return 0;
}


template <class T1, class T2>
  int NESTGPU::_RemoteConnectOneToOne
  (RemoteNode<T1> source, RemoteNode<T2> target, int n_node,
   SynSpec &syn_spec)
{
  if (MpiId()==source.i_host_ && source.i_host_==target.i_host_) {
    return _ConnectOneToOne<T1, T2>(source.i_node_, target.i_node_,
				    n_node, syn_spec);
  }
  else if (MpiId()==source.i_host_ || MpiId()==target.i_host_) {
    int *i_remote_node_arr = new int[n_node];
    if (MpiId() == target.i_host_) {
      connect_mpi_->MPI_Send_int(&n_remote_node_, 1, source.i_host_);
      connect_mpi_->MPI_Recv_int(&n_remote_node_, 1, source.i_host_);
      connect_mpi_->MPI_Recv_int(i_remote_node_arr, n_node, source.i_host_);

      for (int in=0; in<n_node; in++) {
	int i_remote_node = i_remote_node_arr[in];
	_RemoteSingleConnect<T2>(i_remote_node, target.i_node_, in,
				 in, syn_spec);
      }
    }
    else if (MpiId() == source.i_host_) {
      int i_new_remote_node;
      connect_mpi_->MPI_Recv_int(&i_new_remote_node, 1, target.i_host_);
      for (int in=0; in<n_node; in++) {
	int i_source_node = source.GetINode(in);
	int i_remote_node = -1;
	for (std::vector<ExternalConnectionNode >::iterator it =
	       connect_mpi_->extern_connection_[i_source_node].begin();
	     it <  connect_mpi_->extern_connection_[i_source_node].end();
	     it++) {
	  if ((*it).target_host_id == target.i_host_) {
	    i_remote_node = (*it).remote_node_id;
	    break;
	  }
	}
	if (i_remote_node == -1) {
	  i_remote_node = i_new_remote_node;
	  i_new_remote_node++;
	  ExternalConnectionNode conn_node = {target.i_host_, i_remote_node};
	  connect_mpi_->extern_connection_[i_source_node].push_back(conn_node);
	}
	i_remote_node_arr[in] = i_remote_node;
      }
      connect_mpi_->MPI_Send_int(&i_new_remote_node, 1, target.i_host_);
      connect_mpi_->MPI_Send_int(i_remote_node_arr, n_node, target.i_host_);
    }
    delete[] i_remote_node_arr;
  }
  MPI_Barrier( MPI_COMM_WORLD );

  return 0;
}


template <class T1, class T2>
  int NESTGPU::_RemoteConnectAllToAll
  (RemoteNode<T1> source, int n_source, RemoteNode<T2> target, int n_target,
   SynSpec &syn_spec)
{
  if (MpiId()==source.i_host_ && source.i_host_==target.i_host_) {
    return _ConnectAllToAll<T1, T2>(source.i_node_, n_source,
					 target.i_node_, n_target, syn_spec);
  }
  else if (MpiId()==source.i_host_ || MpiId()==target.i_host_) {
    int *i_remote_node_arr = new int[n_target*n_source];
    if (MpiId() == target.i_host_) {
      connect_mpi_->MPI_Send_int(&n_remote_node_, 1, source.i_host_);
      connect_mpi_->MPI_Recv_int(&n_remote_node_, 1, source.i_host_);
      connect_mpi_->MPI_Recv_int(i_remote_node_arr, n_target*n_source,
				 source.i_host_);

      for (int k=0; k<n_target; k++) {
	for (int i=0; i<n_source; i++) {
      	  int i_remote_node = i_remote_node_arr[k*n_source+i];
	  size_t i_array = (size_t)k*n_source + i;
	  _RemoteSingleConnect<T2>(i_remote_node, target.i_node_, k,
				   i_array, syn_spec);
	}
      }
    }
    else if (MpiId() == source.i_host_) {
      int i_new_remote_node;
      connect_mpi_->MPI_Recv_int(&i_new_remote_node, 1, target.i_host_);
      for (int k=0; k<n_target; k++) {
	for (int i=0; i<n_source; i++) {
	  int i_source_node = source.GetINode(i);	  
	  int i_remote_node = -1;
	  for (std::vector<ExternalConnectionNode >::iterator it =
		 connect_mpi_->extern_connection_[i_source_node].begin();
	       it <  connect_mpi_->extern_connection_[i_source_node].end(); it++) {
	    if ((*it).target_host_id == target.i_host_) {
	      i_remote_node = (*it).remote_node_id;
	      break;
	    }
	  }
	  if (i_remote_node == -1) {
	    i_remote_node = i_new_remote_node;
	    i_new_remote_node++;
	    ExternalConnectionNode conn_node = {target.i_host_, i_remote_node};
	    connect_mpi_->extern_connection_[i_source_node].push_back(conn_node);
	  }
	  i_remote_node_arr[k*n_source+i] = i_remote_node;
	}
      }
      connect_mpi_->MPI_Send_int(&i_new_remote_node, 1, target.i_host_);
      connect_mpi_->MPI_Send_int(i_remote_node_arr, n_target*n_source,
				 target.i_host_);
    }
    delete[] i_remote_node_arr;
  }
  MPI_Barrier( MPI_COMM_WORLD );

  return 0;
}

template <class T1, class T2>
  int NESTGPU::_RemoteConnectFixedTotalNumber
  (RemoteNode<T1> source, int n_source, RemoteNode<T2> target, int n_target,
   int n_conn, SynSpec &syn_spec)
{
  if (MpiId()==source.i_host_ && source.i_host_==target.i_host_) {
    return _ConnectFixedTotalNumber<T1, T2>(source.i_node_, n_source,
					 target.i_node_, n_target,
					    n_conn, syn_spec);
  }
  unsigned int *rnd = RandomInt(2*n_conn);
  if (MpiId()==source.i_host_ || MpiId()==target.i_host_) {
    int *i_remote_node_arr = new int[n_conn];
    if (MpiId() == target.i_host_) {
      connect_mpi_->MPI_Send_int(&n_remote_node_, 1, source.i_host_);
      connect_mpi_->MPI_Recv_int(&n_remote_node_, 1, source.i_host_);
      connect_mpi_->MPI_Recv_int(i_remote_node_arr, n_conn, source.i_host_);
      
      for (int i_conn=0; i_conn<n_conn; i_conn++) {
	int i_remote_node = i_remote_node_arr[i_conn];
	int itn = rnd[2*i_conn+1] % n_target;
	_RemoteSingleConnect<T2>(i_remote_node, target.i_node_, itn,
			       i_conn, syn_spec);
      }
    }
    else if (MpiId() == source.i_host_) {
      int i_new_remote_node;
      connect_mpi_->MPI_Recv_int(&i_new_remote_node, 1, target.i_host_);
      for (int i_conn=0; i_conn<n_conn; i_conn++) {
	int isn = rnd[2*i_conn] % n_source;
	int i_source_node = source.GetINode(isn);	  
	int i_remote_node = -1;
	for (std::vector<ExternalConnectionNode >::iterator it =
	       connect_mpi_->extern_connection_[i_source_node].begin();
	     it <  connect_mpi_->extern_connection_[i_source_node].end();
	     it++) {
	  if ((*it).target_host_id == target.i_host_) {
	    i_remote_node = (*it).remote_node_id;
	    break;
	  }
	}
	if (i_remote_node == -1) {
	  i_remote_node = i_new_remote_node;
	  i_new_remote_node++;
	  ExternalConnectionNode conn_node = {target.i_host_, i_remote_node};
	  connect_mpi_->extern_connection_[i_source_node].push_back(conn_node);
	}
	i_remote_node_arr[i_conn] = i_remote_node;
      }
      connect_mpi_->MPI_Send_int(&i_new_remote_node, 1, target.i_host_);
      connect_mpi_->MPI_Send_int(i_remote_node_arr, n_conn,
				 target.i_host_);
    }
    delete[] i_remote_node_arr;
  }
  MPI_Barrier( MPI_COMM_WORLD );

  return 0;
}

template <class T1, class T2>
  int NESTGPU::_RemoteConnectFixedIndegree
  (RemoteNode<T1> source, int n_source, RemoteNode<T2> target, int n_target,
   int indegree, SynSpec &syn_spec)
{
  const int method_thresh = 5;
  if (indegree>n_source) {
    throw ngpu_exception("Indegree larger than number of source nodes");
  }
  if (MpiId()==source.i_host_ && source.i_host_==target.i_host_) {
    return _ConnectFixedIndegree<T1, T2>(source.i_node_, n_source,
					 target.i_node_, n_target, indegree,
					 syn_spec);
  }
  else if (MpiId()==source.i_host_ || MpiId()==target.i_host_) {
    int *i_remote_node_arr = new int[n_target*indegree];
    if (MpiId() == target.i_host_) {
      connect_mpi_->MPI_Send_int(&n_remote_node_, 1, source.i_host_);
      connect_mpi_->MPI_Recv_int(&n_remote_node_, 1, source.i_host_);
      connect_mpi_->MPI_Recv_int(i_remote_node_arr, n_target*indegree,
				 source.i_host_);

      for (int k=0; k<n_target; k++) {
	for (int i=0; i<indegree; i++) {
      	  int i_remote_node = i_remote_node_arr[k*indegree+i];
	  size_t i_array = (size_t)k*indegree + i;
	  _RemoteSingleConnect<T2>(i_remote_node, target.i_node_, k,
				   i_array, syn_spec);
	  
	}
      }
    }
    else if (MpiId() == source.i_host_) {
      int i_new_remote_node;
      connect_mpi_->MPI_Recv_int(&i_new_remote_node, 1, target.i_host_);
      int n_rnd = indegree;
      if (n_source>=method_thresh*indegree) { // choose method
	n_rnd *= 5; 
      } 
      unsigned int *rnd = RandomInt(n_rnd);

      //std::vector<int> input_array;
      //for (int i=0; i<n_source; i++) {
      //input_array.push_back(source.GetINode(i));
      //}
      for (int k=0; k<n_target; k++) {
	std::vector<int> int_vect;
	if (n_source<method_thresh*indegree) { // choose method
	  //https://stackoverflow.com/questions/18625223
	  // v = sequence(0, n_source-1)
	  int_vect.resize(n_source);
	  int start = 0;
	  std::iota(int_vect.begin(), int_vect.end(), start);
	  for (int i=0; i<indegree; i++) {
	    int j = i + rnd[i] % (n_source - i);
	    if (j != i) {
	      std::swap(int_vect[i], int_vect[j]);
	    }
	  }
	}
	else { // nuovo metodo
	  std::vector<int> sorted_vect;
	  for (int i=0; i<indegree; i++) {
	    int i1 = 0;
	    std::vector<int>::iterator iter;
	    int j;
	    do {
	      j = rnd[i1*indegree + i]
		% n_source;
	      // https://riptutorial.com/cplusplus/example/7270/using-a-sorted-vector-for-fast-element-lookup
	      // check if j is in target_vect
	      iter = std::lower_bound(sorted_vect.begin(),
				      sorted_vect.end(), j);
	      i1++;
	    } while (iter != sorted_vect.end() && *iter == j); // we found j 
	    sorted_vect.insert(iter, j);
	    int_vect.push_back(j);
	  }
	}
	for (int i=0; i<indegree; i++) {
	  int i_source_node = source.GetINode(int_vect[i]);
	  int i_remote_node = -1;
	  for (std::vector<ExternalConnectionNode >::iterator it =
		 connect_mpi_->extern_connection_[i_source_node].begin();
	       it <  connect_mpi_->extern_connection_[i_source_node].end();
	       it++) {
	    if ((*it).target_host_id == target.i_host_) {
	      i_remote_node = (*it).remote_node_id;
	      break;
	    }
	  }
	  if (i_remote_node == -1) {
	    i_remote_node = i_new_remote_node;
	    i_new_remote_node++;
	    ExternalConnectionNode conn_node = {target.i_host_, i_remote_node};
	    connect_mpi_->extern_connection_[i_source_node].push_back(conn_node);
	  }
	  i_remote_node_arr[k*indegree+i] = i_remote_node;
	}
      }
      connect_mpi_->MPI_Send_int(&i_new_remote_node, 1, target.i_host_);
      connect_mpi_->MPI_Send_int(i_remote_node_arr, n_target*indegree,
				 target.i_host_);
      delete[] rnd;
    }
    delete[] i_remote_node_arr;
  }
  MPI_Barrier( MPI_COMM_WORLD );

  return 0;
}


template <class T1, class T2>
  int NESTGPU::_RemoteConnectFixedOutdegree
  (RemoteNode<T1> source, int n_source, RemoteNode<T2> target, int n_target,
   int outdegree, SynSpec &syn_spec)
{
  const int method_thresh = 5;
  if (outdegree>n_target) {
    throw ngpu_exception("Outdegree larger than number of target nodes");
  }
  if (MpiId()==source.i_host_ && source.i_host_==target.i_host_) {
    return _ConnectFixedOutdegree<T1, T2>(source.i_node_, n_source,
					 target.i_node_, n_target, outdegree,
					 syn_spec);
  }
  else if (MpiId()==source.i_host_ || MpiId()==target.i_host_) {
    int *i_remote_node_arr = new int[n_source];
    if (MpiId() == target.i_host_) {
      connect_mpi_->MPI_Send_int(&n_remote_node_, 1, source.i_host_);
      connect_mpi_->MPI_Recv_int(&n_remote_node_, 1, source.i_host_);
      connect_mpi_->MPI_Recv_int(i_remote_node_arr, n_source,
				 source.i_host_);

      int n_rnd = outdegree;
      if (n_target>=method_thresh*outdegree) { // choose method
	n_rnd *= 5; 
      }
      unsigned int *rnd = RandomInt(n_rnd);

      for (int isn=0; isn<n_source; isn++) {
	std::vector<int> int_vect;
 	if (n_target<method_thresh*outdegree) { // choose method
	  //https://stackoverflow.com/questions/18625223
	  // v = sequence(0, n_target-1)
	  int_vect.resize(n_target);
	  int start = 0;
	  std::iota(int_vect.begin(), int_vect.end(), start);
	  for (int i=0; i<outdegree; i++) {
	    int j = i + rnd[i] % (n_target - i);
	    if (j != i) {
	      std::swap(int_vect[i], int_vect[j]);
	    }
	  }
	}
	else { // other method
	  std::vector<int> sorted_vect;
	  for (int i=0; i<outdegree; i++) {
	    int i1 = 0;
	    std::vector<int>::iterator iter;
	    int j;
	    do {
	      j = rnd[i1*outdegree + i] % n_target;
	      // https://riptutorial.com/cplusplus/example/7270/using-a-sorted-vector-for-fast-element-lookup
	      // check if j is in target_vect
	      iter = std::lower_bound(sorted_vect.begin(),
				      sorted_vect.end(), j);
	      i1++;
	    } while (iter != sorted_vect.end() && *iter == j); // we found j 
	    sorted_vect.insert(iter, j);
	    int_vect.push_back(j);
	  }
	}	
	for (int k=0; k<outdegree; k++) {
      	  int i_remote_node = i_remote_node_arr[isn];
	  int itn = int_vect[k];
	  size_t i_array = (size_t)isn*outdegree + k;
	  _RemoteSingleConnect<T2>(i_remote_node, target.i_node_, itn,
				   i_array, syn_spec);
	}
      }
      delete[] rnd;
    }
    else if (MpiId() == source.i_host_) {
      int i_new_remote_node;
      connect_mpi_->MPI_Recv_int(&i_new_remote_node, 1, target.i_host_);
      for (int isn=0; isn<n_source; isn++) {
	int i_source_node = source.GetINode(isn);
	int i_remote_node = -1;
	for (std::vector<ExternalConnectionNode >::iterator it =
	       connect_mpi_->extern_connection_[i_source_node].begin();
	     it <  connect_mpi_->extern_connection_[i_source_node].end();
	     it++) {
	  if ((*it).target_host_id == target.i_host_) {
	    i_remote_node = (*it).remote_node_id;
	    break;
	  }
	}
	if (i_remote_node == -1) {
	  i_remote_node = i_new_remote_node;
	  i_new_remote_node++;
	  ExternalConnectionNode conn_node = {target.i_host_, i_remote_node};
	  connect_mpi_->extern_connection_[i_source_node].push_back(conn_node);
	}
	i_remote_node_arr[isn] = i_remote_node;
      }
      connect_mpi_->MPI_Send_int(&i_new_remote_node, 1, target.i_host_);
      connect_mpi_->MPI_Send_int(i_remote_node_arr, n_source, target.i_host_);
    }
    delete[] i_remote_node_arr;
  }
  MPI_Barrier( MPI_COMM_WORLD );

  return 0;
}

#endif

#endif
