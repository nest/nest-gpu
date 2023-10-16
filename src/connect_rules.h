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

extern bool ConnectionSpikeTimeFlag;


template <class T1, class T2>
int NESTGPU::_Connect(curandGenerator_t &gen, T1 source, int n_source,
		      T2 target, int n_target,
		      ConnSpec &conn_spec, SynSpec &syn_spec)
{
  CheckUncalibrated("Connections cannot be created after calibration");
  ////////////////////////
    //TEMPORARY, TO BE IMPROVED
  if (syn_spec.syn_group_>=1) {
    ConnectionSpikeTimeFlag = true;
    rev_conn_flag_ = true;
  }

  switch (conn_spec.rule_) {
  case ONE_TO_ONE:
    if (n_source != n_target) {
      throw ngpu_exception("Number of source and target nodes must be equal "
			   "for the one-to-one connection rule");
    }
    return _ConnectOneToOne<T1, T2>(gen, source, target, n_source, syn_spec);
    break;

  case ALL_TO_ALL:
    return _ConnectAllToAll<T1, T2>(gen, source, n_source, target, n_target,
				    syn_spec);
    break;
  case FIXED_TOTAL_NUMBER:
    return _ConnectFixedTotalNumber<T1, T2>(gen, source, n_source,
					    target, n_target,
					    conn_spec.total_num_, syn_spec);
    break;
  case FIXED_INDEGREE:
    return _ConnectFixedIndegree<T1, T2>(gen, source, n_source,
					 target, n_target,
					 conn_spec.indegree_, syn_spec);
    break;
  case FIXED_OUTDEGREE:
    return _ConnectFixedOutdegree<T1, T2>(gen, source, n_source,
					  target, n_target,
					  conn_spec.outdegree_, syn_spec);
    break;
  default:
    throw ngpu_exception("Unknown connection rule");
  }
  return 0;
}

template
int NESTGPU::_Connect<int, int>(curandGenerator_t &gen,
				int source, int n_source,
				int target, int n_target,
				ConnSpec &conn_spec, SynSpec &syn_spec);

template
int NESTGPU::_Connect<int, int*>(curandGenerator_t &gen,
				 int source, int n_source,
				 int *target, int n_target,
				 ConnSpec &conn_spec, SynSpec &syn_spec);

template
int NESTGPU::_Connect<int*, int>(curandGenerator_t &gen,
				 int *source, int n_source,
				 int target, int n_target,
				 ConnSpec &conn_spec, SynSpec &syn_spec);

template
int NESTGPU::_Connect<int*, int*>(curandGenerator_t &gen,
				  int *source, int n_source,
				  int *target, int n_target,
				  ConnSpec &conn_spec, SynSpec &syn_spec);

template <class T1, class T2>
int NESTGPU::_Connect(T1 source, int n_source,
		      T2 target, int n_target,
		      ConnSpec &conn_spec, SynSpec &syn_spec)
{
  return _Connect(conn_random_generator_[this_host_][this_host_],
		  source, n_source, target, n_target, conn_spec, syn_spec);
}

template
int NESTGPU::_Connect<int, int>(int source, int n_source,
				int target, int n_target,
				ConnSpec &conn_spec, SynSpec &syn_spec);

template
int NESTGPU::_Connect<int, int*>(int source, int n_source,
				 int *target, int n_target,
				 ConnSpec &conn_spec, SynSpec &syn_spec);

template
int NESTGPU::_Connect<int*, int>(int *source, int n_source,
				 int target, int n_target,
				 ConnSpec &conn_spec, SynSpec &syn_spec);

template
int NESTGPU::_Connect<int*, int*>(int *source, int n_source,
				  int *target, int n_target,
				  ConnSpec &conn_spec, SynSpec &syn_spec);





#endif
