/*
 *  This file is part of NESTGPU.
 *
 *  Copyright (C) 2021 The NEST Initiative
 *
 *  NESTGPU is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NESTGPU is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NESTGPU.  If not, see <http://www.gnu.org/licenses/>.
 *
 */





#ifdef HAVE_MPI
#ifndef CONNECTMPI_H
#define CONNECTMPI_H
#include <vector>
#include <mpi.h>
#include "connect.h"

struct ExternalConnectionNode
{
  int target_host_id;
  int remote_node_id;
};

class ConnectMpi
{
 public:
  NetConnection *net_connection_;
  int mpi_id_;
  int mpi_np_;
  int mpi_master_;
  bool remote_spike_height_;
  
  double SendSpikeToRemote_MPI_time_;
  double RecvSpikeFromRemote_MPI_time_;
  double SendSpikeToRemote_CUDAcp_time_;
  double RecvSpikeFromRemote_CUDAcp_time_;
  double JoinSpike_time_;
  
  
  std::vector<std::vector<ExternalConnectionNode > > extern_connection_;

  int MPI_Recv_int(int *int_val, int n, int sender_id);
  
  int MPI_Recv_float(float *float_val, int n, int sender_id);

  int MPI_Recv_uchar(unsigned char *uchar_val, int n, int sender_id);
  
  int MPI_Send_int(int *int_val, int n, int target_id);
  
  int MPI_Send_float(float *float_val, int n, int target_id);

  int MPI_Send_uchar(unsigned char *uchar_val, int n, int target_id);

  /*
  int RemoteConnect(int i_source_host, int i_source_node,
		    int i_target_host, int i_target_node,
		    int port, unsigned char syn_group,
		    float weight, float delay);
  */
  int MpiInit(int argc, char *argv[]);
  
  bool ProcMaster();
  
  int ExternalSpikeInit(int n_node, int n_hosts, int max_spike_per_host);

  int SendSpikeToRemote(int n_hosts, int max_spike_per_host);

  int RecvSpikeFromRemote(int n_hosts, int max_spike_per_host);

  int CopySpikeFromRemote(int n_hosts, int max_spike_per_host,
			  int i_remote_node_0);

  int JoinSpikes(int n_hosts, int max_spike_per_host);

};

#endif
#endif
