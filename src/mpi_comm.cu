/*
 *  mpi_comm.cu
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

#include <config.h>

#include <list>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_error.h"
#include "getRealTime.h"
#include "nestgpu.h"

#include "mpi_comm.h"
#include "remote_connect.h"
#include "remote_spike.h"

#ifdef HAVE_MPI
#include <mpi.h>
MPI_Request *recv_mpi_request;
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// bit packing compression/decompression  used with MPI send/receive taken from
// https://stackoverflow.com/questions/49462207/how-to-compress-a-32-bit-array-elements-into-minimum-required-bit-elements
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define MASK32 ((uint64_t)0xffffffff)

void bitPackWrite(uint32_t *arr, int bits, int i, int value) {
  int bitoffset = i * bits;
  int index = bitoffset / 32;
  int shift = bitoffset % 32;
  uint64_t maskbits = (~(uint64_t)0) >> (64 - bits);
  uint64_t twoval = ((uint64_t)arr[index + 1] << 32) + arr[index];
  twoval = twoval & ~(maskbits << shift) | ((value & maskbits) << shift);
  arr[index] = (twoval & MASK32);
  arr[index + 1] = (twoval >> 32) & MASK32;
}

int bitPackRead(const uint32_t *arr, int bits, int i) {
  int bitoffset = i * bits;
  int index = bitoffset / 32;
  int shift = bitoffset % 32;
  uint64_t maskbits = (~(uint64_t)0) >> (64 - bits);
  int value =
      ((((uint64_t)arr[index + 1] << 32) + arr[index]) >> shift) & maskbits;
  return (value);
}

void bitPackInPlace(uint32_t *arr, int size, int *packed_size_pt, int bits) {
  *packed_size_pt = ((size - 1) * bits) / 32 + 2;
  for (int i = 0; i < size; i++) {
    bitPackWrite(arr, bits, i, arr[i]);
  }
}

int bitPackedSize(int size, int bits) {
  if (size == 0) {
    return 0;
  } else {
    return ((size - 1) * bits) / 32 + 2;
  }
}

void bitPackInPlace(uint32_t *arr, int size, int bits) {
  for (int i = 0; i < size; i++) {
    bitPackWrite(arr, bits, i, arr[i]);
  }
}

void bitUnpackInPlace(uint32_t *arr, int size, int bits) {
  for (int i = size - 1; i >= 0; i--) {
    arr[i] = bitPackRead(arr, bits, i);
  }
}

// Send spikes to remote MPI processes
int NESTGPU::SendSpikeToRemote(int n_ext_spikes) {
#ifdef HAVE_MPI
  int mpi_id, tag = 1; // id is already in the class, can be removed
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_id);

  double time_mark = getRealTime();

  // get point-to-point MPI communication activation matrix
  std::vector<std::vector<bool>> &p2p_host_conn_matrix =
      conn_->getP2PHostConnMatrix();

  gpuErrchk(cudaMemcpy(&h_ExternalTargetSpikeNum[0], d_ExternalTargetSpikeNum,
                       n_hosts_ * sizeof(int), cudaMemcpyDeviceToHost));
  SendSpikeToRemote_CUDAcp_time_ += (getRealTime() - time_mark);

  time_mark = getRealTime();
  int n_spike_tot = 0;
  // copy spikes from GPU to CPU memory
  if (n_ext_spikes > 0) {
    gpuErrchk(cudaMemcpy(&n_spike_tot, d_ExternalTargetSpikeIdx0 + n_hosts_,
                         sizeof(int), cudaMemcpyDeviceToHost));
    if (n_spike_tot >= max_remote_spike_num_) {
      throw ngpu_exception(
          std::string("Number of spikes to be sent remotely ") +
          std::to_string(n_spike_tot) + " larger than limit " +
          std::to_string(max_remote_spike_num_));
    }

    gpuErrchk(cudaMemcpy(&h_ExternalTargetSpikeNodeId[0],
                         d_ExternalTargetSpikeNodeId, n_spike_tot * sizeof(int),
                         cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&h_ExternalTargetSpikeIdx0[0],
                         d_ExternalTargetSpikeIdx0,
                         (n_hosts_ + 1) * sizeof(int), cudaMemcpyDeviceToHost));
  } else {
    for (int i = 0; i < n_hosts_ + 1; i++) {
      h_ExternalTargetSpikeIdx0[i] = 0;
    }
  }

  // prepare array for sending spikes to host groups through MPI communicators
  int n_hg_spike_tot = 0;
  // copy spikes from GPU to CPU memory
  if (n_ext_spikes > 0) {
    gpuErrchk(cudaMemcpy(&h_ExternalHostGroupSpikeIdx0[0],
                         d_ExternalHostGroupSpikeIdx0,
                         (conn_->getHostGroup().size() + 1) * sizeof(uint),
                         cudaMemcpyDeviceToHost));
    n_hg_spike_tot = h_ExternalHostGroupSpikeIdx0[conn_->getHostGroup().size()];

    if (n_hg_spike_tot > 0) {
      if (n_hg_spike_tot >= max_remote_spike_num_) {
        throw ngpu_exception(
            std::string(
                "Number of spikes to be sent remotely to host groups ") +
            std::to_string(n_hg_spike_tot) + " larger than limit " +
            std::to_string(max_remote_spike_num_));
      }
      gpuErrchk(cudaMemcpy(
          &h_ExternalHostGroupSpikeNodeId[0], d_ExternalHostGroupSpikeNodeId,
          n_hg_spike_tot * sizeof(int), cudaMemcpyDeviceToHost));
    }
  } else {
    for (uint i = 0; i < conn_->getHostGroup().size() + 1; i++) {
      h_ExternalHostGroupSpikeIdx0[i] = 0;
    }
  }

  SendSpikeToRemote_CUDAcp_time_ += (getRealTime() - time_mark);
  time_mark = getRealTime();

  // loop on remote MPI proc
  for (int ih = 0; ih < n_hosts_; ih++) {
    if (ih == mpi_id || p2p_host_conn_matrix[this_host_][ih] ==
                            false) { // skip self MPI proc and unused
                                     // point-to-point MPI communications
      recv_mpi_request[n_hosts_ + ih] = MPI_REQUEST_NULL;
      continue;
    }
    // get index and size of spike packet that must be sent to MPI proc ih
    // array_idx is the first index of the packet for host ih
    int array_idx = h_ExternalTargetSpikeIdx0[ih];
    int n_spikes = h_ExternalTargetSpikeIdx0[ih + 1] - array_idx;
    // nonblocking sent of spike packet to MPI proc ih
    if (n_spikes >= max_spike_per_host_) {
      throw ngpu_exception(
          std::string("MPI_Isend error from host ") +
          std::to_string(this_host_) + " to host " + std::to_string(ih) +
          "\nNumber of spikes to be sent remotely " + std::to_string(n_spikes) +
          " larger than limit " + std::to_string(max_spike_per_host_) +
          "\nYou can try to increase the kernel parameter "
          "\"max_spike_per_host_fact_\".");
    }

    MPI_Isend(&h_ExternalTargetSpikeNodeId[array_idx], n_spikes, MPI_UNSIGNED,
              ih, tag, MPI_COMM_WORLD, &recv_mpi_request[n_hosts_ + ih]);

    // printf("MPI_Send nspikes (src,tgt,nspike): "
    //	   "%d %d %d\n", mpi_id, ih, n_spikes);
    // printf("MPI_Send 1st-neuron-idx (src,tgt,idx): "
    //	   "%d %d %d\n", mpi_id, ih,
    //	   h_ExternalTargetSpikeNodeId[array_idx]);
  }
  SendSpikeToRemote_comm_time_ += (getRealTime() - time_mark);

  return 0;
#else
  throw ngpu_exception("MPI is not available in your build");
#endif
}

// Receive spikes from remote MPI processes
int NESTGPU::RecvSpikeFromRemote()

{
#ifdef HAVE_MPI
  int mpi_id, tag = 1; // id is already in the class, can be removed
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_id);

  double time_mark = getRealTime();

  // get point-to-point MPI communication activation matrix
  std::vector<std::vector<bool>> &p2p_host_conn_matrix =
      conn_->getP2PHostConnMatrix();

  // loop on remote MPI proc
  for (int i_host = 0; i_host < n_hosts_; i_host++) {
    if (i_host == mpi_id || p2p_host_conn_matrix[i_host][this_host_] == false) {
      recv_mpi_request[i_host] = MPI_REQUEST_NULL;
      continue;
    }
    // start nonblocking MPI receive from MPI proc i_host
    MPI_Irecv(&h_ExternalSourceSpikeNodeId[0][i_host * max_spike_per_host_],
              max_spike_per_host_, MPI_UNSIGNED, i_host, tag, MPI_COMM_WORLD,
              &recv_mpi_request[i_host]);
  }
  MPI_Status statuses[2 * n_hosts_];
  // recv_mpi_request[ mpi_id ] = MPI_REQUEST_NULL;
  // MPI_Waitall( n_hosts_ + nhg - 1, recv_mpi_request, statuses );
  MPI_Waitall(2 * n_hosts_, recv_mpi_request, statuses);

  std::vector<std::vector<int>> &host_group = conn_->getHostGroup();
  std::vector<MPI_Comm> &mpi_comm_vect = conn_->getMPIComm();
  uint nhg = host_group.size();
  std::vector<int> &host_group_local_id = conn_->getHostGroupLocalId();
  std::vector<std::vector<int>> &bit_pack_nbits = conn_->getBitPackNbits();
  std::vector<int> &bit_pack_nbits_this_host = conn_->getBitPackNbitsThisHost();

  for (uint abs_ihg = 0; abs_ihg < host_group_local_id.size(); abs_ihg++) {
    int ihg = host_group_local_id[abs_ihg];
    if (ihg < 0) {
      continue;
    }
    int idx0 = h_ExternalHostGroupSpikeIdx0[ihg]; // position of subarray of
                                                  // spikes that must be sent to
                                                  // host group ihg
    uint *sendbuf = &h_ExternalHostGroupSpikeNodeId[idx0]; // send address
    int sendcount = h_ExternalHostGroupSpikeNum[ihg];      // send count

    uint *recvbuf =
        &h_ExternalSourceSpikeNodeId[ihg][0]; //[ i_host * max_spike_per_host_ ]
                                              //// receiving buffers
    int *recvcounts = &h_ExternalSourceSpikeNum[ihg][0];
    int *displs =
        &h_ExternalSourceSpikeDispl[0]; // displacememnts of receiving buffers,
                                        // all equal to max_spike_per_host_

    MPI_Allgather(&sendcount, 1, MPI_INT, recvcounts, 1, MPI_INT,
                  mpi_comm_vect[ihg - 1]);

    if (mpi_bitpack_) {
      int sendcount_packed = 0;
      if (sendcount > 0) {
        double time_mark = getRealTime();
        bitPackInPlace(sendbuf, sendcount, bit_pack_nbits_this_host[ihg]);
        MpiBitPack_time_ += (getRealTime() - time_mark);
        sendcount_packed =
            bitPackedSize(sendcount, bit_pack_nbits_this_host[ihg]);
      }
      uint nh = host_group[ihg].size(); // number of hosts in the group
      // loop on hosts
      for (uint gi_host = 0; gi_host < nh; gi_host++) {
        int i_host = host_group[ihg][gi_host];
        // if (i_host != this_host_) {
        h_ExternalSourceSpikeNumBitPacked[ihg][gi_host] =
            bitPackedSize(h_ExternalSourceSpikeNum[ihg][gi_host],
                          bit_pack_nbits[ihg][gi_host]);
        //}
      }
      int *recvcounts_packed = &h_ExternalSourceSpikeNumBitPacked[ihg][0];

      MPI_Allgatherv(sendbuf, sendcount_packed, MPI_INT, recvbuf,
                     recvcounts_packed, displs, MPI_INT,
                     mpi_comm_vect[ihg - 1]);

      SpikeNumAllgather_send_ += sendcount;
      SpikeNumAllgather_send_packed_ += sendcount_packed;
      double time_mark = getRealTime();
      // loop on hosts
      for (uint gi_host = 0; gi_host < nh; gi_host++) {
        int i_host = host_group[ihg][gi_host];
        if (i_host != this_host_) {
          SpikeNumAllgather_recv_packed_ += recvcounts_packed[i_host];
          SpikeNumAllgather_recv_ += recvcounts[i_host];
          uint *recvbuf_of_host =
              &h_ExternalSourceSpikeNodeId[ihg][gi_host * max_spike_per_host_];
          int count = h_ExternalSourceSpikeNum[ihg][gi_host];
          bitUnpackInPlace(recvbuf_of_host, count,
                           bit_pack_nbits[ihg][gi_host]);
        }
      }
      MpiBitUnpack_time_ += (getRealTime() - time_mark);
    } else {
      MPI_Allgatherv(sendbuf, sendcount, MPI_INT, recvbuf, recvcounts, displs,
                     MPI_INT, mpi_comm_vect[ihg - 1]);
    }
  }

  /*
  MPI_Status statuses[ n_hosts_ + nhg - 1];
  recv_mpi_request[ mpi_id ] = MPI_REQUEST_NULL;
  //MPI_Waitall( n_hosts_ + nhg - 1, recv_mpi_request, statuses );
  MPI_Waitall( n_hosts_, recv_mpi_request, statuses );
  */
  for (int i_host = 0; i_host < n_hosts_; i_host++) {
    if ((int)i_host == mpi_id) {
      h_ExternalSourceSpikeNum[0][i_host] = 0;
      continue;
    }
    int count = 0;
    if (p2p_host_conn_matrix[i_host][this_host_] == true) {
      MPI_Get_count(&statuses[i_host], MPI_UNSIGNED, &count);
    }
    if (count < 0 || count > max_spike_per_host_) {
      throw ngpu_exception(
          std::string("MPI_Irecv error in host ") + std::to_string(this_host_) +
          " from host " + std::to_string(i_host) +
          "\nNumber of spikes sent remotely larger than limit " +
          std::to_string(max_spike_per_host_) +
          "\nYou can try to increase the kernel parameter "
          "\"max_spike_per_host_fact_\".");
    }
    h_ExternalSourceSpikeNum[0][i_host] = count;
  }

  // Maybe the barrier is not necessary?
  // MPI_Barrier( MPI_COMM_WORLD );
  RecvSpikeFromRemote_comm_time_ += (getRealTime() - time_mark);

  return 0;
#else
  throw ngpu_exception("MPI is not available in your build");
#endif
}

int NESTGPU::ConnectMpiInit(int argc, char **argv) {
#ifdef HAVE_MPI
  CheckUncalibrated("MPI connections cannot be initialized after calibration");
  int initialized;
  MPI_Initialized(&initialized);
  if (!initialized) {
    int provided_thread_level;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_thread_level);
  }
  int n_hosts;
  int this_host;
  MPI_Comm_size(MPI_COMM_WORLD, &n_hosts);
  MPI_Comm_rank(MPI_COMM_WORLD, &this_host);
  mpi_flag_ = true;
  setNHosts(n_hosts);
  setThisHost(this_host);
  // conn_->remoteConnectionMapInit();
  recv_mpi_request = new MPI_Request[2 * n_hosts_];
  return 0;
#else
  throw ngpu_exception("MPI is not available in your build");
#endif
}

int NESTGPU::FakeConnectMpiInit(int n_hosts, int this_host) {
  setNHosts(n_hosts);
  setThisHost(this_host);

  return 0;
}

int NESTGPU::MpiFinalize() {
#ifdef HAVE_MPI
  if (mpi_flag_) {
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized) {
      MPI_Finalize();
    }
  }

  return 0;
#else
  throw ngpu_exception("MPI is not available in your build");
#endif
}
