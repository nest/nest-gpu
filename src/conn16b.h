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

#ifndef CONN16B_H
#define CONN16B_H
#include "connect.h"
#include "remote_connect.h"

struct conn16b_struct
{
  uint target_port_syn;
  float weight;
};

//typedef uint conn16b_key;
typedef uint64_t conn16b_key;

template <>
__device__ __forceinline__ void
setConnDelay<conn16b_key> 
(conn16b_key &conn_key, int delay) {
  conn_key = (conn_key & (~DelayMask)) | delay;
}

template <>
__device__ __forceinline__ void
setConnSource<conn16b_key> 
(conn16b_key &conn_key, inode_t source) {
  conn_key = (conn_key & (~SourceMask)) | (source << MaxDelayNBits);
}

template <>
__device__ __forceinline__ void
setConnTarget<conn16b_struct> 
(conn16b_struct &conn, inode_t target) {
  conn.target_port_syn = (conn.target_port_syn & (~TargetMask))
    | (target << MaxPortSynNBits);
}

template <>
__device__ __forceinline__ void
setConnPort<conn16b_key, conn16b_struct> 
(conn16b_key &conn_key, conn16b_struct &conn, int port) {
  conn.target_port_syn = (conn.target_port_syn & (~PortMask))
    | (port << (MaxSynNBits + 1));
}

template <>
__device__ __forceinline__ void
setConnSyn<conn16b_key, conn16b_struct> 
(conn16b_key &conn_key, conn16b_struct &conn, int syn) {
  conn.target_port_syn = (conn.target_port_syn & (~SynMask))
    | syn;
}

template <>
__device__ __forceinline__ int
getConnDelay<conn16b_key> 
(const conn16b_key &conn_key) {
  return conn_key & DelayMask;
}

template <>
__device__ __forceinline__ inode_t
getConnSource<conn16b_key> 
(conn16b_key &conn_key) {
  return (conn_key & SourceMask) >> MaxDelayNBits;
}

template <>
__device__ __forceinline__ inode_t
getConnTarget<conn16b_struct> 
(conn16b_struct &conn) {
  return (conn.target_port_syn & TargetMask) >>  MaxPortSynNBits;
}

template <>
__device__ __forceinline__ int
getConnPort<conn16b_key, conn16b_struct> 
(conn16b_key &conn_key, conn16b_struct &conn) {
  return (conn.target_port_syn & PortMask) >> (MaxSynNBits + 1);
}

template <>
__device__ __forceinline__ int
getConnSyn<conn16b_key, conn16b_struct> 
(conn16b_key &conn_key, conn16b_struct &conn) {
  return conn.target_port_syn & SynMask;
}

// TEMPORARY TO BE IMPROVED!!!!
template <>
__device__ __forceinline__ bool
getConnRemoteFlag<conn16b_key, conn16b_struct> 
(conn16b_key &conn_key, conn16b_struct &conn) {
  return (conn.target_port_syn >> MaxSynNBits) & (uint)1;
}

template <>
__device__ __forceinline__ void
clearConnRemoteFlag<conn16b_key, conn16b_struct> 
(conn16b_key &conn_key, conn16b_struct &conn) {
  conn.target_port_syn = conn.target_port_syn &
    ~((uint)1 << MaxSynNBits);
}

template<>
int ConnectionTemplate<conn16b_key, conn16b_struct>::setMaxNodeNBits
(int max_node_nbits);

template<>
int ConnectionTemplate<conn16b_key, conn16b_struct>::setMaxSynNBits
(int max_syn_nbits);

#endif
