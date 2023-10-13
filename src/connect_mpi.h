/*
 *  connect_mpi.h
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





#ifdef HAVE_MPI
#ifndef CONNECTMPI_H
#define CONNECTMPI_H
#include <vector>
#include <mpi.h>
#include "connect.h"

namespace ConnectMpi
{
  // public:
  //int mpi_id_;
  //int mpi_np_;
  //int mpi_master_;
  //bool remote_spike_height_;
  
  int MPI_Recv_int(int *int_val, int n, int sender_id);
  
  int MPI_Recv_float(float *float_val, int n, int sender_id);

  int MPI_Recv_uchar(unsigned char *uchar_val, int n, int sender_id);
  
  int MPI_Send_int(int *int_val, int n, int target_id);
  
  int MPI_Send_float(float *float_val, int n, int target_id);

  int MPI_Send_uchar(unsigned char *uchar_val, int n, int target_id);

  //int MpiInit(int argc, char *argv[]);
  
  bool ProcMaster();
  
};

#endif
#endif
