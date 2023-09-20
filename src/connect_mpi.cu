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





#include <config.h>

#ifdef HAVE_MPI

#include <iostream>
#include <cmath>
#include <stdlib.h>

#include "connect_mpi.h"


int ConnectMpi::MPI_Recv_int(int *int_val, int n, int sender_id)
{
  MPI_Status Stat;
  int tag = 1;

  MPI_Recv(int_val, n, MPI_INT, sender_id, tag, MPI_COMM_WORLD, &Stat);

  return 0;
}

int ConnectMpi::MPI_Recv_float(float *float_val, int n, int sender_id)
{
  MPI_Status Stat;
  int tag = 1;

  MPI_Recv(float_val, n, MPI_FLOAT, sender_id, tag, MPI_COMM_WORLD, &Stat);

  return 0;
}

int ConnectMpi::MPI_Recv_uchar(unsigned char *uchar_val, int n, int sender_id)
{
  MPI_Status Stat;
  int tag = 1;

  MPI_Recv(uchar_val, n, MPI_UNSIGNED_CHAR, sender_id, tag, MPI_COMM_WORLD,
	   &Stat);

  return 0;
}

int ConnectMpi::MPI_Send_int(int *int_val, int n, int target_id)
{
  int tag = 1;

  MPI_Send(int_val, n, MPI_INT, target_id, tag, MPI_COMM_WORLD);

  return 0;
}

int ConnectMpi::MPI_Send_float(float *float_val, int n, int target_id)
{
  int tag = 1;

  MPI_Send(float_val, n, MPI_FLOAT, target_id, tag, MPI_COMM_WORLD);

  return 0;
}

int ConnectMpi::MPI_Send_uchar(unsigned char *uchar_val, int n, int target_id)
{
  int tag = 1;

  MPI_Send(uchar_val, n, MPI_UNSIGNED_CHAR, target_id, tag, MPI_COMM_WORLD);

  return 0;
}

int ConnectMpi::MpiInit(int argc, char *argv[])
{
  int initialized;
  MPI_Initialized(&initialized);
  if (!initialized) {
    MPI_Init(&argc,&argv);
  }
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_np_);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_id_);
  mpi_master_ = 0;
  
  return 0;
}

//bool ConnectMpi::ProcMaster()
//{
//  if (mpi_id_==mpi_master_) return true;
//  else return false;
//}

#endif
