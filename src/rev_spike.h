/*
 *  rev_spike.h
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


#ifndef REVSPIKE_H
#define REVSPIKE_H

extern unsigned int* d_RevSpikeNum;
extern unsigned int* d_RevSpikeTarget;
extern int* d_RevSpikeNConn;

__global__ void RevSpikeReset();

__global__ void RevSpikeBufferUpdate( unsigned int n_node );

__global__ void SynapseUpdateKernel( int n_rev_spikes, int* RevSpikeNConn );

int RevSpikeInit( NetConnection* net_connection );

int RevSpikeFree();

int ResetConnectionSpikeTimeDown( NetConnection* net_connection );

int ResetConnectionSpikeTimeUp( NetConnection* net_connection );

#endif
