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





#ifndef GETSPIKE_H
#define GETSPIKE_H

__global__ void GetSpikes(double *spike_array, int array_size, int n_port,
			  int n_var,
			  float *port_weight_arr,
			  int port_weight_arr_step,
			  int port_weight_port_step, //float *y_arr);
			  float *port_input_arr,
			  int port_input_arr_step,
			  int port_input_port_step);


__global__ void CollectSpikeKernel(int n_spikes, int *SpikeTargetNum);

#endif
