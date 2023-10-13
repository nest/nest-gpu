/*
 *  spike_generator.h
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





#ifndef SPIKEGENERATOR_H
#define SPIKEGENERATOR_H

#include <iostream>
#include <string>
#include "cuda_error.h"
				    //#include "node_group.h"
#include "base_neuron.h"
#include "neuron_models.h"

/* BeginUserDocs: device, generator

Short description
+++++++++++++++++

Generate spikes from an array of spike-times

Description
+++++++++++

A spike generator can be used to generate spikes at specific times
which are given to the spike generator as an array.

The following parameters can be set in the status dictionary.

============== ======= =======================================
**Parameters:**
--------------------------------------------------------------
 spike_times   list    Times in which spikes are emitted
 spike_heights list    Weight of the spikes emitted
============== ======= =======================================

Spike times are given in milliseconds, and must be sorted with the
earliest spike first. All spike times must be strictly in the future
(i.e. greater than the current time step). 

It is possible that spike times do not coincide with a time step,
i.e., are not a multiple of the simulation resolution.
In that case, spike times will be rounded to the nearest
simulation steps (i.e. multiples of the resolution).

Sending multiple occurrences at same time step is not possible,
however the spike height can be regulated to be the equivalent of
having multiple spikes at once.
The spikes are thus delivered with the weight indicated by
the spike height multiplied with the weight of the connection.


See also
++++++++

poisson_generator

EndUserDocs
*/

class spike_generator : public BaseNeuron
{
  int *d_n_spikes_;
  int *d_i_spike_;	    
  int **d_spike_time_idx_;
  float **d_spike_height_;
  int **h_spike_time_idx_;
  float ** h_spike_height_;
  std::vector<std::vector<float> > spike_time_vect_;
  std::vector<std::vector<float> > spike_height_vect_;

  int SetSpikes(int irel_node, int n_spikes, float *spike_time,
		float *spike_height, float time_min, float time_resolution);
  
 public:
  ~spike_generator();
  
  int Init(int i_node_0, int n_node, int n_port, int i_group);

  int Free();
  
  int Update(long long i_time, double t1);

  int Calibrate(double time_min, float time_resolution);

  int SetArrayParam(int i_neuron, int n_neuron, std::string param_name,
		    float *array, int array_size);
  
  int SetArrayParam(int *i_neuron, int n_neuron, std::string param_name,
		    float *array, int array_size);
  
  int GetArrayParamSize(int i_neuron, std::string param_name);

  float *GetArrayParam(int i_neuron, std::string param_name);

};


#endif
