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





#ifndef POISSGEN_H
#define POISSGEN_H

#include <iostream>
#include <string>
#include <curand.h>
#include <curand_kernel.h>
#include "cuda_error.h"
#include "node_group.h"
#include "base_neuron.h"
#include "neuron_models.h"

/*
const int N_POISS_GEN_SCAL_PARAM = 4;
const std::string poiss_gen_scal_param_name[] = {
  "rate",
  "origin"
  "start",
  "stop",
};
*/

/* BeginUserDocs: device, generator

Short description
+++++++++++++++++

Generate spikes with Poisson process statistics

Description
+++++++++++

The poisson_generator simulates a neuron that is firing with Poisson
statistics, i.e. exponentially distributed interspike intervals. It will
generate a `unique` spike train for each of it's targets. If you do not want
this behavior and need the same spike train for all targets, you have to use a
``parrot_neuron`` between the poisson generator and the targets.

Parameters
++++++++++

The following parameters can be set in the status dictionary.

======== ======= =======================================
 rate    Hz      Mean firing rate
 origin  ms      Reference time for start and stop
 start   ms      Activation time, relative to origin
 stop    ms      Deactivation time, relative to origin
======== ======= =======================================


EndUserDocs */

class poiss_gen : public BaseNeuron
{
  curandState *d_curand_state_;
 public:
  
  int Init(int i_node_0, int n_node, int n_port, int i_group,
	   unsigned long long *seed);

  int Calibrate(double, float);
		
  int Update(long long it, double t1);
  int SendDirectSpikes(double t, float time_step);

};


#endif
