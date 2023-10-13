/*
 *  parrot_neuron.h
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





#ifndef PARROTNEURON_H
#define PARROTNEURON_H

#include <iostream>
#include <string>
//#include "node_group.h"
#include "base_neuron.h"
//#include "neuron_models.h"


/* BeginUserDocs: neuron, parrot

Short description
+++++++++++++++++

Neuron that repeats incoming spikes

Description
+++++++++++

The parrot neuron simply emits one spike for every incoming spike.
An important application is to provide identical poisson spike
trains to a group of neurons. The ``poisson_generator`` sends a different
spike train to each of its target neurons. By connecting one
``poisson_generator`` to a ``parrot_neuron`` and then that ``parrot_neuron`` to
a group of neurons, all target neurons will receive the same poisson
spike train.

Remarks
.......

- Weights of connections *to* the ``parrot_neuron`` are ignored.
- Weights on connections *from* the ``parrot_neuron`` are handled as usual.
- Delays are honored on incoming and outgoing connections.

Only spikes arriving on connections to port (``receptor``) 0 will 
be repeated. Connections onto port 1 will be accepted, but spikes
incoming through port 1 will be ignored. This allows setting
exact pre- and postsynaptic spike times for STDP protocols by 
connecting two parrot neurons spiking at desired times by, e.g.,
a `stdp` onto port 1 on the postsynaptic parrot neuron.


EndUserDocs */


class parrot_neuron : public BaseNeuron
{
 public:
  ~parrot_neuron();

  int Init(int i_node_0, int n_node, int n_port, int i_group);

  int Free();
  
  int Update(long long it, double t1);

};


#endif
