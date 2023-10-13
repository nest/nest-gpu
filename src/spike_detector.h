/*
 *  spike_detector.h
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





#ifndef SPIKEDETECTOR_H
#define SPIKEDETECTOR_H

#include <iostream>
#include <string>
//#include "node_group.h"
#include "base_neuron.h"

/* BeginUserDocs: device, recorder, spike

Short description
+++++++++++++++++

Collecting spikes from neurons

Description
+++++++++++

The ``spike_detector`` collects and records all spikes it receives
from neurons that are connected to it.

Any neuron from which spikes have to be recorded must be connected to
the spike recorder using the standard ``Connect`` command. 

.. warning::

  Differently from NEST, the connection ``weights`` and ``delays`` are
  taken into account by the spike recorder. This device will be modified
  in the future in order to have a device more similar to the one of NEST.

To record and retrieve the spikes emitted by a neuron, a new recorder has
to be created using the command ``CreateRecord``. For more details about
the continuous recording of variables see the :doc:`multimeter <multimeter>`
documentation.

Here follows an example:

::

  neuron = nestgpu.Create("aeif_cond_beta", 3)
  spike_det = nestgpu.Create("spike_detector")
  nestgpu.Connect([neuron[0]], spike_det, {"rule": "one_to_one"}, {"weight": 1.0, "delay": 1.0, "receptor":0})

  recorder = nestgpu.CreateRecord("", ["spike_height"], [spike_det[0]], [0])

  nestgpu.Simulate()
   
  recorded_data = nestgpu.GetRecordData(record)
  time = [row[0] for row in recorded_data]
  spike_height = [row[1] for row in recorded_data]

The output is thus a continuous variable, which is 0 when no spikes are emitted
by the neuron, and is ``weights`` when a spike is emitted. 

.. note::

  A faster implementation for spike recording, which is also similar to 
  the one of NEST in terms of output, is described in the guide of
  :doc:`how to record spikes <../guides/how_to_record_spikes>`.


See also
++++++++

multimeter

EndUserDocs */

class spike_detector : public BaseNeuron
{
 public:
  ~spike_detector();

  int Init(int i_node_0, int n_node, int n_port, int i_group);

  int Free();
  
  int Update(long long it, double t1);

};


#endif
