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





#ifndef SPIKEDETECTOR_H
#define SPIKEDETECTOR_H

#include <iostream>
#include <string>
//#include "node_group.h"
#include "base_neuron.h"


class spike_detector : public BaseNeuron
{
 public:
  ~spike_detector();

  int Init(int i_node_0, int n_node, int n_port, int i_group,
	   unsigned long long *seed);

  int Free();
  
  int Update(long long it, double t1);

};


#endif
