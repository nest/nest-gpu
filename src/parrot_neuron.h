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


class parrot_neuron : public BaseNeuron
{
 public:
  ~parrot_neuron();

  int Init(int i_node_0, int n_node, int n_port, int i_group,
	   unsigned long long *seed);

  int Free();
  
  int Update(long long it, double t1);

};


#endif
