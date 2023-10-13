/*
 *  ext_neuron.h
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





#ifndef EXTNEURON_H
#define EXTNEURON_H

#include <iostream>
#include <string>
#include "cuda_error.h"
#include "node_group.h"
#include "base_neuron.h"
#include "neuron_models.h"


namespace ext_neuron_ns
{
enum ScalVarIndexes {
  N_SCAL_VAR = 0
};

enum PortVarIndexes {
  i_port_input = 0,
  i_port_value,
  N_PORT_VAR
};

enum ScalParamIndexes {
  i_den_delay = 0,
  N_SCAL_PARAM
};

enum PortParamIndexes {
  i_port_weight = 0,
  N_PORT_PARAM
};

//const std::string *ext_neuron_scal_var_name[N_SCAL_VAR] = {};

const std::string ext_neuron_port_var_name[N_PORT_VAR] = {
  "port_input", "port_value"
};

const std::string ext_neuron_scal_param_name[N_SCAL_PARAM] = {
  "den_delay"
};

const std::string ext_neuron_port_param_name[N_PORT_PARAM] = {
  "port_weight"
};

}

class ext_neuron : public BaseNeuron
{
 public:
  ~ext_neuron();
  int Init(int i_node_0, int n_neuron, int n_port, int i_group);
	   

  //int Calibrate(double time_min, float time_resolution);
		
  int Update(long long it, double t1);

  int Free();

  float *GetExtNeuronInputSpikes(int *n_node, int *n_port);
  
};


#endif
