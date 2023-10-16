/*
 *  user_m1_iaf_psc_exp_g.h
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





// adapted from:
// https://github.com/nest/nest-simulator/blob/master/models/iaf_psc_exp.h


#ifndef USERM1IAFPSCEXPG_H
#define USERM1IAFPSCEXPG_H

#include <iostream>
#include <string>
#include "cuda_error.h"
#include "node_group.h"
#include "base_neuron.h"
#include "neuron_models.h"


namespace user_m1_ns
{
enum ScalVarIndexes {
  i_I_syn = 0,        // postsynaptic current for exc. inputs
  i_V_m_rel,          // membrane potential relative to E_L
  i_refractory_step,  // refractory step counter
  N_SCAL_VAR
};

enum ScalParamIndexes {
  i_I_e = 0,         // External current in pA
  N_SCAL_PARAM
};

enum GroupParamIndexes {
  i_tau_m = 0,       // Membrane time constant in ms
  i_C_m,             // Membrane capacitance in pF
  i_E_L,             // Resting potential in mV
  i_Theta_rel,       // Threshold, RELATIVE TO RESTING POTENTIAL(!)
                     // i.e. the real threshold is (E_L_+Theta_rel_)
  i_V_reset_rel,     // relative reset value of the membrane potential
  i_tau_syn,         // Time constant of synaptic current in ms
  i_t_ref,           // Refractory period in ms
  N_GROUP_PARAM
};


 
const std::string user_m1_scal_var_name[N_SCAL_VAR] = {
  "I_syn",
  "V_m_rel",
  "refractory_step"
};

const std::string user_m1_scal_param_name[N_SCAL_PARAM] = {
  "I_e"
};

const std::string user_m1_group_param_name[N_GROUP_PARAM] = {
  "tau_m",
  "C_m",
  "E_L",
  "Theta_rel",
  "V_reset_rel",
  "tau_syn",
  "t_ref"
};
 
} // namespace
 



class user_m1 : public BaseNeuron
{
  float time_resolution_;

 public:
  ~user_m1();
  
  int Init(int i_node_0, int n_neuron, int n_port, int i_group);
	   
  int Calibrate(double /*time_min*/, float time_res) {
    time_resolution_ = time_res;
    return 0;
  }
  
  int Update(long long it, double t1);

  int Free();

};


#endif
