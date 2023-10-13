/*
 *  user_m2_iaf_psc_exp.h
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
// https://github.com/nest/nest-simulator/blob/master/models/user_m2.h


#ifndef USERM2IAFPSCEXP_H
#define USERM2IAFPSCEXP_H

#include <iostream>
#include <string>
#include "cuda_error.h"
#include "node_group.h"
#include "base_neuron.h"
#include "neuron_models.h"


namespace user_m2_ns
{
enum ScalVarIndexes {
  i_I_syn_ex = 0,        // postsynaptic current for exc. inputs
  i_I_syn_in,            // postsynaptic current for inh. inputs
  i_V_m_rel,                 // membrane potential
  i_refractory_step,     // refractory step counter
  N_SCAL_VAR
};

enum ScalParamIndexes {
  i_tau_m = 0,       // Membrane time constant in ms
  i_C_m,             // Membrane capacitance in pF
  i_E_L,             // Resting potential in mV
  i_I_e,             // External current in pA
  i_Theta_rel,       // Threshold, RELATIVE TO RESTING POTENTAIL(!)
                     // i.e. the real threshold is (E_L_+Theta_rel_)
  i_V_reset_rel,     // relative reset value of the membrane potential
  i_tau_ex,          // Time constant of excitatory synaptic current in ms
  i_tau_in,          // Time constant of inhibitory synaptic current in ms
  // i_rho,          // Stochastic firing intensity at threshold in 1/s
  // i_delta,        // Width of threshold region in mV
  i_t_ref,           // Refractory period in ms
  i_den_delay, // dendritic backpropagation delay
  // time evolution operator
  i_P20,
  i_P11ex,
  i_P11in,
  i_P21ex,
  i_P21in,
  i_P22,
  N_SCAL_PARAM
};

 
const std::string user_m2_scal_var_name[N_SCAL_VAR] = {
  "I_syn_ex",
  "I_syn_in",
  "V_m_rel",
  "refractory_step"
};


const std::string user_m2_scal_param_name[N_SCAL_PARAM] = {
  "tau_m",
  "C_m",
  "E_L",
  "I_e",
  "Theta_rel",
  "V_reset_rel",
  "tau_ex",
  "tau_in",
  // "rho",
  //"delta",
  "t_ref",
  "den_delay",
  "P20",
  "P11ex",
  "P11in",
  "P21ex",
  "P21in",
  "P22"
};

} // namespace
 
class user_m2 : public BaseNeuron
{
 public:
  ~user_m2();
  
  int Init(int i_node_0, int n_neuron, int n_port, int i_group);
	   

  int Calibrate(double, float time_resolution);
		
  int Update(long long it, double t1);

  int Free();

};


#endif
