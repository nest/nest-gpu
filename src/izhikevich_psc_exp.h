/*
 *  izhikevich_psc_exp.h
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






#ifndef IZHIKEVICHPSCEXP_H
#define IZHIKEVICHPSCEXP_H

#include <iostream>
#include <string>
#include "cuda_error.h"
#include "node_group.h"
#include "base_neuron.h"
#include "neuron_models.h"


/* BeginUserDocs: neuron, integrate-and-fire

Short description
+++++++++++++++++

Izhikevich neuron model with exponential postsynaptic currents

Description
+++++++++++

Implementation of the simple spiking neuron model introduced by Izhikevich
[1]_, with postsynaptic currents in the form of truncated exponentials.
The dynamics are given by:

.. math::

  \frac{dV_m}{dt} &= 0.04 V_m^2 + 5 V_m + 140 - u + I \\
  \frac{du}{dt} &= a (b V_m - u)


.. math::

  &\text{if}\;\;\; V_m \geq V_{th}:\\
  &\;\;\;\; V_m \text{ is set to } c\\
  &\;\;\;\; u \text{ is incremented by } d\\
  & \, \\
  &v \text{ jumps on each spike arrival by the weight of the spike}

This implementation uses the standard technique for forward Euler integration.

Parameters
++++++++++

The following parameters can be set in the status dictionary.

======================= =======  ==============================================
 V_m                    mV       Membrane potential
 I_syn                  pA       Synaptic current
 u                      mV       Membrane potential recovery variable
 V_th                   mV       Spike threshold
 a                      real     Describes time scale of recovery variable
 b                      real     Sensitivity of recovery variable
 c                      mV       After-spike reset value of V_m
 d                      mV       After-spike reset value of u
 I_e                    pA       Constant input current
 t_ref                  ms       Refractory time
 tau_syn                ms       Time constant of synaptic current
 den_delay              ms       Dendritic delay
======================= =======  ==============================================

References
++++++++++

.. [1] Izhikevich EM (2003). Simple model of spiking neurons. IEEE Transactions
       on Neural Networks, 14:1569-1572. DOI: https://doi.org/10.1109/TNN.2003.820440


EndUserDocs */


namespace izhikevich_psc_exp_ns
{
enum ScalVarIndexes {
  i_I_syn = 0,        // postsynaptic current for exc. inputs
  i_V_m,              // membrane potential
  i_u,
  i_refractory_step,  // refractory step counter
  N_SCAL_VAR
};

enum ScalParamIndexes {
  i_I_e = 0,         // External current in pA
  i_den_delay,
  N_SCAL_PARAM
};

enum GroupParamIndexes {
  i_V_th = 0,
  i_a,
  i_b,
  i_c,
  i_d,
  i_tau_syn,         // Time constant of synaptic current in ms
  i_t_ref,           // Refractory period in ms
  N_GROUP_PARAM
};


 
const std::string izhikevich_psc_exp_scal_var_name[N_SCAL_VAR] = {
  "I_syn",
  "V_m",
  "u",
  "refractory_step"
};

const std::string izhikevich_psc_exp_scal_param_name[N_SCAL_PARAM] = {
  "I_e",
  "den_delay"
};

const std::string izhikevich_psc_exp_group_param_name[N_GROUP_PARAM] = {
  "V_th",
  "a",
  "b",
  "c",
  "d",
  "tau_syn",
  "t_ref"
};
 
} // namespace
 



class izhikevich_psc_exp : public BaseNeuron
{
  float time_resolution_;

 public:
  ~izhikevich_psc_exp();
  
  int Init(int i_node_0, int n_neuron, int n_port, int i_group);
	   
  int Calibrate(double /*time_min*/, float time_res) {
    time_resolution_ = time_res;
    return 0;
  }
  
  int Update(long long it, double t1);

  int Free();

};


#endif
