/*
 *  iaf_psc_exp.h
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

#ifndef IAFPSCEXP_H
#define IAFPSCEXP_H

#include "base_neuron.h"
#include "cuda_error.h"
#include "neuron_models.h"
#include "node_group.h"
#include <iostream>
#include <string>

/* BeginUserDocs: neuron, integrate-and-fire, current-based

Short description
+++++++++++++++++

Leaky integrate-and-fire neuron model with exponential PSCs

Description
+++++++++++

iaf_psc_exp is an implementation of a leaky integrate-and-fire model
with exponential shaped postsynaptic currents (PSCs) according to
equations 1, 2, 4 and 5 of [1]_ and equation 3 of [2]_.
Thus, postsynaptic currents have an infinitely short rise time.

The threshold crossing is followed by an absolute refractory period (t_ref)
during which the membrane potential is clamped to the resting potential
and spiking is prohibited.

The linear subthreshold dynamics are integrated by the Exact
Integration scheme [3]_. The neuron dynamics are solved on the time
grid given by the computational step size. Incoming as well as emitted
spikes are forced into that grid.

An additional state variable and the corresponding differential
equation represent a piecewise constant external current.

For conversion between postsynaptic potentials (PSPs) and PSCs,
please refer to the ``postsynaptic_potential_to_current`` function in
the ``helpers.py`` script of the Cortical Microcircuit model of [4]_.

Parameters
++++++++++

The following parameters can be set in the status dictionary.

============  =======  ========================================================
 V_m_rel       mV      Membrane potential in mV (relative to resting potential)
 I_syn_ex      pA      Excitatory synaptic current
 I_syn_in      pA      Inhibitory synaptic current
 tau_m         ms      Membrane time constant
 C_m           pF      Capacity of the membrane
 E_L           mV      Resting membrane potential
 I_e           pA      Constant input current
 Theta_rel     mV      Spike threshold in mV (relative to resting potential)
 V_reset_rel   mV      Reset membrane potential after a spike
 tau_ex        ms      Exponential decay time constant of excitatory synaptic
                       current kernel
 tau_in        ms      Exponential decay time constant of inhibitory synaptic
                       current kernel
 t_ref         ms      Duration of refractory period (V_m = V_reset)
 den_delay     ms      Dendritic delay
============  =======  ========================================================

References
++++++++++

.. [1] Burkitt A N (2006). A review of the integrate-and-fire neuron model:
       I. Homogeneous synaptic input. Biologial Cybernetics 95:1-19.
       DOI: https://doi.org/10.1007/s00422-006-0068-6
.. [2] Hanuschkin A, Kunkel S, Helias M, Morrison A, Diesmann M (2010).
       A general and efficient methof for incorporating precise spike
       times in globally time-driven simulations. Frontiers in Neuroinformatics.
       DOI: https://doi.org/10.3389/fninf.2010.00113
.. [3] Rotter S,  Diesmann M (1999). Exact simulation of
       time-invariant linear systems with applications to neuronal
       modeling. Biologial Cybernetics 81:381-402.
       DOI: https://doi.org/10.1007/s004220050570
.. [4] Potjans TC. and Diesmann M. 2014. The cell-type specific cortical
       microcircuit: relating structure and activity in a full-scale spiking
       network model. Cerebral Cortex. 24(3):785–806.
       DOI: https://doi.org/10.1093/cercor/bhs358.

See also
++++++++

iaf_psc_exp_g

EndUserDocs */

namespace iaf_psc_exp_ns
{
enum ScalVarIndexes
{
  i_I_syn_ex = 0,    // postsynaptic current for exc. inputs
  i_I_syn_in,        // postsynaptic current for inh. inputs
  i_V_m_rel,         // membrane potential
  i_refractory_step, // refractory step counter
  N_SCAL_VAR
};

enum ScalParamIndexes
{
  i_tau_m = 0,   // Membrane time constant in ms
  i_C_m,         // Membrane capacitance in pF
  i_E_L,         // Resting potential in mV
  i_I_e,         // External current in pA
  i_Theta_rel,   // Threshold, RELATIVE TO RESTING POTENTAIL(!)
                 // i.e. the real threshold is (E_L_+Theta_rel_)
  i_V_reset_rel, // relative reset value of the membrane potential
  i_tau_ex,      // Time constant of excitatory synaptic current in ms
  i_tau_in,      // Time constant of inhibitory synaptic current in ms
  // i_rho,          // Stochastic firing intensity at threshold in 1/s
  // i_delta,        // Width of threshold region in mV
  i_t_ref,     // Refractory period in ms
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

const std::string iaf_psc_exp_scal_var_name[ N_SCAL_VAR ] = { "I_syn_ex", "I_syn_in", "V_m_rel", "refractory_step" };

const std::string iaf_psc_exp_scal_param_name[ N_SCAL_PARAM ] = { "tau_m",
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
  "P22" };

} // namespace iaf_psc_exp_ns

class iaf_psc_exp : public BaseNeuron
{
public:
  ~iaf_psc_exp();

  int Init( int i_node_0, int n_neuron, int n_port, int i_group );

  int Calibrate( double, float time_resolution );

  int Update( long long it, double t1 );

  int Free();
};

#endif
