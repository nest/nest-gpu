/*
 *  aeif_psc_alpha.h
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





#ifndef AEIFPSCALPHA_H
#define AEIFPSCALPHA_H

#include <iostream>
#include <string>
#include "cuda_error.h"
#include "rk5.h"
#include "node_group.h"
#include "base_neuron.h"
#include "neuron_models.h"

/* BeginUserDocs: neuron, adaptive threshold, integrate-and-fire, current-based

Short description
+++++++++++++++++

Current-based exponential integrate-and-fire neuron model

Description
+++++++++++

``aeif_psc_alpha`` is the adaptive exponential integrate and fire neuron according
to [1]_. Synaptic currents are modeled as alpha functions.

This implementation uses the 5th order Runge-Kutta solver with
adaptive step size to integrate the differential equation.

The membrane potential is given by the following differential equation:

.. math::

  C_m \frac{dV}{dt} = -g_L(V-E_L) + g_L\Delta_T \exp\left(\frac{V-V_{th}}{\Delta_T}\right)
  + I_{syn\_ ex}(V, t) - I_{syn\_ in}(V, t) - w + I_e

where `I_syn_ex` and `I_syn_in` are the excitatory and inhibitory synaptic currents
modeled as alpha functions.

The differential equation for the spike-adaptation current `w` is:

.. math::

 \tau_w dw/dt= a(V-E_L) - w

.. note::

  Although this model is not multisynapse, the port (excitatory or inhibitory)
  to be chosen must be specified using the synapse property ``receptor``.
  The excitatory port has index 0, whereas the inhibitory one has index 1. Differently from
  NEST, the connection weights related to the inhibitory port must be positive.

Parameters
++++++++++

The following parameters can be set in the status dictionary.

========== ======= =======================================
**Dynamic state variables:**
----------------------------------------------------------
 V_m       mV      Membrane potential
 I_syn_ex  pA      Excitatory synaptic current
 I_syn_in  pA      Inhibitory synaptic current
 w         pA      Spike-adaptation current
========== ======= =======================================

========== ======= =======================================
**Membrane Parameters**
----------------------------------------------------------
 V_th      mV      Spike initiation threshold
 Delta_T   mV      Slope factor
 g_L       nS      Leak conductance
 E_L       mV      Leak reversal potential
 C_m       pF      Capacity of the membrane
 I_e       pA      Constant external input current
 V_peak    mV      Spike detection threshold
 V_reset   mV      Reset value for V_m after a spike
 t_ref     ms      Duration of refractory period
 den_delay ms      Dendritic delay
========== ======= =======================================

======== ======= ==================================
**Spike adaptation parameters**
---------------------------------------------------
 a       ns      Subthreshold adaptation
 b       pA      Spike-triggered adaptation
 tau_w   ms      Adaptation time constant
======== ======= ==================================

=========== ======= ===========================================================
**Synaptic parameters**
-------------------------------------------------------------------------------
 tau_syn_ex ms      Time constant of excitatory synaptic conductance
 tau_syn_ex ms      Time constant of inhibitory synaptic conductance
=========== ======= ===========================================================

============= ======= =========================================================
**Integration parameters**
-------------------------------------------------------------------------------
h0_rel        real    Starting step in ODE integration relative to time 
                      resolution
h_min_rel     real    Minimum step in ODE integration relative to time 
                      resolution
============= ======= =========================================================

References
++++++++++

.. [1] Brette R and Gerstner W (2005). Adaptive Exponential
       Integrate-and-Fire Model as an Effective Description of Neuronal
       Activity. J Neurophysiol 94:3637-3642.
       DOI: https://doi.org/10.1152/jn.00686.2005

See also
++++++++

aeif_psc_alpha_multisynapse, iaf_psc_alpha, aeif_cond_alpha

EndUserDocs */

//#define MAX_PORT_NUM 20

struct aeif_psc_alpha_rk5
{
  int i_node_0_;
};

class aeif_psc_alpha : public BaseNeuron
{
 public:
  RungeKutta5<aeif_psc_alpha_rk5> rk5_;
  float h_min_;
  float h_;
  aeif_psc_alpha_rk5 rk5_data_struct_;
    
  int Init(int i_node_0, int n_neuron, int n_port, int i_group);
	   

  int Calibrate(double time_min, float time_resolution);
		
  int Update(long long it, double t1);
  
  int GetX(int i_neuron, int n_node, double *x) {
    return rk5_.GetX(i_neuron, n_node, x);
  }
  
  int GetY(int i_var, int i_neuron, int n_node, float *y) {
    return rk5_.GetY(i_var, i_neuron, n_node, y);
  }
  
  template<int N_PORT>
    int UpdateNR(long long it, double t1);

};

#endif
