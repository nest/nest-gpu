/*
 *  aeif_cond_alpha_multisynapse.h
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





#ifndef AEIFCONDALPHAMULTISYNAPSE_H
#define AEIFCONDALPHAMULTISYNAPSE_H

#include <iostream>
#include <string>
#include "cuda_error.h"
#include "rk5.h"
#include "node_group.h"
#include "base_neuron.h"
#include "neuron_models.h"


/* BeginUserDocs: neuron, integrate-and-fire, adaptive threshold, conductance-based

Short description
+++++++++++++++++

Conductance-based adaptive exponential integrate-and-fire neuron model

Description
+++++++++++

``aeif_cond_alpha_multisynapse`` is a conductance-based adaptive exponential 
integrate-and-fire neuron model according to [1]_ with multiple
synaptic time constants, and synaptic conductance modeled by an
alpha function.

This implementation uses the 5th order Runge-Kutta solver with
adaptive step size to integrate the differential equation.

It allows an arbitrary number of synaptic time constants. Synaptic
conductance is modeled by an alpha function, as described in [2]_.

The membrane potential is given by the following differential equation:

.. math::

  C_m \frac{dV}{dt} = -g_L(V-E_L) + g_L\Delta_T \exp\left(\frac{V-V_{th}}{\Delta_T}\right)
  + I_{syn_{tot}}(V, t)- w + I_e

where

.. math::

  I_{syn_{tot}}(V,t) = \sum_i g_i(t) (V - E_{rev,i}) ,

the synapse `i` is excitatory or inhibitory depending on the value of
:math:`E_{rev,i}` and the differential equation for the
spike-adaptation current `w` is

.. math::

  \tau_w dw/dt = a(V - E_L) - w

When the neuron fires a spike, the adaptation current :math:`w <- w + b`.

.. note::

  The number of receptor ports must be specified at neuron creation (default value is 1) and
  the receptor index starts from 0 (and not from 1 as in NEST multisynapse models).
  The time constants are supplied by an array, ``tau_syn``, and the pertaining
  synaptic reversal potentials are supplied by the array ``E_rev``. Port numbers
  are automatically assigned in the range 0 to ``n_receptors-1``.
  During connection, the ports are selected with the synapse property ``receptor``.

Parameters
++++++++++

The following parameters can be set in the status dictionary.

======== ======= =======================================
**Dynamic state variables:**
--------------------------------------------------------
 V_m     mV      Membrane potential
 w       pA      Spike-adaptation current
======== ======= =======================================

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

======== ============= ========================================================
**Synaptic parameters**
-------------------------------------------------------------------------------
E_rev    list of mV    Reversal potential
tau_syn  list of ms    Time constant of synaptic conductance
======== ============= ========================================================

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

.. [1] Brette R and Gerstner W (2005). Adaptive exponential
       integrate-and-fire model as an effective description of neuronal
       activity. Journal of Neurophysiology. 943637-3642
       DOI: https://doi.org/10.1152/jn.00686.2005

.. [2] A. Roth and M. C. W. van Rossum, Computational Modeling Methods
       for Neuroscientists, MIT Press 2013, Chapter 6.
       DOI: https://doi.org/10.7551/mitpress/9780262013277.003.0007

See also
+++++++

aeif_cond_beta_multisynapse

EndUserDocs */


#define MAX_PORT_NUM 20

struct aeif_cond_alpha_multisynapse_rk5
{
  int i_node_0_;
};

class aeif_cond_alpha_multisynapse : public BaseNeuron
{
 public:
  RungeKutta5<aeif_cond_alpha_multisynapse_rk5> rk5_;
  float h_min_;
  float h_;
  aeif_cond_alpha_multisynapse_rk5 rk5_data_struct_;
    
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
