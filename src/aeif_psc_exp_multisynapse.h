/*
 *  aeif_psc_exp_multisynapse.h
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

#ifndef AEIFPSCEXPMULTISYNAPSE_H
#define AEIFPSCEXPMULTISYNAPSE_H

#include "base_neuron.h"
#include "cuda_error.h"
#include "neuron_models.h"
#include "node_group.h"
#include "rk5.h"
#include <iostream>
#include <string>

/* BeginUserDocs: neuron, integrate-and-fire, adaptive threshold, current-based

Short description
+++++++++++++++++

Current-based exponential integrate-and-fire neuron model

Description
+++++++++++

aeif_psc_exp_multisynapse is the adaptive exponential integrate and fire neuron
according to [1]_, with postsynaptic currents in the form of
truncated exponentials.

This implementation uses the embedded 5th order Runge-Kutta
solver with adaptive stepsize to integrate the differential equation.

The membrane potential is given by the following differential equation:

.. math::

 C_m \frac{dV}{dt} = -g_L(V-E_L) + g_L\Delta_T
\exp\left(\frac{V-V_{th}}{\Delta_T}\right)
    + I_{syn}(t)- w + I_e

where ``I_syn (t)`` is the sum of the synaptic currents modeled as truncated
exponentials with time constant ``tau_syn``.

The differential equation for the spike-adaptation current `w` is:

.. math::

 \tau_w dw/dt= a(V-E_L) -w

.. note::

  The number of receptor ports must be specified at neuron creation (default
value is 1) and the receptor index starts from 0 (and not from 1 as in NEST
multisynapse models). The time constants are supplied by an array, ``tau_syn``.
Port numbers are automatically assigned in the range 0 to ``n_receptors-1``.
  During connection, the ports are selected with the synapse property
``receptor``.

Parameters
++++++++++

The following parameters can be set in the status dictionary.

======== ======= =======================================
**Dynamic state variables:**
--------------------------------------------------------
 V_m     mV      Membrane potential
 I_syn   pA      Synaptic current
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

=========== ======= ===========================================================
**Synaptic parameters**
-------------------------------------------------------------------------------
 tau_syn    ms      Exponential decay time constant of synaptic
                    current
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

iaf_psc_exp

EndUserDocs */

#define MAX_PORT_NUM 20

struct aeif_psc_exp_multisynapse_rk5
{
  int i_node_0_;
};

class aeif_psc_exp_multisynapse : public BaseNeuron
{
public:
  RungeKutta5< aeif_psc_exp_multisynapse_rk5 > rk5_;
  float h_min_;
  float h_;
  aeif_psc_exp_multisynapse_rk5 rk5_data_struct_;

  int Init( int i_node_0, int n_neuron, int n_port, int i_group );

  int Calibrate( double time_min, float time_resolution );

  int Update( long long it, double t1 );

  int
  GetX( int i_neuron, int n_node, double* x )
  {
    return rk5_.GetX( i_neuron, n_node, x );
  }

  int
  GetY( int i_var, int i_neuron, int n_node, float* y )
  {
    return rk5_.GetY( i_var, i_neuron, n_node, y );
  }

  template < int N_PORT >
  int UpdateNR( long long it, double t1 );
};

#endif
