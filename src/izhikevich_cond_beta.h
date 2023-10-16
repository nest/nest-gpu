/*
 *  izhikevich_cond_beta.h
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





#ifndef IZHIKEVICHCONDBETA_H
#define IZHIKEVICHCONDBETA_H

#include <iostream>
#include <string>
#include "cuda_error.h"
#include "rk5.h"
#include "node_group.h"
#include "base_neuron.h"
#include "neuron_models.h"

/* BeginUserDocs: neuron, integrate-and-fire

Short description
+++++++++++++++++

Conductance-based Izhikevich neuron model

Description
+++++++++++

Implementation of the simple spiking neuron model introduced by Izhikevich
[1]_ with synaptic conductance modeled by a beta function, as described in [2]_.
The dynamics are given by:

.. math::

  \frac{dV_m}{dt} &= 0.04 V_m^2 + 5 V_m + 140 - u + I \\
  \frac{du}{dt} &= a (b V_m - u))


.. math::

   &\text{if}\;\;\; V_m \geq V_{th}:\\
   &\;\;\;\; V_m \text{ is set to } c\\
   &\;\;\;\; u \text{ is incremented by } d\\
   & \, \\
   &v \text{ jumps on each spike arrival by the weight of the spike}

This implementation uses the standard technique for forward Euler integration.
This model is multisynapse, so it allows an arbitrary number of synaptic 
rise time and decay time constants. The number of receptor ports must be specified 
at neuron creation (default value is 1) and the receptor index starts from 0 
(and not from 1 as in NEST multisynapse models).
The time constants are supplied by by two arrays, ``tau_rise`` and ``tau_decay`` for
the synaptic rise time and decay time, respectively. The synaptic
reversal potentials are supplied by the array ``E_rev``. Port numbers
are automatically assigned in the range from 0 to ``n_receptors-1``.
During connection, the ports are selected with the synapse property ``receptor``.

Parameters
++++++++++

The following parameters can be set in the status dictionary.

======================= =======  ==============================================
 V_m                    mV       Membrane potential
 u                      mV       Membrane potential recovery variable
 V_th                   mV       Spike threshold
 a                      real     Describes time scale of recovery variable
 b                      real     Sensitivity of recovery variable
 c                      mV       After-spike reset value of V_m
 d                      mV       After-spike reset value of u
 I_e                    pA       Constant input current
 t_ref                  ms       Refractory time
 den_delay              ms       Dendritic delay
 E_rev                  mV       Leak reversal potential
 tau_rise               ms       Rise time constant of synaptic conductance
 tau_decay              ms       Decay time constant of synaptic conductance
 h_min_rel              real     Starting step in ODE integration relative to
                                 time resolution
 h0_rel                 real     Minimum step in ODE integration relative to 
                                 time resolution
======================= =======  ==============================================

References
++++++++++

.. [1] Izhikevich EM (2003). Simple model of spiking neurons. IEEE Transactions
       on Neural Networks, 14:1569-1572. DOI: https://doi.org/10.1109/TNN.2003.820440

.. [2] A. Roth and M. C. W. van Rossum, Computational Modeling Methods
       for Neuroscientists, MIT Press 2013, Chapter 6.
       DOI: https://doi.org/10.7551/mitpress/9780262013277.003.0007


See also
++++++++

izhikevich, aeif_conf_beta

EndUserDocs */

#define MAX_PORT_NUM 20

struct izhikevich_cond_beta_rk5
{
  int i_node_0_;
};

class izhikevich_cond_beta : public BaseNeuron
{
 public:
  RungeKutta5<izhikevich_cond_beta_rk5> rk5_;
  float h_min_;
  float h_;
  izhikevich_cond_beta_rk5 rk5_data_struct_;
    
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
