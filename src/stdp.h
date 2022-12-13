/*
 *  This file is part of NESTGPU.
 *
 *  Copyright (C) 2021 The NEST Initiative
 *
 *  NESTGPU is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NESTGPU is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NESTGPU.  If not, see <http://www.gnu.org/licenses/>.
 *
 */





#ifndef STDP_H
#define STDP_H

#include "syn_model.h"

/* BeginUserDocs: synapse, spike-timing-dependent plasticity

Short description
+++++++++++++++++

Synapse type for spike-timing dependent plasticity

Description
+++++++++++

The STDP class is a type of synapse model used to create
synapses that enable spike timing dependent plasticity
(as defined in [1]_). 
Here the weight dependence exponent can be set separately
for potentiation and depression.


Parameters
++++++++++

========== =======  ======================================================
 tau_plus  ms       Time constant of STDP window, potentiation
 tau_minus ms       Time constant of STDP window, depression
 lambda    real     Step size
 alpha     real     Asymmetry parameter (scales depression increments as
                    alpha*lambda)
 mu_plus   real     Weight dependence exponent, potentiation
 mu_minus  real     Weight dependence exponent, depression
 Wmax      real     Maximum allowed weight
========== =======  ======================================================


References
++++++++++

.. [1] Guetig et al. (2003). Learning input correlations through nonlinear
       temporally asymmetric hebbian plasticity. Journal of Neuroscience,
       23:3697-3714 DOI: https://doi.org/10.1523/JNEUROSCI.23-09-03697.2003


EndUserDocs */


class STDP : public SynModel
{
 public:
  STDP() {Init();}
  int Init();
};

namespace stdp_ns
{
  enum ParamIndexes {
    i_tau_plus = 0, i_tau_minus, i_lambda, i_alpha, i_mu_plus, i_mu_minus,
    i_Wmax, // i_den_delay,
    N_PARAM
  };

  const std::string stdp_param_name[N_PARAM] = {
    "tau_plus", "tau_minus", "lambda", "alpha", "mu_plus", "mu_minus", "Wmax"
    //, "den_delay"
  };

}

#endif
