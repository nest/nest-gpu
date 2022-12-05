/*
 *  stdp.h
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





#ifndef STDP_H
#define STDP_H

#include "syn_model.h"

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
