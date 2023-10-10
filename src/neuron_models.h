/*
 *  neuron_models.h
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





#ifndef NEURONMODELS_H
#define NEURONMODELS_H

enum NeuronModels {
  i_null_model = 0,
  i_iaf_psc_exp_g_model,  
  i_iaf_psc_exp_hc_model,
  i_iaf_psc_exp_model,
  i_iaf_psc_alpha_model,
  i_ext_neuron_model,
  i_aeif_cond_alpha_model,
  i_aeif_cond_beta_model,
  i_aeif_psc_alpha_model,
  i_aeif_psc_delta_model,
  i_aeif_psc_exp_model,
  i_aeif_cond_beta_multisynapse_model,
  i_aeif_cond_alpha_multisynapse_model,
  i_aeif_psc_exp_multisynapse_model,
  i_aeif_psc_alpha_multisynapse_model,
  i_poisson_generator_model,
  i_spike_generator_model,
  i_parrot_neuron_model,
  i_spike_detector_model,
  i_izhikevich_cond_beta_model,
  i_izhikevich_model,
  i_izhikevich_psc_exp_5s_model,
  i_izhikevich_psc_exp_2s_model,
  i_izhikevich_psc_exp_model,
  i_user_m1_model,
  i_user_m2_model,
  N_NEURON_MODELS
};

const std::string neuron_model_name[N_NEURON_MODELS] = {
  "",
  "iaf_psc_exp_g",
  "iaf_psc_exp_hc",
  "iaf_psc_exp",
  "iaf_psc_alpha",
  "ext_neuron",
  "aeif_cond_alpha",
  "aeif_cond_beta",
  "aeif_psc_alpha",
  "aeif_psc_delta",
  "aeif_psc_exp",
  "aeif_cond_beta_multisynapse",
  "aeif_cond_alpha_multisynapse",
  "aeif_psc_exp_multisynapse",
  "aeif_psc_alpha_multisynapse",
  "poisson_generator",
  "spike_generator",
  "parrot_neuron",
  "spike_detector",
  "izhikevich_cond_beta",
  "izhikevich",
  "izhikevich_psc_exp_5s",
  "izhikevich_psc_exp_2s",
  "izhikevich_psc_exp",
  "user_m1",
  "user_m2"
};

#endif
