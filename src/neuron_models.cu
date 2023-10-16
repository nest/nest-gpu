/*
 *  neuron_models.cu
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





#include <config.h>
#include <iostream>
#include <string>

#include "ngpu_exception.h"
#include "cuda_error.h"
#include "getRealTime.h"
#include "neuron_models.h"
#include "nestgpu.h"
#include "iaf_psc_exp.h"
#include "iaf_psc_exp_hc.h"
#include "iaf_psc_exp_g.h"
#include "iaf_psc_alpha.h"
#include "ext_neuron.h"
#include "aeif_cond_alpha.h"
#include "aeif_cond_beta.h"
#include "aeif_psc_alpha.h"
#include "aeif_psc_delta.h"
#include "aeif_psc_exp.h"
#include "aeif_cond_beta_multisynapse.h"
#include "aeif_cond_alpha_multisynapse.h"
#include "aeif_psc_alpha_multisynapse.h"
#include "aeif_psc_exp_multisynapse.h"
#include "poiss_gen.h"
#include "spike_generator.h"
#include "parrot_neuron.h"
#include "spike_detector.h"
#include "izhikevich_cond_beta.h"
#include "izhikevich.h"
#include "izhikevich_psc_exp_5s.h"
#include "izhikevich_psc_exp_2s.h"
#include "izhikevich_psc_exp.h"
#include "user_m1.h"
#include "user_m2.h"

NodeSeq NESTGPU::_Create(std::string model_name, int n_nodes /*=1*/,
			 int n_ports /*=1*/)
{
  if (!create_flag_) {
    create_flag_ = true;
    start_real_time_ = getRealTime();
  }
  CheckUncalibrated("Nodes cannot be created after calibration");
  if (n_nodes <= 0) {
    throw ngpu_exception("Number of nodes must be greater than zero.");
  }
  else if (n_ports < 0) {
    throw ngpu_exception("Number of ports must be >= zero.");
  }
  if (model_name == neuron_model_name[i_iaf_psc_exp_g_model]) {
    n_ports = 1;
    iaf_psc_exp_g *iaf_psc_exp_g_group = new iaf_psc_exp_g;
    node_vect_.push_back(iaf_psc_exp_g_group);
  }
  else if (model_name == neuron_model_name[i_iaf_psc_exp_hc_model]) {
    n_ports = 1;
    iaf_psc_exp_hc *iaf_psc_exp_hc_group = new iaf_psc_exp_hc;
    node_vect_.push_back(iaf_psc_exp_hc_group);
  }
  else if (model_name == neuron_model_name[i_iaf_psc_exp_model]) {
    n_ports = 2;
    iaf_psc_exp *iaf_psc_exp_group = new iaf_psc_exp;
    node_vect_.push_back(iaf_psc_exp_group);
  }
  else if (model_name == neuron_model_name[i_iaf_psc_alpha_model]) {
    n_ports = 2;
    iaf_psc_alpha *iaf_psc_alpha_group = new iaf_psc_alpha;
    node_vect_.push_back(iaf_psc_alpha_group);
  }
  else if (model_name == neuron_model_name[i_ext_neuron_model]) {
    ext_neuron *ext_neuron_group = new ext_neuron;
    node_vect_.push_back(ext_neuron_group);
  }
  else if (model_name == neuron_model_name[i_aeif_cond_alpha_model]) {
    n_ports = 2;
    aeif_cond_alpha *aeif_cond_alpha_group = new aeif_cond_alpha;
    node_vect_.push_back(aeif_cond_alpha_group);
  }
  else if (model_name == neuron_model_name[i_aeif_cond_beta_model]) {
    n_ports = 2;
    aeif_cond_beta *aeif_cond_beta_group = new aeif_cond_beta;
    node_vect_.push_back(aeif_cond_beta_group);
  }
  else if (model_name == neuron_model_name[i_aeif_psc_alpha_model]) {
    n_ports = 2;
    aeif_psc_alpha *aeif_psc_alpha_group = new aeif_psc_alpha;
    node_vect_.push_back(aeif_psc_alpha_group);
  }
  else if (model_name == neuron_model_name[i_aeif_psc_delta_model]) {
    n_ports = 1;
    aeif_psc_delta *aeif_psc_delta_group = new aeif_psc_delta;
    node_vect_.push_back(aeif_psc_delta_group);
  }
  else if (model_name == neuron_model_name[i_aeif_psc_exp_model]) {
    n_ports = 2;
    aeif_psc_exp *aeif_psc_exp_group = new aeif_psc_exp;
    node_vect_.push_back(aeif_psc_exp_group);
  }
  else if (model_name == neuron_model_name[i_aeif_cond_beta_multisynapse_model]) {
    aeif_cond_beta_multisynapse *aeif_cond_beta_multisynapse_group = new aeif_cond_beta_multisynapse;
    node_vect_.push_back(aeif_cond_beta_multisynapse_group);
  }
  else if (model_name == neuron_model_name[i_aeif_cond_alpha_multisynapse_model]) {
    aeif_cond_alpha_multisynapse *aeif_cond_alpha_multisynapse_group = new aeif_cond_alpha_multisynapse;
    node_vect_.push_back(aeif_cond_alpha_multisynapse_group);
  }
  else if (model_name == neuron_model_name[i_aeif_psc_exp_multisynapse_model]) {
    aeif_psc_exp_multisynapse *aeif_psc_exp_multisynapse_group = new aeif_psc_exp_multisynapse;
    node_vect_.push_back(aeif_psc_exp_multisynapse_group);
  }
  else if (model_name == neuron_model_name[i_aeif_psc_alpha_multisynapse_model]) {
    aeif_psc_alpha_multisynapse *aeif_psc_alpha_multisynapse_group = new aeif_psc_alpha_multisynapse;
    node_vect_.push_back(aeif_psc_alpha_multisynapse_group);
  }
  else if (model_name == neuron_model_name[i_user_m1_model]) {
    user_m1 *user_m1_group = new user_m1;
    node_vect_.push_back(user_m1_group);
  }
  else if (model_name == neuron_model_name[i_user_m2_model]) {
    user_m2 *user_m2_group = new user_m2;
    node_vect_.push_back(user_m2_group);
  }
  else if (model_name == neuron_model_name[i_poisson_generator_model]) {
    n_ports = 0;
    poiss_gen *poiss_gen_group = new poiss_gen;
    node_vect_.push_back(poiss_gen_group);
  }
  else if (model_name == neuron_model_name[i_spike_generator_model]) {
    n_ports = 0;
    spike_generator *spike_generator_group = new spike_generator;
    node_vect_.push_back(spike_generator_group);
  }
  else if (model_name == neuron_model_name[i_parrot_neuron_model]) {
    n_ports = 2;
    parrot_neuron *parrot_neuron_group = new parrot_neuron;
    node_vect_.push_back(parrot_neuron_group);
  }
  else if (model_name == neuron_model_name[i_spike_detector_model]) {
    n_ports = 1;
    spike_detector *spike_detector_group = new spike_detector;
    node_vect_.push_back(spike_detector_group);
  }
  else if (model_name == neuron_model_name[i_izhikevich_model]) {
    izhikevich *izhikevich_group = new izhikevich;
    node_vect_.push_back(izhikevich_group);
  }
  else if (model_name == neuron_model_name[i_izhikevich_cond_beta_model]) {
    izhikevich_cond_beta *izhikevich_cond_beta_group = new izhikevich_cond_beta;
    node_vect_.push_back(izhikevich_cond_beta_group);
  }
  else if (model_name == neuron_model_name[i_izhikevich_psc_exp_5s_model]) {
    izhikevich_psc_exp_5s *izhikevich_psc_exp_5s_group =
      new izhikevich_psc_exp_5s;
    node_vect_.push_back(izhikevich_psc_exp_5s_group);
  }
  else if (model_name == neuron_model_name[i_izhikevich_psc_exp_2s_model]) {
    izhikevich_psc_exp_2s *izhikevich_psc_exp_2s_group =
      new izhikevich_psc_exp_2s;
    node_vect_.push_back(izhikevich_psc_exp_2s_group);
  }
  else if (model_name == neuron_model_name[i_izhikevich_psc_exp_model]) {
    izhikevich_psc_exp *izhikevich_psc_exp_group = new izhikevich_psc_exp;
    node_vect_.push_back(izhikevich_psc_exp_group);
  }
  else {
    throw ngpu_exception(std::string("Unknown neuron model name: ")
			 + model_name);
  }
  return NodeSeq(CreateNodeGroup(n_nodes, n_ports), n_nodes);
}

NodeSeq NESTGPU::Create(std::string model_name, int n_nodes,
			int n_ports)
{
  for (int i_host=0; i_host<n_hosts_; i_host++) {
    n_remote_nodes_[i_host] += n_nodes;
  }
  
  return _Create(model_name, n_nodes, n_ports);
}
