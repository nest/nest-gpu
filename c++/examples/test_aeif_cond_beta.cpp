/*
 *  test_aeif_cond_beta.cpp
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





#include <stdio.h>
#include <iostream>
#include <string>
#include <algorithm>
#include "nestgpu.h"

using namespace std;

int main(int argc, char *argv[])
{
  NESTGPU ngpu;
  cout << "Building ...\n";
  
  srand(12345);
  int n_neurons = 10000;
  
  // create n_neurons neurons with 3 receptor ports
  NodeSeq neuron = ngpu.Create("aeif_cond_beta", n_neurons, 3);

  // neuron parameters
  float E_rev[] = {20.0, 0.0, -85.0};
  float tau_decay[] = {40.0, 20.0, 30.0};
  float tau_rise[] = {20.0, 10.0, 5.0};
  ngpu.SetNeuronParam(neuron, "E_rev", E_rev, 3);
  ngpu.SetNeuronParam(neuron, "tau_decay", tau_decay, 3);
  ngpu.SetNeuronParam(neuron, "tau_rise", tau_rise, 3);
  ngpu.SetNeuronParam(neuron, "a", 4.0);
  ngpu.SetNeuronParam(neuron, "b", 80.5);
  ngpu.SetNeuronParam(neuron, "E_L", -70.6);
  ngpu.SetNeuronParam(neuron, "g_L", 300.0);

  NodeSeq sg = ngpu.Create("spike_generator"); // create spike generator

  float spike_times[] = {10.0, 400.0};
  float spike_heights[] = {1.0, 0.5};
  int n_spikes = 2;
  // set spike times and height
  ngpu.SetNeuronParam(sg, "spike_times", spike_times, n_spikes);
  ngpu.SetNeuronParam(sg, "spike_heights", spike_heights, n_spikes);
  
  float delay[] = {1.0, 100.0, 130.0};
  float weight[] = {0.1, 0.2, 0.5};

  for (int i_port=0; i_port<3; i_port++) {
    ConnSpec conn_spec(ALL_TO_ALL);
    SynSpec syn_spec(STANDARD_SYNAPSE, weight[i_port], delay[i_port], i_port);
    ngpu.Connect(sg, neuron, conn_spec, syn_spec);
  }
  string filename = "test_aeif_cond_beta.dat";
  int i_neuron[] = {neuron[rand()%n_neurons]}; // any set of neuron indexes
  string var_name[] = {"V_m"};
  // create multimeter record of V_m
  ngpu.CreateRecord(filename, var_name, i_neuron, 1);

  ngpu.Simulate(800.0);

  return 0;
}
