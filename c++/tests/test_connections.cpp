/*
 *  test_connections.cpp
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
#include <stdlib.h>
#include <iostream>
#include <string>
#include <algorithm>
#include "nestgpu.h"

using namespace std;

int main(int argc, char *argv[])
{
  // Intializes C random number generator
  // srand((unsigned) time(&t));

  NESTGPU ngpu;
  cout << "Building ...\n";

  ngpu.SetRandomSeed(1234ULL); // seed for GPU random numbers
  
  // poisson generator parameters
  float poiss_rate = 5000.0; // poisson signal rate in Hz
  float poiss_weight = 1.0;
  float poiss_delay = 0.2; // poisson signal delay in ms

  // create poisson generator
  NodeSeq pg = ngpu.Create("poisson_generator");
  ngpu.SetNeuronParam(pg, "rate", poiss_rate);

  int n_recept = 3; // number of receptors
  // create 3 neuron groups
  int n_neur1 = 100; // number of neurons
  int n_neur2 = 20;
  int n_neur3 = 50;
  int n_neurons = n_neur1 + n_neur2 + n_neur3;
  
  NodeSeq neur_group = ngpu.Create("aeif_cond_beta", n_neurons, n_recept);
  NodeSeq neur_group1 = neur_group.Subseq(0, n_neur1 - 1);
  NodeSeq neur_group2 = neur_group.Subseq(n_neur1, n_neur1 + n_neur2 - 1);
  NodeSeq neur_group3 = neur_group.Subseq(n_neur1 + n_neur2, n_neurons - 1);
  
  // neuron parameters
  float E_rev[] = {0.0, 0.0, 0.0};
  float tau_decay[] = {1.0, 1.0, 1.0};
  float tau_rise[] = {1.0, 1.0, 1.0};
  ngpu.SetNeuronParam(neur_group1, "E_rev", E_rev, 3);
  ngpu.SetNeuronParam(neur_group1, "tau_decay", tau_decay, 3);
  ngpu.SetNeuronParam(neur_group1, "tau_rise", tau_rise, 3);
  ngpu.SetNeuronParam(neur_group2, "E_rev", E_rev, 3);
  ngpu.SetNeuronParam(neur_group2, "tau_decay", tau_decay, 3);
  ngpu.SetNeuronParam(neur_group2, "tau_rise", tau_rise, 3);
  ngpu.SetNeuronParam(neur_group3, "E_rev", E_rev, 3);
  ngpu.SetNeuronParam(neur_group3, "tau_decay", tau_decay, 3);
  ngpu.SetNeuronParam(neur_group3, "tau_rise", tau_rise, 3);

  int i11 = neur_group1[rand()%n_neur1];
  int i12 = neur_group2[rand()%n_neur2];
  int i13 = neur_group2[rand()%n_neur2];
  int i14 = neur_group3[rand()%n_neur3];

  int i21 = neur_group2[rand()%n_neur2];

  int i31 = neur_group1[rand()%n_neur1];
  int i32 = neur_group3[rand()%n_neur3];

  int it1 = neur_group1[rand()%n_neur1];
  int it2 = neur_group2[rand()%n_neur2];
  int it3 = neur_group3[rand()%n_neur3];
  
  // connect poisson generator to port 0 of all neurons
  ngpu.Connect(pg[0], i11, 0, 0, poiss_weight, poiss_delay);
  ngpu.Connect(pg[0], i12, 0, 0, poiss_weight, poiss_delay);
  ngpu.Connect(pg[0], i13, 0, 0, poiss_weight, poiss_delay);
  ngpu.Connect(pg[0], i14, 0, 0, poiss_weight, poiss_delay);
  ngpu.Connect(pg[0], i21, 0, 0, poiss_weight, poiss_delay);
  ngpu.Connect(pg[0], i31, 0, 0, poiss_weight, poiss_delay);
  ngpu.Connect(pg[0], i32, 0, 0, poiss_weight, poiss_delay);

  float weight = 0.01; // connection weight
  float delay = 0.2; // connection delay in ms

  // connect neurons to target neuron n. 1
  ngpu.Connect(i11, it1, 0, 0, weight, delay);
  ngpu.Connect(i12, it1, 1, 0, weight, delay);
  ngpu.Connect(i13, it1, 1, 0, weight, delay);
  ngpu.Connect(i14, it1, 2, 0, weight, delay);

  // connect neuron to target neuron n. 2
  ngpu.Connect(i21, it2, 0, 0, weight, delay);

    // connect neurons to target neuron n. 3
  ngpu.Connect(i31, it3, 0, 0, weight, delay);
  ngpu.Connect(i32, it3, 1, 0, weight, delay);
  
  // create multimeter record n.1
  string filename1 = "test_connections_voltage.dat";
  int i_neuron_arr1[] = {i11, i12, i13, i14, i21, i31, i32, it1, it2, it3};
  std::string var_name_arr1[] = {"V_m", "V_m", "V_m", "V_m", "V_m", "V_m",
				"V_m", "V_m", "V_m", "V_m"};
  ngpu.CreateRecord(filename1, var_name_arr1, i_neuron_arr1, 10);

  // create multimeter record n.2
  string filename2 = "test_connections_g1.dat";
  int i_neuron_arr2[] = {it1, it1, it1, it2, it3, it3};
  int i_receptor_arr[] = {0, 1, 2, 0, 0, 1};
  std::string var_name_arr2[] = {"g1", "g1", "g1", "g1", "g1", "g1"};
  ngpu.CreateRecord(filename2, var_name_arr2, i_neuron_arr2,
		    i_receptor_arr, 6);

  // create multimeter record n.3
  string filename3 = "test_connections_spikes.dat";
  int i_neuron_arr3[] = {i11, i12, i13, i14, i21, i31, i32};
  std::string var_name_arr3[] = {"spike", "spike", "spike", "spike", "spike",
				 "spike", "spike"};
  ngpu.CreateRecord(filename3, var_name_arr3, i_neuron_arr3, 7);

  ngpu.Simulate();

  return 0;
}
