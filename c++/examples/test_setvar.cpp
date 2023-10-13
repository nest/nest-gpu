/*
 *  test_setvar.cpp
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
  int n_neurons = 3;
  
  // create n_neurons neurons with 2 receptor ports
  NodeSeq neuron = ngpu.Create("aeif_cond_beta", n_neurons, 2);
  float tau_decay[] = {60.0, 10.0};
  float tau_rise[] = {40.0, 5.0};
  ngpu.SetNeuronParam(neuron, "tau_decay", tau_decay, 2);
  ngpu.SetNeuronParam(neuron, "tau_rise", tau_rise, 2);
  
  NodeSeq neuron0 = neuron.Subseq(0,0);
  NodeSeq neuron1 = neuron.Subseq(1,1);
  NodeSeq neuron2 = neuron.Subseq(2,2);
  float g11[] = {0.0, 0.1};
  float g12[] = {0.1, 0.0};
  
  // neuron variables
  ngpu.SetNeuronVar(neuron0, "V_m", -80.0);
  ngpu.SetNeuronVar(neuron1, "g1", g11, 2);
  ngpu.SetNeuronVar(neuron2, "g1", g12, 2);

  // reading parameters and variables test
  float *read_td = ngpu.GetNeuronParam(neuron, "tau_decay");
  float *read_tr = ngpu.GetNeuronParam(neuron, "tau_rise");
  float *read_Vm = ngpu.GetNeuronVar(neuron, "V_m");
  float *read_Vth = ngpu.GetNeuronParam(neuron, "V_th");
  float *read_g1 = ngpu.GetNeuronVar(neuron, "g1");

  for (int in=0; in<3; in++) {
    printf("Neuron n. %d\n", in);
    printf("\tV_m: %f\n", read_Vm[in]);
    printf("\tV_th: %f\n", read_Vth[in]); 
    for (int ip=0; ip<2; ip++) {
      printf("\tg1: %f\n", read_g1[in*2+ip]);
      printf("\ttau_rise: %f\n", read_tr[in*2+ip]);
      printf("\ttau_decay: %f\n", read_td[in*2+ip]); 
    }
    printf("\n");
  }

  string filename = "test_setvar.dat";
  int i_neurons[] = {neuron[0], neuron[1], neuron[2]};
  string var_name[] = {"V_m", "V_m", "V_m"};

  // create multimeter record of V_m
  ngpu.CreateRecord(filename, var_name, i_neurons, 3);

  ngpu.Simulate();

  return 0;
}
