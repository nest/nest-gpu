/*
 *  test_constcurr.cpp
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
  
  // create n_neurons neurons with 1 receptor ports
  NodeSeq neuron = ngpu.Create("aeif_cond_beta", n_neurons, 1);

  // neuron parameters
  ngpu.SetNeuronParam(neuron, "a", 4.0);
  ngpu.SetNeuronParam(neuron, "b",  80.5);
  ngpu.SetNeuronParam(neuron, "E_L", -70.6);
  ngpu.SetNeuronParam(neuron, "I_e", 800.0);

  string filename = "test_constcurr.dat";
  int i_neurons[] = {neuron[rand()%n_neurons]}; // any set of neuron indexes
  string var_name[] = {"V_m"};

  // create multimeter record of V_m
  ngpu.CreateRecord(filename, var_name, i_neurons, 1);

  ngpu.Simulate();

  return 0;
}
