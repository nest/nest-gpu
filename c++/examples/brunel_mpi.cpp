/*
 *  brunel_mpi.cpp
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
  ngpu.ConnectMpiInit(argc, argv);
  int mpi_np = ngpu.MpiNp();
  if (argc != 2 || mpi_np != 2) {
    cout << "Usage: mpirun -np 2 " << argv[0] << " n_neurons\n";
    ngpu.MpiFinalize();
    return 0;
  }
  int arg1;
  sscanf(argv[1], "%d", &arg1);

  int mpi_id = ngpu.MpiId();
  cout << "Building on host " << mpi_id << " ..." <<endl;

  ngpu.SetRandomSeed(1234ULL + mpi_id); // seed for GPU random numbers
 
  //////////////////////////////////////////////////////////////////////
  // WRITE HERE COMMANDS THAT ARE EXECUTED ON ALL HOSTS
  //////////////////////////////////////////////////////////////////////
  int n_receptors = 2;

  float delay = 1.0;       // synaptic delay in ms

  int order = arg1/5;
  int NE = 4 * order;      // number of excitatory neurons
  int NI = 1 * order;      // number of inhibitory neurons
  int n_neurons = NE + NI; // number of neurons in total

  int CE = 800;  // number of excitatory synapses per neuron
  int CI = CE/4;  // number of inhibitory synapses per neuron

  float Wex = 0.05;
  float Win = 0.35;

  // each host has a poisson generator
  float poiss_rate = 20000.0; // poisson signal rate in Hz
  float poiss_weight = 0.37;
  float poiss_delay = 0.2; // poisson signal delay in ms

  // create poisson generator
  NodeSeq pg = ngpu.Create("poisson_generator");
  ngpu.SetNeuronParam(pg, "rate", poiss_rate);

  // each host has n_neurons neurons with n_receptor receptor ports
  NodeSeq neuron = ngpu.Create("aeif_cond_beta", n_neurons,
					   n_receptors);
  NodeSeq exc_neuron = neuron.Subseq(0,NE-1); // excitatory neuron group
  NodeSeq inh_neuron = neuron.Subseq(NE, n_neurons-1); //inhibitory neuron group
  
  // the following parameters are set to the same values on all hosts
  float E_rev[] = {0.0, -85.0};
  float tau_decay[] = {1.0, 1.0};
  float tau_rise[] = {1.0, 1.0};
  ngpu.SetNeuronParam(neuron, "E_rev", E_rev, 2);
  ngpu.SetNeuronParam(neuron, "tau_decay", tau_decay, 2);
  ngpu.SetNeuronParam(neuron, "tau_rise", tau_rise, 2);

  // Excitatory local connections, defined on all hosts
  // connect excitatory neurons to port 0 of all neurons
  // weight Wex and fixed indegree CE*3/4
  ConnSpec conn_spec1(FIXED_INDEGREE, CE*3/4);
  SynSpec syn_spec1;
  syn_spec1.SetParam("receptor", 0);
  syn_spec1.SetParam("weight", Wex);
  syn_spec1.SetParam("delay", delay);
  ngpu.Connect(exc_neuron, neuron, conn_spec1, syn_spec1);

  // Inhibitory local connections, defined on all hosts
  // connect inhibitory neurons to port 1 of all neurons
  // weight Win and fixed indegree CI*3/4
  ConnSpec conn_spec2(FIXED_INDEGREE, CI*3/4);
  SynSpec syn_spec2;
  syn_spec2.SetParam("receptor", 1);
  syn_spec2.SetParam("weight", Win);
  syn_spec2.SetParam("delay", delay);
  ngpu.Connect(inh_neuron, neuron, conn_spec2, syn_spec2);

  ConnSpec conn_spec3(ALL_TO_ALL);
  SynSpec syn_spec3(STANDARD_SYNAPSE, poiss_weight, poiss_delay, 0);
  // connect poisson generator to port 0 of all neurons
  ngpu.Connect(pg, neuron, conn_spec3, syn_spec3);


  char filename[100];
  sprintf(filename, "test_brunel_mpi_%d.dat", mpi_id);

  int i_neuron_arr[] = {neuron[0], neuron[rand()%n_neurons],
		     neuron[n_neurons-1]}; // any set of neuron indexes
  // create multimeter record of V_m
  std::string var_name_arr[] = {"V_m", "V_m", "V_m"};
  ngpu.CreateRecord(string(filename), var_name_arr, i_neuron_arr, 3);

  //////////////////////////////////////////////////////////////////////
  // WRITE HERE REMOTE CONNECTIONS
  //////////////////////////////////////////////////////////////////////

  // Excitatory remote connections
  // connect excitatory neurons to port 0 of all neurons
  // weight Wex and fixed indegree CE/4
  // host 0 to host 1
  ConnSpec conn_spec4(FIXED_INDEGREE, CE/4);
  SynSpec syn_spec4;
  syn_spec4.SetParam("receptor", 0);
  syn_spec4.SetParam("weight", Wex);
  syn_spec4.SetParam("delay", delay);
  // host 0 to host 1
  ngpu.RemoteConnect(0, exc_neuron, 1, neuron, conn_spec4, syn_spec4);
  // host 1 to host 0
  ngpu.RemoteConnect(1, exc_neuron, 0, neuron, conn_spec4, syn_spec4);

  // Inhibitory remote connections
  // connect inhibitory neurons to port 1 of all neurons
  // weight Win and fixed indegree CI/4
  // host 0 to host 1
  ConnSpec conn_spec5(FIXED_INDEGREE, CI/4);
  SynSpec syn_spec5;
  syn_spec5.SetParam("receptor", 1);
  syn_spec5.SetParam("weight", Win);
  syn_spec5.SetParam("delay", delay);
  // host 0 to host 1
  ngpu.RemoteConnect(0, inh_neuron, 1, neuron, conn_spec5, syn_spec5);
  // host 1 to host 0
  ngpu.RemoteConnect(1, inh_neuron, 0, neuron, conn_spec5, syn_spec5);

  ngpu.Simulate();

  ngpu.MpiFinalize();

  return 0;
}
