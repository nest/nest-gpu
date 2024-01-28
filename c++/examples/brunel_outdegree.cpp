/*
 *  brunel_outdegree.cpp
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

#include "nestgpu.h"
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <string>

using namespace std;

int
main( int argc, char* argv[] )
{
  if ( argc != 2 )
  {
    cout << "Usage: " << argv[ 0 ] << " n_neurons\n";
    return 0;
  }
  int arg1;
  sscanf( argv[ 1 ], "%d", &arg1 );
  NESTGPU ngpu;
  cout << "Building ...\n";

  ngpu.SetRandomSeed( 12345ULL ); // seed for GPU random numbers

  int n_receptors = 2;

  int order = arg1 / 5;
  int NE = 4 * order;      // number of excitatory neurons
  int NI = 1 * order;      // number of inhibitory neurons
  int n_neurons = NE + NI; // number of neurons in total

  int CPN = 1000; // number of output connections per neuron

  float Wex = 0.05;
  float Win = 0.35;

  // poisson generator parameters
  float poiss_rate = 20000.0; // poisson signal rate in Hz
  float poiss_weight = 0.37;
  float poiss_delay = 0.2; // poisson signal delay in ms

  // create poisson generator
  NodeSeq pg = ngpu.Create( "poisson_generator" );
  ngpu.SetNeuronParam( pg, "rate", poiss_rate );

  // create n_neurons neurons with n_receptor receptor ports
  NodeSeq neuron = ngpu.Create( "aeif_cond_beta", n_neurons, n_receptors );
  NodeSeq exc_neuron = neuron.Subseq( 0, NE - 1 );         // excitatory neuron group
  NodeSeq inh_neuron = neuron.Subseq( NE, n_neurons - 1 ); // inhibitory neuron group

  // neuron parameters
  float E_rev[] = { 0.0, -85.0 };
  float tau_decay[] = { 1.0, 1.0 };
  float tau_rise[] = { 1.0, 1.0 };
  ngpu.SetNeuronParam( neuron, "E_rev", E_rev, 2 );
  ngpu.SetNeuronParam( neuron, "tau_decay", tau_decay, 2 );
  ngpu.SetNeuronParam( neuron, "tau_rise", tau_rise, 2 );

  float mean_delay = 0.5;
  float std_delay = 0.25;
  float min_delay = 0.1;
  // Excitatory connections
  // connect excitatory neurons to port 0 of all neurons
  // normally distributed delays, weight Wex and CPN connections per neuron
  float* exc_delays =
    ngpu.RandomNormalClipped( CPN * NE, mean_delay, std_delay, min_delay, mean_delay + 3 * std_delay );

  ConnSpec conn_spec1( FIXED_OUTDEGREE, CPN );
  SynSpec syn_spec1;
  syn_spec1.SetParam( "receptor", 0 );
  syn_spec1.SetParam( "weight", Wex );
  syn_spec1.SetParam( "delay_array", exc_delays );
  ngpu.Connect( exc_neuron, neuron, conn_spec1, syn_spec1 );
  delete[] exc_delays;

  // Inhibitory connections
  // connect inhibitory neurons to port 1 of all neurons
  // normally distributed delays, weight Win and CPN connections per neuron
  float* inh_delays =
    ngpu.RandomNormalClipped( CPN * NI, mean_delay, std_delay, min_delay, mean_delay + 3 * std_delay );

  ConnSpec conn_spec2( FIXED_OUTDEGREE, CPN );
  SynSpec syn_spec2;
  syn_spec2.SetParam( "receptor", 1 );
  syn_spec2.SetParam( "weight", Win );
  syn_spec2.SetParam( "delay_array", inh_delays );
  ngpu.Connect( inh_neuron, neuron, conn_spec2, syn_spec2 );

  delete[] inh_delays;

  ConnSpec conn_spec3( ALL_TO_ALL );
  SynSpec syn_spec3( STANDARD_SYNAPSE, poiss_weight, poiss_delay, 0 );
  // connect poisson generator to port 0 of all neurons
  ngpu.Connect( pg, neuron, conn_spec3, syn_spec3 );

  char filename[] = "test_brunel_outdegree.dat";
  // any set of neuron indexes
  int i_neuron_arr[] = { neuron[ 0 ],
    neuron[ rand() % n_neurons ],
    neuron[ rand() % n_neurons ],
    neuron[ rand() % n_neurons ],
    neuron[ rand() % n_neurons ],
    neuron[ rand() % n_neurons ],
    neuron[ rand() % n_neurons ],
    neuron[ rand() % n_neurons ],
    neuron[ rand() % n_neurons ],
    neuron[ n_neurons - 1 ] };
  // create multimeter record of V_m
  std::string var_name_arr[] = { "V_m", "V_m", "V_m", "V_m", "V_m", "V_m", "V_m", "V_m", "V_m", "V_m" };
  ngpu.CreateRecord( string( filename ), var_name_arr, i_neuron_arr, 10 );

  ngpu.Simulate();

  return 0;
}
