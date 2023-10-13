/*
 *  test_connect.cpp
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

#include "nestgpu.h"

int main(int argc, char *argv[])
{
  const int N = 5;
  
  NESTGPU ngpu;

  NodeSeq neuron = ngpu.Create("aeif_cond_beta", 2*N);
  std::vector<int> neuron_even;
  std::vector<int> neuron_odd;
  for (int i=0; i<N; i++) {
    neuron_even.push_back(neuron[2*i]);
    neuron_odd.push_back(neuron[2*i+1]);
  }
  float even_to_odd_delay[N*N];
  float even_to_odd_weight[N*N];
  float odd_to_even_delay[N*N];
  float odd_to_even_weight[N*N];
  for (int is=0; is<N; is++) {
    int ise = 2*is;
    int iso = 2*is + 1;
    for (int it=0; it<N; it++) {
      int ite = 2*it;
      int ito = 2*it + 1;
      even_to_odd_delay[it*N+is] = 2.0*N*ise + ito;
      even_to_odd_weight[it*N+is] = 100.0*(2.0*N*ise + ito);
      odd_to_even_delay[it*N+is] = 2.0*N*iso + ite;
      odd_to_even_weight[it*N+is] = 100.0*(2.0*N*iso + ite);
    }
  }

  ConnSpec conn_spec(ALL_TO_ALL);
  SynSpec even_to_odd_syn_spec;
  even_to_odd_syn_spec.SetParam("weight_array", even_to_odd_weight);
  even_to_odd_syn_spec.SetParam("delay_array", even_to_odd_delay);
  SynSpec odd_to_even_syn_spec;
  odd_to_even_syn_spec.SetParam("weight_array", odd_to_even_weight);
  odd_to_even_syn_spec.SetParam("delay_array", odd_to_even_delay);
  
  ngpu.Connect(neuron_even, neuron_odd, conn_spec, even_to_odd_syn_spec);
  ngpu.Connect(neuron_odd, neuron_even, conn_spec, odd_to_even_syn_spec);

  // Even to all
  std::vector<ConnectionId> conn_id
    = ngpu.GetConnections(neuron_even, neuron);
  std::vector<ConnectionStatus> conn_stat_vect
    = ngpu.GetConnectionStatus(conn_id);
  std::cout << "########################################\n";
  std::cout << "Even to all\n";
  for (unsigned int i=0; i<conn_stat_vect.size(); i++) {
    int i_source = conn_stat_vect[i].i_source;
    int i_target = conn_stat_vect[i].i_target;
    int port = conn_stat_vect[i].port;
    int syn_group = conn_stat_vect[i].syn_group;
    float weight = conn_stat_vect[i].weight;
    float delay = conn_stat_vect[i].delay;
    std::cout << "  i_source " << i_source << "\n";
    std::cout << "  i_target " << i_target << "\n";
    std::cout << "  port " << port << "\n";
    std::cout << "  syn_group " << syn_group << "\n";
    std::cout << "  weight " << weight << "\n";
    std::cout << "  delay " << delay << "\n";
    std::cout << "\n";
  }
  std::cout << "########################################\n";

  
  // All to odd
  conn_id = ngpu.GetConnections(neuron, neuron_odd);
  conn_stat_vect = ngpu.GetConnectionStatus(conn_id);
  std::cout << "########################################\n";
  std::cout << "All to odd\n";
  for (unsigned int i=0; i<conn_stat_vect.size(); i++) {
    int i_source = conn_stat_vect[i].i_source;
    int i_target = conn_stat_vect[i].i_target;
    int port = conn_stat_vect[i].port;
    int syn_group = conn_stat_vect[i].syn_group;
    float weight = conn_stat_vect[i].weight;
    float delay = conn_stat_vect[i].delay;
    std::cout << "  i_source " << i_source << "\n";
    std::cout << "  i_target " << i_target << "\n";
    std::cout << "  port " << port << "\n";
    std::cout << "  syn_group " << syn_group << "\n";
    std::cout << "  weight " << weight << "\n";
    std::cout << "  delay " << delay << "\n";
    std::cout << "\n";
  }
  std::cout << "########################################\n";

  
  // Even to 3,4,5,6
  NodeSeq neuron_3_6 = neuron.Subseq(3,6);
  conn_id = ngpu.GetConnections(neuron_even, neuron_3_6);
  conn_stat_vect = ngpu.GetConnectionStatus(conn_id);
  std::cout << "########################################\n";
  std::cout << "Even to 3,4,5,6\n";
  for (unsigned int i=0; i<conn_stat_vect.size(); i++) {
    int i_source = conn_stat_vect[i].i_source;
    int i_target = conn_stat_vect[i].i_target;
    int port = conn_stat_vect[i].port;
    int syn_group = conn_stat_vect[i].syn_group;
    float weight = conn_stat_vect[i].weight;
    float delay = conn_stat_vect[i].delay;
    std::cout << "  i_source " << i_source << "\n";
    std::cout << "  i_target " << i_target << "\n";
    std::cout << "  port " << port << "\n";
    std::cout << "  syn_group " << syn_group << "\n";
    std::cout << "  weight " << weight << "\n";
    std::cout << "  delay " << delay << "\n";
    std::cout << "\n";
  }
  std::cout << "########################################\n";

  
  // 3,4,5,6 to odd
  conn_id = ngpu.GetConnections(neuron_3_6, neuron_odd);
  conn_stat_vect = ngpu.GetConnectionStatus(conn_id);
  std::cout << "########################################\n";
  std::cout << "3,4,5,6 to odd\n";
  for (unsigned int i=0; i<conn_stat_vect.size(); i++) {
    int i_source = conn_stat_vect[i].i_source;
    int i_target = conn_stat_vect[i].i_target;
    int port = conn_stat_vect[i].port;
    int syn_group = conn_stat_vect[i].syn_group;
    float weight = conn_stat_vect[i].weight;
    float delay = conn_stat_vect[i].delay;
    std::cout << "  i_source " << i_source << "\n";
    std::cout << "  i_target " << i_target << "\n";
    std::cout << "  port " << port << "\n";
    std::cout << "  syn_group " << syn_group << "\n";
    std::cout << "  weight " << weight << "\n";
    std::cout << "  delay " << delay << "\n";
    std::cout << "\n";
  }
  std::cout << "########################################\n";

  
  return 0;
}
