/*
 *  connect_spec.h
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





#ifndef CONNECTSPEC_H
#define CONNECTSPEC_H

#include <iostream>

#define STANDARD_SYNAPSE 0

class NESTGPU;

template <class T>
class RemoteNode
{
 public:
  int i_host_;
  T i_node_;
 RemoteNode(int i_host, T node): i_host_(i_host), i_node_(node) {}
  int GetINode(int in);
};

enum ConnectionRules
  {
   ONE_TO_ONE=0, ALL_TO_ALL, FIXED_TOTAL_NUMBER, FIXED_INDEGREE,
   FIXED_OUTDEGREE, N_CONN_RULE
  };

const std::string conn_rule_name[N_CONN_RULE] =
  {
   "one_to_one", "all_to_all", "fixed_total_number", "fixed_indegree",
   "fixed_outdegree"
};

class ConnSpec
{
  int rule_;
  int total_num_;
  int indegree_;
  int outdegree_;
public:
  ConnSpec();
  ConnSpec(int rule, int degree=0);
  int Init();
  int Init(int rule, int degree=0);
  int SetParam(std::string param_name, int value);
  int GetParam(std::string param_name);
  static bool IsParam(std::string param_name);

  friend class NESTGPU;
};


class SynSpec
{
 public:
  unsigned char syn_group_;
  int port_;
  int weight_distr_;
  float *weight_h_array_pt_;
  float weight_;
  int delay_distr_;
  float *delay_h_array_pt_;
  float delay_;
  float weight_mu_;
  float weight_low_;
  float weight_high_;
  float weight_sigma_;
  float delay_mu_;
  float delay_low_;
  float delay_high_;
  float delay_sigma_;
  
 public:
  SynSpec();
  SynSpec(float weight, float delay);
  SynSpec(int syn_group, float weight, float delay, int port=0);
  int Init();
  int Init(float weight, float delay);
  int Init(int syn_group, float weight, float delay, int port=0);
  int SetParam(std::string param_name, int value);
  int SetParam(std::string param_name, float value);
  int SetParam(std::string param_name, float *array_pt);
  float GetParam(std::string param_name);
  static bool IsIntParam(std::string param_name);
  static bool IsFloatParam(std::string param_name);
  static bool IsFloatPtParam(std::string param_name);

  friend class NESTGPU;
};

#endif
