/*
 *  connect_rules.cu
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
#include "ngpu_exception.h"
//#include "connect.h"
#include "nestgpu.h"
#include "connect_rules.h"
#include "connect.h"
#include "distribution.h"

int ConnSpec::Init()
{
  rule_ = ALL_TO_ALL;
  total_num_ = 0;
  indegree_ = 0;
  outdegree_ = 0;
  return 0;
}
			    
ConnSpec::ConnSpec()
{
  Init();
}

int ConnSpec::Init(int rule, int degree /*=0*/)
{
  Init();
  if (rule<0 || rule>N_CONN_RULE) {
    throw ngpu_exception("Unknown connection rule");
  }
  if ((rule==ALL_TO_ALL || rule==ONE_TO_ONE) && (degree != 0)) {
    throw ngpu_exception(std::string("Connection rule ") + conn_rule_name[rule]
			 + " does not have a degree");
  }
  rule_ = rule;
  if (rule==FIXED_TOTAL_NUMBER) {
    total_num_ = degree;
  }
  else if (rule==FIXED_INDEGREE) {
    indegree_ = degree;
  }
  else if (rule==FIXED_OUTDEGREE) {
    outdegree_ = degree;
  }
  
  return 0;
}

ConnSpec::ConnSpec(int rule, int degree /*=0*/)
{
  Init(rule, degree);
}

int ConnSpec::SetParam(std::string param_name, int value)
{
  if (param_name=="rule") {
    if (value<0 || value>N_CONN_RULE) {
      throw ngpu_exception("Unknown connection rule");
    }
    rule_ = value;
    return 0;
  }
  else if (param_name=="indegree") {
    if (value<0) {
      throw ngpu_exception("Indegree must be >=0");
    }
    indegree_ = value;
    return 0;
  }
  else if (param_name=="outdegree") {
    if (value<0) {
      throw ngpu_exception("Outdegree must be >=0");
    }
    outdegree_ = value;
    return 0;
  }
  else if (param_name=="total_num") {
    if (value<0) {
      throw ngpu_exception("total_num must be >=0");
    }
    total_num_ = value;
    return 0;
  }

  throw ngpu_exception("Unknown connection int parameter");
}

bool ConnSpec::IsParam(std::string param_name)
{
  if (param_name=="rule" || param_name=="indegree" || param_name=="outdegree"
      || param_name=="total_num") {
    return true;
  }
  else {
    return false;
  }
}

SynSpec::SynSpec()
{
  Init();
}


int SynSpec::Init()
{
  syn_group_ = 0;
  port_ = 0;
  weight_ = 0;
  delay_ = 0;
  weight_distr_ = DISTR_TYPE_NONE;
  delay_distr_ = DISTR_TYPE_NONE;
  weight_h_array_pt_ = NULL;
  delay_h_array_pt_ = NULL;

  return 0;
}


SynSpec::SynSpec(float weight, float delay)
{
  Init(weight, delay);
}

int SynSpec::Init(float weight, float delay)
{
  if (delay<0) {
    throw ngpu_exception("Delay must be >=0");
  }
  Init();
  weight_ = weight;
  delay_ = delay;

  return 0;
 }

SynSpec::SynSpec(int syn_group, float weight, float delay, int port /*=0*/)
{
  Init(syn_group, weight, delay, port);
}

int SynSpec::Init(int syn_group, float weight, float delay, int port /*=0*/)
{
  if (syn_group<0) { // || syn_group>n_syn_group) {
    throw ngpu_exception("Unknown synapse group");
  }
  if (port<0) {
    throw ngpu_exception("Port index must be >=0");
  }
  Init(weight, delay);
  syn_group_ = syn_group;
  port_ = port;

  return 0;
 }

int SynSpec::SetParam(std::string param_name, int value)
{
  if (param_name=="synapse_group") {
    if (value<0) { // || value>n_syn_group) {
      throw ngpu_exception("Unknown synapse group");
    }
    syn_group_ = value;
  }
  else if (param_name=="receptor") {
    if (value<0) {
      throw ngpu_exception("Port index must be >=0");
    }
    port_ = value;
  }
  else if (param_name=="weight_distribution") {
    weight_distr_ = value;
    //printf("weight_distribution_ idx: %d\n", value);
  }
  else if (param_name=="delay_distribution") {
    delay_distr_ = value;
    //printf("delay_distribution_ idx: %d\n", value);
  }
  else  {
    throw ngpu_exception("Unknown synapse int parameter");
  }
  
  return 0;
}

bool SynSpec::IsIntParam(std::string param_name)
{
  if (param_name=="synapse_group" || param_name=="receptor"
      || param_name=="weight_distribution"
      || param_name=="delay_distribution"
      ) {
    return true;
  }
  else {
    return false;
  }
}

int SynSpec::SetParam(std::string param_name, float value)
{
  if (param_name=="weight") {
    weight_ = value;
  }
  else if (param_name=="delay") {
    if (value<0) {
      throw ngpu_exception("Delay must be >=0");
    }
    delay_ = value;
  }
  else if (param_name=="weight_mu") {
    weight_mu_ = value;
    //printf("weight_mu_: %f\n", value);
  }
  else if (param_name=="weight_low") {
    weight_low_ = value;
    //printf("weight_low_: %f\n", value);
  }
  else if (param_name=="weight_high") {
    weight_high_ = value;
    //printf("weight_high_: %f\n", value);
  }
  else if (param_name=="weight_sigma") {
    weight_sigma_ = value;
    //printf("weight_sigma_: %f\n", value);
  }
  else if (param_name=="delay_mu") {
    delay_mu_ = value;
    //printf("delay_mu_: %f\n", value);
  }
  else if (param_name=="delay_low") {
    delay_low_ = value;
    //printf("delay_low_: %f\n", value);
  }
  else if (param_name=="delay_high") {
    delay_high_ = value;
    //printf("delay_high_: %f\n", value);
  }
  else if (param_name=="delay_sigma") {
    delay_sigma_ = value;
    //printf("delay_sigma_: %f\n", value);
  }
  else {
    throw ngpu_exception("Unknown synapse float parameter");
  }
  return 0;
}

bool SynSpec::IsFloatParam(std::string param_name)
{
  if (param_name=="weight" || param_name=="delay"
      || param_name=="weight_mu" || param_name=="weight_low"
      || param_name=="weight_high" || param_name=="weight_sigma"
      || param_name=="delay_mu" || param_name=="delay_low"
      || param_name=="delay_high" || param_name=="delay_sigma"
      ) {
    return true;
  }
  else {
    return false;
  }
}
 
int SynSpec::SetParam(std::string param_name, float *array_pt)
{
  if (param_name=="weight_array") {
    weight_h_array_pt_ = array_pt;
    weight_distr_ = DISTR_TYPE_ARRAY;
  }
  else if (param_name=="delay_array") {
    delay_h_array_pt_ = array_pt;
    delay_distr_ = DISTR_TYPE_ARRAY;
  }
  else {
    throw ngpu_exception("Unknown synapse array parameter");
  }
  
  return 0;
}

bool SynSpec::IsFloatPtParam(std::string param_name)
{
  if (param_name=="weight_array" || param_name=="delay_array") {
    return true;
  }
  else {
    return false;
  }
}



int NESTGPU::Connect(int i_source, int n_source, int i_target, int n_target,
		       ConnSpec &conn_spec, SynSpec &syn_spec)
{
  return _Connect<int, int>(i_source, n_source, i_target, n_target,
			    conn_spec, syn_spec);
}

int NESTGPU::Connect(int i_source, int n_source, int* target, int n_target,
		       ConnSpec &conn_spec, SynSpec &syn_spec)
{
  int *d_target;
  CUDAMALLOCCTRL("&d_target",&d_target, n_target*sizeof(int));
  gpuErrchk(cudaMemcpy(d_target, target, n_target*sizeof(int),
		       cudaMemcpyHostToDevice));
  int ret = _Connect<int, int*>(i_source, n_source, d_target, n_target,
				conn_spec, syn_spec);
  CUDAFREECTRL("d_target",d_target);

  return ret;
}
int NESTGPU::Connect(int* source, int n_source, int i_target, int n_target,
		       ConnSpec &conn_spec, SynSpec &syn_spec)
{
  int *d_source;
  CUDAMALLOCCTRL("&d_source",&d_source, n_source*sizeof(int));
  gpuErrchk(cudaMemcpy(d_source, source, n_source*sizeof(int),
		       cudaMemcpyHostToDevice));
  int ret = _Connect<int*, int>(d_source, n_source, i_target, n_target,
				conn_spec, syn_spec);
  CUDAFREECTRL("d_source",d_source);
  
  return ret;
}
int NESTGPU::Connect(int* source, int n_source, int* target, int n_target,
		       ConnSpec &conn_spec, SynSpec &syn_spec)
{
  int *d_source;
  CUDAMALLOCCTRL("&d_source",&d_source, n_source*sizeof(int));
  gpuErrchk(cudaMemcpy(d_source, source, n_source*sizeof(int),
		       cudaMemcpyHostToDevice));
  int *d_target;
  CUDAMALLOCCTRL("&d_target",&d_target, n_target*sizeof(int));
  gpuErrchk(cudaMemcpy(d_target, target, n_target*sizeof(int),
		       cudaMemcpyHostToDevice));
  int ret = _Connect<int*, int*>(d_source, n_source, d_target, n_target,
				 conn_spec, syn_spec);
  CUDAFREECTRL("d_source",d_source);
  CUDAFREECTRL("d_target",d_target);

  return ret;
}

int NESTGPU::Connect(NodeSeq source, NodeSeq target,
		     ConnSpec &conn_spec, SynSpec &syn_spec)
{
  return Connect(source.i0, source.n, target.i0, target.n,
		 conn_spec, syn_spec);
}

int NESTGPU::Connect(NodeSeq source, std::vector<int> target,
		     ConnSpec &conn_spec, SynSpec &syn_spec)
{
  return Connect(source.i0, source.n, target.data(),
		 target.size(), conn_spec, syn_spec);
}

int NESTGPU::Connect(std::vector<int> source, NodeSeq target,
		     ConnSpec &conn_spec, SynSpec &syn_spec)
{
  return Connect(source.data(), source.size(), target.i0,
		 target.n, conn_spec, syn_spec);
}

int NESTGPU::Connect(std::vector<int> source, std::vector<int> target,
		     ConnSpec &conn_spec, SynSpec &syn_spec)
{
  return Connect(source.data(), source.size(), target.data(),
		 target.size(), conn_spec, syn_spec);
}


int NESTGPU::RemoteConnect(int i_source_host, int i_source, int n_source,
			   int i_target_host, int i_target, int n_target,
			   ConnSpec &conn_spec, SynSpec &syn_spec)
{
  return _RemoteConnect<int, int>(i_source_host, i_source, n_source,
				  i_target_host, i_target, n_target,
				  conn_spec, syn_spec);
}

int NESTGPU::RemoteConnect(int i_source_host, int i_source, int n_source,
			   int i_target_host, int* target, int n_target,
			   ConnSpec &conn_spec, SynSpec &syn_spec)
{
  int *d_target;
  CUDAMALLOCCTRL("&d_target",&d_target, n_target*sizeof(int));
  gpuErrchk(cudaMemcpy(d_target, target, n_target*sizeof(int),
		       cudaMemcpyHostToDevice));
  int ret = _RemoteConnect<int, int*>(i_source_host, i_source, n_source,
				      i_target_host, d_target, n_target,
				      conn_spec, syn_spec);
  CUDAFREECTRL("d_target",d_target);

  return ret;
}

int NESTGPU::RemoteConnect(int i_source_host, int* source, int n_source,
			   int i_target_host, int i_target, int n_target,
			   ConnSpec &conn_spec, SynSpec &syn_spec)
{
  int *d_source;
  CUDAMALLOCCTRL("&d_source",&d_source, n_source*sizeof(int));
  gpuErrchk(cudaMemcpy(d_source, source, n_source*sizeof(int),
		       cudaMemcpyHostToDevice));
  int ret = _RemoteConnect<int*, int>(i_source_host, d_source, n_source,
				      i_target_host, i_target, n_target,
				      conn_spec, syn_spec);
  CUDAFREECTRL("d_source",d_source);
  
  return ret;
}

int NESTGPU::RemoteConnect(int i_source_host, int* source, int n_source,
			   int i_target_host, int* target, int n_target,
			   ConnSpec &conn_spec, SynSpec &syn_spec)
{
  int *d_source;
  CUDAMALLOCCTRL("&d_source",&d_source, n_source*sizeof(int));
  gpuErrchk(cudaMemcpy(d_source, source, n_source*sizeof(int),
		       cudaMemcpyHostToDevice));
  int *d_target;
  CUDAMALLOCCTRL("&d_target",&d_target, n_target*sizeof(int));
  gpuErrchk(cudaMemcpy(d_target, target, n_target*sizeof(int),
		       cudaMemcpyHostToDevice));
  int ret = _RemoteConnect<int*, int*>(i_source_host, d_source, n_source,
				       i_target_host, d_target, n_target,
				       conn_spec, syn_spec);
  CUDAFREECTRL("d_source",d_source);
  CUDAFREECTRL("d_target",d_target);

  return ret;
}

int NESTGPU::RemoteConnect(int i_source_host, NodeSeq source,
			   int i_target_host, NodeSeq target,
			   ConnSpec &conn_spec, SynSpec &syn_spec)
{
  return RemoteConnect(i_source_host, source.i0, source.n,
		       i_target_host, target.i0, target.n,
		       conn_spec, syn_spec);
}

int NESTGPU::RemoteConnect(int i_source_host, NodeSeq source,
			   int i_target_host, std::vector<int> target,
			   ConnSpec &conn_spec, SynSpec &syn_spec)
{
  return RemoteConnect(i_source_host, source.i0, source.n,
		       i_target_host, target.data(), target.size(),
		       conn_spec, syn_spec);
}

int NESTGPU::RemoteConnect(int i_source_host, std::vector<int> source,
			   int i_target_host, NodeSeq target,
			   ConnSpec &conn_spec, SynSpec &syn_spec)
{
  return RemoteConnect(i_source_host, source.data(), source.size(),
			i_target_host, target.i0, target.n,
			conn_spec, syn_spec);
}

int NESTGPU::RemoteConnect(int i_source_host, std::vector<int> source,
			   int i_target_host, std::vector<int> target,
			   ConnSpec &conn_spec, SynSpec &syn_spec)
{
  return RemoteConnect(i_source_host, source.data(), source.size(),
		       i_target_host, target.data(), target.size(),
		       conn_spec, syn_spec);
}

