/*
 *  spike_generator.cu
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
#include <cmath>
#include <iostream>
//#include <stdio.h>

#include "nestgpu.h"
#include "neuron_models.h"
#include "spike_generator.h"
#include "spike_buffer.h"
#include "cuda_error.h"
//#include "spike_generator_variables.h"
const int N_SPIKE_GEN_SCAL_PARAM = 0;
const std::string *spike_gen_scal_param_name = NULL;
enum {
  i_SPIKE_TIME_ARRAY_PARAM=0,
  i_SPIKE_HEIGHT_ARRAY_PARAM,
  N_SPIKE_GEN_ARRAY_PARAM
};

const std::string spike_gen_array_param_name[N_SPIKE_GEN_ARRAY_PARAM]
= {"spike_times", "spike_heights"};

__global__
void spike_generatorUpdate(int i_node_0, int n_node, long long i_time,
			  int *n_spikes, int *i_spike, int **spike_time_idx,
			  float **spike_height)
{
  int irel_node = threadIdx.x + blockIdx.x * blockDim.x;
  if (irel_node < n_node) {
    if (n_spikes[irel_node] > 0) {
      int is = i_spike[irel_node];
      if (is<n_spikes[irel_node]
          && spike_time_idx[irel_node][is]==i_time) {
	int i_node = i_node_0 + irel_node;
	float height = spike_height[irel_node][is];
	PushSpike(i_node, height);
	i_spike[irel_node]++;
      }
    }
  }
}

int spike_generator::Init(int i_node_0, int n_node, int /*n_port*/,
			  int i_group)
{
  BaseNeuron::Init(i_node_0, n_node, 0 /*n_port*/, i_group);
  node_type_ = i_spike_generator_model;
  n_scal_param_ = N_SPIKE_GEN_SCAL_PARAM;
  n_param_ = n_scal_param_;
  scal_param_name_ = spike_gen_scal_param_name;

  for (int i=0; i<N_SPIKE_GEN_ARRAY_PARAM; i++) {
    array_param_name_.push_back(spike_gen_array_param_name[i]);
  }				
  std::vector<float> empty_vect;
  spike_time_vect_.clear();
  spike_time_vect_.insert(spike_time_vect_.begin(), n_node, empty_vect);
  spike_height_vect_.clear();
  spike_height_vect_.insert(spike_height_vect_.begin(), n_node, empty_vect);
  
  CUDAMALLOCCTRL("&param_arr_",&param_arr_, n_node_*n_param_*sizeof(float));

  //SetScalParam(0, n_node, "origin", 0.0);
  
  h_spike_time_idx_ = new int*[n_node_];
  h_spike_height_ = new float*[n_node_];
  for (int i_node=0; i_node<n_node_; i_node++) {
    h_spike_time_idx_[i_node] = 0;
    h_spike_height_[i_node] = 0;
  }
  
  CUDAMALLOCCTRL("&d_n_spikes_",&d_n_spikes_, n_node_*sizeof(int));
  CUDAMALLOCCTRL("&d_i_spike_",&d_i_spike_, n_node_*sizeof(int));
  CUDAMALLOCCTRL("&d_spike_time_idx_",&d_spike_time_idx_, n_node_*sizeof(int*));
  CUDAMALLOCCTRL("&d_spike_height_",&d_spike_height_, n_node_*sizeof(float*));
  
  gpuErrchk(cudaMemset(d_n_spikes_, 0, n_node_*sizeof(int)));
  gpuErrchk(cudaMemset(d_i_spike_, 0, n_node_*sizeof(int)));
  gpuErrchk(cudaMemset(d_spike_time_idx_, 0, n_node_*sizeof(int*)));
  gpuErrchk(cudaMemset(d_spike_height_, 0, n_node_*sizeof(float*)));
  
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  
  return 0;
}


int spike_generator::Free()
{
  for (int i_node=0; i_node<n_node_; i_node++) {
    if(h_spike_time_idx_[i_node] != 0) {
      CUDAFREECTRL("h_spike_time_idx_[i_node]",h_spike_time_idx_[i_node]);
      CUDAFREECTRL("h_spike_height_[i_node]",h_spike_height_[i_node]);
    }
  }  
  CUDAFREECTRL("d_n_spikes_",d_n_spikes_);
  CUDAFREECTRL("d_i_spike_",d_i_spike_);	    
  CUDAFREECTRL("d_spike_time_idx_",d_spike_time_idx_);
  CUDAFREECTRL("d_spike_height_",d_spike_height_);

  delete[] h_spike_time_idx_;
  delete[] h_spike_height_;
  
  return 0;
}

spike_generator::~spike_generator()
{
  if (n_node_>0) {
    Free();
  }
}

int spike_generator::Update(long long i_time, double /*t1*/)
{
  spike_generatorUpdate<<<(n_node_+1023)/1024, 1024>>>
    (i_node_0_, n_node_, i_time, d_n_spikes_, d_i_spike_, d_spike_time_idx_,
     d_spike_height_);
  //gpuErrchk( cudaPeekAtLastError() );
  //gpuErrchk( cudaDeviceSynchronize() );

  return 0;
}

int spike_generator::SetArrayParam(int i_neuron, int n_neuron,
				   std::string param_name, float *array,
				   int array_size)
{
  CheckNeuronIdx(i_neuron);
  CheckNeuronIdx(i_neuron + n_neuron - 1);

  if (param_name==array_param_name_[i_SPIKE_TIME_ARRAY_PARAM]) {
    for (int in=i_neuron; in<i_neuron+n_neuron; in++) {
      spike_time_vect_[in] = std::vector<float>(array, array+array_size);
    }
  }
  else if (param_name==array_param_name_[i_SPIKE_HEIGHT_ARRAY_PARAM]) {
    for (int in=i_neuron; in<i_neuron+n_neuron; in++) {
      spike_height_vect_[in] = std::vector<float>(array, array+array_size);
    }
  }
  else {
    throw ngpu_exception(std::string("Unrecognized array parameter ")
			 + param_name);
  }

  return 0;
}
  
int spike_generator::SetArrayParam(int *i_neuron, int n_neuron,
				   std::string param_name, float *array,
				   int array_size)
{
  if (param_name==array_param_name_[i_SPIKE_TIME_ARRAY_PARAM]) {
    for (int i=0; i<n_neuron; i++) {
      int in = i_neuron[i];
      CheckNeuronIdx(in);
      spike_time_vect_[in] = std::vector<float>(array, array+array_size);
    }
  }
  else if (param_name==array_param_name_[i_SPIKE_HEIGHT_ARRAY_PARAM]) {
    for (int i=0; i<n_neuron; i++) {
      int in = i_neuron[i];
      CheckNeuronIdx(in);      
      spike_height_vect_[in] = std::vector<float>(array, array+array_size);
    }
  }
  else {
    throw ngpu_exception(std::string("Unrecognized array parameter ")
			 + param_name);
  }

  return 0;
}

int spike_generator::Calibrate(double time_min, float time_resolution)
{
  for (int in=0; in<n_node_; in++) {
    unsigned int n_spikes = spike_time_vect_[in].size();
    if (n_spikes>0) {
      if (spike_height_vect_[in].size()==0) {
	spike_height_vect_[in].insert(spike_height_vect_[in].begin(),
				      n_spikes, 1.0);
      }
      else if (spike_height_vect_[in].size()!=n_spikes) {
	throw ngpu_exception("spike time array and spike height array "
			     "must have the same size in spike generator");
      }
      SetSpikes(in, n_spikes, spike_time_vect_[in].data(),
		spike_height_vect_[in].data(), (float)time_min, time_resolution);
    }
  }
  
  return 0;
}
  


int spike_generator::SetSpikes(int irel_node, int n_spikes, float *spike_time,
			       float *spike_height, float time_min,
			       float time_resolution)
{
  if (n_spikes <=0) {
    throw ngpu_exception("Number of spikes must be greater than 0 "
			 "in spike generator setting");
  }
  
  cudaMemcpy(&d_n_spikes_[irel_node], &n_spikes, sizeof(int),
	     cudaMemcpyHostToDevice);
  if (h_spike_time_idx_[irel_node] != 0) {
    CUDAFREECTRL("h_spike_time_idx_[irel_node]",h_spike_time_idx_[irel_node]);
    CUDAFREECTRL("h_spike_height_[irel_node]",h_spike_height_[irel_node]);
  }
  CUDAMALLOCCTRL("&h_spike_time_idx_[irel_node]",&h_spike_time_idx_[irel_node], n_spikes*sizeof(int));
  CUDAMALLOCCTRL("&h_spike_height_[irel_node]",&h_spike_height_[irel_node], n_spikes*sizeof(float));

  cudaMemcpy(&d_spike_time_idx_[irel_node], &h_spike_time_idx_[irel_node],
	     sizeof(int*), cudaMemcpyHostToDevice);
  cudaMemcpy(&d_spike_height_[irel_node], &h_spike_height_[irel_node], sizeof(float*),
	     cudaMemcpyHostToDevice);

  int *spike_time_idx = new int[n_spikes];
  for(int i=0; i<n_spikes; i++) {
    spike_time_idx[i] = (int)round((spike_time[i] - time_min)
				   /time_resolution);
    if (i>0 && spike_time_idx[i]<=spike_time_idx[i-1]) {
      throw ngpu_exception("Spike times must be ordered, and the difference "
			   "between\nconsecutive spikes must be >= the "
			   "time resolution");
    }
    //cout << "ti " << spike_time_idx[i] << endl;
    //cout << spike_time[i] << " " << time_min << endl;
      
  }
  
  cudaMemcpy(h_spike_time_idx_[irel_node], spike_time_idx, n_spikes*sizeof(int),
	     cudaMemcpyHostToDevice);
  cudaMemcpy(h_spike_height_[irel_node], spike_height, n_spikes*sizeof(float),
	     cudaMemcpyHostToDevice);

  return 0;
}

int spike_generator::GetArrayParamSize(int i_neuron, std::string param_name)
{
  if (param_name==array_param_name_[i_SPIKE_TIME_ARRAY_PARAM]) {
    return spike_time_vect_[i_neuron].size();
  }
  else if (param_name==array_param_name_[i_SPIKE_HEIGHT_ARRAY_PARAM]) {
    return spike_height_vect_[i_neuron].size();
  }
  else {
    throw ngpu_exception(std::string("Unrecognized parameter ")
			 + param_name);
  }
}

float *spike_generator::GetArrayParam(int i_neuron, std::string param_name)
{
  if (param_name==array_param_name_[i_SPIKE_TIME_ARRAY_PARAM]) {
    return spike_time_vect_[i_neuron].data();
  }
  else if (param_name==array_param_name_[i_SPIKE_HEIGHT_ARRAY_PARAM]) {
    return spike_height_vect_[i_neuron].data();
  }
  else {
    throw ngpu_exception(std::string("Unrecognized parameter ")
			 + param_name);
  }
}
