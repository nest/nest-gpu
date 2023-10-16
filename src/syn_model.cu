/*
 *  syn_model.cu
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
#include "cuda_error.h"
#include "nestgpu.h"
#include "syn_model.h"
#include "test_syn_model.h"
#include "stdp.h"

int *d_SynGroupTypeMap;
__device__ int *SynGroupTypeMap;

float **d_SynGroupParamMap;
__device__ float **SynGroupParamMap;

__device__ void TestSynModelUpdate(float *w, float Dt, float *param);

__global__ void SynGroupInit(int *syn_group_type_map,
			     float **syn_group_param_map)
{
  SynGroupTypeMap = syn_group_type_map;
  SynGroupParamMap = syn_group_param_map;
  
}

int SynModel::GetNParam()
{
  return n_param_;
}

std::vector<std::string> SynModel::GetParamNames()
{
  std::vector<std::string> param_name_vect;
  for (int i=0; i<n_param_; i++) {
    param_name_vect.push_back(param_name_[i]);
  }
  
  return param_name_vect;
}

bool SynModel::IsParam(std::string param_name)
{
  int i_param;
  for (i_param=0; i_param<n_param_; i_param++) {
    if (param_name == param_name_[i_param]) return true;
  }
  return false;
}

int SynModel::GetParamIdx(std::string param_name)
{
  int i_param;
  for (i_param=0; i_param<n_param_; i_param++) {
    if (param_name == param_name_[i_param]) break;
  }
  if (i_param == n_param_) {
    throw ngpu_exception(std::string("Unrecognized parameter ")
			 + param_name);
  }
  
  return i_param;
}

float SynModel::GetParam(std::string param_name)
{
  if (!IsParam(param_name)) {
    throw ngpu_exception(std::string("Unrecognized synapse parameter ")
			 + param_name);
  }
  int i_param =  GetParamIdx(param_name);
  float *d_param_pt = d_param_arr_ + i_param;
  float param_val;
  gpuErrchk(cudaMemcpy(&param_val, d_param_pt, sizeof(float),
		       cudaMemcpyDeviceToHost));
  return param_val;
}


int SynModel::SetParam(std::string param_name, float val)
{
  if (!IsParam(param_name)) {
    throw ngpu_exception(std::string("Unrecognized synapse parameter ")
			 + param_name);
  }
  int i_param =  GetParamIdx(param_name);
  float *d_param_pt = d_param_arr_ + i_param;
  gpuErrchk(cudaMemcpy(d_param_pt, &val, sizeof(float),
		       cudaMemcpyHostToDevice));
  return 0;
}

  
int NESTGPU::CreateSynGroup(std::string model_name)
{
  CheckUncalibrated("Nodes cannot be created after calibration");
  if (model_name == syn_model_name[i_test_syn_model]) {
    TestSynModel *test_syn_model_group = new TestSynModel;
    syn_group_vect_.push_back(test_syn_model_group);
  }
  else if (model_name == syn_model_name[i_stdp_model]) {
    STDP *stdp_group = new STDP;
    syn_group_vect_.push_back(stdp_group);
  }
  else {
    throw ngpu_exception(std::string("Unknown synapse model name: ")
			 + model_name);
  }
  return syn_group_vect_.size(); // 0 is standard synapse
}

int NESTGPU::GetSynGroupNParam(int syn_group)
{
  if (syn_group<1 || syn_group>(int)syn_group_vect_.size()) {
    throw ngpu_exception("Unrecognized synapse group");
  }

  return syn_group_vect_[syn_group-1]->GetNParam();
}

std::vector<std::string> NESTGPU::GetSynGroupParamNames(int syn_group)
{
  if (syn_group<1 || syn_group>(int)syn_group_vect_.size()) {
    throw ngpu_exception("Unrecognized synapse group");
  }

  return syn_group_vect_[syn_group-1]->GetParamNames();
}

bool NESTGPU::IsSynGroupParam(int syn_group, std::string param_name)
{
  if (syn_group<1 || syn_group>(int)syn_group_vect_.size()) {
    throw ngpu_exception("Unrecognized synapse group");
  }

  return syn_group_vect_[syn_group-1]->IsParam(param_name);
}

int NESTGPU::GetSynGroupParamIdx(int syn_group, std::string param_name)
{
  if (syn_group<1 || syn_group>(int)syn_group_vect_.size()) {
    throw ngpu_exception("Unrecognized synapse group");
  }

  return syn_group_vect_[syn_group-1]->GetParamIdx(param_name);
}

float NESTGPU::GetSynGroupParam(int syn_group, std::string param_name)
{
  if (syn_group<1 || syn_group>(int)syn_group_vect_.size()) {
    throw ngpu_exception("Unrecognized synapse group");
  }

  return syn_group_vect_[syn_group-1]->GetParam(param_name);
}

int NESTGPU::SetSynGroupParam(int syn_group, std::string param_name,
				float val)
{
  if (syn_group<1 || syn_group>(int)syn_group_vect_.size()) {
    throw ngpu_exception("Unrecognized synapse group");
  }

  return syn_group_vect_[syn_group-1]->SetParam(param_name, val);
}


int NESTGPU::SynGroupCalibrate()
{
  int n_group = syn_group_vect_.size();
  int *h_SynGroupTypeMap = new int[n_group];
  float **h_SynGroupParamMap = new float*[n_group];

  for (int syn_group=1; syn_group<=n_group; syn_group++) {
    h_SynGroupTypeMap[syn_group-1] = syn_group_vect_[syn_group-1]->type_;
    h_SynGroupParamMap[syn_group-1]
      = syn_group_vect_[syn_group-1]->d_param_arr_;
  }
  CUDAMALLOCCTRL("&d_SynGroupTypeMap",&d_SynGroupTypeMap, n_group*sizeof(int));
  CUDAMALLOCCTRL("&d_SynGroupParamMap",&d_SynGroupParamMap, n_group*sizeof(float*));

  // Memcopies will be synchronised with SynGroupInit kernel
  gpuErrchk(cudaMemcpyAsync(d_SynGroupTypeMap, h_SynGroupTypeMap,
		       n_group*sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyAsync(d_SynGroupParamMap, h_SynGroupParamMap,
		       n_group*sizeof(float*), cudaMemcpyHostToDevice));

  SynGroupInit<<<1,1>>>(d_SynGroupTypeMap, d_SynGroupParamMap);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  
  delete[] h_SynGroupTypeMap;
  delete[] h_SynGroupParamMap;

  return 0;
}
