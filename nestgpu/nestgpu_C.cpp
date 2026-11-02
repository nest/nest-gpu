/*
 *  nestgpu_C.cpp
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

#include <array>
#include <config.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include "nestgpu.h"
#include "nestgpu_C.h"
#include "propagate_error.h"

#include "api_converters.h"
#include "c_api.h"

extern "C" {
static sapi::CAPI capi;
static std::unordered_map<std::size_t, sapi::RankNodeSequenceMap>
    spatial_node_sequence_map;
static NESTGPU *NESTGPU_instance = nullptr;
ConnSpec ConnSpec_instance;
SynSpec SynSpec_instance;

void checkNESTGPUInstance() {
  if (NESTGPU_instance == nullptr) {
    NESTGPU_instance = new NESTGPU();
  }
}

char *NESTGPU_GetErrorMessage() {
  checkNESTGPUInstance();
  char *cstr = NESTGPU_instance->GetErrorMessage();
  return cstr;
}

unsigned char NESTGPU_GetErrorCode() {
  checkNESTGPUInstance();
  return NESTGPU_instance->GetErrorCode();
}

void NESTGPU_SetOnException(int on_exception) {
  checkNESTGPUInstance();
  NESTGPU_instance->SetOnException(on_exception);
}

unsigned int *RandomInt(size_t n);

int NESTGPU_SetRandomSeed(unsigned long long seed) {
  int ret = 0;
  BEGIN_ERR_PROP { ret = NESTGPU_instance->SetRandomSeed(seed); }
  END_ERR_PROP return ret;
}

int NESTGPU_SetTimeResolution(float time_res) {
  int ret = 0;
  BEGIN_ERR_PROP { ret = NESTGPU_instance->SetTimeResolution(time_res); }
  END_ERR_PROP return ret;
}

float NESTGPU_GetTimeResolution() {
  float ret = 0.0;
  BEGIN_ERR_PROP { ret = NESTGPU_instance->GetTimeResolution(); }
  END_ERR_PROP return ret;
}

int NESTGPU_SetMaxSpikeBufferSize(int max_size) {
  int ret = 0;
  BEGIN_ERR_PROP { ret = NESTGPU_instance->SetMaxSpikeBufferSize(max_size); }
  END_ERR_PROP return ret;
}

int NESTGPU_GetMaxSpikeBufferSize() {
  int ret = 0;
  BEGIN_ERR_PROP { ret = NESTGPU_instance->GetMaxSpikeBufferSize(); }
  END_ERR_PROP return ret;
}

int NESTGPU_SetSimTime(float sim_time) {
  int ret = 0;
  BEGIN_ERR_PROP { ret = NESTGPU_instance->SetSimTime(sim_time); }
  END_ERR_PROP return ret;
}

int NESTGPU_SetVerbosityLevel(int verbosity_level) {
  int ret = 0;
  BEGIN_ERR_PROP { ret = NESTGPU_instance->SetVerbosityLevel(verbosity_level); }
  END_ERR_PROP return ret;
}

int NESTGPU_SetNestedLoopAlgo(int nested_loop_algo) {
  int ret = 0;
  BEGIN_ERR_PROP {
    ret = NESTGPU_instance->SetNestedLoopAlgo(nested_loop_algo);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_Create(char *model_name, int n_neuron, int n_port) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string model_name_str = std::string(model_name);
    NodeSeq neur = NESTGPU_instance->Create(model_name_str, n_neuron, n_port);
    ret = neur[0];
  }
  END_ERR_PROP return ret;
}

int NESTGPU_CreateRecord(char *file_name, char *var_name_arr[], int *i_node_arr,
                         int *port_arr, int n_node) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string file_name_str = std::string(file_name);
    std::vector<std::string> var_name_vect;
    for (int i = 0; i < n_node; i++) {
      std::string var_name = std::string(var_name_arr[i]);
      var_name_vect.push_back(var_name);
    }
    ret = NESTGPU_instance->CreateRecord(file_name_str, var_name_vect.data(),
                                         i_node_arr, port_arr, n_node);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_GetRecordDataRows(int i_record) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::vector<std::vector<float>> *data_vect_pt =
        NESTGPU_instance->GetRecordData(i_record);

    ret = data_vect_pt->size();
  }
  END_ERR_PROP return ret;
}

int NESTGPU_GetRecordDataColumns(int i_record) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::vector<std::vector<float>> *data_vect_pt =
        NESTGPU_instance->GetRecordData(i_record);

    ret = data_vect_pt->at(0).size();
  }
  END_ERR_PROP return ret;
}

float **NESTGPU_GetRecordData(int i_record) {
  float **ret = nullptr;
  BEGIN_ERR_PROP {
    std::vector<std::vector<float>> *data_vect_pt =
        NESTGPU_instance->GetRecordData(i_record);
    int nr = data_vect_pt->size();
    ret = new float *[nr];
    for (int i = 0; i < nr; i++) {
      ret[i] = data_vect_pt->at(i).data();
    }
  }
  END_ERR_PROP return ret;
}

int NESTGPU_SetNeuronScalParam(int i_node, int n_neuron, char *param_name,
                               float val) {
  int ret = 0;
  BEGIN_ERR_PROP {

    std::string param_name_str = std::string(param_name);
    ret =
        NESTGPU_instance->SetNeuronParam(i_node, n_neuron, param_name_str, val);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_SetNeuronArrayParam(int i_node, int n_neuron, char *param_name,
                                float *param, int array_size) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->SetNeuronParam(i_node, n_neuron, param_name_str,
                                           param, array_size);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_SetNeuronPtScalParam(int *i_node, int n_neuron, char *param_name,
                                 float val) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret =
        NESTGPU_instance->SetNeuronParam(i_node, n_neuron, param_name_str, val);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_SetNeuronPtArrayParam(int *i_node, int n_neuron, char *param_name,
                                  float *param, int array_size) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->SetNeuronParam(i_node, n_neuron, param_name_str,
                                           param, array_size);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_IsNeuronScalParam(int i_node, char *param_name) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);

    ret = NESTGPU_instance->IsNeuronScalParam(i_node, param_name_str);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_IsNeuronPortParam(int i_node, char *param_name) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);

    ret = NESTGPU_instance->IsNeuronPortParam(i_node, param_name_str);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_IsNeuronArrayParam(int i_node, char *param_name) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);

    ret = NESTGPU_instance->IsNeuronArrayParam(i_node, param_name_str);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_SetNeuronIntVar(int i_node, int n_neuron, char *var_name, int val) {
  int ret = 0;
  BEGIN_ERR_PROP {

    std::string var_name_str = std::string(var_name);
    ret =
        NESTGPU_instance->SetNeuronIntVar(i_node, n_neuron, var_name_str, val);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_SetNeuronScalVar(int i_node, int n_neuron, char *var_name,
                             float val) {
  int ret = 0;
  BEGIN_ERR_PROP {

    std::string var_name_str = std::string(var_name);
    ret = NESTGPU_instance->SetNeuronVar(i_node, n_neuron, var_name_str, val);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_SetNeuronArrayVar(int i_node, int n_neuron, char *var_name,
                              float *var, int array_size) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string var_name_str = std::string(var_name);
    ret = NESTGPU_instance->SetNeuronVar(i_node, n_neuron, var_name_str, var,
                                         array_size);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_SetNeuronPtIntVar(int *i_node, int n_neuron, char *var_name,
                              int val) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string var_name_str = std::string(var_name);
    ret =
        NESTGPU_instance->SetNeuronIntVar(i_node, n_neuron, var_name_str, val);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_SetNeuronPtScalVar(int *i_node, int n_neuron, char *var_name,
                               float val) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string var_name_str = std::string(var_name);
    ret = NESTGPU_instance->SetNeuronVar(i_node, n_neuron, var_name_str, val);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_SetNeuronPtArrayVar(int *i_node, int n_neuron, char *var_name,
                                float *var, int array_size) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string var_name_str = std::string(var_name);
    ret = NESTGPU_instance->SetNeuronVar(i_node, n_neuron, var_name_str, var,
                                         array_size);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_SetNeuronScalParamDistr(int i_node, int n_neuron,
                                    char *param_name) {
  int ret = 0;
  BEGIN_ERR_PROP {

    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->SetNeuronScalParamDistr(i_node, n_neuron,
                                                    param_name_str);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_SetNeuronScalVarDistr(int i_node, int n_neuron, char *var_name) {
  int ret = 0;
  BEGIN_ERR_PROP {

    std::string var_name_str = std::string(var_name);
    ret =
        NESTGPU_instance->SetNeuronScalVarDistr(i_node, n_neuron, var_name_str);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_SetNeuronPortParamDistr(int i_node, int n_neuron,
                                    char *param_name) {
  int ret = 0;
  BEGIN_ERR_PROP {

    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->SetNeuronPortParamDistr(i_node, n_neuron,
                                                    param_name_str);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_SetNeuronPortVarDistr(int i_node, int n_neuron, char *var_name) {
  int ret = 0;
  BEGIN_ERR_PROP {

    std::string var_name_str = std::string(var_name);
    ret =
        NESTGPU_instance->SetNeuronPortVarDistr(i_node, n_neuron, var_name_str);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_SetNeuronPtScalParamDistr(int *i_node, int n_neuron,
                                      char *param_name) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->SetNeuronPtScalParamDistr(i_node, n_neuron,
                                                      param_name_str);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_SetNeuronPtScalVarDistr(int *i_node, int n_neuron, char *var_name) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string var_name_str = std::string(var_name);
    ret = NESTGPU_instance->SetNeuronPtScalVarDistr(i_node, n_neuron,
                                                    var_name_str);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_SetNeuronPtPortParamDistr(int *i_node, int n_neuron,
                                      char *param_name) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->SetNeuronPtPortParamDistr(i_node, n_neuron,
                                                      param_name_str);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_SetNeuronPtPortVarDistr(int *i_node, int n_neuron, char *var_name) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string var_name_str = std::string(var_name);
    ret = NESTGPU_instance->SetNeuronPtPortVarDistr(i_node, n_neuron,
                                                    var_name_str);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_SetDistributionIntParam(char *param_name, int val) {
  int ret = 0;
  BEGIN_ERR_PROP {

    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->SetDistributionIntParam(param_name_str, val);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_SetDistributionScalParam(char *param_name, float val) {
  int ret = 0;
  BEGIN_ERR_PROP {

    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->SetDistributionScalParam(param_name_str, val);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_SetDistributionVectParam(char *param_name, float val, int i) {
  int ret = 0;
  BEGIN_ERR_PROP {

    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->SetDistributionVectParam(param_name_str, val, i);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_SetDistributionFloatPtParam(char *param_name, float *array_pt) {
  int ret = 0;
  BEGIN_ERR_PROP {

    std::string param_name_str = std::string(param_name);
    ret =
        NESTGPU_instance->SetDistributionFloatPtParam(param_name_str, array_pt);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_IsDistributionFloatParam(char *param_name) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);

    ret = NESTGPU_instance->IsDistributionFloatParam(param_name_str);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_IsNeuronIntVar(int i_node, char *var_name) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string var_name_str = std::string(var_name);

    ret = NESTGPU_instance->IsNeuronIntVar(i_node, var_name_str);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_IsNeuronScalVar(int i_node, char *var_name) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string var_name_str = std::string(var_name);

    ret = NESTGPU_instance->IsNeuronScalVar(i_node, var_name_str);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_IsNeuronPortVar(int i_node, char *var_name) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string var_name_str = std::string(var_name);

    ret = NESTGPU_instance->IsNeuronPortVar(i_node, var_name_str);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_IsNeuronArrayVar(int i_node, char *var_name) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string var_name_str = std::string(var_name);

    ret = NESTGPU_instance->IsNeuronArrayVar(i_node, var_name_str);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_GetNeuronParamSize(int i_node, char *param_name) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);

    ret = NESTGPU_instance->GetNeuronParamSize(i_node, param_name_str);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_GetNeuronVarSize(int i_node, char *var_name) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string var_name_str = std::string(var_name);

    ret = NESTGPU_instance->GetNeuronVarSize(i_node, var_name_str);
  }
  END_ERR_PROP return ret;
}

float *NESTGPU_GetNeuronParam(int i_node, int n_neuron, char *param_name) {
  float *ret = nullptr;
  BEGIN_ERR_PROP {

    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->GetNeuronParam(i_node, n_neuron, param_name_str);
  }
  END_ERR_PROP return ret;
}

float *NESTGPU_GetNeuronPtParam(int *i_node, int n_neuron, char *param_name) {
  float *ret = nullptr;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->GetNeuronParam(i_node, n_neuron, param_name_str);
  }
  END_ERR_PROP return ret;
}

float *NESTGPU_GetArrayParam(int i_node, char *param_name) {
  float *ret = nullptr;
  BEGIN_ERR_PROP {

    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->GetArrayParam(i_node, param_name_str);
  }
  END_ERR_PROP return ret;
}

int *NESTGPU_GetNeuronIntVar(int i_node, int n_neuron, char *param_name) {
  int *ret = nullptr;
  BEGIN_ERR_PROP {

    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->GetNeuronIntVar(i_node, n_neuron, param_name_str);
  }
  END_ERR_PROP return ret;
}

int *NESTGPU_GetNeuronPtIntVar(int *i_node, int n_neuron, char *param_name) {
  int *ret = nullptr;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->GetNeuronIntVar(i_node, n_neuron, param_name_str);
  }
  END_ERR_PROP return ret;
}

float *NESTGPU_GetNeuronVar(int i_node, int n_neuron, char *param_name) {
  float *ret = nullptr;
  BEGIN_ERR_PROP {

    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->GetNeuronVar(i_node, n_neuron, param_name_str);
  }
  END_ERR_PROP return ret;
}

float *NESTGPU_GetNeuronPtVar(int *i_node, int n_neuron, char *param_name) {
  float *ret = nullptr;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->GetNeuronVar(i_node, n_neuron, param_name_str);
  }
  END_ERR_PROP return ret;
}

float *NESTGPU_GetArrayVar(int i_node, char *var_name) {
  float *ret = nullptr;
  BEGIN_ERR_PROP {

    std::string var_name_str = std::string(var_name);
    ret = NESTGPU_instance->GetArrayVar(i_node, var_name_str);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_Calibrate() {
  int ret = 0;
  BEGIN_ERR_PROP { ret = NESTGPU_instance->Calibrate(); }
  END_ERR_PROP return ret;
}

int NESTGPU_Simulate() {
  int ret = 0;
  BEGIN_ERR_PROP { ret = NESTGPU_instance->Simulate(); }
  END_ERR_PROP return ret;
}

int NESTGPU_StartSimulation() {
  int ret = 0;
  BEGIN_ERR_PROP { ret = NESTGPU_instance->StartSimulation(); }
  END_ERR_PROP return ret;
}

int NESTGPU_SimulationStep() {
  int ret = 0;
  BEGIN_ERR_PROP { ret = NESTGPU_instance->SimulationStep(); }
  END_ERR_PROP return ret;
}

int NESTGPU_PrintTimers() {
  int ret = 0;
  BEGIN_ERR_PROP { ret = NESTGPU_instance->PrintTimers(); }
  END_ERR_PROP return ret;
}

int NESTGPU_ConnectMpiInit(int argc, char **argv) {
  int ret = 0;
  BEGIN_ERR_PROP {
    ret = NESTGPU_instance->ConnectMpiInit(argc, argv);
    capi.set_rank(NESTGPU_instance->HostId());
    capi.set_num_processes(NESTGPU_instance->HostNum());
  }
  END_ERR_PROP return ret;
}

int NESTGPU_FakeConnectMpiInit(int n_hosts, int this_host) {
  int ret = 0;
  BEGIN_ERR_PROP {
    ret = NESTGPU_instance->FakeConnectMpiInit(n_hosts, this_host);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_SetNHosts(int n_hosts) {
  int ret = 0;
  BEGIN_ERR_PROP { ret = NESTGPU_instance->setNHosts(n_hosts); }
  END_ERR_PROP return ret;
}

int NESTGPU_MpiFinalize() {
  int ret = 0;
  BEGIN_ERR_PROP { ret = NESTGPU_instance->MpiFinalize(); }
  END_ERR_PROP return ret;
}

int NESTGPU_HostId() {
  int ret = 0;
  BEGIN_ERR_PROP { ret = NESTGPU_instance->HostId(); }
  END_ERR_PROP return ret;
}

int NESTGPU_HostNum() {
  int ret = 0;
  BEGIN_ERR_PROP { ret = NESTGPU_instance->HostNum(); }
  END_ERR_PROP return ret;
}

size_t NESTGPU_getCUDAMemHostUsed() {
  size_t ret = 0;
  BEGIN_ERR_PROP { ret = NESTGPU_instance->getCUDAMemHostUsed(); }
  END_ERR_PROP return ret;
}

size_t NESTGPU_getCUDAMemHostPeak() {
  size_t ret = 0;
  BEGIN_ERR_PROP { ret = NESTGPU_instance->getCUDAMemHostPeak(); }
  END_ERR_PROP return ret;
}

size_t NESTGPU_getCUDAMemTotal() {
  size_t ret = 0;
  BEGIN_ERR_PROP { ret = NESTGPU_instance->getCUDAMemTotal(); }
  END_ERR_PROP return ret;
}

size_t NESTGPU_getCUDAMemFree() {
  size_t ret = 0;
  BEGIN_ERR_PROP { ret = NESTGPU_instance->getCUDAMemFree(); }
  END_ERR_PROP return ret;
}

unsigned int *NESTGPU_RandomInt(size_t n) {
  unsigned int *ret = nullptr;
  BEGIN_ERR_PROP { ret = NESTGPU_instance->RandomInt(n); }
  END_ERR_PROP return ret;
}

float *NESTGPU_RandomUniform(size_t n) {
  float *ret = nullptr;
  BEGIN_ERR_PROP { ret = NESTGPU_instance->RandomUniform(n); }
  END_ERR_PROP return ret;
}

float *NESTGPU_RandomNormal(size_t n, float mean, float stddev) {
  float *ret = nullptr;
  BEGIN_ERR_PROP { ret = NESTGPU_instance->RandomNormal(n, mean, stddev); }
  END_ERR_PROP return ret;
}

float *NESTGPU_RandomLognormalClipped(size_t n, float mean, float stddev,
                                      float vmin, float vmax, float vstep) {
  float *ret = nullptr;
  BEGIN_ERR_PROP {
    ret = NESTGPU_instance->RandomLognormalClipped(n, mean, stddev, vmin, vmax,
                                                   vstep);
  }
  END_ERR_PROP return ret;
}

float *NESTGPU_RandomNormalClipped(size_t n, float mean, float stddev,
                                   float vmin, float vmax, float vstep) {
  float *ret = nullptr;
  BEGIN_ERR_PROP {
    ret = NESTGPU_instance->RandomNormalClipped(n, mean, stddev, vmin, vmax,
                                                vstep);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_ConnSpecInit() {
  int ret = 0;
  BEGIN_ERR_PROP { ret = ConnSpec_instance.Init(); }
  END_ERR_PROP return ret;
}

int NESTGPU_SetConnSpecParam(char *param_name, int value) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = ConnSpec_instance.SetParam(param_name_str, value);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_ConnSpecIsParam(char *param_name) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = ConnSpec::IsParam(param_name_str);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_SynSpecInit() {
  int ret = 0;
  BEGIN_ERR_PROP { ret = SynSpec_instance.Init(); }
  END_ERR_PROP return ret;
}

int NESTGPU_SetSynSpecIntParam(char *param_name, int value) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = SynSpec_instance.SetParam(param_name_str, value);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_SetSynSpecFloatParam(char *param_name, float value) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = SynSpec_instance.SetParam(param_name_str, value);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_SetSynSpecFloatPtParam(char *param_name, float *array_pt) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = SynSpec_instance.SetParam(param_name_str, array_pt);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_SynSpecIsIntParam(char *param_name) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = SynSpec_instance.IsIntParam(param_name_str);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_SynSpecIsFloatParam(char *param_name) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = SynSpec_instance.IsFloatParam(param_name_str);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_SynSpecIsFloatPtParam(char *param_name) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = SynSpec_instance.IsFloatPtParam(param_name_str);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_ConnectSeqSeq(uint i_source, uint n_source, uint i_target,
                          uint n_target) {
  int ret = 0;
  BEGIN_ERR_PROP {
    ret = NESTGPU_instance->Connect(i_source, n_source, i_target, n_target,
                                    ConnSpec_instance, SynSpec_instance);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_ConnectSeqGroup(uint i_source, uint n_source, uint *i_target,
                            uint n_target) {
  int ret = 0;
  BEGIN_ERR_PROP {
    ret = NESTGPU_instance->Connect(i_source, n_source, i_target, n_target,
                                    ConnSpec_instance, SynSpec_instance);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_ConnectGroupSeq(uint *i_source, uint n_source, uint i_target,
                            uint n_target) {
  int ret = 0;
  BEGIN_ERR_PROP {
    ret = NESTGPU_instance->Connect(i_source, n_source, i_target, n_target,
                                    ConnSpec_instance, SynSpec_instance);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_ConnectGroupGroup(uint *i_source, uint n_source, uint *i_target,
                              uint n_target) {
  int ret = 0;
  BEGIN_ERR_PROP {
    ret = NESTGPU_instance->Connect(i_source, n_source, i_target, n_target,
                                    ConnSpec_instance, SynSpec_instance);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_RemoteConnectSeqSeq(int i_source_host, uint i_source, uint n_source,
                                int i_target_host, uint i_target, uint n_target,
                                int i_host_group) {
  int ret = 0;
  BEGIN_ERR_PROP {
    ret = NESTGPU_instance->RemoteConnect(
        i_source_host, i_source, n_source, i_target_host, i_target, n_target,
        i_host_group, ConnSpec_instance, SynSpec_instance);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_RemoteConnectSeqGroup(int i_source_host, uint i_source,
                                  uint n_source, int i_target_host,
                                  uint *i_target, uint n_target,
                                  int i_host_group) {
  int ret = 0;
  BEGIN_ERR_PROP {
    ret = NESTGPU_instance->RemoteConnect(
        i_source_host, i_source, n_source, i_target_host, i_target, n_target,
        i_host_group, ConnSpec_instance, SynSpec_instance);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_RemoteConnectGroupSeq(int i_source_host, uint *i_source,
                                  uint n_source, int i_target_host,
                                  uint i_target, uint n_target,
                                  int i_host_group) {
  int ret = 0;
  BEGIN_ERR_PROP {
    ret = NESTGPU_instance->RemoteConnect(
        i_source_host, i_source, n_source, i_target_host, i_target, n_target,
        i_host_group, ConnSpec_instance, SynSpec_instance);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_RemoteConnectGroupGroup(int i_source_host, uint *i_source,
                                    uint n_source, int i_target_host,
                                    uint *i_target, uint n_target,
                                    int i_host_group) {
  int ret = 0;
  BEGIN_ERR_PROP {
    ret = NESTGPU_instance->RemoteConnect(
        i_source_host, i_source, n_source, i_target_host, i_target, n_target,
        i_host_group, ConnSpec_instance, SynSpec_instance);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_CreateHostGroup(int *host_arr, int n_hosts) {
  int ret = 0;
  BEGIN_ERR_PROP { ret = NESTGPU_instance->CreateHostGroup(host_arr, n_hosts); }
  END_ERR_PROP return ret;
}

char **NESTGPU_GetIntVarNames(uint i_node) {
  char **ret = nullptr;
  BEGIN_ERR_PROP {
    std::vector<std::string> var_name_vect =
        NESTGPU_instance->GetIntVarNames(i_node);
    char **var_name_array =
        (char **)malloc(var_name_vect.size() * sizeof(char *));
    for (unsigned int i = 0; i < var_name_vect.size(); i++) {
      uint vl = var_name_vect[i].length() + 1;
      char *var_name = (char *)malloc((vl) * sizeof(char));

      strncpy(var_name, var_name_vect[i].c_str(), vl);
      var_name_array[i] = var_name;
    }
    ret = var_name_array;
  }
  END_ERR_PROP return ret;
}

char **NESTGPU_GetScalVarNames(uint i_node) {
  char **ret = nullptr;
  BEGIN_ERR_PROP {
    std::vector<std::string> var_name_vect =
        NESTGPU_instance->GetScalVarNames(i_node);
    char **var_name_array =
        (char **)malloc(var_name_vect.size() * sizeof(char *));
    for (unsigned int i = 0; i < var_name_vect.size(); i++) {
      uint vl = var_name_vect[i].length() + 1;
      char *var_name = (char *)malloc((vl) * sizeof(char));

      strncpy(var_name, var_name_vect[i].c_str(), vl);
      var_name_array[i] = var_name;
    }
    ret = var_name_array;
  }
  END_ERR_PROP return ret;
}

int NESTGPU_GetNIntVar(uint i_node) {
  int ret = 0;
  BEGIN_ERR_PROP { ret = NESTGPU_instance->GetNIntVar(i_node); }
  END_ERR_PROP return ret;
}

int NESTGPU_GetNScalVar(uint i_node) {
  int ret = 0;
  BEGIN_ERR_PROP { ret = NESTGPU_instance->GetNScalVar(i_node); }
  END_ERR_PROP return ret;
}

char **NESTGPU_GetPortVarNames(uint i_node) {
  char **ret = nullptr;
  BEGIN_ERR_PROP {
    std::vector<std::string> var_name_vect =
        NESTGPU_instance->GetPortVarNames(i_node);
    char **var_name_array =
        (char **)malloc(var_name_vect.size() * sizeof(char *));
    for (unsigned int i = 0; i < var_name_vect.size(); i++) {
      uint vl = var_name_vect[i].length() + 1;
      char *var_name = (char *)malloc((vl) * sizeof(char));

      strncpy(var_name, var_name_vect[i].c_str(), vl);
      var_name_array[i] = var_name;
    }
    ret = var_name_array;
  }
  END_ERR_PROP return ret;
}

int NESTGPU_GetNPortVar(uint i_node) {
  int ret = 0;
  BEGIN_ERR_PROP { ret = NESTGPU_instance->GetNPortVar(i_node); }
  END_ERR_PROP return ret;
}

char **NESTGPU_GetScalParamNames(uint i_node) {
  char **ret = nullptr;
  BEGIN_ERR_PROP {
    std::vector<std::string> var_name_vect =
        NESTGPU_instance->GetScalParamNames(i_node);
    char **var_name_array =
        (char **)malloc(var_name_vect.size() * sizeof(char *));
    for (unsigned int i = 0; i < var_name_vect.size(); i++) {
      uint vl = var_name_vect[i].length() + 1;
      char *var_name = (char *)malloc((vl) * sizeof(char));

      strncpy(var_name, var_name_vect[i].c_str(), vl);
      var_name_array[i] = var_name;
    }
    ret = var_name_array;
  }
  END_ERR_PROP return ret;
}

int NESTGPU_GetNScalParam(uint i_node) {
  int ret = 0;
  BEGIN_ERR_PROP { ret = NESTGPU_instance->GetNScalParam(i_node); }
  END_ERR_PROP return ret;
}

char **NESTGPU_GetGroupParamNames(uint i_node) {
  char **ret = nullptr;
  BEGIN_ERR_PROP {
    std::vector<std::string> var_name_vect =
        NESTGPU_instance->GetGroupParamNames(i_node);
    char **var_name_array =
        (char **)malloc(var_name_vect.size() * sizeof(char *));
    for (unsigned int i = 0; i < var_name_vect.size(); i++) {
      uint vl = var_name_vect[i].length() + 1;
      char *var_name = (char *)malloc((vl) * sizeof(char));

      strncpy(var_name, var_name_vect[i].c_str(), vl);
      var_name_array[i] = var_name;
    }
    ret = var_name_array;
  }
  END_ERR_PROP return ret;
}

int NESTGPU_GetNGroupParam(uint i_node) {
  int ret = 0;
  BEGIN_ERR_PROP { ret = NESTGPU_instance->GetNGroupParam(i_node); }
  END_ERR_PROP return ret;
}

char **NESTGPU_GetPortParamNames(uint i_node) {
  char **ret = nullptr;
  BEGIN_ERR_PROP {
    std::vector<std::string> var_name_vect =
        NESTGPU_instance->GetPortParamNames(i_node);
    char **var_name_array =
        (char **)malloc(var_name_vect.size() * sizeof(char *));
    for (unsigned int i = 0; i < var_name_vect.size(); i++) {
      uint vl = var_name_vect[i].length() + 1;
      char *var_name = (char *)malloc((vl) * sizeof(char));

      strncpy(var_name, var_name_vect[i].c_str(), vl);
      var_name_array[i] = var_name;
    }
    ret = var_name_array;
  }
  END_ERR_PROP return ret;
}

int NESTGPU_GetNPortParam(uint i_node) {
  int ret = 0;
  BEGIN_ERR_PROP { ret = NESTGPU_instance->GetNPortParam(i_node); }
  END_ERR_PROP return ret;
}

char **NESTGPU_GetArrayParamNames(uint i_node) {
  char **ret = nullptr;
  BEGIN_ERR_PROP {
    std::vector<std::string> var_name_vect =
        NESTGPU_instance->GetArrayParamNames(i_node);
    char **var_name_array =
        (char **)malloc(var_name_vect.size() * sizeof(char *));
    for (unsigned int i = 0; i < var_name_vect.size(); i++) {
      uint vl = var_name_vect[i].length() + 1;
      char *var_name = (char *)malloc((vl) * sizeof(char));

      strncpy(var_name, var_name_vect[i].c_str(), vl);
      var_name_array[i] = var_name;
    }
    ret = var_name_array;
  }
  END_ERR_PROP return ret;
}

int NESTGPU_GetNArrayParam(uint i_node) {
  int ret = 0;
  BEGIN_ERR_PROP { ret = NESTGPU_instance->GetNArrayParam(i_node); }
  END_ERR_PROP return ret;
}

char **NESTGPU_GetArrayVarNames(uint i_node) {
  char **ret = nullptr;
  BEGIN_ERR_PROP {
    std::vector<std::string> var_name_vect =
        NESTGPU_instance->GetArrayVarNames(i_node);
    char **var_name_array =
        (char **)malloc(var_name_vect.size() * sizeof(char *));
    for (unsigned int i = 0; i < var_name_vect.size(); i++) {
      uint vl = var_name_vect[i].length() + 1;
      char *var_name = (char *)malloc((vl) * sizeof(char));

      strncpy(var_name, var_name_vect[i].c_str(), vl);
      var_name_array[i] = var_name;
    }
    ret = var_name_array;
  }
  END_ERR_PROP return ret;
}

int NESTGPU_GetNArrayVar(uint i_node) {
  int ret = 0;
  BEGIN_ERR_PROP { ret = NESTGPU_instance->GetNArrayVar(i_node); }
  END_ERR_PROP return ret;
}

int64_t *NESTGPU_GetSeqSeqConnections(uint i_source, uint n_source,
                                      uint i_target, uint n_target,
                                      int syn_group, int64_t *n_conn) {
  int64_t *ret = nullptr;
  BEGIN_ERR_PROP {
    ret = NESTGPU_instance->GetConnections(i_source, n_source, i_target,
                                           n_target, syn_group, n_conn);
  }
  END_ERR_PROP return ret;
}

int64_t *NESTGPU_GetSeqGroupConnections(uint i_source, uint n_source,
                                        uint *i_target_pt, uint n_target,
                                        int syn_group, int64_t *n_conn) {
  int64_t *ret = nullptr;
  BEGIN_ERR_PROP {
    ret = NESTGPU_instance->GetConnections(i_source, n_source, i_target_pt,
                                           n_target, syn_group, n_conn);
  }
  END_ERR_PROP return ret;
}

int64_t *NESTGPU_GetGroupSeqConnections(uint *i_source_pt, uint n_source,
                                        uint i_target, uint n_target,
                                        int syn_group, int64_t *n_conn) {
  int64_t *ret = nullptr;
  BEGIN_ERR_PROP {
    ret = NESTGPU_instance->GetConnections(i_source_pt, n_source, i_target,
                                           n_target, syn_group, n_conn);
  }
  END_ERR_PROP return ret;
}

int64_t *NESTGPU_GetGroupGroupConnections(uint *i_source_pt, uint n_source,
                                          uint *i_target_pt, uint n_target,
                                          int syn_group, int64_t *n_conn) {
  int64_t *ret = nullptr;
  BEGIN_ERR_PROP {
    ret = NESTGPU_instance->GetConnections(i_source_pt, n_source, i_target_pt,
                                           n_target, syn_group, n_conn);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_GetConnectionStatus(int64_t *conn_ids, int64_t n_conn,
                                uint *i_source, uint *i_target, int *port,
                                int *syn_group, float *delay, float *weight) {
  int ret = 0;
  BEGIN_ERR_PROP {
    ret = NESTGPU_instance->GetConnectionStatus(
        conn_ids, n_conn, i_source, i_target, port, syn_group, delay, weight);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_IsConnectionFloatParam(char *param_name) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->IsConnectionFloatParam(param_name_str);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_IsConnectionIntParam(char *param_name) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->IsConnectionIntParam(param_name_str);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_GetConnectionFloatParam(int64_t *conn_ids, int64_t n_conn,
                                    float *param_arr, char *param_name) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->GetConnectionFloatParam(conn_ids, n_conn, param_arr,
                                                    param_name_str);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_GetConnectionIntParam(int64_t *conn_ids, int64_t n_conn,
                                  int *param_arr, char *param_name) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->GetConnectionIntParam(conn_ids, n_conn, param_arr,
                                                  param_name_str);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_SetConnectionFloatParamDistr(int64_t *conn_ids, int64_t n_conn,
                                         char *param_name) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->SetConnectionFloatParamDistr(conn_ids, n_conn,
                                                         param_name_str);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_SetConnectionIntParamArr(int64_t *conn_ids, int64_t n_conn,
                                     int *param_arr, char *param_name) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->SetConnectionIntParamArr(conn_ids, n_conn,
                                                     param_arr, param_name_str);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_SetConnectionFloatParam(int64_t *conn_ids, int64_t n_conn,
                                    float val, char *param_name) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->SetConnectionFloatParam(conn_ids, n_conn, val,
                                                    param_name_str);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_SetConnectionIntParam(int64_t *conn_ids, int64_t n_conn, int val,
                                  char *param_name) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->SetConnectionIntParam(conn_ids, n_conn, val,
                                                  param_name_str);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_CreateSynGroup(char *model_name) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string model_name_str = std::string(model_name);
    ret = NESTGPU_instance->CreateSynGroup(model_name_str);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_GetSynGroupNParam(int i_syn_group) {
  int ret = 0;
  BEGIN_ERR_PROP { ret = NESTGPU_instance->GetSynGroupNParam(i_syn_group); }
  END_ERR_PROP return ret;
}

char **NESTGPU_GetSynGroupParamNames(int i_syn_group) {
  char **ret = nullptr;
  BEGIN_ERR_PROP {
    std::vector<std::string> name_vect =
        NESTGPU_instance->GetSynGroupParamNames(i_syn_group);
    char **name_array = (char **)malloc(name_vect.size() * sizeof(char *));
    for (unsigned int i = 0; i < name_vect.size(); i++) {
      uint vl = name_vect[i].length() + 1;
      char *param_name = (char *)malloc((vl) * sizeof(char));

      strncpy(param_name, name_vect[i].c_str(), vl);
      name_array[i] = param_name;
    }
    ret = name_array;
  }
  END_ERR_PROP return ret;
}

int NESTGPU_IsSynGroupParam(int i_syn_group, char *param_name) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);

    ret = NESTGPU_instance->IsSynGroupParam(i_syn_group, param_name_str);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_GetSynGroupParamIdx(int i_syn_group, char *param_name) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);

    ret = NESTGPU_instance->GetSynGroupParamIdx(i_syn_group, param_name_str);
  }
  END_ERR_PROP return ret;
}

float NESTGPU_GetSynGroupParam(int i_syn_group, char *param_name) {
  float ret = 0.0;
  BEGIN_ERR_PROP {

    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->GetSynGroupParam(i_syn_group, param_name_str);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_SetSynGroupParam(int i_syn_group, char *param_name, float val) {
  float ret = 0.0;
  BEGIN_ERR_PROP {

    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->SetSynGroupParam(i_syn_group, param_name_str, val);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_ActivateSpikeCount(uint i_node, int n_node) {
  int ret = 0;
  BEGIN_ERR_PROP { ret = NESTGPU_instance->ActivateSpikeCount(i_node, n_node); }
  END_ERR_PROP return ret;
}

int NESTGPU_ActivateRecSpikeTimes(uint i_node, int n_node,
                                  int max_n_rec_spike_times) {
  int ret = 0;
  BEGIN_ERR_PROP {

    ret = NESTGPU_instance->ActivateRecSpikeTimes(i_node, n_node,
                                                  max_n_rec_spike_times);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_SetRecSpikeTimesStep(uint i_node, int n_node,
                                 int rec_spike_times_step) {
  int ret = 0;
  BEGIN_ERR_PROP {

    ret = NESTGPU_instance->SetRecSpikeTimesStep(i_node, n_node,
                                                 rec_spike_times_step);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_GetNRecSpikeTimes(uint i_node) {
  int ret = 0;
  BEGIN_ERR_PROP { ret = NESTGPU_instance->GetNRecSpikeTimes(i_node); }
  END_ERR_PROP return ret;
}

int NESTGPU_GetRecSpikeTimes(uint i_node, int n_node, int **n_spike_times_pt,
                             float ***spike_times_pt) {
  int ret = 0;
  BEGIN_ERR_PROP {
    ret = NESTGPU_instance->GetRecSpikeTimes(i_node, n_node, n_spike_times_pt,
                                             spike_times_pt);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_PushSpikesToNodes(int n_spikes, int *node_id) {
  int ret = 0;
  BEGIN_ERR_PROP {

    ret = NESTGPU_instance->PushSpikesToNodes(n_spikes, node_id);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_GetExtNeuronInputSpikes(int *n_spikes, int **node, int **port,
                                    float **spike_mul, int include_zeros) {
  int ret = 0;
  BEGIN_ERR_PROP {

    ret = NESTGPU_instance->GetExtNeuronInputSpikes(
        n_spikes, node, port, spike_mul, include_zeros > 0);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_SetNeuronGroupParam(uint i_node, int n_node, char *param_name,
                                float val) {
  float ret = 0.0;
  BEGIN_ERR_PROP {

    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->SetNeuronGroupParam(i_node, n_node, param_name_str,
                                                val);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_IsNeuronGroupParam(uint i_node, char *param_name) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);

    ret = NESTGPU_instance->IsNeuronGroupParam(i_node, param_name_str);
  }
  END_ERR_PROP return ret;
}

float NESTGPU_GetNeuronGroupParam(uint i_node, char *param_name) {
  float ret = 0;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);

    ret = NESTGPU_instance->GetNeuronGroupParam(i_node, param_name_str);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_GetNBoolParam() {
  int ret = 0;
  BEGIN_ERR_PROP { ret = NESTGPU_instance->GetNBoolParam(); }
  END_ERR_PROP return ret;
}

char **NESTGPU_GetBoolParamNames() {
  char **ret = nullptr;
  BEGIN_ERR_PROP {
    std::vector<std::string> name_vect = NESTGPU_instance->GetBoolParamNames();
    char **name_array = (char **)malloc(name_vect.size() * sizeof(char *));
    for (unsigned int i = 0; i < name_vect.size(); i++) {
      uint vl = name_vect[i].length() + 1;
      char *param_name = (char *)malloc((vl) * sizeof(char));

      strncpy(param_name, name_vect[i].c_str(), vl);
      name_array[i] = param_name;
    }
    ret = name_array;
  }
  END_ERR_PROP return ret;
}

int NESTGPU_IsBoolParam(char *param_name) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);

    ret = NESTGPU_instance->IsBoolParam(param_name_str);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_GetBoolParamIdx(char *param_name) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);

    ret = NESTGPU_instance->GetBoolParamIdx(param_name_str);
  }
  END_ERR_PROP return ret;
}

bool NESTGPU_GetBoolParam(char *param_name) {
  bool ret = true;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);

    ret = NESTGPU_instance->GetBoolParam(param_name_str);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_SetBoolParam(char *param_name, bool val) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);

    ret = NESTGPU_instance->SetBoolParam(param_name_str, val);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_GetNFloatParam() {
  int ret = 0;
  BEGIN_ERR_PROP { ret = NESTGPU_instance->GetNFloatParam(); }
  END_ERR_PROP return ret;
}

char **NESTGPU_GetFloatParamNames() {
  char **ret = nullptr;
  BEGIN_ERR_PROP {
    std::vector<std::string> name_vect = NESTGPU_instance->GetFloatParamNames();
    char **name_array = (char **)malloc(name_vect.size() * sizeof(char *));
    for (unsigned int i = 0; i < name_vect.size(); i++) {
      uint vl = name_vect[i].length() + 1;
      char *param_name = (char *)malloc((vl) * sizeof(char));

      strncpy(param_name, name_vect[i].c_str(), vl);
      name_array[i] = param_name;
    }
    ret = name_array;
  }
  END_ERR_PROP return ret;
}

int NESTGPU_IsFloatParam(char *param_name) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);

    ret = NESTGPU_instance->IsFloatParam(param_name_str);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_GetFloatParamIdx(char *param_name) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);

    ret = NESTGPU_instance->GetFloatParamIdx(param_name_str);
  }
  END_ERR_PROP return ret;
}

float NESTGPU_GetFloatParam(char *param_name) {
  float ret = 0.0;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);

    ret = NESTGPU_instance->GetFloatParam(param_name_str);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_SetFloatParam(char *param_name, float val) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);

    ret = NESTGPU_instance->SetFloatParam(param_name_str, val);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_GetNIntParam() {
  int ret = 0;
  BEGIN_ERR_PROP { ret = NESTGPU_instance->GetNIntParam(); }
  END_ERR_PROP return ret;
}

char **NESTGPU_GetIntParamNames() {
  char **ret = nullptr;
  BEGIN_ERR_PROP {
    std::vector<std::string> name_vect = NESTGPU_instance->GetIntParamNames();
    char **name_array = (char **)malloc(name_vect.size() * sizeof(char *));
    for (unsigned int i = 0; i < name_vect.size(); i++) {
      uint vl = name_vect[i].length() + 1;
      char *param_name = (char *)malloc((vl) * sizeof(char));

      strncpy(param_name, name_vect[i].c_str(), vl);
      name_array[i] = param_name;
    }
    ret = name_array;
  }
  END_ERR_PROP return ret;
}

int NESTGPU_IsIntParam(char *param_name) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);

    ret = NESTGPU_instance->IsIntParam(param_name_str);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_GetIntParamIdx(char *param_name) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);

    ret = NESTGPU_instance->GetIntParamIdx(param_name_str);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_GetIntParam(char *param_name) {
  int ret = 0;
  BEGIN_ERR_PROP {

    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->GetIntParam(param_name_str);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_SetIntParam(char *param_name, int val) {
  int ret = 0;
  BEGIN_ERR_PROP {

    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->SetIntParam(param_name_str, val);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_RemoteCreate(int i_host, char *model_name, int n_neuron,
                         int n_port) {
  int ret = 0;
  BEGIN_ERR_PROP {
    std::string model_name_str = std::string(model_name);
    RemoteNodeSeq rneur = NESTGPU_instance->RemoteCreate(i_host, model_name_str,
                                                         n_neuron, n_port);
    ret = rneur.node_seq[0];
  }
  END_ERR_PROP return ret;
}

int NESTGPU_ConnectDistributedFixedIndegreeSeqSeq(
    int *source_host_arr, int n_source_host, uint *source_arr,
    uint *n_source_arr, int *target_host_arr, int n_target_host,
    uint *target_arr, uint *n_target_arr, int indegree, int i_host_group) {
  int ret = 0;
  BEGIN_ERR_PROP {
    ret = NESTGPU_instance->ConnectDistributedFixedIndegree(
        source_host_arr, n_source_host, source_arr, n_source_arr,
        target_host_arr, n_target_host, target_arr, n_target_arr, indegree,
        i_host_group, SynSpec_instance);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_ConnectDistributedFixedIndegreeSeqGroup(
    int *source_host_arr, int n_source_host, uint *source_arr,
    uint *n_source_arr, int *target_host_arr, int n_target_host,
    uint **target_arr, uint *n_target_arr, int indegree, int i_host_group) {
  int ret = 0;
  BEGIN_ERR_PROP {
    ret = NESTGPU_instance->ConnectDistributedFixedIndegree(
        source_host_arr, n_source_host, source_arr, n_source_arr,
        target_host_arr, n_target_host, target_arr, n_target_arr, indegree,
        i_host_group, SynSpec_instance);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_ConnectDistributedFixedIndegreeGroupSeq(
    int *source_host_arr, int n_source_host, uint **source_arr,
    uint *n_source_arr, int *target_host_arr, int n_target_host,
    uint *target_arr, uint *n_target_arr, int indegree, int i_host_group) {
  int ret = 0;
  BEGIN_ERR_PROP {
    ret = NESTGPU_instance->ConnectDistributedFixedIndegree(
        source_host_arr, n_source_host, source_arr, n_source_arr,
        target_host_arr, n_target_host, target_arr, n_target_arr, indegree,
        i_host_group, SynSpec_instance);
  }
  END_ERR_PROP return ret;
}

int NESTGPU_ConnectDistributedFixedIndegreeGroupGroup(
    int *source_host_arr, int n_source_host, uint **source_arr,
    uint *n_source_arr, int *target_host_arr, int n_target_host,
    uint **target_arr, uint *n_target_arr, int indegree, int i_host_group) {
  int ret = 0;
  BEGIN_ERR_PROP {
    ret = NESTGPU_instance->ConnectDistributedFixedIndegree(
        source_host_arr, n_source_host, source_arr, n_source_arr,
        target_host_arr, n_target_host, target_arr, n_target_arr, indegree,
        i_host_group, SynSpec_instance);
  }
  END_ERR_PROP return ret;
}

sapi::ParameterNamesPairArray *get_parameter_names() {
  BEGIN_ERR_PROP { return capi.get_parameter_names(); }
  END_ERR_PROP
  return nullptr;
}

bool free_gc() {
  BEGIN_ERR_PROP {
    capi.free_gc();
    return true;
  }
  END_ERR_PROP
  return false;
}

bool free_view_gc() {
  BEGIN_ERR_PROP {
    capi.free_view_gc();
    return true;
  }
  END_ERR_PROP
  return false;
}

sapi::OptionalIndex get_rank() {
  sapi::OptionalIndex opt;
  BEGIN_ERR_PROP {
    opt.second_ = static_cast<std::size_t>(capi.get_rank());
    opt.first_ = true;
    return opt;
  }
  END_ERR_PROP
  opt.first_ = false;
  return opt;
}

sapi::OptionalIndex get_num_processes() {
  sapi::OptionalIndex opt;
  BEGIN_ERR_PROP {
    opt.second_ = static_cast<std::size_t>(capi.get_num_processes());
    opt.first_ = true;
    return opt;
  }
  END_ERR_PROP
  opt.first_ = false;
  return opt;
}

sapi::OptionalIndex get_num_threads() {
  sapi::OptionalIndex opt;
  BEGIN_ERR_PROP {
    opt.second_ = static_cast<std::size_t>(capi.get_num_threads());
    opt.first_ = true;
    return opt;
  }
  END_ERR_PROP
  opt.first_ = false;
  return opt;
}

bool set_num_threads(sapi::vp_t num_threads) {
  BEGIN_ERR_PROP {
    capi.set_num_threads(num_threads);
    return true;
  }
  END_ERR_PROP
  return false;
}

sapi::OptionalIndex get_rng_seed() {
  sapi::OptionalIndex opt;
  BEGIN_ERR_PROP {
    opt.second_ = static_cast<std::size_t>(capi.get_rng_seed());
    opt.first_ = true;
    return opt;
  }
  END_ERR_PROP
  opt.first_ = false;
  return opt;
}

bool set_rng_seed(sapi::rng_seed_t seed) {
  BEGIN_ERR_PROP {
    capi.set_rng_seed(seed);
    NESTGPU_instance->SetRandomSeed(seed);
    return true;
  }
  END_ERR_PROP
  return false;
}

sapi::CharArray *get_rng_type() {
  BEGIN_ERR_PROP { return capi.get_rng_type(); }
  END_ERR_PROP
  return nullptr;
}

bool set_rng_type(const sapi::CharArray &rng_type) {
  BEGIN_ERR_PROP {
    capi.set_rng_type(rng_type);
    return true;
  }
  END_ERR_PROP
  return false;
}

bool generate_tile_grid(const sapi::NestedTileIdxArray &rank_tiles_ownership,
                        const sapi::GPStruct &grid_parameters) {
  BEGIN_ERR_PROP {
    capi.generate_tile_grid(rank_tiles_ownership, grid_parameters);
    return true;
  }
  END_ERR_PROP
  return false;
}

sapi::RankNodeSequenceMap
update_node_counts_per_rank(const sapi::NodeCountVector &nodes_per_rank,
                            const sapi::CharArray &model_name,
                            const int num_ports) {
  sapi::RankNodeSequenceMap rns_map;
  const auto model = sapi::charray_to_string(model_name);
  if (1 < capi.get_num_processes()) {
    sapi::vp_t rank = 0;
    for (const auto &node_count : nodes_per_rank) {
      if (0 < node_count) {
        const auto remote_nodeseq = NESTGPU_instance->RemoteCreate(
            static_cast<int>(rank), model, static_cast<inode_t>(node_count),
            num_ports);

        const auto emplace_res = rns_map.emplace(
            rank, sapi::NodeSequence(remote_nodeseq.node_seq.i0,
                                     remote_nodeseq.node_seq.n));

        if (!emplace_res.second)
          throw std::runtime_error("Corrupted rank node sequence map");
      }
      ++rank;
    }
  } else {
    const auto local_rank = capi.get_rank();
    const auto nodeseq = NESTGPU_instance->Create(
        model, static_cast<inode_t>(nodes_per_rank[local_rank]), num_ports);

    const auto emplace_res =
        rns_map.emplace(local_rank, sapi::NodeSequence(nodeseq.i0, nodeseq.n));

    if (!emplace_res.second)
      throw std::runtime_error("Corrupted rank node sequence map");
  }

  return rns_map;
}

sapi::PairT<bool, sapi::SpatialNodeSequence> generate_nodes_in_grid(
    const sapi::CharArray &model_name, sapi::largenodeidx_t num_nodes,
    const sapi::TileIdxArray &target_tiles, int num_ports,
    uint8_t grid_distribution_mode, uint8_t tile_distribution_mode) {
  sapi::PairT<bool, sapi::SpatialNodeSequence> pair;
  BEGIN_ERR_PROP {
    const auto nodes_per_rank = capi.generate_nodes_in_grid(
        num_nodes, target_tiles, grid_distribution_mode);

    auto rank_map =
        update_node_counts_per_rank(nodes_per_rank, model_name, num_ports);

    if (const auto search = rank_map.find(capi.get_rank());
        search != rank_map.end()) {
      pair.second_.second_ = search->second.first;
      pair.second_.third_ = search->second.second;
    } else {
      pair.second_.second_ = -1;
      pair.second_.third_ = -1;
    }

    pair.second_.first_ =
        capi.generate_nodes_in_tiles(rank_map, tile_distribution_mode);

    const auto emplace_res = spatial_node_sequence_map.emplace(
        pair.second_.first_, std::move(rank_map));
    if (!emplace_res.second)
      throw std::runtime_error("Corrupted spatial node sequence map");

    pair.first_ = true;
    return pair;
  }
  END_ERR_PROP
  pair.first_ = false;
  return pair;
}

sapi::TripletT<bool, sapi::SpatialNodeSequence, sapi::PositionViewStruct *>
insert_positions_in_grid(const sapi::CharArray &model_name,
                         const sapi::PositionViewStruct &positions,
                         int num_ports) {
  sapi::TripletT<bool, sapi::SpatialNodeSequence, sapi::PositionViewStruct *>
      triplet;
  BEGIN_ERR_PROP {
    const auto [nodes_per_rank, leftovers] =
        capi.insert_positions_in_grid(positions);

    if (leftovers->coord_count_ == positions.coord_count_)
      throw std::runtime_error("Failed to insert any provided node position");

    auto rank_map =
        update_node_counts_per_rank(nodes_per_rank, model_name, num_ports);

    if (const auto search = rank_map.find(capi.get_rank());
        search != rank_map.end()) {
      triplet.second_.second_ = search->second.first;
      triplet.second_.third_ = search->second.second;
    } else {
      triplet.second_.second_ = -1;
      triplet.second_.third_ = -1;
    }

    triplet.second_.first_ = capi.insert_positions_in_tiles(rank_map);

    const auto emplace_res = spatial_node_sequence_map.emplace(
        triplet.second_.first_, std::move(rank_map));
    if (!emplace_res.second)
      throw std::runtime_error("Corrupted spatial node sequence map");

    triplet.first_ = true;
    triplet.third_ = leftovers;
    return triplet;
  }
  END_ERR_PROP
  triplet.first_ = false;
  triplet.third_ = nullptr;
  return triplet;
}

void create_spatial_connections(const int source_rank, const int target_rank,
                                sapi::ConnectionVectors &conn_vec,
                                const bool remote, const bool remote_source) {
  if (conn_vec.sizes_ < 1)
    return;

  ConnSpec_instance.rule_ = ConnectionRules::ASSIGNED_CONNECTIONS;
  ConnSpec_instance.use_all_remote_source_nodes_ = false;
  ConnSpec_instance.assigned_connections_ = conn_vec.sizes_;

  if (remote) {
    for (auto &source : conn_vec.connection_sources_)
      source -= conn_vec.bounds_.first_source_index_;

    NESTGPU_instance->manual_assign_connections(
        conn_vec.connection_sources_.data(),
        conn_vec.connection_targets_.data(),
        conn_vec.connection_weights_.data(), conn_vec.connection_delays_.data(),
        conn_vec.sizes_, remote_source, SynSpec_instance);

    for (auto &source : conn_vec.connection_sources_)
      source += conn_vec.bounds_.first_source_index_;

    NESTGPU_instance->RemoteConnect(
        source_rank, conn_vec.bounds_.first_source_index_,
        conn_vec.bounds_.last_source_index_ -
            conn_vec.bounds_.first_source_index_ + 1,
        target_rank, conn_vec.bounds_.first_target_index_,
        conn_vec.bounds_.last_target_index_ -
            conn_vec.bounds_.last_target_index_ + 1,
        -1, ConnSpec_instance, SynSpec_instance);
  } else {
    NESTGPU_instance->manual_assign_connections(
        conn_vec.connection_sources_.data(),
        conn_vec.connection_targets_.data(),
        conn_vec.connection_weights_.data(), conn_vec.connection_delays_.data(),
        conn_vec.sizes_, false, SynSpec_instance);

    NESTGPU_instance->Connect(conn_vec.bounds_.first_source_index_,
                              conn_vec.bounds_.last_source_index_ -
                                  conn_vec.bounds_.first_source_index_ + 1,
                              conn_vec.bounds_.first_target_index_,
                              conn_vec.bounds_.last_target_index_ -
                                  conn_vec.bounds_.last_target_index_ + 1,
                              ConnSpec_instance, SynSpec_instance);
  }
}

sapi::OptionalIndex
compute_spatial_connections(std::size_t source_index, std::size_t target_index,
                            const sapi::MPStruct &mask_parameters,
                            const sapi::CPStruct &connection_parameters) {
  sapi::OptionalIndex opt;
  BEGIN_ERR_PROP {
    const auto [conn_index, conn_map_ptr] = capi.compute_spatial_connections(
        source_index, target_index, mask_parameters, connection_parameters);

    try {
      if (NESTGPU_instance->GetBoolParam("check_node_maps")) {
        const std::array<std::unordered_map<
                             std::size_t, sapi::RankNodeSequenceMap>::iterator,
                         2>
            dtns_it_array = {spatial_node_sequence_map.find(source_index),
                             spatial_node_sequence_map.find(target_index)};
        if (std::any_of(dtns_it_array.cbegin(), dtns_it_array.cend(),
                        [&](const auto &it) {
                          return it == spatial_node_sequence_map.end();
                        }))
          throw std::runtime_error("Corrupted spatial node sequence map");

        uint8_t idx = 0;
        const auto host_str = "KERNEL RANK " + std::to_string(capi.get_rank());
        const auto check_str =
            host_str + std::string(": checking spatial node sequence maps on ");
        for (const auto dtns_it : dtns_it_array) {
          const auto check_incoming = idx == 0;
          if (const auto local_ns = dtns_it->second.find(capi.get_rank());
              local_ns != dtns_it->second.end()) {
            std::cout << check_str +
                             std::string(check_incoming
                                             ? "incoming connections\n"
                                             : "outgoing connections\n");
            const auto first_local = local_ns->second.first;
            const auto one_after_last_local =
                first_local + local_ns->second.second;
            for (const auto &[rank, conn_map] :
                 check_incoming ? conn_map_ptr->incoming_connections_
                                : conn_map_ptr->outgoing_connections_) {
              const auto rank_ns =
                  &dtns_it_array[(idx + 1) % 2]->second.at(rank);
              const auto first_remote = rank_ns->first;
              const auto one_after_last_remote = first_remote + rank_ns->second;
              for (const auto &conn_vec : conn_map.partitioned_connections_) {
                for (std::size_t idx = 0; idx < conn_vec.sizes_; ++idx) {
                  const auto source = static_cast<sapi::nodeidx_t>(
                      conn_vec.connection_sources_[idx]);
                  const auto target = static_cast<sapi::nodeidx_t>(
                      conn_vec.connection_targets_[idx]);

                  if (check_incoming) {
                    if (target < first_local || one_after_last_local <= target)
                      throw std::runtime_error(
                          "Corrupted incoming connection map local side");
                    if (source < first_remote ||
                        one_after_last_remote <= source)
                      throw std::runtime_error(
                          "Corrupted incoming connection map remote side");
                  } else {
                    if (source < first_local || one_after_last_local <= source)
                      throw std::runtime_error("Corrupted inverted outgoing "
                                               "connection map local side");
                    if (target < first_remote ||
                        one_after_last_remote <= target)
                      throw std::runtime_error("Corrupted inverted outgoing "
                                               "connection map remote side");
                  }
                }
              }
            }
          }
          ++idx;
        }

        std::cout << host_str + " finished check successfully\n";
      }

      const auto local_rank = static_cast<int>(capi.get_rank());

      if (!conn_map_ptr->incoming_connections_.empty()) {
        for (auto &[remote_rank, rank_connection_info] :
             conn_map_ptr->incoming_connections_) {
          const auto is_remote = remote_rank != local_rank;
          for (auto &conn_vec : rank_connection_info.partitioned_connections_)
            create_spatial_connections(static_cast<int>(remote_rank),
                                       local_rank, conn_vec, is_remote, false);
        }
      }

      if (!conn_map_ptr->outgoing_connections_.empty()) {
        for (auto &[remote_rank, rank_connection_info] :
             conn_map_ptr->outgoing_connections_) {
          const auto is_remote = remote_rank != local_rank;
          for (auto &conn_vec : rank_connection_info.partitioned_connections_)
            create_spatial_connections(local_rank,
                                       static_cast<int>(remote_rank), conn_vec,
                                       is_remote, is_remote);
        }
      }
    } catch (const std::exception &e) {
      std::cerr << e.what() << "\n";
      capi.clear_spatial_connections(conn_index);
      throw e;
    }

    opt.second_ = conn_index;
    opt.first_ = true;
    return opt;
  }
  END_ERR_PROP
  opt.first_ = false;
  return opt;
}

sapi::NodesViewStruct *
view_node_positions(sapi::OptionalIndex index,
                    const sapi::MPStruct &mask_parameters) {
  BEGIN_ERR_PROP { return capi.view_node_positions(index, mask_parameters); }
  END_ERR_PROP
  return nullptr;
}

sapi::RemoteConnectionViewPair *view_spatial_connections(std::size_t index) {
  BEGIN_ERR_PROP { return capi.view_spatial_connections(index); }
  END_ERR_PROP
  return nullptr;
}

sapi::ConnectionCountsViewStruct *view_connection_counts(std::size_t index) {
  BEGIN_ERR_PROP { return capi.view_connection_counts(index); }
  END_ERR_PROP
  return nullptr;
}

sapi::GridViewStruct *view_grid_vertices() {
  BEGIN_ERR_PROP { return capi.view_grid_vertices(); }
  END_ERR_PROP
  return nullptr;
}

sapi::TiledNodeSequencePairArray *
get_distributed_node_sequences(std::size_t index) {
  BEGIN_ERR_PROP { return capi.get_distributed_node_sequences(index); }
  END_ERR_PROP
  return nullptr;
}

sapi::RecordedTimesArrayPair *get_timer_data() {
  BEGIN_ERR_PROP { return capi.get_timer_data(); }
  END_ERR_PROP
  return nullptr;
}

bool clear_spatial_connections(std::size_t index) {
  BEGIN_ERR_PROP {
    capi.clear_spatial_connections(index);
    return true;
  }
  END_ERR_PROP
  return false;
}
}
