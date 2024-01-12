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





#include <config.h>
#include <iostream>
#include <cstdlib>
#include <string>
#include <cstring>

#include "nestgpu.h"
#include "nestgpu_C.h"
#include "propagate_error.h"

extern "C" {
  static NESTGPU *NESTGPU_instance = NULL;
  ConnSpec ConnSpec_instance;
  SynSpec SynSpec_instance;

  void checkNESTGPUInstance() {
    if (NESTGPU_instance == NULL) {
      NESTGPU_instance = new NESTGPU();
    }
  }
  
  char *NESTGPU_GetErrorMessage()
  {
    checkNESTGPUInstance();
    char *cstr = NESTGPU_instance->GetErrorMessage();
    return cstr;
  }

  unsigned char NESTGPU_GetErrorCode()
  {
    checkNESTGPUInstance();
    return NESTGPU_instance->GetErrorCode();
  }

  void NESTGPU_SetOnException(int on_exception)
  {
    checkNESTGPUInstance();
    NESTGPU_instance->SetOnException(on_exception);
  }

  unsigned int *RandomInt(size_t n);
  
  int NESTGPU_SetRandomSeed(unsigned long long seed)
  { int ret = 0; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->SetRandomSeed(seed);
  } END_ERR_PROP return ret; }

  int NESTGPU_SetTimeResolution(float time_res)
  { int ret = 0; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->SetTimeResolution(time_res);
  } END_ERR_PROP return ret; }

  float NESTGPU_GetTimeResolution()
  { float ret = 0.0; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->GetTimeResolution();
  } END_ERR_PROP return ret; }

  int NESTGPU_SetMaxSpikeBufferSize(int max_size)
  { int ret = 0; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->SetMaxSpikeBufferSize(max_size);
  } END_ERR_PROP return ret; }

  int NESTGPU_GetMaxSpikeBufferSize()
  { int ret = 0; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->GetMaxSpikeBufferSize();
  } END_ERR_PROP return ret; }

  int NESTGPU_SetSimTime(float sim_time)
  { int ret = 0; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->SetSimTime(sim_time);
  } END_ERR_PROP return ret; }

  int NESTGPU_SetVerbosityLevel(int verbosity_level)
  { int ret = 0; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->SetVerbosityLevel(verbosity_level);
  } END_ERR_PROP return ret; }

  int NESTGPU_SetNestedLoopAlgo(int nested_loop_algo)
  { int ret = 0; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->SetNestedLoopAlgo(nested_loop_algo);
  } END_ERR_PROP return ret; }

  
  int NESTGPU_Create(char *model_name, int n_neuron, int n_port)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string model_name_str = std::string(model_name);
    NodeSeq neur = NESTGPU_instance->Create(model_name_str, n_neuron,
						    n_port);
    ret = neur[0];
  } END_ERR_PROP return ret; }

  int NESTGPU_CreateRecord(char *file_name, char *var_name_arr[],
			     int *i_node_arr, int *port_arr,
			     int n_node)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string file_name_str = std::string(file_name);
    std::vector<std::string> var_name_vect;
    for (int i=0; i<n_node; i++) {
      std::string var_name = std::string(var_name_arr[i]);
      var_name_vect.push_back(var_name);
    }
    ret = NESTGPU_instance->CreateRecord
      (file_name_str, var_name_vect.data(), i_node_arr, port_arr,
       n_node);		       
  } END_ERR_PROP return ret; }
  
  int NESTGPU_GetRecordDataRows(int i_record)
  { int ret = 0; BEGIN_ERR_PROP {
    std::vector<std::vector<float> > *data_vect_pt
      = NESTGPU_instance->GetRecordData(i_record);

    ret = data_vect_pt->size();
  } END_ERR_PROP return ret; }
  
  int NESTGPU_GetRecordDataColumns(int i_record)
  { int ret = 0; BEGIN_ERR_PROP {
    std::vector<std::vector<float> > *data_vect_pt
      = NESTGPU_instance->GetRecordData(i_record);
    
    ret = data_vect_pt->at(0).size();
  } END_ERR_PROP return ret; }

  float **NESTGPU_GetRecordData(int i_record)
  { float **ret = NULL; BEGIN_ERR_PROP {
    std::vector<std::vector<float> > *data_vect_pt
      = NESTGPU_instance->GetRecordData(i_record);
    int nr = data_vect_pt->size();
    ret = new float*[nr];
    for (int i=0; i<nr; i++) {
      ret[i] = data_vect_pt->at(i).data();
    }
  } END_ERR_PROP return ret; }

  int NESTGPU_SetNeuronScalParam(int i_node, int n_neuron, char *param_name,
				   float val)
  { int ret = 0; BEGIN_ERR_PROP {
    
    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->SetNeuronParam(i_node, n_neuron,
					     param_name_str, val);
  } END_ERR_PROP return ret; }

  int NESTGPU_SetNeuronArrayParam(int i_node, int n_neuron,
				    char *param_name, float *param,
				    int array_size)
  { int ret = 0; BEGIN_ERR_PROP {
      std::string param_name_str = std::string(param_name);    
      ret = NESTGPU_instance->SetNeuronParam(i_node, n_neuron,
					       param_name_str, param,
					       array_size);
  } END_ERR_PROP return ret; }

  int NESTGPU_SetNeuronPtScalParam(int *i_node, int n_neuron,
				     char *param_name,float val)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->SetNeuronParam(i_node, n_neuron,
					     param_name_str, val);
  } END_ERR_PROP return ret; }

  int NESTGPU_SetNeuronPtArrayParam(int *i_node, int n_neuron,
				     char *param_name, float *param,
				     int array_size)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);    
    ret = NESTGPU_instance->SetNeuronParam(i_node, n_neuron,
					     param_name_str, param,
					     array_size);
  } END_ERR_PROP return ret; }
  
  int NESTGPU_IsNeuronScalParam(int i_node, char *param_name)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    
    ret = NESTGPU_instance->IsNeuronScalParam(i_node, param_name_str);
  } END_ERR_PROP return ret; }
  
  int NESTGPU_IsNeuronPortParam(int i_node, char *param_name)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    
    ret = NESTGPU_instance->IsNeuronPortParam(i_node, param_name_str);
  } END_ERR_PROP return ret; }
  
  int NESTGPU_IsNeuronArrayParam(int i_node, char *param_name)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    
    ret = NESTGPU_instance->IsNeuronArrayParam(i_node, param_name_str);
  } END_ERR_PROP return ret; }
  

  int NESTGPU_SetNeuronIntVar(int i_node, int n_neuron, char *var_name,
				int val)
  { int ret = 0; BEGIN_ERR_PROP {
    
    std::string var_name_str = std::string(var_name);
    ret = NESTGPU_instance->SetNeuronIntVar(i_node, n_neuron,
					     var_name_str, val);
  } END_ERR_PROP return ret; }

  int NESTGPU_SetNeuronScalVar(int i_node, int n_neuron, char *var_name,
				   float val)
  { int ret = 0; BEGIN_ERR_PROP {
    
    std::string var_name_str = std::string(var_name);
    ret = NESTGPU_instance->SetNeuronVar(i_node, n_neuron,
					     var_name_str, val);
  } END_ERR_PROP return ret; }

  int NESTGPU_SetNeuronArrayVar(int i_node, int n_neuron,
				    char *var_name, float *var,
				    int array_size)
  { int ret = 0; BEGIN_ERR_PROP {
      std::string var_name_str = std::string(var_name);    
      ret = NESTGPU_instance->SetNeuronVar(i_node, n_neuron,
					       var_name_str, var,
					       array_size);
  } END_ERR_PROP return ret; }

  int NESTGPU_SetNeuronPtIntVar(int *i_node, int n_neuron,
				     char *var_name, int val)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string var_name_str = std::string(var_name);
    ret = NESTGPU_instance->SetNeuronIntVar(i_node, n_neuron,
					      var_name_str, val);
  } END_ERR_PROP return ret; }

  int NESTGPU_SetNeuronPtScalVar(int *i_node, int n_neuron,
				     char *var_name, float val)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string var_name_str = std::string(var_name);
    ret = NESTGPU_instance->SetNeuronVar(i_node, n_neuron,
					     var_name_str, val);
  } END_ERR_PROP return ret; }

  int NESTGPU_SetNeuronPtArrayVar(int *i_node, int n_neuron,
				     char *var_name, float *var,
				     int array_size)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string var_name_str = std::string(var_name);    
    ret = NESTGPU_instance->SetNeuronVar(i_node, n_neuron,
					     var_name_str, var,
					     array_size);
  } END_ERR_PROP return ret; }


  
  int NESTGPU_SetNeuronScalParamDistr(int i_node, int n_neuron,
				      char *param_name)
  { int ret = 0; BEGIN_ERR_PROP {
    
    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->SetNeuronScalParamDistr(i_node, n_neuron,
						    param_name_str);
  } END_ERR_PROP return ret; }

  int NESTGPU_SetNeuronScalVarDistr(int i_node, int n_neuron,
				    char *var_name)
  { int ret = 0; BEGIN_ERR_PROP {
    
    std::string var_name_str = std::string(var_name);
    ret = NESTGPU_instance->SetNeuronScalVarDistr(i_node, n_neuron,
						  var_name_str);
  } END_ERR_PROP return ret; }


  int NESTGPU_SetNeuronPortParamDistr(int i_node, int n_neuron,
				      char *param_name)
  { int ret = 0; BEGIN_ERR_PROP {
    
    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->SetNeuronPortParamDistr(i_node, n_neuron,
						    param_name_str);
  } END_ERR_PROP return ret; }

  int NESTGPU_SetNeuronPortVarDistr(int i_node, int n_neuron,
				    char *var_name)
  { int ret = 0; BEGIN_ERR_PROP {
    
    std::string var_name_str = std::string(var_name);
    ret = NESTGPU_instance->SetNeuronPortVarDistr(i_node, n_neuron,
						  var_name_str);
  } END_ERR_PROP return ret; }

  int NESTGPU_SetNeuronPtScalParamDistr(int *i_node, int n_neuron,
					char *param_name)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->SetNeuronPtScalParamDistr(i_node, n_neuron,
						      param_name_str);
  } END_ERR_PROP return ret; }

  int NESTGPU_SetNeuronPtScalVarDistr(int *i_node, int n_neuron,
					char *var_name)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string var_name_str = std::string(var_name);
    ret = NESTGPU_instance->SetNeuronPtScalVarDistr(i_node, n_neuron,
						    var_name_str);
  } END_ERR_PROP return ret; }

  int NESTGPU_SetNeuronPtPortParamDistr(int *i_node, int n_neuron,
					char *param_name)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->SetNeuronPtPortParamDistr(i_node, n_neuron,
						      param_name_str);
  } END_ERR_PROP return ret; }

  int NESTGPU_SetNeuronPtPortVarDistr(int *i_node, int n_neuron,
				      char *var_name)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string var_name_str = std::string(var_name);
    ret = NESTGPU_instance->SetNeuronPtPortVarDistr(i_node, n_neuron,
						    var_name_str);
  } END_ERR_PROP return ret; }


  int NESTGPU_SetDistributionIntParam(char *param_name, int val)
  { int ret = 0; BEGIN_ERR_PROP {
    
    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->SetDistributionIntParam(param_name_str, val);
  } END_ERR_PROP return ret; }

  int NESTGPU_SetDistributionScalParam(char *param_name, float val)
  { int ret = 0; BEGIN_ERR_PROP {
    
    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->SetDistributionScalParam(param_name_str, val);
  } END_ERR_PROP return ret; }

  int NESTGPU_SetDistributionVectParam(char *param_name, float val, int i)
  { int ret = 0; BEGIN_ERR_PROP {
    
    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->SetDistributionVectParam(param_name_str, val, i);
  } END_ERR_PROP return ret; }

  int NESTGPU_SetDistributionFloatPtParam(char *param_name, float *array_pt)
  { int ret = 0; BEGIN_ERR_PROP {
    
    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->SetDistributionFloatPtParam(param_name_str,
							array_pt);
  } END_ERR_PROP return ret; }

  int NESTGPU_IsDistributionFloatParam(char *param_name)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    
    ret = NESTGPU_instance->IsDistributionFloatParam(param_name_str);
  } END_ERR_PROP return ret; }

  
  int NESTGPU_IsNeuronIntVar(int i_node, char *var_name)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string var_name_str = std::string(var_name);

    ret = NESTGPU_instance->IsNeuronIntVar(i_node, var_name_str);
  } END_ERR_PROP return ret; }
  
  int NESTGPU_IsNeuronScalVar(int i_node, char *var_name)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string var_name_str = std::string(var_name);
    
    ret = NESTGPU_instance->IsNeuronScalVar(i_node, var_name_str);
  } END_ERR_PROP return ret; }
  
  int NESTGPU_IsNeuronPortVar(int i_node, char *var_name)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string var_name_str = std::string(var_name);
    
    ret = NESTGPU_instance->IsNeuronPortVar(i_node, var_name_str);
  } END_ERR_PROP return ret; }
  
  int NESTGPU_IsNeuronArrayVar(int i_node, char *var_name)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string var_name_str = std::string(var_name);
    
    ret = NESTGPU_instance->IsNeuronArrayVar(i_node, var_name_str);
  } END_ERR_PROP return ret; }
  

  int NESTGPU_GetNeuronParamSize(int i_node, char *param_name)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    
    ret = NESTGPU_instance->GetNeuronParamSize(i_node, param_name_str);
  } END_ERR_PROP return ret; }
  
  
  int NESTGPU_GetNeuronVarSize(int i_node, char *var_name)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string var_name_str = std::string(var_name);
    
    ret = NESTGPU_instance->GetNeuronVarSize(i_node, var_name_str);
  } END_ERR_PROP return ret; }
  
  
  float *NESTGPU_GetNeuronParam(int i_node, int n_neuron,
				  char *param_name)
  { float *ret = NULL; BEGIN_ERR_PROP {
    
    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->GetNeuronParam(i_node, n_neuron,
					     param_name_str);
  } END_ERR_PROP return ret; }


  float *NESTGPU_GetNeuronPtParam(int *i_node, int n_neuron,
				 char *param_name)
  { float *ret = NULL; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->GetNeuronParam(i_node, n_neuron,
					     param_name_str);
  } END_ERR_PROP return ret; }


  float *NESTGPU_GetArrayParam(int i_node, char *param_name)
  { float *ret = NULL; BEGIN_ERR_PROP {
    
    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->GetArrayParam(i_node, param_name_str);
  } END_ERR_PROP return ret; }

  
  int *NESTGPU_GetNeuronIntVar(int i_node, int n_neuron,
				 char *param_name)
  { int *ret = NULL; BEGIN_ERR_PROP {
    
    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->GetNeuronIntVar(i_node, n_neuron,
					      param_name_str);
  } END_ERR_PROP return ret; }


  int *NESTGPU_GetNeuronPtIntVar(int *i_node, int n_neuron,
				   char *param_name)
  { int *ret = NULL; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->GetNeuronIntVar(i_node, n_neuron,
					      param_name_str);
  } END_ERR_PROP return ret; }

  float *NESTGPU_GetNeuronVar(int i_node, int n_neuron,
				char *param_name)
  { float *ret = NULL; BEGIN_ERR_PROP {
    
    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->GetNeuronVar(i_node, n_neuron,
					   param_name_str);
  } END_ERR_PROP return ret; }


  float *NESTGPU_GetNeuronPtVar(int *i_node, int n_neuron,
				 char *param_name)
  { float *ret = NULL; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->GetNeuronVar(i_node, n_neuron,
					   param_name_str);
  } END_ERR_PROP return ret; }

  float *NESTGPU_GetArrayVar(int i_node, char *var_name)
  { float *ret = NULL; BEGIN_ERR_PROP {
    
    std::string var_name_str = std::string(var_name);
    ret = NESTGPU_instance->GetArrayVar(i_node, var_name_str);
  } END_ERR_PROP return ret; }


  int NESTGPU_Calibrate()
  { int ret = 0; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->Calibrate();
  } END_ERR_PROP return ret; }

  int NESTGPU_Simulate()
  { int ret = 0; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->Simulate();
  } END_ERR_PROP return ret; }

  int NESTGPU_StartSimulation()
  { int ret = 0; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->StartSimulation();
  } END_ERR_PROP return ret; }

  int NESTGPU_SimulationStep()
  { int ret = 0; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->SimulationStep();
  } END_ERR_PROP return ret; }

  int NESTGPU_EndSimulation()
  { int ret = 0; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->EndSimulation();
  } END_ERR_PROP return ret; }

  int NESTGPU_ConnectMpiInit(int argc, char *argv[])
  { int ret = 0; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->ConnectMpiInit(argc, argv);
  } END_ERR_PROP return ret; }

  int NESTGPU_MpiFinalize()
  { int ret = 0; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->MpiFinalize();
  } END_ERR_PROP return ret; }

  int NESTGPU_HostId()
  { int ret = 0; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->HostId();
  } END_ERR_PROP return ret; }

  int NESTGPU_HostNum()
  { int ret = 0; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->HostNum();
  } END_ERR_PROP return ret; }

  unsigned int *NESTGPU_RandomInt(size_t n)
  { unsigned int *ret = NULL; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->RandomInt(n);
  } END_ERR_PROP return ret; }
  
  float *NESTGPU_RandomUniform(size_t n)
  { float* ret = NULL; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->RandomUniform(n);
  } END_ERR_PROP return ret; }
  
  float *NESTGPU_RandomNormal(size_t n, float mean, float stddev)
  { float *ret = NULL; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->RandomNormal(n, mean, stddev);
  } END_ERR_PROP return ret; }
  
  float *NESTGPU_RandomNormalClipped(size_t n, float mean, float stddev,
				       float vmin, float vmax, float vstep)
  { float *ret = NULL; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->RandomNormalClipped(n, mean, stddev, vmin,
						  vmax, vstep);
  } END_ERR_PROP return ret; }
  
  int NESTGPU_ConnSpecInit()
  { int ret = 0; BEGIN_ERR_PROP {
    ret = ConnSpec_instance.Init();
  } END_ERR_PROP return ret; }

  int NESTGPU_SetConnSpecParam(char *param_name, int value)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = ConnSpec_instance.SetParam(param_name_str, value);
  } END_ERR_PROP return ret; }

  int NESTGPU_ConnSpecIsParam(char *param_name)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = ConnSpec::IsParam(param_name_str);
  } END_ERR_PROP return ret; }

  int NESTGPU_SynSpecInit()
  { int ret = 0; BEGIN_ERR_PROP {
    ret = SynSpec_instance.Init();
  } END_ERR_PROP return ret; }

  int NESTGPU_SetSynSpecIntParam(char *param_name, int value)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = SynSpec_instance.SetParam(param_name_str, value);
  } END_ERR_PROP return ret; }

  int NESTGPU_SetSynSpecFloatParam(char *param_name, float value)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = SynSpec_instance.SetParam(param_name_str, value);
  } END_ERR_PROP return ret; }

  int NESTGPU_SetSynSpecFloatPtParam(char *param_name, float *array_pt)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = SynSpec_instance.SetParam(param_name_str, array_pt);
  } END_ERR_PROP return ret; }

  int NESTGPU_SynSpecIsIntParam(char *param_name)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = SynSpec_instance.IsIntParam(param_name_str);
  } END_ERR_PROP return ret; }

  int NESTGPU_SynSpecIsFloatParam(char *param_name)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = SynSpec_instance.IsFloatParam(param_name_str);
  } END_ERR_PROP return ret; }

  int NESTGPU_SynSpecIsFloatPtParam(char *param_name)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = SynSpec_instance.IsFloatPtParam(param_name_str);
  } END_ERR_PROP return ret; }

  int NESTGPU_ConnectSeqSeq(int i_source, int n_source, int i_target,
			      int n_target)
  { int ret = 0; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->Connect(i_source, n_source, i_target, n_target,
				      ConnSpec_instance, SynSpec_instance); 
  } END_ERR_PROP return ret; }

  int NESTGPU_ConnectSeqGroup(int i_source, int n_source, int *i_target,
				int n_target)
  { int ret = 0; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->Connect(i_source, n_source, i_target, n_target,
				      ConnSpec_instance, SynSpec_instance); 
  } END_ERR_PROP return ret; }

  int NESTGPU_ConnectGroupSeq(int *i_source, int n_source, int i_target,
				int n_target)
  { int ret = 0; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->Connect(i_source, n_source, i_target, n_target,
				      ConnSpec_instance, SynSpec_instance);
  } END_ERR_PROP return ret; }

  int NESTGPU_ConnectGroupGroup(int *i_source, int n_source, int *i_target,
				  int n_target)
  { int ret = 0; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->Connect(i_source, n_source, i_target, n_target,
				      ConnSpec_instance, SynSpec_instance);
  } END_ERR_PROP return ret; }

  int NESTGPU_RemoteConnectSeqSeq(int i_source_host, int i_source,
				    int n_source, int i_target_host,
				    int i_target, int n_target)
  { int ret = 0; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->RemoteConnect(i_source_host, i_source, n_source,
					    i_target_host, i_target, n_target,
					    ConnSpec_instance,
					    SynSpec_instance); 
  } END_ERR_PROP return ret; }

  int NESTGPU_RemoteConnectSeqGroup(int i_source_host, int i_source,
				      int n_source, int i_target_host,
				      int *i_target, int n_target)
  { int ret = 0; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->RemoteConnect(i_source_host, i_source, n_source,
					    i_target_host, i_target, n_target,
					    ConnSpec_instance,
					    SynSpec_instance); 
  } END_ERR_PROP return ret; }

  int NESTGPU_RemoteConnectGroupSeq(int i_source_host, int *i_source,
				      int n_source, int i_target_host,
				      int i_target, int n_target)
  { int ret = 0; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->RemoteConnect(i_source_host, i_source, n_source,
					    i_target_host, i_target, n_target,
					    ConnSpec_instance,
					    SynSpec_instance);
  } END_ERR_PROP return ret; }


  int NESTGPU_RemoteConnectGroupGroup(int i_source_host, int *i_source,
					int n_source, int i_target_host,
					int *i_target, int n_target)
  { int ret = 0; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->RemoteConnect(i_source_host, i_source, n_source,
					    i_target_host, i_target, n_target,
					    ConnSpec_instance,
					    SynSpec_instance);
  } END_ERR_PROP return ret; }


  char **NESTGPU_GetIntVarNames(int i_node)
  { char **ret = NULL; BEGIN_ERR_PROP {
    std::vector<std::string> var_name_vect =
      NESTGPU_instance->GetIntVarNames(i_node);
    char **var_name_array = (char**)malloc(var_name_vect.size()
					   *sizeof(char*));
    for (unsigned int i=0; i<var_name_vect.size(); i++) {
      char *var_name = (char*)malloc((var_name_vect[i].length() + 1)
				      *sizeof(char));
      
      strcpy(var_name, var_name_vect[i].c_str());
      var_name_array[i] = var_name;
    }
    ret = var_name_array;
    
  } END_ERR_PROP return ret; }

  char **NESTGPU_GetScalVarNames(int i_node)
  { char **ret = NULL; BEGIN_ERR_PROP {
    std::vector<std::string> var_name_vect =
      NESTGPU_instance->GetScalVarNames(i_node);
    char **var_name_array = (char**)malloc(var_name_vect.size()
					   *sizeof(char*));
    for (unsigned int i=0; i<var_name_vect.size(); i++) {
      char *var_name = (char*)malloc((var_name_vect[i].length() + 1)
				      *sizeof(char));
      
      strcpy(var_name, var_name_vect[i].c_str());
      var_name_array[i] = var_name;
    }
    ret = var_name_array;
    
  } END_ERR_PROP return ret; }

  int NESTGPU_GetNIntVar(int i_node)
  { int ret = 0; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->GetNIntVar(i_node);
  } END_ERR_PROP return ret; }

  int NESTGPU_GetNScalVar(int i_node)
  { int ret = 0; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->GetNScalVar(i_node);
  } END_ERR_PROP return ret; }


  char **NESTGPU_GetPortVarNames(int i_node)
  { char **ret = NULL; BEGIN_ERR_PROP {
    std::vector<std::string> var_name_vect =
      NESTGPU_instance->GetPortVarNames(i_node);
    char **var_name_array = (char**)malloc(var_name_vect.size()
					   *sizeof(char*));
    for (unsigned int i=0; i<var_name_vect.size(); i++) {
      char *var_name = (char*)malloc((var_name_vect[i].length() + 1)
				      *sizeof(char));
      
      strcpy(var_name, var_name_vect[i].c_str());
      var_name_array[i] = var_name;
    }
    ret = var_name_array;
    
  } END_ERR_PROP return ret; }

  
  int NESTGPU_GetNPortVar(int i_node)
  { int ret = 0; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->GetNPortVar(i_node);
  } END_ERR_PROP return ret; }

  
  char **NESTGPU_GetScalParamNames(int i_node)
  { char **ret = NULL; BEGIN_ERR_PROP {
    std::vector<std::string> var_name_vect =
      NESTGPU_instance->GetScalParamNames(i_node);
    char **var_name_array = (char**)malloc(var_name_vect.size()
					   *sizeof(char*));
    for (unsigned int i=0; i<var_name_vect.size(); i++) {
      char *var_name = (char*)malloc((var_name_vect[i].length() + 1)
				      *sizeof(char));
      
      strcpy(var_name, var_name_vect[i].c_str());
      var_name_array[i] = var_name;
    }
    ret = var_name_array;
    
  } END_ERR_PROP return ret; }

  
  int NESTGPU_GetNScalParam(int i_node)
  { int ret = 0; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->GetNScalParam(i_node);
  } END_ERR_PROP return ret; }


  char **NESTGPU_GetGroupParamNames(int i_node)
  { char **ret = NULL; BEGIN_ERR_PROP {
    std::vector<std::string> var_name_vect =
      NESTGPU_instance->GetGroupParamNames(i_node);
    char **var_name_array = (char**)malloc(var_name_vect.size()
					   *sizeof(char*));
    for (unsigned int i=0; i<var_name_vect.size(); i++) {
      char *var_name = (char*)malloc((var_name_vect[i].length() + 1)
				      *sizeof(char));
      
      strcpy(var_name, var_name_vect[i].c_str());
      var_name_array[i] = var_name;
    }
    ret = var_name_array;
    
  } END_ERR_PROP return ret; }

  
  int NESTGPU_GetNGroupParam(int i_node)
  { int ret = 0; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->GetNGroupParam(i_node);
  } END_ERR_PROP return ret; }


  char **NESTGPU_GetPortParamNames(int i_node)
  { char **ret = NULL; BEGIN_ERR_PROP {
    std::vector<std::string> var_name_vect =
      NESTGPU_instance->GetPortParamNames(i_node);
    char **var_name_array = (char**)malloc(var_name_vect.size()
					   *sizeof(char*));
    for (unsigned int i=0; i<var_name_vect.size(); i++) {
      char *var_name = (char*)malloc((var_name_vect[i].length() + 1)
				      *sizeof(char));
      
      strcpy(var_name, var_name_vect[i].c_str());
      var_name_array[i] = var_name;
    }
    ret = var_name_array;
    
  } END_ERR_PROP return ret; }

  
  int NESTGPU_GetNPortParam(int i_node)
  { int ret = 0; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->GetNPortParam(i_node);
  } END_ERR_PROP return ret; }


  char **NESTGPU_GetArrayParamNames(int i_node)
  { char **ret = NULL; BEGIN_ERR_PROP {
    std::vector<std::string> var_name_vect =
      NESTGPU_instance->GetArrayParamNames(i_node);
    char **var_name_array = (char**)malloc(var_name_vect.size()
					   *sizeof(char*));
    for (unsigned int i=0; i<var_name_vect.size(); i++) {
      char *var_name = (char*)malloc((var_name_vect[i].length() + 1)
				      *sizeof(char));
      
      strcpy(var_name, var_name_vect[i].c_str());
      var_name_array[i] = var_name;
    }
    ret = var_name_array;
    
  } END_ERR_PROP return ret; }

  
  int NESTGPU_GetNArrayParam(int i_node)
  { int ret = 0; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->GetNArrayParam(i_node);
  } END_ERR_PROP return ret; }

  char **NESTGPU_GetArrayVarNames(int i_node)
  { char **ret = NULL; BEGIN_ERR_PROP {
    std::vector<std::string> var_name_vect =
      NESTGPU_instance->GetArrayVarNames(i_node);
    char **var_name_array = (char**)malloc(var_name_vect.size()
					   *sizeof(char*));
    for (unsigned int i=0; i<var_name_vect.size(); i++) {
      char *var_name = (char*)malloc((var_name_vect[i].length() + 1)
				      *sizeof(char));
      
      strcpy(var_name, var_name_vect[i].c_str());
      var_name_array[i] = var_name;
    }
    ret = var_name_array;
    
  } END_ERR_PROP return ret; }

  int NESTGPU_GetNArrayVar(int i_node)
  { int ret = 0; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->GetNArrayVar(i_node);
  } END_ERR_PROP return ret; }


  int64_t *NESTGPU_GetSeqSeqConnections(int i_source, int n_source,
					int i_target, int n_target,
					int syn_group, int64_t *n_conn)
  { int64_t *ret = NULL; BEGIN_ERR_PROP {
      ret = NESTGPU_instance->GetConnections(i_source, n_source, i_target,
					     n_target, syn_group, n_conn);
  } END_ERR_PROP return ret; }

  int64_t *NESTGPU_GetSeqGroupConnections(int i_source, int n_source,
					  int *i_target_pt, int n_target,
					  int syn_group, int64_t *n_conn)
  { int64_t *ret = NULL; BEGIN_ERR_PROP {
      ret = NESTGPU_instance->GetConnections(i_source, n_source, i_target_pt,
					     n_target, syn_group, n_conn);
  } END_ERR_PROP return ret; }

  int64_t *NESTGPU_GetGroupSeqConnections(int *i_source_pt, int n_source,
					  int i_target, int n_target,
					  int syn_group, int64_t *n_conn)
  { int64_t *ret = NULL; BEGIN_ERR_PROP {
      ret = NESTGPU_instance->GetConnections(i_source_pt, n_source, i_target,
					     n_target, syn_group, n_conn);
  } END_ERR_PROP return ret; }

  int64_t *NESTGPU_GetGroupGroupConnections(int *i_source_pt, int n_source,
					    int *i_target_pt, int n_target,
					    int syn_group, int64_t *n_conn)
  { int64_t *ret = NULL; BEGIN_ERR_PROP {
      ret = NESTGPU_instance->GetConnections(i_source_pt, n_source,
					     i_target_pt, n_target,
					     syn_group, n_conn);
  } END_ERR_PROP return ret; }

  int NESTGPU_GetConnectionStatus(int64_t *conn_ids, int64_t n_conn,
				  int *i_source, int *i_target,
				  int *port,
				  unsigned char *syn_group, float *delay,
				  float *weight)
  { int ret = 0; BEGIN_ERR_PROP {
      ret = NESTGPU_instance->GetConnectionStatus
	(conn_ids, n_conn, i_source, i_target, port, syn_group, delay,
	 weight);
  } END_ERR_PROP return ret; }

  int NESTGPU_IsConnectionFloatParam(char *param_name)
  { int ret = 0; BEGIN_ERR_PROP {
      std::string param_name_str = std::string(param_name);
      ret = NESTGPU_instance->IsConnectionFloatParam(param_name_str);
  } END_ERR_PROP return ret; }
  
  int NESTGPU_IsConnectionIntParam(char *param_name)
  { int ret = 0; BEGIN_ERR_PROP {
      std::string param_name_str = std::string(param_name);
      ret = NESTGPU_instance->IsConnectionIntParam(param_name_str);
  } END_ERR_PROP return ret; }
  
  int NESTGPU_GetConnectionFloatParam(int64_t *conn_ids, int64_t n_conn,
				      float *param_arr, char *param_name)
  { int ret = 0; BEGIN_ERR_PROP {
      std::string param_name_str = std::string(param_name);
      ret = NESTGPU_instance->GetConnectionFloatParam(conn_ids, n_conn,
						      param_arr,
						      param_name_str);
  } END_ERR_PROP return ret; }
  
  int NESTGPU_GetConnectionIntParam(int64_t *conn_ids, int64_t n_conn,
				    int *param_arr, char *param_name)
  { int ret = 0; BEGIN_ERR_PROP {
      std::string param_name_str = std::string(param_name);
      ret = NESTGPU_instance->GetConnectionIntParam(conn_ids, n_conn,
						    param_arr,
						    param_name_str);
  } END_ERR_PROP return ret; }

  int NESTGPU_SetConnectionFloatParamDistr(int64_t *conn_ids, int64_t n_conn,
					   char *param_name)
  { int ret = 0; BEGIN_ERR_PROP {
      std::string param_name_str = std::string(param_name);
      ret = NESTGPU_instance->SetConnectionFloatParamDistr(conn_ids, n_conn,
							   param_name_str);
    } END_ERR_PROP return ret; }
  
  int NESTGPU_SetConnectionIntParamArr(int64_t *conn_ids, int64_t n_conn,
				       int *param_arr, char *param_name)
  { int ret = 0; BEGIN_ERR_PROP {
      std::string param_name_str = std::string(param_name);
      ret = NESTGPU_instance->SetConnectionIntParamArr(conn_ids, n_conn,
						       param_arr,
						       param_name_str);
  } END_ERR_PROP return ret; }

  int NESTGPU_SetConnectionFloatParam(int64_t *conn_ids, int64_t n_conn,
				      float val, char *param_name)
  { int ret = 0; BEGIN_ERR_PROP {
      std::string param_name_str = std::string(param_name);
      ret = NESTGPU_instance->SetConnectionFloatParam(conn_ids, n_conn,
						      val, param_name_str);
  } END_ERR_PROP return ret; }

  int NESTGPU_SetConnectionIntParam(int64_t *conn_ids, int64_t n_conn,
				    int val, char *param_name)
  { int ret = 0; BEGIN_ERR_PROP {
      std::string param_name_str = std::string(param_name);
      ret = NESTGPU_instance->SetConnectionIntParam(conn_ids, n_conn,
						    val, param_name_str);
  } END_ERR_PROP return ret; }
  
  int NESTGPU_CreateSynGroup(char *model_name)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string model_name_str = std::string(model_name);
    ret = NESTGPU_instance->CreateSynGroup(model_name_str);
  } END_ERR_PROP return ret; }


  int NESTGPU_GetSynGroupNParam(int i_syn_group)
  { int ret = 0; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->GetSynGroupNParam(i_syn_group);
  } END_ERR_PROP return ret; }

  
  char **NESTGPU_GetSynGroupParamNames(int i_syn_group)
  { char **ret = NULL; BEGIN_ERR_PROP {
    std::vector<std::string> name_vect =
      NESTGPU_instance->GetSynGroupParamNames(i_syn_group);
    char **name_array = (char**)malloc(name_vect.size()
				       *sizeof(char*));
    for (unsigned int i=0; i<name_vect.size(); i++) {
      char *param_name = (char*)malloc((name_vect[i].length() + 1)
				       *sizeof(char));
      
      strcpy(param_name, name_vect[i].c_str());
      name_array[i] = param_name;
    }
    ret = name_array;
    
  } END_ERR_PROP return ret; }

  
  int NESTGPU_IsSynGroupParam(int i_syn_group, char *param_name)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    
    ret = NESTGPU_instance->IsSynGroupParam(i_syn_group, param_name_str);
  } END_ERR_PROP return ret; }

  
  int NESTGPU_GetSynGroupParamIdx(int i_syn_group, char *param_name)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    
    ret = NESTGPU_instance->GetSynGroupParamIdx(i_syn_group, param_name_str);
  } END_ERR_PROP return ret; }

  
  float NESTGPU_GetSynGroupParam(int i_syn_group, char *param_name)
  { float ret = 0.0; BEGIN_ERR_PROP {
    
    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->GetSynGroupParam(i_syn_group, param_name_str);
  } END_ERR_PROP return ret; }

  
  int NESTGPU_SetSynGroupParam(int i_syn_group, char *param_name, float val)
  { float ret = 0.0; BEGIN_ERR_PROP {
    
    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->SetSynGroupParam(i_syn_group, param_name_str,
					       val);
  } END_ERR_PROP return ret; }

  
  int NESTGPU_ActivateSpikeCount(int i_node, int n_node)
  { int ret = 0; BEGIN_ERR_PROP {
    
    ret = NESTGPU_instance->ActivateSpikeCount(i_node, n_node);
  } END_ERR_PROP return ret; }

  
  int NESTGPU_ActivateRecSpikeTimes(int i_node, int n_node,
				      int max_n_rec_spike_times)
  { int ret = 0; BEGIN_ERR_PROP {
    
      ret = NESTGPU_instance->ActivateRecSpikeTimes(i_node, n_node,
						      max_n_rec_spike_times);
  } END_ERR_PROP return ret; }

  int NESTGPU_SetRecSpikeTimesStep(int i_node, int n_node,
				   int rec_spike_times_step)
  { int ret = 0; BEGIN_ERR_PROP {
    
      ret = NESTGPU_instance->SetRecSpikeTimesStep(i_node, n_node,
						   rec_spike_times_step);
  } END_ERR_PROP return ret; }

  
  int NESTGPU_GetNRecSpikeTimes(int i_node)
  { int ret = 0; BEGIN_ERR_PROP {
    
      ret = NESTGPU_instance->GetNRecSpikeTimes(i_node);
  } END_ERR_PROP return ret; }

  int NESTGPU_GetRecSpikeTimes(int i_node, int n_node,
			       int **n_spike_times_pt,
			       float ***spike_times_pt)
  { int ret = 0; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->GetRecSpikeTimes(i_node, n_node, n_spike_times_pt,
					     spike_times_pt);
    
  } END_ERR_PROP return ret; }
  
  int NESTGPU_PushSpikesToNodes(int n_spikes, int *node_id)
  { int ret = 0; BEGIN_ERR_PROP {
    
      ret = NESTGPU_instance->PushSpikesToNodes(n_spikes, node_id);
  } END_ERR_PROP return ret; }

  int NESTGPU_GetExtNeuronInputSpikes(int *n_spikes, int **node, int **port,
			      float **spike_height, int include_zeros)
  { int ret = 0; BEGIN_ERR_PROP {
    
      ret = NESTGPU_instance->GetExtNeuronInputSpikes(n_spikes, node, port,
							spike_height,
							include_zeros>0);
  } END_ERR_PROP return ret; }

  int NESTGPU_SetNeuronGroupParam(int i_node, int n_node, char *param_name,
				    float val)
  { float ret = 0.0; BEGIN_ERR_PROP {
    
    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->SetNeuronGroupParam(i_node, n_node,
						  param_name_str,
						  val);
  } END_ERR_PROP return ret; }

  int NESTGPU_IsNeuronGroupParam(int i_node, char *param_name)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    
    ret = NESTGPU_instance->IsNeuronGroupParam(i_node, param_name_str);
  } END_ERR_PROP return ret; }

  float NESTGPU_GetNeuronGroupParam(int i_node, char *param_name)
  { float ret = 0; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    
    ret = NESTGPU_instance->GetNeuronGroupParam(i_node, param_name_str);
  } END_ERR_PROP return ret; }


  int NESTGPU_GetNBoolParam()
  { int ret = 0; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->GetNBoolParam();
  } END_ERR_PROP return ret; }

  
  char **NESTGPU_GetBoolParamNames()
  { char **ret = NULL; BEGIN_ERR_PROP {
    std::vector<std::string> name_vect =
      NESTGPU_instance->GetBoolParamNames();
    char **name_array = (char**)malloc(name_vect.size()
				       *sizeof(char*));
    for (unsigned int i=0; i<name_vect.size(); i++) {
      char *param_name = (char*)malloc((name_vect[i].length() + 1)
				       *sizeof(char));
      
      strcpy(param_name, name_vect[i].c_str());
      name_array[i] = param_name;
    }
    ret = name_array;
    
  } END_ERR_PROP return ret; }

  
  int NESTGPU_IsBoolParam(char *param_name)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    
    ret = NESTGPU_instance->IsBoolParam(param_name_str);
  } END_ERR_PROP return ret; }

  
  int NESTGPU_GetBoolParamIdx(char *param_name)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    
    ret = NESTGPU_instance->GetBoolParamIdx(param_name_str);
  } END_ERR_PROP return ret; }

  
  bool NESTGPU_GetBoolParam(char *param_name)
  { bool ret = true; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    
    ret = NESTGPU_instance->GetBoolParam(param_name_str);
  } END_ERR_PROP return ret; }

  
  int NESTGPU_SetBoolParam(char *param_name, bool val)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    
    ret = NESTGPU_instance->SetBoolParam(param_name_str, val);
  } END_ERR_PROP return ret; }


  int NESTGPU_GetNFloatParam()
  { int ret = 0; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->GetNFloatParam();
  } END_ERR_PROP return ret; }

  
  char **NESTGPU_GetFloatParamNames()
  { char **ret = NULL; BEGIN_ERR_PROP {
    std::vector<std::string> name_vect =
      NESTGPU_instance->GetFloatParamNames();
    char **name_array = (char**)malloc(name_vect.size()
				       *sizeof(char*));
    for (unsigned int i=0; i<name_vect.size(); i++) {
      char *param_name = (char*)malloc((name_vect[i].length() + 1)
				       *sizeof(char));
      
      strcpy(param_name, name_vect[i].c_str());
      name_array[i] = param_name;
    }
    ret = name_array;
    
  } END_ERR_PROP return ret; }

  
  int NESTGPU_IsFloatParam(char *param_name)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    
    ret = NESTGPU_instance->IsFloatParam(param_name_str);
  } END_ERR_PROP return ret; }

  
  int NESTGPU_GetFloatParamIdx(char *param_name)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    
    ret = NESTGPU_instance->GetFloatParamIdx(param_name_str);
  } END_ERR_PROP return ret; }

  
  float NESTGPU_GetFloatParam(char *param_name)
  { float ret = 0.0; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    
    ret = NESTGPU_instance->GetFloatParam(param_name_str);
  } END_ERR_PROP return ret; }

  
  int NESTGPU_SetFloatParam(char *param_name, float val)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    
    ret = NESTGPU_instance->SetFloatParam(param_name_str, val);
  } END_ERR_PROP return ret; }

  int NESTGPU_GetNIntParam()
  { int ret = 0; BEGIN_ERR_PROP {
    ret = NESTGPU_instance->GetNIntParam();
  } END_ERR_PROP return ret; }

  
  char **NESTGPU_GetIntParamNames()
  { char **ret = NULL; BEGIN_ERR_PROP {
    std::vector<std::string> name_vect =
      NESTGPU_instance->GetIntParamNames();
    char **name_array = (char**)malloc(name_vect.size()
				       *sizeof(char*));
    for (unsigned int i=0; i<name_vect.size(); i++) {
      char *param_name = (char*)malloc((name_vect[i].length() + 1)
				       *sizeof(char));
      
      strcpy(param_name, name_vect[i].c_str());
      name_array[i] = param_name;
    }
    ret = name_array;
    
  } END_ERR_PROP return ret; }

  
  int NESTGPU_IsIntParam(char *param_name)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    
    ret = NESTGPU_instance->IsIntParam(param_name_str);
  } END_ERR_PROP return ret; }

  
  int NESTGPU_GetIntParamIdx(char *param_name)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    
    ret = NESTGPU_instance->GetIntParamIdx(param_name_str);
  } END_ERR_PROP return ret; }

  
  int NESTGPU_GetIntParam(char *param_name)
  { int ret = 0; BEGIN_ERR_PROP {
    
    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->GetIntParam(param_name_str);
  } END_ERR_PROP return ret; }

  
  int NESTGPU_SetIntParam(char *param_name, int val)
  { int ret = 0; BEGIN_ERR_PROP {
    
    std::string param_name_str = std::string(param_name);
    ret = NESTGPU_instance->SetIntParam(param_name_str, val);
  } END_ERR_PROP return ret; }

  int NESTGPU_RemoteCreate(int i_host, char *model_name, int n_neuron,
			     int n_port)
  { int ret = 0; BEGIN_ERR_PROP {
    std::string model_name_str = std::string(model_name);
    RemoteNodeSeq rneur = NESTGPU_instance->RemoteCreate(i_host,
							   model_name_str,
							   n_neuron,
							   n_port);
    ret = rneur.node_seq[0];
  } END_ERR_PROP return ret; }

}
