/*
 *  nestgpu_C.h
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





#ifndef NESTGPUC_H
#define NESTGPUC_H

#ifdef __cplusplus
extern "C" {
#endif
  
  char *NESTGPU_GetErrorMessage();

  unsigned char NESTGPU_GetErrorCode();
  
  void NESTGPU_SetOnException(int on_exception);

  int NESTGPU_SetRandomSeed(unsigned long long seed);

  int NESTGPU_SetTimeResolution(float time_res);

  float NESTGPU_GetTimeResolution();

  int NESTGPU_SetMaxSpikeBufferSize(int max_size);

  int NESTGPU_GetMaxSpikeBufferSize();

  int NESTGPU_SetSimTime(float sim_time);

  int NESTGPU_SetVerbosityLevel(int verbosity_level);

  int NESTGPU_SetNestedLoopAlgo(int nested_loop_algo);

  int NESTGPU_Create(char *model_name, int n_neuron, int n_port);

  int NESTGPU_CreateRecord(char *file_name, char *var_name_arr[],
			     int *i_node_arr, int *port_arr,
			     int n_node);
  
  int NESTGPU_GetRecordDataRows(int i_record);
  
  int NESTGPU_GetRecordDataColumns(int i_record);

  float **NESTGPU_GetRecordData(int i_record);

  int NESTGPU_SetNeuronScalParam(int i_node, int n_neuron, char *param_name,
				   float val);

  int NESTGPU_SetNeuronArrayParam(int i_node, int n_neuron,
				    char *param_name, float *param,
				    int array_size);

  int NESTGPU_SetNeuronPtScalParam(int *i_node, int n_neuron,
				     char *param_name, float val);

  int NESTGPU_SetNeuronPtArrayParam(int *i_node, int n_neuron,
				      char *param_name, float *param,
				      int array_size);
  
  int NESTGPU_IsNeuronScalParam(int i_node, char *param_name);
  
  int NESTGPU_IsNeuronPortParam(int i_node, char *param_name);

  int NESTGPU_IsNeuronArrayParam(int i_node, char *param_name);
  

  int NESTGPU_SetNeuronIntVar(int i_node, int n_neuron, char *var_name,
				int val);

  int NESTGPU_SetNeuronScalVar(int i_node, int n_neuron, char *var_name,
				   float val);

  int NESTGPU_SetNeuronArrayVar(int i_node, int n_neuron,
				    char *var_name, float *var,
				    int array_size);

  int NESTGPU_SetNeuronPtIntVar(int *i_node, int n_neuron,
				  char *var_name, int val);

  int NESTGPU_SetNeuronPtScalVar(int *i_node, int n_neuron,
				     char *var_name, float val);

  int NESTGPU_SetNeuronPtArrayVar(int *i_node, int n_neuron,
				      char *var_name, float *var,
				      int array_size);
  
  int NESTGPU_SetNeuronScalParamDistr(int i_node, int n_neuron,
				      char *param_name);

  int NESTGPU_SetNeuronScalVarDistr(int i_node, int n_neuron,
				    char *var_name);


  int NESTGPU_SetNeuronPortParamDistr(int i_node, int n_neuron,
				      char *param_name);

  int NESTGPU_SetNeuronPortVarDistr(int i_node, int n_neuron,
				    char *var_name);

  int NESTGPU_SetNeuronPtScalParamDistr(int *i_node, int n_neuron,
					char *param_name);

  int NESTGPU_SetNeuronPtScalVarDistr(int *i_node, int n_neuron,
				      char *var_name);

  int NESTGPU_SetNeuronPtPortParamDistr(int *i_node, int n_neuron,
					char *param_name);

  int NESTGPU_SetNeuronPtPortVarDistr(int *i_node, int n_neuron,
				      char *var_name);

  int NESTGPU_SetDistributionIntParam(char *param_name, int val);

  int NESTGPU_SetDistributionScalParam(char *param_name, float val);

  int NESTGPU_SetDistributionVectParam(char *param_name, float val, int i);

  int NESTGPU_SetDistributionFloatPtParam(char *param_name, float *array_pt);

  int NESTGPU_IsDistributionFloatParam(char *param_name);
  
  int NESTGPU_IsNeuronIntVar(int i_node, char *var_name);

  int NESTGPU_IsNeuronScalVar(int i_node, char *var_name);
  
  int NESTGPU_IsNeuronPortVar(int i_node, char *var_name);

  int NESTGPU_IsNeuronArrayVar(int i_node, char *var_name);

  int NESTGPU_GetNeuronParamSize(int i_node, char *param_name);

  int NESTGPU_GetNeuronVarSize(int i_node, char *var_name);
  
  float *NESTGPU_GetNeuronParam(int i_node, int n_neuron,
				  char *param_name);

  float *NESTGPU_GetNeuronPtParam(int *i_node, int n_neuron,
				    char *param_name);

  float *NESTGPU_GetArrayParam(int i_node, char *param_name);

  int *NESTGPU_GetNeuronIntVar(int i_node, int n_neuron,
				 char *param_name);

  int *NESTGPU_GetNeuronPtIntVar(int *i_node, int n_neuron,
				   char *param_name);
  
  float *NESTGPU_GetNeuronVar(int i_node, int n_neuron,
				char *param_name);

  float *NESTGPU_GetNeuronPtVar(int *i_node, int n_neuron,
				  char *param_name);
  
  float *NESTGPU_GetArrayVar(int i_node, char *var_name);
  
  int NESTGPU_Calibrate();

  int NESTGPU_Simulate();

  int NESTGPU_StartSimulation();

  int NESTGPU_SimulationStep();

  int NESTGPU_EndSimulation();

  int NESTGPU_ConnectMpiInit(int argc, char *argv[]);

  int NESTGPU_MpiFinalize();
  
  int NESTGPU_HostId();

  int NESTGPU_HostNum();

  unsigned int *NESTGPU_RandomInt(size_t n);
  
  float *NESTGPU_RandomUniform(size_t n);
  
  float *NESTGPU_RandomNormal(size_t n, float mean, float stddev);
  
  float *NESTGPU_RandomNormalClipped(size_t n, float mean, float stddev,
				       float vmin, float vmax, float vstep);
  
  int NESTGPU_ConnSpecInit();

  int NESTGPU_SetConnSpecParam(char *param_name, int value);

  int NESTGPU_ConnSpecIsParam(char *param_name);

  int NESTGPU_SynSpecInit();

  int NESTGPU_SetSynSpecIntParam(char *param_name, int value);

  int NESTGPU_SetSynSpecFloatParam(char *param_name, float value);

  int NESTGPU_SetSynSpecFloatPtParam(char *param_name, float *array_pt);

  int NESTGPU_SynSpecIsIntParam(char *param_name);

  int NESTGPU_SynSpecIsFloatParam(char *param_name);

  int NESTGPU_SynSpecIsFloatPtParam(char *param_name);

  int NESTGPU_ConnectSeqSeq(int i_source, int n_source, int i_target,
			      int n_target);

  int NESTGPU_ConnectSeqGroup(int i_source, int n_source, int *i_target,
				int n_target);

  int NESTGPU_ConnectGroupSeq(int *i_source, int n_source, int i_target,
				int n_target);

  int NESTGPU_ConnectGroupGroup(int *i_source, int n_source, int *i_target,
				  int n_target);

  int NESTGPU_RemoteConnectSeqSeq(int i_source_host, int i_source,
				    int n_source, int i_target_host,
				    int i_target, int n_target);

  int NESTGPU_RemoteConnectSeqGroup(int i_source_host, int i_source,
				      int n_source, int i_target_host,
				      int *i_target, int n_target);

  int NESTGPU_RemoteConnectGroupSeq(int i_source_host, int *i_source,
				      int n_source, int i_target_host,
				      int i_target, int n_target);

  int NESTGPU_RemoteConnectGroupGroup(int i_source_host, int *i_source,
					int n_source, int i_target_host,
					int *i_target, int n_target);

  char **NESTGPU_GetIntVarNames(int i_node);

  char **NESTGPU_GetScalVarNames(int i_node);
  
  int NESTGPU_GetNIntVar(int i_node);

  int NESTGPU_GetNScalVar(int i_node);
    
  char **NESTGPU_GetPortVarNames(int i_node);
  
  int NESTGPU_GetNPortVar(int i_node);
    
  char **NESTGPU_GetScalParamNames(int i_node);
  
  int NESTGPU_GetNScalParam(int i_node);
    
  char **NESTGPU_GetPortParamNames(int i_node);

  int NESTGPU_GetNGroupParam(int i_node);
  
  char **NESTGPU_GetGroupParamNames(int i_node);

  int NESTGPU_GetNPortParam(int i_node);

  char **NESTGPU_GetArrayParamNames(int i_node);
  
  int NESTGPU_GetNArrayParam(int i_node);

  char **NESTGPU_GetArrayVarNames(int i_node);
  
  int NESTGPU_GetNArrayVar(int i_node);
    
  int64_t *NESTGPU_GetSeqSeqConnections(int i_source, int n_source,
					int i_target, int n_target,
					int syn_group, int64_t *n_conn);

  int64_t *NESTGPU_GetSeqGroupConnections(int i_source, int n_source,
					  int *i_target_pt, int n_target,
					  int syn_group, int64_t *n_conn);

  int64_t *NESTGPU_GetGroupSeqConnections(int *i_source_pt, int n_source,
					  int i_target, int n_target,
					  int syn_group, int64_t *n_conn);

  int64_t *NESTGPU_GetGroupGroupConnections(int *i_source_pt, int n_source,
					    int *i_target_pt, int n_target,
					    int syn_group, int64_t *n_conn);
  
  int NESTGPU_GetConnectionStatus(int64_t *conn_ids, int64_t n_conn,
				  int *i_source, int *i_target,
				  int *port,
				  unsigned char *syn_group, float *delay,
				  float *weight);
  
  int NESTGPU_IsConnectionFloatParam(char *param_name);
  
  int NESTGPU_IsConnectionIntParam(char *param_name);
  
  int NESTGPU_GetConnectionFloatParam(int64_t *conn_ids, int64_t n_conn,
				      float *param_arr, char *param_name);
  
  int NESTGPU_GetConnectionIntParam(int64_t *conn_ids, int64_t n_conn,
				    int *param_arr, char *param_name);
  
  int NESTGPU_SetConnectionFloatParamDistr(int64_t *conn_ids, int64_t n_conn,
					   char *param_name);
  
  int NESTGPU_SetConnectionIntParamArr(int64_t *conn_ids, int64_t n_conn,
				       int *param_arr, char *param_name);
  
  int NESTGPU_SetConnectionFloatParam(int64_t *conn_ids, int64_t n_conn,
				      float val, char *param_name);
  
  int NESTGPU_SetConnectionIntParam(int64_t *conn_ids, int64_t n_conn,
				    int val, char *param_name);

  int NESTGPU_CreateSynGroup(char *model_name);
  
  int NESTGPU_GetSynGroupNParam(int i_syn_group);
  
  char **NESTGPU_GetSynGroupParamNames(int i_syn_group);
  
  int NESTGPU_IsSynGroupParam(int i_syn_group, char *param_name);
  
  int NESTGPU_GetSynGroupParamIdx(int i_syn_group, char *param_name);
  
  float NESTGPU_GetSynGroupParam(int i_syn_group, char *param_name);
  
  int NESTGPU_SetSynGroupParam(int i_syn_group, char *param_name, float val);

  int NESTGPU_ActivateSpikeCount(int i_node, int n_node);

  int NESTGPU_ActivateRecSpikeTimes(int i_node, int n_node,
				      int max_n_rec_spike_times);
  
  int NESTGPU_SetRecSpikeTimesStep(int i_node, int n_node,
				   int rec_spike_times_step);
  
  int NESTGPU_GetNRecSpikeTimes(int i_node);

  int NESTGPU_GetRecSpikeTimes(int i_node, int n_node,
			       int **n_spike_times_pt,
			       float ***spike_times_pt);

  int NESTGPU_PushSpikesToNodes(int n_spikes, int *node_id);
 
  int NESTGPU_GetExtNeuronInputSpikes(int *n_spikes, int **node, int **port,
					float **spike_height,
					int include_zeros);

  int NESTGPU_SetNeuronGroupParam(int i_node, int n_node, char *param_name,
				    float val);

  int NESTGPU_IsNeuronGroupParam(int i_node, char *param_name);

  float NESTGPU_GetNeuronGroupParam(int i_node, char *param_name);

  int NESTGPU_GetNBoolParam();
  
  char **NESTGPU_GetBoolParamNames();
  
  int NESTGPU_IsBoolParam(char *param_name);
  
  int NESTGPU_GetBoolParamIdx(char *param_name);
  
  bool NESTGPU_GetBoolParam(char *param_name);
  
  int NESTGPU_SetBoolParam(char *param_name, bool val);

  int NESTGPU_GetNFloatParam();
  
  char **NESTGPU_GetFloatParamNames();
  
  int NESTGPU_IsFloatParam(char *param_name);
  
  int NESTGPU_GetFloatParamIdx(char *param_name);
  
  float NESTGPU_GetFloatParam(char *param_name);
  
  int NESTGPU_SetFloatParam(char *param_name, float val);

  int NESTGPU_GetNIntParam();
  
  char **NESTGPU_GetIntParamNames();
  
  int NESTGPU_IsIntParam(char *param_name);
  
  int NESTGPU_GetIntParamIdx(char *param_name);
  
  int NESTGPU_GetIntParam(char *param_name);
  
  int NESTGPU_SetIntParam(char *param_name, int val);

  int NESTGPU_RemoteCreate(int i_host, char *model_name, int n_neuron,
			     int n_port);

#ifdef __cplusplus
}
#endif


#endif
