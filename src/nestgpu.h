/*
 *  nestgpu.h
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





#ifndef NESTGPU_H
#define NESTGPU_H

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>

#include "ngpu_exception.h"
#include "node_group.h"
#include "base_neuron.h"
#include "connect_spec.h"
#include "connect.h"
#include "syn_model.h"

#ifdef HAVE_MPI
class ConnectMpi;
#endif

class PoissonGenerator;
class Multimeter;
class NetConnection;
struct curandGenerator_st;
typedef struct curandGenerator_st* curandGenerator_t;
class ConnSpec;
class SynSpec;

class Sequence
{
 public:
  int i0;
  int n;

 Sequence(int i0=0, int n=0) : i0(i0), n(n) {}

  inline int operator[](int i) {
    if (i<0) {
      throw ngpu_exception("Sequence index cannot be negative");
    }
    if (i>=n) {
      throw ngpu_exception("Sequence index out of range");
    }
    return i0 + i;
  }

  inline Sequence Subseq(int first, int last) {
    if (first<0 || first>last) {
      throw ngpu_exception("Sequence subset range error");
    }
    if (last>=n) {
      throw ngpu_exception("Sequence subset out of range");
    }
    return Sequence(i0 + first, last - first + 1);
  }

  // https://stackoverflow.com/questions/18625223
  inline std::vector<int> ToVector() {
    int start = i0;
    std::vector<int> v(n);
    std::iota(v.begin(), v.end(), start);
    return v;
  }
};

typedef Sequence NodeSeq;

class RemoteNodeSeq
{
 public:
  int i_host;
  NodeSeq node_seq;

  RemoteNodeSeq(int i_host=0, NodeSeq node_seq=NodeSeq(0,0)) :
    i_host(i_host), node_seq(node_seq) {}
};

enum {ON_EXCEPTION_EXIT=0, ON_EXCEPTION_HANDLE};

class NESTGPU
{
  float time_resolution_; // time resolution in ms
  curandGenerator_t *random_generator_;
  unsigned long long kernel_seed_;
  bool calibrate_flag_; // becomes true after calibration

  PoissonGenerator *poiss_generator_;
  Multimeter *multimeter_;
  std::vector<BaseNeuron*> node_vect_; // -> node_group_vect
  std::vector<SynModel*> syn_group_vect_;

  NetConnection *net_connection_;

  bool mpi_flag_; // true if MPI is initialized
#ifdef HAVE_MPI
  ConnectMpi *connect_mpi_;
#endif

  std::vector<signed char> node_group_map_;
  signed char *d_node_group_map_;


  int max_spike_buffer_size_;
  int max_spike_num_;
  int max_spike_per_host_;

  double max_spike_num_fact_;
  double max_spike_per_host_fact_;

  double t_min_;
  double neural_time_; // Neural activity time
  double sim_time_; // Simulation time in ms
  double neur_t0_; // Neural activity simulation time origin
  long long it_; // simulation time index
  long long Nt_; // number of simulation time steps
  int n_poiss_node_;
  int n_remote_node_;
  int i_remote_node_0_;

  double start_real_time_;
  double build_real_time_;
  double end_real_time_;

  bool error_flag_;
  std::string error_message_;
  unsigned char error_code_;
  int on_exception_;

  int verbosity_level_;
  bool print_time_;

  std::vector<RemoteConnection> remote_connection_vect_;
  std::vector<int> ext_neuron_input_spike_node_;
  std::vector<int> ext_neuron_input_spike_port_;
  std::vector<float> ext_neuron_input_spike_height_;

  int CreateNodeGroup(int n_neuron, int n_port);
  int CheckUncalibrated(std::string message);
  double *InitGetSpikeArray(int n_node, int n_port);
  int NodeGroupArrayInit();
  int ClearGetSpikeArrays();
  int FreeGetSpikeArrays();
  int FreeNodeGroupMap();


  template <class T1, class T2>
    int _Connect(T1 source, int n_source, T2 target, int n_target,
		 ConnSpec &conn_spec, SynSpec &syn_spec);

  template<class T1, class T2>
    int _SingleConnect(T1 source, int i_source, T2 target, int i_target,
		       int i_array, SynSpec &syn_spec);
  template<class T1, class T2>
    int _SingleConnect(T1 source, int i_source, T2 target, int i_target,
		       float weight, float delay, int i_array,
		       SynSpec &syn_spec);

  template<class T>
    int _RemoteSingleConnect(int i_source, T target, int i_target,
			     int i_array, SynSpec &syn_spec);
  template<class T>
    int _RemoteSingleConnect(int i_source, T target, int i_target,
			     float weight, float delay, int i_array,
			     SynSpec &syn_spec);

  template <class T1, class T2>
    int _ConnectOneToOne(T1 source, T2 target, int n_node, SynSpec &syn_spec);

  template <class T1, class T2>
    int _ConnectAllToAll
    (T1 source, int n_source, T2 target, int n_target, SynSpec &syn_spec);

  template <class T1, class T2>
    int _ConnectFixedTotalNumber
    (T1 source, int n_source, T2 target, int n_target, int n_conn,
     SynSpec &syn_spec);

  template <class T1, class T2>
    int _ConnectFixedIndegree
    (
     T1 source, int n_source, T2 target, int n_target, int indegree,
     SynSpec &syn_spec
     );

  template <class T1, class T2>
    int _ConnectFixedOutdegree
    (
     T1 source, int n_source, T2 target, int n_target, int outdegree,
     SynSpec &syn_spec
     );

#ifdef HAVE_MPI
  template <class T1, class T2>
    int _RemoteConnect(RemoteNode<T1> source, int n_source,
		       RemoteNode<T2> target, int n_target,
		       ConnSpec &conn_spec, SynSpec &syn_spec);

  template <class T1, class T2>
    int _RemoteConnectOneToOne
    (RemoteNode<T1> source, RemoteNode<T2> target, int n_node,
     SynSpec &syn_spec);

  template <class T1, class T2>
    int _RemoteConnectAllToAll
    (RemoteNode<T1> source, int n_source, RemoteNode<T2> target, int n_target,
     SynSpec &syn_spec);

  template <class T1, class T2>
    int _RemoteConnectFixedTotalNumber
    (RemoteNode<T1> source, int n_source, RemoteNode<T2> target, int n_target,
     int n_conn, SynSpec &syn_spec);

  template <class T1, class T2>
    int _RemoteConnectFixedIndegree
    (RemoteNode<T1> source, int n_source, RemoteNode<T2> target, int n_target,
     int indegree, SynSpec &syn_spec);

  template <class T1, class T2>
    int _RemoteConnectFixedOutdegree
    (RemoteNode<T1> source, int n_source, RemoteNode<T2> target, int n_target,
     int outdegree, SynSpec &syn_spec);
#endif
  int ConnectRemoteNodes();

  double SpikeBufferUpdate_time_;
  double poisson_generator_time_;
  double neuron_Update_time_;
  double copy_ext_spike_time_;
  double SendExternalSpike_time_;
  double SendSpikeToRemote_time_;
  double RecvSpikeFromRemote_time_;
  double NestedLoop_time_;
  double GetSpike_time_;
  double SpikeReset_time_;
  double ExternalSpikeReset_time_;

  bool first_simulation_flag_;

 public:
  NESTGPU();

  ~NESTGPU();

  int SetRandomSeed(unsigned long long seed);

  int SetTimeResolution(float time_res);

  inline float GetTimeResolution() {
    return time_resolution_;
  }

  inline int SetSimTime(float sim_time) {
    sim_time_ = sim_time;
    return 0;
  }

  inline float GetSimTime() {
    return sim_time_;
  }

  inline int SetVerbosityLevel(int verbosity_level) {
    verbosity_level_ = verbosity_level;
    return 0;
  }

  inline int SetPrintTime(bool print_time) {
    print_time_ = print_time;
    return 0;
  }


  int SetMaxSpikeBufferSize(int max_size);
  int GetMaxSpikeBufferSize();

  int GetNBoolParam();
  std::vector<std::string> GetBoolParamNames();
  bool IsBoolParam(std::string param_name);
  int GetBoolParamIdx(std::string param_name);
  bool GetBoolParam(std::string param_name);
  int SetBoolParam(std::string param_name, bool val);

  int GetNFloatParam();
  std::vector<std::string> GetFloatParamNames();
  bool IsFloatParam(std::string param_name);
  int GetFloatParamIdx(std::string param_name);
  float GetFloatParam(std::string param_name);
  int SetFloatParam(std::string param_name, float val);

  int GetNIntParam();
  std::vector<std::string> GetIntParamNames();
  bool IsIntParam(std::string param_name);
  int GetIntParamIdx(std::string param_name);
  int GetIntParam(std::string param_name);
  int SetIntParam(std::string param_name, int val);

  NodeSeq Create(std::string model_name, int n_neuron=1, int n_port=1);
  NodeSeq CreatePoissonGenerator(int n_node, float rate);
  RemoteNodeSeq RemoteCreate(int i_host, std::string model_name,
			     int n_node=1, int n_port=1);

  int CreateRecord(std::string file_name, std::string *var_name_arr,
		   int *i_node_arr, int n_node);
  int CreateRecord(std::string file_name, std::string *var_name_arr,
		   int *i_node_arr, int *port_arr, int n_node);
  std::vector<std::vector<float> > *GetRecordData(int i_record);

  int SetNeuronParam(int i_node, int n_neuron, std::string param_name,
		     float val);

  int SetNeuronParam(int *i_node, int n_neuron, std::string param_name,
		     float val);

  int SetNeuronParam(int i_node, int n_neuron, std::string param_name,
		     float *param, int array_size);

  int SetNeuronParam(int *i_node, int n_neuron, std::string param_name,
		     float *param, int array_size);

  int SetNeuronParam(NodeSeq nodes, std::string param_name, float val) {
    return SetNeuronParam(nodes.i0, nodes.n, param_name, val);
  }

  int SetNeuronParam(NodeSeq nodes, std::string param_name, float *param,
		      int array_size) {
    return SetNeuronParam(nodes.i0, nodes.n, param_name, param, array_size);
  }

  int SetNeuronParam(std::vector<int> nodes, std::string param_name,
		     float val) {
    return SetNeuronParam(nodes.data(), nodes.size(), param_name, val);
  }

  int SetNeuronParam(std::vector<int> nodes, std::string param_name,
		     float *param, int array_size) {
    return SetNeuronParam(nodes.data(), nodes.size(), param_name, param,
			  array_size);
  }

  int SetNeuronIntVar(int i_node, int n_neuron, std::string var_name,
		     int val);

  int SetNeuronIntVar(int *i_node, int n_neuron, std::string var_name,
		     int val);

  int SetNeuronIntVar(NodeSeq nodes, std::string var_name, int val) {
    return SetNeuronIntVar(nodes.i0, nodes.n, var_name, val);
  }

  int SetNeuronIntVar(std::vector<int> nodes, std::string var_name,
		     int val) {
    return SetNeuronIntVar(nodes.data(), nodes.size(), var_name, val);
  }

  int SetNeuronVar(int i_node, int n_neuron, std::string var_name,
		     float val);

  int SetNeuronVar(int *i_node, int n_neuron, std::string var_name,
		     float val);

  int SetNeuronVar(int i_node, int n_neuron, std::string var_name,
		     float *var, int array_size);

  int SetNeuronVar(int *i_node, int n_neuron, std::string var_name,
		     float *var, int array_size);

  int SetNeuronVar(NodeSeq nodes, std::string var_name, float val) {
    return SetNeuronVar(nodes.i0, nodes.n, var_name, val);
  }

  int SetNeuronVar(NodeSeq nodes, std::string var_name, float *var,
		      int array_size) {
    return SetNeuronVar(nodes.i0, nodes.n, var_name, var, array_size);
  }

  int SetNeuronVar(std::vector<int> nodes, std::string var_name,
		     float val) {
    return SetNeuronVar(nodes.data(), nodes.size(), var_name, val);
  }

  int SetNeuronVar(std::vector<int> nodes, std::string var_name,
		     float *var, int array_size) {
    return SetNeuronVar(nodes.data(), nodes.size(), var_name, var,
			  array_size);
  }

  int GetNeuronParamSize(int i_node, std::string param_name);

  int GetNeuronVarSize(int i_node, std::string var_name);

  float *GetNeuronParam(int i_node, int n_neuron, std::string param_name);

  float *GetNeuronParam(int *i_node, int n_neuron, std::string param_name);

  float *GetNeuronParam(NodeSeq nodes, std::string param_name) {
    return GetNeuronParam(nodes.i0, nodes.n, param_name);
  }

  float *GetNeuronParam(std::vector<int> nodes, std::string param_name) {
    return GetNeuronParam(nodes.data(), nodes.size(), param_name);
  }

  float *GetArrayParam(int i_node, std::string param_name);

  int *GetNeuronIntVar(int i_node, int n_neuron, std::string var_name);

  int *GetNeuronIntVar(int *i_node, int n_neuron, std::string var_name);

  int *GetNeuronIntVar(NodeSeq nodes, std::string var_name) {
    return GetNeuronIntVar(nodes.i0, nodes.n, var_name);
  }

  int *GetNeuronIntVar(std::vector<int> nodes, std::string var_name) {
    return GetNeuronIntVar(nodes.data(), nodes.size(), var_name);
  }

  float *GetNeuronVar(int i_node, int n_neuron, std::string var_name);

  float *GetNeuronVar(int *i_node, int n_neuron, std::string var_name);

  float *GetNeuronVar(NodeSeq nodes, std::string var_name) {
    return GetNeuronVar(nodes.i0, nodes.n, var_name);
  }

  float *GetNeuronVar(std::vector<int> nodes, std::string var_name) {
    return GetNeuronVar(nodes.data(), nodes.size(), var_name);
  }

  float *GetArrayVar(int i_node, std::string param_name);

  int GetNodeSequenceOffset(int i_node, int n_node, int &i_group);

  std::vector<int> GetNodeArrayWithOffset(int *i_node, int n_node,
					  int &i_group);

  int IsNeuronScalParam(int i_node, std::string param_name);

  int IsNeuronPortParam(int i_node, std::string param_name);

  int IsNeuronArrayParam(int i_node, std::string param_name);

  int IsNeuronIntVar(int i_node, std::string var_name);

  int IsNeuronScalVar(int i_node, std::string var_name);

  int IsNeuronPortVar(int i_node, std::string var_name);

  int IsNeuronArrayVar(int i_node, std::string var_name);

  int SetSpikeGenerator(int i_node, int n_spikes, float *spike_time,
			float *spike_height);

  int Calibrate();

  int Simulate();

  int Simulate(float sim_time);

  int StartSimulation();

  int SimulationStep();

  int EndSimulation();

  int ConnectMpiInit(int argc, char *argv[]);

  int MpiId();

  int MpiNp();

  int ProcMaster();

  int MpiFinalize();

  std::string MpiRankStr();

  void SetErrorFlag(bool error_flag) {error_flag_ = error_flag;}

  void SetErrorMessage(std::string error_message) { error_message_
      = error_message; }

  void SetErrorCode(unsigned char error_code) {error_code_ = error_code;}

  void SetOnException(int on_exception) {on_exception_ = on_exception;}

  bool GetErrorFlag() {return error_flag_;}

  char *GetErrorMessage() {return &error_message_[0];}

  unsigned char GetErrorCode() {return error_code_;}

  int OnException() {return on_exception_;}

  unsigned int *RandomInt(size_t n);

  float *RandomUniform(size_t n);

  float *RandomNormal(size_t n, float mean, float stddev);

  float *RandomNormalClipped(size_t n, float mean, float stddev, float vmin,
			     float vmax, float vstep);

  int Connect
    (
     int i_source_node, int i_target_node, unsigned char port,
     unsigned char syn_group, float weight, float delay
     );

  int Connect(int i_source, int n_source, int i_target, int n_target,
	      ConnSpec &conn_spec, SynSpec &syn_spec);

  int Connect(int i_source, int n_source, int* target, int n_target,
	      ConnSpec &conn_spec, SynSpec &syn_spec);

  int Connect(int* source, int n_source, int i_target, int n_target,
	      ConnSpec &conn_spec, SynSpec &syn_spec);

  int Connect(int* source, int n_source, int* target, int n_target,
	      ConnSpec &conn_spec, SynSpec &syn_spec);

  int Connect(NodeSeq source, NodeSeq target,
	      ConnSpec &conn_spec, SynSpec &syn_spec);

  int Connect(NodeSeq source, std::vector<int> target,
	      ConnSpec &conn_spec, SynSpec &syn_spec);

  int Connect(std::vector<int> source, NodeSeq target,
	      ConnSpec &conn_spec, SynSpec &syn_spec);

  int Connect(std::vector<int> source, std::vector<int> target,
	      ConnSpec &conn_spec, SynSpec &syn_spec);

  int RemoteConnect(int i_source_host, int i_source, int n_source,
		    int i_target_host, int i_target, int n_target,
		    ConnSpec &conn_spec, SynSpec &syn_spec);

  int RemoteConnect(int i_source_host, int i_source, int n_source,
		    int i_target_host, int* target, int n_target,
		    ConnSpec &conn_spec, SynSpec &syn_spec);

  int RemoteConnect(int i_source_host, int* source, int n_source,
		    int i_target_host, int i_target, int n_target,
		    ConnSpec &conn_spec, SynSpec &syn_spec);

  int RemoteConnect(int i_source_host, int* source, int n_source,
		    int i_target_host, int* target, int n_target,
		    ConnSpec &conn_spec, SynSpec &syn_spec);

  int RemoteConnect(int i_source_host, NodeSeq source,
		    int i_target_host, NodeSeq target,
		    ConnSpec &conn_spec, SynSpec &syn_spec);

  int RemoteConnect(int i_source_host, NodeSeq source,
		    int i_target_host, std::vector<int> target,
		    ConnSpec &conn_spec, SynSpec &syn_spec);

  int RemoteConnect(int i_source_host, std::vector<int> source,
		    int i_target_host, NodeSeq target,
		    ConnSpec &conn_spec, SynSpec &syn_spec);

  int RemoteConnect(int i_source_host, std::vector<int> source,
		    int i_target_host, std::vector<int> target,
		    ConnSpec &conn_spec, SynSpec &syn_spec);

  int BuildDirectConnections();

  std::vector<std::string> GetScalVarNames(int i_node);

  int GetNIntVar(int i_node);

  std::vector<std::string> GetIntVarNames(int i_node);

  int GetNScalVar(int i_node);

  std::vector<std::string> GetPortVarNames(int i_node);

  int GetNPortVar(int i_node);

  std::vector<std::string> GetScalParamNames(int i_node);

  int GetNScalParam(int i_node);

  std::vector<std::string> GetPortParamNames(int i_node);

  int GetNPortParam(int i_node);

  std::vector<std::string> GetArrayParamNames(int i_node);

  int GetNArrayParam(int i_node);

  std::vector<std::string> GetArrayVarNames(int i_node);

  std::vector<std::string> GetGroupParamNames(int i_node);

  int GetNGroupParam(int i_node);

  int GetNArrayVar(int i_node);

  ConnectionStatus GetConnectionStatus(ConnectionId conn_id);

  std::vector<ConnectionStatus> GetConnectionStatus(std::vector<ConnectionId>
						    &conn_id_vect);

  std::vector<ConnectionId> GetConnections(int i_source, int n_source,
					   int i_target, int n_target,
					   int syn_group=-1);

  std::vector<ConnectionId> GetConnections(int *i_source, int n_source,
					   int i_target, int n_target,
					   int syn_group=-1);

  std::vector<ConnectionId> GetConnections(int i_source, int n_source,
					   int *i_target, int n_target,
					   int syn_group=-1);

  std::vector<ConnectionId> GetConnections(int *i_source, int n_source,
					   int *i_target, int n_target,
					   int syn_group=-1);

  std::vector<ConnectionId> GetConnections(NodeSeq source, NodeSeq target,
					   int syn_group=-1);

  std::vector<ConnectionId> GetConnections(std::vector<int> source,
					   NodeSeq target, int syn_group=-1);

  std::vector<ConnectionId> GetConnections(NodeSeq source,
					   std::vector<int> target,
					   int syn_group=-1);

  std::vector<ConnectionId> GetConnections(std::vector<int> source,
					   std::vector<int> target,
					   int syn_group=-1);

  int CreateSynGroup(std::string model_name);

  int GetSynGroupNParam(int syn_group);

  std::vector<std::string> GetSynGroupParamNames(int syn_group);

  bool IsSynGroupParam(int syn_group, std::string param_name);

  int GetSynGroupParamIdx(int syn_group, std::string param_name);

  float GetSynGroupParam(int syn_group, std::string param_name);

  int SetSynGroupParam(int syn_group, std::string param_name, float val);

  int SynGroupCalibrate();

  int ActivateSpikeCount(int i_node, int n_node);

  int ActivateSpikeCount(NodeSeq nodes) {
    return ActivateSpikeCount(nodes.i0, nodes.n);
  }

  int ActivateRecSpikeTimes(int i_node, int n_node, int max_n_rec_spike_times);

  int ActivateRecSpikeTimes(NodeSeq nodes, int max_n_rec_spike_times) {
    return ActivateRecSpikeTimes(nodes.i0, nodes.n, max_n_rec_spike_times);
  }

  int SetRecSpikeTimesStep(int i_node, int n_node, int rec_spike_times_step);

  int SetRecSpikeTimesStep(NodeSeq nodes, int rec_spike_times_step) {
    return SetRecSpikeTimesStep(nodes.i0, nodes.n, rec_spike_times_step);
  }

  int GetNRecSpikeTimes(int i_node);

  int GetRecSpikeTimes(int i_node, int n_node, int **n_spike_times_pt,
		       float ***spike_times_pt);

  int GetRecSpikeTimes(NodeSeq nodes, int **n_spike_times_pt,
		       float ***spike_times_pt) {
    return GetRecSpikeTimes(nodes.i0, nodes.n, n_spike_times_pt,
			    spike_times_pt);
  }

  int PushSpikesToNodes(int n_spikes, int *node_id, float *spike_height);

  int PushSpikesToNodes(int n_spikes, int *node_id);

  int GetExtNeuronInputSpikes(int *n_spikes, int **node, int **port,
			      float **spike_height, bool include_zeros);

  int SetNeuronGroupParam(int i_node, int n_node,
			  std::string param_name, float val);

  int IsNeuronGroupParam(int i_node, std::string param_name);

  float GetNeuronGroupParam(int i_node, std::string param_name);

};



#endif
