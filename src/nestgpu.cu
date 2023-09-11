/*
 *  This file is part of NESTGPU.
 *
 *  Copyright (C) 2021 The NEST Initiative
 *
 *  NESTGPU is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NESTGPU is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NESTGPU.  If not, see <http://www.gnu.org/licenses/>.
 *
 */





#include <config.h>
#include <stdio.h>
#include <stdint.h>
#include <cmath>
#include <iostream>
#include <string>
#include <algorithm>
#include <curand.h>
#include "distribution.h"
#include "syn_model.h"
#include "spike_buffer.h"
#include "cuda_error.h"
#include "send_spike.h"
#include "get_spike.h"
//#include "connect_mpi.h"

#include "spike_generator.h"
#include "multimeter.h"
#include "getRealTime.h"
#include "random.h"
#include "nestgpu.h"
#include "nested_loop.h"
#include "rev_spike.h"
#include "spike_mpi.h"
#include "connect.h"
#include "poiss_gen.h"

#include "remote_connect.h"

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#define THREAD_MAXNUM omp_get_max_threads()
#define THREAD_IDX omp_get_thread_num()
#else
#define THREAD_MAXNUM 1
#define THREAD_IDX 0
#endif

				    //#define VERBOSE_TIME

__constant__ double NESTGPUTime;
__constant__ long long NESTGPUTimeIdx;
__constant__ float NESTGPUTimeResolution;

enum KernelFloatParamIndexes {
  i_time_resolution = 0,
  i_max_spike_num_fact,
  i_max_spike_per_host_fact,
  N_KERNEL_FLOAT_PARAM
};

enum KernelIntParamIndexes {
  i_rnd_seed = 0,
  i_verbosity_level,
  i_max_spike_buffer_size,
  i_remote_spike_height_flag,
  N_KERNEL_INT_PARAM
};

enum KernelBoolParamIndexes {
  i_print_time,
  N_KERNEL_BOOL_PARAM
};

const std::string kernel_float_param_name[N_KERNEL_FLOAT_PARAM] = {
  "time_resolution",
  "max_spike_num_fact",
  "max_spike_per_host_fact"
};

const std::string kernel_int_param_name[N_KERNEL_INT_PARAM] = {
  "rnd_seed",
  "verbosity_level",
  "max_spike_buffer_size",
  "remote_spike_height_flag"
};

const std::string kernel_bool_param_name[N_KERNEL_BOOL_PARAM] = {
  "print_time"
};

NESTGPU::NESTGPU()
{
  random_generator_ = new curandGenerator_t;
  CURAND_CALL(curandCreateGenerator(random_generator_,
				    CURAND_RNG_PSEUDO_DEFAULT));
  distribution_ = new Distribution;
  multimeter_ = new Multimeter;
  
  //SetRandomSeed(54321ULL);
  //SetRandomSeed(54322ULL);
  //SetRandomSeed(54323ULL);
  SetRandomSeed(54328ULL);
  
  calibrate_flag_ = false;
  create_flag_ = false;
  ConnectionSpikeTimeFlag = false;
  rev_conn_flag_ = false;
  h_NRevConn = 0;
  
  start_real_time_ = getRealTime();
  max_spike_buffer_size_ = 20;
  t_min_ = 0.0;
  sim_time_ = 1000.0;        //Simulation time in ms
  n_poiss_node_ = 0;
  n_remote_node_ = 0;
  SetTimeResolution(0.1);  // time resolution in ms
  max_spike_num_fact_ = 1.0;
  max_spike_per_host_fact_ = 1.0;
  setMaxNodeNBits(20); // maximum number of nodes is 2^20

  error_flag_ = false;
  error_message_ = "";
  error_code_ = 0;
  
  on_exception_ = ON_EXCEPTION_EXIT;

  verbosity_level_ = 4;
  print_time_ = false;
  
  mpi_flag_ = false;
#ifdef HAVE_MPI
  //connect_mpi_ = new ConnectMpi;
  //connect_mpi_->remote_spike_height_ = false;
#endif

  RemoteConnectionMapInit(4); // (uint n_hosts)
  // TEMPORARY, REMOVE!!!!!!!!!!!!!!!!!
  int n_neurons = 30;
  int CE = 3;
  Create("iaf_psc_exp", n_neurons);

  float mean_delay = 0.5;
  float std_delay = 0.25;
  float min_delay = 0.1;
  float w = 1.0;

  ConnSpec conn_spec1(FIXED_INDEGREE, CE);
  SynSpec syn_spec1;
  syn_spec1.SetParam("receptor", 0);
  syn_spec1.SetParam("weight", w);
  syn_spec1.SetParam("delay_distribution", DISTR_TYPE_NORMAL_CLIPPED);
  syn_spec1.SetParam("delay_mu", mean_delay);
  syn_spec1.SetParam("delay_low", min_delay);
  syn_spec1.SetParam("delay_high", mean_delay+3*std_delay);
  syn_spec1.SetParam("delay_sigma", std_delay);

  const int n_source = 10;
  int h_source_node_index[n_source] = {21, 24, 21, 24, 22, 21, 23, 25, 26, 22};
  int *d_source_node_index;
  gpuErrchk(cudaMalloc(&d_source_node_index, n_source*sizeof(int)));
  gpuErrchk(cudaMemcpy(d_source_node_index, h_source_node_index,
		       n_source*sizeof(int), cudaMemcpyHostToDevice));

  //_RemoteConnectSource(1, 20, 10, 10, 3, conn_spec1, syn_spec1);
  _RemoteConnectSource(1, d_source_node_index, 10, 10, 3, conn_spec1, syn_spec1);

  std::cout << "##################################################\n";
  std::cout << "##################################################\n";
  std::cout << "SECOND CONNECT COMMAND\n";
  std::cout << "##################################################\n";
  std::cout << "##################################################\n";
  _RemoteConnectSource(1, 20, 10, 10, 3, conn_spec1, syn_spec1);

  
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  
  // NestedLoop::Init(); moved to calibrate
  nested_loop_algo_ = CumulSumNestedLoopAlgo;

  SpikeBufferUpdate_time_ = 0;
  poisson_generator_time_ = 0;
  neuron_Update_time_ = 0;
  copy_ext_spike_time_ = 0;
  SendExternalSpike_time_ = 0;
  SendSpikeToRemote_time_ = 0;
  RecvSpikeFromRemote_time_ = 0;
  NestedLoop_time_ = 0;
  GetSpike_time_ = 0;
  SpikeReset_time_ = 0;
  ExternalSpikeReset_time_ = 0;

  first_simulation_flag_ = true;
}

NESTGPU::~NESTGPU()
{
  multimeter_->CloseFiles();
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  if (calibrate_flag_) {
    FreeNodeGroupMap();
    FreeGetSpikeArrays();
  }

  for (unsigned int i=0; i<node_vect_.size(); i++) {
    delete node_vect_[i];
  }

#ifdef HAVE_MPI
  //delete connect_mpi_;
#endif

  delete multimeter_;
  curandDestroyGenerator(*random_generator_);
  delete random_generator_;
}

int NESTGPU::SetRandomSeed(unsigned long long seed)
{
  kernel_seed_ = seed + 12345;
  CURAND_CALL(curandDestroyGenerator(*random_generator_));
  random_generator_ = new curandGenerator_t;
  CURAND_CALL(curandCreateGenerator(random_generator_,
				    CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(*random_generator_, seed));
  distribution_->setCurandGenerator(random_generator_);
  
  return 0;
}

int NESTGPU::SetTimeResolution(float time_res)
{
  time_resolution_ = time_res;
  
  return 0;
}

int NESTGPU::SetNestedLoopAlgo(int nested_loop_algo)
{
  nested_loop_algo_ = nested_loop_algo;
  
  return 0;
}

int NESTGPU::SetMaxSpikeBufferSize(int max_size)
{
  max_spike_buffer_size_ = max_size;
  
  return 0;
}

int NESTGPU::GetMaxSpikeBufferSize()
{
  return max_spike_buffer_size_;
}

uint NESTGPU::GetNNode()
{
  return node_group_map_.size();
}

int NESTGPU::CreateNodeGroup(int n_node, int n_port)
{
  int i_node_0 = GetNNode();
  int max_n_neurons = IntPow(2,h_MaxNodeNBits);
  int max_n_ports = IntPow(2,h_MaxPortNBits);
  
  if ((i_node_0 + n_node) > max_n_neurons) {
    throw ngpu_exception(std::string("Maximum number of neurons ")
			 + std::to_string(max_n_neurons) + " exceeded");
  }
  if (n_port > max_n_ports) {
    throw ngpu_exception(std::string("Maximum number of ports ")
			 + std::to_string(max_n_ports) + " exceeded");
  }
  int i_group = node_vect_.size() - 1;
  node_group_map_.insert(node_group_map_.end(), n_node, i_group);
    
  node_vect_[i_group]->Init(i_node_0, n_node, n_port, i_group, &kernel_seed_);
  node_vect_[i_group]->get_spike_array_ = InitGetSpikeArray(n_node, n_port);
  
  return i_node_0;
}

int NESTGPU::CheckUncalibrated(std::string message)
{
  if (calibrate_flag_ == true) {
    throw ngpu_exception(message);
  }
  
  return 0;
}

int NESTGPU::Calibrate()
{
  CheckUncalibrated("Calibration can be made only once");

  gpuErrchk(cudaMemcpyToSymbol(NESTGPUTimeResolution, &time_resolution_,
			       sizeof(float)));
///////////////////////////////////
 
  organizeConnections(time_resolution_, GetNNode(),
		      NConn, h_ConnBlockSize,
		      KeySubarray, ConnectionSubarray);
  NewConnectInit();

  poiss_conn::OrganizeDirectConnections();

  int max_delay_num = h_MaxDelayNum;
  
  unsigned int n_spike_buffers = GetNNode();
  NestedLoop::Init(n_spike_buffers);
		   
  ConnectRemoteNodes();
  // temporary
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  
  calibrate_flag_ = true;
  
  //gpuErrchk(cudaMemcpyToSymbolAsync(NESTGPUMpiFlag, &mpi_flag_,
  // sizeof(bool)));

  if (verbosity_level_>=1) {
    std::cout << MpiRankStr() << "Calibrating ...\n";
  }
  
  neural_time_ = t_min_;
  	    
  NodeGroupArrayInit();
  
  max_spike_num_ = (int)round(max_spike_num_fact_
			      * GetNNode()
			      * max_delay_num);
  max_spike_num_ = (max_spike_num_>1) ? max_spike_num_ : 1;

  max_spike_per_host_ = (int)round(max_spike_per_host_fact_
				   * GetNNode()
				   * max_delay_num);
  max_spike_per_host_ = (max_spike_per_host_>1) ? max_spike_per_host_ : 1;
  
  SpikeInit(max_spike_num_);
  SpikeBufferInit(GetNNode(), max_spike_buffer_size_);
  
#ifdef HAVE_MPI
  /*
  if (mpi_flag_) {
    // remove superfluous argument mpi_np
    connect_mpi_->ExternalSpikeInit(connect_mpi_->extern_connection_.size(),
				    connect_mpi_->mpi_np_,
				    max_spike_per_host_);
  }
  */
#endif
  if (rev_conn_flag_) {
    RevSpikeInit(GetNNode());
  }
 
  multimeter_->OpenFiles();
  
  for (unsigned int i=0; i<node_vect_.size(); i++) {
    node_vect_[i]->Calibrate(t_min_, time_resolution_);
  }
  
  SynGroupCalibrate();
  
  gpuErrchk(cudaMemcpyToSymbolAsync(NESTGPUTimeResolution, &time_resolution_,
				    sizeof(float)));

  return 0;
}

int NESTGPU::Simulate(float sim_time) {
  sim_time_ = sim_time;
  return Simulate();
}

int NESTGPU::Simulate()
{
  StartSimulation();
  
  for (long long it=0; it<Nt_; it++) {
    if (it%100==0 && verbosity_level_>=2 && print_time_==true) {
      printf("\r[%.2lf %%] Model time: %.3lf ms", 100.0*(neural_time_-neur_t0_)/sim_time_, neural_time_);
    }
    SimulationStep();
  }
  EndSimulation();

  return 0;
}

int NESTGPU::StartSimulation()
{
  if (!calibrate_flag_) {
    Calibrate();
  }
#ifdef HAVE_MPI                                                                                                            
  if (mpi_flag_) {
    MPI_Barrier(MPI_COMM_WORLD);
  }
#endif
  if (first_simulation_flag_) {
    gpuErrchk(cudaMemcpyToSymbolAsync(NESTGPUTime, &neural_time_, sizeof(double)));
    multimeter_->WriteRecords(neural_time_);
    build_real_time_ = getRealTime();
    first_simulation_flag_ = false;
  }
  if (verbosity_level_>=1) {
    std::cout << MpiRankStr() << "Simulating ...\n";
    printf("Neural activity simulation time: %.3lf ms\n", sim_time_);
  }
  
  neur_t0_ = neural_time_;
  it_ = 0;
  Nt_ = (long long)round(sim_time_/time_resolution_);
  
  return 0;
}

int NESTGPU::EndSimulation()
{
  if (verbosity_level_>=2  && print_time_==true) {
    printf("\r[%.2lf %%] Model time: %.3lf ms", 100.0*(neural_time_-neur_t0_)/sim_time_, neural_time_);
  }
#ifdef HAVE_MPI                                        
  if (mpi_flag_) {
    MPI_Barrier(MPI_COMM_WORLD);
  }
#endif

  end_real_time_ = getRealTime();

  //multimeter_->CloseFiles();
  //neuron.rk5.Free();

  if (verbosity_level_>=3) {
    std::cout << "\n";
    std::cout << MpiRankStr() << "  SpikeBufferUpdate_time: " <<
      SpikeBufferUpdate_time_ << "\n";
    std::cout << MpiRankStr() << "  poisson_generator_time: " <<
      poisson_generator_time_ << "\n";
    std::cout << MpiRankStr() << "  neuron_Update_time: " <<
      neuron_Update_time_ << "\n";
    std::cout << MpiRankStr() << "  copy_ext_spike_time: " <<
      copy_ext_spike_time_ << "\n";
    std::cout << MpiRankStr() << "  SendExternalSpike_time: " <<
      SendExternalSpike_time_ << "\n";
    std::cout << MpiRankStr() << "  SendSpikeToRemote_time: " <<
      SendSpikeToRemote_time_ << "\n";
    std::cout << MpiRankStr() << "  RecvSpikeFromRemote_time: " <<
      RecvSpikeFromRemote_time_ << "\n";
    std::cout << MpiRankStr() << "  NestedLoop_time: " <<
      NestedLoop_time_ << "\n";
    std::cout << MpiRankStr() << "  GetSpike_time: " <<
      GetSpike_time_ << "\n";
    std::cout << MpiRankStr() << "  SpikeReset_time: " <<
      SpikeReset_time_ << "\n";
    std::cout << MpiRankStr() << "  ExternalSpikeReset_time: " <<
      ExternalSpikeReset_time_ << "\n";
  }
  /*
  if (mpi_flag_ && verbosity_level_>=4) {
    std::cout << MpiRankStr() << "  SendSpikeToRemote_MPI_time: " <<
      connect_mpi_->SendSpikeToRemote_MPI_time_ << "\n";
    std::cout << MpiRankStr() << "  RecvSpikeFromRemote_MPI_time: " <<
      connect_mpi_->RecvSpikeFromRemote_MPI_time_ << "\n";
    std::cout << MpiRankStr() << "  SendSpikeToRemote_CUDAcp_time: " <<
      connect_mpi_->SendSpikeToRemote_CUDAcp_time_  << "\n";
    std::cout << MpiRankStr() << "  RecvSpikeFromRemote_CUDAcp_time: " <<
      connect_mpi_->RecvSpikeFromRemote_CUDAcp_time_  << "\n";
    std::cout << MpiRankStr() << "  JoinSpike_time: " <<
      connect_mpi_->JoinSpike_time_  << "\n";
  }
  */
  if (verbosity_level_>=1) {
    std::cout << MpiRankStr() << "Building time: " <<
      (build_real_time_ - start_real_time_) << "\n";
    std::cout << MpiRankStr() << "Simulation time: " <<
      (end_real_time_ - build_real_time_) << "\n";
  }

  return 0;
}


int NESTGPU::SimulationStep()
{
  if (first_simulation_flag_) {
    StartSimulation();
  }
  double time_mark;

  time_mark = getRealTime();
  SpikeBufferUpdate<<<(GetNNode()+1023)/1024, 1024>>>();
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  SpikeBufferUpdate_time_ += (getRealTime() - time_mark);
  time_mark = getRealTime();
  neural_time_ = neur_t0_ + (double)time_resolution_*(it_+1);
  gpuErrchk(cudaMemcpyToSymbolAsync(NESTGPUTime, &neural_time_, sizeof(double)));
  long long time_idx = (int)round(neur_t0_/time_resolution_) + it_ + 1;
  gpuErrchk(cudaMemcpyToSymbolAsync(NESTGPUTimeIdx, &time_idx, sizeof(long long)));

  /*
  if (ConnectionSpikeTimeFlag) {
    if ( (time_idx & 0xffff) == 0x8000) {
      ResetConnectionSpikeTimeUp(net_connection_);
    }
    else if ( (time_idx & 0xffff) == 0) {
      ResetConnectionSpikeTimeDown(net_connection_);
    }
  }
  */
  
  for (unsigned int i=0; i<node_vect_.size(); i++) {
    node_vect_[i]->Update(it_, neural_time_);
  }
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  
  neuron_Update_time_ += (getRealTime() - time_mark);
  multimeter_->WriteRecords(neural_time_);
  /*
#ifdef HAVE_MPI
  if (mpi_flag_) {
    int n_ext_spike;
    time_mark = getRealTime();
    gpuErrchk(cudaMemcpy(&n_ext_spike, d_ExternalSpikeNum, sizeof(int),
			 cudaMemcpyDeviceToHost));
    copy_ext_spike_time_ += (getRealTime() - time_mark);

    if (n_ext_spike != 0) {
      time_mark = getRealTime();
      SendExternalSpike<<<(n_ext_spike+1023)/1024, 1024>>>();
      gpuErrchk( cudaPeekAtLastError() );
      gpuErrchk( cudaDeviceSynchronize() );
      SendExternalSpike_time_ += (getRealTime() - time_mark);
    }
    //for (int ih=0; ih<connect_mpi_->mpi_np_; ih++) {
    //if (ih == connect_mpi_->mpi_id_) {

    time_mark = getRealTime();
    connect_mpi_->SendSpikeToRemote(connect_mpi_->mpi_np_,
				    max_spike_per_host_);
    SendSpikeToRemote_time_ += (getRealTime() - time_mark);
    time_mark = getRealTime();
    connect_mpi_->RecvSpikeFromRemote(connect_mpi_->mpi_np_,
				      max_spike_per_host_);
				      
    RecvSpikeFromRemote_time_ += (getRealTime() - time_mark);
    connect_mpi_->CopySpikeFromRemote(connect_mpi_->mpi_np_,
				      max_spike_per_host_,
				      i_remote_node_0_);
    MPI_Barrier(MPI_COMM_WORLD);
 
  }
#endif
*/    
  int n_spikes;
  time_mark = getRealTime();
  gpuErrchk(cudaMemcpy(&n_spikes, d_SpikeNum, sizeof(int),
		       cudaMemcpyDeviceToHost));

  ClearGetSpikeArrays();    
  if (n_spikes > 0) {
    time_mark = getRealTime();
    NestedLoop::Run<0>(nested_loop_algo_, n_spikes, d_SpikeTargetNum);
    NestedLoop_time_ += (getRealTime() - time_mark);
  }
  time_mark = getRealTime();
  for (unsigned int i=0; i<node_vect_.size(); i++) {
    if (node_vect_[i]->has_dir_conn_) {
      node_vect_[i]->SendDirectSpikes(time_idx);
    }
  }
  poisson_generator_time_ += (getRealTime() - time_mark);
  time_mark = getRealTime();
  for (unsigned int i=0; i<node_vect_.size(); i++) {
    if (node_vect_[i]->n_port_>0) {

      int grid_dim_x = (node_vect_[i]->n_node_+1023)/1024;
      int grid_dim_y = node_vect_[i]->n_port_;
      dim3 grid_dim(grid_dim_x, grid_dim_y);
      //dim3 block_dim(1024,1);
					    
      GetSpikes<<<grid_dim, 1024>>> //block_dim>>>
	(node_vect_[i]->get_spike_array_, node_vect_[i]->n_node_,
	 node_vect_[i]->n_port_,
	 node_vect_[i]->n_var_,
	 node_vect_[i]->port_weight_arr_,
	 node_vect_[i]->port_weight_arr_step_,
	 node_vect_[i]->port_weight_port_step_,
	 node_vect_[i]->port_input_arr_,
	 node_vect_[i]->port_input_arr_step_,
	 node_vect_[i]->port_input_port_step_);
      // gpuErrchk( cudaPeekAtLastError() );
      // gpuErrchk( cudaDeviceSynchronize() );
    }
  }
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  GetSpike_time_ += (getRealTime() - time_mark);

  time_mark = getRealTime();
  SpikeReset<<<1, 1>>>();
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  SpikeReset_time_ += (getRealTime() - time_mark);

  /*
#ifdef HAVE_MPI
  if (mpi_flag_) {
    time_mark = getRealTime();
    ExternalSpikeReset<<<1, 1>>>();
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    ExternalSpikeReset_time_ += (getRealTime() - time_mark);
  }
#endif
  */
  if (h_NRevConn > 0) {
    //time_mark = getRealTime();
    RevSpikeReset<<<1, 1>>>();
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    RevSpikeBufferUpdate<<<(GetNNode()+1023)/1024, 1024>>>(GetNNode());
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    unsigned int n_rev_spikes;
    gpuErrchk(cudaMemcpy(&n_rev_spikes, d_RevSpikeNum, sizeof(unsigned int),
			 cudaMemcpyDeviceToHost));
    if (n_rev_spikes > 0) {
      NestedLoop::Run<1>(nested_loop_algo_, n_rev_spikes, d_RevSpikeNConn);
    }      
    //RevSpikeBufferUpdate_time_ += (getRealTime() - time_mark);
  }

  for (unsigned int i=0; i<node_vect_.size(); i++) {
    // if spike times recording is activated for node group...
    if (node_vect_[i]->max_n_rec_spike_times_>0) {
      // and if buffering is activated every n_step time steps...
      int n_step = node_vect_[i]->rec_spike_times_step_;
      if (n_step>0 && (time_idx%n_step == n_step-1)) {
	// extract recorded spike times and put them in buffers
	node_vect_[i]->BufferRecSpikeTimes();
      }
    }
  }

  it_++;
  
  return 0;
}

int NESTGPU::CreateRecord(std::string file_name, std::string *var_name_arr,
			    int *i_node_arr, int *port_arr,
			    int n_node)
{
  std::vector<BaseNeuron*> neur_vect;
  std::vector<int> i_neur_vect;
  std::vector<int> port_vect;
  std::vector<std::string> var_name_vect;
  for (int i=0; i<n_node; i++) {
    var_name_vect.push_back(var_name_arr[i]);
    int i_group = node_group_map_[i_node_arr[i]];
    i_neur_vect.push_back(i_node_arr[i] - node_vect_[i_group]->i_node_0_);
    port_vect.push_back(port_arr[i]);
    neur_vect.push_back(node_vect_[i_group]);
  }

  return multimeter_->CreateRecord(neur_vect, file_name, var_name_vect,
  				   i_neur_vect, port_vect);

}

int NESTGPU::CreateRecord(std::string file_name, std::string *var_name_arr,
			    int *i_node_arr, int n_node)
{
  std::vector<int> port_vect(n_node, 0);
  return CreateRecord(file_name, var_name_arr, i_node_arr,
		      port_vect.data(), n_node);
}

std::vector<std::vector<float> > *NESTGPU::GetRecordData(int i_record)
{
  return multimeter_->GetRecordData(i_record);
}

int NESTGPU::GetNodeSequenceOffset(int i_node, int n_node, int &i_group)
{
  if (i_node<0 || (i_node+n_node > (int)node_group_map_.size())) {
    throw ngpu_exception("Unrecognized node in getting node sequence offset");
  }
  i_group = node_group_map_[i_node];  
  if (node_group_map_[i_node+n_node-1] != i_group) {
    throw ngpu_exception("Nodes belong to different node groups "
			 "in setting parameter");
  }
  return node_vect_[i_group]->i_node_0_;
}
  
std::vector<int> NESTGPU::GetNodeArrayWithOffset(int *i_node, int n_node,
						   int &i_group)
{
  int in0 = i_node[0];
  if (in0<0 || in0>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in setting parameter");
  }
  i_group = node_group_map_[in0];
  int i0 = node_vect_[i_group]->i_node_0_;
  std::vector<int> nodes;
  nodes.assign(i_node, i_node+n_node);
  for(int i=0; i<n_node; i++) {
    int in = nodes[i];
    if (in<0 || in>=(int)node_group_map_.size()) {
      throw ngpu_exception("Unrecognized node in setting parameter");
    }
    if (node_group_map_[in] != i_group) {
      throw ngpu_exception("Nodes belong to different node groups "
			   "in setting parameter");
    }
    nodes[i] -= i0;
  }
  return nodes;
}

int NESTGPU::SetNeuronParam(int i_node, int n_node,
			      std::string param_name, float val)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, n_node, i_group);
  
  return node_vect_[i_group]->SetScalParam(i_neuron, n_node, param_name, val);
}

int NESTGPU::SetNeuronParam(int *i_node, int n_node,
			      std::string param_name, float val)
{
  int i_group;
  std::vector<int> nodes = GetNodeArrayWithOffset(i_node, n_node,
						  i_group);
  return node_vect_[i_group]->SetScalParam(nodes.data(), n_node,
					   param_name, val);
}

int NESTGPU::SetNeuronParam(int i_node, int n_node, std::string param_name,
			      float *param, int array_size)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, n_node, i_group);
  if (node_vect_[i_group]->IsPortParam(param_name)) {
      return node_vect_[i_group]->SetPortParam(i_neuron, n_node, param_name,
					       param, array_size);
  }
  else {
    return node_vect_[i_group]->SetArrayParam(i_neuron, n_node, param_name,
					      param, array_size);
  }
}

int NESTGPU::SetNeuronParam( int *i_node, int n_node,
			       std::string param_name, float *param,
			       int array_size)
{
  int i_group;
  std::vector<int> nodes = GetNodeArrayWithOffset(i_node, n_node,
						  i_group);
  if (node_vect_[i_group]->IsPortParam(param_name)) {  
    return node_vect_[i_group]->SetPortParam(nodes.data(), n_node,
					     param_name, param, array_size);
  }
  else {
    return node_vect_[i_group]->SetArrayParam(nodes.data(), n_node,
					      param_name, param, array_size);
  }    
}

////////////////////////////////////////////////////////////////////////

int NESTGPU::SetNeuronScalParamDistr(int i_node, int n_node,
				     std::string param_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, n_node, i_group);
  
  return node_vect_[i_group]->SetScalParamDistr(i_neuron, n_node, param_name,
						distribution_);
}

int NESTGPU::SetNeuronScalVarDistr(int i_node, int n_node,
				   std::string var_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, n_node, i_group);
  
  return node_vect_[i_group]->SetScalVarDistr(i_neuron, n_node, var_name,
						distribution_);
}

int NESTGPU::SetNeuronPortParamDistr(int i_node, int n_node,
				     std::string param_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, n_node, i_group);
  
  return node_vect_[i_group]->SetPortParamDistr(i_neuron, n_node, param_name,
						distribution_);
}

int NESTGPU::SetNeuronPortVarDistr(int i_node, int n_node,
				   std::string var_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, n_node, i_group);
  
  return node_vect_[i_group]->SetPortVarDistr(i_neuron, n_node, var_name,
						distribution_);
}

int NESTGPU::SetNeuronPtScalParamDistr(int *i_node, int n_node,
				       std::string param_name)
{
  int i_group;
  std::vector<int> nodes = GetNodeArrayWithOffset(i_node, n_node,
						  i_group);
  return node_vect_[i_group]->SetScalParamDistr(nodes.data(), n_node,
						param_name, distribution_);
}

int NESTGPU::SetNeuronPtScalVarDistr(int *i_node, int n_node,
				       std::string var_name)
{
  int i_group;
  std::vector<int> nodes = GetNodeArrayWithOffset(i_node, n_node,
						  i_group);
  return node_vect_[i_group]->SetScalVarDistr(nodes.data(), n_node,
						var_name, distribution_);
}

int NESTGPU::SetNeuronPtPortParamDistr(int *i_node, int n_node,
				       std::string param_name)
{
  int i_group;
  std::vector<int> nodes = GetNodeArrayWithOffset(i_node, n_node,
						  i_group);
  return node_vect_[i_group]->SetPortParamDistr(nodes.data(), n_node,
						param_name, distribution_);
}

int NESTGPU::SetNeuronPtPortVarDistr(int *i_node, int n_node,
				       std::string var_name)
{
  int i_group;
  std::vector<int> nodes = GetNodeArrayWithOffset(i_node, n_node,
						  i_group);
  return node_vect_[i_group]->SetPortVarDistr(nodes.data(), n_node,
						var_name, distribution_);
}

int NESTGPU::SetDistributionIntParam(std::string param_name, int val)
{
  return distribution_->SetIntParam(param_name, val);
}

int NESTGPU::SetDistributionScalParam(std::string param_name, float val)
{
  return distribution_->SetScalParam(param_name, val);
}

int NESTGPU::SetDistributionVectParam(std::string param_name, float val, int i)
{
  return distribution_->SetVectParam(param_name, val, i);
}

int NESTGPU::SetDistributionFloatPtParam(std::string param_name,
					 float *array_pt)
{
  return distribution_->SetFloatPtParam(param_name, array_pt);
}

int NESTGPU::IsDistributionFloatParam(std::string param_name)
{
  return distribution_->IsFloatParam(param_name);
}

////////////////////////////////////////////////////////////////////////

int NESTGPU::IsNeuronScalParam(int i_node, std::string param_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);
  
  return node_vect_[i_group]->IsScalParam(param_name);
}

int NESTGPU::IsNeuronPortParam(int i_node, std::string param_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);
  
  return node_vect_[i_group]->IsPortParam(param_name);
}

int NESTGPU::IsNeuronArrayParam(int i_node, std::string param_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);
  
  return node_vect_[i_group]->IsArrayParam(param_name);
}

int NESTGPU::SetNeuronIntVar(int i_node, int n_node,
			      std::string var_name, int val)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, n_node, i_group);
  
  return node_vect_[i_group]->SetIntVar(i_neuron, n_node, var_name, val);
}

int NESTGPU::SetNeuronIntVar(int *i_node, int n_node,
			      std::string var_name, int val)
{
  int i_group;
  std::vector<int> nodes = GetNodeArrayWithOffset(i_node, n_node,
						  i_group);
  return node_vect_[i_group]->SetIntVar(nodes.data(), n_node,
					var_name, val);
}

int NESTGPU::SetNeuronVar(int i_node, int n_node,
			      std::string var_name, float val)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, n_node, i_group);
  
  return node_vect_[i_group]->SetScalVar(i_neuron, n_node, var_name, val);
}

int NESTGPU::SetNeuronVar(int *i_node, int n_node,
			      std::string var_name, float val)
{
  int i_group;
  std::vector<int> nodes = GetNodeArrayWithOffset(i_node, n_node,
						  i_group);
  return node_vect_[i_group]->SetScalVar(nodes.data(), n_node,
					   var_name, val);
}

int NESTGPU::SetNeuronVar(int i_node, int n_node, std::string var_name,
			      float *var, int array_size)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, n_node, i_group);
  if (node_vect_[i_group]->IsPortVar(var_name)) {
      return node_vect_[i_group]->SetPortVar(i_neuron, n_node, var_name,
					       var, array_size);
  }
  else {
    return node_vect_[i_group]->SetArrayVar(i_neuron, n_node, var_name,
					      var, array_size);
  }
}

int NESTGPU::SetNeuronVar( int *i_node, int n_node,
			       std::string var_name, float *var,
			       int array_size)
{
  int i_group;
  std::vector<int> nodes = GetNodeArrayWithOffset(i_node, n_node,
						  i_group);
  if (node_vect_[i_group]->IsPortVar(var_name)) {  
    return node_vect_[i_group]->SetPortVar(nodes.data(), n_node,
					   var_name, var, array_size);
  }
  else {
    return node_vect_[i_group]->SetArrayVar(nodes.data(), n_node,
					    var_name, var, array_size);
  }    
}

int NESTGPU::IsNeuronIntVar(int i_node, std::string var_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);
  
  return node_vect_[i_group]->IsIntVar(var_name);
}

int NESTGPU::IsNeuronScalVar(int i_node, std::string var_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);
  
  return node_vect_[i_group]->IsScalVar(var_name);
}

int NESTGPU::IsNeuronPortVar(int i_node, std::string var_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);
  
  return node_vect_[i_group]->IsPortVar(var_name);
}

int NESTGPU::IsNeuronArrayVar(int i_node, std::string var_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);
  
  return node_vect_[i_group]->IsArrayVar(var_name);
}


int NESTGPU::GetNeuronParamSize(int i_node, std::string param_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);
  if (node_vect_[i_group]->IsArrayParam(param_name)!=0) {
    return node_vect_[i_group]->GetArrayParamSize(i_neuron, param_name);
  }
  else {
    return node_vect_[i_group]->GetParamSize(param_name);
  }
}

int NESTGPU::GetNeuronVarSize(int i_node, std::string var_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);
  if (node_vect_[i_group]->IsArrayVar(var_name)!=0) {
    return node_vect_[i_group]->GetArrayVarSize(i_neuron, var_name);
  }
  else {
    return node_vect_[i_group]->GetVarSize(var_name);
  }
}


float *NESTGPU::GetNeuronParam(int i_node, int n_node,
				 std::string param_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, n_node, i_group);
  if (node_vect_[i_group]->IsScalParam(param_name)) {
    return node_vect_[i_group]->GetScalParam(i_neuron, n_node, param_name);
  }
  else if (node_vect_[i_group]->IsPortParam(param_name)) {
    return node_vect_[i_group]->GetPortParam(i_neuron, n_node, param_name);
  }
  else if (node_vect_[i_group]->IsArrayParam(param_name)) {
    if (n_node != 1) {
      throw ngpu_exception("Cannot get array parameters for more than one node"
			   "at a time");
    }
    return node_vect_[i_group]->GetArrayParam(i_neuron, param_name);
  }
  else {
    throw ngpu_exception(std::string("Unrecognized parameter ")
			 + param_name);
  }
}

float *NESTGPU::GetNeuronParam( int *i_node, int n_node,
				  std::string param_name)
{
  int i_group;
  std::vector<int> nodes = GetNodeArrayWithOffset(i_node, n_node,
						  i_group);
  if (node_vect_[i_group]->IsScalParam(param_name)) {
    return node_vect_[i_group]->GetScalParam(nodes.data(), n_node,
					     param_name);
  }
  else if (node_vect_[i_group]->IsPortParam(param_name)) {  
    return node_vect_[i_group]->GetPortParam(nodes.data(), n_node,
					     param_name);
  }
  else if (node_vect_[i_group]->IsArrayParam(param_name)) {
    if (n_node != 1) {
      throw ngpu_exception("Cannot get array parameters for more than one node"
			   "at a time");
    }
    return node_vect_[i_group]->GetArrayParam(nodes[0], param_name);
  }
  else {
    throw ngpu_exception(std::string("Unrecognized parameter ")
			 + param_name);
  }
}

float *NESTGPU::GetArrayParam(int i_node, std::string param_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);

  return node_vect_[i_group]->GetArrayParam(i_neuron, param_name);
}

int *NESTGPU::GetNeuronIntVar(int i_node, int n_node,
				std::string var_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, n_node, i_group);
  if (node_vect_[i_group]->IsIntVar(var_name)) {
    return node_vect_[i_group]->GetIntVar(i_neuron, n_node, var_name);
  }
  else {
    throw ngpu_exception(std::string("Unrecognized integer variable ")
			 + var_name);
  }
}

int *NESTGPU::GetNeuronIntVar(int *i_node, int n_node,
			       std::string var_name)
{
  int i_group;
  std::vector<int> nodes = GetNodeArrayWithOffset(i_node, n_node,
						  i_group);
  if (node_vect_[i_group]->IsIntVar(var_name)) {
    return node_vect_[i_group]->GetIntVar(nodes.data(), n_node,
					     var_name);
  }
  else {
    throw ngpu_exception(std::string("Unrecognized variable ")
			 + var_name);
  }
}

float *NESTGPU::GetNeuronVar(int i_node, int n_node,
			       std::string var_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, n_node, i_group);
  if (node_vect_[i_group]->IsScalVar(var_name)) {
    return node_vect_[i_group]->GetScalVar(i_neuron, n_node, var_name);
  }
  else if (node_vect_[i_group]->IsPortVar(var_name)) {
    return node_vect_[i_group]->GetPortVar(i_neuron, n_node, var_name);
  }
  else if (node_vect_[i_group]->IsArrayVar(var_name)) {
    if (n_node != 1) {
      throw ngpu_exception("Cannot get array variables for more than one node"
			   "at a time");
    }
    return node_vect_[i_group]->GetArrayVar(i_neuron, var_name);
  }
  else {
    throw ngpu_exception(std::string("Unrecognized variable ")
			 + var_name);
  }
}

float *NESTGPU::GetNeuronVar(int *i_node, int n_node,
			       std::string var_name)
{
  int i_group;
  std::vector<int> nodes = GetNodeArrayWithOffset(i_node, n_node,
						  i_group);
  if (node_vect_[i_group]->IsScalVar(var_name)) {
    return node_vect_[i_group]->GetScalVar(nodes.data(), n_node,
					     var_name);
  }
  else if (node_vect_[i_group]->IsPortVar(var_name)) {  
    return node_vect_[i_group]->GetPortVar(nodes.data(), n_node,
					   var_name);
  }
  else if (node_vect_[i_group]->IsArrayVar(var_name)) {
    if (n_node != 1) {
      throw ngpu_exception("Cannot get array variables for more than one node"
			   "at a time");
    }
    return node_vect_[i_group]->GetArrayVar(nodes[0], var_name);
  }
  else {
    throw ngpu_exception(std::string("Unrecognized variable ")
			 + var_name);
  }
}

float *NESTGPU::GetArrayVar(int i_node, std::string var_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);

  return node_vect_[i_group]->GetArrayVar(i_neuron, var_name);
}

int NESTGPU::ConnectMpiInit(int argc, char *argv[])
{
#ifdef HAVE_MPI
  CheckUncalibrated("MPI connections cannot be initialized after calibration");
  /*
  int err = connect_mpi_->MpiInit(argc, argv);
  if (err==0) {
    mpi_flag_ = true;
  }
  
  return err;
  */
  return 0;
#else
  throw ngpu_exception("MPI is not available in your build");
#endif
}

int NESTGPU::MpiId()
{
#ifdef HAVE_MPI
  /*
  return connect_mpi_->mpi_id_;
  */
  return 0;
#else
  throw ngpu_exception("MPI is not available in your build");
#endif
}

int NESTGPU::MpiNp()
{
#ifdef HAVE_MPI
  /*
  return connect_mpi_->mpi_np_;
  */
  return 0;
#else
  throw ngpu_exception("MPI is not available in your build");
#endif

}

int NESTGPU::ProcMaster()
{
#ifdef HAVE_MPI
  //return connect_mpi_->ProcMaster();
  return 0;
#else
  throw ngpu_exception("MPI is not available in your build");
#endif  
}

int NESTGPU::MpiFinalize()
{
#ifdef HAVE_MPI
  if (mpi_flag_) {
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized) {
      MPI_Finalize();
    }
  }
  
  return 0;
#else
  throw ngpu_exception("MPI is not available in your build");
#endif
}

std::string NESTGPU::MpiRankStr()
{
  /*
  if (mpi_flag_) {
    return std::string("MPI rank ") + std::to_string(connect_mpi_->mpi_id_)
      + " : ";
  }
  else {
  */
    return "";
    //}
}

unsigned int *NESTGPU::RandomInt(size_t n)
{
  return curand_int(*random_generator_, n);
}

float *NESTGPU::RandomUniform(size_t n)
{
  return curand_uniform(*random_generator_, n);
}

float *NESTGPU::RandomNormal(size_t n, float mean, float stddev)
{
  return curand_normal(*random_generator_, n, mean, stddev);
}

float *NESTGPU::RandomNormalClipped(size_t n, float mean, float stddev,
				      float vmin, float vmax, float vstep)
{
  const float epsi = 1.0e-6;
  
  n = (n/4 + 1)*4; 
  int n_extra = n/10;
  n_extra = (n_extra/4 + 1)*4; 
  if (n_extra<1024) {
    n_extra=1024;
  }
  int i_extra = 0;
  float *arr = curand_normal(*random_generator_, n, mean, stddev);
  float *arr_extra = NULL;
  for (size_t i=0; i<n; i++) {
    while (arr[i]<vmin || arr[i]>vmax) {
      if (i_extra==0) {
	arr_extra = curand_normal(*random_generator_, n_extra, mean, stddev);
      }
      arr[i] = arr_extra[i_extra];
      i_extra++;
      if (i_extra==n_extra) {
	i_extra = 0;
	delete[](arr_extra);
	arr_extra = NULL;
      }
    }
  }
  if (arr_extra != NULL) {
    delete[](arr_extra);
  }
  if (vstep>stddev*epsi) {
    for (size_t i=0; i<n; i++) {
      arr[i] = vmin + vstep*round((arr[i] - vmin)/vstep);
    }
  }

  return arr; 
}

std::vector<std::string> NESTGPU::GetIntVarNames(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading variable names");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetIntVarNames();
}

std::vector<std::string> NESTGPU::GetScalVarNames(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading variable names");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetScalVarNames();
}

int NESTGPU::GetNIntVar(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading number of "
			 "variables");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetNIntVar();
}

int NESTGPU::GetNScalVar(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading number of "
			 "variables");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetNScalVar();
}

std::vector<std::string> NESTGPU::GetPortVarNames(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading variable names");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetPortVarNames();
}

int NESTGPU::GetNPortVar(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading number of "
			 "variables");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetNPortVar();
}


std::vector<std::string> NESTGPU::GetScalParamNames(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading parameter names");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetScalParamNames();
}

int NESTGPU::GetNScalParam(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading number of "
			 "parameters");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetNScalParam();
}

std::vector<std::string> NESTGPU::GetPortParamNames(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading parameter names");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetPortParamNames();
}

int NESTGPU::GetNPortParam(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading number of "
			 "parameters");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetNPortParam();
}


std::vector<std::string> NESTGPU::GetArrayParamNames(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading array parameter names");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetArrayParamNames();
}

int NESTGPU::GetNArrayParam(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading number of array "
			 "parameters");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetNArrayParam();
}


std::vector<std::string> NESTGPU::GetArrayVarNames(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading array variable names");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetArrayVarNames();
}

int NESTGPU::GetNArrayVar(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading number of array "
			 "variables");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetNArrayVar();
}

int64_t *NESTGPU::GetConnections(int i_source, int n_source,
				 int i_target, int n_target,
				 int syn_group, int64_t *n_conn)
{
  if (n_source<=0) {
    i_source = 0;
    n_source = GetNNode();
  }
  if (n_target<=0) {
    i_target = 0;
    n_target = GetNNode();
  }
  int *i_source_pt = new int[n_source];
  for (int i=0; i<n_source; i++) {
    i_source_pt[i] = i_source + i;
  }
  int *i_target_pt = new int[n_target];
  for (int i=0; i<n_target; i++) {
    i_target_pt[i] = i_target + i;
  }
  
  int64_t *conn_ids =
    GetConnections(i_source_pt, n_source, i_target_pt, n_target, syn_group,
		   n_conn);
  delete[] i_source_pt;
  delete[] i_target_pt;

  return conn_ids;
}

int64_t *NESTGPU::GetConnections(int *i_source_pt, int n_source,
				 int i_target, int n_target,
				 int syn_group, int64_t *n_conn)
{
  if (n_target<=0) {
    i_target = 0;
    n_target = GetNNode();
  }
  int *i_target_pt = new int[n_target];
  for (int i=0; i<n_target; i++) {
    i_target_pt[i] = i_target + i;
  }
  
  int64_t *conn_ids =
    GetConnections(i_source_pt, n_source, i_target_pt, n_target, syn_group,
		   n_conn);
  delete[] i_target_pt;

  return conn_ids;
}


int64_t *NESTGPU::GetConnections(int i_source, int n_source,
				 int *i_target_pt, int n_target,
				 int syn_group, int64_t *n_conn)
{
  if (n_source<=0) {
    i_source = 0;
    n_source = GetNNode();
  }
  int *i_source_pt = new int[n_source];
  for (int i=0; i<n_source; i++) {
    i_source_pt[i] = i_source + i;
  }

  int64_t *conn_ids =
    GetConnections(i_source_pt, n_source, i_target_pt, n_target, syn_group,
		   n_conn);
  delete[] i_source_pt;

  return conn_ids;
}

int64_t *NESTGPU::GetConnections(NodeSeq source, NodeSeq target,
				 int syn_group, int64_t *n_conn)
{
  return GetConnections(source.i0, source.n, target.i0, target.n, syn_group,
			n_conn);
}

int64_t *NESTGPU::GetConnections(std::vector<int> source, NodeSeq target,
				 int syn_group, int64_t *n_conn)
{
  return GetConnections(source.data(), source.size(), target.i0, target.n,
			syn_group, n_conn);
}


int64_t *NESTGPU::GetConnections(NodeSeq source, std::vector<int> target,
				 int syn_group, int64_t *n_conn)
{
  return GetConnections(source.i0, source.n, target.data(), target.size(),
			syn_group, n_conn);
}

int64_t *NESTGPU::GetConnections(std::vector<int> source,
				 std::vector<int> target,
				 int syn_group, int64_t *n_conn)
{
  return GetConnections(source.data(), source.size(),
			target.data(), target.size(),
			syn_group, n_conn);
}


int NESTGPU::ActivateSpikeCount(int i_node, int n_node)
{
  CheckUncalibrated("Spike count must be activated before calibration");
  int i_group;
  int i_node_0 = GetNodeSequenceOffset(i_node, n_node, i_group);
  if (i_node_0!=i_node || node_vect_[i_group]->n_node_!=n_node) {
    throw ngpu_exception("Spike count must be activated for all and only "
			 " the nodes of the same group");
  }
  node_vect_[i_group]->ActivateSpikeCount();

  return 0;
}

int NESTGPU::ActivateRecSpikeTimes(int i_node, int n_node,
				     int max_n_rec_spike_times)
{
  CheckUncalibrated("Spike time recording must be activated "
		    "before calibration");
  int i_group;
  int i_node_0 = GetNodeSequenceOffset(i_node, n_node, i_group);
  if (i_node_0!=i_node || node_vect_[i_group]->n_node_!=n_node) {
    throw ngpu_exception("Spike count must be activated for all and only "
			 " the nodes of the same group");
  }
  node_vect_[i_group]->ActivateRecSpikeTimes(max_n_rec_spike_times);

  return 0;
}

int NESTGPU::SetRecSpikeTimesStep(int i_node, int n_node,
				     int rec_spike_times_step)
{
  int i_group;
  int i_node_0 = GetNodeSequenceOffset(i_node, n_node, i_group);
  if (i_node_0!=i_node || node_vect_[i_group]->n_node_!=n_node) {
    throw ngpu_exception("Time step for buffering spike time recording "
			 "must be set for all and only "
			 "the nodes of the same group");
  }
  node_vect_[i_group]->SetRecSpikeTimesStep(rec_spike_times_step);

  return 0;
}

// get number of recorded spike times for a node
int NESTGPU::GetNRecSpikeTimes(int i_node)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);
  return node_vect_[i_group]->GetNRecSpikeTimes(i_neuron);
}

// get recorded spike times for node group
int NESTGPU::GetRecSpikeTimes(int i_node, int n_node, int **n_spike_times_pt,
			      float ***spike_times_pt)
{
  int i_group;
  int i_node_0 = GetNodeSequenceOffset(i_node, n_node, i_group);
  if (i_node_0!=i_node || node_vect_[i_group]->n_node_!=n_node) {
    throw ngpu_exception("Spike times must be extracted for all and only "
			 " the nodes of the same group");
  }
  
  return node_vect_[i_group]->GetRecSpikeTimes(n_spike_times_pt,
					       spike_times_pt);
					       
}

int NESTGPU::PushSpikesToNodes(int n_spikes, int *node_id,
				 float *spike_height)
{
  /*
  int *d_node_id;
  float *d_spike_height;
  gpuErrchk(cudaMalloc(&d_node_id, n_spikes*sizeof(int)));
  gpuErrchk(cudaMalloc(&d_spike_height, n_spikes*sizeof(float)));
  // Memcpy are synchronized by PushSpikeFromRemote kernel
  gpuErrchk(cudaMemcpyAsync(d_node_id, node_id, n_spikes*sizeof(int),
		       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyAsync(d_spike_height, spike_height, n_spikes*sizeof(float),
		       cudaMemcpyHostToDevice));
  PushSpikeFromRemote<<<(n_spikes+1023)/1024, 1024>>>(n_spikes, d_node_id,
						     d_spike_height);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  gpuErrchk(cudaFree(d_node_id));
  gpuErrchk(cudaFree(d_spike_height));
  */
  
  return 0;
}

int NESTGPU::PushSpikesToNodes(int n_spikes, int *node_id)
{
  /*
  //std::cout << "n_spikes: " << n_spikes << "\n";
  //for (int i=0; i<n_spikes; i++) {
  //  std::cout << node_id[i] << " ";
  //}
  //std::cout << "\n";

  int *d_node_id;
  gpuErrchk(cudaMalloc(&d_node_id, n_spikes*sizeof(int)));
  // memcopy data transfer is overlapped with PushSpikeFromRemote kernel
  gpuErrchk(cudaMemcpyAsync(d_node_id, node_id, n_spikes*sizeof(int),
		       cudaMemcpyHostToDevice));  
  PushSpikeFromRemote<<<(n_spikes+1023)/1024, 1024>>>(n_spikes, d_node_id);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  gpuErrchk(cudaFree(d_node_id));
  */
  
  return 0;
}

int NESTGPU::GetExtNeuronInputSpikes(int *n_spikes, int **node, int **port,
				       float **spike_height, bool include_zeros)
{
  ext_neuron_input_spike_node_.clear();
  ext_neuron_input_spike_port_.clear();
  ext_neuron_input_spike_height_.clear();
  
  for (unsigned int i=0; i<node_vect_.size(); i++) {
    if (node_vect_[i]->IsExtNeuron()) {
      int n_node;
      int n_port;
      float *sh = node_vect_[i]->GetExtNeuronInputSpikes(&n_node, &n_port);
      for (int i_neur=0; i_neur<n_node; i_neur++) {
	int i_node = i_neur + node_vect_[i]->i_node_0_;
	for (int i_port=0; i_port<n_port; i_port++) {
	  int j = i_neur*n_port + i_port;
	  if (sh[j] != 0.0 || include_zeros) {
	    ext_neuron_input_spike_node_.push_back(i_node);
	    ext_neuron_input_spike_port_.push_back(i_port);
	    ext_neuron_input_spike_height_.push_back(sh[j]);
	  }
	}
      }	
    }
  }
  *n_spikes = ext_neuron_input_spike_node_.size();
  *node = ext_neuron_input_spike_node_.data();
  *port = ext_neuron_input_spike_port_.data();
  *spike_height = ext_neuron_input_spike_height_.data();
  
  return 0;
}

int NESTGPU::SetNeuronGroupParam(int i_node, int n_node,
				   std::string param_name, float val)
{
  int i_group;
  int i_node_0 = GetNodeSequenceOffset(i_node, n_node, i_group);
  if (i_node_0!=i_node || node_vect_[i_group]->n_node_!=n_node) {
    throw ngpu_exception(std::string("Group parameter ") + param_name
			 + " can only be set for all and only "
			 " the nodes of the same group");
  }
  return node_vect_[i_group]->SetGroupParam(param_name, val);
}

int NESTGPU::IsNeuronGroupParam(int i_node, std::string param_name)
{
  int i_group;
  int i_node_0 = GetNodeSequenceOffset(i_node, 1, i_group);

  return node_vect_[i_group]->IsGroupParam(param_name);
}

float NESTGPU::GetNeuronGroupParam(int i_node, std::string param_name)
{
  int i_group;
  int i_node_0 = GetNodeSequenceOffset(i_node, 1, i_group);

  return node_vect_[i_group]->GetGroupParam(param_name);
}

std::vector<std::string> NESTGPU::GetGroupParamNames(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading group parameter names");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetGroupParamNames();
}

int NESTGPU::GetNGroupParam(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading number of "
			 "group parameters");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetNGroupParam();
}

// Connect spike buffers of remote source nodes to local target nodes
// Maybe move this in connect_rules.cpp ? And parallelize with OpenMP?
int NESTGPU::ConnectRemoteNodes()
{
  if (n_remote_node_>0) {
    i_remote_node_0_ = node_group_map_.size();
    BaseNeuron *bn = new BaseNeuron;
    node_vect_.push_back(bn);  
    CreateNodeGroup(n_remote_node_, 0);
    /*
    for (unsigned int i=0; i<remote_connection_vect_.size(); i++) {
      RemoteConnection rc = remote_connection_vect_[i];
      net_connection_->Connect(i_remote_node_0_ + rc.i_source_rel, rc.i_target,
			       rc.port, rc.syn_group, rc.weight, rc.delay);

    }
    */
  }
  
  return 0;
}


int NESTGPU::GetNBoolParam()
{
  return N_KERNEL_BOOL_PARAM;
}

std::vector<std::string> NESTGPU::GetBoolParamNames()
{
  std::vector<std::string> param_name_vect;
  for (int i=0; i<N_KERNEL_BOOL_PARAM; i++) {
    param_name_vect.push_back(kernel_bool_param_name[i]);
  }
  
  return param_name_vect;
}

bool NESTGPU::IsBoolParam(std::string param_name)
{
  int i_param;
  for (i_param=0; i_param<N_KERNEL_BOOL_PARAM; i_param++) {
    if (param_name == kernel_bool_param_name[i_param]) return true;
  }
  return false;
}

int NESTGPU::GetBoolParamIdx(std::string param_name)
{
  int i_param;
  for (i_param=0; i_param<N_KERNEL_BOOL_PARAM; i_param++) {
    if (param_name == kernel_bool_param_name[i_param]) break;
  }
  if (i_param == N_KERNEL_BOOL_PARAM) {
    throw ngpu_exception(std::string("Unrecognized kernel boolean parameter ")
			 + param_name);
  }
  
  return i_param;
}

bool NESTGPU::GetBoolParam(std::string param_name)
{
  int i_param =  GetBoolParamIdx(param_name);
  switch (i_param) {
  case i_print_time:
    return print_time_;
  default:
    throw ngpu_exception(std::string("Unrecognized kernel boolean parameter ")
			 + param_name);
  }
}

int NESTGPU::SetBoolParam(std::string param_name, bool val)
{
  int i_param =  GetBoolParamIdx(param_name);

  switch (i_param) {
  case i_time_resolution:
    print_time_ = val;
    break;
  default:
    throw ngpu_exception(std::string("Unrecognized kernel boolean parameter ")
			 + param_name);
  }
  
  return 0;
}


int NESTGPU::GetNFloatParam()
{
  return N_KERNEL_FLOAT_PARAM;
}

std::vector<std::string> NESTGPU::GetFloatParamNames()
{
  std::vector<std::string> param_name_vect;
  for (int i=0; i<N_KERNEL_FLOAT_PARAM; i++) {
    param_name_vect.push_back(kernel_float_param_name[i]);
  }
  
  return param_name_vect;
}

bool NESTGPU::IsFloatParam(std::string param_name)
{
  int i_param;
  for (i_param=0; i_param<N_KERNEL_FLOAT_PARAM; i_param++) {
    if (param_name == kernel_float_param_name[i_param]) return true;
  }
  return false;
}

int NESTGPU::GetFloatParamIdx(std::string param_name)
{
  int i_param;
  for (i_param=0; i_param<N_KERNEL_FLOAT_PARAM; i_param++) {
    if (param_name == kernel_float_param_name[i_param]) break;
  }
  if (i_param == N_KERNEL_FLOAT_PARAM) {
    throw ngpu_exception(std::string("Unrecognized kernel float parameter ")
			 + param_name);
  }
  
  return i_param;
}

float NESTGPU::GetFloatParam(std::string param_name)
{
  int i_param =  GetFloatParamIdx(param_name);
  switch (i_param) {
  case i_time_resolution:
    return time_resolution_;
  case i_max_spike_num_fact:
    return max_spike_num_fact_;
  case i_max_spike_per_host_fact:
    return max_spike_per_host_fact_;
  default:
    throw ngpu_exception(std::string("Unrecognized kernel float parameter ")
			 + param_name);
  }
}

int NESTGPU::SetFloatParam(std::string param_name, float val)
{
  int i_param =  GetFloatParamIdx(param_name);

  switch (i_param) {
  case i_time_resolution:
    time_resolution_ = val;
    break;
  case i_max_spike_num_fact:
    max_spike_num_fact_ = val;
    break;
  case i_max_spike_per_host_fact:
    max_spike_per_host_fact_ = val;
    break;
  default:
    throw ngpu_exception(std::string("Unrecognized kernel float parameter ")
			 + param_name);
  }
  
  return 0;
}

int NESTGPU::GetNIntParam()
{
  return N_KERNEL_INT_PARAM;
}

std::vector<std::string> NESTGPU::GetIntParamNames()
{
  std::vector<std::string> param_name_vect;
  for (int i=0; i<N_KERNEL_INT_PARAM; i++) {
    param_name_vect.push_back(kernel_int_param_name[i]);
  }
  
  return param_name_vect;
}

bool NESTGPU::IsIntParam(std::string param_name)
{
  int i_param;
  for (i_param=0; i_param<N_KERNEL_INT_PARAM; i_param++) {
    if (param_name == kernel_int_param_name[i_param]) return true;
  }
  return false;
}

int NESTGPU::GetIntParamIdx(std::string param_name)
{
  int i_param;
  for (i_param=0; i_param<N_KERNEL_INT_PARAM; i_param++) {
    if (param_name == kernel_int_param_name[i_param]) break;
  }
  if (i_param == N_KERNEL_INT_PARAM) {
    throw ngpu_exception(std::string("Unrecognized kernel int parameter ")
			 + param_name);
  }
  
  return i_param;
}

int NESTGPU::GetIntParam(std::string param_name)
{
  int i_param =  GetIntParamIdx(param_name);
  switch (i_param) {
  case i_rnd_seed:
    return kernel_seed_ - 12345; // see nestgpu.cu
  case i_verbosity_level:
    return verbosity_level_;
  case i_max_spike_buffer_size:
    return max_spike_buffer_size_;
  case i_remote_spike_height_flag:
#ifdef HAVE_MPI
    /*
    if (connect_mpi_->remote_spike_height_) {
      return 1;
    }
    else {
    */
      return 0;
      //}
#else
    return 0;
#endif
  default:
    throw ngpu_exception(std::string("Unrecognized kernel int parameter ")
			 + param_name);
  }
}

int NESTGPU::SetIntParam(std::string param_name, int val)
{
  int i_param =  GetIntParamIdx(param_name);
  switch (i_param) {
  case i_rnd_seed:
    SetRandomSeed(val);
    break;
  case i_verbosity_level:
    SetVerbosityLevel(val);
    break;
  case i_max_spike_per_host_fact:
    SetMaxSpikeBufferSize(val);
    break;
  case i_remote_spike_height_flag:
#ifdef HAVE_MPI
    /*
    if (val==0) {
      connect_mpi_->remote_spike_height_ = false;
    }
    else if (val==1) {
      connect_mpi_->remote_spike_height_ = true;
    }
    else {
      throw ngpu_exception("Admissible values of remote_spike_height_flag are only 0 or 1");
    }
    */
    break;
#else
    throw ngpu_exception("remote_spike_height_flag cannot be changed in an installation without MPI support");
#endif
  default:
    throw ngpu_exception(std::string("Unrecognized kernel int parameter ")
			 + param_name);
  }
  
  return 0;
}

RemoteNodeSeq NESTGPU::RemoteCreate(int i_host, std::string model_name,
				      int n_node /*=1*/, int n_port /*=1*/)
{
  if (!create_flag_) {
    create_flag_ = true;
    start_real_time_ = getRealTime();
  }
#ifdef HAVE_MPI
  if (i_host<0 || i_host>=MpiNp()) {
    throw ngpu_exception("Invalid host index in RemoteCreate");
  }
  NodeSeq node_seq;
  if (i_host == MpiId()) {
    node_seq = Create(model_name, n_node, n_port);
  }
  MPI_Bcast(&node_seq, sizeof(NodeSeq), MPI_BYTE, i_host, MPI_COMM_WORLD);
  return RemoteNodeSeq(i_host, node_seq);
#else
  throw ngpu_exception("MPI is not available in your build");
#endif
}
