/*
 *  nestgpu.cu
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
#include <stdio.h>
#include <stdint.h>
#include <cmath>
#include <iostream>
#include <string>
#include <algorithm>
#include <curand.h>
#include "spike_buffer.h"
#include "cuda_error.h"
#include "send_spike.h"
#include "get_spike.h"
#include "connect_mpi.h"

#include "spike_generator.h"
#include "multimeter.h"
#include "poisson.h"
#include "getRealTime.h"
#include "random.h"
#include "nestgpu.h"
#include "dir_connect.h"
#include "rev_spike.h"
#include "spike_mpi.h"

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#define THREAD_MAXNUM 1
#define THREAD_IDX 0

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
  poiss_generator_ = new PoissonGenerator;
  multimeter_ = new Multimeter;
  net_connection_ = new NetConnection;
  
  
  calibrate_flag_ = false;

  start_real_time_ = getRealTime();
  max_spike_buffer_size_ = 20;
  t_min_ = 0.0;
  sim_time_ = 1000.0;        //Simulation time in ms
  n_poiss_node_ = 0;
  n_remote_node_ = 0;
  SetTimeResolution(0.1);  // time resolution in ms
  max_spike_num_fact_ = 1.0;
  max_spike_per_host_fact_ = 1.0;
  
  error_flag_ = false;
  error_message_ = "";
  error_code_ = 0;
  
  on_exception_ = ON_EXCEPTION_EXIT;

  verbosity_level_ = 4;
  print_time_ = false;
  
  mpi_flag_ = false;
#ifdef HAVE_MPI
  connect_mpi_ = new ConnectMpi;
  connect_mpi_->net_connection_ = net_connection_;
  connect_mpi_->remote_spike_height_ = false;
#endif
  
  SetRandomSeed(54321ULL);

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
  delete connect_mpi_;
#endif

  delete net_connection_;
  delete multimeter_;
  delete poiss_generator_;
  curandDestroyGenerator(*random_generator_);
  delete random_generator_;
}

int NESTGPU::SetRandomSeed(unsigned long long seed)
{
  kernel_seed_ = seed + 12345;
  #ifdef HAVE_MPI
  kernel_seed_ += connect_mpi_->mpi_id_;
  #endif
  CURAND_CALL(curandDestroyGenerator(*random_generator_));
  random_generator_ = new curandGenerator_t;
  CURAND_CALL(curandCreateGenerator(random_generator_,
				    CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(*random_generator_, seed));
  poiss_generator_->random_generator_ = random_generator_;

  return 0;
}

int NESTGPU::SetTimeResolution(float time_res)
{
  time_resolution_ = time_res;
  net_connection_->time_resolution_ = time_res;
  
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

int NESTGPU::CreateNodeGroup(int n_node, int n_port)
{
  int i_node_0 = node_group_map_.size();

#ifdef HAVE_MPI
  if ((int)connect_mpi_->extern_connection_.size() != i_node_0) {
    throw ngpu_exception("Error: connect_mpi_.extern_connection_ and "
			 "node_group_map_ must have the same size!");
  }
#endif

  if ((int)net_connection_->connection_.size() != i_node_0) {
    throw ngpu_exception("Error: net_connection_.connection_ and "
			 "node_group_map_ must have the same size!");
  }
  if ((net_connection_->connection_.size() + n_node) > MAX_N_NEURON) {
    throw ngpu_exception(std::string("Maximum number of neurons ")
			 + std::to_string(MAX_N_NEURON) + " exceeded");
  }
  if (n_port > MAX_N_PORT) {
    throw ngpu_exception(std::string("Maximum number of ports ")
			 + std::to_string(MAX_N_PORT) + " exceeded");
  }
  int i_group = node_vect_.size() - 1;
  node_group_map_.insert(node_group_map_.end(), n_node, i_group);
  
  std::vector<ConnGroup> conn;
  std::vector<std::vector<ConnGroup> >::iterator it
    = net_connection_->connection_.end();
  net_connection_->connection_.insert(it, n_node, conn);

#ifdef HAVE_MPI
  std::vector<ExternalConnectionNode > conn_node;
  std::vector<std::vector< ExternalConnectionNode> >::iterator it1
    = connect_mpi_->extern_connection_.end();
  connect_mpi_->extern_connection_.insert(it1, n_node, conn_node);
#endif
  
  node_vect_[i_group]->Init(i_node_0, n_node, n_port, i_group, &kernel_seed_);
  node_vect_[i_group]->get_spike_array_ = InitGetSpikeArray(n_node, n_port);
  
  return i_node_0;
}

NodeSeq NESTGPU::CreatePoissonGenerator(int n_node, float rate)
{
  CheckUncalibrated("Poisson generator cannot be created after calibration");
  if (n_poiss_node_ != 0) {
    throw ngpu_exception("Number of poisson generators cannot be modified.");
  }
  else if (n_node <= 0) {
    throw ngpu_exception("Number of nodes must be greater than zero.");
  }
  
  n_poiss_node_ = n_node;               
 
  BaseNeuron *bn = new BaseNeuron;
  node_vect_.push_back(bn);
  int i_node_0 = CreateNodeGroup( n_node, 0);
  
  float lambda = rate*time_resolution_ / 1000.0; // rate is in Hz, time in ms
  poiss_generator_->Create(random_generator_, i_node_0, n_node, lambda);
    
  return NodeSeq(i_node_0, n_node);
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
  ConnectRemoteNodes();
  calibrate_flag_ = true;
  BuildDirectConnections();

#ifdef HAVE_MPI
  gpuErrchk(cudaMemcpyToSymbolAsync(NESTGPUMpiFlag, &mpi_flag_, sizeof(bool)));
#endif

  if (verbosity_level_>=1) {
    std::cout << MpiRankStr() << "Calibrating ...\n";
  }
  
  neural_time_ = t_min_;
  	    
  NodeGroupArrayInit();
  
  max_spike_num_ = (int)round(max_spike_num_fact_
                 * net_connection_->connection_.size()
  		 * net_connection_->MaxDelayNum());
  max_spike_num_ = (max_spike_num_>1) ? max_spike_num_ : 1;

  max_spike_per_host_ = (int)round(max_spike_per_host_fact_
                 * net_connection_->connection_.size()
  		 * net_connection_->MaxDelayNum());
  max_spike_per_host_ = (max_spike_per_host_>1) ? max_spike_per_host_ : 1;
  
  SpikeInit(max_spike_num_);
  SpikeBufferInit(net_connection_, max_spike_buffer_size_);

#ifdef HAVE_MPI
  if (mpi_flag_) {
    // remove superfluous argument mpi_np
    connect_mpi_->ExternalSpikeInit(connect_mpi_->extern_connection_.size(),
				    connect_mpi_->mpi_np_,
				    max_spike_per_host_);
  }
#endif
  
  if (net_connection_->NRevConnections()>0) {
    RevSpikeInit(net_connection_); 
  }
  
  multimeter_->OpenFiles();
  
  for (unsigned int i=0; i<node_vect_.size(); i++) {
    node_vect_[i]->Calibrate(t_min_, time_resolution_);
  }
  
  SynGroupCalibrate();
  
  gpuErrchk(cudaMemcpyToSymbolAsync(NESTGPUTimeResolution, &time_resolution_,
			       sizeof(float)));
///////////////////////////////////

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
#ifdef HAVE_MPI
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
#endif
  
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
  SpikeBufferUpdate<<<(net_connection_->connection_.size()+1023)/1024,
    1024>>>();
  gpuErrchk( cudaPeekAtLastError() );
  SpikeBufferUpdate_time_ += (getRealTime() - time_mark);
  time_mark = getRealTime();
  if (n_poiss_node_>0) {
    poiss_generator_->Update(Nt_-it_);
    poisson_generator_time_ += (getRealTime() - time_mark);
  }
  time_mark = getRealTime();
  neural_time_ = neur_t0_ + (double)time_resolution_*(it_+1);
  gpuErrchk(cudaMemcpyToSymbolAsync(NESTGPUTime, &neural_time_, sizeof(double)));
  long long time_idx = (int)round(neur_t0_/time_resolution_) + it_ + 1;
  gpuErrchk(cudaMemcpyToSymbolAsync(NESTGPUTimeIdx, &time_idx, sizeof(long long)));

  if (ConnectionSpikeTimeFlag) {
    if ( (time_idx & 0xffff) == 0x8000) {
      ResetConnectionSpikeTimeUp(net_connection_);
    }
    else if ( (time_idx & 0xffff) == 0) {
      ResetConnectionSpikeTimeDown(net_connection_);
    }
  }
    
  for (unsigned int i=0; i<node_vect_.size(); i++) {
    node_vect_[i]->Update(it_, neural_time_);
  }
  gpuErrchk( cudaPeekAtLastError() );
  
  neuron_Update_time_ += (getRealTime() - time_mark);
  multimeter_->WriteRecords(neural_time_);

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
    
  int n_spikes;
  time_mark = getRealTime();
  // Call will get delayed until ClearGetSpikesArrays()
  // afterwards the value of n_spikes will be available
  gpuErrchk(cudaMemcpyAsync(&n_spikes, d_SpikeNum, sizeof(int),
		       cudaMemcpyDeviceToHost));

  ClearGetSpikeArrays(); 
  gpuErrchk( cudaDeviceSynchronize() );   
  if (n_spikes > 0) {
    time_mark = getRealTime();
    CollectSpikeKernel<<<n_spikes, 1024>>>(n_spikes, d_SpikeTargetNum);
    gpuErrchk(cudaPeekAtLastError());

    NestedLoop_time_ += (getRealTime() - time_mark);
  }
  time_mark = getRealTime();
  for (unsigned int i=0; i<node_vect_.size(); i++) {
    if (node_vect_[i]->has_dir_conn_) {
      node_vect_[i]->SendDirectSpikes(neural_time_, time_resolution_/1000.0);
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
    }
  }
  gpuErrchk( cudaPeekAtLastError() );

  GetSpike_time_ += (getRealTime() - time_mark);

  time_mark = getRealTime();
  SpikeReset<<<1, 1>>>();
  gpuErrchk( cudaPeekAtLastError() );
  SpikeReset_time_ += (getRealTime() - time_mark);

#ifdef HAVE_MPI
  if (mpi_flag_) {
    time_mark = getRealTime();
    ExternalSpikeReset<<<1, 1>>>();
    gpuErrchk( cudaPeekAtLastError() );
    ExternalSpikeReset_time_ += (getRealTime() - time_mark);
  }
#endif

  if (net_connection_->NRevConnections()>0) {
    //time_mark = getRealTime();
    RevSpikeReset<<<1, 1>>>();
    gpuErrchk( cudaPeekAtLastError() );
    RevSpikeBufferUpdate<<<(net_connection_->connection_.size()+1023)/1024,
      1024>>>(net_connection_->connection_.size());
    gpuErrchk( cudaPeekAtLastError() );
    unsigned int n_rev_spikes;
    gpuErrchk(cudaMemcpy(&n_rev_spikes, d_RevSpikeNum, sizeof(unsigned int),
			 cudaMemcpyDeviceToHost));
    if (n_rev_spikes > 0) {
      SynapseUpdateKernel<<<n_rev_spikes, 1024>>>(n_rev_spikes, d_RevSpikeNConn);
      gpuErrchk(cudaPeekAtLastError());
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
  int err = connect_mpi_->MpiInit(argc, argv);
  if (err==0) {
    mpi_flag_ = true;
  }
  
  return err;
#else
  throw ngpu_exception("MPI is not available in your build");
#endif
}

int NESTGPU::MpiId()
{
#ifdef HAVE_MPI
  return connect_mpi_->mpi_id_;
#else
  throw ngpu_exception("MPI is not available in your build");
#endif
}

int NESTGPU::MpiNp()
{
#ifdef HAVE_MPI
  return connect_mpi_->mpi_np_;
#else
  throw ngpu_exception("MPI is not available in your build");
#endif

}

int NESTGPU::ProcMaster()
{
#ifdef HAVE_MPI
  return connect_mpi_->ProcMaster();
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
#ifdef HAVE_MPI
  if (mpi_flag_) {
    return std::string("MPI rank ") + std::to_string(connect_mpi_->mpi_id_)
      + " : ";
  }
  else {
    return "";
  }
#else
  return "";
#endif
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

int NESTGPU::BuildDirectConnections()
{
  for (unsigned int iv=0; iv<node_vect_.size(); iv++) {
    if (node_vect_[iv]->has_dir_conn_) {
      std::vector<DirectConnection> dir_conn_vect;
      int i0 = node_vect_[iv]->i_node_0_;
      int n = node_vect_[iv]->n_node_;
      for (int i_source=i0; i_source<i0+n; i_source++) {
	std::vector<ConnGroup> &conn = net_connection_->connection_[i_source];
	for (unsigned int id=0; id<conn.size(); id++) {
	  std::vector<TargetSyn> tv = conn[id].target_vect;
	  for (unsigned int i=0; i<tv.size(); i++) {
	    DirectConnection dir_conn;
	    dir_conn.irel_source_ = i_source - i0;
	    dir_conn.i_target_ = tv[i].node;
	    dir_conn.port_ = tv[i].port;
	    dir_conn.weight_ = tv[i].weight;
	    dir_conn.delay_ = time_resolution_*(conn[id].delay+1);
	    dir_conn_vect.push_back(dir_conn);
	  }
	}
      }
      uint64_t n_dir_conn = dir_conn_vect.size();
      node_vect_[iv]->n_dir_conn_ = n_dir_conn;
      
      DirectConnection *d_dir_conn_array;
      gpuErrchk(cudaMalloc(&d_dir_conn_array,
			   n_dir_conn*sizeof(DirectConnection )));
      gpuErrchk(cudaMemcpy(d_dir_conn_array, dir_conn_vect.data(),
			   n_dir_conn*sizeof(DirectConnection),
			   cudaMemcpyHostToDevice));
      node_vect_[iv]->d_dir_conn_array_ = d_dir_conn_array;
    }
  }

  return 0;
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

ConnectionStatus NESTGPU::GetConnectionStatus(ConnectionId conn_id) {
  ConnectionStatus conn_stat = net_connection_->GetConnectionStatus(conn_id);
  if (calibrate_flag_ == true) {
    int i_source = conn_id.i_source_;
    int i_group = conn_id.i_group_;
    int i_conn = conn_id.i_conn_;
    int n_spike_buffer = net_connection_->connection_.size();
    conn_stat.weight = 0;
    float *d_weight_pt
      = h_ConnectionGroupTargetWeight[i_group*n_spike_buffer+i_source] + i_conn;
    gpuErrchk(cudaMemcpy(&conn_stat.weight, d_weight_pt, sizeof(float),
			 cudaMemcpyDeviceToHost));
  }
  return conn_stat;
}

std::vector<ConnectionStatus> NESTGPU::GetConnectionStatus(std::vector
							     <ConnectionId>
							     &conn_id_vect) {
  std::vector<ConnectionStatus> conn_stat_vect;
  for (unsigned int i=0; i<conn_id_vect.size(); i++) {
    ConnectionStatus conn_stat = GetConnectionStatus(conn_id_vect[i]);
    conn_stat_vect.push_back(conn_stat);
  }
  return conn_stat_vect;
}
  
std::vector<ConnectionId> NESTGPU::GetConnections(int i_source, int n_source,
						    int i_target, int n_target,
						    int syn_group) {
  if (n_source<=0) {
    i_source = 0;
    n_source = net_connection_->connection_.size();
  }
  if (n_target<=0) {
    i_target = 0;
    n_target = net_connection_->connection_.size();
  }

  return net_connection_->GetConnections<int>(i_source, n_source, i_target,
					      n_target, syn_group);    
}

std::vector<ConnectionId> NESTGPU::GetConnections(int *i_source, int n_source,
						    int i_target, int n_target,
						    int syn_group) {
  if (n_target<=0) {
    i_target = 0;
    n_target = net_connection_->connection_.size();
  }
    
  return net_connection_->GetConnections<int*>(i_source, n_source, i_target,
					       n_target, syn_group);
  
}


std::vector<ConnectionId> NESTGPU::GetConnections(int i_source, int n_source,
						    int *i_target, int n_target,
						    int syn_group) {
  if (n_source<=0) {
    i_source = 0;
    n_source = net_connection_->connection_.size();
  }
  
  return net_connection_->GetConnections<int>(i_source, n_source, i_target,
					      n_target, syn_group);    
}

std::vector<ConnectionId> NESTGPU::GetConnections(int *i_source, int n_source,
						    int *i_target, int n_target,
						    int syn_group) {
  
  return net_connection_->GetConnections<int*>(i_source, n_source, i_target,
					       n_target, syn_group);
  
}


std::vector<ConnectionId> NESTGPU::GetConnections(NodeSeq source,
						    NodeSeq target,
						    int syn_group) {
  return net_connection_->GetConnections<int>(source.i0, source.n, target.i0,
					      target.n, syn_group);
}

std::vector<ConnectionId> NESTGPU::GetConnections(std::vector<int> source,
						    NodeSeq target,
						    int syn_group) {
  return net_connection_->GetConnections<int*>(source.data(), source.size(),
					       target.i0, target.n,
					       syn_group);
}


std::vector<ConnectionId> NESTGPU::GetConnections(NodeSeq source,
						    std::vector<int> target,
						    int syn_group) {
  return net_connection_->GetConnections<int>(source.i0, source.n,
					      target.data(), target.size(),
					      syn_group);
}

std::vector<ConnectionId> NESTGPU::GetConnections(std::vector<int> source,
						    std::vector<int> target,
						    int syn_group) {
  return net_connection_->GetConnections<int*>(source.data(), source.size(),
					       target.data(), target.size(),
					       syn_group);
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

  return 0;
}

int NESTGPU::PushSpikesToNodes(int n_spikes, int *node_id)
{
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
    for (unsigned int i=0; i<remote_connection_vect_.size(); i++) {
      RemoteConnection rc = remote_connection_vect_[i];
      net_connection_->Connect(i_remote_node_0_ + rc.i_source_rel, rc.i_target,
			       rc.port, rc.syn_group, rc.weight, rc.delay);

    }
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
    if (connect_mpi_->remote_spike_height_) {
      return 1;
    }
    else {
      return 0;
    }
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
    if (val==0) {
      connect_mpi_->remote_spike_height_ = false;
    }
    else if (val==1) {
      connect_mpi_->remote_spike_height_ = true;
    }
    else {
      throw ngpu_exception("Admissible values of remote_spike_height_flag are only 0 or 1");
    }
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
