/*
 *  base_neuron.cu
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

#include <vector>
#include <algorithm>

#include <config.h>

#include <iostream>
#include "utilities.h"
#include "ngpu_exception.h"
#include "cuda_error.h"
#include "distribution.h"
#include "base_neuron.h"
#include "spike_buffer.h"
#include "scan.h"

// set equally spaced (index i*step) elements of array arr to value val
__global__ void BaseNeuronSetIntArray(int *arr, int n_elem, int step,
					int val)
{
  int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (array_idx<n_elem) {
    arr[array_idx*step] = val;
  }
}

// set elements of array arr to value val using indexes from pointer pos
// and given step: index = pos[array_idx]*step
__global__ void BaseNeuronSetIntPtArray(int *arr, int *pos, int n_elem,
					  int step, int val)
{
  int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (array_idx<n_elem) {
    arr[pos[array_idx]*step] = val;
  }
}

// copy equally spaced elements of array arr1 to equally spaced positions
// of array arr2
__global__ void BaseNeuronGetIntArray(int *arr1, int *arr2, int n_elem,
					int step1, int step2)
{
  int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (array_idx<n_elem) {
    arr2[array_idx*step2] = arr1[array_idx*step1];
  }
}

// copy elements of array arr1 with indexes from pointer pos
// and given step (index = pos[array_idx]*step1)
// to equally spaced positions of array arr2
__global__ void BaseNeuronGetIntPtArray(int *arr1, int *arr2, int *pos,
					  int n_elem, int step1, int step2)
{
  int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (array_idx<n_elem) {
    arr2[array_idx*step2] = arr1[pos[array_idx]*step1];
  }
}

// set equally spaced (index i*step) elements of array arr to value val
__global__ void BaseNeuronSetFloatArray(float *arr, int n_elem, int step,
					float val)
{
  int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (array_idx<n_elem) {
    arr[array_idx*step] = val;
  }
}

// copy array src_arr to equally spaced (index i*step) elements of target_arr
__global__ void BaseNeuronCopyFloatArray(float *target_arr, int n_elem,
					 int step, float *src_arr)
{
  int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (array_idx<n_elem) {
    target_arr[array_idx*step] = src_arr[array_idx];
  }
}

// set elements of array arr to value val using indexes from pointer pos
// and given step: index = pos[array_idx]*step
__global__ void BaseNeuronSetFloatPtArray(float *arr, int *pos, int n_elem,
					  int step, float val)
{
  int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (array_idx<n_elem) {
    arr[pos[array_idx]*step] = val;
  }
}

// copy array src_arr to elements of array target_arr using indexes
// from pointer pos and given step: index = pos[array_idx]*step
__global__ void BaseNeuronCopyFloatPtArray(float *target_arr, int *pos,
					   int n_elem, int step, float *src_arr)
{
  int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (array_idx<n_elem) {
    target_arr[pos[array_idx]*step] = src_arr[array_idx];
  }
}

// copy equally spaced elements of array arr1 to equally spaced positions
// of array arr2
__global__ void BaseNeuronGetFloatArray(float *arr1, float *arr2, int n_elem,
					int step1, int step2)
{
  int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (array_idx<n_elem) {
    arr2[array_idx*step2] = arr1[array_idx*step1];
  }
}

// copy elements of array arr1 with indexes from pointer pos
// and given step (index = pos[array_idx]*step1)
// to equally spaced positions of array arr2
__global__ void BaseNeuronGetFloatPtArray(float *arr1, float *arr2, int *pos,
					  int n_elem, int step1, int step2)
{
  int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (array_idx<n_elem) {
    arr2[array_idx*step2] = arr1[pos[array_idx]*step1];
  }
}

// Initialization method for class BaseNeuron
int BaseNeuron::Init(int i_node_0, int n_node, int n_port,
		     int i_group)
{
  node_type_= 0; // NULL MODEL
  ext_neuron_flag_ = false; // by default neuron is not external
  i_node_0_ = i_node_0; // first neuron of group index in spike buffer array
  n_node_ = n_node; // number of nodes in the group
  n_port_ = n_port; // number of receptor ports
  i_group_ = i_group; // neuron group index

  n_scal_var_ = 0; // number of scalar state variables
  n_port_var_ = 0; // number of receptor-port state variables
  n_scal_param_ = 0; // number of scalar parameters
  n_port_param_ = 0; // number of receptor-port parameters
  n_group_param_ = 0; // number of neuron-group parameters
  n_var_ = 0; // total number of state variables
  n_param_ = 0; // total number of parameters

  get_spike_array_ = NULL;
  port_weight_arr_ = NULL; // pointer to array of receptor-port weights
  port_weight_arr_step_ = 0; // step between elements for different neurons
  port_weight_port_step_ = 0; // step between elements for different ports
  port_input_arr_ = NULL; // pointer to array of receptor-port input
  port_input_arr_step_ = 0; // step between elements for different neurons
  port_input_port_step_ = 0; // step between elements for different ports
  var_arr_ = NULL; // pointer to state-variables array
  param_arr_ = NULL; // pointer to parameter array
  group_param_ = NULL; // pointer to neuron-group parameters
  int_var_name_.clear(); // vector of integer-variable names
  scal_var_name_ = NULL; // array of scalar state-variable names
  port_var_name_= NULL;// array of receptor-port state variable names
  scal_param_name_ = NULL; // array of scalar parameter names
  port_param_name_ = NULL; // array of receptor-port parameter names
  group_param_name_ = NULL; // array of neuron-group parameter names
  array_var_name_.clear(); // vector of array-variable names
  array_param_name_.clear(); // vector of array-parameter names

  has_dir_conn_ = false; // true if neur. group has outgoing direct connections

  spike_count_ = NULL; // array of spike counters
  rec_spike_times_ = NULL; // array of spike-time records
  n_rec_spike_times_ = NULL; // array of number of recorded spike times
  max_n_rec_spike_times_ = 0; // max number of recorded spike times
  rec_spike_times_step_ = 0; // number of time steps for spike times buffering
                             // 0 for no buffering
  den_delay_arr_ = NULL; // array of dendritic backward delays

  return 0;
}			    

// allocate state-variable array
int BaseNeuron::AllocVarArr()
{
  CUDAMALLOCCTRL("&var_arr_",&var_arr_, n_node_*n_var_*sizeof(float));
  return 0;
}

// allocate parameter array
int BaseNeuron::AllocParamArr()
{
  CUDAMALLOCCTRL("&param_arr_",&param_arr_, n_node_*n_param_*sizeof(float));
  return 0;
}

// deallocate state-variable array
int BaseNeuron::FreeVarArr()
{
  if (var_arr_ != NULL) {
    CUDAFREECTRL("var_arr_",var_arr_);
    var_arr_ = NULL;
  }
  return 0;
}

// deallocate parameter array
int BaseNeuron::FreeParamArr()
{
  if (param_arr_ != NULL) {
    CUDAFREECTRL("param_arr_",param_arr_);
    param_arr_ = NULL;
  }
  return 0;
}

// set scalar parameter param_name of neurons
// i_neuron, ..., i_neuron + n_neuron -1
// to value val
int BaseNeuron::SetScalParam(int i_neuron, int n_neuron,
			     std::string param_name, float val)
{
  if (!IsScalParam(param_name)) {
    throw ngpu_exception(std::string("Unrecognized scalar parameter ")
			 + param_name);
  }
  CheckNeuronIdx(i_neuron);
  CheckNeuronIdx(i_neuron + n_neuron - 1);
  float *param_pt = GetParamPt(i_neuron, param_name);
  BaseNeuronSetFloatArray<<<(n_neuron+1023)/1024, 1024>>>
    (param_pt, n_neuron, n_param_, val);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  
  return 0;
}

// set scalar parameter param_name of neurons
// i_neuron[0], ..., i_neuron[n_neuron -1]
// to value val
int BaseNeuron::SetScalParam(int *i_neuron, int n_neuron,
			     std::string param_name, float val)
{
  if (!IsScalParam(param_name)) {
    throw ngpu_exception(std::string("Unrecognized scalar parameter ")
				     + param_name);
  }
  int *d_i_neuron;
  CUDAMALLOCCTRL("&d_i_neuron",&d_i_neuron, n_neuron*sizeof(int));
  // Memcopy will be synchronized with BaseNeuronSetFloatPtArray kernel
  gpuErrchk(cudaMemcpyAsync(d_i_neuron, i_neuron, n_neuron*sizeof(int),
		       cudaMemcpyHostToDevice));
  float *param_pt = GetParamPt(0, param_name);
  BaseNeuronSetFloatPtArray<<<(n_neuron+1023)/1024, 1024>>>
    (param_pt, d_i_neuron, n_neuron, n_param_, val);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  CUDAFREECTRL("d_i_neuron",d_i_neuron);
  
  return 0;
}

// set receptor-port parameter param_name of neurons
// i_neuron, ..., i_neuron + n_neuron -1
// to value val
int BaseNeuron::SetPortParam(int i_neuron, int n_neuron,
			     std::string param_name, float *param,
			     int vect_size)
{
  if (!IsPortParam(param_name)) {
    throw ngpu_exception(std::string("Unrecognized port parameter ")
			 + param_name);
  }
  CheckNeuronIdx(i_neuron);
  CheckNeuronIdx(i_neuron + n_neuron - 1);
  if (vect_size != n_port_) {
    throw ngpu_exception("Parameter array size must be equal "
			 "to the number of ports.");
  }
  float *param_pt;
    
  for (int i_vect=0; i_vect<vect_size; i_vect++) {
    param_pt = GetParamPt(i_neuron, param_name, i_vect);
    BaseNeuronSetFloatArray<<<(n_neuron+1023)/1024, 1024>>>
      (param_pt, n_neuron, n_param_, param[i_vect]);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }
  return 0;
}

// set receptor-port parameter param_name of neurons
// i_neuron[0], ..., i_neuron[n_neuron -1]
// to value val
int BaseNeuron::SetPortParam(int *i_neuron, int n_neuron,
			     std::string param_name, float *param,
			     int vect_size)
{
  if (!IsPortParam(param_name)) {
    throw ngpu_exception(std::string("Unrecognized port parameter ")
			 + param_name);
  }
  if (vect_size != n_port_) {
    throw ngpu_exception("Parameter array size must be equal "
			 "to the number of ports.");
  }
  int *d_i_neuron;
  CUDAMALLOCCTRL("&d_i_neuron",&d_i_neuron, n_neuron*sizeof(int));
  // Memcopy will be synchronized with BaseNeuronSetFloatPtArray kernel
  gpuErrchk(cudaMemcpyAsync(d_i_neuron, i_neuron, n_neuron*sizeof(int),
		       cudaMemcpyHostToDevice));
  for (int i_vect=0; i_vect<vect_size; i_vect++) {
    float *param_pt = GetParamPt(0, param_name, i_vect);
    BaseNeuronSetFloatPtArray<<<(n_neuron+1023)/1024, 1024>>>
      (param_pt, d_i_neuron, n_neuron, n_param_, param[i_vect]);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }
  CUDAFREECTRL("d_i_neuron",d_i_neuron);

  return 0;
}

// set array parameter param_name of neurons
// i_neuron, ..., i_neuron + n_neuron -1
// to values array[0], ... , array[array_size-1]
// Must be defined in derived classes
int BaseNeuron::SetArrayParam(int i_neuron, int n_neuron,
			      std::string param_name, float *array,
			      int array_size)
{
  throw ngpu_exception(std::string("Unrecognized parameter ")
		       + param_name);
}

// set array parameter param_name of neurons
// i_neuron[0], ..., i_neuron[n_neuron -1]
// to values array[0], ... , array[array_size-1]
// Must be defined in derived classes
int BaseNeuron::SetArrayParam(int *i_neuron, int n_neuron,
			      std::string param_name, float *array,
			      int array_size)
{
  throw ngpu_exception(std::string("Unrecognized parameter ")
		       + param_name);
}

// set neuron-group parameter param_name to value val
int BaseNeuron::SetGroupParam(std::string param_name, float val)
{
  int i_param;
  for (i_param=0; i_param<n_group_param_; i_param++) {
    if (param_name == group_param_name_[i_param]) {
      group_param_[i_param] = val;
      return 0;
    }
  }
  throw ngpu_exception(std::string("Unrecognized group parameter ")
		       + param_name);
}

// set integer variable var_name of neurons
// i_neuron, ..., i_neuron + n_neuron -1
// to value val
int BaseNeuron::SetIntVar(int i_neuron, int n_neuron,
			  std::string var_name, int val)
{
  if (!IsIntVar(var_name)) {
    throw ngpu_exception(std::string("Unrecognized integer variable ")
			 + var_name);
  }
  CheckNeuronIdx(i_neuron);
  CheckNeuronIdx(i_neuron + n_neuron - 1);
  int *var_pt = GetIntVarPt(i_neuron, var_name);
  BaseNeuronSetIntArray<<<(n_neuron+1023)/1024, 1024>>>
    (var_pt, n_neuron, 1, val);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  
  return 0;
}

// set integer variable var_name of neurons
// i_neuron[0], ..., i_neuron[n_neuron -1]
// to value val
int BaseNeuron::SetIntVar(int *i_neuron, int n_neuron,
			  std::string var_name, int val)
{
  if (!IsIntVar(var_name)) {
    throw ngpu_exception(std::string("Unrecognized integer variable ")
			 + var_name);
  }
  int *d_i_neuron;
  CUDAMALLOCCTRL("&d_i_neuron",&d_i_neuron, n_neuron*sizeof(int));
  // Memcopy will be synchronized with BaseNeuronSetIntPtArray kernel
  gpuErrchk(cudaMemcpyAsync(d_i_neuron, i_neuron, n_neuron*sizeof(int),
		       cudaMemcpyHostToDevice));
  int *var_pt = GetIntVarPt(0, var_name);
  BaseNeuronSetIntPtArray<<<(n_neuron+1023)/1024, 1024>>>
    (var_pt, d_i_neuron, n_neuron, 1, val);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  CUDAFREECTRL("d_i_neuron",d_i_neuron);
  
  return 0;
}

// set scalar state-variable var_name of neurons
// i_neuron, ..., i_neuron + n_neuron -1
// to value val
int BaseNeuron::SetScalVar(int i_neuron, int n_neuron,
			     std::string var_name, float val)
{
  if (!IsScalVar(var_name)) {
    throw ngpu_exception(std::string("Unrecognized scalar variable ")
			 + var_name);
  }
  CheckNeuronIdx(i_neuron);
  CheckNeuronIdx(i_neuron + n_neuron - 1);
  float *var_pt = GetVarPt(i_neuron, var_name);
  BaseNeuronSetFloatArray<<<(n_neuron+1023)/1024, 1024>>>
    (var_pt, n_neuron, n_var_, val);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  
  return 0;
}

// set scalar state-variable var_name of neurons
// i_neuron[0], ..., i_neuron[n_neuron -1]
// to value val
int BaseNeuron::SetScalVar(int *i_neuron, int n_neuron,
			   std::string var_name, float val)
{
  if (!IsScalVar(var_name)) {
    throw ngpu_exception(std::string("Unrecognized scalar variable ")
				     + var_name);
  }
  int *d_i_neuron;
  CUDAMALLOCCTRL("&d_i_neuron",&d_i_neuron, n_neuron*sizeof(int));
  // Memcopy will be synchronized with BaseNeuronSetFloatPtArray kernel
  gpuErrchk(cudaMemcpyAsync(d_i_neuron, i_neuron, n_neuron*sizeof(int),
		       cudaMemcpyHostToDevice));
  float *var_pt = GetVarPt(0, var_name);
  BaseNeuronSetFloatPtArray<<<(n_neuron+1023)/1024, 1024>>>
    (var_pt, d_i_neuron, n_neuron, n_var_, val);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  CUDAFREECTRL("d_i_neuron",d_i_neuron);
  
  return 0;
}

// set receptor-port state-variable var_name of neurons
// i_neuron, ..., i_neuron + n_neuron -1
// to value val
int BaseNeuron::SetPortVar(int i_neuron, int n_neuron,
			   std::string var_name, float *var,
			   int vect_size)
{
  if (!IsPortVar(var_name)) {
    throw ngpu_exception(std::string("Unrecognized port variable ")
			 + var_name);
  }
  CheckNeuronIdx(i_neuron);
  CheckNeuronIdx(i_neuron + n_neuron - 1);
  if (vect_size != n_port_) {
    throw ngpu_exception("Variable array size must be equal "
			 "to the number of ports.");
  }
  float *var_pt;
    
  for (int i_vect=0; i_vect<vect_size; i_vect++) {
    var_pt = GetVarPt(i_neuron, var_name, i_vect);
    BaseNeuronSetFloatArray<<<(n_neuron+1023)/1024, 1024>>>
      (var_pt, n_neuron, n_var_, var[i_vect]);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }
  return 0;
}

// set receptor-port state-variable var_name of neurons
// i_neuron[0], ..., i_neuron[n_neuron -1]
// to value val
int BaseNeuron::SetPortVar(int *i_neuron, int n_neuron,
			   std::string var_name, float *var,
			   int vect_size)
{
  if (!IsPortVar(var_name)) {
    throw ngpu_exception(std::string("Unrecognized port variable ")
			 + var_name);
  }
  if (vect_size != n_port_) {
    throw ngpu_exception("Variable array size must be equal "
			 "to the number of ports.");
  }
  int *d_i_neuron;
  CUDAMALLOCCTRL("&d_i_neuron",&d_i_neuron, n_neuron*sizeof(int));
  // Memcopy will be synchronized with BaseNeuronSetFloatPtArray kernel
  gpuErrchk(cudaMemcpyAsync(d_i_neuron, i_neuron, n_neuron*sizeof(int),
		       cudaMemcpyHostToDevice));
  for (int i_vect=0; i_vect<vect_size; i_vect++) {
    float *var_pt = GetVarPt(0, var_name, i_vect);
    BaseNeuronSetFloatPtArray<<<(n_neuron+1023)/1024, 1024>>>
      (var_pt, d_i_neuron, n_neuron, n_var_, var[i_vect]);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }
  CUDAFREECTRL("d_i_neuron",d_i_neuron);

  return 0;
}

// set array variable var_name of neurons
// i_neuron, ..., i_neuron + n_neuron -1
// to values array[0], ... , array[array_size-1]
// Must be defined in derived classes
int BaseNeuron::SetArrayVar(int i_neuron, int n_neuron,
			      std::string var_name, float *array,
			      int array_size)
{
  throw ngpu_exception(std::string("Unrecognized variable ")
		       + var_name);
}

// set array variable var_name of neurons
// i_neuron[0], ..., i_neuron[n_neuron -1]
// to values array[0], ... , array[array_size-1]
// Must be defined in derived classes
int BaseNeuron::SetArrayVar(int *i_neuron, int n_neuron,
			      std::string var_name, float *array,
			      int array_size)
{
  throw ngpu_exception(std::string("Unrecognized variable ")
		       + var_name);
}




//////////////////////////////////////////////////////////////////////

// set scalar parameter param_name of neurons
// i_neuron, ..., i_neuron + n_neuron -1
// using distribution or array
int BaseNeuron::SetScalParamDistr(int i_neuron, int n_neuron,
				  std::string param_name,
				  Distribution *distribution)
{
  if (!IsScalParam(param_name)) {
    throw ngpu_exception(std::string("Unrecognized scalar parameter ")
			 + param_name);
  }
  CheckNeuronIdx(i_neuron);
  CheckNeuronIdx(i_neuron + n_neuron - 1);
  float *param_pt = GetParamPt(i_neuron, param_name);
  float *d_arr = distribution->getArray(*random_generator_, n_neuron);
  BaseNeuronCopyFloatArray<<<(n_neuron+1023)/1024, 1024>>>
    (param_pt, n_neuron, n_param_, d_arr);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  CUDAFREECTRL("d_arr",d_arr);
  
  return 0;
}

// set scalar parameter param_name of neurons
// i_neuron[0], ..., i_neuron[n_neuron -1]
// using distribution or array
int BaseNeuron::SetScalParamDistr(int *i_neuron, int n_neuron,
				  std::string param_name,
				  Distribution *distribution)
{
  if (!IsScalParam(param_name)) {
    throw ngpu_exception(std::string("Unrecognized scalar parameter ")
				     + param_name);
  }
  int *d_i_neuron;
  CUDAMALLOCCTRL("&d_i_neuron",&d_i_neuron, n_neuron*sizeof(int));
  gpuErrchk(cudaMemcpy(d_i_neuron, i_neuron, n_neuron*sizeof(int),
		       cudaMemcpyHostToDevice));
  float *param_pt = GetParamPt(0, param_name);
  float *d_arr = distribution->getArray(*random_generator_, n_neuron);
  BaseNeuronCopyFloatPtArray<<<(n_neuron+1023)/1024, 1024>>>
    (param_pt, d_i_neuron, n_neuron, n_param_, d_arr);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  CUDAFREECTRL("d_i_neuron",d_i_neuron);
  CUDAFREECTRL("d_arr",d_arr);
  
  return 0;
}

// set scalar variable var_name of neurons
// i_neuron, ..., i_neuron + n_neuron -1
// using distribution or array
int BaseNeuron::SetScalVarDistr(int i_neuron, int n_neuron,
				  std::string var_name,
				  Distribution *distribution)
{
  //printf("okk0\n");
  if (!IsScalVar(var_name)) {
    throw ngpu_exception(std::string("Unrecognized scalar variable ")
			 + var_name);
  }
  CheckNeuronIdx(i_neuron);
  CheckNeuronIdx(i_neuron + n_neuron - 1);
  float *var_pt = GetVarPt(i_neuron, var_name);
  //printf("okk1\n");
  float *d_arr = distribution->getArray(*random_generator_, n_neuron);
  //printf("okk2\n");
  BaseNeuronCopyFloatArray<<<(n_neuron+1023)/1024, 1024>>>
    (var_pt, n_neuron, n_var_, d_arr);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  //printf("okk3\n");
  CUDAFREECTRL("d_arr",d_arr);
  //printf("okk4\n");
  
  return 0;
}

// set scalar state-variable var_name of neurons
// i_neuron[0], ..., i_neuron[n_neuron -1]
// using distribution or array
int BaseNeuron::SetScalVarDistr(int *i_neuron, int n_neuron,
				std::string var_name,
				Distribution *distribution)
{
  if (!IsScalVar(var_name)) {
    throw ngpu_exception(std::string("Unrecognized scalar variable ")
				     + var_name);
  }
  int *d_i_neuron;
  CUDAMALLOCCTRL("&d_i_neuron",&d_i_neuron, n_neuron*sizeof(int));
  gpuErrchk(cudaMemcpy(d_i_neuron, i_neuron, n_neuron*sizeof(int),
		       cudaMemcpyHostToDevice));
  float *var_pt = GetVarPt(0, var_name);
  float *d_arr = distribution->getArray(*random_generator_, n_neuron);
  BaseNeuronCopyFloatPtArray<<<(n_neuron+1023)/1024, 1024>>>
    (var_pt, d_i_neuron, n_neuron, n_var_, d_arr);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  CUDAFREECTRL("d_i_neuron",d_i_neuron);
  CUDAFREECTRL("d_arr",d_arr);
  
  return 0;
}

// set receptor-port parameter param_name of neurons
// i_neuron, ..., i_neuron + n_neuron -1
// using distribution or array
int BaseNeuron::SetPortParamDistr(int i_neuron, int n_neuron,
				  std::string param_name,
				  Distribution *distribution)
{
  if (!IsPortParam(param_name)) {
    throw ngpu_exception(std::string("Unrecognized port parameter ")
			 + param_name);
  }
  CheckNeuronIdx(i_neuron);
  CheckNeuronIdx(i_neuron + n_neuron - 1);
  int vect_size = distribution->vectSize();
  if (vect_size != n_port_) {
    throw ngpu_exception("Distribution vector dimension must be "
			 "equal to the number of ports.");
  }
  float *param_pt;
    
  for (int i_vect=0; i_vect<vect_size; i_vect++) {
    param_pt = GetParamPt(i_neuron, param_name, i_vect);
    float *d_arr = distribution->getArray(*random_generator_, n_neuron, i_vect);
    BaseNeuronCopyFloatArray<<<(n_neuron+1023)/1024, 1024>>>
      (param_pt, n_neuron, n_param_, d_arr);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    CUDAFREECTRL("d_arr",d_arr);
  }
  return 0;
}

// set receptor-port parameter param_name of neurons
// i_neuron[0], ..., i_neuron[n_neuron -1]
// using distribution or array
int BaseNeuron::SetPortParamDistr(int *i_neuron, int n_neuron,
				  std::string param_name,
				  Distribution *distribution)
			     
{
  if (!IsPortParam(param_name)) {
    throw ngpu_exception(std::string("Unrecognized port parameter ")
			 + param_name);
  }
  int vect_size = distribution->vectSize();
  if (vect_size != n_port_) {
    throw ngpu_exception("Distribution vector dimension must be "
			 "equal to the number of ports.");
  }
  int *d_i_neuron;
  CUDAMALLOCCTRL("&d_i_neuron",&d_i_neuron, n_neuron*sizeof(int));
  gpuErrchk(cudaMemcpy(d_i_neuron, i_neuron, n_neuron*sizeof(int),
		       cudaMemcpyHostToDevice));
  for (int i_vect=0; i_vect<vect_size; i_vect++) {
    float *param_pt = GetParamPt(0, param_name, i_vect);
    float *d_arr = distribution->getArray(*random_generator_, n_neuron, i_vect);
    BaseNeuronCopyFloatPtArray<<<(n_neuron+1023)/1024, 1024>>>
      (param_pt, d_i_neuron, n_neuron, n_param_, d_arr);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    CUDAFREECTRL("d_arr",d_arr);
  }
  CUDAFREECTRL("d_i_neuron",d_i_neuron);

  return 0;
}

// set receptor-port variable var_name of neurons
// i_neuron, ..., i_neuron + n_neuron -1
// using distribution or array
int BaseNeuron::SetPortVarDistr(int i_neuron, int n_neuron,
				std::string var_name,
				Distribution *distribution)
{
  if (!IsPortVar(var_name)) {
    throw ngpu_exception(std::string("Unrecognized port variable ")
			 + var_name);
  }
  CheckNeuronIdx(i_neuron);
  CheckNeuronIdx(i_neuron + n_neuron - 1);
  int vect_size = distribution->vectSize();
  if (vect_size != n_port_) {
    throw ngpu_exception("Distribution vector dimension must be "
			 "equal to the number of ports.");
  }
  float *var_pt;
    
  for (int i_vect=0; i_vect<vect_size; i_vect++) {
    var_pt = GetVarPt(i_neuron, var_name, i_vect);
    float *d_arr = distribution->getArray(*random_generator_, n_neuron, i_vect);
    BaseNeuronCopyFloatArray<<<(n_neuron+1023)/1024, 1024>>>
      (var_pt, n_neuron, n_var_, d_arr);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    CUDAFREECTRL("d_arr",d_arr);
  }
  return 0;
}

int BaseNeuron::SetPortVarDistr(int *i_neuron, int n_neuron,
				std::string var_name,
				Distribution *distribution)
{
  if (!IsPortVar(var_name)) {
    throw ngpu_exception(std::string("Unrecognized port variable ")
			 + var_name);
  }
  int vect_size = distribution->vectSize();
  if (vect_size != n_port_) {
    throw ngpu_exception("Distribution vector dimension must be "
			 "equal to the number of ports.");
  }
  int *d_i_neuron;
  CUDAMALLOCCTRL("&d_i_neuron",&d_i_neuron, n_neuron*sizeof(int));
  gpuErrchk(cudaMemcpy(d_i_neuron, i_neuron, n_neuron*sizeof(int),
		       cudaMemcpyHostToDevice));
  for (int i_vect=0; i_vect<vect_size; i_vect++) {
    float *var_pt = GetVarPt(0, var_name, i_vect);
    float *d_arr = distribution->getArray(*random_generator_, n_neuron, i_vect);
    BaseNeuronCopyFloatPtArray<<<(n_neuron+1023)/1024, 1024>>>
      (var_pt, d_i_neuron, n_neuron, n_var_, d_arr);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    CUDAFREECTRL("d_arr",d_arr);
  }
  CUDAFREECTRL("d_i_neuron",d_i_neuron);

  return 0;
}


    
//////////////////////////////////////////////////////////////////////



// get scalar parameters param_name of neurons
// i_neuron, ..., i_neuron + n_neuron -1
float *BaseNeuron::GetScalParam(int i_neuron, int n_neuron,
				std::string param_name)
{
  if (!IsScalParam(param_name)) {
    throw ngpu_exception(std::string("Unrecognized scalar parameter ")
			 + param_name);
  }
  CheckNeuronIdx(i_neuron);
  CheckNeuronIdx(i_neuron + n_neuron - 1);
  float *param_pt = GetParamPt(i_neuron, param_name);

  float *d_param_arr;
  CUDAMALLOCCTRL("&d_param_arr",&d_param_arr, n_neuron*sizeof(float));
  float *h_param_arr = (float*)malloc(n_neuron*sizeof(float));

  BaseNeuronGetFloatArray<<<(n_neuron+1023)/1024, 1024>>>
    (param_pt, d_param_arr, n_neuron, n_param_, 1);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  
  gpuErrchk(cudaMemcpy(h_param_arr, d_param_arr, n_neuron*sizeof(float),
		       cudaMemcpyDeviceToHost));
  CUDAFREECTRL("d_param_arr",d_param_arr);
  
  return h_param_arr;
}

// get scalar parameters param_name of neurons
// i_neuron[0], ..., i_neuron[n_neuron -1]
float *BaseNeuron::GetScalParam(int *i_neuron, int n_neuron,
				std::string param_name)
{
  if (!IsScalParam(param_name)) {
    throw ngpu_exception(std::string("Unrecognized scalar parameter ")
				     + param_name);
  }
  int *d_i_neuron;
  CUDAMALLOCCTRL("&d_i_neuron",&d_i_neuron, n_neuron*sizeof(int));
  // Memcopy will be synchronized with BaseNeuronGetFloatPtArray kernel
  gpuErrchk(cudaMemcpyAsync(d_i_neuron, i_neuron, n_neuron*sizeof(int),
		       cudaMemcpyHostToDevice));
  float *param_pt = GetParamPt(0, param_name);

  float *d_param_arr;
  CUDAMALLOCCTRL("&d_param_arr",&d_param_arr, n_neuron*sizeof(float));
  float *h_param_arr = (float*)malloc(n_neuron*sizeof(float));
  
  BaseNeuronGetFloatPtArray<<<(n_neuron+1023)/1024, 1024>>>
    (param_pt, d_param_arr, d_i_neuron, n_neuron, n_param_, 1);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  CUDAFREECTRL("d_i_neuron",d_i_neuron);

  gpuErrchk(cudaMemcpy(h_param_arr, d_param_arr, n_neuron*sizeof(float),
		       cudaMemcpyDeviceToHost));
  CUDAFREECTRL("d_param_arr",d_param_arr);

  return h_param_arr;
}

// get receptor-port parameters param_name of neurons
// i_neuron, ..., i_neuron + n_neuron -1
float *BaseNeuron::GetPortParam(int i_neuron, int n_neuron,
			      std::string param_name)
{
  if (!IsPortParam(param_name)) {
    throw ngpu_exception(std::string("Unrecognized port parameter ")
			 + param_name);
  }
  CheckNeuronIdx(i_neuron);
  CheckNeuronIdx(i_neuron + n_neuron - 1);
  float *param_pt;

  float *d_param_arr;
  CUDAMALLOCCTRL("&d_param_arr",&d_param_arr, n_neuron*n_port_*sizeof(float));
  float *h_param_arr = (float*)malloc(n_neuron*n_port_*sizeof(float));
  
  for (int port=0; port<n_port_; port++) {
    param_pt = GetParamPt(i_neuron, param_name, port);
    BaseNeuronGetFloatArray<<<(n_neuron+1023)/1024, 1024>>>
      (param_pt, d_param_arr + port, n_neuron, n_param_, n_port_);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }

  gpuErrchk(cudaMemcpy(h_param_arr, d_param_arr, n_neuron*n_port_
		       *sizeof(float), cudaMemcpyDeviceToHost));
  CUDAFREECTRL("d_param_arr",d_param_arr);
  
  return h_param_arr;
}

// get receptor-port parameters param_name of neurons
// i_neuron[0], ..., i_neuron[n_neuron -1]
float *BaseNeuron::GetPortParam(int *i_neuron, int n_neuron,
				std::string param_name)
{
  if (!IsPortParam(param_name)) {
    throw ngpu_exception(std::string("Unrecognized port parameter ")
			 + param_name);
  }
  int *d_i_neuron;
  CUDAMALLOCCTRL("&d_i_neuron",&d_i_neuron, n_neuron*sizeof(int));
  // Memcopy will be synchronized with BaseNeuronGetFloatPtArray kernel
  gpuErrchk(cudaMemcpyAsync(d_i_neuron, i_neuron, n_neuron*sizeof(int),
		       cudaMemcpyHostToDevice));

  float *d_param_arr;
  CUDAMALLOCCTRL("&d_param_arr",&d_param_arr, n_neuron*n_port_*sizeof(float));
  float *h_param_arr = (float*)malloc(n_neuron*n_port_*sizeof(float));
    
  for (int port=0; port<n_port_; port++) {
    float *param_pt = GetParamPt(0, param_name, port);
    BaseNeuronGetFloatPtArray<<<(n_neuron+1023)/1024, 1024>>>
      (param_pt, d_param_arr+port, d_i_neuron, n_neuron, n_param_,
       n_port_);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }
  CUDAFREECTRL("d_i_neuron",d_i_neuron);
  
  gpuErrchk(cudaMemcpy(h_param_arr, d_param_arr, n_neuron*n_port_
		       *sizeof(float), cudaMemcpyDeviceToHost));
  CUDAFREECTRL("d_param_arr",d_param_arr);
  
  return h_param_arr;
}

// get array-parameter param_name of neuron i_neuron
// must be defined in the derived classes
float *BaseNeuron::GetArrayParam(int i_neuron, std::string param_name)
{
  throw ngpu_exception(std::string("Unrecognized parameter ")
		       + param_name);
}

// get neuron-group parameter param_name
float BaseNeuron::GetGroupParam(std::string param_name)
{
  int i_param;
  for (i_param=0; i_param<n_group_param_; i_param++) {
    if (param_name == group_param_name_[i_param]) {
      return group_param_[i_param];
    }
  }
    
  throw ngpu_exception(std::string("Unrecognized group parameter ")
		       + param_name);
}

 
// get integer variable var_name of neurons
// i_neuron, ..., i_neuron + n_neuron -1
int *BaseNeuron::GetIntVar(int i_neuron, int n_neuron,
				std::string var_name)
{
  if (!IsIntVar(var_name)) {
    throw ngpu_exception(std::string("Unrecognized integer variable ")
			 + var_name);
  }
  CheckNeuronIdx(i_neuron);
  CheckNeuronIdx(i_neuron + n_neuron - 1);
  int *var_pt = GetIntVarPt(i_neuron, var_name);

  int *d_var_arr;
  CUDAMALLOCCTRL("&d_var_arr",&d_var_arr, n_neuron*sizeof(int));
  int *h_var_arr = (int*)malloc(n_neuron*sizeof(int));

  BaseNeuronGetIntArray<<<(n_neuron+1023)/1024, 1024>>>
    (var_pt, d_var_arr, n_neuron, 1, 1);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  
  gpuErrchk(cudaMemcpy(h_var_arr, d_var_arr, n_neuron*sizeof(int),
		       cudaMemcpyDeviceToHost));
  CUDAFREECTRL("d_var_arr",d_var_arr);
  
  return h_var_arr;
}

// get integer variable var_name of neurons
// i_neuron[0], ..., i_neuron[n_neuron -1]
int *BaseNeuron::GetIntVar(int *i_neuron, int n_neuron,
			   std::string var_name)
{
  if (!IsIntVar(var_name)) {
    throw ngpu_exception(std::string("Unrecognized integer variable ")
			 + var_name);
  }
  int *d_i_neuron;
  CUDAMALLOCCTRL("&d_i_neuron",&d_i_neuron, n_neuron*sizeof(int));
  // Memcopy will be synchronized with BaseNeuronGetIntPtArray kernel
  gpuErrchk(cudaMemcpyAsync(d_i_neuron, i_neuron, n_neuron*sizeof(int),
		       cudaMemcpyHostToDevice));
  int *var_pt = GetIntVarPt(0, var_name);

  int *d_var_arr;
  CUDAMALLOCCTRL("&d_var_arr",&d_var_arr, n_neuron*sizeof(int));
  int *h_var_arr = (int*)malloc(n_neuron*sizeof(int));
  
  BaseNeuronGetIntPtArray<<<(n_neuron+1023)/1024, 1024>>>
    (var_pt, d_var_arr, d_i_neuron, n_neuron, 1, 1);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  CUDAFREECTRL("d_i_neuron",d_i_neuron);

  gpuErrchk(cudaMemcpy(h_var_arr, d_var_arr, n_neuron*sizeof(int),
		       cudaMemcpyDeviceToHost));
  CUDAFREECTRL("d_var_arr",d_var_arr);
  
  return h_var_arr;
}

// get scalar state-variable var_name of neurons
// i_neuron, ..., i_neuron + n_neuron -1
float *BaseNeuron::GetScalVar(int i_neuron, int n_neuron,
				std::string var_name)
{
  if (!IsScalVar(var_name)) {
    throw ngpu_exception(std::string("Unrecognized scalar variable ")
			 + var_name);
  }
  CheckNeuronIdx(i_neuron);
  CheckNeuronIdx(i_neuron + n_neuron - 1);
  float *var_pt = GetVarPt(i_neuron, var_name);

  float *d_var_arr;
  CUDAMALLOCCTRL("&d_var_arr",&d_var_arr, n_neuron*sizeof(float));
  float *h_var_arr = (float*)malloc(n_neuron*sizeof(float));

  BaseNeuronGetFloatArray<<<(n_neuron+1023)/1024, 1024>>>
    (var_pt, d_var_arr, n_neuron, n_var_, 1);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  
  gpuErrchk(cudaMemcpy(h_var_arr, d_var_arr, n_neuron*sizeof(float),
		       cudaMemcpyDeviceToHost));
  CUDAFREECTRL("d_var_arr",d_var_arr);
  
  return h_var_arr;
}

// get scalar state-variable var_name of neurons
// i_neuron[0], ..., i_neuron[n_neuron -1]
float *BaseNeuron::GetScalVar(int *i_neuron, int n_neuron,
				std::string var_name)
{
  if (!IsScalVar(var_name)) {
    throw ngpu_exception(std::string("Unrecognized scalar variable ")
				     + var_name);
  }
  int *d_i_neuron;
  CUDAMALLOCCTRL("&d_i_neuron",&d_i_neuron, n_neuron*sizeof(int));
  // Memcopy will be synchronized with BaseNeuronGetFloatPtArray kernel
  gpuErrchk(cudaMemcpyAsync(d_i_neuron, i_neuron, n_neuron*sizeof(int),
		       cudaMemcpyHostToDevice));
  float *var_pt = GetVarPt(0, var_name);

  float *d_var_arr;
  CUDAMALLOCCTRL("&d_var_arr",&d_var_arr, n_neuron*sizeof(float));
  float *h_var_arr = (float*)malloc(n_neuron*sizeof(float));
  
  BaseNeuronGetFloatPtArray<<<(n_neuron+1023)/1024, 1024>>>
    (var_pt, d_var_arr, d_i_neuron, n_neuron, n_var_, 1);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  CUDAFREECTRL("d_i_neuron",d_i_neuron);

  gpuErrchk(cudaMemcpy(h_var_arr, d_var_arr, n_neuron*sizeof(float),
		       cudaMemcpyDeviceToHost));
  CUDAFREECTRL("d_var_arr",d_var_arr);

  return h_var_arr;
}

// get receptor-port state-variable var_name of neurons
// i_neuron, ..., i_neuron + n_neuron -1
float *BaseNeuron::GetPortVar(int i_neuron, int n_neuron,
			      std::string var_name)
{
  if (!IsPortVar(var_name)) {
    throw ngpu_exception(std::string("Unrecognized port variable ")
			 + var_name);
  }
  CheckNeuronIdx(i_neuron);
  CheckNeuronIdx(i_neuron + n_neuron - 1);
  float *var_pt;

  float *d_var_arr;
  CUDAMALLOCCTRL("&d_var_arr",&d_var_arr, n_neuron*n_port_*sizeof(float));
  float *h_var_arr = (float*)malloc(n_neuron*n_port_*sizeof(float));
  
  for (int port=0; port<n_port_; port++) {
    var_pt = GetVarPt(i_neuron, var_name, port);
    BaseNeuronGetFloatArray<<<(n_neuron+1023)/1024, 1024>>>
      (var_pt, d_var_arr + port, n_neuron, n_var_, n_port_);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }

  gpuErrchk(cudaMemcpy(h_var_arr, d_var_arr, n_neuron*n_port_
		       *sizeof(float), cudaMemcpyDeviceToHost));
  CUDAFREECTRL("d_var_arr",d_var_arr);
  
  return h_var_arr;
}

// get receptor-port state-variable var_name of neurons
// i_neuron[0], ..., i_neuron[n_neuron -1]
float *BaseNeuron::GetPortVar(int *i_neuron, int n_neuron,
			      std::string var_name)
{
  if (!IsPortVar(var_name)) {
    throw ngpu_exception(std::string("Unrecognized port variable ")
			 + var_name);
  }
  int *d_i_neuron;
  CUDAMALLOCCTRL("&d_i_neuron",&d_i_neuron, n_neuron*sizeof(int));
  // Memcopy will be synchronized with BaseNeuronGetFloatPtArray kernel
  gpuErrchk(cudaMemcpyAsync(d_i_neuron, i_neuron, n_neuron*sizeof(int),
		       cudaMemcpyHostToDevice));

  float *d_var_arr;
  CUDAMALLOCCTRL("&d_var_arr",&d_var_arr, n_neuron*n_port_*sizeof(float));
  float *h_var_arr = (float*)malloc(n_neuron*n_port_*sizeof(float));
    
  for (int port=0; port<n_port_; port++) {
    float *var_pt = GetVarPt(0, var_name, port);
    BaseNeuronGetFloatPtArray<<<(n_neuron+1023)/1024, 1024>>>
      (var_pt, d_var_arr+port, d_i_neuron, n_neuron, n_var_, n_port_);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }
  CUDAFREECTRL("d_i_neuron",d_i_neuron);
  
  gpuErrchk(cudaMemcpy(h_var_arr, d_var_arr, n_neuron*n_port_
		       *sizeof(float), cudaMemcpyDeviceToHost));
  CUDAFREECTRL("d_var_arr",d_var_arr);
  
  return h_var_arr;
}

// get array variable var_name of neuron  i_neuron
float *BaseNeuron::GetArrayVar(int i_neuron, std::string var_name)
{
  throw ngpu_exception(std::string("Unrecognized variable ")
		       + var_name);
}

// get index of integer variable var_name
int BaseNeuron::GetIntVarIdx(std::string var_name)
{
  int i_var;
  for (i_var=0; i_var<GetNIntVar(); i_var++) {
    if (var_name == int_var_name_[i_var]) break;
  }
  if (i_var == GetNIntVar()) {
    throw ngpu_exception(std::string("Unrecognized integer variable ")
			 + var_name);
  }
  
  return i_var;
}

// get index of scalar variable var_name
int BaseNeuron::GetScalVarIdx(std::string var_name)
{
  int i_var;
  for (i_var=0; i_var<n_scal_var_; i_var++) {
    if (var_name == scal_var_name_[i_var]) break;
  }
  if (i_var == n_scal_var_) {
    throw ngpu_exception(std::string("Unrecognized scalar variable ")
			 + var_name);
  }
  
  return i_var;
}

// get index of receptor-port variable var_name
int BaseNeuron::GetPortVarIdx(std::string var_name)
{
  int i_var;
  for (i_var=0; i_var<n_port_var_; i_var++) {
    if (var_name == port_var_name_[i_var]) break;
  }
  if (i_var == n_port_var_) {
    throw ngpu_exception(std::string("Unrecognized port variable ")
				     + var_name);
  }
  
  return i_var;
}

// get index of scalar parameter param_name
int BaseNeuron::GetScalParamIdx(std::string param_name)
{
  int i_param;
  for (i_param=0; i_param<n_scal_param_; i_param++) {
    if (param_name == scal_param_name_[i_param]) break;
  }
  if (i_param == n_scal_param_) {
    throw ngpu_exception(std::string("Unrecognized parameter ")
			 + param_name);
  }
  
  return i_param;
}

// get index of receptor-port parameter param_name
int BaseNeuron::GetPortParamIdx(std::string param_name)
{  
  int i_param;
  for (i_param=0; i_param<n_port_param_; i_param++) {
    if (param_name == port_param_name_[i_param]) break;
  }
  if (i_param == n_port_param_) {
    throw ngpu_exception(std::string("Unrecognized port parameter ")
			 + param_name);
  }
  
  return i_param;
}

// return pointer to state variable array
float *BaseNeuron::GetVarArr()
{
  return var_arr_;
}

// return pointer to parameter array
float *BaseNeuron::GetParamArr()
{
  return param_arr_;
}

// return array size for array variable var_name
// Must be defined in derived class
int BaseNeuron::GetArrayVarSize(int i_neuron, std::string var_name)
{
  throw ngpu_exception(std::string("Unrecognized variable ")
		       + var_name);

}
  
// return array size for array parameter param_name
// Must be defined in derived class
int BaseNeuron::GetArrayParamSize(int i_neuron, std::string param_name)
{
  throw ngpu_exception(std::string("Unrecognized parameter ")
		       + param_name);

}

// return size of variable var_name
// 1 for scalar variables, n_port for receptor-port variables
int BaseNeuron::GetVarSize(std::string var_name)
{
  if (IsScalVar(var_name)) {
    return 1;
  }
  else if (IsPortVar(var_name)) {
    return n_port_;
  }
  else if (IsArrayVar(var_name)) {
    throw ngpu_exception(std::string("Node index must be specified to get "
				     "array variable size for ")+ var_name);
  }
  else {
    throw ngpu_exception(std::string("Unrecognized variable ")
			 + var_name);
  }
}

// return size of parameter param_name
// 1 for scalar parameters, n_port for receptor-port parameters
int BaseNeuron::GetParamSize(std::string param_name)
{
  if (IsScalParam(param_name)) {
    return 1;
  }
  else if (IsPortParam(param_name)) {
    return n_port_;
  }
  else if (IsArrayParam(param_name)) {
    throw ngpu_exception(std::string("Node index must be specified to get "
				     "array parameter size for ")+ param_name);
  }
  else {
    throw ngpu_exception(std::string("Unrecognized parameter ")
			 + param_name);
  }
}

// check if var_name is an integer variable
bool BaseNeuron::IsIntVar(std::string var_name)
{
  int i_var;
  for (i_var=0; i_var<GetNIntVar(); i_var++) {
    if (var_name == int_var_name_[i_var]) return true;
  }
  return false;
}

// check if var_name is a scalar variable
bool BaseNeuron::IsScalVar(std::string var_name)
{
  int i_var;
  for (i_var=0; i_var<n_scal_var_; i_var++) {
    if (var_name == scal_var_name_[i_var]) return true;
  }
  return false;
}

// check if var_name is a receptor-port variable
bool BaseNeuron::IsPortVar(std::string var_name)
{
  int i_var;
  for (i_var=0; i_var<n_port_var_; i_var++) {
    if (var_name == port_var_name_[i_var]) return true;
  }
  return false;
}

// check if var_name is an array variable
bool BaseNeuron::IsArrayVar(std::string var_name)
{
  int i_var;
  for (i_var=0; i_var<GetNArrayVar(); i_var++) {
    if (var_name == array_var_name_[i_var]) return true;
  }
  return false;
}

// check if param_name is a scalar parameter
bool BaseNeuron::IsScalParam(std::string param_name)
{
  int i_param;
  for (i_param=0; i_param<n_scal_param_; i_param++) {
    if (param_name == scal_param_name_[i_param]) return true;
  }
  return false;
}

// check if param_name is a receptor-port parameter
bool BaseNeuron::IsPortParam(std::string param_name)
{  
  int i_param;
  for (i_param=0; i_param<n_port_param_; i_param++) {
    if (param_name == port_param_name_[i_param]) return true;
  }
  return false;
}

// check if param_name is an array parameter
bool BaseNeuron::IsArrayParam(std::string param_name)
{
  int i_param;
  for (i_param=0; i_param<GetNArrayParam(); i_param++) {
    if (param_name == array_param_name_[i_param]) return true;
  }
  return false;
}

// check if param_name is a neuron-group parameter
bool BaseNeuron::IsGroupParam(std::string param_name)
{
  int i_param;
  for (i_param=0; i_param<n_group_param_; i_param++) {
    if (param_name == group_param_name_[i_param]) return true;
  }
  return false;
}

// check if index i_neuron is >=0 and <n_node for the neuron group
int BaseNeuron::CheckNeuronIdx(int i_neuron)
{
  if (i_neuron>=n_node_) {
    throw ngpu_exception("Neuron index must be lower then n. of neurons");
  }
  else if (i_neuron<0) {
    throw ngpu_exception("Neuron index must be >= 0");
  }
  return 0;
}

// check if index port is >=0 and <n_port
int BaseNeuron::CheckPortIdx(int port)
{
  if (port>=n_port_) {
    throw ngpu_exception("Port index must be lower then n. of ports");
  }
  else if (port<0) {
    throw ngpu_exception("Port index must be >= 0");
  }
  return 0;
}

// return pointer to integer variable var_name for neuron i_neuron
int *BaseNeuron::GetIntVarPt(int i_neuron, std::string var_name)
{
  CheckNeuronIdx(i_neuron);
    
  if (IsIntVar(var_name)) {
    int i_var =  GetIntVarIdx(var_name);
    return int_var_pt_[i_var] + i_neuron; 
  }
  else {
    throw ngpu_exception(std::string("Unrecognized integer variable ")
			 + var_name);
  }
}

// return pointer to variable var_name for neuron i_neuron
// (and specified receptor port in case of a port variable)
float *BaseNeuron::GetVarPt(int i_neuron, std::string var_name,
			    int port /*=0*/)
{
  CheckNeuronIdx(i_neuron);
  if (port!=0) {
    CheckPortIdx(port);
  }
    
  if (IsScalVar(var_name)) {
    int i_var =  GetScalVarIdx(var_name);
    return GetVarArr() + i_neuron*n_var_ + i_var;
  }
  else if (IsPortVar(var_name)) {
    int i_vvar =  GetPortVarIdx(var_name);
    return GetVarArr() + i_neuron*n_var_ + n_scal_var_
      + port*n_port_var_ + i_vvar;
  }
  else {
    throw ngpu_exception(std::string("Unrecognized variable ")
			 + var_name);
  }
}

// return pointer to parameter param_name for neuron i_neuron
// (and specified receptor port in case of a port parameter)
float *BaseNeuron::GetParamPt(int i_neuron, std::string param_name,
			      int port /*=0*/)
{
  CheckNeuronIdx(i_neuron);
  if (port!=0) {
    CheckPortIdx(port);
  }
  if (IsScalParam(param_name)) {
    int i_param =  GetScalParamIdx(param_name);
    return GetParamArr() + i_neuron*n_param_ + i_param;
  }
  else if (IsPortParam(param_name)) {
    int i_vparam =  GetPortParamIdx(param_name);
    return GetParamArr() + i_neuron*n_param_ + n_scal_param_
      + port*n_port_param_ + i_vparam;
  }
  else {
    throw ngpu_exception(std::string("Unrecognized parameter ")
			 + param_name);
  }
}

// return spike multiplicity (spike_height) of neuron i_neuron 
// if neuron emitted a spike in the current time step
// otherwise return 0
float BaseNeuron::GetSpikeActivity(int i_neuron)
{
  CheckNeuronIdx(i_neuron);
  int i_spike_buffer = i_neuron + i_node_0_;
  int Ns;
  gpuErrchk(cudaMemcpy(&Ns, d_SpikeBufferSize + i_spike_buffer,
		       sizeof(int), cudaMemcpyDeviceToHost));
  if (Ns==0) {
    return 0.0;
  }
  
  int is0;
  gpuErrchk(cudaMemcpy(&is0, d_SpikeBufferIdx0 + i_spike_buffer,
		       sizeof(int), cudaMemcpyDeviceToHost));
  int i_arr = is0*h_NSpikeBuffer+i_spike_buffer; // spike index in array

  int time_idx;
  // get first (most recent) spike from buffer
  gpuErrchk(cudaMemcpy(&time_idx, d_SpikeBufferTimeIdx + i_arr,
		       sizeof(int), cudaMemcpyDeviceToHost));
  if (time_idx!=0) { // neuron is not spiking now
    return 0.0;
  }
  float spike_height;
  gpuErrchk(cudaMemcpy(&spike_height, d_SpikeBufferHeight + i_arr,
		       sizeof(float), cudaMemcpyDeviceToHost));

  return spike_height;
}

// get all names of integer variables
std::vector<std::string> BaseNeuron::GetIntVarNames()
{
  return int_var_name_;
}

// get all names of scalar state variables
std::vector<std::string> BaseNeuron::GetScalVarNames()
{
  std::vector<std::string> var_name_vect;
  for (int i=0; i<n_scal_var_; i++) {
    var_name_vect.push_back(scal_var_name_[i]);
  }
  
  return var_name_vect;
}

// get number of scalar state variables
int BaseNeuron::GetNScalVar()
{
  return n_scal_var_;
}

// get number of integer variables
int BaseNeuron::GetNIntVar()
{
  return (int)int_var_name_.size();
}

// get all names of receptor-port state variables
std::vector<std::string> BaseNeuron::GetPortVarNames()
{
  std::vector<std::string> var_name_vect;
  for (int i=0; i<n_port_var_; i++) {
    var_name_vect.push_back(port_var_name_[i]);
  }
  
  return var_name_vect;
}

// get number of receptor-port variables 
int BaseNeuron::GetNPortVar()
{
  return n_port_var_;
}

// get all names of scalar parameters
std::vector<std::string> BaseNeuron::GetScalParamNames()
{
  std::vector<std::string> param_name_vect;
  for (int i=0; i<n_scal_param_; i++) {
    param_name_vect.push_back(scal_param_name_[i]);
  }
  
  return param_name_vect;
}

// get number of scalar parameters
int BaseNeuron::GetNScalParam()
{
  return n_scal_param_;
}

// get all names of receptor-port parameters
std::vector<std::string> BaseNeuron::GetPortParamNames()
{
  std::vector<std::string> param_name_vect;
  for (int i=0; i<n_port_param_; i++) {
    param_name_vect.push_back(port_param_name_[i]);
  }
  
  return param_name_vect;
}

// get number of receptor-port parameters
int BaseNeuron::GetNPortParam()
{
  return n_port_param_;
}

// get all names of neuron-group parameters
std::vector<std::string> BaseNeuron::GetGroupParamNames()
{
  std::vector<std::string> param_name_vect;
  for (int i=0; i<n_group_param_; i++) {
    param_name_vect.push_back(group_param_name_[i]);
  }
  
  return param_name_vect;
}

// get number of neuron-group parameters
int BaseNeuron::GetNGroupParam()
{
  return n_group_param_;
}

// get all names of array variables
std::vector<std::string> BaseNeuron::GetArrayVarNames()
{
  std::vector<std::string> var_name_vect;
  for (int i=0; i<GetNArrayVar(); i++) {
    var_name_vect.push_back(array_var_name_[i]);
  }
  
  return var_name_vect;
}

// get number of array variables
int BaseNeuron::GetNArrayVar()
{
  return (int)array_var_name_.size();
}

// get all names of array parameters
std::vector<std::string> BaseNeuron::GetArrayParamNames()
{
  std::vector<std::string> param_name_vect;
  for (int i=0; i<GetNArrayParam(); i++) {
    param_name_vect.push_back(array_param_name_[i]);
  }
  
  return param_name_vect;
}

// get number of array parameters
int BaseNeuron::GetNArrayParam()
{
  return (int)array_param_name_.size();
}

// activate spike count for all neurons of the group
int BaseNeuron::ActivateSpikeCount()
{
  const std::string s = "spike_count";
  if (std::find(int_var_name_.begin(), int_var_name_.end(), s)
      == int_var_name_.end()) { // add it if not already present 
    int_var_name_.push_back(s);

    CUDAMALLOCCTRL("&spike_count_",&spike_count_, n_node_*sizeof(int));
    gpuErrchk(cudaMemset(spike_count_, 0, n_node_*sizeof(int)));
    int_var_pt_.push_back(spike_count_);
  }
  else {
    throw ngpu_exception("Spike count already activated");
  }


  return 0;
}

// activate spike-time recording for all neurons of the group
int BaseNeuron::ActivateRecSpikeTimes(int max_n_rec_spike_times)
{
  if(max_n_rec_spike_times<=0) {
    throw ngpu_exception("Maximum number of recorded spike times "
			 "must be greater than 0");
  }
  const std::string s = "n_rec_spike_times";
  if (std::find(int_var_name_.begin(), int_var_name_.end(), s)
      == int_var_name_.end()) { // add it if not already present 
    int_var_name_.push_back(s);

    CUDAMALLOCCTRL("&n_rec_spike_times_",&n_rec_spike_times_, n_node_*sizeof(int));
    CUDAMALLOCCTRL("&n_rec_spike_times_cumul_",&n_rec_spike_times_cumul_,
			 (n_node_+1)*sizeof(int));
    gpuErrchk(cudaMemset(n_rec_spike_times_, 0, n_node_*sizeof(int)));
    int_var_pt_.push_back(n_rec_spike_times_);
    
    max_n_rec_spike_times_ = max_n_rec_spike_times;
    CUDAMALLOCCTRL("&rec_spike_times_",&rec_spike_times_, n_node_*max_n_rec_spike_times
			 *sizeof(int));
    CUDAMALLOCCTRL("&rec_spike_times_pack_",&rec_spike_times_pack_, n_node_*max_n_rec_spike_times
			 *sizeof(int));
    spike_times_pt_vect_.resize(n_node_, NULL);
    n_spike_times_vect_.resize(n_node_, 0);
    spike_times_vect_.resize(n_node_);
  }
  else {
    throw ngpu_exception("Spike times recording already activated");
  }

  return 0;
}

// set number of time steps for buffering recorded spike times
int BaseNeuron::SetRecSpikeTimesStep(int rec_spike_times_step)
{
  rec_spike_times_step_ = rec_spike_times_step;

  return 0;
}

// get number of spikes recorded for neuron i_neuron
int BaseNeuron::GetNRecSpikeTimes(int i_neuron)
{
  CheckNeuronIdx(i_neuron);
  if(max_n_rec_spike_times_<=0) {
    throw ngpu_exception("Spike times recording was not activated");
  }
  int n_spikes;
  
  gpuErrchk(cudaMemcpy(&n_spikes, &n_rec_spike_times_[i_neuron], sizeof(int),
		       cudaMemcpyDeviceToHost));
  return n_spikes;
}

// get input spikes from external interface
// Must be defined in derived classes
float *BaseNeuron::GetExtNeuronInputSpikes(int *n_node, int *n_port)
{
  throw ngpu_exception("Cannot get extern neuron input spikes from this model");
}

// set neuron-group parameter param_name to value val
// Must be defined in derived classes
int BaseNeuron::SetNeuronGroupParam(std::string param_name, float val)
{
  throw ngpu_exception(std::string("Unrecognized neuron group parameter ")
		       + param_name);
}



// kernel for packing spike times of neurons
// i_neuron, ..., i_neuron + n_neuron -1
// in contiguous locations in GPU memory 
__global__ void PackSpikeTimesKernel(int n_neuron, int *n_rec_spike_times_cumul,
		     float *rec_spike_times, float *rec_spike_times_pack,
		     int n_spike_tot, int max_n_rec_spike_times)
{
  // array_idx: index on one-dimensional packed spike array 
  int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (array_idx<n_spike_tot) {
    // a locate of array_idx on the cumulative sum of the number of spikes
    // of the neurons is used to get the neuron index
    int i_neuron = locate(array_idx, n_rec_spike_times_cumul, n_neuron + 1);
    // if neuron has no spikes, go to the next
    while ((i_neuron < n_neuron) && (n_rec_spike_times_cumul[i_neuron+1]
				  == n_rec_spike_times_cumul[i_neuron])) {
      i_neuron++;
      if (i_neuron==n_neuron) return;
    }
    // the difference gives the spike index
    int i_spike = array_idx - n_rec_spike_times_cumul[i_neuron];
    // copy the spike to the packed array
    rec_spike_times_pack[array_idx] =
      rec_spike_times[i_neuron*max_n_rec_spike_times + i_spike];
  }
}

// extract recorded spike times
// and put them in a buffer
int BaseNeuron::BufferRecSpikeTimes()
{  
  if(max_n_rec_spike_times_<=0) {
    throw ngpu_exception("Spike times recording was not activated");
  }
  // a cumulative sum is used by the spike-packing algorithm
  prefix_scan(n_rec_spike_times_cumul_, n_rec_spike_times_,
	      n_node_+1, true);
  int *h_n_rec_spike_times_cumul = new int[n_node_+1];
  gpuErrchk(cudaMemcpy(h_n_rec_spike_times_cumul,
			    n_rec_spike_times_cumul_,
			    (n_node_+1)*sizeof(int), cudaMemcpyDeviceToHost));
  // the last element of the cumulative sum is the total number of spikes
  int n_spike_tot = h_n_rec_spike_times_cumul[n_node_];

  if (n_spike_tot>0) {
    // pack spike times in GPU memory
    PackSpikeTimesKernel<<<(n_spike_tot+1023)/1024, 1024>>>(n_node_,
		     n_rec_spike_times_cumul_,
		     rec_spike_times_,
		     rec_spike_times_pack_,
		     n_spike_tot, max_n_rec_spike_times_);

    float *h_rec_spike_times_pack = new float[n_spike_tot];
    gpuErrchk(cudaMemcpy(h_rec_spike_times_pack,
			 rec_spike_times_pack_,
			 sizeof(float)*n_spike_tot, cudaMemcpyDeviceToHost));
    // push the packed spike array and the cumulative sum in the buffers
    spike_times_buffer_.push_back(h_rec_spike_times_pack);
    n_spike_times_cumul_buffer_.push_back(h_n_rec_spike_times_cumul);
    gpuErrchk(cudaMemset(n_rec_spike_times_, 0, n_node_*sizeof(int)));
  }
  else {
    delete[] h_n_rec_spike_times_cumul;
  }
  
  return 0;
}

// get recorded spike times
int BaseNeuron::GetRecSpikeTimes(int **n_spike_times_pt,
				 float ***spike_times_pt)
{
  if(max_n_rec_spike_times_<=0) {
    throw ngpu_exception("Spike times recording was not activated");
  }
  // push all spikes and cumulative sums left in the buffers
  BufferRecSpikeTimes();

  // first evaluate the total number of spikes for each node
  for (int i_node=0; i_node<n_node_; i_node++) {
    n_spike_times_vect_[i_node] = 0;
    // loop on buffer entries
    for (uint i_buf=0; i_buf<spike_times_buffer_.size(); i_buf++) {
      int *n_spike_times_cumul = n_spike_times_cumul_buffer_[i_buf];
      // get the number of spikes of each buffer entry
      int n_spike = n_spike_times_cumul[i_node+1] - n_spike_times_cumul[i_node];
      // and add it to the number of spikes of the node
      n_spike_times_vect_[i_node] += n_spike;
    }
    // allocate the spike vector for the considered node
    spike_times_vect_[i_node].resize(n_spike_times_vect_[i_node]);

    int k = 0;
    // loop on buffer entries
    for (uint i_buf=0; i_buf<spike_times_buffer_.size(); i_buf++) {
      float *spike_times_pack = spike_times_buffer_[i_buf];
      int *n_spike_times_cumul = n_spike_times_cumul_buffer_[i_buf];
      // array_idx: index of the first spike of node i_node
      // on one-dimensional packed spike array      
      int array_idx = n_spike_times_cumul[i_node];
      int n_spike = n_spike_times_cumul[i_node+1] - array_idx;
     
      float *pt = spike_times_pack + array_idx;
      // insert the spikes of node i_node in its spike vector 
      spike_times_vect_[i_node].insert(spike_times_vect_[i_node].begin()+k,
				       pt, pt+n_spike);
      k += n_spike;
    }
  }
  for (int i_node=0; i_node<n_node_; i_node++) {
    // pointer to spike vector data of node i_node
    spike_times_pt_vect_[i_node] = spike_times_vect_[i_node].data();
  }
  spike_times_buffer_.clear();
  n_spike_times_cumul_buffer_.clear();                                  
 
  *n_spike_times_pt = n_spike_times_vect_.data();
  *spike_times_pt = spike_times_pt_vect_.data();
  
  return 0;
}
  
