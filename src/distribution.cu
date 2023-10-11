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

#include <vector>
#include <cuda.h>
#include <curand.h>
#include "cuda_error.h"
#include "ngpu_exception.h"
#include "distribution.h"


__global__ void randomNormalClippedKernel(float *arr, int64_t n, float mu,
					  float sigma, float low, float high,
					  double normal_cdf_alpha,
					  double normal_cdf_beta)
{
  const double epsilon=1.0e-15;
  int64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid>=n) return;
  float uniform = arr[tid];
  double p = normal_cdf_alpha + (normal_cdf_beta - normal_cdf_alpha) * uniform;
  double v = p * 2.0 - 1.0;
  v = max(v,  epsilon - 1.0);
  v = min(v, -epsilon + 1.0);
  double x = (double)sigma * sqrt(2.0) * erfinv(v) + mu;
  x = max(x, low);
  x = min(x, high);
  arr[tid] = (float)x;
}

double normalCDF(double value)
{
   return 0.5 * erfc(-value * M_SQRT1_2);
}

int randomNormalClipped(float *arr, int64_t n, float mu,
			float sigma, float low, float high)
{
  double alpha = ((double)low - mu) / sigma;
  double beta = ((double)high - mu) / sigma;
  double normal_cdf_alpha = normalCDF(alpha);
  double normal_cdf_beta = normalCDF(beta);

  //printf("mu: %f\tsigma: %f\tlow: %f\thigh: %f\tn: %ld\n",
  //	 mu, sigma, low, high, n);
  //n = 10000;
  randomNormalClippedKernel<<<(n+1023)/1024, 1024>>>(arr, n, mu, sigma,
						     low, high,
						     normal_cdf_alpha,
						     normal_cdf_beta);
  DBGCUDASYNC
  // temporary test, remove!!!!!!!!!!!!!
  //gpuErrchk( cudaDeviceSynchronize() );
  //float h_arr[10000];
  //gpuErrchk(cudaMemcpy(h_arr, arr, n*sizeof(float), cudaMemcpyDeviceToHost));
  //for (int i=0; i<n; i++) {
  //  printf("arr: %f\n", h_arr[i]);
  //}
  //exit(0);

  return 0;
}



bool Distribution::isDistribution(int distr_idx)
{
  if (distr_idx>DISTR_TYPE_ARRAY && distr_idx<N_DISTR_TYPE) {
    return true;
  }
  else {
    return false;
  }
}
  
bool Distribution::isArray(int distr_idx)
{
  if (distr_idx==DISTR_TYPE_ARRAY) {
    return true;
  }
  else {
    return false;
  }
}

void Distribution::checkDistributionInitialized()
{
  if (distr_idx_<DISTR_TYPE_ARRAY || distr_idx_>=N_DISTR_TYPE) {
    throw ngpu_exception("Distribution was not initialized");
  }
}

int Distribution::vectSize()
{
  return vect_size_;
}

float *Distribution::getArray(curandGenerator_t &gen, int64_t n_elem,
			      int i_vect)
{
  checkDistributionInitialized();
  if (distr_idx_>=DISTR_TYPE_ARRAY) {
    CUDAMALLOCCTRL("&d_array_pt_",&d_array_pt_, n_elem*sizeof(float));
  }
  if (distr_idx_==DISTR_TYPE_ARRAY) {
    gpuErrchk(cudaMemcpy(d_array_pt_, h_array_pt_, n_elem*sizeof(float),
			 cudaMemcpyHostToDevice));    
  }
  else if (distr_idx_==DISTR_TYPE_NORMAL_CLIPPED) {
    //printf("ok0\n");
    CURAND_CALL(curandGenerateUniform(gen, d_array_pt_, n_elem));
    //printf("ok1\n");
    randomNormalClipped(d_array_pt_, n_elem, mu_[i_vect], sigma_[i_vect],
			low_[i_vect], high_[i_vect]);
    //printf("ok2\n");
  }
  else if (distr_idx_==DISTR_TYPE_NORMAL) {
    float low = mu_[i_vect] - 5.0*sigma_[i_vect];
    float high = mu_[i_vect] + 5.0*sigma_[i_vect];
    CURAND_CALL(curandGenerateUniform(gen, d_array_pt_, n_elem));
    randomNormalClipped(d_array_pt_, n_elem, mu_[i_vect], sigma_[i_vect],
			low, high);
  }
  return d_array_pt_;
}

int Distribution::SetIntParam(std::string param_name, int val)
{
  if (param_name=="distr_idx") {
    if (isDistribution(val) || isArray(val)) {
      distr_idx_ = val;
      vect_size_ = 0;
      mu_.clear();
      sigma_.clear();
      low_.clear();
      high_.clear();

    }
    else {
      throw ngpu_exception("Invalid distribution type");
    }
  }
  else if (param_name=="vect_size") {
    vect_size_ = val;
    mu_.resize(vect_size_);
    sigma_.resize(vect_size_);
    low_.resize(vect_size_);
    high_.resize(vect_size_);
  }
  else {
    throw ngpu_exception(std::string("Unrecognized distribution "
				     "integer parameter ") + param_name);
  }

  return 0;
}

int Distribution::SetScalParam(std::string param_name, float val)
{
  //printf("dok0\n");
  checkDistributionInitialized();
  //printf("dok1\n");
  if (vect_size_ <= 0) {
    throw ngpu_exception("Distribution parameter vector dimension "
			 "was not initialized");
  }
  else if (vect_size_>1) {
    throw ngpu_exception("Distribution parameter vector dimension"
			 " inconsistent for scalar parameter");
  }
  //printf("dok2\n");
  SetVectParam(param_name, val, 0);
  
  return 0;
}

int Distribution::SetVectParam(std::string param_name, float val, int i)
{
  //printf("dok3\n");
  checkDistributionInitialized();
  //printf("dok4\n");
  if (vect_size_ <= 0) {
    throw ngpu_exception("Distribution parameter vector dimension "
			 "was not initialized");
  }
  if (i > vect_size_) {
    throw ngpu_exception("Vector parameter index for distribution "
			 "out of range");
  }
  //printf("dok5\n");
  if (param_name=="mu") {
    // aggiungere && distr_idx==NORMAL || distr_idx==NORMAL_CLIPPED
    //printf("dok6 i: %d val: %f\n", i, val);
    mu_[i] = val;
  }
  else if (param_name=="sigma") {
    sigma_[i] = val;
  }
  else if (param_name=="low") {
    low_[i] = val;
  }
  else if (param_name=="high") {
    high_[i] = val;
  }
  else {
    throw ngpu_exception(std::string("Unrecognized distribution "
				     "float parameter ") + param_name);
  }
  //printf("dok7\n");
  
  return 0;
}

int Distribution::SetFloatPtParam(std::string param_name, float *h_array_pt)
{
  if (param_name=="array_pt") {
    distr_idx_ = DISTR_TYPE_ARRAY;
    h_array_pt_ = h_array_pt;
  }
  else {
    throw ngpu_exception(std::string("Unrecognized distribution "
				     "float pointer parameter ") + param_name);
  }

  return 0;
}

bool Distribution::IsFloatParam(std::string param_name)
{
  if ((param_name=="mu")
      || (param_name=="sigma")
      || (param_name=="low")
      || (param_name=="high")) {
    return true;
  }
  else {
    return false;
  }
}
