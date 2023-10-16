/*
 *  distribution.h
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

#ifndef DISTRIBUTION_H
#define DISTRIBUTION_H
#include <vector>
#include <cuda.h>
#include <curand.h>

class Distribution
{
  //curandGenerator_t *curand_generator_;
  int distr_idx_;
  int vect_size_;
  float *h_array_pt_;
  float *d_array_pt_;
  std::vector<float> mu_;
  std::vector<float> sigma_;
  std::vector<float> low_;
  std::vector<float> high_;

public:
  //void setCurandGenerator(curandGenerator_t *gen)
  //{curand_generator_ = gen;}
  
  bool isDistribution(int distr_idx);
  
  bool isArray(int distr_idx);

  void checkDistributionInitialized();

  int vectSize();

  float *getArray(curandGenerator_t &gen, int64_t n_elem, int i_vect = 0);
  
  int SetIntParam(std::string param_name, int val);

  int SetScalParam(std::string param_name, float val);

  int SetVectParam(std::string param_name, float val, int i);

  int SetFloatPtParam(std::string param_name, float *h_array_pt);

  bool IsFloatParam(std::string param_name);

};

enum DistributionType {
  DISTR_TYPE_NONE=0,
  DISTR_TYPE_ARRAY,
  DISTR_TYPE_NORMAL,
  DISTR_TYPE_NORMAL_CLIPPED,
  N_DISTR_TYPE
};

int randomNormalClipped(float *arr, int64_t n, float mu,
			float sigma, float low, float high);

#endif
