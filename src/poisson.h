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





#ifndef POISSON_H
#define POISSON_H
#include <cuda.h>
#include <curand.h>

extern __device__ unsigned int *PoissonData;

class PoissonGenerator
{
  unsigned int *dev_poisson_data_;
  int poisson_data_size_;

  int buffer_size_;
  int n_steps_;
  int i_step_;
  float lambda_;
  int more_steps_;
  int i_node_0_;

  
  int Init(curandGenerator_t *random_generator, unsigned int n);

 public:
  curandGenerator_t *random_generator_;
  int n_node_;
    
  PoissonGenerator();

  ~PoissonGenerator();

  int Free();
  
  int Create(curandGenerator_t *random_generator,
	     int i_node_0, int n_node, float lambda);
  
  int Generate();

  int Generate(int max_n_steps);

  int Update(int max_n_steps);

};

#endif
