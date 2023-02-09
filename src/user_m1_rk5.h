/*
 *  user_m1_rk5.h
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





#ifndef USERM1RK5_H
#define USERM1RK5_H

struct user_m1_rk5;


template<int NVAR, int NPARAM>
__device__
void Derivatives(double x, float *y, float *dydx, float *param,
		 user_m1_rk5 data_struct);

template<int NVAR, int NPARAM>
__device__
void ExternalUpdate(double x, float *y, float *param, bool end_time_step,
		    user_m1_rk5 data_struct);

__device__
void NodeInit(int n_var, int n_param, double x, float *y,
	      float *param, user_m1_rk5 data_struct);

__device__
void NodeCalibrate(int n_var, int n_param, double x, float *y,
		   float *param, user_m1_rk5 data_struct);

#endif
