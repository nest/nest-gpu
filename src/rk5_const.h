/*
 *  rk5_const.h
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

#ifndef RK5CONST_H
#define RK5CONST_H


extern __constant__ float c2;
extern __constant__ float c3;
extern __constant__ float c4;
extern __constant__ float c5;
extern __constant__ float c6;
extern __constant__ float a21;
extern __constant__ float a31;
extern __constant__ float a32;
extern __constant__ float a41;
extern __constant__ float a42;
extern __constant__ float a43;
extern __constant__ float a51;
extern __constant__ float a52;
extern __constant__ float a53;
extern __constant__ float a54;
extern __constant__ float a61;
extern __constant__ float a62;
extern __constant__ float a63;
extern __constant__ float a64;
extern __constant__ float a65;

extern __constant__ float a71;
extern __constant__ float a73;
extern __constant__ float a74;
extern __constant__ float a76;

extern __constant__ float e1;
extern __constant__ float e3;
extern __constant__ float e4;
extern __constant__ float e5;
extern __constant__ float e6;

extern __constant__ float eps;
extern __constant__ float coeff;
extern __constant__ float exp_inc;
extern __constant__ float exp_dec;
extern __constant__ float err_min;
extern __constant__ float scal_min;

#endif
