/*
 *  rk5.h
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





#ifndef RK5_H
#define RK5_H

#include "cuda_error.h"
#include "rk5_const.h"
#include "rk5_interface.h"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

__global__ void SetFloatArray(float *arr, int n_elem, int step, float val);

template<class DataStruct>
__global__
void ArrayInit(int array_size, int n_var, int n_param, double *x_arr,
	       float *h_arr, float *y_arr, float *par_arr, double x_min,
	       float h, DataStruct data_struct)
{
  int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (array_idx<array_size) {
    NodeInit(n_var, n_param, x_min, &y_arr[array_idx*n_var],
	     &par_arr[array_idx*n_param], data_struct);
    x_arr[array_idx] = x_min;
    h_arr[array_idx] = h;
  }
}

template<class DataStruct>
__global__
void ArrayCalibrate(int array_size, int n_var, int n_param, double *x_arr,
		    float *h_arr, float *y_arr, float *par_arr, double x_min,
		    float h, DataStruct data_struct)
{
  int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (array_idx<array_size) {
    NodeCalibrate(n_var, n_param, x_min, &y_arr[array_idx*n_var],
		  &par_arr[array_idx*n_param], data_struct);
    x_arr[array_idx] = x_min;
    h_arr[array_idx] = h;
  }
}

template<int NVAR, int NPARAM, class DataStruct>
__device__
void RK5Step(double &x, float *y, float &h, float h_min, float h_max,
	     float *param, DataStruct data_struct)
{
  float y_new[NVAR];
  float k1[NVAR];
  float k2[NVAR];
  float k3[NVAR];
  float k4[NVAR];
  float k5[NVAR];
  float k6[NVAR];
  float y_scal[NVAR];

  Derivatives<NVAR, NPARAM>(x, y, k1, param, data_struct);
  for (int i=0; i<NVAR; i++) {
    y_scal[i] = fabs(y[i]) + fabs(k1[i]*h) + scal_min;
  }
  
  float err;
  for(;;) {
    if (h > h_max) h = h_max;
    if (h < h_min) h = h_min;
    
    for (int i=0; i<NVAR; i++) {
      y_new[i] = y[i] + h*a21*k1[i];
    }

    Derivatives<NVAR, NPARAM>(x+c2*h, y_new, k2, param,
			      data_struct);
  
    for (int i=0; i<NVAR; i++) {
      y_new[i] = y[i] + h*(a31*k1[i] + a32*k2[i]);
    }
    Derivatives<NVAR, NPARAM>(x+c3*h, y_new, k3, param,
			      data_struct);

    for (int i=0; i<NVAR; i++) {
      y_new[i] = y[i] + h*(a41*k1[i] + a42*k2[i] + a43*k3[i]);
    }
    Derivatives<NVAR, NPARAM>(x+c4*h, y_new, k4, param,
			      data_struct);
  
    for (int i=0; i<NVAR; i++) {
      y_new[i] = y[i] + h*(a51*k1[i] + a52*k2[i] + a53*k3[i] + a54*k4[i]);
    }
    Derivatives<NVAR, NPARAM>(x+c5*h, y_new, k5, param,
			      data_struct);
  
    for (int i=0; i<NVAR; i++) {
      y_new[i] = y[i] + h*(a61*k1[i] + a62*k2[i] + a63*k3[i] + a64*k4[i]
			  + a65*k5[i]);
    }
    Derivatives<NVAR, NPARAM>(x+c6*h, y_new, k6, param, data_struct);
    
    for (int i=0; i<NVAR; i++) {
      y_new[i] = y[i] + h*(a71*k1[i] + a73*k3[i] + a74*k4[i] + a76*k6[i]);
    }
  
    err = 0.0;
    for (int i=0; i<NVAR; i++) {
      float val = h*(e1*k1[i] + e3*k3[i] + e4*k4[i] + e5*k5[i] + e6*k6[i]);
      val /= y_scal[i]; ///// check for overflow!!!!!!!!!!!
      err = MAX(err,fabs(val));
    }
    err /= eps;
    if (err <= 1.0 || h<=h_min*(1.0+1.0e-5)) break;

    float h_new = h*coeff*pow(err,exp_dec);
    h = MAX(h_new, 0.1*h);
    
    //if (h <= h_min) {
    //  h = h_min;
    //}
    //x_new = x + h;
  }

  x += h;  

  if (err > err_min) {
    h = h*coeff*pow(err,exp_inc);
  }
  else {
    h = 5.0*h;
  }
  
  for (int i=0; i<NVAR; i++) {
    y[i] = y_new[i];
  }

}

template<int NVAR, int NPARAM, class DataStruct>
__device__
void RK5Update(double &x, float *y, double x1, float &h, float h_min,
	       float *param, DataStruct data_struct)
{
  bool end_time_step=false;
  while(!end_time_step) {
    float hmax=(float)(x1-x);
    RK5Step<NVAR, NPARAM, DataStruct>(x, y, h, h_min, hmax, param,
				       data_struct);
    end_time_step = (x >= x1-h_min);
    ExternalUpdate<NVAR, NPARAM>(x, y, param, end_time_step, data_struct);
  }
}

template<int NVAR, int NPARAM, class DataStruct>
__global__
void ArrayUpdate(int array_size, double *x_arr, float *h_arr, float *y_arr,
		 float *par_arr, double x1, float h_min, DataStruct data_struct)
{
  int ArrayIdx = threadIdx.x + blockIdx.x * blockDim.x;
  if (ArrayIdx<array_size) {
    double x = x_arr[ArrayIdx];
    float h = h_arr[ArrayIdx];
    float y[NVAR];
    float param[NPARAM];

    for(int i=0; i<NVAR; i++) {
      y[i] = y_arr[ArrayIdx*NVAR + i];
    }
    for(int j=0; j<NPARAM; j++) {
      param[j] = par_arr[ArrayIdx*NPARAM + j];
    }

    RK5Update<NVAR, NPARAM, DataStruct>(x, y, x1, h, h_min, param,
					 data_struct);

    x_arr[ArrayIdx] = x;
    h_arr[ArrayIdx] = h;
    for(int i=0; i<NVAR; i++) {
      y_arr[ArrayIdx*NVAR + i] = y[i];
    }
    for(int j=0; j<NPARAM; j++) {
      par_arr[ArrayIdx*NPARAM + j] = param[j]; 
    }
  }
}

template<class DataStruct>
class RungeKutta5
{
  int array_size_;
  int n_var_;
  int n_param_;
    
  double *d_XArr;
  float *d_HArr;
  float *d_YArr;
  float *d_ParamArr;

  public:

  ~RungeKutta5();
 
  double *GetXArr() {return d_XArr;}
  float *GetHArr() {return d_HArr;}
  float *GetYArr() {return d_YArr;}
  float *GetParamArr() {return d_ParamArr;}
  int Init(int array_size, int n_var, int n_param, double x_min, float h,
	   DataStruct data_struct);
  int Calibrate(double x_min, float h, DataStruct data_struct);

  int Free();

  int GetX(int i_array, int n_elem, double *x);
  int GetY(int i_var, int i_array, int n_elem, float *y);
  int SetParam(int i_param, int i_array, int n_param, int n_elem, float val);
  int SetVectParam(int i_param, int i_array, int n_param, int n_elem,
		    float *param, int vect_size);
  template<int NVAR, int NPARAM> int Update(double x1, float h_min,
					     DataStruct data_struct);

};


template<class DataStruct>
template<int NVAR, int NPARAM>
  int RungeKutta5<DataStruct>::Update(double x1, float h_min,
				      DataStruct data_struct)
{
  ArrayUpdate<NVAR, NPARAM, DataStruct><<<(array_size_+1023)/1024, 1024>>>
    (array_size_, d_XArr, d_HArr, d_YArr, d_ParamArr, x1, h_min, data_struct);
  //gpuErrchk( cudaPeekAtLastError() );
  //gpuErrchk( cudaDeviceSynchronize() );

  return 0;
}

template<class DataStruct>
RungeKutta5<DataStruct>::~RungeKutta5()
{
  Free();
}

template<class DataStruct>
int RungeKutta5<DataStruct>::Free()
{
  cudaFree(d_XArr);
  cudaFree(d_HArr);
  cudaFree(d_YArr);
  cudaFree(d_ParamArr);

  return 0;
}

template<class DataStruct>
int RungeKutta5<DataStruct>::Init(int array_size, int n_var, int n_param,
				  double x_min, float h,
				  DataStruct data_struct)
{
  array_size_ = array_size;
  n_var_ = n_var;
  n_param_ = n_param; 

  CUDAMALLOCCTRL("&d_XArr",&d_XArr, array_size_*sizeof(double));
  CUDAMALLOCCTRL("&d_HArr",&d_HArr, array_size_*sizeof(float));
  CUDAMALLOCCTRL("&d_YArr",&d_YArr, array_size_*n_var_*sizeof(float));
  CUDAMALLOCCTRL("&d_ParamArr",&d_ParamArr, array_size_*n_param_*sizeof(float));

  ArrayInit<DataStruct><<<(array_size+1023)/1024, 1024>>>
    (array_size_, n_var, n_param, d_XArr, d_HArr, d_YArr, d_ParamArr,
     x_min, h, data_struct);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  
  return 0;
}

template<class DataStruct>
int RungeKutta5<DataStruct>::Calibrate(double x_min, float h,
				       DataStruct data_struct)
{
  ArrayCalibrate<DataStruct><<<(array_size_+1023)/1024, 1024>>>
    (array_size_, n_var_, n_param_, d_XArr, d_HArr, d_YArr, d_ParamArr,
     x_min, h, data_struct);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  return 0;
}

template<class DataStruct>
int RungeKutta5<DataStruct>::GetX(int i_array, int n_elem, double *x)
{
  cudaMemcpy(x, &d_XArr[i_array], n_elem*sizeof(double),
	     cudaMemcpyDeviceToHost);

  return 0;
}

template<class DataStruct>
int RungeKutta5<DataStruct>::GetY(int i_var, int i_array, int n_elem, float *y)
{
  cudaMemcpy(y, &d_YArr[i_array*n_var_ + i_var], n_elem*sizeof(float),
	     cudaMemcpyDeviceToHost);

  return 0;
}

template<class DataStruct>
int RungeKutta5<DataStruct>::SetParam(int i_param, int i_array, int n_param,
			   int n_elem, float val)
{
  SetFloatArray<<<(n_elem+1023)/1024, 1024>>>
    (&d_ParamArr[i_array*n_param_ + i_param], n_elem, n_param, val);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  
  return 0;
}

#endif
