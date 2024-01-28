/*
 *  user_m2_psc_delta_kernel.h
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

#ifndef USERM2PSCDELTAKERNEL_H
#define USERM2PSCDELTAKERNEL_H

#include "node_group.h"
#include "spike_buffer.h"
#include "user_m2.h"
#include <cmath>
#include <string>

#define MIN( a, b ) ( ( ( a ) < ( b ) ) ? ( a ) : ( b ) )

extern __constant__ float NESTGPUTimeResolution;

namespace user_m2_ns
{
enum ScalVarIndexes
{
  i_V_m = 0,
  i_w,
  N_SCAL_VAR
};

enum PortVarIndexes
{
  N_PORT_VAR = 0
};

enum ScalParamIndexes
{
  i_V_th = 0,
  i_Delta_T,
  i_g_L,
  i_E_L,
  i_C_m,
  i_a,
  i_b,
  i_tau_w,
  i_I_e,
  i_V_peak,
  i_V_reset,
  i_t_ref,
  i_refractory_step,
  i_den_delay,
  N_SCAL_PARAM
};

enum GroupParamIndexes
{
  i_h_min_rel = 0, // Min. step in ODE integr. relative to time resolution
  i_h0_rel,        // Starting step in ODE integr. relative to time resolution
  N_GROUP_PARAM
};

const std::string user_m2_scal_var_name[ N_SCAL_VAR ] = { "V_m", "w" };

const std::string user_m2_scal_param_name[ N_SCAL_PARAM ] = { "V_th",
  "Delta_T",
  "g_L",
  "E_L",
  "C_m",
  "a",
  "b",
  "tau_w",
  "I_e",
  "V_peak",
  "V_reset",
  "t_ref",
  "refractory_step",
  "den_delay" };

const std::string user_m2_group_param_name[ N_GROUP_PARAM ] = { "h_min_rel", "h0_rel" };

//
// I know that defines are "bad", but the defines below make the
// following equations much more readable.
// For every rule there is some exceptions!
//
#define V_m y[ i_V_m ]
#define w y[ i_w ]

#define dVdt dydx[ i_V_m ]
#define dwdt dydx[ i_w ]

#define V_th param[ i_V_th ]
#define Delta_T param[ i_Delta_T ]
#define g_L param[ i_g_L ]
#define E_L param[ i_E_L ]
#define C_m param[ i_C_m ]
#define a param[ i_a ]
#define b param[ i_b ]
#define tau_w param[ i_tau_w ]
#define I_e param[ i_I_e ]
#define V_peak param[ i_V_peak ]
#define V_reset param[ i_V_reset ]
#define t_ref param[ i_t_ref ]
#define refractory_step param[ i_refractory_step ]
#define den_delay param[ i_den_delay ]

#define h_min_rel_ group_param_[ i_h_min_rel ]
#define h0_rel_ group_param_[ i_h0_rel ]

template < int NVAR, int NPARAM > //, class DataStruct>
__device__ void
Derivatives( double x, float* y, float* dydx, float* param, user_m2_rk5 data_struct )
{

  float V = ( refractory_step > 0 ) ? V_reset : MIN( V_m, V_peak );

  float V_spike = Delta_T == 0. ? 0. : Delta_T * exp( ( V - V_th ) / Delta_T );

  dVdt = ( refractory_step > 0 ) ? 0 : ( -g_L * ( V - E_L - V_spike ) - w + I_e ) / C_m;
  // Adaptation current w.
  dwdt = ( a * ( V - E_L ) - w ) / tau_w;
}

template < int NVAR, int NPARAM > //, class DataStruct>
__device__ void
ExternalUpdate( double x, float* y, float* param, bool end_time_step, user_m2_rk5 data_struct )
{
  if ( V_m < -1.0e3 )
  { // numerical instability
    printf( "V_m out of lower bound\n" );
    V_m = V_reset;
    w = 0;
    return;
  }
  if ( w < -1.0e6 || w > 1.0e6 )
  { // numerical instability
    printf( "w out of bound\n" );
    V_m = V_reset;
    w = 0;
    return;
  }
  if ( refractory_step > 0.0 )
  {
    V_m = V_reset;
    if ( end_time_step )
    {
      refractory_step -= 1.0;
    }
  }
  else
  {
    if ( V_m >= V_peak )
    { // send spike
      int neuron_idx = threadIdx.x + blockIdx.x * blockDim.x;
      PushSpike( data_struct.i_node_0_ + neuron_idx, 1.0 );
      V_m = V_reset;
      w += b; // spike-driven adaptation
      refractory_step = ( int ) round( t_ref / NESTGPUTimeResolution );
      if ( refractory_step < 0 )
      {
        refractory_step = 0;
      }
    }
  }
}

}; // namespace user_m2_ns

template < int NVAR, int NPARAM >
__device__ void
Derivatives( double x, float* y, float* dydx, float* param, user_m2_rk5 data_struct )
{
  user_m2_ns::Derivatives< NVAR, NPARAM >( x, y, dydx, param, data_struct );
}

template < int NVAR, int NPARAM >
__device__ void
ExternalUpdate( double x, float* y, float* param, bool end_time_step, user_m2_rk5 data_struct )
{
  user_m2_ns::ExternalUpdate< NVAR, NPARAM >( x, y, param, end_time_step, data_struct );
}

#endif
