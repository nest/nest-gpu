Implement new neuron models
===========================

NEST GPU supports the definition of new models through three possible
approaches, two that do not require any experience with programming
languages, the other more advanced, which allows to use an arbitrary
number of new models in the same simulation and greater flexibility in
the model definition.

Basic approach with 5-th order Runge Kutta
------------------------------------------

The 5-th order Runge Kutta method gives more accurate solution of the
differential equations underling the neuron dynamics compared to the
Euler forward method, however it is slower. The user has to choose the
synapse type among one of those already available in the code: either
current based with exponential, alpha or delta function, or conductance
based with alpha or beta function. The corresponding template files are
user_m1_psc_exp\ *, user_m1_psc_alpha*, etc. To use them, the user has
to overwrite the files user_m1.cu user_m1.h user_m1_kernel.h
user_m1_rk5.h. For instance:

.. code-block:: bash

   cp user_m1_cond_beta.cu user_m1.cu
   cp user_m1_cond_beta.h user_m1.h
   cp user_m1_cond_beta_kernel.h user_m1_kernel.h
   cp user_m1_cond_beta_rk5.h user_m1_rk5.h

Those files must be opened and modified appropriately using a text
editor. In the file user_m1_kernel.h, in the lines:

.. code-block:: cpp

   enum ScalVarIndexes {
     i_V_m = 0,
     i_w,
     N_SCAL_VAR
   };
   ...
   enum ScalParamIndexes {
     i_V_th = 0,
     i_Delta_T,
    ....
     i_t_ref,
     i_refractory_step,
     i_den_delay,
     N_SCAL_PARAM
   };
   ...
   const std::string user_m1_scal_var_name[N_SCAL_VAR] = {
     "V_m",
     "w"
   };
   ...
   const std::string user_m1_scal_param_name[N_SCAL_PARAM] = {
     "V_th",
     "Delta_T",
     ....
     "t_ref",
     "refractory_step",
     "den_delay"
   };
   ...
   #define V_m y[i_V_m]
   #define w y[i_w]
   ...
   #define dVdt dydx[i_V_m]
   #define dwdt dydx[i_w]
   ...
   #define V_th param[i_V_th]
   #define Delta_T param[i_Delta_T]
   ...
   #define t_ref param[i_t_ref]
   #define refractory_step param[i_refractory_step]
   #define den_delay param[i_den_delay]
   ...

V_m and w should be replaced with the names of the state variables that
describe the neuron dynamics, and V_th, Delta_t, etc. should be replaced
with the names of the parameters of the model. It is advised not to
modify the parameters t_ref, refractory_step and den_delay, unless the
user knows what he is doing. In the lines:

.. code-block:: cpp

     dVdt = ( refractory_step > 0 ) ? 0 :
       ( -g_L*(V - E_L - V_spike) + I_syn - w + I_e) / C_m;
     // Adaptation current w.
     dwdt = (a*(V - E_L) - w) / tau_w;

should be replaced with the differential equations of the state
variables, i.e. the derivatives of the state variables. The lines:

.. code-block:: cpp

     if ( V_m < -1.0e3) { // numerical instability
   ...
     if ( w < -1.0e6 || w > 1.0e6) { // numerical instability

should be replaced with proper limits for the state variables. In the
file user_m1.cu, the lines

.. code-block:: cpp

     V_th = -50.4;
     Delta_T = 2.0;
     g_L = 30.0;
   ...

should be replaced with proper initial values of the parameters and
state variables.

After those modifications, the code must be recompiled and reinstalled
with the commands ``make`` and ``make install``

Basic approach with Euler forward method
----------------------------------------

This integration method is faster than the Runge-Kutta but it is less
accurate. The corresponding template files are user_m1_psc_exp\ *,
user_m1_psc_alpha*, etc. To use them, the user has to overwrite the
files user_m1.cu user_m1.h user_m1_kernel.h user_m1_rk5.h. For instance:

.. code-block:: bash

   cp user_m1_iaf_psc_exp.cu user_m1.cu
   cp user_m1_iaf_psc_exp.h user_m1.h
   cp user_m1_iaf_psc_exp_kernel.h user_m1_kernel.h
   cp user_m1_iaf_psc_exp_rk5.h user_m1_rk5.h

Those files must be opened and modified appropriately using a text
editor. In the file user_m1.h, in the lines:

.. code-block:: cpp

   enum ScalVarIndexes {
     i_I_syn_ex = 0,        // postsynaptic current for exc. inputs
     i_I_syn_in,            // postsynaptic current for inh. inputs
     i_V_m_rel,                 // membrane potential
     i_refractory_step,     // refractory step counter
     N_SCAL_VAR
   };

   enum ScalParamIndexes {
     i_tau_m = 0,       // Membrane time constant in ms
     i_C_m,             // Membrane capacitance in pF
   ...
     N_SCAL_PARAM
   };

    
   const std::string user_m1_scal_var_name[N_SCAL_VAR] = {
     "I_syn_ex",
     "I_syn_in",
     "V_m_rel",
     "refractory_step"
   };


   const std::string user_m1_scal_param_name[N_SCAL_PARAM] = {
     "tau_m",
     "C_m",
   ...
   };

I_syn_ex, I_syn_in and V_m_rel should be replaced with the names of the
state variables that describe the neuron and synaptic current dynamics,
and tau_m, C_m, etc. should be replaced with the names of the parameters
of the model. It is advised not to modify the variable refractory step
and the parameters t_ref and den_delay, unless the user knows what he is
doing. The same replacements should be done in the file user_m1.cu, in
the lines:

.. code-block:: cpp

   #define I_syn_ex var[i_I_syn_ex]
   #define I_syn_in var[i_I_syn_in]
   ...

In the file user_m1.cu, the lines

.. code-block:: cpp

      else { // neuron is not refractory, so evolve V
         V_m_rel = V_m_rel * P22 + I_syn_ex * P21ex + I_syn_in * P21in + I_e * P20;
       }
       // exponential decaying PSCs
       I_syn_ex *= P11ex;
       I_syn_in *= P11in;

should be replaced with a proper update formula for the state variables.

After those modifications, the code must be recompiled and reinstalled
with the commands ``make`` and ``make install``

Advanced approach
-----------------

With this approach it is possible to define an arbitrary number of
models. The user can start either from the files user_m1\* or from the
files defining one of the models already included in NEST GPU, edit the
files, replace the model name (user_m1 or the name of the model used as
a starting point) with a new name and modify the state variables, the
parameters and the equations of the dynamics according to the new model.

In the file neuron_models.h, in the lines

.. code-block:: cpp

   enum NeuronModels {
     i_null_model = 0, i_iaf_psc_exp_g_model,
     i_iaf_psc_exp_hc_model, i_iaf_psc_exp_model,
   ...

and

.. code-block:: cpp

   const std::string neuron_model_name[N_NEURON_MODELS] = {
     "", "iaf_psc_exp_g", "iaf_psc_exp_hc", "iaf_psc_exp", "ext_neuron",
   ...

the user should add the name of the new model. In the file
neuron_models.cu, after the line:

.. code-block:: cpp

   #include "user_m2.h"

the user should include the header of the new model. In the body of the
function

.. code-block:: cpp

   NodeSeq NESTGPU::Create(std::string model_name, int n_node /*=1*/,
                             int n_port /*=1*/)

the user should add a new block, as:

.. code-block:: cpp

     else if (model_name == neuron_model_name[i_my_model]) {
       my_model *my_model_group = new my_model;
       node_vect_.push_back(my_model_group);
     }

where my_model should be replaced by the model name. In the file
Makefile.am, after the lines

.. code-block:: cpp

   $(top_srcdir)/src/user_m2_kernel.h \
   $(top_srcdir)/src/user_m2_rk5.h

add the header files of your new model. After the lines

.. code-block:: cpp

   $(top_srcdir)/src/user_m1.cu \
   $(top_srcdir)/src/user_m2.cu

add the .cu files of your new model.

After those modifications, from the main directory, run

.. code-block:: bash

   autoreconf -i

then the code must be recompiled and reinstalled following the
instructions for compiling from source.
