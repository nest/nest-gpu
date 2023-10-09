# -*- coding: utf-8 -*-
#
# hpc_benchmark.py
#
# This file is part of NEST GPU.
#
# Copyright (C) 2021 The NEST Initiative
#
# NEST GPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST GPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST GPU.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import os
import sys
from time import perf_counter_ns
import scipy.special as sp
from mpi4py import MPI

import nestgpu as ngpu
#import nest
#import nest.raster_plot

M_INFO = 10
M_ERROR = 30

ngpu.ConnectMpiInit()
mpi_np = ngpu.MpiNp()

print("Simulation with {} MPI processes".format(mpi_np))

mpi_id = ngpu.MpiId()

comm = MPI.COMM_WORLD


###############################################################################
# Parameter section
# Define all relevant parameters: changes should be made here


params = {
    #'nvp': 1,               # total number of virtual processes
    'scale': 1.,            # scaling factor of the network size
                            # total network size = scale*11250 neurons
    'simtime': 250.,        # total simulation time in ms
    'presimtime': 50.,      # simulation time until reaching equilibrium
    'dt': 0.1,              # simulation step
    'record_spikes': True,  # switch to record spikes of excitatory
                            # neurons to file
    'path_name': '.',       # path where all files will have to be written
    'log_file': 'log',      # naming scheme for the log files
}

def lambertwm1(x):
    """Wrapper for LambertWm1 function"""
    # Using scipy to mimic the gsl_sf_lambert_Wm1 function.
    return sp.lambertw(x, k=-1 if x < 0 else 0).real

def convert_synapse_weight(tau_m, tau_syn, C_m):
    """
    Computes conversion factor for synapse weight from mV to pA
    This function is specific to the leaky integrate-and-fire neuron
    model with alpha-shaped postsynaptic currents.
    """

    # compute time to maximum of V_m after spike input
    # to neuron at rest
    a = tau_m / tau_syn
    b = 1.0 / tau_syn - 1.0 / tau_m
    t_rise = 1.0 / b * (-lambertwm1(-np.exp(-1.0 / a) / a).real - 1.0 / a)

    v_max = np.exp(1.0) / (tau_syn * C_m * b) * (
        (np.exp(-t_rise / tau_m) - np.exp(-t_rise / tau_syn)) /
        b - t_rise * np.exp(-t_rise / tau_syn))
    return 1. / v_max

tau_syn = 0.32582722403722841

brunel_params = {
    'NE': int(9000 * params['scale']),  # number of excitatory neurons
    'NI': int(2250 * params['scale']),  # number of inhibitory neurons
    
    # Todo change parameter names!!!
    'model_params': {  # Set variables for iaf_psc_alpha
        'E_L': 0.0,  # Resting membrane potential(mV)
        'C_m': 250.0,  # Capacity of the membrane(pF)
        'tau_m': 10.0,  # Membrane time constant(ms)
        't_ref': 0.5,  # Duration of refractory period(ms)
        'Theta_rel': 20.0,  # Threshold(mV)
        'V_reset_rel': 0.0,  # Reset Potential(mV)
        # time const. postsynaptic excitatory currents(ms)
        'tau_syn_ex': tau_syn,
        # time const. postsynaptic inhibitory currents(ms)
        'tau_syn_in': tau_syn,
        #'tau_minus': 30.0,  # time constant for STDP(depression)
        # V can be randomly initialized see below
        'V_m_rel': 5.7  # mean value of membrane potential
    },
    'randomize_Vm': True,
    'mean_potential': 5.7,
    'sigma_potential': 7.2,

    'delay': 1.5,  # synaptic delay, all alphaonnections(ms)

    # synaptic weight
    'JE': 0.14,  # peak of EPSP

    'sigma_w': 3.47,  # standard dev. of E->E synapses(pA)
    'g': -5.0,

    'stdp_params': {
        'alpha': 0.0513,
        'lambda': 0.1,  # STDP step size
        'mu_plus': 0.4,  # STDP weight dependence exponent(potentiation)
        # added
        'mu_minus': 0.4,  # STDP weight dependence exponent(depression)
        'tau_plus': 15.0,  # time constant for potentiation
        # added
        'tau_minus': 15.0,  # time constant for depression
    },
    'stdp_delay': 1.5,

    'eta': 1.685,  # scaling of external stimulus
    'filestem': params['path_name']
}

def compute_rate(spike_times):
    """Compute local approximation of average firing rate
    This approximation is based on the number of local nodes, number
    of local spikes and total time. Since this also considers devices,
    the actual firing rate is usually underestimated.
    """

    #n_local_spikes = sr.n_events
    n_local_neurons = brunel_params['NE'] + brunel_params['NI']

    # discard spikes during presimtime
    n_spikes = sum(sum(i>params['presimtime'] for i in j) for j in spike_times)

    simtime = params['simtime']
    return (1. * n_spikes / (n_local_neurons * simtime) * 1e3)


# unpack a few variables for convenience
NE = brunel_params['NE']
NI = brunel_params['NI']
model_params = brunel_params['model_params']
stdp_params = brunel_params['stdp_params']

ngpu.SetKernelStatus({"time_resolution": params['dt']})

print('Creating neuron populations.')

neuron = []; E_pops = []; I_pops = []

for i in range(mpi_np):
    neuron.append(ngpu.RemoteCreate(i, 'iaf_psc_alpha', NE+NI, 1, model_params).node_seq)
    E_pops.append(neuron[i][0:NE])
    I_pops.append(neuron[i][NE:NE+NI])

if brunel_params['randomize_Vm']:
    print('Randomizing membrane potentials.')
    #ngpu.SetStatus(E_pops[mpi_id], {"V_m_rel": {"distribution": "normal", "mu": brunel_params['mean_potential'], "sigma": brunel_params['sigma_potential']}})
    #ngpu.SetStatus(I_pops[mpi_id], {"V_m_rel": {"distribution": "normal", "mu": brunel_params['mean_potential'], "sigma": brunel_params['sigma_potential']}})
    ngpu.SetStatus(neuron[mpi_id], {"V_m_rel": {"distribution": "normal", "mu": brunel_params['mean_potential'], "sigma": brunel_params['sigma_potential']}})


# total number of incoming excitatory connections
CE = int(1. * NE / params['scale'])
# total number of incomining inhibitory connections
CI = int(1. * NI / params['scale'])

# number of indegrees from each MPI process
# here the indegrees are equally distributed among the
# neuron populations in all the MPI processes

CE_distrib = int(1.0 * CE / mpi_np)
CI_distrib = int(1.0 * CI / mpi_np)

print('Creating excitatory stimulus generator.')

# Convert synapse weight from mV to pA
conversion_factor = convert_synapse_weight(model_params['tau_m'], model_params['tau_syn_ex'], model_params['C_m'])
JE_pA = conversion_factor * brunel_params['JE']

nu_thresh = model_params['Theta_rel'] / ( CE * model_params['tau_m'] / model_params['C_m'] * JE_pA * np.exp(1.) * tau_syn)
nu_ext = nu_thresh * brunel_params['eta']

E_stim= ngpu.Create('poisson_generator', 1, 1, {'rate': nu_ext * CE * 1000.})

print('Creating excitatory spike recorder.')

if params['record_spikes']:
    recorder_label = os.path.join(brunel_params['filestem'], 'alpha_' + str(stdp_params['alpha']) + '_spikes')
    ngpu.ActivateRecSpikeTimes(neuron[mpi_id], 1000)

syn_dict_ex = {'weight': JE_pA, 'delay': brunel_params['delay']}
syn_dict_in = {'weight': brunel_params['g'] * JE_pA, 'delay': brunel_params['delay']}

syn_group_stdp = ngpu.CreateSynGroup('stdp', stdp_params)
syn_dict_stdp = {"weight": JE_pA, "delay": brunel_params['stdp_delay'], "synapse_group": syn_group_stdp}

print('Connecting stimulus generators.')

# Connect Poisson generator to neuron
ngpu.Connect(E_stim, E_pops[mpi_id], {'rule': 'all_to_all'}, syn_dict_ex)
ngpu.Connect(E_stim, I_pops[mpi_id], {'rule': 'all_to_all'}, syn_dict_ex)

print("Inside process {}".format(mpi_id))


print('Creating local connections.')
print('Connecting excitatory -> excitatory population.')

ngpu.Connect(E_pops[mpi_id], E_pops[mpi_id],
                {'rule': 'fixed_indegree', 'indegree': CE_distrib}, syn_dict_ex)
                #'allow_autapses': False, 'allow_multapses': True},
                #syn_dict_ex)
                #syn_dict_stdp)

print('Connecting inhibitory -> excitatory population.')

ngpu.Connect(I_pops[mpi_id], E_pops[mpi_id],
                {'rule': 'fixed_indegree', 'indegree': CI_distrib},
                #'allow_autapses': False, 'allow_multapses': True},
                syn_dict_in)

print('Connecting excitatory -> inhibitory population.')

ngpu.Connect(E_pops[mpi_id], I_pops[mpi_id],
                {'rule': 'fixed_indegree', 'indegree': CE_distrib},
                #'allow_autapses': False, 'allow_multapses': True},
                syn_dict_ex)

print('Connecting inhibitory -> inhibitory population.')

ngpu.Connect(I_pops[mpi_id], I_pops[mpi_id],
                {'rule': 'fixed_indegree', 'indegree': CI_distrib},
                #'allow_autapses': False, 'allow_multapses': True},
                syn_dict_in)


print('Creating local and remote connections.')
for i in range(mpi_np):
    for j in range(mpi_np):
        if(i!=j):

            print('Connecting excitatory {} -> excitatory {} population. mpi_id = {}'.format(i, j, mpi_id))

            ngpu.RemoteConnect(i, E_pops[i], j, E_pops[j],
                        {'rule': 'fixed_indegree', 'indegree': CE_distrib},
                        #'allow_autapses': False, 'allow_multapses': True},
                        syn_dict_ex)
                        #syn_dict_stdp)

            
            comm.Barrier()

            print('Connecting inhibitory {} -> excitatory {} population. mpi_id = {}'.format(i, j, mpi_id))

            ngpu.RemoteConnect(i, I_pops[i], j, E_pops[j],
                        {'rule': 'fixed_indegree', 'indegree': CI_distrib},
                        #'allow_autapses': False, 'allow_multapses': True},
                        syn_dict_in)

            comm.Barrier()
            print('Connecting excitatory {} -> inhibitory {} population. mpi_id = {}'.format(i, j, mpi_id))

            ngpu.RemoteConnect(i, E_pops[i], j, I_pops[j],
                        {'rule': 'fixed_indegree', 'indegree': CE_distrib},
                        #'allow_autapses': False, 'allow_multapses': True},
                        syn_dict_ex)

            comm.Barrier()
            print('Connecting inhibitory {} -> inhibitory {} population. mpi_id = {}'.format(i, j, mpi_id))

            ngpu.RemoteConnect(i, I_pops[i], j, I_pops[j],
                        {'rule': 'fixed_indegree', 'indegree': CI_distrib},
                        #'allow_autapses': False, 'allow_multapses': True},
                        syn_dict_in)
            
            comm.Barrier()
            

ngpu.Calibrate()

ngpu.Simulate(params['presimtime'])

ngpu.Simulate(params['simtime'])

if params['record_spikes']:
    spike_times = ngpu.GetRecSpikeTimes(neuron[mpi_id])
    rate = compute_rate(spike_times)






