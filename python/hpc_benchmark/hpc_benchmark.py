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


"""
Random balanced network HPC benchmark
-------------------------------------
This script produces a balanced random network of `scale*11250` neurons in
which the excitatory-excitatory neurons exhibit STDP with
multiplicative depression and power-law potentiation. A mutual
equilibrium is obtained between the activity dynamics (low rate in
asynchronous irregular regime) and the synaptic weight distribution
(unimodal). The number of incoming connections per neuron is fixed
and independent of network size (indegree=11250).
This is the standard network investigated in [1]_, [2]_, [3]_.
A note on connectivity
~~~~~~~~~~~~~~~~~~~~~~
.. image:: ../examples/hpc_benchmark_connectivity.svg
   :width: 50 %
   :alt: HPC Benchmark network architecture
   :align: right
Each neuron receives :math:`K_{in,{\\tau} E}` excitatory connections randomly
drawn from population E and :math:`K_{in,\\tau I}` inhibitory connections from
population I. Autapses are prohibited (denoted by the crossed out A next to
the connections) while multapses are allowed (denoted by the M). Each neuron
receives additional input from an external stimulation device. All delays are
constant, all weights but excitatory onto excitatory are constant. Excitatory
onto excitatory weights are time dependent. Figure taken from [4]_.
A note on scaling
~~~~~~~~~~~~~~~~~
This benchmark was originally developed for very large-scale simulations on
supercomputers with more than 1 million neurons in the network and
11.250 incoming synapses per neuron. For such large networks, synaptic input
to a single neuron will be little correlated across inputs and network
activity will remain stable over long periods of time.
The original network size corresponds to a scale parameter of 100 or more.
In order to make it possible to test this benchmark script on desktop
computers, the scale parameter is set to 1 below, while the number of
11.250 incoming synapses per neuron is retained. In this limit, correlations
in input to neurons are large and will lead to increasing synaptic weights.
Over time, network dynamics will therefore become unstable and all neurons
in the network will fire in synchrony, leading to extremely slow simulation
speeds.
Therefore, the presimulation time is reduced to 50 ms below and the
simulation time to 250 ms, while we usually use 100 ms presimulation and
1000 ms simulation time.
For meaningful use of this benchmark, you should use a scale > 10 and check
that the firing rate reported at the end of the benchmark is below 10 spikes
per second.
References
~~~~~~~~~~
.. [1] Morrison A, Aertsen A, Diesmann M (2007). Spike-timing-dependent
       plasticity in balanced random networks. Neural Comput 19(6):1437-67
.. [2] Helias et al (2012). Supercomputers ready for use as discovery machines
       for neuroscience. Front. Neuroinform. 6:26
.. [3] Kunkel et al (2014). Spiking network simulation code for petascale
       computers. Front. Neuroinform. 8:78
.. [4] Senk et al (2021). Connectivity Concepts in Neuronal Network Modeling.
       arXiv. 2110.02883
"""

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

mpI_popsd = ngpu.MpiId()

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

###############################################################################
# For compatibility with earlier benchmarks, we require a rise time of
# ``t_rise = 1.700759 ms`` and we choose ``tau_syn`` to achieve this for given
# ``tau_m``. This requires numerical inversion of the expression for ``t_rise``
# in ``convert_synapse_weight``. We computed this value once and hard-code
# it here.


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

    ####################################################################
    # Note that Kunkel et al. (2014) report different values. The values
    # in the paper were used for the benchmarks on K, the values given
    # here were used for the benchmark on JUQUEEN.

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

###############################################################################
# Function Section


def build_network(logger):
    """Builds the network including setting of simulation and neuron
    parameters, creation of neurons and connections
    Requires an instance of Logger as argument
    """

    tic = perf_counter_ns()  # start timer on construction

    # unpack a few variables for convenience
    NE = brunel_params['NE']
    NI = brunel_params['NI']
    model_params = brunel_params['model_params']
    stdp_params = brunel_params['stdp_params']

    ngpu.SetKernelStatus({"time_resolution": params['dt']})

    print('Creating neuron populations.')

    #E_pops = np.empty(mpi_np, dtype=ngpu.NodeSeq)
    #I_pops = np.empty(mpi_np, dtype=ngpu.NodeSeq)
    #Egids = np.empty(mpi_np, dtype=tuple)
    #Igids = np.empty(mpi_np, dtype=tuple)

    #E_pops[mpI_popsd] = ngpu.RemoteCreate(mpI_popsd, 'iaf_psc_alpha', NE, 1, model_params)
    #I_pops[mpI_popsd] = ngpu.RemoteCreate(mpI_popsd, 'iaf_psc_alpha', NI, 1, model_params)

    #for i in range(mpi_np):
    #    E_pops[i] = ngpu.RemoteCreate(i, 'iaf_psc_alpha', NE, 1, model_params).node_seq
    #    I_pops[i] = ngpu.RemoteCreate(i, 'iaf_psc_alpha', NI, 1, model_params).node_seq

    E_pops = ngpu.RemoteCreate(mpI_popsd, 'iaf_psc_alpha', NE, 1, model_params).node_seq
    I_pops = ngpu.RemoteCreate(mpI_popsd, 'iaf_psc_alpha', NI, 1, model_params).node_seq


    if brunel_params['randomize_Vm']:
        print('Randomizing membrane potentials.')
        ngpu.SetStatus(E_pops, {"V_m_rel": {"distribution": "normal", "mu": brunel_params['mean_potential'], "sigma": brunel_params['sigma_potential']}})
        ngpu.SetStatus(I_pops, {"V_m_rel": {"distribution": "normal", "mu": brunel_params['mean_potential'], "sigma": brunel_params['sigma_potential']}})

    #Egids[mpI_popsd] = (E_pops[mpI_popsd].node_seq[0], E_pops[mpI_popsd].node_seq[-1])
    #Igids[mpI_popsd] = (I_pops[mpI_popsd].node_seq[0], I_pops[mpI_popsd].node_seq[-1])

    # local neurons
    #E_neurons = E_pops[mpI_popsd].node_seq
    #I_neurons = I_pops[mpI_popsd].node_seq

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
        ngpu.ActivateRecSpikeTimes(E_pops, 1000)

    BuildNodeTime = perf_counter_ns() - tic

    logger.log(str(BuildNodeTime) + ' # build_time_nodes')
    #logger.log(str(memory_thisjob()) + ' # virt_mem_after_nodes')

    tic = perf_counter_ns()

    syn_dict_ex = {'weight': JE_pA, 'delay': brunel_params['delay']}
    syn_dict_in = {'weight': brunel_params['g'] * JE_pA, 'delay': brunel_params['delay']}
    
    syn_group_stdp = ngpu.CreateSynGroup('stdp', stdp_params)
    syn_dict_stdp = {"weight": JE_pA, "delay": brunel_params['stdp_delay'], "synapse_group": syn_group_stdp}

    print('Connecting stimulus generators.')

    # Connect Poisson generator to neuron
    ngpu.Connect(E_stim, E_pops, {'rule': 'all_to_all'}, syn_dict_ex)
    ngpu.Connect(E_stim, I_pops, {'rule': 'all_to_all'}, syn_dict_ex)

    """
    print('Creating local connections.')
    print('Connecting excitatory -> excitatory population.')

    ngpu.Connect(E_pops[mpI_popsd].node_seq, E_pops[mpI_popsd].node_seq,
                 {'rule': 'fixed_indegree', 'indegree': CE_distrib}, syn_dict_ex)
                  #'allow_autapses': False, 'allow_multapses': True},
                 #syn_dict_ex)
                 #syn_dict_stdp)

    print('Connecting inhibitory -> excitatory population.')

    ngpu.Connect(I_pops[mpI_popsd].node_seq, E_pops[mpI_popsd].node_seq,
                 {'rule': 'fixed_indegree', 'indegree': CI_distrib},
                  #'allow_autapses': False, 'allow_multapses': True},
                 syn_dict_in)

    print('Connecting excitatory -> inhibitory population.')

    ngpu.Connect(E_pops[mpI_popsd].node_seq, I_pops[mpI_popsd].node_seq,
                 {'rule': 'fixed_indegree', 'indegree': CE_distrib},
                  #'allow_autapses': False, 'allow_multapses': True},
                 syn_dict_ex)

    print('Connecting inhibitory -> inhibitory population.')

    ngpu.Connect(I_pops[mpI_popsd].node_seq, I_pops[mpI_popsd].node_seq,
                 {'rule': 'fixed_indegree', 'indegree': CI_distrib},
                  #'allow_autapses': False, 'allow_multapses': True},
                 syn_dict_in)
    """

    print('Creating local and remote connections.')
    for i in range(mpi_np):
        for j in range(mpi_np):

            print('Connecting excitatory {} -> excitatory {} population.'.format(i, j))

            ngpu.RemoteConnect(i, E_pops, j, E_pops,
                        {'rule': 'fixed_indegree', 'indegree': CE_distrib},
                        #'allow_autapses': False, 'allow_multapses': True},
                        syn_dict_ex)
                        #syn_dict_stdp)

            comm.Barrier()

            print('Connecting inhibitory {} -> excitatory {} population.'.format(i, j))

            ngpu.RemoteConnect(i, I_pops, j, E_pops,
                        {'rule': 'fixed_indegree', 'indegree': CI_distrib},
                        #'allow_autapses': False, 'allow_multapses': True},
                        syn_dict_in)

            comm.Barrier()
            print('Connecting excitatory {} -> inhibitory {} population.'.format(i, j))

            ngpu.RemoteConnect(i, E_pops, j, I_pops,
                        {'rule': 'fixed_indegree', 'indegree': CE_distrib},
                        #'allow_autapses': False, 'allow_multapses': True},
                        syn_dict_ex)

            comm.Barrier()
            print('Connecting inhibitory {} -> inhibitory {} population.'.format(mpI_popsd, j))

            ngpu.RemoteConnect(i, I_pops, j, I_pops,
                        {'rule': 'fixed_indegree', 'indegree': CI_distrib},
                        #'allow_autapses': False, 'allow_multapses': True},
                        syn_dict_in)
            
            comm.Barrier()

    # read out time used for building
    BuildEdgeTime = perf_counter_ns() - tic

    logger.log(str(BuildEdgeTime) + ' # build_edge_time')
    #logger.log(str(memory_thisjob()) + ' # virt_mem_after_edges')

    return E_pops, I_pops if params['record_spikes'] else None


def run_simulation():
    """Performs a simulation, including network construction"""

    # open log file
    with Logger(params['log_file']) as logger:

        ngpu.SetKernelStatus({"verbosity_level": 4})

        #logger.log(str(memory_thisjob()) + ' # virt_mem_0')

        E_neurons, I_neurons = build_network(logger)

        tic = perf_counter_ns()

        ngpu.Calibrate()

        CalibrationTime = perf_counter_ns() - tic

        #logger.log(str(memory_thisjob()) + ' # virt_mem_after_presim')
        logger.log(str(CalibrationTime) + ' # calib_time')

        tic = perf_counter_ns()

        ngpu.Simulate(params['presimtime'])

        PreparationTime = perf_counter_ns() - tic

        #logger.log(str(memory_thisjob()) + ' # virt_mem_after_presim')
        logger.log(str(PreparationTime) + ' # presim_time')

        tic = perf_counter_ns()

        ngpu.Simulate(params['simtime'])

        SimGPUTime = perf_counter_ns() - tic

        #logger.log(str(memory_thisjob()) + ' # virt_mem_after_sim')
        logger.log(str(SimGPUTime) + ' # sim_time')

        if params['record_spikes']:
            spike_times = ngpu.GetRecSpikeTimes(E_neurons)
            rate = compute_rate(spike_times)
            logger.log(str(rate) + ' # average rate')

        print(ngpu.GetKernelStatus())


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


#def memory_thisjob():
#    """Wrapper to obtain current memory usage"""
#    nest.ll_api.sr('memory_thisjob')
#    return nest.ll_api.spp()


def lambertwm1(x):
    """Wrapper for LambertWm1 function"""
    # Using scipy to mimic the gsl_sf_lambert_Wm1 function.
    return sp.lambertw(x, k=-1 if x < 0 else 0).real


class Logger:
    """Logger context manager used to properly log memory and timing
    information from network simulations.
    """

    def __init__(self, file_name):
        # copy output to cout for ranks 0..max_rank_cout-1
        self.max_rank_cout = 5
        # write to log files for ranks 0..max_rank_log-1
        self.max_rank_log = 30
        self.line_counter = 0
        self.file_name = file_name

    def __enter__(self):
        if ngpu.Rank() < self.max_rank_log:

            # convert rank to string, prepend 0 if necessary to make
            # numbers equally wide for all ranks
            rank = '{:0' + str(len(str(self.max_rank_log))) + '}'
            fn = '{fn}_{rank}.dat'.format(
                fn=self.file_name, rank=rank.format(ngpu.Rank()))

            self.f = open(fn, 'w')

        return self

    def log(self, value):
        if ngpu.Rank() < self.max_rank_log:
            line = '{lc} {rank} {value} \n'.format(
                lc=self.line_counter, rank=ngpu.Rank(), value=value)
            self.f.write(line)
            self.line_counter += 1

        if ngpu.Rank() < self.max_rank_cout:
            print(str(ngpu.Rank()) + ' ' + value + '\n', file=sys.stdout)
            print(str(ngpu.Rank()) + ' ' + value + '\n', file=sys.stderr)

    def __exit__(self, exc_type, exc_val, traceback):
        if ngpu.Rank() < self.max_rank_log:
            self.f.close()


if __name__ == '__main__':
    run_simulation()