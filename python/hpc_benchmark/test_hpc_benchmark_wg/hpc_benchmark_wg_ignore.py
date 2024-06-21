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
Warning: the NEST GPU implementation still presents differencies with respect
to NEST in the average firing rate of neurons.

This script produces a balanced random network of `scale*11250` neurons
connected with static connections. The number of incoming connections 
per neuron is fixed and independent of network size (indegree=11250).

Furthermore, the scale can be also increased through running the script
using several MPI processes using the command mpirun -np nproc python hpc_benchmark.py
This way, a network of `scale*11250` neurons is built in every MPI process,
with indegrees equally distributed across the processes.

This is the standard network investigated in [1]_, [2]_, [3]_.
A note on connectivity
~~~~~~~~~~~~~~~~~~~~~~
Each neuron receives :math:`K_{in,{\\tau} E}` excitatory connections randomly
drawn from population E and :math:`K_{in,\\tau I}` inhibitory connections from
population I. Autapses are prohibited while multapses are allowed. Each neuron
receives additional input from an external stimulation device. All delays are
constant, all weights but excitatory onto excitatory are constant.

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
11.250 incoming synapses per neuron is retained.
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

import os
import sys
import json
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

from time import perf_counter_ns

import nestgpu as ngpu

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--path", type=str, default=".")
parser.add_argument("--seed", type=int, default=12345)
args = parser.parse_args()

M_INFO = 10
M_ERROR = 30

ngpu.ConnectMpiInit()

mpi_id = ngpu.HostId()
mpi_np = ngpu.HostNum()

hg = ngpu.CreateHostGroup(list(range(mpi_np)))

###############################################################################
# Parameter section
# Define all relevant parameters: changes should be made here


params = {
    'scale': 0.1,             # scaling factor of the network size
                             # total network size = scale*11250 neurons
    'seed': args.seed,       # seed for random number generation
    'simtime': 250.,         # total simulation time in ms
    'presimtime': 50.,       # simulation time until reaching equilibrium
    'dt': 0.1,               # simulation step
    'stdp': False,           # enable plastic connections [feature not properlyly implemented yet!]
    'record_spikes': True,  # switch to record spikes of excitatory
                             # neurons to file
    'show_plot': False,      # switch to show plot at the end of simulation
                             # disabled by default for benchmarking
    'raster_plot': True,    # when record_spikes=True, depicts a raster plot
    'path_name': args.path,  # path where all files will have to be written
    'log_file': 'log',       # naming scheme for the log files
    'use_all_to_all': False, # Connect using all to all rule
    'check_conns': False,    # Get ConnectionId objects after build. VERY SLOW!
    'use_dc_input': False,   # Use DC input instead of Poisson generators
    'verbose_log': False,    # Enable verbose output per MPI process
    'ignore_and_fire': True,  # if True, all connection weights but those from poisson generator are set to zero
    'ignore_and_fire_rate': 50000.0, # rate of poisson generator when the 'ignore_and_fire' flag is True
    'ignore_and_fire_weight': 10.75 # weight of poisson generator connections when the 'ignore_and_fire' flag is True 
}


def rank_print(message):
    """Prints message and attaches MPI rank"""
    if params['verbose_log']:
        print(f"MPI RANK {mpi_id}: {message}")

rank_print("Simulation with {} MPI processes".format(mpi_np))


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

def dc_input_compensating_poisson(*args, **kwargs):
    """TEST FUNCTION
    DC amplitude tuned to obtain a target firing rate of 13Hz for E and I pops.
    """
    return 500.1

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
        'V_m_rel': 0.0 #5.7  # mean value of membrane potential
    },

    ####################################################################
    # Note that Kunkel et al. (2014) report different values. The values
    # in the paper were used for the benchmarks on K, the values given
    # here were used for the benchmark on JUQUEEN.

    'randomize_Vm': True,
    'mean_potential': 5.7,
    'sigma_potential': 7.2,

    'delay': 1.5,  # synaptic delay, all alpha connections(ms)

    # synaptic weight
    'JE': 0.14,  # peak of EPSP

    'sigma_w': 3.47,  # standard dev. of E->E synapses(pA)
    'g': -5.0,

    # stdp synapses still to be implemented correctly
    'stdp_params': {
        'alpha': 0.0513,
        'lambda': 0.1,  # STDP step size
        'mu_plus': 0.4,  # STDP weight dependence exponent(potentiation)
        'mu_minus': 0.4,  # STDP weight dependence exponent(depression)
        'tau_plus': 15.0,  # time constant for potentiation
        'tau_minus': 15.0,  # time constant for depression
    },
    'stdp_delay': 1.5,

    'eta': 1.685,  # scaling of external stimulus
    'filestem': params['path_name'],
}

###############################################################################
# Function Section

def build_network():
    """Builds the network including setting of simulation and neuron
    parameters, creation of neurons and connections.
    Uses a dictionary to store information about the network construction times.
    
    Returns recorded neuron ids if spike recording is enabled, and the time dictionary.
    """

    time_start = perf_counter_ns()  # start timer on construction

    # unpack a few variables for convenience
    NE = brunel_params['NE']
    NI = brunel_params['NI']
    model_params = brunel_params['model_params']
    stdp_params = brunel_params['stdp_params']

    rank_print('Creating neuron populations.')

    neurons = []; E_pops = []; I_pops = []

    if(mpi_np > 1):
        for i in range(mpi_np):
            neurons.append(ngpu.RemoteCreate(i, 'iaf_psc_alpha', NE+NI, 1, model_params).node_seq)
            E_pops.append(neurons[i][0:NE])
            I_pops.append(neurons[i][NE:NE+NI])

    else:
        neurons.append(ngpu.Create('iaf_psc_alpha', NE+NI, 1, model_params))
        E_pops.append(neurons[mpi_id][0:NE])
        I_pops.append(neurons[mpi_id][NE:NE+NI])

    if brunel_params['randomize_Vm']:
        rank_print('Randomizing membrane potentials.')
        ngpu.SetStatus(neurons[mpi_id], {"V_m_rel": {"distribution": "normal", "mu": brunel_params['mean_potential'], "sigma": brunel_params['sigma_potential']}})

    # total number of incoming excitatory connections
    CE = int(1. * NE / params['scale'])
    # total number of incomining inhibitory connections
    CI = int(1. * NI / params['scale'])

    rank_print('Creating excitatory stimulus generator.')

    # Convert synapse weight from mV to pA
    conversion_factor = convert_synapse_weight(model_params['tau_m'], model_params['tau_syn_ex'], model_params['C_m'])
    JE_pA = conversion_factor * brunel_params['JE']
    if mpi_id==0:
       rank_print("Synaptic weights: JE={}; JI={}".format(JE_pA, JE_pA*brunel_params['g']))

    nu_thresh = model_params['Theta_rel'] / ( CE * model_params['tau_m'] / model_params['C_m'] * JE_pA * np.exp(1.) * tau_syn)
    nu_ext = nu_thresh * brunel_params['eta']
    rate = nu_ext * CE * 1000.
    if params['ignore_and_fire']:
        brunel_params["poisson_rate"] = params['ignore_and_fire_rate']
        E_stim= ngpu.Create('poisson_generator', 1, 1, {'rate': params['ignore_and_fire_rate']})
        JE_pA = 0.0
    elif not params['use_dc_input']:
        brunel_params["poisson_rate"] = rate
        E_stim= ngpu.Create('poisson_generator', 1, 1, {'rate': rate})
    else:
        inh_amp = dc_input_compensating_poisson(rate, CI, tau_syn, brunel_params['g'] * JE_pA)
        ex_amp = dc_input_compensating_poisson(rate, CE, tau_syn, JE_pA)
        brunel_params["DC_amp_I"] = inh_amp
        brunel_params["DC_amp_E"] = ex_amp
        ngpu.SetStatus(I_pops[mpi_id], {"I_e": inh_amp})
        ngpu.SetStatus(E_pops[mpi_id], {"I_e": ex_amp})


    rank_print('Creating excitatory spike recorder.')

    if params['record_spikes']:
        recorder_label = 'alpha_' + str(stdp_params['alpha']) + '_spikes_' + str(mpi_id)
        brunel_params["recorder_label"] = recorder_label
        ngpu.ActivateRecSpikeTimes(neurons[mpi_id], 1000)
        record = ngpu.CreateRecord("", ["V_m_rel"], [neurons[mpi_id][0]], [0])

    time_create = perf_counter_ns()

    syn_dict_ex = None
    syn_dict_in = {'weight': brunel_params['g'] * JE_pA, 'delay': brunel_params['delay']}
    if params['stdp']:
        syn_group_stdp = ngpu.CreateSynGroup('stdp', stdp_params)
        syn_dict_ex = {"weight": JE_pA, "delay": brunel_params['stdp_delay'], "synapse_group": syn_group_stdp}
    else:
        syn_dict_ex = {'weight': JE_pA, 'delay': brunel_params['delay']}

    if params['ignore_and_fire']:
        rank_print('Connecting stimulus generators.')
        syn_dict_stim = {'weight': params['ignore_and_fire_weight'], 'delay': brunel_params['delay']}
        # connect Poisson generator to neuron
        my_connect(E_stim, neurons[mpi_id], {'rule': 'all_to_all'}, syn_dict_stim)
    elif not params["use_dc_input"]:
        rank_print('Connecting stimulus generators.')
        # connect Poisson generator to neuron
        my_connect(E_stim, neurons[mpi_id], {'rule': 'all_to_all'}, syn_dict_ex)

    rank_print('Creating local connections.')
    rank_print('Connecting excitatory -> excitatory population.')

    # number of indegrees from current MPI process
    CE_local = CE // mpi_np
    if ( (2*mpi_id) % mpi_np ) < ( CE % mpi_np ):
        CE_local = CE_local + 1
        
    CI_local = CI // mpi_np
    if ( (2*mpi_id) % mpi_np ) < ( CI % mpi_np ):
        CI_local = CI_local + 1

    if params['use_all_to_all']:
        i_conn_rule = {'rule': 'all_to_all'}
        e_conn_rule = {'rule': 'all_to_all'}
    else:
        i_conn_rule = {'rule': 'fixed_indegree', 'indegree': CI_local}
        e_conn_rule = {'rule': 'fixed_indegree', 'indegree': CE_local}

    brunel_params["connection_rules"] = {"inhibitory": i_conn_rule, "excitatory": e_conn_rule}
    
    my_connect(E_pops[mpi_id], neurons[mpi_id],
                 e_conn_rule, syn_dict_ex)

    my_connect(I_pops[mpi_id], neurons[mpi_id],
                 i_conn_rule, syn_dict_in)
        
    time_connect_local = perf_counter_ns()

    rank_print('Creating remote connections.')
    
    for i in range(mpi_np):
        for j in range(mpi_np):
            if(i!=j):
                rank_print('Connecting excitatory {} -> excitatory {} population.'.format(i, j))

                # number of indegrees from each MPI process
                # here the indegrees are equally distributed among the
                # neuron populations in all the MPI processes
                CE_distrib = CE // mpi_np
                if ( (i + j) % mpi_np ) < ( CE % mpi_np ):
                    CE_distrib = CE_distrib + 1
                    
                CI_distrib = CI // mpi_np
                if ( (i + j) % mpi_np ) < ( CI % mpi_np ):
                    CI_distrib = CI_distrib + 1
                    
                if params['use_all_to_all']:
                    i_conn_rule = {'rule': 'all_to_all'}
                    e_conn_rule = {'rule': 'all_to_all'}
                else:
                    i_conn_rule = {'rule': 'fixed_indegree', 'indegree': CI_distrib}
                    e_conn_rule = {'rule': 'fixed_indegree', 'indegree': CE_distrib}

                my_remoteconnect(i, E_pops[i], j, neurons[j],
                                    e_conn_rule, syn_dict_ex, hg)

                rank_print('Connecting inhibitory {} -> excitatory {} population.'.format(i, j))
                
                my_remoteconnect(i, I_pops[i], j, neurons[j],
                                    i_conn_rule, syn_dict_in, hg)

                rank_print('Connecting excitatory {} -> inhibitory {} population.'.format(i, j))

                rank_print('Connecting inhibitory {} -> inhibitory {} population.'.format(i, j))

    # read out time used for building
    time_connect_remote = perf_counter_ns()

    time_dict = {
        "time_create": time_create - time_start,
        "time_connect_local": time_connect_local - time_create,
        "time_connect_remote": time_connect_remote - time_connect_local,
        "time_connect": time_connect_remote - time_create
    }

    conns = None
    if params['check_conns']:
        conns = dict()
        for i in range(mpi_np):
            if mpi_id == i:
                conns[i] = dict()
                for j in range(mpi_np):
                    conns[i][j] = dict()
                    conns[i][j]["E"] = dict()
                    conns[i][j]["I"] = dict()
                    conns[i][j]["E"]["E"] = get_conn_dict_array(E_pops[i], E_pops[j])
                    conns[i][j]["I"]["E"] = get_conn_dict_array(I_pops[i], E_pops[j])
                    conns[i][j]["E"]["I"] = get_conn_dict_array(E_pops[i], I_pops[j])
                    conns[i][j]["I"]["I"] = get_conn_dict_array(I_pops[i], I_pops[j])

        time_check_connect = perf_counter_ns()

        time_dict["time_check_connect"] = time_check_connect - time_connect_remote

    return neurons[mpi_id], record if params['record_spikes'] else None, conns, time_dict


def run_simulation():
    """Performs a simulation, including network construction"""

    time_start = perf_counter_ns()

    ngpu.SetKernelStatus({
        "verbosity_level": 4,
        "rnd_seed": params["seed"],
        "time_resolution": params['dt']
        })
    seed = ngpu.GetKernelStatus("rnd_seed")
    
    time_initialize = perf_counter_ns()

    neurons, record, conns, time_dict = build_network()

    time_construct = perf_counter_ns()

    ngpu.Calibrate()

    time_calibrate = perf_counter_ns()

    ngpu.Simulate(params['presimtime'])

    time_presimulate = perf_counter_ns()

    ngpu.Simulate(params['simtime'])

    time_simulate = perf_counter_ns()

    time_dict.update({
            "time_initialize": time_initialize - time_start,
            "time_construct": time_construct - time_initialize,
            "time_calibrate": time_calibrate - time_construct,
            "time_presimulate": time_presimulate - time_calibrate,
            "time_simulate": time_simulate - time_presimulate,
            "time_total": time_simulate - time_start
        })

    conf_dict = {
        "num_processes": mpi_np,
        "brunel_params": brunel_params,
        "simulation_params": params
    }

    info_dict = {
        "rank": mpi_id,
        "seed": seed,
        "conf": conf_dict,
        "timers": time_dict
    }

    if params['record_spikes']:
        e_stats, e_data, i_stats, i_data= get_spike_times(neurons)
        e_rate = compute_rate(*e_stats)
        i_rate = compute_rate(*i_stats)
        info_dict["stats"] = {
            "excitatory_firing_rate": e_rate,
            "inhibitory_firing_rate": i_rate
        }
        
        if params['show_plot']:
            recorded_data = ngpu.GetRecordData(record)
            time = [row[0] for row in recorded_data]
            V_m = [row[1] for row in recorded_data]
            plt.figure(mpi_id)
            plt.plot(time, V_m, '-r')
            plt.draw()
        if params['raster_plot']:
            raster_plot(e_data, i_data)

    if params['check_conns']:
        with open(os.path.join(params['path_name'], f"connections_{mpi_id}.json"), 'w') as f:
            json.dump(conns, f, indent=4)

    k_status = ngpu.GetKernelStatus()
    info_dict["kernel_status"] = k_status

    rank_print(json.dumps(info_dict, indent=4))

    with open(os.path.join(params['path_name'], params['log_file'] + f"_{mpi_id}.json"), 'w') as f:
        json.dump(info_dict, f, indent=4)

def my_connect(source, target, conn_dict, syn_dict):
    rank_print("MY id {} LOCAL Source {} {} | Target {} {}".format(mpi_id, source.i0, source.n, target.i0, target.n))
    ngpu.Connect(source, target, conn_dict, syn_dict)


def my_remoteconnect(source_host, source, target_host, target, conn_dict, syn_dict, hg):
    rank_print("MY id {} REMOTE Source {} {} {} | Target {} {} {}".format(mpi_id, source_host, source.i0, source.n, target_host, target.i0, target.n))
    ngpu.RemoteConnect(source_host, source, target_host, target, conn_dict, syn_dict, hg)


def get_conn_dict_array(source, target):
    """Retrieve neural connections as an array of dictionaries."""
    connectionIdArray = ngpu.GetConnections(source, target)
    res = [{"i_source": i.i_source, "i_group": i.i_group, "i_conn": i.i_conn} for i in connectionIdArray]
    return res


def get_spike_times(neurons):
    """Retrieve spike times from local neuron population
    and filter through inhibitory neurons to store only excitatory spikes.
    """

    # get spikes
    spike_times = ngpu.GetRecSpikeTimes(neurons)

    # select excitatory neurons
    e_count = 0
    e_data = []
    e_bound = brunel_params['NE']
    i_count = 0
    i_data = []
    i_bound = brunel_params['NE'] + brunel_params['NI']
    for i_neur in range(i_bound):
        spikes = spike_times[i_neur]
        if len(spikes) != 0:
            if i_neur < e_bound:
                for t in spikes:
                    if t > params['presimtime']:
                        e_count += 1
                        e_data.append([i_neur, t])
            else:
                for t in spikes:
                    if t > params['presimtime']:
                        i_count += 1
                        i_data.append([i_neur, t])

    # Save data
    if len(e_data) > 0:
        e_array = np.array(e_data)
        e_fn = os.path.join(brunel_params['filestem'], brunel_params["recorder_label"] + "_e_pop.dat")
        np.savetxt(e_fn, e_array, fmt='%d\t%.3f', header="sender time_ms", comments='')

    if len(i_data) > 0:
        i_array = np.array(i_data)
        i_fn = os.path.join(brunel_params['filestem'], brunel_params["recorder_label"] + "_i_pop.dat")
        np.savetxt(i_fn, i_array, fmt='%d\t%.3f', header="sender time_ms", comments='')

    return (brunel_params['NE'], e_count), e_data, (brunel_params['NI'], i_count), i_data


def compute_rate(num_neurons, spike_count):
    """Compute local approximation of average firing rate
    This approximation is based on the number of local nodes, number
    of local spikes and total time. Since this also considers devices,
    the actual firing rate is usually underestimated.
    """
    if spike_count < 1:
        return 0

    time_frame = params['simtime']

    return (1. * spike_count / (num_neurons * time_frame) * 1e3)

def raster_plot(e_st, i_st):
    fs = 18 # fontsize
    colors = ['#595289', '#af143c']
    e_ids = np.zeros(len(e_st)); i_ids = np.zeros(len(i_st))
    e_times = np.zeros(len(e_st)); i_times = np.zeros(len(i_st))
    for i in range(len(e_st)):
        e_ids[i]=e_st[i][0]
        e_times[i]=e_st[i][1]
    for i in range(len(i_st)):
        i_ids[i]=i_st[i][0]
        i_times[i]=i_st[i][1]
    
    plt.figure(1, figsize=(16, 10))
    plt.plot(e_times, e_ids, '.', color=colors[0])
    plt.plot(i_times, i_ids, '.', color=colors[1])
    plt.xlabel('time [ms]', fontsize=fs)
    plt.ylabel('neuron ID', fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.tight_layout()
    plt.savefig(os.path.join(brunel_params['filestem'], 'raster_plot'+ str(mpi_id) +'.png'), dpi=300)

if __name__ == '__main__':
    run_simulation()
    if params['show_plot']:
        plt.show()
    ngpu.MpiFinalize()
