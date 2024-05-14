#
#  network.py
#
#  This file is part of NEST GPU.
#
#  Copyright (C) 2021 The NEST Initiative
#
#  NEST GPU is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 2 of the License, or
#  (at your option) any later version.
#
#  NEST GPU is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with NEST GPU.  If not, see <http://www.gnu.org/licenses/>.

""" Microcircuit: Network Class
----------------------------------------

Main file of the microcircuit defining the ``Network`` class with functions to
build and simulate the network.

"""

import os
import numpy as np
import nestgpu as ngpu
import helpers


class Network:
    """ Provides functions to setup NEST GPU, to create and connect all nodes
    of the network, to simulate, and to evaluate the resulting spike data.

    Instantiating a Network object derives dependent parameters and already
    initializes NEST GPU.

    Parameters
    ---------
    sim_dict
        Dictionary containing all parameters specific to the simulation
        (see: ``sim_params.py``).
    net_dict
         Dictionary containing all parameters specific to the neuron and
         network models (see: ``network_params.py``).
    stim_dict
        Optional dictionary containing all parameter specific to the stimulus
        (see: ``stimulus_params.py``)

    """

    def __init__(self, sim_dict, net_dict, stim_dict=None):
        self.sim_dict = sim_dict
        self.net_dict = net_dict
        self.stim_dict = stim_dict
        self.Rank = 0

        # data directory
        self.data_path = sim_dict['data_path']
        if self.Rank == 0:
            if os.path.isdir(self.data_path):
                message = '  Directory already existed.'
                if self.sim_dict['overwrite_files']:
                    message += ' Old data will be overwritten.'
            else:
                os.mkdir(self.data_path)
                message = '  Directory has been created.'
            print('Data will be written to: {}\n{}\n'.format(self.data_path,
                                                             message))

        # derive parameters based on input dictionaries
        self.__derive_parameters()

        # initialize NEST GPU
        self.__setup_ngpu()

    def create(self):
        """ Creates all network nodes.

        Neuronal populations and recording and stimulating devices are created.

        """
        self.__create_neuronal_populations()
        if len(self.sim_dict['rec_dev']) > 0:
            self.__create_recording_devices()
        if self.net_dict['poisson_input']:
            self.__create_poisson_bg_input()
        if self.stim_dict['thalamic_input']:
            self.__create_thalamic_stim_input()
        if self.stim_dict['dc_input']:
            self.__create_dc_stim_input()

    def connect(self):
        """ Connects the network.

        Recurrent connections among neurons of the neuronal populations are
        established, and recording and stimulating devices are connected.

        The ``self.__connect_*()`` functions use ``ngpu.Connect()`` calls which
        set up the postsynaptic connectivity.
        The full
        connection infrastructure including presynaptic connectivity is set up
        afterwards in the preparation phase of the simulation.
        The preparation phase is usually induced by the first
        ``ngpu.Simulate()`` call.
        For including this phase in measurements of the connection time,
        we induce it here explicitly by calling ``ngpu.Prepare()``.

        """
        self.__connect_neuronal_populations()

        #if len(self.sim_dict['rec_dev']) > 0:
        #    self.__connect_recording_devices()
        if self.net_dict['poisson_input']:
            self.__connect_poisson_bg_input()
        if self.stim_dict['thalamic_input']:
            self.__connect_thalamic_stim_input()
        if self.stim_dict['dc_input']:
            self.__connect_dc_stim_input()

        #ngpu.Prepare()
        #ngpu.Cleanup()

    def simulate(self, t_sim):
        """ Simulates the microcircuit.

        Parameters
        ----------
        t_sim
            Simulation time (in ms).

        """
        if self.Rank == 0:
            print('Simulating {} ms.'.format(t_sim))

        ngpu.Simulate(t_sim)

    def evaluate(self, raster_plot_interval, firing_rates_interval):
        """ Displays simulation results.

        Creates a spike raster plot.
        Calculates the firing rate of each population and displays them as a
        box plot.

        Parameters
        ----------
        raster_plot_interval
            Times (in ms) to start and stop loading spike times for raster plot
            (included).
        firing_rates_interval
            Times (in ms) to start and stop lading spike times for computing
            firing rates (included).

        Returns
        -------
            None

        """
        
        spike_times_net = ngpu.GetRecSpikeTimes(self.neurons)
        popid = 0
        for i_pop in range(len(self.pops)):
            population = self.pops[i_pop]
            data = []
            spike_times_list = spike_times_net[popid:popid+len(population)]
            popid += len(population)
            for i_neur in range(len(population)):
                spike_times = spike_times_list[i_neur]
                if (len(spike_times) != 0):
                    # print("i_pop:", i_pop, " i_neur:", i_neur, " n_spikes:",
                    #      len(spike_times))
                    for t in spike_times:
                        data.append([population[i_neur], t])
            arr = np.array(data)
            fn = os.path.join(self.data_path, 'spike_times_' + str(i_pop) +
                              '.dat')
            fmt='%d\t%.3f'
            np.savetxt(fn, arr, fmt=fmt, header="sender time_ms",
                       comments='')
        if self.Rank == 0:
            print('Interval to plot spikes: {} ms'.format(raster_plot_interval))
            helpers.plot_raster(
                self.data_path,
                'spike_detector',
                raster_plot_interval[0],
                raster_plot_interval[1],
                self.net_dict['N_scaling'])

            print('Interval to compute firing rates: {} ms'.format(
                firing_rates_interval))
            helpers.firing_rates(
                self.data_path, 'spike_detector',
                firing_rates_interval[0], firing_rates_interval[1])
            helpers.boxplot(self.data_path, self.net_dict['populations'])

    def __derive_parameters(self):
        """
        Derives and adjusts parameters and stores them as class attributes.
        """
        self.num_pops = len(self.net_dict['populations'])

        # total number of synapses between neuronal populations before scaling
        full_num_synapses = helpers.num_synapses_from_conn_probs(
            self.net_dict['conn_probs'],
            self.net_dict['full_num_neurons'],
            self.net_dict['full_num_neurons'])

        # scaled numbers of neurons and synapses
        self.num_neurons = np.round((self.net_dict['full_num_neurons'] *
                                     self.net_dict['N_scaling'])).astype(int)
        self.num_synapses = np.round((full_num_synapses *
                                      self.net_dict['N_scaling'] *
                                      self.net_dict['K_scaling'])).astype(int)
        self.ext_indegrees = np.round((self.net_dict['K_ext'] *
                                       self.net_dict['K_scaling'])).astype(int)

        # conversion from PSPs to PSCs
        PSC_over_PSP = helpers.postsynaptic_potential_to_current(
            self.net_dict['neuron_params']['C_m'],
            self.net_dict['neuron_params']['tau_m'],
            self.net_dict['neuron_params']['tau_syn'])
        PSC_matrix_mean = self.net_dict['PSP_matrix_mean'] * PSC_over_PSP
        PSC_ext = self.net_dict['PSP_exc_mean'] * PSC_over_PSP

        # DC input compensates for potentially missing Poisson input
        if self.net_dict['poisson_input']:
            DC_amp = np.zeros(self.num_pops)
        else:
            if self.Rank == 0:
                print('DC input compensates for missing Poisson input.\n')
            DC_amp = helpers.dc_input_compensating_poisson(
                self.net_dict['bg_rate'], self.net_dict['K_ext'],
                self.net_dict['neuron_params']['tau_syn'],
                PSC_ext)

        # adjust weights and DC amplitude if the indegree is scaled
        if self.net_dict['K_scaling'] != 1:
            PSC_matrix_mean, PSC_ext, DC_amp = \
                helpers.adjust_weights_and_input_to_synapse_scaling(
                    self.net_dict['full_num_neurons'],
                    full_num_synapses, self.net_dict['K_scaling'],
                    PSC_matrix_mean, PSC_ext,
                    self.net_dict['neuron_params']['tau_syn'],
                    self.net_dict['full_mean_rates'],
                    DC_amp,
                    self.net_dict['poisson_input'],
                    self.net_dict['bg_rate'], self.net_dict['K_ext'])

        # store final parameters as class attributes
        self.weight_matrix_mean = PSC_matrix_mean
        self.weight_ext = PSC_ext
        self.DC_amp = DC_amp

        # thalamic input
        if self.stim_dict['thalamic_input']:
            num_th_synapses = helpers.num_synapses_from_conn_probs(
                self.stim_dict['conn_probs_th'],
                self.stim_dict['num_th_neurons'],
                self.net_dict['full_num_neurons'])[0]
            self.weight_th = self.stim_dict['PSP_th'] * PSC_over_PSP
            if self.net_dict['K_scaling'] != 1:
                num_th_synapses *= self.net_dict['K_scaling']
                self.weight_th /= np.sqrt(self.net_dict['K_scaling'])
            self.num_th_synapses = np.round(num_th_synapses).astype(int)

        if self.Rank == 0:
            message = ''
            if self.net_dict['N_scaling'] != 1:
                message += \
                    'Neuron numbers are scaled by a factor of {:.3f}.\n'.format(
                        self.net_dict['N_scaling'])
            if self.net_dict['K_scaling'] != 1:
                message += \
                    'Indegrees are scaled by a factor of {:.3f}.'.format(
                        self.net_dict['K_scaling'])
                message += '\n  Weights and DC input are adjusted to compensate.\n'
            print(message)

    def __setup_ngpu(self):
        """ 
        Initializes NEST GPU and sets the nested loop algorithm if NEST GPU version < 2.0.

        """

        # set seeds for random number generation
        master_seed = self.sim_dict['master_seed']
        ngpu.SetRandomSeed(master_seed)
        if('remove_conn_key' in ngpu.GetKernelStatus()):
            ngpu.SetKernelStatus({'print_time': self.sim_dict['print_time'],
                                  'spike_buffer_algo': 0,
                                  'remove_conn_key': self.sim_dict['remove_conn_key']})
        else:
            ngpu.SetKernelStatus({'print_time': self.sim_dict['print_time'],
                                  'spike_buffer_algo': 0})
            print("Cannot set remove_conn_key in NEST GPU version < 2.0")
        # 'conn_struct_type': self.sim_dict['conn_struct_type'],

        self.sim_resolution = self.sim_dict['sim_resolution']

        if hasattr(ngpu, "SetNestedLoopAlgo"):
            ngpu.SetNestedLoopAlgo(self.sim_dict['nl_algo'])
        else:
            print("Cannot set nested loop algorithm in NEST GPU version < 2.0")

    def __create_neuronal_populations(self):
        """ Creates the neuronal populations.

        The neuronal populations are created and the parameters are assigned
        to them. The initial membrane potential of the neurons is drawn from
        normal distributions dependent on the parameter ``V0_type``.

        The first and last neuron id of each population is written to file.
        """
        if self.Rank == 0:
            print('Creating neuronal populations.')

        self.n_tot_neurons = 0
        for i in np.arange(self.num_pops):
            self.n_tot_neurons = self.n_tot_neurons + self.num_neurons[i]

        self.neurons = ngpu.Create(self.net_dict['neuron_model'],
                              self.n_tot_neurons)

        tau_syn=self.net_dict['neuron_params']['tau_syn']
        E_L=self.net_dict['neuron_params']['E_L']
        V_th=self.net_dict['neuron_params']['V_th']
        V_reset=self.net_dict['neuron_params']['V_reset']
        t_ref=self.net_dict['neuron_params']['t_ref']
        ngpu.SetStatus(self.neurons, {"tau_syn":tau_syn,
                                      "E_L":E_L,
                                      "Theta_rel":V_th - E_L,
                                      "V_reset_rel":V_reset - E_L,
                                      "t_ref":t_ref})
                                     
        self.pops = []
        for i in np.arange(self.num_pops):
            if i==0:
                i_node_0 = 0
            i_node_1 = i_node_0 + self.num_neurons[i]
            #print("i_node_1 ", i_node_1)
            population = self.neurons[i_node_0:i_node_1]
            i_node_0 = i_node_1
            
            I_e=self.DC_amp[i]
            ngpu.SetStatus(population, {"I_e":I_e})
            
            #print(population.i0)
            #print(population.n)

            if self.net_dict['V0_type'] == 'optimized':
                V_rel_mean = self.net_dict['neuron_params']['V0_mean'] \
                ['optimized'][i] - E_L
                V_std = self.net_dict['neuron_params']['V0_std'] \
                        ['optimized'][i]
            elif self.net_dict['V0_type'] == 'original':
                V_rel_mean = self.net_dict['neuron_params']['V0_mean'] \
                             ['original'] - E_L,
                V_std = self.net_dict['neuron_params']['V0_std']['original']
            else:
                raise Exception(
                    'V0_type incorrect. ' +
                    'Valid options are "optimized" and "original".')

            #print("V_rel_mean", V_rel_mean)
            #print("V_std", V_std)
            #print("pop size: ", len(population))
            ngpu.SetStatus(population, {"V_m_rel": {"distribution":"normal",
                                                    "mu":V_rel_mean,
                                                    "sigma":V_std } } )

            self.pops.append(population)

        # write node ids to file
        if self.Rank == 0:
            fn = os.path.join(self.data_path, 'population_nodeids.dat')
            with open(fn, 'w+') as f:
                for pop in self.pops:
                    f.write('{} {}\n'.format(pop[0],
                                             pop[len(pop)-1]))

    def __create_recording_devices(self):
        """ Creates one recording device of each kind per population.

        Only devices which are given in ``sim_dict['rec_dev']`` are created.

        """
        if self.Rank == 0:
            print('Creating recording devices.')

        if 'spike_detector' in self.sim_dict['rec_dev']:
            if self.Rank == 0:
                print('  Activating spike time recording.')
                #for pop in self.pops:
                ngpu.ActivateRecSpikeTimes(self.neurons, 1000)
                    
            #self.spike_detectors = ngpu.Create('spike_detector',
            #                                   self.num_pops)
        #if 'voltmeter' in self.sim_dict['rec_dev']:
        #    if self.Rank == 0:
        #        print('  Creating voltmeters.')
        #    self.voltmeters = ngpu.CreateRecord('V_m_rel',
        #                                  n=self.num_pops,
        #                                  params=vm_dict)

    def __create_poisson_bg_input(self):
        """ Creates the Poisson generators for ongoing background input if
        specified in ``network_params.py``.

        If ``poisson_input`` is ``False``, DC input is applied for compensation
        in ``create_neuronal_populations()``.

        """
        if self.Rank == 0:
            print('Creating Poisson generators for background input.')

        self.poisson_bg_input = ngpu.Create('poisson_generator',
                                            self.num_pops)
        rate_list = self.net_dict['bg_rate'] * self.ext_indegrees
        for i_pop in range(self.num_pops):
            ngpu.SetStatus([self.poisson_bg_input[i_pop]],
                           "rate", rate_list[i_pop]) 

    def __create_thalamic_stim_input(self):
        """ Creates the thalamic neuronal population if specified in
        ``stim_dict``.

        Thalamic neurons are of type ``parrot_neuron`` and receive input from a
        Poisson generator.
        Note that the number of thalamic neurons is not scaled with
        ``N_scaling``.

        """
        if self.Rank == 0:
            print('Creating thalamic input for external stimulation.')

        self.thalamic_population = ngpu.Create(
            'parrot_neuron', n=self.stim_dict['num_th_neurons'])

        self.poisson_th = ngpu.Create('poisson_generator')
        self.poisson_th.set(
            rate=self.stim_dict['th_rate'],
            start=self.stim_dict['th_start'],
            stop=(self.stim_dict['th_start'] + self.stim_dict['th_duration']))

    def __connect_neuronal_populations(self):
        """ Creates the recurrent connections between neuronal populations. """
        if self.Rank == 0:
            print('Connecting neuronal populations recurrently.')

        for i, target_pop in enumerate(self.pops):
            for j, source_pop in enumerate(self.pops):
                if self.num_synapses[i][j] >= 0.:
                    conn_dict_rec = {
                        'rule': 'fixed_total_number',
                        'total_num': self.num_synapses[i][j]}

                    w_mean = self.weight_matrix_mean[i][j]
                    w_std = abs(self.weight_matrix_mean[i][j] *
                                self.net_dict['weight_rel_std'])
                    
                    if w_mean < 0:
                        w_min = w_mean-3.0*w_std
                        w_max = 0.0
                        # i_receptor = 1
                    else:
                        w_min = 0.0
                        w_max = w_mean+3.0*w_std
                        # i_receptor = 0
                        
                    d_mean = self.net_dict['delay_matrix_mean'][i][j]
                    d_std = (self.net_dict['delay_matrix_mean'][i][j] *
                             self.net_dict['delay_rel_std'])
                    d_min = self.sim_resolution
                    d_max = d_mean+3.0*d_std

                    syn_dict = {
                        'weight': {'distribution':'normal_clipped',
                                   'mu':w_mean, 'low':w_min,
                                   'high':w_max,
                                   'sigma':w_std},
                        'delay': {'distribution':'normal_clipped',
                                       'mu':d_mean, 'low':d_min,
                                       'high':d_max,
                                       'sigma':d_std}}
                        #'receptor':i_receptor}

                    ngpu.Connect(
                        source_pop, target_pop, conn_dict_rec, syn_dict)

    #def __connect_recording_devices(self):
    #    """ Connects the recording devices to the microcircuit."""
    #    if self.Rank == 0:
    #        print('Connecting recording devices.')
    #
    #    for i, target_pop in enumerate(self.pops):
    #        if 'spike_detector' in self.sim_dict['rec_dev']:
    #            conn_dict = {'rule': 'all_to_all'}
    #            syn_dict = {'weight': 1.0, 'delay': self.sim_resolution}
    #            ngpu.Connect(target_pop, [self.spike_detectors[i]],
    #                         conn_dict, syn_dict)

    def __connect_poisson_bg_input(self):
        """ Connects the Poisson generators to the microcircuit."""
        if self.Rank == 0:
            print('Connecting Poisson generators for background input.')

        for i, target_pop in enumerate(self.pops):
            conn_dict_poisson = {'rule': 'all_to_all'}

            syn_dict_poisson = {
                'weight': self.weight_ext,
                'delay': self.net_dict['delay_poisson']}

            ngpu.Connect(
                [self.poisson_bg_input[i]], target_pop,
                conn_dict_poisson, syn_dict_poisson)

    def __connect_thalamic_stim_input(self):
        """ Connects the thalamic input to the neuronal populations."""
        if self.Rank == 0:
            print('Connecting thalamic input.')

        # connect Poisson input to thalamic population
        ngpu.Connect(self.poisson_th, self.thalamic_population)

        # connect thalamic population to neuronal populations
        for i, target_pop in enumerate(self.pops):
            conn_dict_th = {
                'rule': 'fixed_total_number',
                'N': self.num_th_synapses[i]}

            w_mean = self.weight_th,
            w_std = self.weight_th * self.net_dict['weight_rel_std']
            w_min = 0.0,
            w_max = w_mean + 3.0*w_std

            d_mean = self.stim_dict['delay_th_mean']
            d_std = (self.stim_dict['delay_th_mean'] *
                     self.stim_dict['delay_th_rel_std'])
            d_min = self.sim_resolution
            d_max = d_mean + 3.0*d_std

            syn_dict_th = {
                'weight': {"distribution":"normal_clipped",
                           "mu":w_mean, "low":w_min,
                           "high":w_max,
                           "sigma":w_std},
                'delay': {"distribution":"normal_clipped",
                          "mu":d_mean, "low":d_min,
                          "high":d_max,
                          "sigma":d_std}}
 
            ngpu.Connect(
                self.thalamic_population, target_pop,
                conn_spec=conn_dict_th, syn_spec=syn_dict_th)

