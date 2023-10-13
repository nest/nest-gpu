Differences in usage between NEST GPU and NEST
==============================================

Aeif neuron models
------------------

Aeif neuron models in NEST GPU have both a non-multisynapse and 
a multisynapse implementation.
For the multisynapse implementation, the number
of receptor ports must be specified at neuron creation:

.. code-block:: python

   # Create n_neurons neurons with n_receptor receptor ports
   neuron = ngpu.Create("aeif_cond_beta_multisynapse", n_neurons, n_receptors)

If not specified, the number of neurons and the number of receptors are
set to 1 and cannot be changed. The receptor index starts from 0 (and
not from 1 as in NEST multisynapse models).

The non-multisynapse implementation of aeif neuron models has two receptor
ports (i.e. excitatory and inhibitory), and thus the connections require to
specify the receptor port through the synapse property ``receptor`` (0 
for the excitatory port and 1 for the inhibitory port). Differently from
NEST, the connection weights related to the inhibitory port must be positive.

The only exception is the ``aeif_psc_delta`` neuron model, which has only
one receptor port. For this model, an inhibitory connection must have
a negative weight.

Multimeter device
-----------------

The multimeter devices in NEST GPU are used in a different way from
NEST. To record a variable, you have to create a record, as in the
following example:

.. code-block:: python

   import nestgpu as ngpu

   neuron = ngpu.Create("aeif_cond_beta", 3) # create a population of 3 neurons 

   ngpu.SetStatus(neuron, {"I_e":1000.0}) # set a constant input current

   filename = "test.dat" # file where the record will be saved. If empty ("") no file is produced

   i_neurons = [neuron[0], neuron[1], neuron[2]] # any set of neuron indexes

   var_name_arr = ["V_m", "V_m", "V_m"] # variables to be recorded

   i_receptor_arr = [0, 0, 0] # receptor ports from which the variable should be recorded
                              # (0 for scalar variables)

   # create multimeter record of V_m
   record = ngpu.CreateRecord(filename, var_name_arr, i_neuron_arr,
                                   i_receptor_arr)

   ngpu.Simulate(800) #simulate 800 ms of biological time

   data_list = ngpu.GetRecordData(record) # get data from record
   t=[row[0] for row in data_list] # extract time from data
   V1=[row[1] for row in data_list] # extract membrane potential of neuron 0
   V2=[row[2] for row in data_list] # extract membrane potential of neuron 1
   V3=[row[3] for row in data_list] # extract membrane potential of neuron 2
