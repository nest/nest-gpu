.. _randomness_in_nestgpu_simulations:

==================================
Randomness in NEST GPU Simulations
==================================

As in NEST, random numbers are used in several occasions for neural network creation, such
as the randomization of node and connection parameters and when stochastic input or stochastic 
connection rules are employed in the simulation. NEST GPU uses random generators from the 
`curand <https://docs.nvidia.com/cuda/curand/index.html>`_ library of CUDA to obtain random
numbers following different distributions.

.. _random_number_seed:

Random numbers for simulation
=============================

Similar to the CPU version of NEST, the randomness for a simulation can be set
throughout a master seed, which is part of the kernel parameters (see 
:doc:`kernel_parameters` for more information in this regard). This is used
both for the probabilistic connection rules, the creation of parameter 
distributions as described below and the stochastic input generation. It can be
set as follows:

.. code-block:: python

   nestgpu.SetKernelStatus("rnd_seed", 1234)


.. _random_number_params:

Random numbers for network parameters
=====================================

Normal distribution
-------------------

Draws a normal distribution given a mean (``mu``) and standard deviation (``sigma``).
Values for mean and standard deviation must be always specified. The following example shows
how the distribution can be used to randomize the membrane potential of a neuron population.

.. code-block:: python

   n=nestgpu.Create('aeif_cond_beta', 10000, 3)
   nestgpu.SetStatus(n, 'V_m', {'distribution':'normal','mu':1.0, 'sigma':0.5})


Normal clipped distribution
---------------------------

Draws a normal clipped distribution given a mean (``mu``) and standard deviation (``sigma``).
The distribution is clipped on the range [low, high], where ``low`` and ``high`` are specified
alongside mean and standard deviation, as shown in an example similar to the one above.

.. code-block:: python

   n=nestgpu.Create('aeif_cond_beta', 10000, 3)
   nestgpu.SetStatus(n, 'V_m', {'distribution':'normal_clipped','mu':1.0, 'sigma':0.5, 'low':0.1, 'high':2.0})


Lognormal clipped distribution
------------------------------

Draws a lognormal clipped distribution with mean ``mu`` and standard deviation ``sigma``,
where the mean and standard deviation are of the underlying normal distribution. This is 
the same approach used in Python libraries such as Numpy, and also in the CPU version of
NEST. The following code shows how to lognormally distribute connection parameters such 
as synaptic weights and delays.

.. code-block:: python

   mu = 10.0
   sigma = 5.0
   low = 0.1
   high = 100

   neuron1 = nestgpu.Create("aeif_cond_beta_multisynapse", 5000)
   neuron2 = nestgpu.Create("aeif_cond_beta_multisynapse", 5000)
   nestgpu.Connect(neuron1, neuron2, {'rule': 'one_to_one'}, {'delay': {'distribution': 'lognormal_clipped', 'mu': mu, 'sigma':sigma, 'low': low, 'high': high},                   
                                                         'weight': {'distribution': 'lognormal_clipped', 'mu': mu, 'sigma': sigma, 'low': low, 'high': high}})


Other distributions
-------------------

If other distributions need to be generated, NEST GPU enables
passing a Python array to the parameter. This way, also the
distributions implemented in Python scientific libraries can
be used for simulation. The array can be simply passed when
setting the node or connection parameter.
