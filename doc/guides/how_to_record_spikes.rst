How to record spikes in NEST GPU
================================

There are different ways to record spikes in NEST GPU.
The first way is to continuously record the spikes
using the device :doc:`spike_detector <../models/spike_detector>`.

An alternative way, which is computationally faster than
the ``spike_detector`` device, can be achieved using
the ``RecSpikeTimes`` method. 
This method has to be activated before the ``Simulate`` 
function through the command ``ActivateRecSpikeTimes`` in this way:

::

    nestgpu.ActivateRecSpikeTimes(neurons, N_max_spike_times)

where ``neurons`` is a population of N neurons created using the
``Create`` function, and ``N_max_spike_times`` is an integer
which sets the maximum amount of spikes that can be recorded
from each neuron of the population (needed to optimize GPU
memory). This method does not enable the recording of 
a subset of neurons belonging to a population created in a 
single ``Create`` function.

After the simulation, the spike times of the recorded population
can be obtained using the command ``GetRecSpikeTimes``, which
returns a list of N lists with the spike times for every neuron
of the population:

::

    spike_times = nestgpu.GetRecSpikeTimes(neurons)

    