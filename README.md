# NEST GPU

**NEST GPU was developed under the name [NeuronGPU](https://github.com/golosio/NeuronGPU) before it has been integrated in the NEST Initiative, see [Golosio et al. (2021)](https://www.frontiersin.org/articles/10.3389/fncom.2021.627620/full). Currently this repository is being adapted to the NEST development workflow.**

NEST GPU is a GPU-MPI library for simulation of large-scale networks of spiking neurons.
Can be used in Python, in C++ and in C.

With this library it is possible to run relatively fast simulations of large-scale networks of spiking neurons. For instance, on a single Nvidia GeForce RTX 2080 Ti GPU board it is possible to simulate the activity of 1 million multisynapse AdEx neurons with 1000 synapse per neurons, for a total of 1 billion synapse, using the fifth-order Runge-Kutta method with adaptive stepsize as differential equations solver, in little more than 70 seconds per second of neural activity. The MPI communication is also very efficient. 
The Python interface is very similar to that of the NEST simulator: the most used commands are practically identical, dictionaries are used to define neurons, connections and synapses properties in the same way.

## Documentation
To get started with NEST GPU, please see the [NEST GPU Documentation](https://nest-gpu.readthedocs.io/en/latest/).

## License
NEST GPU is an open source software licensed under the [GNU General Public Lincese version 2](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html).

## Installing NEST GPU
To install NEST GPU see the [NEST GPU Installation Intructions](https://nest-gpu.readthedocs.io/en/latest/installation/installation.html) on our Documentation.

## Citing NEST GPU
If you use NEST GPU in your work, please cite the publications in our [publication list](https://nest-gpu.readthedocs.io/en/latest/publications.html).
