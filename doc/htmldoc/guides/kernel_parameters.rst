.. _kernel_parameters:

NEST GPU kernel parameters
==========================

NEST GPU provides a set of configurable kernel parameters that control the global simulation state, performance tuning, memory allocation, buffer sizes, and communication behavior across GPU and MPI nodes.

To better guide users, these parameters are divided into two categories based on their relevance and usage frequency:

- :ref:`kernel-general-parameters`: Common parameters that standard users frequently adjust for their simulations.

- :ref:`kernel-advanced-parameters`: Low-level, internal, or optimization parameters that are typically managed by advanced developers and are best left at their default values.

---

.. _kernel-general-parameters:

General Parameters
------------------

These parameters are commonly accessed and modified by standard users to control basic simulation settings, output behavior, and stochastic properties.

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter Name
     - Type
     - Description
   * - ``time_resolution``
     - float
     - Simulation time step (resolution) :math:`h` in milliseconds.
   * - ``rnd_seed``
     - int
     - Base random number generator (RNG) seed for stochastic processes and network initialization as mentioned in :ref:`random_number_seed`.
   * - ``min_allowed_delay``
     - float
     - Minimum allowed synaptic delay in the network (typically bounded by the simulation resolution).
   * - ``verbosity_level``
     - int
     - Controls the amount of logging information and runtime messages printed to standard output.
   * - ``print_time``
     - bool
     - Enables or disables periodic printing of the simulation progress (current simulation time) to stdout.

---

.. _kernel-advanced-parameters:

Advanced Parameters
-------------------

These parameters regulate fine-grained memory allocation, buffer capacities, hardware bit-widths, and MPI communication strategies. Modifying them incorrectly may affect simulation stability or performance.

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter Name
     - Type
     - Description
   * - ``max_spike_num_fact``
     - float
     - Scaling factor used to allocate memory buffers for spikes on the GPU dynamically.
   * - ``max_spike_per_host_fact``
     - float
     - Safety factor for estimating the maximum number of spikes handled per host/node.
   * - ``max_remote_spike_num_fact``
     - float
     - Scaling factor for remote spike communication buffers across MPI ranks.
   * - ``use_all_source_node_fact``
     - float
     - Factor regulating memory allocation strategies when handling all-to-all or dense source-node mappings.
   * - ``max_spike_buffer_size``
     - int
     - Maximum capacity of the spike buffers allocated during simulation.
   * - ``max_node_n_bits``
     - int
     - Bit-width allocated for encoding node identifiers (IDs) in data structures.
   * - ``max_syn_n_bits``
     - int
     - Bit-width allocated for synapse identification and indexing.
   * - ``max_delay_n_bits``
     - int
     - Bit-width allocated for representing discrete synaptic delays.
   * - ``conn_struct_type``
     - int
     - Selects the structural representation format for network connectivity.
   * - ``spike_buffer_algo``
     - int
     - Selects the underlying algorithm used for managing and sorting spike buffers as listed in :ref:`spike_buffer_algorithms`.
   * - ``remove_conn_key``
     - bool
     - If enabled, removes connection keys to optimize memory footprint after initialization.
   * - ``remote_spike_mul``
     - bool
     - Multiplier/modifier policy for remote spike exchange optimization across MPI processes.
   * - ``check_node_maps``
     - bool
     - Enables rigorous consistency checks on node mapping structures between host and device.
   * - ``mpi_bitpack``
     - bool
     - Enables bitpacking techniques for MPI communication to reduce network traffic bandwidth.
   * - ``max_n_ports_warning``
     - bool
     - Toggles warning messages when approaching or exceeding the maximum number of structural ports.
   * - ``first_out_conn_in_device``
     - bool
     - Optimization flag determining placement of the first outgoing connection structures within device memory.
   * - ``have_n_out_conn``
     - bool
     - Flag indicating whether nodes track the total count of outgoing connections explicitly.
   * - ``delete_remote_node_map``
     - bool
     - Frees remote node mapping data structures from memory once they are no longer required.
   * - ``delete_image_node_map``
     - bool
     - Frees image node map structures to reclaim device/host memory post-setup.



.. _spike_buffer_algorithms:

Spike buffer algorithms
-----------------------

When constructing large-scale spiking neural networks, NEST GPU executes 
intensive iteration patterns over populations and synapses. 
To optimize performance and maximize hardware utilization across different
network topologies and densities, NEST GPU implements several **nested loop algorithms**.
These algorithms dictate how the iteration spaces (loops over source and target neurons)
are parallelized and mapped onto the GPU threads.

The available algorithms, which can be selected via configuration scripts, include:

.. list-table::
   :widths: 5 25 70
   :header-rows: 1

   * - ID
     - Name
     - Description
   * - **0**
     - **BlockStep**
     - Divides the iteration space into blocks, stepping through segments to balance the workload across GPU threads.
   * - **1**
     - **CumulSum**
     - Utilizes prefix-sum (*scan*) operations to compute memory offsets dynamically, preventing race conditions during parallel construction.
   * - **2**
     - **Simple**
     - A straightforward, baseline implementation with minimal optimization. Useful primarily for debugging or very small networks where advanced overhead is unnecessary.
   * - **3**
     - **ParallelInner**
     - Parallelizes the innermost loop of the connection generation routine.
   * - **4**
     - **ParallelOuter**
     - Parallelizes the outermost loop of the connection generation routine, often preferred when dealing with large source populations.
   * - **5**
     - **Frame1D**
     - Subdivides the network space into 1D spatial frames to enhance memory locality and coalescing during execution.
   * - **6**
     - **Frame2D**
     - Extends spatial frame decomposition into 2D grids, ideal for topologically structured or layered 2D neural sheets.
   * - **7**
     - **Smart1D**
     - An adaptive, heuristic-driven 1D algorithm that automatically optimizes thread mapping based on network density and population size.
   * - **8**
     - **Smart2D**
     - An adaptive, heuristic-driven 2D algorithm designed to dynamically select the best spatial mapping strategy for complex 2D network topographies.

The default choice for the ``spike_buffer_algo`` parameter is 0, i.e., the BlockStep algorithm. 
This was verified to be the most efficient algorithm in several large-scale simulations :footcite:p:`Golosio2023`.

References
----------

.. footbibliography::