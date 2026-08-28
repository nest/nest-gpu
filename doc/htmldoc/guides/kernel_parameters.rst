.. _kernel_parameters:

NEST GPU Kernel parameters
==========================

NEST GPU provides a set of configurable kernel parameters that control the global simulation state, performance tuning, memory allocation, buffer sizes, and communication behavior across GPU and MPI nodes.

The parameters are divided into three categories based on their data type:
- :ref:`kernel-float-parameters`

- :ref:`kernel-int-parameters`

- :ref:`kernel-bool-parameters`

---

.. _kernel-float-parameters:

Floating-Point Parameters
-------------------------

The following parameters accept floating-point values and typically regulate time resolution, integration steps, and scaling factors.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Parameter Name
     - Description
   * - ``time_resolution``
     - Simulation time step (resolution) :math:`h` in milliseconds.
   * - ``max_spike_num_fact``
     - Scaling factor used to allocate memory buffers for spikes on the GPU dynamically.
   * - ``max_spike_per_host_fact``
     - Safety factor for estimating the maximum number of spikes handled per host/node.
   * - ``max_remote_spike_num_fact``
     - Scaling factor for remote spike communication buffers across MPI ranks.
   * - ``min_allowed_delay``
     - Minimum allowed synaptic delay in the network (typically bounded by the simulation resolution).
   * - ``use_all_source_node_fact``
     - Factor regulating memory allocation strategies when handling all-to-all or dense source-node mappings.

---

.. _kernel-int-parameters:

Integer Parameters
------------------

Integer parameters manage seed generation, verbosity, bit-width constraints for hardware/data compression mapping, and algorithmic choices.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Parameter Name
     - Description
   * - ``rnd_seed``
     - Base random number generator (RNG) seed for stochastic processes and network initialization as mentioned in :ref:`random_number_seed`.
   * - ``verbosity_level``
     - Controls the amount of logging information and runtime messages printed to standard output.
   * - ``max_spike_buffer_size``
     - Maximum capacity of the spike buffers allocated during simulation.
   * - ``max_node_n_bits``
     - Bit-width allocated for encoding node identifiers (IDs) in data structures.
   * - ``max_syn_n_bits``
     - Bit-width allocated for synapse identification and indexing.
   * - ``max_delay_n_bits``
     - Bit-width allocated for representing discrete synaptic delays.
   * - ``conn_struct_type``
     - Selects the structural representation format for network connectivity.
   * - ``spike_buffer_algo``
     - Selects the underlying algorithm used for managing and sorting spike buffers.

---

.. _kernel-bool-parameters:

Boolean Parameters
------------------

Boolean flags (enabled/disabled) used to toggle diagnostic checks, debugging outputs, performance optimizations, and MPI communication strategies.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Parameter Name
     - Description
   * - ``print_time``
     - Enables or disables periodic printing of the simulation progress (current simulation time) to stdout.
   * - ``remove_conn_key``
     - If enabled, removes connection keys to optimize memory footprint after initialization.
   * - ``remote_spike_mul``
     - Multiplier/modifier policy for remote spike exchange optimization across MPI processes.
   * - ``check_node_maps``
     - Enables rigorous consistency checks on node mapping structures between host and device.
   * - ``mpi_bitpack``
     - Enables bitpacking techniques for MPI communication to reduce network traffic bandwidth.
   * - ``max_n_ports_warning``
     - Toggles warning messages when approaching or exceeding the maximum number of structural ports.
   * - ``first_out_conn_in_device``
     - Optimization flag determining placement of the first outgoing connection structures within device memory.
   * - ``have_n_out_conn``
     - Flag indicating whether nodes track the total count of outgoing connections explicitly.
   * - ``delete_remote_node_map``
     - Frees remote node mapping data structures from memory once they are no longer required.
   * - ``delete_image_node_map``
     - Frees image node map structures to reclaim device/host memory post-setup.


