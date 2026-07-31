Simulations using multiple GPUs
===============================

NEST GPU can exploit multiple GPUs in parallel to perform simulations. It is common to have a configuration where a single GPU is assigned to each MPI process of the parallel simulation. Although having multiple MPI ranks using the same GPU is possible, that is not the intended use of NEST GPU for multi-GPU simulations.

.. note::
   To ensure that each MPI rank is correctly mapped to a specific GPU, users may need to configure the ``CUDA_VISIBLE_DEVICES`` environment variable before launching the simulation. While workload managers such as SLURM typically handle this resource allocation automatically, manual configuration is necessary when running parallel simulations outside cluster environments.

The library naturally exploits the locality of the network: while the CPU version of NEST distributes the network model using a round-robin approach, the GPU version enables the creation of an entire population inside a specific MPI process.
This is particularly useful in the case of modular networks, where the spike traffic is higher within the same module of the network. However, the simulation code shows some differences with respect to the CPU version, and the user should be aware of how to distribute the neuron populations of the model among different MPI processes.

In the following, we outline the main differences when creating and connecting populations located across different MPI processes.

Differences in Create and Connect calls
---------------------------------------

In single-GPU simulations, ``Create`` and ``Connect`` calls follow the same structure as the CPU version of NEST.
Thus, a typical command for creating neurons is structured as follows:

.. code-block:: python

    neuron_pop = nestgpu.Create('iaf_psc_exp', 1000)

In multi-GPU simulations, where each MPI process is mapped to a GPU, we may need to specify the MPI process in which a population is created. In NEST GPU, this is achieved using the ``RemoteCreate`` function, which accepts the MPI process index as an additional argument. For instance:

.. code-block:: python

    mpi_proc_id = 0
    neuron_pop = nestgpu.RemoteCreate(mpi_proc_id, 'iaf_psc_exp', 1000)

This command creates the ``neuron_pop`` population on Rank 0 of the simulation.
The ``RemoteCreate`` call is required especially when instantiating different model populations within a list, or when populations belonging to different processes differ. If a simulation requires identical populations to be simulated inside each MPI process, the standard ``Create`` call remains valid.

When nodes belonging to different MPI processes need to be connected, a ``RemoteConnect`` call is required:

.. code-block:: python
    
    nestgpu.RemoteConnect(source_host_id, source_pop, target_host_id, target_pop, conn_dict, syn_dict)

This connects the population ``source_pop`` instantiated on Rank ``source_host_id`` to ``target_pop`` instantiated on ``target_host_id``. 
Essentially, you must declare the MPI process ID where each population is instantiated alongside the populations themselves.

.. warning::
    Although each rank can independently construct its share of the network, the correct way to use remote creation and connections in NEST GPU is to have all ranks call all creation and connection functions.
    For this reason, ``RemoteCreate`` and ``RemoteConnect`` calls should not be specified inside a rank conditional block.
    In principle, if both the ranks involved in a connection are included in the condition block, e.g.,
    
    .. code-block:: python

        # don't try this at home!
        if rank == 1 or rank == 3:
            nestgpu.RemoteConnect(1 ,pop_1, 3, pop_3, ...)
    
    the connections would be correctly instantiated. However, this is highly discouraged for regular users.



Different communication strategies
----------------------------------

In [1], an efficient network construction method was implemented in NEST GPU, enabling the runtime building of models in single-GPU simulations. This method has been extended in [2] for multi-GPU simulations.

Each MPI process builds its portion of the network without requiring MPI communication with other processes. Then, it creates the data structures needed to communicate via MPI and transmits spikes across remote connections. Spike delivery during the simulation can be carried out using either point-to-point or collective communication, depending on the user's requirements.
The following sections present the differences between these two methods and provide examples of their usage. For additional details on the implementation, please refer to [2].

Point-to-point communication
----------------------------
*Point-to-point* protocols enable direct communication between specific pairs of processes. They are advantageous for networks with an uneven or sparse distribution of neurons and connections, which results in heterogeneous communication patterns between processes.

This is the default communication method in NEST GPU. The code block below shows how to set up a multi-GPU simulation using this approach. The example is adapted from ``brunel_mpi.py``, which simulates a balanced Brunel network.

.. code-block:: python

    import nestgpu as ngpu

    # MPI initialization
    ngpu.ConnectMpiInit()

    # Every process creates the 'neuron' population locally
    n_neurons = 1000
    neuron = ngpu.Create("aeif_cond_beta_multisynapse", n_neurons)

    exc_neuron = neuron[0:800]           # Excitatory neurons
    inh_neuron = neuron[800:n_neurons]    # Inhibitory neurons

    CE = 800
    Wex = 0.05
    delay = 1.0

    exc_conn_dict = {"rule": "fixed_indegree", "indegree": CE*3//4}
    exc_syn_dict = {"weight": Wex, "delay": delay}
    
    # Connection taking place locally, inside every MPI process
    ngpu.Connect(exc_neuron, neuron, exc_conn_dict, exc_syn_dict)

    # Creating remote connections between the excitatory population 
    # of Rank 0 and the neurons of Rank 1, and vice versa
    re_conn_dict = {"rule": "fixed_indegree", "indegree": CE//4}
    re_syn_dict = {"weight": Wex, "delay": delay}

    # Host 0 to Host 1
    ngpu.RemoteConnect(0, exc_neuron, 1, neuron, re_conn_dict, re_syn_dict)
    # Host 1 to Host 0
    ngpu.RemoteConnect(1, exc_neuron, 0, neuron, re_conn_dict, re_syn_dict)


Collective communication
------------------------
The *collective communication* function utilized in NEST GPU is ``MPI_Allgather``. With this approach, each process in an MPI group communicates the spikes of its source neurons simultaneously to all other processes in the same group, while receiving the spikes sent by the others. When the network load across nodes is balanced and communication payloads are homogeneous, collective protocols can be more efficient than point-to-point methods.

To implement collective communication, a *host group* must be defined to instantiate the collective communicator. When connecting populations collectively, the host group is passed as a parameter to the connection command.

The following example, adapted from ``hpc_benchmark.py``, demonstrates how to initialize a host group and use ``ConnectDistributedFixedIndegree`` to connect populations across processes:

.. code-block:: python

    import nestgpu as ngpu

    # MPI initialization
    ngpu.ConnectMpiInit()
    mpi_np = ngpu.HostNum()
    mpi_id = ngpu.HostId()

    # Create a host group involving all participating MPI processes
    host_list = list(range(mpi_np))
    hg = ngpu.CreateHostGroup(host_list)

    # Model parameters
    NE = 9000
    NI = 2250
    model_params = {
        'E_L': 0.0,
        'C_m': 250.0,
        'tau_m': 10.0,
        't_ref': 0.5,
        'Theta_rel': 20.0,
        'V_reset_rel': 0.0,
        'tau_syn_ex': 0.32,
        'tau_syn_in': 0.32,
        'V_m_rel': 0.0
    }

    # Instantiate populations across the different ranks
    neurons = []
    E_pops = []
    I_pops = []

    for i in range(mpi_np):
        # RemoteCreate instantiates the populations on the designated MPI rank
        node_seq = ngpu.RemoteCreate(i, 'iaf_psc_alpha', NE + NI, 1, model_params).node_seq
        neurons.append(node_seq)
        E_pops.append(node_seq[0:NE])
        I_pops.append(node_seq[NE:NE+NI])

    CE = 11250
    syn_dict_ex = {'weight': 0.14, 'delay': 1.5}

    # Connect the populations collectively using the host group (hg)
    # This distributes the fixed indegree connections across the processes
    ngpu.ConnectDistributedFixedIndegree(
        host_list,       # Source hosts list
        E_pops,          # Source populations list
        host_list,       # Target hosts list
        neurons,         # Target populations list
        CE,              # Indegree
        hg,              # Host group (collective communicator)
        syn_dict_ex      # Synapse dictionary
    )


While this example demonstrates ``ConnectDistributedFixedIndegree``, other connection rules can also be adopted.

References
----------

[1] Golosio B, Villamar J, Tiddia G, Pastorelli E, Stapmanns J, Fanti V, Paolucci PS, Morrison A and Senk J. (2023) Runtime Construction of Large-Scale Spiking Neuronal Network Models on GPU Devices. Applied Sciences; 13(17):9598. doi: https://doi.org/10.3390/app13179598 

[2] Golosio B, Tiddia G, Villamar J, Pontisso L, Sergi L, Simula F, Babu P, Pastorelli E, Morrison A, Lonardo A, Paolucci PS and Senk J. (2026) Scalable construction of spiking neural networks using up to thousands of GPUs. Neuromorph. Comput. Eng. 6 024012. doi: https://doi.org/10.1088/2634-4386/ae65d2