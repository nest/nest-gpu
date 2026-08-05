Welcome to the NEST GPU documentation!
======================================

.. grid::
  :gutter: 2

  .. grid-item::

     .. grid:: 1 1 1 1
       :gutter: 2

       .. grid-item::

           `NEST GPU <https://github.com/nest/nest-gpu>`__ is a GPU library for the simulation of large-scale spiking neural networks.
           It is written in CUDA-C++ and supports multi-GPU simulations through MPI.
           Originally developed under the name NeuronGPU, the code joined the NEST Initiative e.V. and follows the concepts and practices already established by the CPU-based `NEST Simulator <https://nest-simulator.readthedocs.io>`_.
           The Python interface allows users to define neurons, connections, and synapse properties using commands familiar from PyNEST.
           Under the hood, the design mirrors NEST Simulator where applicable, while enabling efficient simulations on the largest GPU-powered computing systems.

           To get started with NEST GPU, install it on your system or browse the latest publications:

           .. grid:: 2

             .. grid-item-card::
                :link-type: doc
                :link: installation/index
                :class-card: nest-button

                Install NEST GPU

             .. grid-item-card::
                :link-type: doc
                :link: publications
                :class-card: nest-button

                NEST GPU publications


  .. grid-item::

     .. grid:: 1 1 1 1
       :gutter: 2

       .. grid-item-card:: Highlights

          .. carousel::
              :show_indicators:
              :show_fade:
              :show_dark:
              :show_captions_below:
              :data-bs-ride: carousel

                .. figure:: static/img/publication_figs/Golosio2026_fig4.jpg
                  :target: publications.html

                  Golosio et al. (2026) Fig. 4

                  Large-scale spiking neural network simulations using up to thousands of GPUs

                .. figure:: static/img/publication_figs/Golosio2023_figA3def.jpg
                  :target: publications.html

                  Golosio et al. (2023) Fig. A3d-f

                  Statistical match of network activity with NEST CPU (here for a cortical microcircuit model)

                .. figure:: static/img/publication_figs/Golosio2026_fig1.jpg
                  :target: publications.html

                  Golosio et al. (2026) Fig. 1

                  Point-to-point and collective communication using MPI for GPU clusters

                .. figure:: static/img/publication_figs/Golosio2026_fig3.jpg
                  :target: publications.html

                  Golosio et al. (2026) Fig. 3

                  Fast onboard network construction and simulation of multi-area model


.. toctree::
   :maxdepth: 1
   :caption: USAGE
   :hidden:

   Install <installation/index>
   Guides <guides/index>
   Examples <examples/index>
   Model Directory <models/index>
   Publications <publications>
   Cite NEST GPU <cite_nest_gpu>
   License <license>

.. toctree::
   :maxdepth: 2
   :caption: COMMUNITY
   :hidden:

   Contact us and contribute <contribute>
   What's new? <whats_new>
   NEST Homepage <https://nest-simulator.org>

.. toctree::
   :caption: RELATED PROJECTS
   :hidden:

   NEST Simulator <https://nest-simulator.readthedocs.io/en/latest/>
   NESTML <https://nestml.readthedocs.io/en/latest/>

..
   Indices and tables
   ==================

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`!