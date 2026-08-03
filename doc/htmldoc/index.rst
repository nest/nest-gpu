Welcome to NEST GPU's documentation!
====================================

.. grid::
  :gutter: 2

  .. grid-item::

     .. grid:: 1 1 1 1
       :gutter: 2

       .. grid-item::

          `NEST GPU <https://github.com/nest/nest-gpu>`__ is a GPU-MPI library for simulation of large-scale networks of spiking
          neurons. Can be used in Python, in C++ and in C.

          .. note::

             NEST GPU was developed under the
             name  `NeuronGPU <https://github.com/golosio/NeuronGPU>`__  before
             it has been integrated in the NEST Initiative, see  `Golosio et
             al. (2021) <https://www.frontiersin.org/articles/10.3389/fncom.2021.627620/full>`__.
             Currently this repository is being adapted to the NEST development
             workflow.

          With this library it is possible to run relatively fast simulations of
          large-scale networks of spiking neurons employing GPUs. 
          For instance, on a single NVIDIA GeForce RTX 2080 Ti GPU board it is 
          possible to simulate the activity of 1 million multisynapse AdEx neurons
          with 1000 synapse per neuron in little more than 70 seconds per second
          of neural activity using the fifth-order Runge-Kutta method with adaptive
          stepsize as differential equations solver.
          The MPI communication is also very efficient. The Python interface is
          very similar to that of the NEST simulator: the most used commands are
          practically identical, dictionaries are used to define neurons,
          connections and synapsis properties in the same way.

          To start using it, have a look at the examples in the `python/examples <https://github.com/nest/nest-gpu/tree/main/python/examples>`_
          and `c++/examples <https://github.com/nest/nest-gpu/tree/main/c%2B%2B/examples>`_ folders.

  .. grid-item::

     .. grid:: 1 1 1 1
       :gutter: 2

       .. grid-item-card::

          .. carousel::
              :show_indicators:
              :show_fade:
              :show_dark:
              :data-bs-ride: carousel

                .. figure:: static/img/placeholder_example_1.svg
                  :target: examples/index.html

                  Placeholder caption 1

                .. figure:: static/img/placeholder_example_2.svg
                  :target: examples/index.html

                  Placeholder caption 2

                .. figure:: static/img/placeholder_example_3.svg
                  :target: examples/index.html

                  Placeholder caption 3

                .. figure:: static/img/placeholder_example_4.svg
                  :target: examples/index.html

                  Placeholder caption 4

How to cite us
--------------

If you use NEST GPU in your work, please cite the publications on our :doc:`publication list <publications>`.



.. toctree::
   :maxdepth: 1
   :caption: USAGE
   :hidden:

   Download <download/download>
   Install <installation/index>
   Guides <guides/index>
   Examples <examples/index>
   Model Directory <models/index>
   Publications <publications>

.. toctree::
   :maxdepth: 2
   :caption: COMMUNITY
   :hidden:

   Community <community>

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