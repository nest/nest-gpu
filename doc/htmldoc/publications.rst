NEST GPU Publications
=====================

2026
----

.. _golosio2026:

.. card::
   :class-card: nest-publication-card

   .. grid:: 1 1 2 2
      :gutter: 3
      :class-row: nest-publication

      .. grid-item::
         :columns: 12 12 5 5

         .. bibliography::
            :list: bullet
            :filter: key == "golosio2026"

      .. grid-item::
         :columns: 12 12 7 7

         .. figure:: static/img/publication_figs/Golosio2026_fig4.jpg

            Golosio et al. (2026) Fig. 4

   .. dropdown:: BibTeX entry

      .. literalinclude:: refs.bib
         :language: latex
         :start-at: @article{Golosio2026,
         :end-before: @article{Potjans2014,

   - Network construction for multi-GPU clusters and upcoming exascale
     supercomputers using MPI
   - Each process builds its local connectivity and prepares the data
     structures for efficient spike exchange across the cluster during state
     propagation
   - Point-to-point communication: network construction of the multi-area
     model :cite:p:`Schmidt2018` more than ten times faster than in
     :ref:`Tiddia et al. (2022) <tiddia2022>`
   - Collective communication: balanced random network scaled up to 1,024
     NVIDIA A100 GPUs (about 230.4 million neurons and :math:`2.59 \times 10^{12}`
     synapses), with network construction in less than a minute

2023
----

.. _golosio2023:

.. card::
   :class-card: nest-publication-card

   .. grid:: 1 1 2 2
      :gutter: 3
      :class-row: nest-publication

      .. grid-item::
         :columns: 12 12 5 5

         .. bibliography::
            :list: bullet
            :filter: key == "golosio2023"

      .. grid-item::
         :columns: 12 12 7 7

         .. figure:: static/img/publication_figs/Golosio2023_fig3a.jpg

            Golosio et al. (2023) Fig. 3a

   .. dropdown:: BibTeX entry

      .. literalinclude:: refs.bib
         :language: latex
         :start-at: @article{Golosio2023,
         :end-before: @article{Golosio2026,

   - New method for creating network connections interactively, dynamically,
     and directly in GPU memory through a set of commonly used high-level
     connection rules :cite:p:`Senk2022`
   - Comparison of different consumer and data-center GPUs
   - Network construction of the cortical microcircuit model
     :cite:p:`Potjans2014` in about 0.5 s; simulation performance result entered constructive community race :cite:p:`Senk2026`
   - Scaling performance tested with a balanced random network on a single
     NVIDIA A100 GPU up to :math:`3 \times 10^{5}` neurons with 10,000
     connections per neuron, limited by GPU memory

2022
----

.. _tiddia2022:

.. card::
   :class-card: nest-publication-card

   .. grid:: 1 1 2 2
      :gutter: 3
      :class-row: nest-publication

      .. grid-item::
         :columns: 12 12 5 5

         .. bibliography::
            :list: bullet
            :filter: key == "tiddia2022"

      .. grid-item::
         :columns: 12 12 7 7

         .. figure:: static/img/publication_figs/Tiddia2022_fig8.jpg

            Tiddia et al. (2022) Fig. 8

   .. dropdown:: BibTeX entry

      .. literalinclude:: refs.bib
         :language: latex
         :start-at: @article{Tiddia2022,

   - Remote spike communication through MPI on a GPU cluster
   - Simulation of the multi-area model of 32 vision-related areas of macaque
     monkey cortex (about 4 million neurons and 24 billion synapses)
     :cite:p:`Schmidt2018`
   - Spiking statistics matched with the NEST simulator
   - 3.1 times (2.4 times) faster than the NEST simulator with the model in
     its metastable (ground) state, running on 32 NVIDIA A100 GPUs

2021
----

.. _golosio2021:

.. card::
   :class-card: nest-publication-card

   .. grid:: 1 1 2 2
      :gutter: 3
      :class-row: nest-publication

      .. grid-item::
         :columns: 12 12 5 5

         .. bibliography::
            :list: bullet
            :filter: key == "golosio2021"

      .. grid-item::
         :columns: 12 12 7 7

         .. figure:: static/img/publication_figs/Golosio2021_fig6a.jpg

            Golosio et al. (2021) Fig. 6a

   .. dropdown:: BibTeX entry

      .. literalinclude:: refs.bib
         :language: latex
         :start-at: @article{Golosio2021,
         :end-before: @article{Golosio2023,

   - First publication of the new GPU library in CUDA-C/C++, tested on a
     single consumer NVIDIA GPU
   - Developed under the name NeuronGPU; the code soon after joined the NEST
     Initiative e.V. and was renamed to NEST GPU
   - Novel spike-delivery algorithm
   - LIF and AdEx neuron models with current- or conductance-based synapses,
     and stimulating and recording devices
   - Match of single-neuron subthreshold dynamics and statistical network
     activity with the NEST simulator
   - Close-to-realtime simulation of the cortical microcircuit model (about
     80,000 neurons and 300 million synapses) :cite:p:`Potjans2014`; simulation performance result entered constructive community race :cite:p:`Senk2026`
   - Simulation of a balanced random network with a million AdEx neurons and
     a thousand connections per neuron

References
----------

.. bibliography::
   :filter: docname in docnames and key not in {"golosio2021", "golosio2023", "golosio2026", "tiddia2022"}
