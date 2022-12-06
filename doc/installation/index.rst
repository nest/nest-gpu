Install NEST GPU
================

Requirements
------------
To build NEST GPU you need `CMake <https://cmake.org/install>`_ (version 3.17 or higher).
You also need the `NVIDIA drivers <https://www.nvidia.com/Download/index.aspx?lang=en-us>`_
for the GPU card installed in your machine and the 
`NVIDIA CUDA development toolkit <https://developer.nvidia.com/cuda-toolkit>`_.
To use the NEST GPU Python interface you need `Python 3 <https://www.python.org/>`_,
`Numpy <https://numpy.org/>`_, `Scipy <https://scipy.org/>`_,
`Matplotlib <https://matplotlib.org/>`_
and `mpi4py <https://mpi4py.readthedocs.io/en/stable/intro.html#>`_
for using MPI from Python.

We also recommend to install the `OpenMP <https://www.openmp.org/>`_ binary and development
packages, which are needed for using MPI.

.. note::

   Currently NEST GPU supports only NVIDIA GPUs and can only be installed
   on linux systems.

Installation instructions
-------------------------

The following instructions describe the steps to compile and install NEST GPU
from source code. To adjust settings please see :doc:`CMake Options <cmake_options>` instructions.

* You can install the mandatory and recommended packages (see `Requirements`_ section) as follows:

.. code-block:: bash

   sudo apt update

   sudo apt install -y \
   nvidia-cuda-toolkit \
   openmpi-bin \
   libopenmpi-dev \
   libomp-dev \
   python3-all-dev \
   python3-numpy \
   python3-scipy \
   python3-matplotlib \
   python3-mpi4py

If you are using Ubuntu, `here <https://linuxhint.com/install-nvidia-drivers-on-ubuntu/>`_
you can find a guide to install the NVIDIA drivers for the GPU card on your machine.

* Download the library (see :doc:`Download <../download/download>` section) and unpack the tarball (if needed):

.. code-block:: sh

    tar -xzvf nest-gpu-x.tar.gz

* Create a build directory:

.. code-block:: sh

    mkdir nest-gpu-x-build

* Change to the build directory:

.. code-block:: sh

    cd nest-gpu-x-build

* Configure NEST GPU. For additional ``cmake`` options see the :doc:`CMake Options <cmake_options>` of this docuentation. 
Without the additional options you can type:

.. code-block:: sh

   cmake -DCMAKE_INSTALL_PREFIX:PATH=<nestgpu_install_dir> <nestgpu_source_dir>

.. note::

   ``nestgpu_install_dir`` should be an absolute path.

* Compile and install NEST GPU:

.. code-block:: sh

    make
    make install

NEST GPU should now be successfully installed on your system.

.. toctree::
   :hidden:

   cmake_options

Environment variables
---------------------

To specify where NEST GPU installation is located you have to use some environment variables.
For your convenience, a shell script setting all required environment variables is provided in
``<nestgpu_install_dir>/bin/nestgpu_vars.sh``. Setting the environment variables in your active
shell session requires sourcing the script:

.. code-block:: sh

   source <nestgpu_install_dir>/bin/nestgpu_vars.sh

You may want to include this line in your ``.bashrc`` file, so that the environment variables
are set automatically whenever you open a new terminal.

The following variables are set in ``nestgpu_vars.sh``:

.. list-table::
   :header-rows: 1
   :widths: 10 30

   * - Variable
     - Description
   * - ``PYTHONPATH``
     - Search path for non-standard Python module locations. Will be newly set or prepended to the already existing
       variable if it is already set.
   * - ``PATH``
     - Search path for binaries. Will be newly set or prepended to the already existing variable if it is already set.

If Python does not find the ``nestgpu`` module, your path variables may not be set correctly.
This may also be the case if Python cannot load the ``nestgpu`` module due to missing or
incompatible libraries.


Installation tests
------------------

To check the correctness of NEST GPU installation you can find some tests in the directory
``<nestgpu_source_dir>/python/test``, where the ``<nestgpu_source_dir>`` is the install path given
to ``cmake``. Each Python script tests a specific feature of the library, and to perform
all the tests you can run the bash scripts `test_all.sh` (which runs all the MPI tests that do
not employ MPI) and `test_mpi.sh`.
If everything worked well, for every test you should see a line indicating `TEST PASSED` or `MPI TEST PASSED`.

If some test did not pass, you can have a look at the `log.txt` file given in output
by the bash scripts to see the output of the Python tests.
