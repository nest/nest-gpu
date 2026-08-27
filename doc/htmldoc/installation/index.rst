Install NEST GPU
================

The NEST GPU installation procedure is similar to the one of NEST CPU (compare with :ref:`nest:dev_install`).

Requirements
------------

* Currently NEST GPU supports only NVIDIA GPUs and can only be installed on linux systems.

* To build NEST GPU you need `CMake <https://cmake.org/install>`_ (version 3.17 or higher).
  You also need the `NVIDIA drivers <https://www.nvidia.com/Download/index.aspx?lang=en-us>`_
  for the GPU card installed in your machine and the 
  `NVIDIA CUDA development toolkit <https://developer.nvidia.com/cuda-toolkit>`_.
  If you are using Ubuntu, `here <https://linuxhint.com/install-nvidia-drivers-on-ubuntu/>`_
  you can find a guide to install the NVIDIA drivers for the GPU card on your machine.

* If you want to run parallel simulations with MPI (on per default), you can use `Open MPI <https://www.open-mpi.org/>`_.

* You can obtain the base packages from your system, for example: 

.. code-block:: sh

   sudo apt update

   sudo apt install -y \
   nvidia-cuda-toolkit \
   libomp-dev \
   cmake \
   openmpi-bin \
   openmpi-common \
   libopenmpi-dev

* We recommend to install the following Python packages using a venv environment:

.. code-block:: sh

   python3 -m venv venv-nestgpu
   source venv-nestgpu/bin/activate
   pip install --upgrade pip

   pip install numpy scipy matplotlib mpi4py

NEST GPU installation from source
---------------------------------

* Define the directory where your source code should be available, for example:

.. code-block:: sh

   export SOURCE_DIR=$HOME/repositories/nest-gpu

* Get the source code from GitHub. If you want to use the main branch, run:

.. code-block:: sh 

   git clone git@github.com:nest/nest-gpu.git $SOURCE_DIR

Alternatively, you can checkout a specific release or also download a version as a tarball from https://github.com/nest/nest-gpu/releases and unpack it:

.. code-block:: sh

    tar -xzvf nest-gpu-x.tar.gz -C $SOURCE_DIR

* Define the name of your installation, and provide the paths to your preferred build and install directories, then change to the build directory:

.. code-block:: sh

   export NAME=nest-gpu-x
   export BUILD_DIR=$HOME/software/nest-gpu/$NAME/build
   export INSTALL_DIR=$HOME/software/nest-gpu/$NAME/install

   mkdir -pv $BUILD_DIR
   cd $BUILD_DIR

* Configure NEST GPU. For additional ``cmake`` options see the :doc:`CMake Options <cmake_options>`, but for a default installation just run:

.. code-block:: sh

   cmake -DCMAKE_INSTALL_PREFIX:PATH=$INSTALL_DIR $SOURCE_DIR

* Compile and install NEST GPU - note that you can speed up make by using as many jobs as you have cores available (e.g., ``make -j 8``):

.. code-block:: sh

    make
    make install

* NEST GPU should now be successfully installed on your system.

Environment variables
---------------------

* To find the installation, the following environmental variables are defined in ``nestgpu_vars.sh``:

.. list-table::
   :header-rows: 1
   :widths: 10 30

   * - Variable
     - Description
   * - ``PYTHONPATH``
     - Search path for non-standard Python module locations. Will be newly set or prepended to the already existing
       variable if it is already set.
   * - ``NESTGPU_LIB``
     - Path to the shared object file of the NEST GPU library.

* You can set the environment variables in your active shell session:

.. code-block:: sh

   source $INSTALL_DIR/bin/nestgpu_vars.sh

* You may want to include this line in your ``.bashrc`` file, so that the environment variables
  are set automatically whenever you open a new terminal.
  If you installed using a venv environment, make sure that this is also loaded.

* If Python does not find the ``nestgpu`` module, your path variables may not be set correctly.
  This may also be the case if Python cannot load the ``nestgpu`` module due to missing or
  incompatible libraries.

Installation tests
------------------

* To quickly check if Python can load the ``nestgpu`` module run:

.. code-block:: sh

   python3 -c "import nestgpu"

* To check the correctness of the NEST GPU installation more in depth you can find some tests in the directory
  ``SOURCE_DIR/python/test``.
  Each Python script tests a specific feature of the library.
  To run all non-MPI tests you can execute the bash script `test_all.sh`, and for the MPI-related tests run `test_mpi.sh`.
  If everything worked well, for every test you should see a line indicating `TEST PASSED` or `MPI TEST PASSED`.

.. toctree::
   :hidden:

   cmake_options
