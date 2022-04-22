Install NEST GPU
================

Requirements
------------

Mandatory packages:

-  NVIDIA drivers
-  NVIDIA CUDA development toolkit 

Recommended packages:

-  OpenMP binary and development package 

Packages needed for using Python interface:

-  Python 3
-  Python - matplotlib 

Packages needed for using MPI:

-  OpenMPI binary and development package
-  Python - mpi4py (for using MPI from Python)


Installation instructions
-------------------------

.. tabs::

   .. tab:: Generic linux or unix systems

      On a terminal, cd to the folder where you unpacked the tarball, and
      type:

      .. code-block:: bash

         autoreconf -i
         ./configure
         make
         sudo make install

      For support: golosio@unica.it

   .. tab:: Ubuntu

      To install all the necessary packages, open a terminal and type:

      .. code-block:: bash

         sudo apt update
         sudo apt install nvidia-cuda-toolkit openmpi-bin libopenmpi-dev libomp-dev python python-matplotlib python-mpi4py

      Then cd to the folder where you unpacked the tarball, and type:

      .. code-block:: bash

         autoreconf -i
         ./configure
         make
         sudo make install
   
   .. tab:: macOS (High Sierra)

      If you have not installed it yet, you must install clang. Open a
      terminal and type:

      .. code-block:: bash

         xcode-select --install

      If you have not installed it yet, install Homebrew with the command:

      .. code-block:: bash

         /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

      Install the NVIDIA Web Drivers appropriate for your MacOS version/build
      number from:

      https://www.insanelymac.com/forum/topic/324195-nvidia-web-driver-updates-for-macos-high-sierra-update-jan-29-2020/

      Install NVIDIA CUDA development toolkit following the instructions in:

      https://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html

      To verify that the CUDA Drivers and the compiler are installed and
      loaded correctly, in a terminal window use the commands:

      .. code-block:: bash

         kextstat | grep -i cuda
         nvcc -V

      Do not forget to setup CUDA development PATH, by adding the following
      lines to your .bash_profile file

      .. code-block:: bash

         export PATH=/Developer/NVIDIA/CUDA-10.2/bin${PATH:+:${PATH}}
         export DYLD_LIBRARY_PATH=/Developer/NVIDIA/CUDA-10.2/lib\
                                 ${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}

      Install necessary packages:

      .. code-block:: bash

         brew install openmpi mpi4py libomp
         sudo easy_install pip
         pip install matplotlib numpy

      Then cd to the folder where you unpacked the tarball, and type:

      .. code-block:: bash

         autoreconf -i
         ./configure
         make
         sudo make install

Installation test
-----------------

To check the correctness of NEST GPU installation you can find some tests in the directory `python/test <https://github.com/nest/nest-gpu>`_.
Each Python script tests a specific feature of the library, and to perform all the tests you can run the bash scripts `test_all.sh` and `test_mpi.sh`.
If everything worked well, for every test you should see a line indicating `TEST PASSED` or `MPI TEST PASSED`.