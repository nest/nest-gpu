CMake Options for NEST GPU
==========================

NEST GPU is installed with ``cmake`` (at least v3.12). In the simplest case, the commands::

    cmake -DCMAKE_INSTALL_PREFIX:PATH=<nestgpu_install_dir> <nestgpu_source_dir>
    make
    make install

should build and install NEST GPU to ``nestgpu_install_dir``, which should be an absolute
path.


Options for configuring NEST
----------------------------

NEST GPU allows for several configuration options for custom builds:

Use Python to build:
    
+-----------------------------------------------+----------------------------------------------------------------+
| ``-Dwith-python=[OFF|ON]``                    | Build with Python. [default=ON]                                |
+-----------------------------------------------+----------------------------------------------------------------+
                            

Set GPU architecture (see `here <https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/>`_
to choose the best NVIDIA GPU architecture for your GPU card):

+-----------------------------------------------+----------------------------------------------------------------+
| ``-Dwith-gpu-arch=[int]``                     | Specify the GPU compute architecture. [default=70]             |
+-----------------------------------------------+----------------------------------------------------------------+

Change parallelization scheme:

+-----------------------------------------------+----------------------------------------------------------------+
| ``-Dwith-mpi=[OFF|ON]``                       | Build with MPI parallelization. [default=ON]                   |
+-----------------------------------------------+----------------------------------------------------------------+
| ``-Dwith-openmp=[OFF|ON|<OpenMP-Flag>]``      | Enable OpenMP multi-threading.                                 |
|                                               | Optionally set OMP compiler flag. [default=ON]                 |
+-----------------------------------------------+----------------------------------------------------------------+

.. warning::

    Currently NEST GPU must be installed with MPI to avoid compilation errors.


Change compilation behavior:

+-----------------------------------------------+----------------------------------------------------------------+
| ``-Dwith-optimize=[OFF|ON|<list;of;flags>]``  | Enable user defined optimizations. When OFF, no '-O' flag is   |
|                                               | passed to the compiler. Separate multiple flags by ';'.        |
|                                               | [default=ON (uses '-O3')]                                      |
+-----------------------------------------------+----------------------------------------------------------------+
| ``-Dwith-warning=[OFF|ON|<list;of;flags>]``   | Enable user defined warnings. Separate multiple flags by ';'.  |
|                                               | [default=ON (uses '-Wall')]                                    |
+-----------------------------------------------+----------------------------------------------------------------+
| ``-Dwith-debug=[OFF|ON|<list;of;flags>]``     | Enable user defined debug flags. Separate multiple flags       |
|                                               | by ';'. [default=OFF, when ON, defaults to '-g']               |
+-----------------------------------------------+----------------------------------------------------------------+
| ``-Dwith-cpp-std=[OFF|ON]``                   | C++ standard to use for compilation. [default=OFF]             |
+-----------------------------------------------+----------------------------------------------------------------+
| ``-Dwith-libraries=<list;of;libraries>``      | Link additional libraries. Give full path.                     |
|                                               | Separate multiple libraries by ';'. [default=OFF]              |
+-----------------------------------------------+----------------------------------------------------------------+
| ``-Dwith-includes=<list;of;includes>``        | Add additional include paths. Give full path without '-I'.     |
|                                               | Separate multiple include paths by ';'. [default=OFF]          |
+-----------------------------------------------+----------------------------------------------------------------+
| ``-Dwith-defines=<list;of;defines>``          | Additional defines, e.g. '-DXYZ=1'.                            |
|                                               | Separate multiple defines by ';'. [default=OFF]                |
+-----------------------------------------------+----------------------------------------------------------------+
| ``-Dwith-version-suffix=[str]``               | Set a user defined version suffix. [default='']                |
+-----------------------------------------------+----------------------------------------------------------------+
| ``-Dwith-max-rreg-count=[int]``               | Set a maximum amount of register used when compiling.          |
|                                               | [default=55]                                                   |
+-----------------------------------------------+----------------------------------------------------------------+
| ``-Dwith-ptxas-options=[<list;of;flags>]``    | Options for ptxas compiling.                                   |
|                                               | Separate multiple flags by ';'. [default='-v']                 |
+-----------------------------------------------+----------------------------------------------------------------+

