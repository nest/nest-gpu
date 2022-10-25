# cmake/ProcessOptions.cmake
#
# This file is part of NEST GPU.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST GPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST GPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST GPU.  If not, see <http://www.gnu.org/licenses/>.

# Here all user defined options will be processed.

function( NEST_PROCESS_STATIC_LIBRARIES )
  # build static or shared libraries
  if ( static-libraries )

    set( BUILD_SHARED_LIBS OFF PARENT_SCOPE )
    # set RPATH stuff
    set( CMAKE_SKIP_RPATH TRUE PARENT_SCOPE )

    # On Linux .a is the static library suffix, on Mac OS X .lib can also
      # be used, so we'll add both to the preference list.
      set( CMAKE_FIND_LIBRARY_SUFFIXES ".a;.lib;.dylib;.so" PARENT_SCOPE )

  else ()
    set( BUILD_SHARED_LIBS ON PARENT_SCOPE )

    # set RPATH stuff
    set( CMAKE_SKIP_RPATH FALSE PARENT_SCOPE )
    # use, i.e. don't skip the full RPATH for the build tree
    set( CMAKE_SKIP_BUILD_RPATH FALSE PARENT_SCOPE )

    # when building, don't use the install RPATH already
    # (but later on when installing)
    set( CMAKE_BUILD_WITH_INSTALL_RPATH FALSE PARENT_SCOPE )

    # set run-time search path (RPATH) so that dynamic libraries in ``lib/nest`` can be located

    # Note: "$ORIGIN" (on Linux) and "@loader_path" (on MacOS) are not CMake variables, but special keywords for the
    # Linux resp. the macOS dynamic loader. They refer to the path in which the object is located, e.g.
    # ``${CMAKE_INSTALL_PREFIX}/bin`` for the nest and sli executables, ``${CMAKE_INSTALL_PREFIX}/lib/nest`` for all
    # dynamic libraries except PyNEST (libnestkernel.so, etc.), and  something like
    # ``${CMAKE_INSTALL_PREFIX}/lib/python3.x/site-packages/nest`` for ``pynestkernel.so``. The RPATH is relative to
    # this origin, so the binary ``bin/nest`` can find the files in the relative location ``../lib/nest``, and
    # similarly for PyNEST and the other libraries. For simplicity, we set all the possibilities on all generated
    # objects.

    # PyNEST can only act as an entry point; it does not need to be included in the other objects' RPATH itself.

    set( CMAKE_INSTALL_RPATH
          # for binaries
          "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/nestgpu"
          # for libraries (except pynestkernel)
          "\$ORIGIN/../../${CMAKE_INSTALL_LIBDIR}/nestgpu"
          # for pynestkernel: origin at <prefix>/lib/python3.x/site-packages/nestgpu
          "\$ORIGIN/../../../nestgpu"
          PARENT_SCOPE )

    # add the automatically determined parts of the RPATH
    # which point to directories outside the build tree to the install RPATH
    set( CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE PARENT_SCOPE )

    # reverse the search order for lib extensions
    set( CMAKE_FIND_LIBRARY_SUFFIXES ".so;.dylib;.a;.lib" PARENT_SCOPE )
  endif ()
endfunction()

function( NEST_POST_PROCESS_WITH_PYTHON )
  if ( Python_FOUND )
    set( PYEXECDIR "${CMAKE_INSTALL_LIBDIR}/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages" PARENT_SCOPE )
  endif()
endfunction()

function( NEST_PROCESS_WITH_OPENMP )
  # Find OPENMP
  if ( with-openmp )
    if ( NOT "${with-openmp}" STREQUAL "ON" )
      printInfo( "Set OpenMP argument: ${with-openmp}")
      # set variables in this scope
      set( OPENMP_FOUND ON )
      set( OpenMP_C_FLAGS "${with-openmp}" )
      set( OpenMP_CXX_FLAGS "${with-openmp}" )
    else ()
      find_package( OpenMP )
    endif ()
    if ( OPENMP_FOUND )
      # export found variables to parent scope
      set( OPENMP_FOUND "${OPENMP_FOUND}" PARENT_SCOPE )
      set( OpenMP_C_FLAGS "${OpenMP_C_FLAGS}" PARENT_SCOPE )
      set( OpenMP_CXX_FLAGS "${OpenMP_CXX_FLAGS}" PARENT_SCOPE )
      # set flags
      set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" PARENT_SCOPE )
      set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" PARENT_SCOPE )
    else()
      printError( "CMake can not find OpenMP." )
    endif ()
  endif ()

  # Provide a dummy OpenMP::OpenMP_CXX if no OpenMP or if flags explicitly
  # given. Needed to avoid problems where OpenMP::OpenMP_CXX is used.
  if ( NOT TARGET OpenMP::OpenMP_CXX )
    add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED)
  endif()

endfunction()

function( NEST_PROCESS_WITH_MPI )
  # Find MPI
  set( HAVE_MPI OFF PARENT_SCOPE )
  if ( with-mpi )
    find_package( MPI REQUIRED )
    if ( MPI_CXX_FOUND )
      set( HAVE_MPI ON PARENT_SCOPE )

      set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${MPI_C_COMPILE_FLAGS}" PARENT_SCOPE )
      set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS}" PARENT_SCOPE )

      set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_CXX_LINK_FLAGS}" PARENT_SCOPE )
      include_directories( ${MPI_CXX_INCLUDE_PATH} )
      # is linked in nestkernel/CMakeLists.txt

      # export found variables to parent scope
      set( MPI_C_FOUND "${MPI_C_FOUND}" PARENT_SCOPE )
      set( MPI_C_COMPILER "${MPI_C_COMPILER}" PARENT_SCOPE )
      set( MPI_C_COMPILE_FLAGS "${MPI_C_COMPILE_FLAGS}" PARENT_SCOPE )
      set( MPI_C_INCLUDE_PATH "${MPI_C_INCLUDE_PATH}" PARENT_SCOPE )
      set( MPI_C_LINK_FLAGS "${MPI_C_LINK_FLAGS}" PARENT_SCOPE )
      set( MPI_C_LIBRARIES "${MPI_C_LIBRARIES}" PARENT_SCOPE )
      set( MPI_CXX_FOUND "${MPI_CXX_FOUND}" PARENT_SCOPE )
      set( MPI_CXX_COMPILER "${MPI_CXX_COMPILER}" PARENT_SCOPE )
      set( MPI_CXX_COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}" PARENT_SCOPE )
      set( MPI_CXX_INCLUDE_PATH "${MPI_CXX_INCLUDE_PATH}" PARENT_SCOPE )
      set( MPI_CXX_LINK_FLAGS "${MPI_CXX_LINK_FLAGS}" PARENT_SCOPE )
      set( MPI_CXX_LIBRARIES "${MPI_CXX_LIBRARIES}" PARENT_SCOPE )
      set( MPIEXEC "${MPIEXEC}" PARENT_SCOPE )
      set( MPIEXEC_NUMPROC_FLAG "${MPIEXEC_NUMPROC_FLAG}" PARENT_SCOPE )
      set( MPIEXEC_PREFLAGS "${MPIEXEC_PREFLAGS}" PARENT_SCOPE )
      set( MPIEXEC_POSTFLAGS "${MPIEXEC_POSTFLAGS}" PARENT_SCOPE )
    endif ()
  endif ()
endfunction()

function( NEST_PROCESS_WITH_MPI4PY )
  if ( HAVE_MPI AND HAVE_PYTHON )
    include( FindPythonModule )
    find_python_module(mpi4py)

    if ( HAVE_MPI4PY )
      include_directories( "${PY_MPI4PY}/include" )
    endif ()

  endif ()
endfunction ()

function( NESTGPU_PROCESS_CUDA_ARCH )
  set( CMAKE_CUDA_ARCHITECTURES ${with-gpu-arch} PARENT_SCOPE )
endfunction ()

function( NEST_PROCESS_VERSION_SUFFIX )
  if ( with-version-suffix )
    foreach ( flag ${with-version-suffix} )
      set( NEST_VERSION_SUFFIX "${flag}" PARENT_SCOPE )
    endforeach ()
  endif ()
endfunction()

# TODO Reimplement for NVCC

#function( NEST_PROCESS_WITH_OPTIMIZE )
#  if ( with-optimize )
#    if ( with-optimize STREQUAL "ON" )
#      set( with-optimize "-O2" )
#    endif ()
#    foreach ( flag ${with-optimize} )
#      set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${flag}" PARENT_SCOPE )
#      set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}" PARENT_SCOPE )
#    endforeach ()
#  endif ()
#endfunction()

#function( NEST_PROCESS_WITH_DEBUG )
#  if ( with-debug )
#    if ( with-debug STREQUAL "ON" )
#      set( with-debug "-g" )
#    endif ()
#    foreach ( flag ${with-debug} )
#      set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${flag}" PARENT_SCOPE )
#      set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}" PARENT_SCOPE )
#    endforeach ()
#  endif ()
#endfunction()

#function( NEST_PROCESS_WITH_STD )
#  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=${with-cpp-std}" PARENT_SCOPE )
#endfunction()

#function( NEST_PROCESS_WITH_WARNING )
#  if ( with-warning )
#    if ( with-warning STREQUAL "ON" )
#      set( with-warning "-Wall" )
#    endif ()
#    foreach ( flag ${with-warning} )
#      set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${flag}" PARENT_SCOPE )
#      set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}" PARENT_SCOPE )
#    endforeach ()
#  endif ()
#endfunction()

#function( NEST_PROCESS_WITH_LIBRARIES )
#  if ( with-libraries )
#    if ( with-libraries STREQUAL "ON" )
#      printError( "-Dwith-libraries requires full library paths." )
#    endif ()
#    foreach ( lib ${with-libraries} )
#      if ( EXISTS "${lib}" )
#        link_libraries( "${lib}" )
#      else ()
#        printError( "Library '${lib}' does not exist!" )
#      endif ()
#    endforeach ()
#  endif ()
#endfunction()

#function( NEST_PROCESS_WITH_INCLUDES )
#  if ( with-includes )
#    if ( with-includes STREQUAL "ON" )
#      printError( "-Dwith-includes requires full paths." )
#    endif ()
#    foreach ( inc ${with-includes} )
#      if ( IS_DIRECTORY "${inc}" )
#        include_directories( "${inc}" )
#      else ()
#        printError( "Include path '${inc}' does not exist!" )
#      endif ()
#    endforeach ()
#  endif ()
#endfunction()

#function( NEST_PROCESS_WITH_DEFINES )
#  if ( with-defines )
#    if ( with-defines STREQUAL "ON" )
#      printError( "-Dwith-defines requires compiler defines -DXYZ=... ." )
#    endif ()
#    foreach ( def ${with-defines} )
#      if ( "${def}" MATCHES "^-D.*" )
#        add_definitions( "${def}" )
#      else ()
#        printError( "Define '${def}' does not match '-D.*' !" )
#      endif ()
#    endforeach ()
#  endif ()
#endfunction()
