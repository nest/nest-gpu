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


function( NEST_PROCESS_WITH_LIBLTDL )
  # Only find libLTDL if we link dynamically
  set( HAVE_LIBLTDL OFF PARENT_SCOPE )
  if ( with-ltdl AND NOT static-libraries )
    if ( NOT ${with-ltdl} STREQUAL "ON" )
      # a path is set
      set( LTDL_ROOT_DIR "${with-ltdl}" )
    endif ()

    find_package( LTDL )
    if ( LTDL_FOUND )
      set( HAVE_LIBLTDL ON PARENT_SCOPE )
      # export found variables to parent scope
      set( LTDL_FOUND ON PARENT_SCOPE )
      set( LTDL_LIBRARIES "${LTDL_LIBRARIES}" PARENT_SCOPE )
      set( LTDL_INCLUDE_DIRS "${LTDL_INCLUDE_DIRS}" PARENT_SCOPE )
      set( LTDL_VERSION "${LTDL_VERSION}" PARENT_SCOPE )

      include_directories( ${LTDL_INCLUDE_DIRS} )
      # is linked in nestkernel/CMakeLists.txt
    endif ()
  endif ()
endfunction()


function( NEST_PROCESS_WITH_STD )
  if ( with-cpp-std )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=${with-cpp-std}" PARENT_SCOPE )
    set( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=${with-cpp-std}" PARENT_SCOPE )
  endif ()
endfunction()


function( NESTGPU_PROCESS_WITH_MAX_RREG_COUNT )
  set( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --maxrregcount=${with-max-rreg-count}" PARENT_SCOPE )
endfunction()


function( NESTGPU_PROCESS_WITH_PTXAS_OPTIONS )
  if ( with-ptxas-options )
    set( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=${with-ptxas-options}" PARENT_SCOPE )
  endif ()
endfunction()


function( NEST_PROCESS_WITH_LIBRARIES )
  if ( with-libraries )
    if ( with-libraries STREQUAL "ON" )
      printError( "-Dwith-libraries requires full library paths." )
    endif ()
    foreach ( lib ${with-libraries} )
      if ( EXISTS "${lib}" )
        link_libraries( "${lib}" )
      else ()
        printError( "Library '${lib}' does not exist!" )
      endif ()
    endforeach ()
  endif ()
endfunction()


function( NEST_PROCESS_WITH_INCLUDES )
  if ( with-includes )
    if ( with-includes STREQUAL "ON" )
      printError( "-Dwith-includes requires full paths." )
    endif ()
    foreach ( inc ${with-includes} )
      if ( IS_DIRECTORY "${inc}" )
        include_directories( "${inc}" )
      else ()
        printError( "Include path '${inc}' does not exist!" )
      endif ()
    endforeach ()
  endif ()
endfunction()


function( NEST_PROCESS_WITH_DEFINES )
  if ( with-defines )
    if ( with-defines STREQUAL "ON" )
      printError( "-Dwith-defines requires compiler defines -DXYZ=... ." )
    endif ()
    foreach ( def ${with-defines} )
      if ( "${def}" MATCHES "^-D.*" )
        add_definitions( "${def}" )
      else ()
        printError( "Define '${def}' does not match '-D.*' !" )
      endif ()
    endforeach ()
  endif ()
endfunction()


function( NESTGPU_PRE_PROCESS_COMPILE_FLAGS )
  set( _CUDA_COMPILE_FLAGS "" PARENT_SCOPE )
endfunction()


function( NEST_PROCESS_WITH_OPTIMIZE )
  if ( with-optimize )
    string( TOUPPER "${with-optimize}" WITHOPTIMIZE )
    if ( WITHOPTIMIZE STREQUAL "ON" )
      set( with-optimize "-O3" )
    endif ()
    set( OPTIMIZATION_FLAGS "" )
    string( JOIN " " OPTIMIZATION_FLAGS  ${with-optimize} )
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OPTIMIZATION_FLAGS}" PARENT_SCOPE )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPTIMIZATION_FLAGS}" PARENT_SCOPE )
    set( _CUDA_COMPILE_FLAGS "${_CUDA_COMPILE_FLAGS} ${OPTIMIZATION_FLAGS}" PARENT_SCOPE )
  endif ()
endfunction()


function( NEST_PROCESS_WITH_DEBUG )
  if ( with-debug )
    string( TOUPPER "${with-debug}" WITHDEBUG )
    if ( WITHDEBUG STREQUAL "ON" )
      set( with-debug "-g" )
    endif ()
    set( DEBUG_FLAGS "" )
    string( JOIN " " DEBUG_FLAGS  ${with-debug} )
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${DEBUG_FLAGS}" PARENT_SCOPE )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DEBUG_FLAGS}" PARENT_SCOPE )
    set( _CUDA_COMPILE_FLAGS "${_CUDA_COMPILE_FLAGS} ${DEBUG_FLAGS}" PARENT_SCOPE )
  endif ()
endfunction()


function( NEST_PROCESS_WITH_WARNING )
  if ( with-warning )
    string( TOUPPER "${with-warning}" WITHWARNING )
    if ( WITHWARNING STREQUAL "ON" )
      set( with-warning "-Wall" )
    endif ()
    set( WARNING_FLAGS "" )
    string( JOIN " " WARNING_FLAGS  ${with-warning} )
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${WARNING_FLAGS}" PARENT_SCOPE )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WARNING_FLAGS}" PARENT_SCOPE )
    set( _CUDA_COMPILE_FLAGS "${_CUDA_COMPILE_FLAGS} ${WARNING_FLAGS}" PARENT_SCOPE )
  endif ()
endfunction()


function( NESTGPU_POST_PROCESS_COMPILE_FLAGS )
  set( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  --compiler-options='${_CUDA_COMPILE_FLAGS}'" PARENT_SCOPE )
endfunction()


function( NEST_PROCESS_VERSION_SUFFIX )
  if ( with-version-suffix )
    foreach ( flag ${with-version-suffix} )
      set( NEST_GPU_VERSION_SUFFIX "${flag}" PARENT_SCOPE )
    endforeach ()
  endif ()
endfunction()
