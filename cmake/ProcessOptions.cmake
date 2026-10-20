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


function( NESTGPU_PROCESS_CUDA_ARCH )
  if( with-gpu-arch )

    if ( NOT "${with-gpu-arch}" MATCHES "^[0-9]+" )
      printError( "Incorrect CUDA architecture" )
    endif ()

    set( CMAKE_CUDA_ARCHITECTURES ${with-gpu-arch} PARENT_SCOPE )

  else()

    set( CMAKE_CUDA_ARCHITECTURES 80 PARENT_SCOPE )

  endif()
endfunction ()


function( NEST_PROCESS_WITH_MPI )
  # Find MPI
  set( HAVE_MPI OFF PARENT_SCOPE )

  string( TOUPPER "${with-mpi}" WITHMPI )
  if( NOT WITHMPI STREQUAL "OFF" )

    if( NOT WITHMPI STREQUAL "ON" )
      # if set, use this prefix
      set( MPI_ROOT "${with-mpi}" )
    endif ()

    find_package( MPI REQUIRED COMPONENTS CXX )

    if( MPI_CXX_FOUND )
      set( HAVE_MPI ON PARENT_SCOPE )

      # export found variables to parent scope
      set( MPI_CXX_FOUND "${MPI_CXX_FOUND}" PARENT_SCOPE )
      set( MPI_CXX_VERSION "${MPI_CXX_VERSION}" PARENT_SCOPE )
      set( MPI_CXX_COMPILER "${MPI_CXX_COMPILER}" PARENT_SCOPE )
      set( MPI_CXX_COMPILE_OPTIONS "${MPI_CXX_COMPILE_OPTIONS}" PARENT_SCOPE )
      set( MPI_CXX_COMPILE_DEFINITIONS "${MPI_CXX_COMPILE_DEFINITIONS}" PARENT_SCOPE )
      set( MPI_CXX_INCLUDE_DIRS "${MPI_CXX_INCLUDE_DIRS}" PARENT_SCOPE )
      set( MPI_CXX_LINK_FLAGS "${MPI_CXX_LINK_FLAGS}" PARENT_SCOPE )
      set( MPI_CXX_LIBRARIES "${MPI_CXX_LIBRARIES}" PARENT_SCOPE )
      set( MPIEXEC_EXECUTABLE "${MPIEXEC_EXECUTABLE}" PARENT_SCOPE )
      set( MPIEXEC_NUMPROC_FLAG "${MPIEXEC_NUMPROC_FLAG}" PARENT_SCOPE )
      set( MPIEXEC_MAX_NUMPROCS "${MPIEXEC_MAX_NUMPROCS}" PARENT_SCOPE )
      set( MPIEXEC_PREFLAGS "${MPIEXEC_PREFLAGS}" PARENT_SCOPE )
      set( MPIEXEC_POSTFLAGS "${MPIEXEC_POSTFLAGS}" PARENT_SCOPE )
    endif ()

  endif ()

  # Provide a dummy MPI::MPI_CXX if no MPI or if flags explicitly
  # given. Needed to avoid problems where MPI::MPI_CXX is used.
  if ( NOT TARGET MPI::MPI_CXX )
    add_library( MPI::MPI_CXX INTERFACE IMPORTED )
  endif()

endfunction()


function( NEST_PROCESS_WITH_OPENMP )
  # Find OPENMP
  set( HAVE_OMP OFF PARENT_SCOPE )

  string( TOUPPER "${with-openmp}" WITHOMP )
  if ( NOT WITHOMP STREQUAL "OFF" )

    if ( NOT WITHOMP STREQUAL "ON" )
      # if set, use this prefix
      set( OpenMP_ROOT "${with-openmp}" )
    endif ()

    find_package( OpenMP REQUIRED COMPONENTS CXX )

    if ( OpenMP_CXX_FOUND )
      set( HAVE_OMP ON PARENT_SCOPE )

      # export found variables to parent scope
      set( OpenMP_CXX_FOUND "${OpenMP_CXX_FOUND}" PARENT_SCOPE )
      set( OpenMP_CXX_VERSION "${OpenMP_CXX_VERSION}" PARENT_SCOPE )
      set( OpenMP_CXX_INCLUDE_DIRS "${OpenMP_CXX_INCLUDE_DIRS}" PARENT_SCOPE )
      set( OpenMP_CXX_FLAGS "${OpenMP_CXX_FLAGS}" PARENT_SCOPE )
      set( OpenMP_CXX_LIBRARIES "${OpenMP_CXX_LIBRARIES}" PARENT_SCOPE )
    endif ()

  endif ()

  # Provide a dummy OpenMP::OpenMP_CXX if no OpenMP or if flags explicitly
  # given. Needed to avoid problems where OpenMP::OpenMP_CXX is used.
  if ( NOT TARGET OpenMP::OpenMP_CXX )
    add_library( OpenMP::OpenMP_CXX INTERFACE IMPORTED )
  endif()

endfunction()


function( NEST_PROCESS_WITH_LIBLTDL )
  # Find LTDL
  set( HAVE_LIBLTDL OFF PARENT_SCOPE )

  string( TOUPPER "${with-ltdl}" WITHLTDL )
  if ( NOT WITHLTDL STREQUAL "OFF" )

    if ( NOT WITHLTDL STREQUAL "ON" )
      # if set, use this prefix
      set( LTDL_ROOT "${with-ltdl}" )
    endif ()

    find_package( LTDL )

    if ( LTDL_FOUND )
      set( HAVE_LIBLTDL ON PARENT_SCOPE )

      include_directories( ${LTDL_INCLUDE_DIRS} )

      # export found variables to parent scope
      set( LTDL_FOUND ON PARENT_SCOPE )
      set( LTDL_VERSION "${LTDL_VERSION}" PARENT_SCOPE )
      set( LTDL_LIBRARIES "${LTDL_LIBRARIES}" PARENT_SCOPE )
      set( LTDL_INCLUDE_DIRS "${LTDL_INCLUDE_DIRS}" PARENT_SCOPE )
    endif ()

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

    string( TOUPPER "${with-defines}" WITHDEFINES )
    if ( WITHDEFINES STREQUAL "ON" )
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


function( NEST_PROCESS_WITH_CPP_STD )
  if ( with-cpp-std )

    if ( NOT "${with-cpp-std}" MATCHES "^[0-9]+" )
      printError( "Incorrect cpp standard" )
    endif ()

    set( CMAKE_CXX_STANDARD ${with-cpp-std} PARENT_SCOPE )
    set( CMAKE_CXX_STANDARD_REQUIRED ON PARENT_SCOPE )
    set( CMAKE_CUDA_STANDARD ${with-cpp-std} PARENT_SCOPE )
    set( CMAKE_CUDA_STANDARD_REQUIRED ON PARENT_SCOPE )

  else()

    set( CMAKE_CXX_STANDARD 17 PARENT_SCOPE )
    set( CMAKE_CXX_STANDARD_REQUIRED ON PARENT_SCOPE )
    set( CMAKE_CUDA_STANDARD 17 PARENT_SCOPE )
    set( CMAKE_CUDA_STANDARD_REQUIRED ON PARENT_SCOPE )

  endif ()
endfunction()


function( NESTGPU_PROCESS_WITH_MAXRREGCOUNT )
  if( with-maxrregcount )

    if ( NOT "${with-maxrregcount}" MATCHES "^[0-9]+" )
      printError( "Incorrect maxrregcount" )
    endif ()

    add_compile_options(
      $<$<COMPILE_LANGUAGE:CUDA>:--maxrregcount=${with-maxrregcount}>
    )

  endif()
endfunction()


function( NESTGPU_PROCESS_WITH_PTXAS_OPTIONS )
  if ( with-ptxas-options )

    set( CUDA_PTXAS "" )
    string( JOIN "," CUDA_PTXAS ${with-ptxas-options} )
    add_compile_options(
      $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=${CUDA_PTXAS}>
    )

  endif ()
endfunction()


function( NEST_PROCESS_WITH_OPTIMIZE )
  if ( with-optimize )

    string( TOUPPER "${with-optimize}" WITHOPTIMIZE )
    if ( WITHOPTIMIZE STREQUAL "ON" )
      set( with-optimize "-O3;-march=native;-mtune=native" ) # For visibility on local scope
    endif ()

    foreach( flag ${with-optimize} )
      add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:${flag}>
        $<$<COMPILE_LANGUAGE:CUDA>:${flag}>
      )
    endforeach()

    if ( WITHOPTIMIZE STREQUAL "ON" )
      set( with-optimize "-O3;-march=native;-mtune=native" PARENT_SCOPE )
    endif ()

  endif ()
endfunction()


function( NEST_PROCESS_WITH_DEBUG )
  if ( with-debug )

    string( TOUPPER "${with-debug}" WITHDEBUG )
    if ( WITHDEBUG STREQUAL "ON" )
      add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:-g>
        $<$<COMPILE_LANGUAGE:CUDA>:-g>
        $<$<COMPILE_LANGUAGE:CUDA>:-G>
      )
    else()
      foreach( flag ${with-debug} )
        add_compile_options(
          $<$<COMPILE_LANGUAGE:CXX>:${flag}>
          $<$<COMPILE_LANGUAGE:CUDA>:${flag}>
        )
      endforeach()
    endif ()

    if ( WITHDEBUG STREQUAL "ON" )
      set( CXX_DBG "-g" PARENT_SCOPE )
      set( CUDA_DBG "-g;-G" PARENT_SCOPE )
    else()
      set( CXX_DBG "${with-debug}" PARENT_SCOPE )
      set( CUDA_DBG "${with-debug}" PARENT_SCOPE )
    endif ()

  endif ()
endfunction()


function( NEST_PROCESS_WITH_WARNING )
  if ( with-warning )

    string( TOUPPER "${with-warning}" WITHWARNING )
    if ( WITHWARNING STREQUAL "ON" )
      set( with-warning "-Wall" ) # For visibility on local scope
    endif ()

    foreach( flag ${with-warning} )
      add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:${flag}>
      )
    endforeach()

    set( CUDA_WARN "" )
    string( JOIN "," CUDA_WARN ${with-warning} )
    add_compile_options(
      $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=${CUDA_WARN}>
    )

    if ( WITHWARNING STREQUAL "ON" )
      set( with-warning "-Wall" PARENT_SCOPE )
    endif ()

  endif ()
endfunction()


function( NEST_PROCESS_WITH_MPI4PY )
  if ( HAVE_MPI AND HAVE_PYTHON )

    include( FindPythonModule )
    find_python_module( mpi4py )

    if ( HAVE_MPI4PY )
      include_directories( "${PY_MPI4PY}/include" )

      set( MPI4PY_INCLUDE_DIRS "${PY_MPI4PY}/include" PARENT_SCOPE )
    endif ()

  endif ()
endfunction ()


function( NEST_PROCESS_VERSION_SUFFIX )
  if ( with-version-suffix )

    foreach ( flag ${with-version-suffix} )
      set( NEST_GPU_VERSION_SUFFIX "${flag}" PARENT_SCOPE )
    endforeach ()

  endif ()
endfunction()
