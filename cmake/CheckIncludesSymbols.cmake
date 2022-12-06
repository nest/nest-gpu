# cmake/CheckIncludeSymbols.cmake
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

# Here we check for all required include headers, types, symbols and functions.

include( CheckIncludeFiles )
check_include_files( "dlfcn.h" HAVE_DLFCN_H )
check_include_files( "inttypes.h" HAVE_INTTYPES_H )
check_include_files( "limits.h" HAVE_LIMITS_H )
check_include_files( "memory.h" HAVE_MEMORY_H )
check_include_files( "stdint.h" HAVE_STDINT_H )
check_include_files( "stdlib.h" HAVE_STDLIB_H )
check_include_files( "strings.h" HAVE_STRINGS_H )
check_include_files( "string.h" HAVE_STRING_H )
check_include_files( "sys/stat.h" HAVE_SYS_STAT_H )
check_include_files( "sys/types.h" HAVE_SYS_TYPES_H )
check_include_files( "unistd.h" HAVE_UNISTD_H )


find_library( HAVE_LIBM m )
if ( HAVE_LIBM )
    link_libraries( m )
endif ()

include(CheckLibraryExists)
check_library_exists( m pow "" HAVE_POW )

find_package( CUDAToolkit REQUIRED )
link_libraries( cuda )
link_libraries( CUDA::cudart )
link_libraries( CUDA::curand )


# Localize the Python interpreter and ABI
find_package( Python 3.8 QUIET COMPONENTS Interpreter Development.Module )
if ( NOT Python_FOUND )
  find_package( Python 3.8 REQUIRED Interpreter Development )
  string( CONCAT PYABI_WARN "Could not locate Python ABI"
    ", using shared libraries and header file instead."
    " Please clear your CMake cache and build folder and verify that CMake"
    " is up-to-date (3.18+)."
  )
  printWarning("${PYABI_WARN}")
else()
  find_package( Python 3.8 REQUIRED Interpreter Development.Module )
endif()

if ( Python_FOUND )
  if ( CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
    execute_process( COMMAND "${Python_EXECUTABLE}" "-c"
      "import sys, os; print(int(bool(os.environ.get('CONDA_DEFAULT_ENV', False)) or (sys.prefix != sys.base_prefix)))"
      OUTPUT_VARIABLE Python_InVirtualEnv OUTPUT_STRIP_TRAILING_WHITESPACE )

    if ( NOT Python_InVirtualEnv AND CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
      printError( "No virtual Python environment found and no installation prefix specified. "
        "Please either build and install NEST in a virtual Python environment or specify CMake option -DCMAKE_INSTALL_PREFIX=<nest_install_dir>.")
    endif()

    # Setting CMAKE_INSTALL_PREFIX effects the inclusion of GNUInstallDirs defining CMAKE_INSTALL_<dir> and CMAKE_INSTALL_FULL_<dir>
    get_filename_component( Python_EnvRoot "${Python_SITELIB}/../../.." ABSOLUTE)
    set ( CMAKE_INSTALL_PREFIX "${Python_EnvRoot}" CACHE PATH "Default install prefix for the active Python interpreter" FORCE )
  endif ( CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )

  # export found variables to parent scope
  set( HAVE_PYTHON ON )
  set( Python_FOUND "${Python_FOUND}" )
  set( Python_EXECUTABLE ${Python_EXECUTABLE} )
  set( PYTHON ${Python_EXECUTABLE} )
  set( Python_VERSION ${Python_VERSION} )
  set( Python_VERSION_MAJOR ${Python_VERSION_MAJOR} )
  set( Python_VERSION_MINOR ${Python_VERSION_MINOR} )
  set( Python_INCLUDE_DIRS "${Python_INCLUDE_DIRS}" )
  set( Python_LIBRARIES "${Python_LIBRARIES}" )
endif ()

# define python excecution path
function( NEST_POST_PROCESS_WITH_PYTHON )
  if ( Python_FOUND )
    set( PYEXECDIR "${CMAKE_INSTALL_LIBDIR}/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages" PARENT_SCOPE )
  endif()
endfunction()


# Enable dynamic library compiling with run-time search PATH
set( BUILD_SHARED_LIBS ON )

# set RPATH stuff
set( CMAKE_SKIP_RPATH FALSE )
# use, i.e. don't skip the full RPATH for the build tree
set( CMAKE_SKIP_BUILD_RPATH FALSE )

# when building, don't use the install RPATH already
# (but later on when installing)
set( CMAKE_BUILD_WITH_INSTALL_RPATH FALSE )

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
      )

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set( CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE )

# reverse the search order for lib extensions
set( CMAKE_FIND_LIBRARY_SUFFIXES ".so;.dylib;.a;.lib" )


# given a list, filter all header files
function( FILTER_HEADERS in_list out_list )
    if( ${CMAKE_VERSION} VERSION_LESS "3.6" )
        unset( tmp_list )
        foreach( fname ${in_list} )
            if( "${fname}" MATCHES "^.*\\.h(pp)?$" )
                list( APPEND tmp_list "${fname}" )
            endif ()
        endforeach ()
    else ()
        set( tmp_list ${in_list} )
        list( FILTER tmp_list INCLUDE REGEX "^.*\\.h(pp)?$")
    endif ()
    set( ${out_list} ${tmp_list} PARENT_SCOPE )
endfunction ()
