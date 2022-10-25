# CMakeLists.txt
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
# along with NEST GPU.  If not, see <http://www.gnu.org/licenses/>

cmake_minimum_required( VERSION 3.17 )

# add cmake modules: for all `include(...)` first look here
list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

project( nestgpu LANGUAGES CUDA CXX C )
set( NEST_GPU_USER_EMAIL "users@nest-simulator.org" )

include( ColorMessages )

# check if the given CMAKE_INSTALL_PREFIX is not empty
if("${CMAKE_INSTALL_PREFIX}" STREQUAL "")
  printError("CMAKE_INSTALL_PREFIX cannot be an empty string")
endif()

# handle relative installation prefixes
if( NOT IS_ABSOLUTE ${CMAKE_INSTALL_PREFIX})
    # convert relative path to absolute path
    get_filename_component(absPath ${CMAKE_INSTALL_PREFIX} ABSOLUTE BASE_DIR ${CMAKE_BINARY_DIR})
    set(CMAKE_INSTALL_PREFIX ${absPath})
    printInfo("Relative CMAKE_INSTALL_PREFIX has been converted to absolute path ${CMAKE_INSTALL_PREFIX}")
endif()

################################################################################
##################         All User Defined options           ##################
################################################################################

# use Python to build PyNEST
set( with-python ON )
option( cythonize-pynest "Use Cython to cythonize pynestkernel.pyx [default=ON]. If OFF, PyNEST has to be build from a pre-cythonized pynestkernel.pyx." ON )

# select parallelization scheme
set( with-gpu-arch "70" CACHE STRING "Specify the GPU compute architecture [default=70]." )
set( with-mpi ON CACHE STRING "Build with MPI parallelization [default=ON]." )
set( with-openmp ON CACHE BOOL "Build with OpenMP multi-threading [default=ON]. Optionally set OMP compiler flags." )

# generic build configuration
option( static-libraries "Build static executable and libraries [default=OFF]" OFF )
#set( with-optimize ON CACHE STRING "Enable user defined optimizations [default=ON (uses '-O2')]. When OFF, no '-O' flag is passed to the compiler. Explicit compiler flags can be given; separate multiple flags by ';'." )
#set( with-warning ON CACHE STRING "Enable user defined warnings [default=ON (uses '-Wall')]. Separate  multiple flags by ';'." )
#set( with-debug OFF CACHE STRING "Enable user defined debug flags [default=OFF]. When ON, '-g' is used. Separate  multiple flags by ';'." )
#set( with-cpp-std "c++11" CACHE STRING "C++ standard to use for compilation [default='c++11']." )
#set( with-libraries OFF CACHE STRING "Link additional libraries [default=OFF]. Give full path. Separate multiple libraries by ';'." )
#set( with-includes OFF CACHE STRING "Add additional include paths [default=OFF]. Give full path without '-I'. Separate multiple include paths by ';'." )
#set( with-defines OFF CACHE STRING "Additional defines, e.g. '-DXYZ=1' [default=OFF]. Separate multiple defines by ';'." )
set( with-version-suffix "" CACHE STRING "Set a user defined version suffix [default='']." )
option( position_independent_code "Use position independent compile for linking multiple libraies [default=OFF]" OFF )

################################################################################
##################      Project Directory variables           ##################
################################################################################

# In general use the CMAKE_INSTALL_<dir> and CMAKE_INSTALL_FULL_<dir> vars from
# GNUInstallDirs (included after calling nest_process_with_python()), but the
# CMAKE_INSTALL_DATADIR is usually just CMAKE_INSTALL_DATAROOTDIR
# and we want it to be CMAKE_INSTALL_DATAROOTDIR/PROJECT_NAME
set( CMAKE_INSTALL_DATADIR "share/${PROJECT_NAME}" CACHE STRING "Relative directory, where NEST GPU installs its data (share/nest)" )

################################################################################
##################           Find utility programs            ##################
################################################################################

# needed for pynest test suite
#if ( ${with-python} STREQUAL "ON" )
#  find_program( NOSETESTS NAMES nosetests )
#endif ()

# needed for target doc and fulldoc
#find_package( Doxygen )
#find_program( SED NAMES sed gsed )

################################################################################
##################                Load includes               ##################
################################################################################

# This include checks the symbols, etc.
include( CheckIncludesSymbols )

# These includes publish function names.
include( ProcessOptions )
#include( ConfigureSummary )
include( GetTriple )

# get triples arch-vendor-os
get_host_triple( NEST_HOST_TRIPLE NEST_HOST_ARCH NEST_HOST_VENDOR NEST_HOST_OS )
get_target_triple( NEST_TARGET_TRIPLE NEST_TARGET_ARCH NEST_TARGET_VENDOR NEST_TARGET_OS )

include( GNUInstallDirs )
nest_process_static_libraries()
nest_post_process_with_python()
nest_process_with_openmp()
nest_process_with_mpi()
nest_process_with_mpi4py()
nest_process_version_suffix()
nestgpu_process_cuda_arch()
#nest_process_with_optimize()
#nest_process_with_debug()
#nest_process_with_std()
#nest_process_with_warning()
#nest_process_with_libraries()
#nest_process_with_includes()
#nest_process_with_defines()

################################################################################
##################           Create version string            ##################
################################################################################

include( NestVersionInfo )
get_version_info()
printInfo("-- NEST GPU version: ${NEST_VERSION_STRING}")

################################################################################
##################         Enable Testing Targets             ##################
################################################################################
#enable_testing()
#set( TEST_OPTS "--with-python=${PYTHON}" )

#add_custom_target( installcheck
#  COMMAND ${CMAKE_COMMAND} -E env
#    ${CMAKE_INSTALL_FULL_DATADIR}/testsuite/do_tests.sh
#	--prefix=${CMAKE_INSTALL_PREFIX}
#	--report-dir="${PROJECT_BINARY_DIR}/reports"
#	${TEST_OPTS}
#  WORKING_DIRECTORY "${PROJECT_BINARY_DIR}"
#  COMMENT "Executing NEST's testsuite..."
#)

# N.B. to ensure "make install" is always run before "make installcheck", we
# would ideally like to add:
#   add_dependencies( installcheck install )
# However, an issue in CMake at time of writing (May 2020, see
# https://gitlab.kitware.com/cmake/cmake/-/issues/8438) precludes us from doing
# so.

################################################################################
##################        Define Subdirectories here          ##################
################################################################################

#add_subdirectory( doc )
add_subdirectory( bin )
add_subdirectory( src )
add_subdirectory( libnestutil )
add_subdirectory( pythonlib )

################################################################################
##################           Summary of flags                 ##################
################################################################################

# used in nest-config

# all compiler flags
#if ( NOT CMAKE_BUILD_TYPE OR "${CMAKE_BUILD_TYPE}" STREQUAL "None" )
#  set( ALL_CFLAGS "${CMAKE_C_FLAGS}" )
#  set( ALL_CXXFLAGS "${CMAKE_CXX_FLAGS}" )
#elseif ( ${CMAKE_BUILD_TYPE} STREQUAL "Debug" )
#  set( ALL_CFLAGS "${CMAKE_C_FLAGS}   ${CMAKE_C_FLAGS_DEBUG}" )
#  set( ALL_CXXFLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}" )
#elseif ( ${CMAKE_BUILD_TYPE} STREQUAL "Release" )
#  set( ALL_CFLAGS "${CMAKE_C_FLAGS}   ${CMAKE_C_FLAGS_RELEASE}" )
#  set( ALL_CXXFLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}" )
#elseif ( ${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo" )
#  set( ALL_CFLAGS "${CMAKE_C_FLAGS}   ${CMAKE_C_FLAGS_RELWITHDEBINFO}" )
#  set( ALL_CXXFLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}" )
#elseif ( ${CMAKE_BUILD_TYPE} STREQUAL "MinSizeRel" )
#  set( ALL_CFLAGS "${CMAKE_C_FLAGS}   ${CMAKE_C_FLAGS_MINSIZEREL}" )
#  set( ALL_CXXFLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_MINSIZEREL}" )
#else ()
#  printError( "Unknown build type: '${CMAKE_BUILD_TYPE}'" )
#endif ()
#if ( with-defines )
#  foreach ( def ${with-defines} )
#    set( ALL_CFLAGS "${def} ${ALL_CFLAGS}" )
#    set( ALL_CXXFLAGS "${def} ${ALL_CXXFLAGS}" )
#  endforeach ()
#endif ()

# libraries required to link extension modules
#set( MODULE_LINK_LIBS
#  "-lnestutil"
#  "-lnestgpukernel"
#  "${OpenMP_CXX_FLAGS}"
#  "${MPI_CXX_LIBRARIES}" )
#
#if ( with-libraries )
#  set( MODULE_LINK_LIBS "${MODULE_LINK_LIBS};${with-libraries}" )
#endif ()
#string( REPLACE ";" " " MODULE_LINK_LIBS "${MODULE_LINK_LIBS}" )

# libraries requied to link NEST
#set( ALL_LIBS
#  "-lnest"
#  ${MODULE_LINK_LIBS} )


# all includes
#set( ALL_INCLUDES_tmp
#  "${CMAKE_INSTALL_FULL_INCLUDEDIR}/nest"
#  "${MPI_CXX_INCLUDE_PATH}" )
#set( ALL_INCLUDES "" )
#foreach ( INC ${ALL_INCLUDES_tmp} ${with-includes} )
#  if ( INC AND NOT INC STREQUAL "" )
#    set( ALL_INCLUDES "${ALL_INCLUDES} -I${INC}" )
#  endif ()
#endforeach ()
#set( ALL_INCLUDES "${ALL_INCLUDES} )

################################################################################
##################           File generation here             ##################
################################################################################

configure_file(
    "${PROJECT_SOURCE_DIR}/libnestutil/config.h.in"
    "${PROJECT_BINARY_DIR}/libnestutil/config.h" @ONLY
)

configure_file(
    "${PROJECT_SOURCE_DIR}/bin/nestgpu_vars.sh.in"
    "${PROJECT_BINARY_DIR}/bin/nestgpu_vars.sh" @ONLY
)

################################################################################
##################            Install Extra Files             ##################
################################################################################

install( FILES LICENSE README.md
    DESTINATION ${CMAKE_INSTALL_DOCDIR}
    )

#add_custom_target( install-nodoc
#  COMMAND $(MAKE) NEST_INSTALL_NODOC=true install
#)

#nest_print_config_summary()
