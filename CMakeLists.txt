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

# select parallelization scheme
set( with-gpu-arch "80" CACHE STRING "Specify the GPU compute architecture [default=80]." )
set( with-mpi ON CACHE STRING "Build with MPI parallelization [default=ON]." )
# Disabled by default due to the mishandling of RNGs in multi-threaded network construction.
set( with-openmp OFF CACHE BOOL "Build with OpenMP multi-threading [default=OFF]. Optionally set OMP compiler flags." )

# external libraries
# libltdl not yet needed but will be useful for NESTML
set( with-ltdl OFF CACHE STRING "Build with ltdl library [default=OFF]. To set a specific ltdl, give the  install path. NEST GPU uses ltdl for dynamic loading of external user modules." )

# compiler options; explicit compiler flags can be given.
set( with-optimize ON CACHE STRING "Enable user defined optimizations [default=ON (uses '-O3')]. When OFF, no '-O' flag is passed to the compiler. Separate multiple flags by ';'." )
set( with-warning ON CACHE STRING "Enable user defined warnings [default=ON (uses '-Wall')]. Separate  multiple flags by ';'." )
set( with-debug OFF CACHE STRING "Enable user defined debug flags [default=OFF]. When ON, '-g' is used. Separate  multiple flags by ';'." )

# CUDA options
set( with-cpp-std OFF CACHE STRING "C++ standard to use for compilation [default=OFF]." )
set( with-libraries OFF CACHE STRING "Link additional libraries [default=OFF]. Give full path. Separate multiple libraries by ';'." )
set( with-includes OFF CACHE STRING "Add additional include paths [default=OFF]. Give full path without '-I'. Separate multiple include paths by ';'." )
set( with-defines OFF CACHE STRING "Additional defines, e.g. '-DXYZ=1' [default=OFF]. Separate multiple defines by ';'." )
set( with-max-rreg-count "55" CACHE STRING "Set a maximum amount of register used when compiling [default=55]. Separate multiple flags by ';'." )
set( with-ptxas-options OFF CACHE STRING "Options for ptxas compiling [default=OFF]. Separate multiple flags by ';'." )

# generic build configuration
set( with-version-suffix OFF CACHE STRING "Set a user defined version suffix [default='']." )

################################################################################
##################      Project Directory variables           ##################
################################################################################

# In general use the CMAKE_INSTALL_<dir> and CMAKE_INSTALL_FULL_<dir> vars from
# GNUInstallDirs (included after calling nest_process_with_python()), but the
# CMAKE_INSTALL_DATADIR is usually just CMAKE_INSTALL_DATAROOTDIR
# and we want it to be CMAKE_INSTALL_DATAROOTDIR/PROJECT_NAME
set( CMAKE_INSTALL_DATADIR "share/${PROJECT_NAME}" CACHE STRING "Relative directory, where NEST GPU installs its data (share/nest)" )

################################################################################
##################  Find utility programs (#TODO adapt for later PR)  #########
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
include( ConfigureSummary )
include( GetTriple )

# get triples arch-vendor-os
get_host_triple( NEST_HOST_TRIPLE NEST_HOST_ARCH NEST_HOST_VENDOR NEST_HOST_OS )
get_target_triple( NEST_TARGET_TRIPLE NEST_TARGET_ARCH NEST_TARGET_VENDOR NEST_TARGET_OS )

include( GNUInstallDirs )
nest_post_process_with_python()

nest_process_with_openmp()
nest_process_with_mpi()
nest_process_with_mpi4py()
nestgpu_process_cuda_arch()

nest_process_with_libltdl()

nest_process_with_std()
nest_process_with_libraries()
nest_process_with_includes()
nest_process_with_defines()
nestgpu_process_with_max_rreg_count()
nestgpu_process_with_ptxas_options()

# IMPORTANT: Please try to keep these two function calls to be the last functions to call in this Process step.
#            Adding wrong keywords to the "-Dwith-optmize" flag might cause wrong errors, and therefore to avoid that
#            the step for processing the optmization phase and the debug phase must be the last steps.
nestgpu_pre_process_compile_flags()
nest_process_with_optimize()
nest_process_with_debug()
nest_process_with_warning()
nestgpu_post_process_compile_flags()

nest_process_version_suffix()

################################################################################
##################           Create version string            ##################
################################################################################

include( VersionInfo )
get_version_info()
printInfo("-- NEST GPU version: ${NEST_GPU_VERSION_STRING}")

################################################################################
##################  Enable Testing Targets  (#TODO adapt for later PR) #########
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
if ( NOT CMAKE_BUILD_TYPE OR "${CMAKE_BUILD_TYPE}" STREQUAL "None" )
  set( ALL_CFLAGS "${CMAKE_C_FLAGS}" )
  set( ALL_CXXFLAGS "${CMAKE_CXX_FLAGS}" )
  set( ALL_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}" )
elseif ( ${CMAKE_BUILD_TYPE} STREQUAL "Debug" )
  set( ALL_CFLAGS "${CMAKE_C_FLAGS}   ${CMAKE_C_FLAGS_DEBUG}" )
  set( ALL_CXXFLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}" )
  set( ALL_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_DEBUG}" )
elseif ( ${CMAKE_BUILD_TYPE} STREQUAL "Release" )
  set( ALL_CFLAGS "${CMAKE_C_FLAGS}   ${CMAKE_C_FLAGS_RELEASE}" )
  set( ALL_CXXFLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}" )
  set( ALL_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_RELEASE}" )
elseif ( ${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo" )
  set( ALL_CFLAGS "${CMAKE_C_FLAGS}   ${CMAKE_C_FLAGS_RELWITHDEBINFO}" )
  set( ALL_CXXFLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}" )
  set( ALL_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_RELWITHDEBINFO}" )
elseif ( ${CMAKE_BUILD_TYPE} STREQUAL "MinSizeRel" )
  set( ALL_CFLAGS "${CMAKE_C_FLAGS}   ${CMAKE_C_FLAGS_MINSIZEREL}" )
  set( ALL_CXXFLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_MINSIZEREL}" )
  set( ALL_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_MINSIZEREL}" )
else ()
  printError( "Unknown build type: '${CMAKE_BUILD_TYPE}'" )
endif ()
if ( with-defines )
  foreach ( def ${with-defines} )
    set( ALL_CFLAGS "${def} ${ALL_CFLAGS}" )
    set( ALL_CXXFLAGS "${def} ${ALL_CXXFLAGS}" )
    set( ALL_CUDA_FLAGS "${def} ${ALL_CUDA_FLAGS}" )
  endforeach ()
endif ()

# libraries required to link extension modules
set( MODULE_LINK_LIBS
  "-lnestutil"
  "-lsrc" # to be renamed to -lnestgpukernel
  "-lcuda"
  "-lcudart"
  "-lcurand"
  "${OpenMP_CXX_FLAGS}"
  "${MPI_CXX_LIBRARIES}"
  "${LTDL_LIBRARIES}" )

if ( with-libraries )
  set( MODULE_LINK_LIBS "${MODULE_LINK_LIBS};${with-libraries}" )
endif ()
string( REPLACE ";" " " MODULE_LINK_LIBS "${MODULE_LINK_LIBS}" )

# libraries requied to link NEST
set( ALL_LIBS
  # "-lnestgpu"  # main function is inside src folder, in the future we will separate files following NEST structure
  ${MODULE_LINK_LIBS} )


# all includes
set( ALL_INCLUDES_tmp
  "${CMAKE_INSTALL_FULL_INCLUDEDIR}/nestgpu"
  "${CUDAToolkit_LIBRARY_DIR}"
  "${MPI_CXX_INCLUDE_PATH}"
  "${LTDL_INCLUDE_DIRS}" )
set( ALL_INCLUDES "" )
foreach ( INC ${ALL_INCLUDES_tmp} ${with-includes} )
  if ( INC AND NOT INC STREQUAL "" )
    set( ALL_INCLUDES "${ALL_INCLUDES} -I${INC}" )
  endif ()
endforeach ()
set( ALL_INCLUDES "${ALL_INCLUDES}" )

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

#TODO adapt for later PR
#add_custom_target( install-nodoc
#  COMMAND $(MAKE) NEST_INSTALL_NODOC=true install
#)

nest_print_config_summary()
