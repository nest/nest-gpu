# testsuite/sapi/CMakeLists.txt
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


set( sapi_tests_helper_headers
test_helpers.h
)

set ( sapi_tests_sources
tile2d_grid_gen_test.cpp
node_generation_test.cpp
node_slicing_test.cpp
masks2d_test.cpp
tiles2d_test.cpp
api_conversion_test.cpp
random_test.cpp
mask_tile_processing_test.cpp
grid_neighborhood_test.cpp
mask_node_processing_test.cpp
coordinate_test.cpp
tile_creators_test.cpp
cg_registry_test.cpp
bounding_box2d_test.cpp
mask_array_test.cpp
mask_registry_test.cpp
node_creation_test.cpp
c_api_test.cpp
spatial_manager_test.cpp
node_distribution_test.cpp
tile2d_indexing_test.cpp
numeric_functors_test.cpp
repeatability_test.cpp
cg_array_test.cpp
spatial_communication_test.cpp
node_insertion_test.cpp
node_sequence_jointure_test.cpp
)

set( sapi_tests )

foreach( source IN LISTS sapi_tests_sources )
    get_filename_component(target "${source}" NAME_WE)

    add_executable(
        "${target}"
        "${source}"
        ${sapi_tests_helper_headers}
    )

    set_target_properties( "${target}" PROPERTIES POSITION_INDEPENDENT_CODE ON )

    if ( with-static-libc )
        target_link_options( "${target}" PRIVATE
            -static-libstdc++ -static-libgcc )
    endif()

    target_link_libraries( "${target}" PRIVATE
        sapi_kernel
        m
        MPI::MPI_CXX
        OpenMP::OpenMP_CXX
    )

    target_include_directories( "${target}" PRIVATE
        ${PROJECT_SOURCE_DIR}/sapi
        ${PROJECT_SOURCE_DIR}/sapi/thirdparty
        ${PROJECT_BINARY_DIR}/libnestutil
        ${PROJECT_SOURCE_DIR}/cpp/tests/sapi
    )

    list( APPEND sapi_tests "${target}" )
endforeach()

install( TARGETS ${sapi_tests}
    RUNTIME DESTINATION "${PROJECT_BINARY_DIR}/tests/sapi" )

string( REPLACE ";" " " ALL_TESTS "${sapi_tests}")


set( USE_MPI "false" )
if( HAVE_MPI )
    set( USE_MPI "true" )
endif()

set( multi_processing_tests
    node_creation_test
    node_distribution_test
    node_slicing_test
    spatial_communication_test
    spatial_manager_test
    repeatability_test
    c_api_test
)

string( REPLACE ";" " " MPI_TESTS "${multi_processing_tests}" )

configure_file(
    "${PROJECT_SOURCE_DIR}/testsuite/sapi/run_tests.sh.in"
    "${PROJECT_BINARY_DIR}/tests/sapi/run_tests.sh"
    @ONLY
    NEWLINE_STYLE UNIX
    FILE_PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ
)

add_custom_target( sapicheck
  COMMAND run_tests.sh
  WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/tests/sapi"
  COMMENT "Executing SAPI testsuite..."
  DEPENDS ${sapi_tests}
)
