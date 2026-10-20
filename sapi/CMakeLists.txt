# sapi/CMakeLists.txt
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

set ( sapi_kernel_sources
nf_creators.cpp
node_slicing.h
tile_geometry.h
api_converters.h
mask_containers.h
split_tree.h
grid_generation.h
sapi_config.h
tile_geometry.cpp
tile2d_geometry.h
timer_manager.h
connection_containers.h
creator_registry.h
c_api.cpp
mask_geometry.h
comparisons.h
bounding_box.h
mask_creators.cpp
node_distribution.h
grid_containers.h
spatial_manager.h
rng_creators.cpp
connection_rules.h
timer_manager.cpp
gf_creators.h
numerics.h
gf2d_geometry.h
node_insertion.h
node_creation.h
grid_functors.h
split_tree.cpp
numeric_functors.cpp
node_containers.h
numeric_functors.h
nf_creators.h
cg_creators.h
spatial_communication.h
connection_generator.h
thread_aligned_array.h
tile2d_geometry.cpp
type_erasure_helpers.h
CMakeLists.txt
grid_neighborhood.h
gf_geometry.h
mask_collection.h
node_distribution.cpp
mask_creators.h
cg_creators.cpp
tile.h
connection_containers.cpp
mask2d_geometry.h
random_manager.h
tile_grid.h
enum_store.h
node_containers_init.h
garbage_collector.h
thirdparty/pcg_extras.hpp
thirdparty/pcg_random.hpp
thirdparty/randutils.hpp
grid_neighborhood.cpp
coordinate_geometry.cpp
random_manager.cpp
payload_preparation.h
coordinates.h
gf_collection.h
rng_creators.h
node_slicing.cpp
api_containers.h
mask_node_processing.h
vp_interface.h
mask2d_geometry.cpp
algebraic_functors.h
mask.h
gf_creators.cpp
coordinate_geometry.h
spatial_containers.h
mask_tile_processing.h
api_converters.cpp
c_api.h
spatial_manager.cpp
)

add_library( sapi_kernel STATIC ${sapi_kernel_sources} )
set_target_properties( sapi_kernel PROPERTIES POSITION_INDEPENDENT_CODE ON )

if ( with-static-libc )
    target_link_options( sapi_kernel PUBLIC -static-libstdc++ -static-libgcc )
endif()

target_link_libraries( sapi_kernel PUBLIC
    m
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
)

target_include_directories( sapi_kernel PUBLIC
    ${PROJECT_SOURCE_DIR}/sapi
    ${PROJECT_SOURCE_DIR}/sapi/thirdparty
    ${PROJECT_BINARY_DIR}/libnestutil
)

FILTER_HEADERS( "${sapi_kernel_sources}" install_headers )
install( FILES ${install_headers}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/nestgpu )
