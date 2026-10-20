/*
 *  c_api.cpp
 *
 *  This file is part of NEST GPU.
 *
 *  Copyright (C) 2021 The NEST Initiative
 *
 *  NEST GPU is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST GPU is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST GPU.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "c_api.h"
#include "api_converters.h"
#include "spatial_manager.h"

namespace sapi {
static bool INIT_OMP_ONCE = true;

CAPI::CAPI() {
  if (INIT_OMP_ONCE) {
    init_omp(1);
    INIT_OMP_ONCE = false;
  }

  generate_param_name_pair_array(param_name_map_, param_name_gc_);
}

ParameterNamesPairArray *CAPI::get_parameter_names() {
  return &param_name_map_;
}

void CAPI::reset() {
  gc_.free_gc();
  view_gc_.free_gc();
  spatial_storage_.free_gc();
  spatial_manager_ = nullptr;
  seed_ = DEFAULT_BASE_SEED_;
  rng_type_ = DEFAULT_RNG_TYPE_;
}

void CAPI::free_gc() { gc_.free_gc(); }

void CAPI::free_view_gc() { view_gc_.free_gc(); }

vp_t CAPI::get_rank() const { return local_rank_; }

void CAPI::set_rank(const vp_t rank) {
  if (spatial_manager_ != nullptr)
    throw std::runtime_error("API is already initialized");
  if (rank < 0)
    throw std::runtime_error("Invalid MPI rank");

  local_rank_ = rank;
}

vp_t CAPI::get_num_processes() const { return num_processes_; }

void CAPI::set_num_processes(const vp_t num_processes) {
  if (spatial_manager_ != nullptr)
    throw std::runtime_error("API is already initialized");
  if (num_processes < 1)
    throw std::runtime_error("Invalid MPI num processes");

  num_processes_ = num_processes;
}

vp_t CAPI::get_num_threads() const { return get_max_omp_threads(); }

void CAPI::set_num_threads(const vp_t num_threads) {
  set_max_omp_threads(num_threads);
  if (spatial_manager_ != nullptr)
    spatial_manager_->update_num_threads();
}

rng_seed_t CAPI::get_rng_seed() const { return seed_; }

void CAPI::set_rng_seed(const rng_seed_t seed) {
  seed_ = seed;
  if (spatial_manager_ != nullptr)
    spatial_manager_->set_rng_seed(seed);
}

CharArray *CAPI::get_rng_type() { return string_to_charray(rng_type_, gc_); }

void CAPI::set_rng_type(const CharArray &rng_type) {
  rng_type_ = charray_to_string(rng_type);
  if (spatial_manager_ != nullptr)
    spatial_manager_->set_rng_type(rng_type_);
}

void CAPI::generate_tile_grid(const NestedTileIdxArray &rank_tiles_ownership,
                              const GPStruct &grid_parameters) {
  if (spatial_manager_ != nullptr)
    throw std::runtime_error(
        "Cannot initialize spatial grid more than once without resetting api");

  switch (grid_parameters.grid_dimensions_.size_) {
  case 2: {
    auto uptr =
        std::make_unique<SpatialManager<Coord2D>>(local_rank_, num_processes_);
    spatial_manager_ = uptr.get();
    spatial_storage_.collect(std::move(uptr));
    break;
  }

  case 3: {
    auto uptr =
        std::make_unique<SpatialManager<Coord3D>>(local_rank_, num_processes_);
    spatial_manager_ = uptr.get();
    spatial_storage_.collect(std::move(uptr));
    break;
  }

  default:
    throw std::invalid_argument("Invalid grid dimensions");
  }

  if (seed_ != DEFAULT_BASE_SEED_)
    spatial_manager_->set_rng_seed(seed_);

  if (rng_type_ != DEFAULT_RNG_TYPE_)
    spatial_manager_->set_rng_type(rng_type_);

  try {
    spatial_manager_->initialize_tile_grid(
        nested_array_to_set_vector(rank_tiles_ownership),
        gpstruct_to_grid_params(grid_parameters));
  } catch (const std::exception &e) {
    spatial_storage_.free_gc();
    spatial_manager_ = nullptr;
    throw e;
  }
}

NodeCountVector CAPI::generate_nodes_in_grid(const largenodeidx_t num_nodes,
                                             const TileIdxArray &target_tiles,
                                             const uint8_t distribution_mode) {
  if (spatial_manager_ == nullptr)
    throw std::runtime_error("Spatial grid not initialized yet");

  std::optional<std::set<tileidx_t>> set;
  if (target_tiles.size_ > 0)
    set.emplace(array_to_set(target_tiles));

  return spatial_manager_->distribute_nodes_in_grid(
      num_nodes, set, get_distribution_mode(distribution_mode));
}

std::size_t CAPI::generate_nodes_in_tiles(
    const RankNodeSequenceMap &node_sequences_per_rank,
    const uint8_t distribution_mode) {
  if (spatial_manager_ == nullptr)
    throw std::runtime_error("Spatial grid not initialized yet");

  return spatial_manager_->generate_nodes_in_tiles(
      node_sequences_per_rank, get_distribution_mode(distribution_mode));
}

std::pair<NodeCountVector, PositionViewStruct *>
CAPI::insert_positions_in_grid(const PositionViewStruct &positions) {
  if (spatial_manager_ == nullptr)
    throw std::runtime_error("Spatial grid not initialized yet");

  auto [node_counts_per_rank, leftovers] =
      spatial_manager_->insert_positions_in_grid(positions.dimensions_,
                                                 positions.coord_count_,
                                                 positions.coordinates_);

  auto uptr = std::make_unique<std::vector<space_t>>(std::move(leftovers));
  const auto ptr = uptr.get();
  gc_.collect(std::move(uptr));

  return std::make_pair(
      std::move(node_counts_per_rank),
      make_view_from_positions(positions.dimensions_, *ptr, gc_));
}

std::size_t CAPI::insert_positions_in_tiles(
    const RankNodeSequenceMap &node_sequences_per_rank) {
  if (spatial_manager_ == nullptr)
    throw std::runtime_error("Spatial grid not initialized yet");

  return spatial_manager_->insert_positions_in_tiles(node_sequences_per_rank);
}

std::pair<std::size_t, DistributedConnectionInfo *>
CAPI::compute_spatial_connections(const std::size_t source_index,
                                  const std::size_t target_index,
                                  const MPStruct &mask_parameters,
                                  const CPStruct &connection_parameters) {
  if (spatial_manager_ == nullptr)
    throw std::runtime_error("Spatial grid not initialized yet");

  return spatial_manager_->compute_spatial_connections(
      source_index, target_index, mpstruct_to_mask_params(mask_parameters),
      cpstruct_to_conn_params(connection_parameters));
}

NodesViewStruct *CAPI::view_node_positions(const OptionalIndex &index,
                                           const MPStruct &mask_parameters) {
  if (spatial_manager_ == nullptr)
    throw std::runtime_error("Spatial grid not initialized yet");

  std::optional<std::size_t> opt_idx;
  if (index.first_)
    opt_idx.emplace(index.second_);

  auto uptr =
      std::make_unique<IndexedNodeCoordinates>(spatial_manager_->get_nodes(
          opt_idx, mpstruct_to_mask_params(mask_parameters)));
  const auto ptr = uptr.get();
  view_gc_.collect(std::move(uptr));

  return make_view_from_indexed_node_coords(*ptr, gc_);
}

RemoteConnectionViewPair *
CAPI::view_spatial_connections(const std::size_t index) {
  if (spatial_manager_ == nullptr)
    throw std::runtime_error("Spatial grid not initialized yet");

  return make_view_from_distributed_connection_info(
      spatial_manager_->get_connection_map(index), gc_);
}

ConnectionCountsViewStruct *
CAPI::view_connection_counts(const std::size_t index) {
  if (spatial_manager_ == nullptr)
    throw std::runtime_error("Spatial grid not initialized yet");

  auto uptr = std::make_unique<ConnectionCounts>(
      spatial_manager_->get_connection_counts(index));
  const auto ptr = uptr.get();
  view_gc_.collect(std::move(uptr));

  return make_view_from_connection_counts(*ptr, gc_);
}

GridViewStruct *CAPI::view_grid_vertices() {
  if (spatial_manager_ == nullptr)
    throw std::runtime_error("Spatial grid not initialized yet");

  auto uptr =
      std::make_unique<GridVertexMap>(spatial_manager_->get_grid_vertices());
  const auto ptr = uptr.get();
  view_gc_.collect(std::move(uptr));

  return make_view_from_grid_vertex_map(*ptr, gc_);
}

TiledNodeSequencePairArray *
CAPI::get_distributed_node_sequences(const std::size_t index) {
  if (spatial_manager_ == nullptr)
    throw std::runtime_error("Spatial grid not initialized yet");

  return dist_tns_map_to_tns_pair_array(
      spatial_manager_->get_distributed_node_sequence(index), gc_);
}

RecordedTimesArrayPair *CAPI::get_timer_data() {
  if (spatial_manager_ == nullptr)
    throw std::runtime_error("Spatial grid not initialized yet");

  return recorded_times_to_array_pair(spatial_manager_->get_timer_data(), gc_);
}

void CAPI::clear_spatial_connections(const std::size_t index) {
  if (spatial_manager_ == nullptr)
    throw std::runtime_error("Spatial grid not initialized yet");

  spatial_manager_->clear_connection_map(index);
}
} // namespace sapi
