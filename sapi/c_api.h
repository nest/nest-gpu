/*
 *  c_api.h
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

#ifndef C_API_H
#define C_API_H

#include <string>

#include "api_containers.h"
#include "connection_containers.h"
#include "node_containers.h"

namespace sapi {
// Forward definition to link with spatial_manager.h
class BaseSpatialManager;

struct CAPI {
  CAPI();
  CAPI(const CAPI &) = delete;
  CAPI(CAPI &&) noexcept = default;
  ~CAPI() noexcept = default;

  CAPI &operator=(const CAPI &) = delete;
  CAPI &operator=(CAPI &&) = delete;

  ParameterNamesPairArray *get_parameter_names();

  void reset();
  void free_gc();
  void free_view_gc();

  vp_t get_rank() const;
  void set_rank(const vp_t rank);

  vp_t get_num_processes() const;
  void set_num_processes(const vp_t num_processes);

  vp_t get_num_threads() const;
  void set_num_threads(const vp_t num_threads);

  rng_seed_t get_rng_seed() const;
  void set_rng_seed(const rng_seed_t seed);

  CharArray *get_rng_type();
  void set_rng_type(const CharArray &rng_type);

  void generate_tile_grid(const NestedTileIdxArray &rank_tiles_ownership,
                          const GPStruct &grid_parameters);

  NodeCountVector generate_nodes_in_grid(const largenodeidx_t num_nodes,
                                         const TileIdxArray &target_tiles,
                                         const uint8_t distribution_mode);

  std::size_t
  generate_nodes_in_tiles(const RankNodeSequenceMap &node_sequences_per_rank,
                          const uint8_t distribution_mode);

  std::pair<NodeCountVector, PositionViewStruct *>
  insert_positions_in_grid(const PositionViewStruct &positions);

  std::size_t
  insert_positions_in_tiles(const RankNodeSequenceMap &node_sequences_per_rank);

  std::pair<std::size_t, DistributedConnectionInfo *>
  compute_spatial_connections(const std::size_t source_index,
                              const std::size_t target_index,
                              const MPStruct &mask_parameters,
                              const CPStruct &connection_parameters);

  NodesViewStruct *view_node_positions(const OptionalIndex &index,
                                       const MPStruct &mask_parameters);

  RemoteConnectionViewPair *view_spatial_connections(const std::size_t index);

  ConnectionCountsViewStruct *view_connection_counts(const std::size_t index);

  GridViewStruct *view_grid_vertices();

  TiledNodeSequencePairArray *
  get_distributed_node_sequences(const std::size_t index);

  RecordedTimesArrayPair *get_timer_data();

  void clear_spatial_connections(const std::size_t index);

private:
  vp_t local_rank_ = 0;
  vp_t num_processes_ = 1;
  rng_seed_t seed_ = DEFAULT_BASE_SEED_;
  std::string rng_type_ = DEFAULT_RNG_TYPE_;
  ParameterNamesPairArray param_name_map_;

  GC gc_;
  GC view_gc_;
  GC spatial_storage_;
  GC param_name_gc_;
  BaseSpatialManager *spatial_manager_ = nullptr;
};
} // namespace sapi

#endif
