/*
 *  spatial_containers.h
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

#ifndef SPATIAL_CONTAINERS_H
#define SPATIAL_CONTAINERS_H

#include <string>
#include <unordered_map>
#include <vector>

#include "sapi_config.h"

namespace sapi {
// This structure is used to export tiled node coordinates
// using a contiguous flat coordinates array
struct IndexedNodeCoordinates {
  dim_t dimensions_ = 0;
  std::vector<nodeidx_t> indexes_;
  std::vector<space_t> coordinates_;
};

// This map is used to export tile vertices
struct GridVertexMap {
  dim_t dimensions_ = 0;
  std::size_t num_tiles_ = 0;
  std::size_t leaves_per_tile_ = 0;
  std::size_t vertices_per_tile_ = 0;
  std::size_t vertices_per_leaf_ = 0;

  std::vector<tileidx_t> tile_indexes_;
  std::vector<space_t> tile_vertices_;
  std::vector<space_t> leaf_vertices_;
};

// This map is used to export connection counts
struct ConnectionCounts {
  std::size_t incoming_ranks_ = 0;
  std::size_t outgoing_ranks_ = 0;
  std::vector<vp_t> source_ranks_;
  std::vector<std::size_t> incoming_counts_;
  std::vector<vp_t> target_ranks_;
  std::vector<std::size_t> outgoing_counts_;
};

// ----- Input parameters -----
struct GridParameters {
  // Grid size and origin
  std::vector<space_t> grid_origin_;
  std::vector<tileidx_t> grid_dimensions_;

  // Tile type and size
  std::string tile_type_;
  std::vector<space_t> tile_side_lengths_;
  std::vector<angle_t> tile_angular_offsets_;

  // Split parameters
  bool compute_splits_ = false;
  split_t num_splits_ = 0;
  nodeidx_t expected_total_nodes_ = 0;
  nodeidx_t expected_nodes_per_leaf_ = 0;
};

struct MaskParameters {
  // At least one required
  std::string mask_blueprint_name_;
  std::vector<space_t> mask_blueprint_params_;
  std::vector<space_t> mask_blueprint_offset_;
  std::string source_mask_name_;
  std::vector<space_t> source_mask_origin_;
  std::vector<space_t> source_mask_params_;
  std::vector<space_t> source_mask_offset_;
  std::string target_mask_name_;
  std::vector<space_t> target_mask_origin_;
  std::vector<space_t> target_mask_params_;
  std::vector<space_t> target_mask_offset_;
};

struct ConnectionParameters {
  // Control parameters
  bool edge_wrap_ = false;
  bool only_neighborhood_ = false;
  bool allow_multiplicity_ = false;
  bool allow_self_connections_ = false;
  bool partition_connections_ = false;
  count_t connection_counts_ = 0;

  // Connection generation
  std::string rule_;

  // Weight computation
  std::string weight_df_name_;
  std::vector<space_t> weight_df_params_;
  std::vector<std::string> weight_ufs_names_;
  std::vector<std::vector<space_t>> weight_ufs_params_;

  // Delay computation
  std::string delay_df_name_;
  std::vector<space_t> delay_df_params_;
  std::vector<std::string> delay_ufs_names_;
  std::vector<std::vector<space_t>> delay_ufs_params_;

  // Probability drawing
  std::string prob_df_name_;
  std::vector<space_t> prob_df_params_;
  std::vector<std::string> prob_ufs_names_;
  std::vector<std::vector<space_t>> prob_ufs_params_;
};
} // namespace sapi

#endif
