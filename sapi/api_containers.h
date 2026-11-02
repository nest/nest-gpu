/*
 *  api_containers.h
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

#ifndef SAPI_CONTAINERS_H
#define SAPI_CONTAINERS_H

#include "garbage_collector.h"
#include "sapi_config.h"

namespace sapi {
template <typename L, typename R> struct PairT {
  L first_;
  R second_;
};

template <typename T0, typename T1, typename T2> struct TripletT {
  T0 first_;
  T1 second_;
  T2 third_;
};

template <typename T> struct ArrayT {
  std::size_t size_ = 0;
  T *array_ = nullptr;

  void resize(const std::size_t size, GC &gc) {
    size_ = size;
    array_ = gc.make_collected<T>(size);
  }
};

template <typename K, typename V> using PairArrayT = ArrayT<PairT<K, V>>;

extern "C" {
struct NodesViewStruct {
  dim_t dimensions_ = 0;
  std::size_t node_count_ = 0;
  const nodeidx_t *indexes_ = nullptr;
  const space_t *coordinates_ = nullptr;
};

struct PositionViewStruct {
  dim_t dimensions_ = 0;
  std::size_t coord_count_ = 0;
  const space_t *coordinates_ = nullptr;
};

struct ConnectionViewStruct {
  std::size_t num_partitions_ = 0;
  std::size_t *partition_sizes_ = nullptr;
  const conn_index_t **sources_ = nullptr;
  const conn_index_t **targets_ = nullptr;
  const conn_param_t **weights_ = nullptr;
  const conn_param_t **delays_ = nullptr;
};

struct ConnectionCountsViewStruct {
  std::size_t incoming_ranks_ = 0;
  std::size_t outgoing_ranks_ = 0;
  const vp_t *source_ranks_;
  const std::size_t *incoming_counts_;
  const vp_t *target_ranks_;
  const std::size_t *outgoing_counts_;
};

struct GridViewStruct {
  dim_t dimensions_ = 0;
  std::size_t num_tiles_ = 0;
  std::size_t leaves_per_tile_ = 0;
  std::size_t vertices_per_tile_ = 0;
  std::size_t vertices_per_leaf_ = 0;

  const tileidx_t *tile_indexes_ = nullptr;
  const space_t *tile_vertices_ = nullptr;
  const space_t *leaf_vertices_ = nullptr;
};

typedef PairT<bool, std::size_t> OptionalIndex;
typedef TripletT<std::size_t, nodeidx_t, nodeidx_t> SpatialNodeSequence;
typedef ArrayT<char> CharArray;
typedef ArrayT<space_t> SpaceTArray;
typedef ArrayT<tileidx_t> TileIdxArray;
typedef ArrayT<angle_t> AngleTArray;

typedef ArrayT<CharArray> NestedCharArray;
typedef ArrayT<SpaceTArray> NestedSpaceTArray;
typedef ArrayT<TileIdxArray> NestedTileIdxArray;

typedef PairArrayT<vp_t, PairArrayT<tileidx_t, PairT<nodeidx_t, nodeidx_t>>>
    TiledNodeSequencePairArray;

typedef PairArrayT<CharArray, double> RankTimerDataPairArray;
typedef PairArrayT<CharArray, ArrayT<double>> ThreadTimerDataPairArray;
typedef PairT<RankTimerDataPairArray, ThreadTimerDataPairArray>
    RecordedTimesArrayPair;

typedef PairArrayT<vp_t, ConnectionViewStruct> ConnectionViewPairArray;
typedef PairT<ConnectionViewPairArray, ConnectionViewPairArray>
    RemoteConnectionViewPair;

typedef PairArrayT<CharArray, NestedCharArray> ParameterNamesPairArray;

struct GPStruct {
  // Grid size and origin
  SpaceTArray grid_origin_;
  TileIdxArray grid_dimensions_;

  // Tile type and size
  CharArray tile_type_;
  SpaceTArray tile_side_lengths_;
  AngleTArray tile_angular_offsets_;

  // Split parameters
  bool compute_splits_ = false;
  split_t num_splits_ = 0;
  nodeidx_t expected_total_nodes_ = 0;
  nodeidx_t expected_nodes_per_leaf_ = 0;
};

struct MPStruct {
  // At least one required
  CharArray mask_blueprint_name_;
  SpaceTArray mask_blueprint_params_;
  SpaceTArray mask_blueprint_offset_;
  CharArray source_mask_name_;
  SpaceTArray source_mask_origin_;
  SpaceTArray source_mask_params_;
  SpaceTArray source_mask_offset_;
  CharArray target_mask_name_;
  SpaceTArray target_mask_origin_;
  SpaceTArray target_mask_params_;
  SpaceTArray target_mask_offset_;
};

struct CPStruct {
  // Control parameters
  bool edge_wrap_ = false;
  bool only_neighborhood_ = false;
  bool allow_multiplicity_ = false;
  bool allow_self_connections_ = false;
  bool partition_connections_ = false;
  count_t connection_counts_ = 0;

  // Connection method
  CharArray rule_;

  // Weight computation
  CharArray weight_df_name_;
  SpaceTArray weight_df_params_;
  NestedCharArray weight_ufs_names_;
  NestedSpaceTArray weight_ufs_params_;

  // Delay computation
  CharArray delay_df_name_;
  SpaceTArray delay_df_params_;
  NestedCharArray delay_ufs_names_;
  NestedSpaceTArray delay_ufs_params_;

  // Probability drawing
  CharArray prob_df_name_;
  SpaceTArray prob_df_params_;
  NestedCharArray prob_ufs_names_;
  NestedSpaceTArray prob_ufs_params_;
};
}
} // namespace sapi

#endif
