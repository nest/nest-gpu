/*
 *  api_converters.cpp
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

#include "api_converters.h"
#include "enum_store.h"
#include "timer_manager.h"

namespace sapi {
std::vector<std::string>
nested_charray_to_string_vector(const NestedCharArray &cstrings) {
  if (cstrings.size_ < 1)
    return {};
  if (cstrings.array_ == nullptr)
    throw std::invalid_argument("Corrupted NestedCharArray");
  std::vector<std::string> strs(cstrings.size_);
  for (std::size_t idx = 0; idx < cstrings.size_; ++idx)
    strs[idx] = charray_to_string(cstrings.array_[idx]);
  return strs;
}

void copy_to_nested_charray_from_string_vector(
    NestedCharArray &cstrings, const std::vector<std::string> &strings,
    GC &gc) {
  cstrings.resize(strings.size(), gc);
  std::size_t idx = 0;
  for (const auto &str : strings)
    copy_to_charray_from_string(cstrings.array_[idx++], str, gc);
}

void make_view_from_indexed_node_coords(NodesViewStruct &ncs,
                                        const IndexedNodeCoordinates &inc) {
  ncs.dimensions_ = inc.dimensions_;
  ncs.node_count_ = inc.indexes_.size();
  const auto total_coords = inc.coordinates_.size();

  if ((0 < ncs.node_count_) != (0 < ncs.dimensions_) ||
      (total_coords != ncs.dimensions_ * ncs.node_count_))
    throw std::runtime_error("Invalid indexed node coordinates");

  if (0 < ncs.node_count_) {
    ncs.indexes_ = inc.indexes_.data();
    ncs.coordinates_ = inc.coordinates_.data();
  } else {
    ncs.indexes_ = nullptr;
    ncs.coordinates_ = nullptr;
  }
}

void make_view_from_positions(PositionViewStruct &pvs, const dim_t dimensions,
                              const std::vector<space_t> &coordinates) {
  if (dimensions < 2 || 3 < dimensions || coordinates.size() % dimensions != 0)
    throw std::invalid_argument("Invalid coordinates");

  pvs.dimensions_ = dimensions;
  if (!coordinates.empty()) {
    pvs.coord_count_ = coordinates.size();
    pvs.coordinates_ = coordinates.data();
  }
}

void make_view_from_grid_vertex_map(GridViewStruct &cgrid_vertices,
                                    const GridVertexMap &grid_vertices) {
  cgrid_vertices.dimensions_ = grid_vertices.dimensions_;
  cgrid_vertices.num_tiles_ = grid_vertices.num_tiles_;
  cgrid_vertices.leaves_per_tile_ = grid_vertices.leaves_per_tile_;
  cgrid_vertices.vertices_per_tile_ = grid_vertices.vertices_per_tile_;
  cgrid_vertices.vertices_per_leaf_ = grid_vertices.vertices_per_leaf_;

  const auto total_tile_size = cgrid_vertices.num_tiles_ *
                               cgrid_vertices.vertices_per_tile_ *
                               cgrid_vertices.dimensions_;
  const auto total_leaf_size =
      cgrid_vertices.num_tiles_ * cgrid_vertices.leaves_per_tile_ *
      grid_vertices.vertices_per_leaf_ * cgrid_vertices.dimensions_;

  if (total_leaf_size < total_tile_size ||
      grid_vertices.tile_indexes_.size() != cgrid_vertices.num_tiles_ ||
      grid_vertices.tile_vertices_.size() != total_tile_size ||
      grid_vertices.leaf_vertices_.size() != total_leaf_size)
    throw std::runtime_error("Invalid grid vertex map");

  if (0 < total_tile_size) {
    cgrid_vertices.tile_indexes_ = grid_vertices.tile_indexes_.data();
    cgrid_vertices.tile_vertices_ = grid_vertices.tile_vertices_.data();
    cgrid_vertices.leaf_vertices_ = grid_vertices.leaf_vertices_.data();
  } else {
    cgrid_vertices.tile_indexes_ = nullptr;
    cgrid_vertices.tile_vertices_ = nullptr;
    cgrid_vertices.leaf_vertices_ = nullptr;
  }
}

void make_view_from_conn_info_map(
    ConnectionViewPairArray &cvpa,
    const std::unordered_map<vp_t, RankConnectionInfo> &ci_map, GC &gc) {
  cvpa.resize(ci_map.size(), gc);

  std::size_t rix = 0;
  for (const auto &[rank, rci] : ci_map) {
    const auto rcv = &cvpa.array_[rix++];
    rcv->first_ = rank;

    auto &cvs = rcv->second_;
    cvs.num_partitions_ = rci.partitioned_connections_.size();
    cvs.partition_sizes_ = gc.make_collected<std::size_t>(cvs.num_partitions_);
    cvs.sources_ = gc.make_collected<const conn_index_t *>(cvs.num_partitions_);
    cvs.targets_ = gc.make_collected<const conn_index_t *>(cvs.num_partitions_);
    cvs.weights_ = gc.make_collected<const conn_param_t *>(cvs.num_partitions_);
    cvs.delays_ = gc.make_collected<const conn_param_t *>(cvs.num_partitions_);

    std::size_t cix = 0;
    for (const auto &cv : rci.partitioned_connections_) {
      if (cv.sizes_ < 1)
        throw std::runtime_error("Invalid connection vector");

      cvs.partition_sizes_[cix] = cv.sizes_;
      cvs.sources_[cix] = cv.connection_sources_.data();
      cvs.targets_[cix] = cv.connection_targets_.data();
      cvs.weights_[cix] = cv.connection_weights_.data();
      cvs.delays_[cix] = cv.connection_delays_.data();
      ++cix;
    }
  }
}

void make_view_from_connection_counts(ConnectionCountsViewStruct &cc_view,
                                      const ConnectionCounts &cc) {
  if (cc.incoming_ranks_ != cc.source_ranks_.size() ||
      cc.incoming_ranks_ != cc.incoming_counts_.size() ||
      cc.outgoing_ranks_ != cc.target_ranks_.size() ||
      cc.outgoing_ranks_ != cc.outgoing_counts_.size())
    throw std::invalid_argument("Invalid connection counts map");

  if (0 < cc.incoming_ranks_) {
    cc_view.incoming_ranks_ = cc.incoming_ranks_;
    cc_view.source_ranks_ = cc.source_ranks_.data();
    cc_view.incoming_counts_ = cc.incoming_counts_.data();
  }

  if (0 < cc.outgoing_ranks_) {
    cc_view.outgoing_ranks_ = cc.outgoing_ranks_;
    cc_view.target_ranks_ = cc.target_ranks_.data();
    cc_view.outgoing_counts_ = cc.outgoing_counts_.data();
  }
}

void copy_to_tns_pair_array_from_dist_tns_map(
    TiledNodeSequencePairArray &cdtns,
    const DistributedTiledNodeSequenceMap &dtns, GC &gc) {
  cdtns.resize(dtns.size(), gc);

  std::size_t rix = 0;
  for (const auto &[rank, tiled_node_sequence] : dtns) {
    const auto rank_tm_pp = &cdtns.array_[rix++];
    rank_tm_pp->first_ = rank;
    rank_tm_pp->second_.resize(tiled_node_sequence.size(), gc);

    std::size_t tix = 0;
    for (const auto &[tile_index, node_sequence] : tiled_node_sequence) {
      const auto tile_ns_pp = &rank_tm_pp->second_.array_[tix++];
      tile_ns_pp->first_ = tile_index;
      tile_ns_pp->second_.first_ = node_sequence.first;
      tile_ns_pp->second_.second_ = node_sequence.second;
    }
  }
}

void copy_to_timer_data_pair_array_from_timer_data_map(
    RecordedTimesArrayPair &ctimes, const RecordedTimes &times, GC &gc) {
  const auto rank_times_pp = &ctimes.first_;
  rank_times_pp->resize(times.rank_times_.size(), gc);

  std::size_t tix = 0;
  for (const auto &[name, time] : times.rank_times_) {
    const auto timer_data_pp = &rank_times_pp->array_[tix++];
    copy_to_charray_from_string(timer_data_pp->first_, name, gc);
    timer_data_pp->second_ = time;
  }

  const auto thread_times_pp = &ctimes.second_;
  thread_times_pp->resize(times.thread_times_.size(), gc);

  tix = 0;
  for (const auto &[name, times] : times.thread_times_) {
    const auto thread_arr_pp = &thread_times_pp->array_[tix++];
    copy_to_charray_from_string(thread_arr_pp->first_, name, gc);
    copy_to_array_from_collection(thread_arr_pp->second_, times, gc);
  }
}

void generate_param_name_pair_array(ParameterNamesPairArray &pnpa, GC &gc) {
  pnpa.resize(7, gc);

  auto &tile_shapes = pnpa.array_[0];
  copy_to_charray_from_string(tile_shapes.first_, "tile_shapes", gc);
  tile_shapes.second_.resize(uint8_t(TILE_SHAPE::NULL_TS), gc);

  for (uint8_t ts = 0; ts < uint8_t(TILE_SHAPE::NULL_TS); ++ts)
    copy_to_charray_from_string(tile_shapes.second_.array_[ts],
                                TILE_SHAPE_NAMES[ts], gc);

  auto &mask_shapes = pnpa.array_[1];
  copy_to_charray_from_string(mask_shapes.first_, "mask_shapes", gc);
  mask_shapes.second_.resize(uint8_t(MASK_SHAPE::NULL_MS), gc);

  for (uint8_t ms = 0; ms < uint8_t(MASK_SHAPE::NULL_MS); ++ms)
    copy_to_charray_from_string(mask_shapes.second_.array_[ms],
                                MASK_SHAPE_NAMES[ms], gc);

  auto &distribution_modes = pnpa.array_[2];
  copy_to_charray_from_string(distribution_modes.first_, "distribution_modes",
                              gc);
  distribution_modes.second_.resize(uint8_t(DISTRIBUTION_MODE::NULL_DM), gc);

  for (uint8_t dm = 0; dm < uint8_t(DISTRIBUTION_MODE::NULL_DM); ++dm)
    copy_to_charray_from_string(distribution_modes.second_.array_[dm],
                                DISTRIBUTION_MODE_NAMES[dm], gc);

  auto &connection_rules = pnpa.array_[3];
  copy_to_charray_from_string(connection_rules.first_, "connection_rules", gc);
  connection_rules.second_.resize(uint8_t(CONNECTION_RULE::NULL_CM), gc);

  for (uint8_t cm = 0; cm < uint8_t(CONNECTION_RULE::NULL_CM); ++cm)
    copy_to_charray_from_string(connection_rules.second_.array_[cm],
                                CONNECTION_RULE_NAMES[cm], gc);

  auto &unary_functions = pnpa.array_[4];
  copy_to_charray_from_string(unary_functions.first_, "unary_functions", gc);
  unary_functions.second_.resize(uint8_t(UNARY_FUNCTION::NULL_UF), gc);

  for (uint8_t uf = 0; uf < uint8_t(UNARY_FUNCTION::NULL_UF); ++uf)
    copy_to_charray_from_string(unary_functions.second_.array_[uf],
                                UNARY_FUNCTION_NAMES[uf], gc);

  auto &displacement_functions = pnpa.array_[5];
  copy_to_charray_from_string(displacement_functions.first_,
                              "displacement_functions", gc);
  displacement_functions.second_.resize(uint8_t(DISPLACEMENT_FUNCTION::NULL_DF),
                                        gc);

  for (uint8_t df = 0; df < uint8_t(DISPLACEMENT_FUNCTION::NULL_DF); ++df)
    copy_to_charray_from_string(displacement_functions.second_.array_[df],
                                DISPLACEMENT_FUNCTION_NAMES[df], gc);

  auto &random_generators = pnpa.array_[6];
  copy_to_charray_from_string(random_generators.first_, "random_generators",
                              gc);

  if constexpr (std::is_same_v<rng_bits_t, uint32_t>) {
    random_generators.second_.resize(uint8_t(RNG32::NULL_RNG), gc);

    for (uint8_t rng = 0; rng < uint8_t(RNG32::NULL_RNG); ++rng)
      copy_to_charray_from_string(random_generators.second_.array_[rng],
                                  RNG_NAMES_32BIT[rng], gc);
  } else {
    random_generators.second_.resize(uint8_t(RNG64::NULL_RNG), gc);

    for (uint8_t rng = 0; rng < uint8_t(RNG64::NULL_RNG); ++rng)
      copy_to_charray_from_string(random_generators.second_.array_[rng],
                                  RNG_NAMES_64BIT[rng], gc);
  }
}

GridParameters gpstruct_to_grid_params(const GPStruct &gps) {
  GridParameters gp;

  gp.grid_origin_ = array_to_vector(gps.grid_origin_);
  gp.grid_dimensions_ = array_to_vector(gps.grid_dimensions_);
  gp.tile_type_ = charray_to_string(gps.tile_type_);
  gp.tile_side_lengths_ = array_to_vector(gps.tile_side_lengths_);
  gp.tile_angular_offsets_ = array_to_vector(gps.tile_angular_offsets_);
  gp.compute_splits_ = gps.compute_splits_;
  gp.num_splits_ = gps.num_splits_;
  gp.expected_total_nodes_ = gps.expected_total_nodes_;
  gp.expected_nodes_per_leaf_ = gps.expected_nodes_per_leaf_;

  return gp;
}

MaskParameters mpstruct_to_mask_params(const MPStruct &mps) {
  MaskParameters mp;

  mp.mask_blueprint_name_ = charray_to_string(mps.mask_blueprint_name_);
  mp.mask_blueprint_params_ = array_to_vector(mps.mask_blueprint_params_);
  mp.mask_blueprint_offset_ = array_to_vector(mps.mask_blueprint_offset_);
  mp.source_mask_name_ = charray_to_string(mps.source_mask_name_);
  mp.source_mask_origin_ = array_to_vector(mps.source_mask_origin_);
  mp.source_mask_params_ = array_to_vector(mps.source_mask_params_);
  mp.source_mask_offset_ = array_to_vector(mps.source_mask_offset_);
  mp.target_mask_name_ = charray_to_string(mps.target_mask_name_);
  mp.target_mask_origin_ = array_to_vector(mps.target_mask_origin_);
  mp.target_mask_params_ = array_to_vector(mps.target_mask_params_);
  mp.target_mask_offset_ = array_to_vector(mps.target_mask_offset_);

  return mp;
}

ConnectionParameters cpstruct_to_conn_params(const CPStruct &cps) {
  ConnectionParameters cp;

  cp.edge_wrap_ = cps.edge_wrap_;
  cp.only_neighborhood_ = cps.only_neighborhood_;
  cp.allow_multiplicity_ = cps.allow_multiplicity_;
  cp.allow_self_connections_ = cps.allow_self_connections_;
  cp.partition_connections_ = cps.partition_connections_;
  cp.connection_counts_ = cps.connection_counts_;

  cp.rule_ = charray_to_string(cps.rule_);

  cp.weight_df_name_ = charray_to_string(cps.weight_df_name_);
  cp.weight_df_params_ = array_to_vector(cps.weight_df_params_);
  cp.weight_ufs_names_ = nested_charray_to_string_vector(cps.weight_ufs_names_);
  cp.weight_ufs_params_ = nested_array_to_nested_vector(cps.weight_ufs_params_);

  cp.delay_df_name_ = charray_to_string(cps.delay_df_name_);
  cp.delay_df_params_ = array_to_vector(cps.delay_df_params_);
  cp.delay_ufs_names_ = nested_charray_to_string_vector(cps.delay_ufs_names_);
  cp.delay_ufs_params_ = nested_array_to_nested_vector(cps.delay_ufs_params_);

  cp.prob_df_name_ = charray_to_string(cps.prob_df_name_);
  cp.prob_df_params_ = array_to_vector(cps.prob_df_params_);
  cp.prob_ufs_names_ = nested_charray_to_string_vector(cps.prob_ufs_names_);
  cp.prob_ufs_params_ = nested_array_to_nested_vector(cps.prob_ufs_params_);

  return cp;
}
} // namespace sapi
