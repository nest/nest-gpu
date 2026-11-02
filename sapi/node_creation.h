/*
 *  node_creation.h
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

#ifndef NODE_CREATION_H
#define NODE_CREATION_H

#include <optional>

#include "grid_neighborhood.h"
#include "node_containers.h"
#include "node_distribution.h"
#include "tile.h"

namespace sapi {
template <typename I,
          typename std::enable_if_t<std::is_integral_v<rcvref<I>>, bool> = true>
constexpr DISTRIBUTION_MODE get_distribution_mode(I &&mode) {
  switch (mode) {
  case uint8_t(DISTRIBUTION_MODE::FREE):
    return DISTRIBUTION_MODE::FREE;
  case uint8_t(DISTRIBUTION_MODE::SQUEEZED):
    return DISTRIBUTION_MODE::SQUEEZED;
  case uint8_t(DISTRIBUTION_MODE::BALANCED):
    return DISTRIBUTION_MODE::BALANCED;
  default:
    throw std::invalid_argument("Invalid distribution mode");
  }
}

template <bool balanced>
void aggregate_tiled_node_count_by_rank(
    NodeCountVector &node_counts_per_rank,
    RankTileIdxNodeCountPairs &tiled_node_counts_per_rank,
    const tileidx_t tile_index, const nodeidx_t node_count_in_tile,
    const GridNeighborhood &grid_neighborhood,
    const RandomManager &rng_manager) {
  assert(0 <= node_count_in_tile);

  if (node_count_in_tile == 0)
    return;

  const auto tile_owners =
      grid_neighborhood.tile_ranks_ownership_map_.cbegin() + tile_index;
  assert(tile_owners != grid_neighborhood.tile_ranks_ownership_map_.end());
  if (tile_owners->empty())
    throw std::runtime_error("Tile with allocated nodes has no owning rank");

  const auto node_counts_per_owning_rank =
      uniform_distribute_node_counts<nodeidx_t, balanced, true>(
          node_count_in_tile, static_cast<tileidx_t>(tile_owners->size()),
          rng_manager);
  assert(node_counts_per_owning_rank.size() == tile_owners->size());

  auto nc_it = node_counts_per_owning_rank.cbegin();
  for (const auto &owner_rank : *tile_owners) {
    const auto count = *nc_it++;
    if (count == 0)
      continue;
    node_counts_per_rank[owner_rank] += count;
    tiled_node_counts_per_rank[owner_rank].emplace_back(
        std::make_pair(tile_index, count));
  }
}

// Here both returned vectors are in size == num processes.
// The first vector contains the total number of nodes
// to be instantiated in each rank.
// The second vector contains the total number of nodes
// to be instantiated per tile owned by each rank.
template <typename I, typename CoordT,
          typename std::enable_if_t<std::is_integral_v<rcvref<I>>, bool> = true>
std::pair<NodeCountVector, RankTileIdxNodeCountPairs>
distribute_node_counts_in_grid(
    const I num_nodes, const std::optional<std::set<tileidx_t>> &tile_set,
    const TileGrid<CoordT> &tile_grid,
    const GridNeighborhood &grid_neighborhood, const RandomManager &rng_manager,
    const DISTRIBUTION_MODE mode) {
  assert(tile_grid.has_split_ && grid_neighborhood.has_owners_ &&
         rng_manager.is_initialized());

  const auto large_distribution =
      std::numeric_limits<nodeidx_t>::max() < num_nodes;

  if (num_nodes < 0)
    throw std::invalid_argument("Cannot distribute a negative number of nodes");
  if (std::numeric_limits<largenodeidx_t>::max() < num_nodes)
    throw std::invalid_argument(
        "Cannot distribute more than " +
        std::to_string(std::numeric_limits<largenodeidx_t>::max()) + " nodes");
  if (large_distribution && tile_set.has_value())
    throw std::invalid_argument("Distribution of a large number of nodes can "
                                "only be done over all tiles");
  if (large_distribution && !grid_neighborhood.rank_tile_bijection_)
    throw std::invalid_argument("Distribution of a large number of nodes can "
                                "only be done over a bijected grid");
  if (large_distribution && mode != DISTRIBUTION_MODE::BALANCED)
    throw std::invalid_argument(
        "Distribution of a large number of nodes can only be done using "
        "balanced distribution mode");
  if (tile_set.has_value() && mode == DISTRIBUTION_MODE::FREE)
    throw std::invalid_argument("Tile specific distribution of nodes cannot be "
                                "done using free distribution mode");

  NodeCountVector node_counts_per_rank(grid_neighborhood.num_processes_, 0);
  RankTileIdxNodeCountPairs tiled_node_counts_per_rank(
      grid_neighborhood.num_processes_);

  if (num_nodes == 0)
    return std::make_pair(std::move(node_counts_per_rank),
                          std::move(tiled_node_counts_per_rank));

#pragma omp parallel default(none)                                             \
    shared(tile_set, tile_grid, grid_neighborhood, rng_manager,                \
               node_counts_per_rank, tiled_node_counts_per_rank)               \
    firstprivate(num_nodes, large_distribution, mode)
#pragma omp master
#pragma omp taskgroup
  {
    NodeCountVector node_counts_per_tile;

    switch (mode) {
    case DISTRIBUTION_MODE::FREE: {
      node_counts_per_tile =
          uniform_distribute_node_counts<nodeidx_t, GridPosition<CoordT>, true>(
              num_nodes, tile_grid.num_tiles_, tile_grid.dimensions_,
              rng_manager);

      break;
    }

    case DISTRIBUTION_MODE::SQUEEZED: {
      node_counts_per_tile =
          uniform_distribute_node_counts<nodeidx_t, false, true>(
              num_nodes,
              tile_set.has_value() ? static_cast<tileidx_t>(tile_set->size())
                                   : tile_grid.num_tiles_,
              rng_manager);

      break;
    }

    default: {
      if (large_distribution)
        node_counts_per_tile =
            uniform_distribute_node_counts<largenodeidx_t, true, true>(
                num_nodes, tile_grid.num_tiles_, rng_manager);
      else
        node_counts_per_tile =
            uniform_distribute_node_counts<nodeidx_t, true, true>(
                num_nodes,
                tile_set.has_value() ? static_cast<tileidx_t>(tile_set->size())
                                     : tile_grid.num_tiles_,
                rng_manager);

      break;
    }
    }

    assert(tile_set.has_value()
               ? node_counts_per_tile.size() == tile_set->size()
               : node_counts_per_tile.size() ==
                     static_cast<std::size_t>(tile_grid.num_tiles_));

    if (grid_neighborhood.rank_tile_bijection_ && !tile_set.has_value()) {
      for (vp_t rank = 0; rank < grid_neighborhood.num_processes_; ++rank) {
        const auto owned_tile =
            *grid_neighborhood.rank_tiles_ownership_map_[rank].cbegin();
        const auto tile_counts = node_counts_per_tile[owned_tile];
        node_counts_per_rank[rank] = tile_counts;
        tiled_node_counts_per_rank[rank].emplace_front(owned_tile, tile_counts);
      }
    } else if (tile_set.has_value()) {
      auto nc_it = node_counts_per_tile.cbegin();
      if (mode == DISTRIBUTION_MODE::BALANCED) {
        for (const auto &tile_index : tile_set.value())
          aggregate_tiled_node_count_by_rank<true>(
              node_counts_per_rank, tiled_node_counts_per_rank, tile_index,
              *nc_it++, grid_neighborhood, rng_manager);
      } else {
        for (const auto &tile_index : tile_set.value())
          aggregate_tiled_node_count_by_rank<false>(
              node_counts_per_rank, tiled_node_counts_per_rank, tile_index,
              *nc_it++, grid_neighborhood, rng_manager);
      }
    } else {
      auto nc_it = node_counts_per_tile.cbegin();
      if (mode == DISTRIBUTION_MODE::BALANCED) {
        for (tileidx_t tile_index = 0; tile_index < tile_grid.num_tiles_;
             ++tile_index)
          aggregate_tiled_node_count_by_rank<true>(
              node_counts_per_rank, tiled_node_counts_per_rank, tile_index,
              *nc_it++, grid_neighborhood, rng_manager);
      } else {
        for (tileidx_t tile_index = 0; tile_index < tile_grid.num_tiles_;
             ++tile_index)
          aggregate_tiled_node_count_by_rank<false>(
              node_counts_per_rank, tiled_node_counts_per_rank, tile_index,
              *nc_it++, grid_neighborhood, rng_manager);
      }
    }
  }

  return std::make_pair(std::move(node_counts_per_rank),
                        std::move(tiled_node_counts_per_rank));
}

template <typename CoordT, bool use_branches, bool balanced>
void distribute_node_counts_in_tiles(
    GridNodeCollection<CoordT> &grid_node_col,
    const TileIdxNodeSequenceMap &node_seq_per_tile,
    const TileGrid<CoordT> &tile_grid, const RandomManager &rng_manager) {
#pragma omp parallel default(none)                                             \
    shared(node_seq_per_tile, grid_node_col, tile_grid, rng_manager)
#pragma omp master
#pragma omp taskgroup
  {
    for (const auto &[tile_index, node_sequence] : node_seq_per_tile) {
      assert(0 <= node_sequence.first && 0 < node_sequence.second);

      const auto tile_nc_it = grid_node_col.begin() + tile_index;
      const auto tile_pos_it = tile_grid.positions_.cbegin() + tile_index;
      assert(tile_nc_it != grid_node_col.end() && !tile_nc_it->empty() &&
             tile_pos_it != tile_grid.positions_.cend() &&
             tile_nc_it->size() == tile_pos_it->tile_.leaf_tiles_.size());

      NodeCountVector node_count_per_sub_tile;

      if constexpr (use_branches) {
        node_count_per_sub_tile =
            uniform_distribute_node_counts<nodeidx_t, std::vector<split_t>,
                                           false>(
                node_sequence.second,
                static_cast<tileidx_t>(tile_nc_it->size()),
                tile_pos_it->tile_.get_possible_sub_tile_branches(
                    tile_grid.splits_),
                rng_manager);
      } else {
        node_count_per_sub_tile =
            uniform_distribute_node_counts<nodeidx_t, balanced, false>(
                node_sequence.second,
                static_cast<tileidx_t>(tile_nc_it->size()), rng_manager);
      }

      assert(node_count_per_sub_tile.size() == tile_nc_it->size());

      auto st_nc_it = tile_nc_it->begin();
      nodeidx_t st_node_index = node_sequence.first;
      auto leaf_it = tile_pos_it->tile_.leaf_tiles_.cbegin();
      for (const auto &node_count : node_count_per_sub_tile) {
        if (0 < node_count) {
#pragma omp task default(none) shared(rng_manager)                             \
    firstprivate(st_nc_it, leaf_it, st_node_index, node_count, tile_index)
          st_nc_it->emplace_back((*leaf_it)->generate_coords_in_tile(
              st_node_index, node_count,
              *rng_manager.reseed_rank_paired_rng(tile_index,
                                                  (*leaf_it)->index_)));

          st_node_index += node_count;
        }

        ++leaf_it;
        ++st_nc_it;
      }
      assert((st_node_index - node_sequence.first) == node_sequence.second);
    }
  }
}

template <typename CoordT>
void distribute_node_counts_in_tiles(
    GridNodeCollection<CoordT> &grid_node_col,
    const TileIdxNodeSequenceMap &node_seq_per_tile,
    const TileGrid<CoordT> &tile_grid, const RandomManager &rng_manager,
    const DISTRIBUTION_MODE mode) {
  assert(tile_grid.has_split_ && !grid_node_col.empty() &&
         rng_manager.is_initialized());

  if (node_seq_per_tile.empty())
    throw std::invalid_argument(
        "Cannot generate nodes with empty node sequences");

  switch (mode) {
  case DISTRIBUTION_MODE::FREE: {
    distribute_node_counts_in_tiles<CoordT, true, false>(
        grid_node_col, node_seq_per_tile, tile_grid, rng_manager);

    break;
  }

  case DISTRIBUTION_MODE::SQUEEZED: {
    distribute_node_counts_in_tiles<CoordT, false, false>(
        grid_node_col, node_seq_per_tile, tile_grid, rng_manager);

    break;
  }

  default: {
    distribute_node_counts_in_tiles<CoordT, false, true>(
        grid_node_col, node_seq_per_tile, tile_grid, rng_manager);

    break;
  }
  }

  rng_manager.update_rank_paired_seed(rng_manager.local_rank_);
}
} // namespace sapi

#endif
