/*
 *  node_slicing.h
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

#ifndef NODE_SLICING_H
#define NODE_SLICING_H

#include <optional>

#include "grid_neighborhood.h"
#include "node_containers.h"

namespace sapi {
// Forward definition to mask.h
template <typename CoordT> struct Mask;

// Forward definition to mask_collection.h
template <typename CoordT> struct MaskCollection;

// Forward definition to thread_aligned_array.h
template <typename T,
          typename std::enable_if_t<std::is_copy_constructible_v<T>, bool>>
class TAArray;

template <typename CoordT>
using FilteringCoordQueue = std::deque<IndexedCoordView<CoordT>>;

typedef DistributedTiledNodeSequenceMap::const_iterator DistTns_IT;
typedef TileIdxNodeSequenceMap::const_iterator Tns_IT;
typedef std::vector<std::pair<tileidx_t, std::optional<Tns_IT>>>
    TiledNodeSequences;

TiledNodeSequences get_optional_tiled_node_sequences(
    const std::optional<DistTns_IT> &opt_dist_tns_it,
    const GridNeighborhood &grid_neighborhood);

template <typename CoordT>
std::deque<const Tile<CoordT> *> inline get_optional_masked_sub_tiles(
    const Tile<CoordT> &tile,
    const MaskCollection<CoordT> *const mask_collection) {
  std::deque<const Tile<CoordT> *> leaf_sub_tiles;

  if (mask_collection->has_source_mask())
    leaf_sub_tiles = mask_collection->source_overlapping_leafs(tile);
  else
    tile.insert_leaf_sub_tiles(leaf_sub_tiles);

  return leaf_sub_tiles;
}

template <typename CoordT, bool ignore_mask>
void check_insert_node_coord_pairs(
    FilteringCoordQueue<CoordT> &node_coord_pairs,
    const IndexedCoordCollection<CoordT> &indexed_coord_col,
    const std::optional<Tns_IT> opt_tns_it, const Mask<CoordT> &mask) {
  static_assert(std::is_trivially_copyable_v<IndexedCoordView<CoordT>>);
  assert(node_coord_pairs.empty() && !indexed_coord_col.empty());

  if (opt_tns_it.has_value()) {
    const auto &filtering_sequence = opt_tns_it.value()->second;
    for (const auto &coord_vec : indexed_coord_col) {
      assert(!coord_vec.empty());

      // Taking into account the fact that vectors within
      // the coord_vec are sorted by coord index and contiguous
      // a quick jointure computation can be performed
      const NodeSequence original_sequence{
          coord_vec.cbegin()->first, static_cast<nodeidx_t>(coord_vec.size())};

      const auto jointure =
          join_sequences(original_sequence, filtering_sequence);

      if (jointure.second < 1)
        continue;

      const auto skip = jointure.first - original_sequence.first;
      assert(0 <= skip && static_cast<std::size_t>(skip + jointure.second) <=
                              coord_vec.size());
      const auto coord_it = coord_vec.begin() + skip;

      if constexpr (ignore_mask) {
        for (nodeidx_t index = 0; index < jointure.second; ++index)
          node_coord_pairs.emplace_back(coord_it + index);
      } else {
        for (nodeidx_t index = 0; index < jointure.second; ++index) {
          if (mask.coord_in_mask((coord_it + index)->second).first)
            node_coord_pairs.emplace_back(coord_it + index);
        }
      }
    }
  } else {
    for (const auto &coord_vec : indexed_coord_col) {
      assert(!coord_vec.empty());

      if constexpr (ignore_mask) {
        for (auto coord_it = coord_vec.cbegin(); coord_it != coord_vec.cend();
             ++coord_it)
          node_coord_pairs.emplace_back(coord_it);
      } else {
        for (auto coord_it = coord_vec.cbegin(); coord_it != coord_vec.cend();
             ++coord_it)
          if (mask.coord_in_mask(coord_it->second).first)
            node_coord_pairs.emplace_back(coord_it);
      }
    }
  }
}

template <typename CoordT>
std::deque<FilteringCoordQueue<CoordT>>
slice_tiled_node_maps(const DistributedTiledNodeSequenceMap *const dist_tns,
                      const TileGrid<CoordT> &tile_grid,
                      const GridNeighborhood &grid_neighborhood,
                      const GridNodeCollection<CoordT> &grid_node_col,
                      const TAArray<MaskCollection<CoordT>, true> &mc_array) {
  assert(tile_grid.has_split_ && grid_neighborhood.has_owners_ &&
         !grid_node_col.empty() && mc_array.is_initialized());

  std::deque<FilteringCoordQueue<CoordT>> tiled_coords;
  std::optional<DistTns_IT> opt_dist_tns_it;
  if (dist_tns != nullptr) {
    assert(!dist_tns->empty());
    auto local_tile_seq_map_it = dist_tns->find(grid_neighborhood.local_rank_);
    if (local_tile_seq_map_it == dist_tns->end())
      return tiled_coords;

    opt_dist_tns_it.emplace(local_tile_seq_map_it);
  }

#pragma omp parallel default(none)                                             \
    shared(tile_grid, grid_neighborhood, grid_node_col, mc_array,              \
               tiled_coords) firstprivate(opt_dist_tns_it)
#pragma omp master
#pragma omp taskgroup
  {
    const auto mask_collection = mc_array.get_local_thread_item();
    for (const auto &[tile_index, opt_tns_it] :
         get_optional_tiled_node_sequences(opt_dist_tns_it,
                                           grid_neighborhood)) {

      const auto tile_nc_it = grid_node_col.cbegin() + tile_index;
      const auto tile_pos_it = tile_grid.positions_.cbegin() + tile_index;
      assert(tile_nc_it != grid_node_col.cend() &&
             tile_pos_it != tile_grid.positions_.cend());

      const auto overlapping_leafs =
          get_optional_masked_sub_tiles(tile_pos_it->tile_, mask_collection);

      if (overlapping_leafs.empty())
        continue;

      for (const auto &leaf_ptr : overlapping_leafs) {
        const auto coord_map_it = tile_nc_it->cbegin() + leaf_ptr->index_;
        assert(coord_map_it != tile_nc_it->cend());

        if (coord_map_it->empty())
          continue;

        const auto leaf_coords = &tiled_coords.emplace_back();

#pragma omp task default(none) shared(mc_array)                                \
    firstprivate(leaf_coords, opt_tns_it, coord_map_it)
        {
          const auto mask_collection = mc_array.get_local_thread_item();
          if (mask_collection->has_source_mask())
            check_insert_node_coord_pairs<CoordT, false>(
                *leaf_coords, *coord_map_it, opt_tns_it,
                mask_collection->source_mask_);
          else
            check_insert_node_coord_pairs<CoordT, true>(
                *leaf_coords, *coord_map_it, opt_tns_it,
                mask_collection->source_mask_);
        }
      }
    }
  }

  return tiled_coords;
}
} // namespace sapi

#endif
