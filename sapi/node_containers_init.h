/*
 *  node_containers_init.h
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

#ifndef NODE_CONTAINERS_INIT_H
#define NODE_CONTAINERS_INIT_H

#include <set>

#include "node_containers.h"
#include "tile.h"
#include "tile_grid.h"

namespace sapi {
template <typename CoordT>
inline void wrapped_split_tile(Tile<CoordT> &tile, const split_t num_splits,
                               const bool generate_total_leaves_vector) {
#pragma omp task default(none) shared(tile)                                    \
    firstprivate(num_splits, generate_total_leaves_vector)
  tile.split(num_splits, generate_total_leaves_vector);
}

template <typename CoordT>
void wrapped_split_position(TilePosition<CoordT> &position,
                            const split_t num_splits,
                            const bool generate_total_leaves_vector) {
  wrapped_split_tile(position.tile_, num_splits, generate_total_leaves_vector);
  for (auto &image : position.tile_images_)
    wrapped_split_tile(image, num_splits, false);
}

template <typename CoordT>
void split_tiles_in_grid(TileGrid<CoordT> &tile_grid,
                         const std::set<tileidx_t> &locally_owned_tiles,
                         const split_t num_splits) {
  static_assert(std::is_unsigned_v<split_t>);
  assert(!tile_grid.positions_.empty() && !tile_grid.has_split_);

#pragma omp parallel default(none) shared(tile_grid, locally_owned_tiles)      \
    firstprivate(num_splits)
#pragma omp master
#pragma omp taskgroup
  for (const auto &owned_tile : locally_owned_tiles) {
    auto &position = tile_grid.positions_.at(owned_tile);
    wrapped_split_position(position, num_splits, true);
    for (const auto &neighbor_index : position.wrapped_tile_neighborhood_)
      if (locally_owned_tiles.find(neighbor_index) == locally_owned_tiles.end())
        wrapped_split_position(tile_grid.positions_.at(neighbor_index),
                               num_splits, false);
  }

  tile_grid.has_split_ = true;
  tile_grid.splits_ = num_splits;
}

template <typename CoordT>
void initialize_local_grid_node_collection(
    GridNodeCollection<CoordT> &grid_collection,
    const std::set<tileidx_t> &locally_owned_tiles,
    const TileGrid<CoordT> &tile_grid) {
  assert(grid_collection.empty() && tile_grid.has_split_);

  if (locally_owned_tiles.empty())
    return;

  const auto num_leaves = tile_grid.positions_.at(*locally_owned_tiles.cbegin())
                              .tile_.leaf_tiles_.size();
  assert(0 < num_leaves);

  grid_collection.resize(tile_grid.num_tiles_);
  const auto gc_it = grid_collection.begin();
  for (const auto &tile_pos : locally_owned_tiles)
    (gc_it + tile_pos)->resize(num_leaves);
}
} // namespace sapi

#endif
