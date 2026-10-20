/*
 *  comparisons.h
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

#ifndef COMPARISONS_H
#define COMPARISONS_H

#include "mask_containers.h"

namespace sapi {
template <typename CoordT>
bool compare_indexed_node_ptr_maps(
    const FilteredTileNodeCollection<CoordT> &left,
    const FilteredTileNodeCollection<CoordT> &right) {
  if (left.size() != right.size())
    return false;

  for (const auto &left_tile : left) {
    const auto right_tile_it = right.find(left_tile.first);
    if (right_tile_it == right.end())
      return false;

    if (left_tile.second.size() != right_tile_it->second.size())
      return false;

    for (const auto &left_leaf : left_tile.second) {
      const auto right_leaf_it = right_tile_it->second.find(left_leaf.first);
      if (right_leaf_it == right_tile_it->second.end())
        return false;

      if (left_leaf.second.size() != right_leaf_it->second.size())
        return false;

      auto right_view_it = right_leaf_it->second.cbegin();
      for (const auto &left_view : left_leaf.second) {
        if (left_view->first != (*right_view_it)->first ||
            !(left_view->second == (*right_view_it)->second))
          return false;
        ++right_view_it;
      }
    }
  }

  return true;
}
} // namespace sapi

#endif
