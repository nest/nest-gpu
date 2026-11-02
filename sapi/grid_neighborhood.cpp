/*
 *  grid_neighborhood.cpp
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

#include "grid_neighborhood.h"

namespace sapi {
std::string GridNeighborhood::to_string() const {
  assert(has_owners_);
  std::string res = "Tile ranks ownership:\n";
  for (std::size_t tix = 0; tix < tile_ranks_ownership_map_.size(); ++tix) {
    res += "{ " + std::to_string(tix) + " : (";
    for (const auto &rix : tile_ranks_ownership_map_[tix])
      res += " " + std::to_string(rix) + ",";
    res += ") }, ";
  }

  res += "\nRank tiles ownership:\n";
  for (vp_t rix = 0; rix < num_processes_; ++rix) {
    res += "{ " + std::to_string(rix) + " : (";
    for (const auto &tix : rank_tiles_ownership_map_[rix])
      res += " " + std::to_string(tix) + ",";
    res += ") }, ";
  }

  res += "\nRank neighborhood:\n";
  for (vp_t rix = 0; rix < num_processes_; ++rix) {
    res += "{ " + std::to_string(rix) + " : (";
    for (const auto &nix : rank_neighbors_map_[rix].second)
      res += " " + std::to_string(nix) + ",";
    res += ") }, ";
  }

  return res;
}
} // namespace sapi
