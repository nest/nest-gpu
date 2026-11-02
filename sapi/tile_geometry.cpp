/*
 *  tile_geometry.cpp
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

#include <cmath>

#include "tile_geometry.h"

namespace sapi {
tileidx_t compute_leaves_count(const split_t splits, const TILE_SHAPE shape) {
  switch (shape) {
  case TILE_SHAPE::RECTANGLE:
    return std::pow(4, splits);

  case TILE_SHAPE::TRIANGLE:
    return std::pow(2, splits);

  case TILE_SHAPE::HEXAGON:
    return splits > 0 ? 6 * std::pow(2, splits - 1) : 1;

  default:
    throw std::invalid_argument("Invalid tile shape");
  }
}

std::vector<split_t> get_possible_sub_tile_branches(const split_t splits,
                                                    const TILE_SHAPE shape) {
  if (splits == 0)
    return {};

  switch (shape) {
  case TILE_SHAPE::RECTANGLE:
    return std::vector<split_t>(splits, 4);

  case TILE_SHAPE::TRIANGLE:
    return std::vector<split_t>(splits, 2);

  case TILE_SHAPE::HEXAGON: {
    std::vector<split_t> split_vec(splits, 2);
    split_vec[0] = 6;
    return split_vec;
  }

  default:
    throw std::invalid_argument("Invalid tile shape");
  }
}

split_t compute_minimal_splits(const double expected_total_nodes,
                               const double total_tiles,
                               const double expected_nodes_per_leaf,
                               const TILE_SHAPE shape) {
  if (expected_total_nodes < 1 || total_tiles < 1 ||
      expected_nodes_per_leaf < 1)
    throw std::invalid_argument("Invalid split parameters");

  switch (shape) {
  case TILE_SHAPE::RECTANGLE:
    return split_t(
        std::max(std::floor(std::log(expected_total_nodes /
                                     (expected_nodes_per_leaf * total_tiles)) /
                            std::log(4.)),
                 0.));

  case TILE_SHAPE::TRIANGLE:
    return split_t(
        std::max(std::floor(std::log(expected_total_nodes /
                                     (expected_nodes_per_leaf * total_tiles)) /
                            std::log(2.)),
                 0.));

  case TILE_SHAPE::HEXAGON:
    return split_t(std::max(
        std::floor(std::log(expected_total_nodes /
                            (6. * expected_nodes_per_leaf * total_tiles)) /
                       std::log(2.) +
                   1.),
        0.));

  default:
    throw std::invalid_argument("Invalid tile shape");
  }
}
} // namespace sapi
