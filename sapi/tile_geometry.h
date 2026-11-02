/*
 *  tile_geometry.h
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

#ifndef TILE_GEOMETRY_H
#define TILE_GEOMETRY_H

#include <stdexcept>
#include <string>

#include "enum_store.h"
#include "tile2d_geometry.h"

namespace sapi {
template <typename CoordT>
void initialize_tile_vertices(std::vector<CoordT> &vertices,
                              CircumscribedRadius<CoordT> &c_radius,
                              const std::vector<space_t> &side_lengths,
                              const std::vector<angle_t> &angular_offsets,
                              const TILE_SHAPE shape) {
  if constexpr (std::is_same_v<CoordT, Coord2D>) {
    if (1 < angular_offsets.size())
      throw std::invalid_argument("Invalid offsets for tile 2D");
    const auto lengths = side_lengths.size();

    switch (shape) {
    case TILE_SHAPE::RECTANGLE: {
      if (lengths < 1 || 2 < lengths)
        throw std::invalid_argument("Invalid rectangle length vector");

      initialize_vertices_2D(vertices, c_radius, side_lengths,
                             angular_offsets.empty() ? 0 : angular_offsets[0],
                             false // triangular vertices
      );

      break;
    }

    case TILE_SHAPE::TRIANGLE: {
      if (lengths < 1 || 2 < lengths)
        throw std::invalid_argument("Invalid triangle length vector");

      initialize_vertices_2D(vertices, c_radius, side_lengths,
                             angular_offsets.empty() ? 0 : angular_offsets[0],
                             true // triangular vertices
      );

      break;
    }

    case TILE_SHAPE::HEXAGON: {
      if (lengths != 1)
        throw std::invalid_argument("Invalid hexagon length vector");

      initialize_hexagon_vertices(vertices, c_radius, side_lengths[0],
                                  angular_offsets.empty() ? 0
                                                          : angular_offsets[0]);

      break;
    }

    default:
      throw std::invalid_argument("Invalid tile shape");
    }
  } else {
    throw std::runtime_error("3D implementation not available yet");
  }
}

template <typename CoordT>
void initialize_tile_helpers(std::vector<CoordT> &helper_vectors,
                             std::vector<space_t> &helper_scalars,
                             const std::vector<CoordT> &vertices,
                             const TILE_SHAPE shape) {
  if constexpr (std::is_same_v<CoordT, Coord2D>) {
    switch (shape) {
    case TILE_SHAPE::RECTANGLE: {
      initialize_helpers_2D(helper_vectors, helper_scalars, vertices);

      break;
    }

    case TILE_SHAPE::TRIANGLE: {
      initialize_helpers_2D(helper_vectors, helper_scalars, vertices);

      break;
    }

    case TILE_SHAPE::HEXAGON: {
      initialize_hexagon_helpers(helper_vectors, helper_scalars, vertices);

      break;
    }

    default:
      throw std::invalid_argument("Invalid tile shape");
    }
  } else {
    throw std::runtime_error("3D implementation not available yet");
  }
}

template <typename CoordT>
void split_tile(std::vector<Tile<CoordT>> &sub_tiles,
                std::vector<const Tile<CoordT> *> &leaf_tiles,
                const std::vector<CoordT> &vertices,
                const CircumscribedRadius<CoordT> &c_radius,
                SplitBranch &split_tree,
                const std::vector<split_t> &possible_branches,
                const split_t splits, const bool generate_total_leaves_vector,
                const TILE_SHAPE shape) {
  if constexpr (std::is_same_v<CoordT, Coord2D>) {
    switch (shape) {
    case TILE_SHAPE::RECTANGLE: {
      split_rectangle(sub_tiles, leaf_tiles, vertices, c_radius, split_tree,
                      possible_branches, splits, generate_total_leaves_vector);

      break;
    }

    case TILE_SHAPE::TRIANGLE: {
      split_triangle(sub_tiles, leaf_tiles, vertices, split_tree,
                     possible_branches, splits, generate_total_leaves_vector);

      break;
    }

    case TILE_SHAPE::HEXAGON: {
      split_hexagon(sub_tiles, leaf_tiles, vertices, c_radius, split_tree,
                    possible_branches, splits, generate_total_leaves_vector);

      break;
    }

    default:
      throw std::invalid_argument("Invalid tile shape");
    }
  } else {
    throw std::runtime_error("3D implementation not available yet");
  }
}

template <typename CoordT>
bool coord_in_tile(const CoordT &coord,
                   const CircumscribedRadius<CoordT> &c_radius,
                   const std::vector<CoordT> &vertices,
                   const std::vector<CoordT> &helper_vectors,
                   const std::vector<space_t> &helper_scalars,
                   const TILE_SHAPE shape) {
  if constexpr (std::is_same_v<CoordT, Coord2D>) {
    switch (shape) {
    case TILE_SHAPE::RECTANGLE:
      return coord_in_rectangle(coord, c_radius, vertices, helper_vectors,
                                helper_scalars);

    case TILE_SHAPE::TRIANGLE:
      return coord_in_triangle(coord, c_radius, vertices, helper_vectors,
                               helper_scalars);

    case TILE_SHAPE::HEXAGON:
      return coord_in_hexagon(coord, c_radius, vertices, helper_vectors,
                              helper_scalars);

    default:
      throw std::invalid_argument("Invalid tile shape");
    }
  } else {
    throw std::runtime_error("3D implementation not available yet");
  }
}

template <typename CoordT>
std::vector<std::pair<nodeidx_t, CoordT>> generate_coords_in_tile(
    const nodeidx_t first_index, const nodeidx_t coord_count, AnyRNG &rng,
    const std::vector<CoordT> &vertices,
    const std::vector<CoordT> &helper_vectors,
    const CircumscribedRadius<CoordT> &c_radius, const TILE_SHAPE shape) {
  if constexpr (std::is_same_v<CoordT, Coord2D>) {
    switch (shape) {
    case TILE_SHAPE::RECTANGLE:
      return generate_coords_in_rectangle(first_index, coord_count, rng,
                                          vertices, helper_vectors);

    case TILE_SHAPE::TRIANGLE:
      return generate_coords_in_triangle(first_index, coord_count, rng,
                                         vertices);

    case TILE_SHAPE::HEXAGON:
      return generate_coords_in_hexagon(first_index, coord_count, rng, vertices,
                                        c_radius);

    default:
      throw std::invalid_argument("Invalid tile shape");
    }
  } else {
    throw std::runtime_error("3D implementation not available yet");
  }
}

template <typename CoordT>
CoordT project_point_to_surface(const CoordT &coord,
                                const std::vector<CoordT> &vertices,
                                const CircumscribedRadius<CoordT> &c_radius,
                                const TILE_SHAPE shape) {
  if constexpr (std::is_same_v<CoordT, Coord2D>) {
    switch (shape) {
    case TILE_SHAPE::RECTANGLE:
      return project_point_to_2D_perimeter(coord, vertices);

    case TILE_SHAPE::HEXAGON:
      return project_point_to_2D_perimeter(coord, vertices);

    case TILE_SHAPE::TRIANGLE:
      return project_point_to_triangle_perimeter(coord, vertices, c_radius);

    default:
      throw std::invalid_argument("Invalid tile shape");
    }
  } else {
    throw std::runtime_error("3D implementation not available yet");
  }
}

tileidx_t compute_leaves_count(const split_t splits, const TILE_SHAPE shape);

std::vector<split_t> get_possible_sub_tile_branches(const split_t splits,
                                                    const TILE_SHAPE shape);

split_t compute_minimal_splits(const double expected_total_nodes,
                               const double total_tiles,
                               const double expected_nodes_per_leaf,
                               const TILE_SHAPE shape);
} // namespace sapi

#endif
