/*
 *  tile2d_geometry.h
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

#ifndef TILE2D_GEOMETRY_H
#define TILE2D_GEOMETRY_H

#include "split_tree.h"

namespace sapi {
// Forward definition to link with coordinates.h
struct Coord2D;
template <typename CoordT> struct CircumscribedRadius;

// Forward definition to link with tile.h
template <typename CoordT> struct Tile;

// Forward definition to link with type_erasure_helpers.h
template <typename RT, typename std::enable_if_t<
                           std::disjunction_v<std::is_same<RT, uint32_t>,
                                              std::is_same<RT, uint64_t>>,
                           bool>
                           b>
class AnyRNG_T;
typedef AnyRNG_T<rng_bits_t, true> AnyRNG;

void initialize_vertices_2D(std::vector<Coord2D> &vertices,
                            CircumscribedRadius<Coord2D> &c_radius,
                            const std::vector<space_t> &side_lengths,
                            const angle_t angular_offset,
                            const bool triangular_vertices);

void initialize_hexagon_vertices(std::vector<Coord2D> &vertices,
                                 CircumscribedRadius<Coord2D> &c_radius,
                                 const space_t side_length,
                                 const angle_t angular_offset);

void initialize_helpers_2D(std::vector<Coord2D> &helper_vectors,
                           std::vector<space_t> &helper_scalars,
                           const std::vector<Coord2D> &vertices);

void initialize_hexagon_helpers(std::vector<Coord2D> &helper_vectors,
                                std::vector<space_t> &helper_scalars,
                                const std::vector<Coord2D> &vertices);

void split_rectangle(std::vector<Tile<Coord2D>> &sub_tiles,
                     std::vector<const Tile<Coord2D> *> &leaf_tiles,
                     const std::vector<Coord2D> &vertices,
                     const CircumscribedRadius<Coord2D> &c_radius,
                     SplitBranch &split_tree,
                     const std::vector<split_t> &possible_branches,
                     const split_t splits,
                     const bool generate_total_leaves_vector);

void split_triangle(std::vector<Tile<Coord2D>> &sub_tiles,
                    std::vector<const Tile<Coord2D> *> &leaf_tiles,
                    const std::vector<Coord2D> &vertices,
                    SplitBranch &split_tree,
                    const std::vector<split_t> &possible_branches,
                    const split_t splits,
                    const bool generate_total_leaves_vector);

void split_hexagon(std::vector<Tile<Coord2D>> &sub_tiles,
                   std::vector<const Tile<Coord2D> *> &leaf_tiles,
                   const std::vector<Coord2D> &vertices,
                   const CircumscribedRadius<Coord2D> &c_radius,
                   SplitBranch &split_tree,
                   const std::vector<split_t> &possible_branches,
                   const split_t splits,
                   const bool generate_total_leaves_vector);

bool coord_in_rectangle(const Coord2D &coord,
                        const CircumscribedRadius<Coord2D> &c_radius,
                        const std::vector<Coord2D> &vertices,
                        const std::vector<Coord2D> &helper_vectors,
                        const std::vector<space_t> &helper_scalars);

bool coord_in_triangle(const Coord2D &coord,
                       const CircumscribedRadius<Coord2D> &c_radius,
                       const std::vector<Coord2D> &vertices,
                       const std::vector<Coord2D> &helper_vectors,
                       const std::vector<space_t> &helper_scalars);

bool coord_in_hexagon(const Coord2D &coord,
                      const CircumscribedRadius<Coord2D> &c_radius,
                      const std::vector<Coord2D> &vertices,
                      const std::vector<Coord2D> &helper_vectors,
                      const std::vector<space_t> &helper_scalars);

std::vector<std::pair<nodeidx_t, Coord2D>>
generate_coords_in_rectangle(const nodeidx_t first_index,
                             const nodeidx_t coord_count, AnyRNG &rng,
                             const std::vector<Coord2D> &vertices,
                             const std::vector<Coord2D> &helper_vectors);

std::vector<std::pair<nodeidx_t, Coord2D>>
generate_coords_in_triangle(const nodeidx_t first_index,
                            const nodeidx_t coord_count, AnyRNG &rng,
                            const std::vector<Coord2D> &vertices);

std::vector<std::pair<nodeidx_t, Coord2D>>
generate_coords_in_hexagon(const nodeidx_t first_index,
                           const nodeidx_t coord_count, AnyRNG &rng,
                           const std::vector<Coord2D> &vertices,
                           const CircumscribedRadius<Coord2D> &c_radius);

Coord2D project_point_to_2D_perimeter(const Coord2D &coord,
                                      const std::vector<Coord2D> &vertices);

Coord2D project_point_to_triangle_perimeter(
    const Coord2D &coord, const std::vector<Coord2D> &vertices,
    const CircumscribedRadius<Coord2D> &c_radius);
} // namespace sapi

#endif
