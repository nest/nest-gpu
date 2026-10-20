/*
 *  gf2d_geometry.h
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

#ifndef GF2D_GEOMETRY_H
#define GF2D_GEOMETRY_H

#include "coordinate_geometry.h"
#include "grid_functors.h"
#include "tile.h"

namespace sapi {
inline void initialize_soc_2D(ShiftedOriginCreator<Coord2D> &soc,
                              const std::vector<space_t> &side_lengths,
                              const angle_t angular_offset) {
  soc.helper_scalars_.resize(2);
  soc.helper_scalars_[0] = side_lengths[0];
  soc.helper_scalars_[1] =
      1 < side_lengths.size() ? side_lengths[1] : side_lengths[0];

  if (angular_offset != 0)
    soc.helper_vectors_.resize(1, create_angular_offset(angular_offset));
}

inline void initialize_hexagonal_soc(ShiftedOriginCreator<Coord2D> &soc,
                                     const std::vector<space_t> &side_lengths,
                                     const angle_t angular_offset) {
  soc.helper_scalars_.resize(3);
  // With side length a, height h of eq triangle is equal to sqrt( 3 ) * a / 2
  // To reach neighboring hexagon center we need 2 * h
  soc.helper_scalars_[0] = std::sqrt(space_t(3)) * side_lengths[0];
  soc.helper_scalars_[1] = soc.helper_scalars_[0] / 2;
  soc.helper_scalars_[2] = 3 * side_lengths[0] / 2;

  if (angular_offset != 0)
    soc.helper_vectors_.resize(1, create_angular_offset(angular_offset));
}

inline Coord2D
create_shifted_rectangular_origin(const Coord2D &grid_origin,
                                  const GridPosition<Coord2D> &grid_position,
                                  const ShiftedOriginCreator<Coord2D> &soc) {
  Coord2D disp_vector =
      construct_coord_2D(soc.helper_scalars_[0] * grid_position[0],
                         soc.helper_scalars_[1] * grid_position[1]);
  if (!soc.helper_vectors_.empty())
    return grid_origin +
           rotate_displacement(disp_vector, soc.helper_vectors_[0]);
  else
    return grid_origin + disp_vector;
}

inline Coord2D create_shifted_triangular_origin(
    const Coord2D &grid_origin, const GridPosition<Coord2D> &grid_position,
    const GridPositionParity<Coord2D> &grid_position_parity,
    const ShiftedOriginCreator<Coord2D> &soc) {
  Coord2D disp_vector = construct_coord_2D(
      soc.helper_scalars_[0] *
          (grid_position[0] - 1 * !grid_position_parity[0]) / 2,
      soc.helper_scalars_[1] * grid_position[1]);
  if (!soc.helper_vectors_.empty())
    return grid_origin +
           rotate_displacement(disp_vector, soc.helper_vectors_[0]);
  else
    return grid_origin + disp_vector;
}

inline Coord2D create_shifted_hexagonal_origin(
    const Coord2D &grid_origin, const GridPosition<Coord2D> &grid_position,
    const GridPositionParity<Coord2D> &grid_position_parity,
    const ShiftedOriginCreator<Coord2D> &soc) {
  Coord2D disp_vector = construct_coord_2D(
      std::fma(soc.helper_scalars_[0], grid_position[0],
               soc.helper_scalars_[1] * !grid_position_parity[1]),
      soc.helper_scalars_[2] * grid_position[1]);
  if (!soc.helper_vectors_.empty())
    return grid_origin +
           rotate_displacement(disp_vector, soc.helper_vectors_[0]);
  else
    return grid_origin + disp_vector;
}

inline void initialize_rectangular_ctc(CachedTileCreator<Coord2D> &ctc,
                                       const std::vector<space_t> &side_lengths,
                                       const angle_t angular_offset) {
  ctc.side_lengths_ = side_lengths;
  ctc.angular_offsets_.resize(1, angular_offset);
}

inline void initialize_triangular_ctc(CachedTileCreator<Coord2D> &ctc,
                                      const std::vector<space_t> &side_lengths,
                                      const angle_t angular_offset) {
  ctc.side_lengths_ = side_lengths;
  ctc.angular_offsets_.resize(2);
  ctc.angular_offsets_[0] = angular_offset;
  ctc.angular_offsets_[1] = angular_offset + 180;
}

inline void initialize_hexagonal_ctc(CachedTileCreator<Coord2D> &ctc,
                                     const std::vector<space_t> &side_lengths,
                                     const angle_t angular_offset) {
  ctc.side_lengths_ = side_lengths;
  ctc.angular_offsets_.resize(1, angular_offset + 30);
}

inline bool
check_rectangular_dimensions(const GridPosition<Coord2D> &grid_dimensions,
                             const CachedTileCreator<Coord2D> &ctc) {
  return (grid_dimensions[0] + grid_dimensions[1] <= 2) ||
         (ctc.angular_offsets_[0] % 45 == 0);
}

inline bool
check_triangular_dimensions(const GridPosition<Coord2D> &grid_dimensions,
                            const CachedTileCreator<Coord2D> &ctc) {
  return (grid_dimensions[0] + grid_dimensions[1] <= 2) ||
         ((ctc.angular_offsets_[1] % 180 == 0) &&
          (grid_dimensions[0] % 2 == 0));
}

inline bool
check_hexagonal_dimensions(const GridPosition<Coord2D> &grid_dimensions,
                           const CachedTileCreator<Coord2D> &ctc) {
  return (grid_dimensions[0] + grid_dimensions[1] <= 2) ||
         ((ctc.angular_offsets_[0] % 30 == 0) && (grid_dimensions[1] % 2 == 0));
}

inline Tile<Coord2D> create_rectangle(const Coord2D &tile_origin,
                                      const CachedTileCreator<Coord2D> &ctc) {
  return Tile<Coord2D>(TILE_SHAPE::RECTANGLE, tile_origin, ctc.side_lengths_,
                       ctc.angular_offsets_);
}

inline Tile<Coord2D>
create_triangle(const Coord2D &tile_origin,
                const GridPositionParity<Coord2D> &grid_position_parity,
                const CachedTileCreator<Coord2D> &ctc) {
  return Tile<Coord2D>(TILE_SHAPE::TRIANGLE, tile_origin, ctc.side_lengths_,
                       {grid_position_parity[0] ? ctc.angular_offsets_[0]
                                                : ctc.angular_offsets_[1]});
}

inline Tile<Coord2D> create_hexagon(const Coord2D &tile_origin,
                                    const CachedTileCreator<Coord2D> &ctc) {
  return Tile<Coord2D>(TILE_SHAPE::HEXAGON, tile_origin, ctc.side_lengths_,
                       ctc.angular_offsets_);
}
} // namespace sapi

#endif
