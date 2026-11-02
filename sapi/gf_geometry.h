/*
 *  gf_geometry.h
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

#ifndef GF_GEOMETRY_H
#define GF_GEOMETRY_H

#include <stdexcept>

#include "coordinates.h"
#include "enum_store.h"
#include "grid_containers.h"

namespace sapi {
// Forward definitions to tile.h
template <typename CoordT> struct Tile;

// Forward definitions to grid_functors.h
template <typename CoordT> struct ShiftedOriginCreator;
template <typename CoordT> struct CachedTileCreator;

void initialize_soc_2D(ShiftedOriginCreator<Coord2D> &soc,
                       const std::vector<space_t> &side_lengths,
                       const angle_t angular_offset);

void initialize_hexagonal_soc(ShiftedOriginCreator<Coord2D> &soc,
                              const std::vector<space_t> &side_lengths,
                              const angle_t angular_offset);

template <typename CoordT>
void initialize_soc(ShiftedOriginCreator<CoordT> &soc,
                    const std::vector<space_t> &side_lengths,
                    const std::vector<angle_t> &angular_offsets) {
  if constexpr (std::is_same_v<CoordT, Coord2D>) {
    if (1 < angular_offsets.size())
      throw std::invalid_argument("Invalid offsets for tile 2D");
    const auto lengths = side_lengths.size();

    switch (soc.shape_) {
    case TILE_SHAPE::RECTANGLE: {
      if (lengths < 1 || 2 < lengths)
        throw std::invalid_argument("Invalid rectangle length vector");

      initialize_soc_2D(soc, side_lengths,
                        angular_offsets.empty() ? 0 : angular_offsets[0]);

      break;
    }

    case TILE_SHAPE::TRIANGLE: {
      if (lengths < 1 || 2 < lengths)
        throw std::invalid_argument("Invalid triangle length vector");

      initialize_soc_2D(soc, side_lengths,
                        angular_offsets.empty() ? 0 : angular_offsets[0]);

      break;
    }

    case TILE_SHAPE::HEXAGON: {
      if (lengths != 1)
        throw std::invalid_argument("Invalid hexagon length vector");

      initialize_hexagonal_soc(
          soc, side_lengths, angular_offsets.empty() ? 0 : angular_offsets[0]);

      break;
    }

    default:
      throw std::invalid_argument("Invalid tile shape");
    }
  } else {
    throw std::runtime_error("3D implementation not available yet");
  }
}

Coord2D
create_shifted_rectangular_origin(const Coord2D &grid_origin,
                                  const GridPosition<Coord2D> &grid_position,
                                  const ShiftedOriginCreator<Coord2D> &soc);

Coord2D create_shifted_triangular_origin(
    const Coord2D &grid_origin, const GridPosition<Coord2D> &grid_position,
    const GridPositionParity<Coord2D> &grid_position_parity,
    const ShiftedOriginCreator<Coord2D> &soc);

Coord2D create_shifted_hexagonal_origin(
    const Coord2D &grid_origin, const GridPosition<Coord2D> &grid_position,
    const GridPositionParity<Coord2D> &grid_position_parity,
    const ShiftedOriginCreator<Coord2D> &soc);

template <typename CoordT>
CoordT
create_shifted_origin(const CoordT &grid_origin,
                      const GridPosition<CoordT> &grid_position,
                      const GridPositionParity<CoordT> &grid_position_parity,
                      const ShiftedOriginCreator<CoordT> &soc) {
  if constexpr (std::is_same_v<CoordT, Coord2D>) {
    switch (soc.shape_) {
    case TILE_SHAPE::RECTANGLE:
      return create_shifted_rectangular_origin(grid_origin, grid_position, soc);

    case TILE_SHAPE::TRIANGLE:
      return create_shifted_triangular_origin(grid_origin, grid_position,
                                              grid_position_parity, soc);

    case TILE_SHAPE::HEXAGON:
      return create_shifted_hexagonal_origin(grid_origin, grid_position,
                                             grid_position_parity, soc);

    default:
      throw std::invalid_argument("Invalid tile shape");
    }
  } else {
    throw std::runtime_error("3D implementation not available yet");
  }
}

void initialize_rectangular_ctc(CachedTileCreator<Coord2D> &ctc,
                                const std::vector<space_t> &side_lengths,
                                const angle_t angular_offset);

void initialize_triangular_ctc(CachedTileCreator<Coord2D> &ctc,
                               const std::vector<space_t> &side_lengths,
                               const angle_t angular_offset);

void initialize_hexagonal_ctc(CachedTileCreator<Coord2D> &ctc,
                              const std::vector<space_t> &side_lengths,
                              const angle_t angular_offset);

template <typename CoordT>
void initialize_ctc(CachedTileCreator<CoordT> &ctc,
                    const std::vector<space_t> &side_lengths,
                    const std::vector<angle_t> &angular_offsets) {
  const auto lengths = side_lengths.size();
  const auto rotations = angular_offsets.size();

  if constexpr (std::is_same_v<CoordT, Coord2D>) {
    if (1 < rotations)
      throw std::invalid_argument("Invalid offsets for tile 2D");

    switch (ctc.shape_) {
    case TILE_SHAPE::RECTANGLE: {
      if (lengths < 1 || 2 < lengths)
        throw std::invalid_argument("Invalid rectangle length vector");

      initialize_rectangular_ctc(
          ctc, side_lengths, angular_offsets.empty() ? 0 : angular_offsets[0]);

      break;
    }

    case TILE_SHAPE::TRIANGLE: {
      if (lengths < 1 || 2 < lengths)
        throw std::invalid_argument("Invalid triangle length vector");

      initialize_triangular_ctc(
          ctc, side_lengths, angular_offsets.empty() ? 0 : angular_offsets[0]);

      break;
    }

    case TILE_SHAPE::HEXAGON: {
      if (lengths != 1)
        throw std::invalid_argument("Invalid hexagon length vector");

      initialize_hexagonal_ctc(
          ctc, side_lengths, angular_offsets.empty() ? 0 : angular_offsets[0]);

      break;
    }

    default:
      throw std::invalid_argument("Invalid tile shape");
    }
  } else {
    throw std::runtime_error("3D implementation not available yet");
  }
}

bool check_rectangular_dimensions(const GridPosition<Coord2D> &grid_dimensions,
                                  const CachedTileCreator<Coord2D> &ctc);

bool check_triangular_dimensions(const GridPosition<Coord2D> &grid_dimensions,
                                 const CachedTileCreator<Coord2D> &ctc);

bool check_hexagonal_dimensions(const GridPosition<Coord2D> &grid_dimensions,
                                const CachedTileCreator<Coord2D> &ctc);

template <typename CoordT>
bool check_dimensions(const GridPosition<CoordT> &grid_dimensions,
                      const CachedTileCreator<CoordT> &ctc) {
  if constexpr (std::is_same_v<CoordT, Coord2D>) {
    switch (ctc.shape_) {
    case TILE_SHAPE::RECTANGLE:
      return check_rectangular_dimensions(grid_dimensions, ctc);

    case TILE_SHAPE::TRIANGLE:
      return check_triangular_dimensions(grid_dimensions, ctc);

    case TILE_SHAPE::HEXAGON:
      return check_hexagonal_dimensions(grid_dimensions, ctc);

    default:
      throw std::invalid_argument("Invalid tile shape");
    }
  } else {
    throw std::runtime_error("3D implementation not available yet");
  }
}

Tile<Coord2D> create_rectangle(const Coord2D &tile_origin,
                               const CachedTileCreator<Coord2D> &ctc);

Tile<Coord2D>
create_triangle(const Coord2D &tile_origin,
                const GridPositionParity<Coord2D> &grid_position_parity,
                const CachedTileCreator<Coord2D> &ctc);

Tile<Coord2D> create_hexagon(const Coord2D &tile_origin,
                             const CachedTileCreator<Coord2D> &ctc);

template <typename CoordT>
Tile<CoordT> create_tile(const CoordT &tile_origin,
                         const GridPositionParity<CoordT> &grid_position_parity,
                         const CachedTileCreator<CoordT> &ctc) {
  if constexpr (std::is_same_v<CoordT, Coord2D>) {
    switch (ctc.shape_) {
    case TILE_SHAPE::RECTANGLE:
      return create_rectangle(tile_origin, ctc);

    case TILE_SHAPE::TRIANGLE:
      return create_triangle(tile_origin, grid_position_parity, ctc);

    case TILE_SHAPE::HEXAGON:
      return create_hexagon(tile_origin, ctc);

    default:
      throw std::invalid_argument("Invalid tile shape");
    }
  } else {
    throw std::runtime_error("3D implementation not available yet");
  }
}
} // namespace sapi

#endif
