/*
 *  grid_functors.h
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

#ifndef GRID_FUNCTORS_H
#define GRID_FUNCTORS_H

#include "gf_geometry.h"

namespace sapi {
template <typename CoordT> struct GridTargetPositionShifts {
  // Position independent GridShiftVector and dimension position dependent
  // GridShiftVector for complete displacement definition of position relative
  // grid targets
  GridShiftVector<CoordT> position_independent_shifts_;
  GridDimensionalShifts<CoordT> position_dependent_shifts_;
};

template <typename CoordT>
inline GridTargetPositionShifts<CoordT> construct_grid_target_position_shifts(
    GridShiftVector<CoordT> &&position_independent_shifts,
    GridDimensionalShifts<CoordT> &&position_dependent_shifts) {
  GridTargetPositionShifts<CoordT> gps;
  gps.position_independent_shifts_ = std::move(position_independent_shifts);
  gps.position_dependent_shifts_ = std::move(position_dependent_shifts);
  return gps;
}

template <typename CoordT> struct ShiftedOriginCreator {
  TILE_SHAPE shape_ = TILE_SHAPE::NULL_TS;
  std::vector<CoordT> helper_vectors_;
  std::vector<space_t> helper_scalars_;

  CoordT create_shifted_origin(
      const CoordT &grid_origin, const GridPosition<CoordT> &grid_position,
      const GridPositionParity<CoordT> &grid_position_parity) const;
};

template <typename CoordT>
inline ShiftedOriginCreator<CoordT>
construct_shifted_origin_creator(const TILE_SHAPE shape,
                                 const std::vector<space_t> &side_lengths,
                                 const std::vector<angle_t> &angular_offsets) {
  ShiftedOriginCreator<CoordT> soc;
  soc.shape_ = shape;
  sapi::initialize_soc(soc, side_lengths, angular_offsets);
  return soc;
}

template <typename CoordT>
inline CoordT ShiftedOriginCreator<CoordT>::create_shifted_origin(
    const CoordT &grid_origin, const GridPosition<CoordT> &grid_position,
    const GridPositionParity<CoordT> &grid_position_parity) const {
  return sapi::create_shifted_origin(grid_origin, grid_position,
                                     grid_position_parity, *this);
}

template <typename CoordT> struct CachedTileCreator {
  TILE_SHAPE shape_ = TILE_SHAPE::NULL_TS;
  std::vector<space_t> side_lengths_;
  std::vector<angle_t> angular_offsets_;

  bool check_dimensions(const GridPosition<CoordT> &grid_dimensions) const;

  Tile<CoordT>
  create_tile(const CoordT &tile_origin,
              const GridPositionParity<CoordT> &grid_position_parity) const;
};

template <typename CoordT>
inline CachedTileCreator<CoordT>
construct_cached_tile_creator(const TILE_SHAPE shape,
                              const std::vector<space_t> &side_lengths,
                              const std::vector<angle_t> &angular_offsets) {
  CachedTileCreator<CoordT> ctc;
  ctc.shape_ = shape;
  sapi::initialize_ctc(ctc, side_lengths, angular_offsets);
  return ctc;
}

template <typename CoordT>
inline bool CachedTileCreator<CoordT>::check_dimensions(
    const GridPosition<CoordT> &grid_dimensions) const {
  return sapi::check_dimensions(grid_dimensions, *this);
}

template <typename CoordT>
inline Tile<CoordT> CachedTileCreator<CoordT>::create_tile(
    const CoordT &tile_origin,
    const GridPositionParity<CoordT> &grid_position_parity) const {
  return sapi::create_tile(tile_origin, grid_position_parity, *this);
}
} // namespace sapi

#endif
