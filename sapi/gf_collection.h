/*
 *  gf_collection.h
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

#ifndef GF_COLLECTION_H
#define GF_COLLECTION_H

#include <algorithm>

#include "grid_functors.h"

namespace sapi {
// Forward definition to creator_registry.h
template <typename RT> class CreatorRegistry;

template <typename CoordT> struct GFCollection {
  CoordT grid_origin_;
  GridPosition<CoordT> grid_dimensions_;
  GridTargetPositionShifts<CoordT> gps_;
  ShiftedOriginCreator<CoordT> soc_;
  CachedTileCreator<CoordT> ctc_;

  tileidx_t compute_index(const GridPosition<CoordT> &position) const;

  CoordT shift_origin(const GridPosition<CoordT> &position,
                      const GridPositionParity<CoordT> &parity) const;

  Tile<CoordT> create_tile(const CoordT &origin,
                           const GridPositionParity<CoordT> &parity) const;
};

template <typename CoordT>
GFCollection<CoordT> construct_gf_collection(
    const std::vector<space_t> &grid_origin,
    const std::vector<tileidx_t> &grid_dimensions, const std::string &tile_type,
    const std::vector<space_t> &tile_side_lengths,
    const std::vector<angle_t> &tile_angular_offsets,
    const CreatorRegistry<GridTargetPositionShifts<CoordT>> &gpsr,
    const CreatorRegistry<ShiftedOriginCreator<CoordT>> &socr,
    const CreatorRegistry<CachedTileCreator<CoordT>> &ctcr) {
  if (grid_origin.empty() ||
      grid_origin.size() != static_cast<std::size_t>(CoordT::D) ||
      grid_dimensions.size() != grid_origin.size() ||
      !std::all_of(grid_dimensions.cbegin(), grid_dimensions.cend(),
                   positiveTix))
    throw std::invalid_argument("Invalid grid params for GFCollection");

  GFCollection<CoordT> gfc;

  std::copy(grid_dimensions.cbegin(), grid_dimensions.cend(),
            gfc.grid_dimensions_.begin());

  gfc.grid_origin_ = CoordT::copy_from_vec(grid_origin.begin());
  gfc.gps_ = gpsr.get_creator(tile_type)->create();
  gfc.soc_ = socr.get_creator(tile_type)->create(tile_side_lengths,
                                                 tile_angular_offsets);
  gfc.ctc_ = ctcr.get_creator(tile_type)->create(tile_side_lengths,
                                                 tile_angular_offsets);

  if (!gfc.ctc_.check_dimensions(gfc.grid_dimensions_))
    throw std::invalid_argument(
        "Grid cannot be instantiated with the dimension | rotation | edge "
        "wrapping combination");

  return gfc;
}

template <typename CoordT>
inline tileidx_t
GFCollection<CoordT>::compute_index(const GridPosition<CoordT> &pos) const {
  return position_to_index(pos, grid_dimensions_);
}

template <typename CoordT>
inline CoordT GFCollection<CoordT>::shift_origin(
    const GridPosition<CoordT> &gpos,
    const GridPositionParity<CoordT> &gpa) const {
  return soc_.create_shifted_origin(grid_origin_, gpos, gpa);
}

template <typename CoordT>
inline Tile<CoordT>
GFCollection<CoordT>::create_tile(const CoordT &tile_origin,
                                  const GridPositionParity<CoordT> &gpa) const {
  return ctc_.create_tile(tile_origin, gpa);
}
} // namespace sapi

#endif
