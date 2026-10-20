/*
 *  mask_creators.cpp
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

#include "mask_creators.h"
#include "creator_registry.h"
#include "mask2d_geometry.h"
#include "type_erasure_helpers.h"

namespace sapi {
template <typename CoordT, MASK_SHAPE shape>
struct MaskCreator final : public StateLessCreator<Mask<CoordT>> {
  Mask<CoordT> create(const std::vector<space_t> &origin,
                      const std::vector<space_t> &params,
                      const std::vector<space_t> &offset) const override {
    return construct_mask<CoordT>(shape, origin, params, offset);
  }
};

void initialize_mk_registry(CreatorRegistry<Mask<Coord2D>> &mkr) {
  mkr.register_creator<MaskCreator<Coord2D, MASK_SHAPE::CIRCULAR>>(
      MASK_SHAPE_NAMES[uint8_t(MASK_SHAPE::CIRCULAR)]);
  mkr.register_creator<MaskCreator<Coord2D, MASK_SHAPE::ELLIPTICAL>>(
      MASK_SHAPE_NAMES[uint8_t(MASK_SHAPE::ELLIPTICAL)]);
  mkr.register_creator<MaskCreator<Coord2D, MASK_SHAPE::PARALLELOGRAM>>(
      MASK_SHAPE_NAMES[uint8_t(MASK_SHAPE::PARALLELOGRAM)]);
  mkr.register_creator<MaskCreator<Coord2D, MASK_SHAPE::TRIANGULAR>>(
      MASK_SHAPE_NAMES[uint8_t(MASK_SHAPE::TRIANGULAR)]);
}

void initialize_mk_registry(CreatorRegistry<Mask<Coord3D>> &) {
  throw std::runtime_error("3D Masks not yet implemented");
}
} // namespace sapi
