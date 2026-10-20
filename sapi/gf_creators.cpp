/*
 *  gf_creators.cpp
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

#include "gf_creators.h"
#include "creator_registry.h"
#include "gf2d_geometry.h"
#include "type_erasure_helpers.h"

namespace sapi {
struct RectangleGSCreator final
    : public StateLessCreator<GridTargetPositionShifts<Coord2D>> {
  GridTargetPositionShifts<Coord2D> create() const override {
    return construct_grid_target_position_shifts<Coord2D>(
        GridShiftVector<Coord2D>{{-1, 1},
                                 {0, 1},
                                 {1, 1},
                                 {-1, 0},
                                 {0, 0},
                                 {1, 0},
                                 {-1, -1},
                                 {0, -1},
                                 {1, -1}},
        GridDimensionalShifts<Coord2D>{});
  }
};

struct TriangleGSCreator final
    : public StateLessCreator<GridTargetPositionShifts<Coord2D>> {
  GridTargetPositionShifts<Coord2D> create() const override {
    return construct_grid_target_position_shifts<Coord2D>(
        GridShiftVector<Coord2D>{// down, up, left, right
                                 {0, -1},
                                 {0, 1},
                                 {-1, 0},
                                 {1, 0},
                                 // 2 * left, 2 * right, up + right, down + left
                                 {-2, 0},
                                 {2, 0},
                                 {1, 1},
                                 {-1, -1},
                                 // up + 2 * right, down + 2 * left, self
                                 {2, 1},
                                 {-2, -1},
                                 {0, 0}},
        GridDimensionalShifts<Coord2D>{
            GridShiftVectorPair<Coord2D>{
                GridShiftVector<Coord2D>{
                    // if x pos is even, up + left, up +  3 * right
                    {-1, 1},
                    {3, 1},
                },
                GridShiftVector<Coord2D>{
                    // if x pos is odd, down + right, down + 3 * left
                    {1, -1},
                    {-3, -1}}},
            GridShiftVectorPair<Coord2D>{}});
  }
};

struct HexagonGSCreator final
    : public StateLessCreator<GridTargetPositionShifts<Coord2D>> {
  GridTargetPositionShifts<Coord2D> create() const override {
    return construct_grid_target_position_shifts<Coord2D>(
        GridShiftVector<Coord2D>{// down, up, left, right, self
                                 {0, -1},
                                 {0, 1},
                                 {-1, 0},
                                 {1, 0},
                                 {0, 0}},
        GridDimensionalShifts<Coord2D>{
            GridShiftVectorPair<Coord2D>{},
            GridShiftVectorPair<Coord2D>{
                GridShiftVector<Coord2D>{
                    // if y pos is even, up + left, down + left
                    {-1, 1},
                    {-1, -1}},
                GridShiftVector<Coord2D>{
                    // if y pos is odd, up + right, down + right
                    {1, 1},
                    {1, -1}}}});
  }
};

template <typename CoordT, TILE_SHAPE shape>
struct SOCCreator final
    : public StateLessCreator<ShiftedOriginCreator<CoordT>> {
  ShiftedOriginCreator<CoordT>
  create(const std::vector<space_t> &side_lengths,
         const std::vector<angle_t> &angular_offsets) const override {
    return construct_shifted_origin_creator<CoordT>(shape, side_lengths,
                                                    angular_offsets);
  }
};

template <typename CoordT, TILE_SHAPE shape>
struct CTCCreator final : public StateLessCreator<CachedTileCreator<CoordT>> {
  CachedTileCreator<CoordT>
  create(const std::vector<space_t> &side_lengths,
         const std::vector<angle_t> &angular_offsets) const override {
    return construct_cached_tile_creator<CoordT>(shape, side_lengths,
                                                 angular_offsets);
  }
};

void initialize_gsc_registry(
    CreatorRegistry<GridTargetPositionShifts<Coord2D>> &gscr) {
  gscr.register_creator<RectangleGSCreator>(
      TILE_SHAPE_NAMES[uint8_t(TILE_SHAPE::RECTANGLE)]);
  gscr.register_creator<TriangleGSCreator>(
      TILE_SHAPE_NAMES[uint8_t(TILE_SHAPE::TRIANGLE)]);
  gscr.register_creator<HexagonGSCreator>(
      TILE_SHAPE_NAMES[uint8_t(TILE_SHAPE::HEXAGON)]);
}

void initialize_gsc_registry(
    CreatorRegistry<GridTargetPositionShifts<Coord3D>> &) {
  throw std::runtime_error("3D GridGenerator not yet implemented");
}

void initialize_soc_registry(
    CreatorRegistry<ShiftedOriginCreator<Coord2D>> &socr) {
  socr.register_creator<SOCCreator<Coord2D, TILE_SHAPE::RECTANGLE>>(
      TILE_SHAPE_NAMES[uint8_t(TILE_SHAPE::RECTANGLE)]);
  socr.register_creator<SOCCreator<Coord2D, TILE_SHAPE::TRIANGLE>>(
      TILE_SHAPE_NAMES[uint8_t(TILE_SHAPE::TRIANGLE)]);
  socr.register_creator<SOCCreator<Coord2D, TILE_SHAPE::HEXAGON>>(
      TILE_SHAPE_NAMES[uint8_t(TILE_SHAPE::HEXAGON)]);
}

void initialize_soc_registry(CreatorRegistry<ShiftedOriginCreator<Coord3D>> &) {
  throw std::runtime_error("3D GridGenerator not yet implemented");
}

void initialize_ctc_registry(
    CreatorRegistry<CachedTileCreator<Coord2D>> &ctcr) {
  ctcr.register_creator<CTCCreator<Coord2D, TILE_SHAPE::RECTANGLE>>(
      TILE_SHAPE_NAMES[uint8_t(TILE_SHAPE::RECTANGLE)]);
  ctcr.register_creator<CTCCreator<Coord2D, TILE_SHAPE::TRIANGLE>>(
      TILE_SHAPE_NAMES[uint8_t(TILE_SHAPE::TRIANGLE)]);
  ctcr.register_creator<CTCCreator<Coord2D, TILE_SHAPE::HEXAGON>>(
      TILE_SHAPE_NAMES[uint8_t(TILE_SHAPE::HEXAGON)]);
}

void initialize_ctc_registry(CreatorRegistry<CachedTileCreator<Coord3D>> &) {
  throw std::runtime_error("3D GridGenerator not yet implemented");
}
} // namespace sapi
