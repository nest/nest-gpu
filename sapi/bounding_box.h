/*
 *  bounding_box.h
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

#ifndef BOUNDING_BOX_H
#define BOUNDING_BOX_H

#include <vector>

#include "numerics.h"

namespace sapi {
// Forward definition to link with coordinate_geometry.h
template <typename CoordT> std::pair<CoordT, CoordT> minmax_coords();
template <typename CoordT> void update_min(CoordT &, const CoordT &);
template <typename CoordT> void update_max(CoordT &, const CoordT &);

template <typename CoordT> struct BoundingBox {
  // As all indexes are defined positives,
  // -1 acts as a sentinel for boxes unmapped to
  // grid positions
  tileidx_t tile_index_ = -1;

  // Boxes are defined as pair of minimum and
  // maximum coordinates
  std::pair<CoordT, CoordT> minmax_bounds_;

  // Bounding box is defined regardless of rotation
  // such that inner boxes bounds are with local bounds
  // minx < x < maxx
  // miny < y < maxy
  // (minz < z < maxz)
  std::vector<BoundingBox<CoordT>> inner_boxes_;

  BoundingBox() noexcept = default;
  BoundingBox(const BoundingBox &) = delete;
  BoundingBox(BoundingBox &&) noexcept = default;
  ~BoundingBox() noexcept = default;

  BoundingBox(std::pair<CoordT, CoordT> &&minmax_bounds,
              const tileidx_t tile_index) noexcept;

  BoundingBox(std::pair<CoordT, CoordT> &&minmax_bounds,
              std::vector<BoundingBox<CoordT>> &&inner_boxes) noexcept;

  BoundingBox &operator=(const BoundingBox &) = delete;
  BoundingBox &operator=(BoundingBox &&) noexcept;

  bool coord_in_box(const CoordT &coord) const;
};

template <typename CoordT>
BoundingBox<CoordT>::BoundingBox(std::pair<CoordT, CoordT> &&minmax_bounds,
                                 const tileidx_t tile_index) noexcept
    : tile_index_(tile_index), minmax_bounds_(std::move(minmax_bounds)) {}

template <typename CoordT>
BoundingBox<CoordT>::BoundingBox(
    std::pair<CoordT, CoordT> &&minmax_bounds,
    std::vector<BoundingBox<CoordT>> &&inner_boxes) noexcept
    : minmax_bounds_(std::move(minmax_bounds)),
      inner_boxes_(std::move(inner_boxes)) {}

template <typename CoordT>
inline BoundingBox<CoordT> &
BoundingBox<CoordT>::operator=(BoundingBox &&bb) noexcept {
  tile_index_ = bb.tile_index_;
  bb.tile_index_ = -1;

  minmax_bounds_.first = bb.minmax_bounds_.first;
  minmax_bounds_.second = bb.minmax_bounds_.second;

  inner_boxes_.swap(bb.inner_boxes_);
  bb.inner_boxes_.clear();

  return *this;
}

template <typename CoordT>
inline bool BoundingBox<CoordT>::coord_in_box(const CoordT &coord) const {
  bool res = interval_test(minmax_bounds_.first.x_, coord.x_,
                           minmax_bounds_.second.x_) &&
             interval_test(minmax_bounds_.first.y_, coord.y_,
                           minmax_bounds_.second.y_);

  if constexpr (std::is_same_v<CoordT, Coord3D>)
    res &= interval_test(minmax_bounds_.first.z_, coord.z_,
                         minmax_bounds_.second.z_);

  return res;
}

template <typename CoordT>
BoundingBox<CoordT> stack_boxes(std::vector<BoundingBox<CoordT>> &&bbs) {
  const auto bb_count = bbs.size();
  assert(bb_count == 1 || bb_count % 2 == 0);

  if (bb_count == 1)
    return std::move(bbs[0]);

  auto min_max = minmax_coords<CoordT>();

  for (const auto &bb : bbs) {
    update_min(min_max.first, bb.minmax_bounds_.first);
    update_max(min_max.second, bb.minmax_bounds_.second);
  }

  return BoundingBox<CoordT>(std::move(min_max), std::move(bbs));
}
} // namespace sapi

#endif
