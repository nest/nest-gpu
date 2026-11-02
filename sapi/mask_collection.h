/*
 *  mask_collection.h
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

#ifndef MASK_COLLECTION_H
#define MASK_COLLECTION_H

#include "mask.h"

namespace sapi {
// Forward definition to creator_registry.h
template <typename RT> class CreatorRegistry;

template <typename CoordT> struct MaskCollection {
  Mask<CoordT> blueprint_;
  Mask<CoordT> source_mask_;
  Mask<CoordT> target_mask_;

  bool has_blueprint() const;
  bool has_source_mask() const;
  bool has_target_mask() const;

  OptDisp<CoordT> blueprint_overlap(const CoordT &a, const CoordT &b) const;

  bool blueprint_overlap(const Tile<CoordT> &a, const Tile<CoordT> &b) const;

  bool blueprint_overlap(const CoordT &a, const Tile<CoordT> &b) const;

  OptDisp<CoordT> source_overlap(const CoordT &c) const;

  bool source_overlap(const Tile<CoordT> &t) const;

  std::deque<const Tile<CoordT> *>
  source_overlapping_leafs(const Tile<CoordT> &t) const;

  OptDisp<CoordT> target_overlap(const CoordT &c) const;

  bool target_overlap(const Tile<CoordT> &t) const;

  std::deque<const Tile<CoordT> *>
  target_overlapping_leafs(const Tile<CoordT> &t) const;
};

template <typename CoordT>
inline MaskCollection<CoordT>
create_mask_collection(const std::string &blueprint_name,
                       const std::vector<space_t> &blueprint_params,
                       const std::vector<space_t> &blueprint_offset,
                       const std::string &source_mask_name,
                       const std::vector<space_t> &source_mask_origin,
                       const std::vector<space_t> &source_mask_params,
                       const std::vector<space_t> &source_mask_offset,
                       const std::string &target_mask_name,
                       const std::vector<space_t> &target_mask_origin,
                       const std::vector<space_t> &target_mask_params,
                       const std::vector<space_t> &target_mask_offset,
                       const CreatorRegistry<Mask<CoordT>> &mc_registry) {
  MaskCollection<CoordT> mc;
  if (!blueprint_name.empty())
    mc.blueprint_ = mc_registry.get_creator(blueprint_name)
                        ->create({}, blueprint_params, blueprint_offset);
  if (!source_mask_name.empty())
    mc.source_mask_ = mc_registry.get_creator(source_mask_name)
                          ->create(source_mask_origin, source_mask_params,
                                   source_mask_offset);
  if (!target_mask_name.empty())
    mc.target_mask_ = mc_registry.get_creator(target_mask_name)
                          ->create(target_mask_origin, target_mask_params,
                                   target_mask_offset);
  return mc;
}

template <typename CoordT>
inline bool MaskCollection<CoordT>::has_blueprint() const {
  return blueprint_.shape_ != MASK_SHAPE::NULL_MS;
}

template <typename CoordT>
inline bool MaskCollection<CoordT>::has_source_mask() const {
  return source_mask_.shape_ != MASK_SHAPE::NULL_MS;
}

template <typename CoordT>
inline bool MaskCollection<CoordT>::has_target_mask() const {
  return target_mask_.shape_ != MASK_SHAPE::NULL_MS;
}

template <typename CoordT>
inline OptDisp<CoordT>
MaskCollection<CoordT>::blueprint_overlap(const CoordT &a,
                                          const CoordT &b) const {
  return blueprint_.coord_in_mask(a, b);
}

template <typename CoordT>
inline bool
MaskCollection<CoordT>::blueprint_overlap(const Tile<CoordT> &a,
                                          const Tile<CoordT> &b) const {
  return blueprint_.tiles_within_mask_range(a, b);
}

template <typename CoordT>
inline bool
MaskCollection<CoordT>::blueprint_overlap(const CoordT &a,
                                          const Tile<CoordT> &b) const {
  return blueprint_.overlap_with_tile_surface(a, b);
}

template <typename CoordT>
inline OptDisp<CoordT>
MaskCollection<CoordT>::source_overlap(const CoordT &c) const {
  return source_mask_.coord_in_mask(c);
}

template <typename CoordT>
inline bool
MaskCollection<CoordT>::source_overlap(const Tile<CoordT> &t) const {
  return source_mask_.tile_overlap(t) != OVERLAP_LEVEL::NONE;
}

template <typename CoordT>
inline std::deque<const Tile<CoordT> *>
MaskCollection<CoordT>::source_overlapping_leafs(const Tile<CoordT> &t) const {
  return source_mask_.get_overlapping_leaf_sub_tiles(t);
}

template <typename CoordT>
inline OptDisp<CoordT>
MaskCollection<CoordT>::target_overlap(const CoordT &c) const {
  return target_mask_.coord_in_mask(c);
}

template <typename CoordT>
inline bool
MaskCollection<CoordT>::target_overlap(const Tile<CoordT> &t) const {
  return target_mask_.tile_overlap(t) != OVERLAP_LEVEL::NONE;
}

template <typename CoordT>
inline std::deque<const Tile<CoordT> *>
MaskCollection<CoordT>::target_overlapping_leafs(const Tile<CoordT> &t) const {
  return target_mask_.get_overlapping_leaf_sub_tiles(t);
}
} // namespace sapi

#endif
