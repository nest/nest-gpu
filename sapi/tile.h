/*
 *  tile.h
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

#ifndef TILE_H
#define TILE_H

#include <cassert>
#include <deque>

#include "tile_geometry.h"

namespace sapi {
template <typename CoordT> struct Tile {
  TILE_SHAPE shape_ = TILE_SHAPE::NULL_TS;
  tileidx_t index_ = -1;
  CircumscribedRadius<CoordT> c_radius_;
  std::vector<CoordT> vertices_;
  std::vector<CoordT> helper_vectors_;
  std::vector<space_t> helper_scalars_;

  // Structures for spliting
  std::vector<Tile> sub_tiles_;
  std::vector<const Tile *> leaf_tiles_;

  Tile() noexcept = default;
  Tile(const Tile &) = delete;
  Tile(Tile &&) noexcept = default;
  ~Tile() noexcept = default;

  Tile(const TILE_SHAPE shape, const CoordT origin,
       const std::vector<space_t> &side_lengths,
       const std::vector<angle_t> &angular_offsets);

  Tile(TILE_SHAPE &&shape, CircumscribedRadius<CoordT> &&c_radius,
       std::vector<CoordT> &&vertices) noexcept;

  Tile &operator=(const Tile &) = delete;
  Tile &operator=(Tile &&) noexcept;

  void split(const split_t splits, const bool generate_total_leaves_vector);

  bool coord_in_tile(const CoordT &coord) const;

  std::vector<std::pair<nodeidx_t, CoordT>>
  generate_coords_in_tile(const nodeidx_t first_index,
                          const nodeidx_t coord_count, AnyRNG &rng) const;

  CoordT project_point_to_surface(const CoordT &coord) const;

  tileidx_t compute_leaves_count(const split_t known_splits) const;

  std::vector<split_t>
  get_possible_sub_tile_branches(const split_t known_splits) const;

  void insert_leaf_sub_tiles(std::deque<const Tile *> &leaves) const;

  bool operator==(const Tile &) const;

  std::string to_string(const uint8_t tabs = 0) const;
};

template <typename CoordT>
Tile<CoordT>::Tile(const TILE_SHAPE shape, const CoordT origin,
                   const std::vector<space_t> &side_lengths,
                   const std::vector<angle_t> &angular_offsets)
    : shape_(shape) {
  if (side_lengths.empty())
    throw std::invalid_argument("Invalid tile side lengths vector");

  c_radius_.origin_ = origin;

  initialize_tile_vertices(vertices_, c_radius_, side_lengths, angular_offsets,
                           shape_);
  initialize_tile_helpers(helper_vectors_, helper_scalars_, vertices_, shape_);
}

template <typename CoordT>
Tile<CoordT>::Tile(TILE_SHAPE &&shape, CircumscribedRadius<CoordT> &&c_radius,
                   std::vector<CoordT> &&vertices) noexcept
    : shape_(shape), c_radius_(std::move(c_radius)),
      vertices_(std::move(vertices)) {}

template <typename CoordT>
inline Tile<CoordT> &Tile<CoordT>::operator=(Tile &&t) noexcept {
  shape_ = t.shape_;
  index_ = t.index_;
  c_radius_ = t.c_radius_;

  vertices_.swap(t.vertices_);
  t.vertices_.clear();

  helper_vectors_.swap(t.helper_vectors_);
  t.helper_vectors_.clear();

  helper_scalars_.swap(t.helper_scalars_);
  t.helper_scalars_.clear();

  sub_tiles_.swap(t.sub_tiles_);
  t.sub_tiles_.clear();

  leaf_tiles_.swap(t.leaf_tiles_);
  t.leaf_tiles_.clear();

  t.shape_ = TILE_SHAPE::NULL_TS;
  t.index_ = -1;

  return *this;
}

template <typename CoordT>
inline void Tile<CoordT>::split(const split_t splits,
                                const bool generate_total_leaves_vector) {
  static_assert(std::is_unsigned_v<split_t>);

  // Leaf tiles will meet first condition
  // intermediate sub tiles second
  // root tiles last
  if (0 <= index_ || !sub_tiles_.empty() || !leaf_tiles_.empty() ||
      shape_ == TILE_SHAPE::NULL_TS)
    return;

  if (0 < splits) {
    SplitBranch split_tree;
    const auto possible_branches =
        sapi::get_possible_sub_tile_branches(splits, shape_);

    if (generate_total_leaves_vector) {
      const auto num_leaves = compute_leaves_count(splits);
      assert(0 < num_leaves);
      leaf_tiles_.resize(num_leaves, nullptr);
    }

    sapi::split_tile(sub_tiles_, leaf_tiles_, vertices_, c_radius_, split_tree,
                     possible_branches, splits, generate_total_leaves_vector,
                     shape_);
  } else {
    index_ = 0;

    if (generate_total_leaves_vector)
      leaf_tiles_.resize(1, this);
  }
}

template <typename CoordT>
inline bool Tile<CoordT>::coord_in_tile(const CoordT &coord) const {
  return sapi::coord_in_tile(coord, c_radius_, vertices_, helper_vectors_,
                             helper_scalars_, shape_);
}

template <typename CoordT>
inline std::vector<std::pair<nodeidx_t, CoordT>>
Tile<CoordT>::generate_coords_in_tile(const nodeidx_t first_index,
                                      const nodeidx_t coord_count,
                                      AnyRNG &rng) const {
  return sapi::generate_coords_in_tile(first_index, coord_count, rng, vertices_,
                                       helper_vectors_, c_radius_, shape_);
}

template <typename CoordT>
inline CoordT
Tile<CoordT>::project_point_to_surface(const CoordT &coord) const {
  return sapi::project_point_to_surface(coord, vertices_, c_radius_, shape_);
}

template <typename CoordT>
inline tileidx_t
Tile<CoordT>::compute_leaves_count(const split_t known_splits) const {
  return sapi::compute_leaves_count(known_splits, shape_);
}

template <typename CoordT>
inline std::vector<split_t>
Tile<CoordT>::get_possible_sub_tile_branches(const split_t known_splits) const {
  return sapi::get_possible_sub_tile_branches(known_splits, shape_);
}

template <typename CoordT>
void Tile<CoordT>::insert_leaf_sub_tiles(
    std::deque<const Tile *> &leaves) const {
  if (!leaf_tiles_.empty()) {
    for (const auto &lt : leaf_tiles_)
      leaves.emplace_back(lt);
  } else {
    std::deque<const Tile *> tree{this};
    do {
      auto tile = tree.front();
      tree.pop_front();

      if (!tile->sub_tiles_.empty())
        for (const auto &st : tile->sub_tiles_)
          tree.emplace_back(&st);
      else
        leaves.emplace_back(tile);

    } while (!tree.empty());
  }
}

template <typename CoordT>
std::string Tile<CoordT>::to_string(const uint8_t tabs) const {
  std::string str = "";
  for (uint8_t t = 0; t < tabs; ++t)
    str += "\t";

  str += std::string(TILE_SHAPE_NAMES[uint8_t(shape_)]) +
         " - index: " + std::to_string(index_) +
         ", properties: " + c_radius_.to_string() + " , vertices: ";

  for (const auto &vt : vertices_)
    str += vt.to_string();

  for (const auto &st : sub_tiles_)
    str += "\n" + st.to_string(tabs + 1);

  return str;
}

template <typename CoordT>
inline bool Tile<CoordT>::operator==(const Tile &t) const {
  // The circumscribed radius and the vertex vectors
  // are enough to uniquely identify a tile
  // sub tiles are generated from the given properties of a tile
  return shape_ == t.shape_ && c_radius_ == t.c_radius_ &&
         vertices_ == t.vertices_;
}
} // namespace sapi

#endif
