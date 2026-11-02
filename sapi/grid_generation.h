/*
 *  grid_generation.h
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

#ifndef GRID_GENERATION_H
#define GRID_GENERATION_H

#include <optional>

#include "gf_collection.h"
#include "tile_grid.h"

namespace sapi {
// Forward definition to link with coordinate_geometry.h
template <typename CoordT>
std::pair<CoordT, CoordT> minmax_vertices(const std::vector<CoordT> &coord_vec);

// Forward definition to link with bounding_box.h
template <typename CoordT>
BoundingBox<CoordT> stack_boxes(std::vector<BoundingBox<CoordT>> &&bbs);

// Forward definition to thread_aligned_array.h
template <typename T,
          typename std::enable_if_t<std::is_copy_constructible_v<T>, bool>>
class TAArray;

template <typename CoordT> struct DimensionalNavigator {
  void initialize(const GridPosition<CoordT> &grid_dimensions);

  std::pair<GridPosition<CoordT>, bool> get_pos_and_advance();

private:
  bool next_ = false;
  GridPosition<CoordT> dimensions_;
  GridPosition<CoordT> current_position_;
};

template <typename CoordT>
inline void DimensionalNavigator<CoordT>::initialize(
    const GridPosition<CoordT> &grid_dimensions) {
  if (!std::all_of(grid_dimensions.cbegin(), grid_dimensions.cend(),
                   positiveTix))
    throw std::invalid_argument("Invalid navigator dimensions");

  next_ = true;
  dimensions_ = grid_dimensions;

  std::fill(current_position_.begin(), current_position_.end(), 0);
}

template <typename CoordT>
std::pair<GridPosition<CoordT>, bool>
DimensionalNavigator<CoordT>::get_pos_and_advance() {
  std::pair<GridPosition<CoordT>, bool> res(current_position_, next_);

  if (next_) {
    auto dim_it = dimensions_.cbegin();
    for (auto pos_it = current_position_.begin();
         pos_it != current_position_.end(); ++pos_it)
      if (++(*pos_it) == *dim_it++)
        *pos_it = 0;
      else
        break;

    // Last advance resets all positions back to 0
    next_ = std::any_of(current_position_.cbegin(), current_position_.cend(),
                        positiveTix);
  }

  return res;
}

template <typename CoordT>
void generate_grid_images(TilePosition<CoordT> &tile_position,
                          const GFCollection<CoordT> *const gf_collection) {
  std::size_t possible_shifts = 1;
  GridPosition<CoordT> shifts_dimensions;
  std::array<std::array<shift_t, 3>, CoordT::D> shifts_array;

  auto dims_it = shifts_dimensions.begin();
  for (auto &shifts : shifts_array) {
    *dims_it++ = 3;
    possible_shifts *= 3;
    shifts = {-1, 0, 1};
  }

  tile_position.tile_images_.resize(possible_shifts);
  tile_position.image_locks_.resize(possible_shifts);
  tile_position.image_displacements_.resize(possible_shifts);
  auto image_it = tile_position.tile_images_.begin();
  auto displacement_it = tile_position.image_displacements_.begin();

  DimensionalNavigator<CoordT> navigator;
  navigator.initialize(shifts_dimensions);

  for (auto position_next_pair = navigator.get_pos_and_advance();
       position_next_pair.second;
       position_next_pair = navigator.get_pos_and_advance()) {
    GridPosition<CoordT> shifted_position = tile_position.position_;

    auto shift_pos_it = shifted_position.begin();
    auto shift_index_it = position_next_pair.first.cbegin();
    auto grid_dim_it = gf_collection->grid_dimensions_.cbegin();
    for (const auto &shifts : shifts_array)
      *shift_pos_it++ += shifts[*shift_index_it++] * (*grid_dim_it++);

    if (shifted_position != tile_position.position_) {
      GridPositionParity<CoordT> shifted_position_parity;
      std::transform(shifted_position.cbegin(), shifted_position.cend(),
                     shifted_position_parity.begin(), evenTix);

      const auto shifted_origin = gf_collection->shift_origin(
          shifted_position, shifted_position_parity);

      *image_it =
          gf_collection->create_tile(shifted_origin, shifted_position_parity);
      *displacement_it = shifted_origin - tile_position.tile_.c_radius_.origin_;
    } else {
      std::iter_swap(image_it, tile_position.tile_images_.begin());
      std::iter_swap(displacement_it,
                     tile_position.image_displacements_.begin());
    }

    ++image_it;
    ++displacement_it;
  }
}

template <typename CoordT>
void insert_neighbor(TilePosition<CoordT> &tile_pos,
                     const GridShift<CoordT> &grid_shift,
                     const GFCollection<CoordT> *const gf_collection) {
  GridPosition<CoordT> neighbor_position;

  bool is_wrapped = false;
  auto gp_it = tile_pos.position_.cbegin();
  auto gd_it = gf_collection->grid_dimensions_.cbegin();
  auto np_it = neighbor_position.begin();
  for (const auto &shift : grid_shift) {
    auto ews =
        edge_wrapped_shift(*gp_it++, static_cast<tileidx_t>(shift), *gd_it++);
    *np_it++ = ews.first;
    is_wrapped |= ews.second;
  }

  if (is_wrapped) {
    tile_pos.wrapped_tile_neighborhood_.insert(
        gf_collection->compute_index(neighbor_position));
  } else {
    const auto index = gf_collection->compute_index(neighbor_position);
    tile_pos.direct_tile_neighborhood_.insert(index);
    tile_pos.wrapped_tile_neighborhood_.insert(index);
  }
}

template <typename CoordT>
void generate_tile_position(std::vector<TilePosition<CoordT>> &positions,
                            const GridPosition<CoordT> &grid_position,
                            const GFCollection<CoordT> *const gf_collection) {
  GridPositionParity<CoordT> position_parity;
  std::transform(grid_position.cbegin(), grid_position.cend(),
                 position_parity.begin(), evenTix);

  TilePosition<CoordT> &tile_pos =
      positions[gf_collection->compute_index(grid_position)];
  tile_pos.position_ = grid_position;
  tile_pos.tile_ = gf_collection->create_tile(
      gf_collection->shift_origin(grid_position, position_parity),
      position_parity);

  generate_grid_images(tile_pos, gf_collection);

  for (const auto &shift : gf_collection->gps_.position_independent_shifts_)
    insert_neighbor(tile_pos, shift, gf_collection);

  auto parity_it = position_parity.cbegin();
  for (const auto &dim_shifts : gf_collection->gps_.position_dependent_shifts_)
    for (const auto &shift :
         *parity_it++ ? dim_shifts.first : dim_shifts.second)
      insert_neighbor(tile_pos, shift, gf_collection);
}

template <typename CoordT>
std::vector<
    std::pair<GridPosition<CoordT>, std::optional<GridPosition<CoordT>>>>
partition_dimensions(const GridPosition<CoordT> &lower_bound,
                     const GridPosition<CoordT> &upper_bound) {
  GridPosition<CoordT> bound_length;
  std::transform(upper_bound.cbegin(), upper_bound.cend(), lower_bound.cbegin(),
                 bound_length.begin(), minusTix);

  bool all_one = true;
  bool some_none = false;
  for (const auto &length : bound_length) {
    all_one &= length == 1;
    some_none |= length < 1;
  }

  assert(!some_none);

  if (all_one)
    return {{lower_bound, std::optional<GridPosition<CoordT>>()}};

  std::size_t possible_combinations = 1;
  GridPosition<CoordT> split_lengths;
  auto sl_it = split_lengths.begin();
  std::array<std::vector<tileidx_t>, CoordT::D> split_combinations;
  auto sc_it = split_combinations.begin();
  for (const auto &length : bound_length) {
    *sc_it = length == 1 ? std::vector<tileidx_t>{0, length}
             : evenTix(length)
                 ? std::vector<tileidx_t>{0, length / 2, length}
                 : std::vector<tileidx_t>{0, (length - 1) / 2, length};

    auto count = (*sc_it++).size();
    possible_combinations *= (count - 1);
    *sl_it++ = static_cast<tileidx_t>(count);
  }

  DimensionalNavigator<CoordT> navigator;
  navigator.initialize(split_lengths);

  std::vector<
      std::pair<GridPosition<CoordT>, std::optional<GridPosition<CoordT>>>>
      dimensional_partitions(possible_combinations);
  auto dim_par = dimensional_partitions.begin();

  for (auto position_next_pair = navigator.get_pos_and_advance();
       position_next_pair.second;
       position_next_pair = navigator.get_pos_and_advance()) {
    GridPosition<CoordT> diagonal_split;
    std::transform(position_next_pair.first.cbegin(),
                   position_next_pair.first.cend(), diagonal_split.begin(),
                   minusOneTix);

    if (std::any_of(diagonal_split.cbegin(), diagonal_split.cend(),
                    negativeTix))
      continue;

    dim_par->first = lower_bound;
    dim_par->second.emplace(lower_bound);

    auto anc_it = dim_par->first.begin();
    auto tar_it = dim_par->second->begin();
    auto ds_it = diagonal_split.cbegin();
    auto sp_it = position_next_pair.first.cbegin();
    for (const auto &split_dimension : split_combinations) {
      *anc_it++ += split_dimension[*ds_it++];
      *tar_it++ += split_dimension[*sp_it++];
    }

    ++dim_par;
  }

  return dimensional_partitions;
}

template <typename CoordT>
BoundingBox<CoordT>
generate_bounding_box_tree(const TileGrid<CoordT> &tile_grid,
                           const GridPosition<CoordT> &grid_lower_bound,
                           const GridPosition<CoordT> &grid_upper_bound) {
  const auto partitioned_bounds =
      partition_dimensions<CoordT>(grid_lower_bound, grid_upper_bound);

  assert(!partitioned_bounds.empty());

  bool all_recursive = true;
  bool all_bounded = true;
  for (const auto &bounds : partitioned_bounds) {
    all_recursive &= bounds.second.has_value();
    all_bounded &= !bounds.second.has_value();
  }

  assert(all_bounded != all_recursive);

  std::vector<BoundingBox<CoordT>> vbb(partitioned_bounds.size());
  auto vbb_it = vbb.begin();

  if (all_recursive)
    for (const auto &bounds : partitioned_bounds)
      *vbb_it++ = generate_bounding_box_tree<CoordT>(tile_grid, bounds.first,
                                                     bounds.second.value());
  else {
    for (const auto &bounds : partitioned_bounds) {
      const auto grid_index =
          position_to_index(bounds.first, tile_grid.dimensions_);
      *vbb_it++ = BoundingBox<CoordT>(
          minmax_vertices(tile_grid.positions_[grid_index].tile_.vertices_),
          grid_index);
    }
  }

  return stack_boxes(std::move(vbb));
}

template <typename CoordT>
TileGrid<CoordT>
generate_tile_grid(const TAArray<GFCollection<CoordT>, true> &gc_array) {
  assert(gc_array.is_initialized());

  const GridPosition<CoordT> &dimensions =
      gc_array.get_local_thread_item()->grid_dimensions_;

  TileGrid<CoordT> grid;
  grid.prepare(dimensions);
  DimensionalNavigator<CoordT> navigator;
  navigator.initialize(dimensions);

#pragma omp parallel default(none) shared(grid, navigator, gc_array)
#pragma omp master
#pragma omp taskgroup
  for (auto position_next_pair = navigator.get_pos_and_advance();
       position_next_pair.second;
       position_next_pair = navigator.get_pos_and_advance())
#pragma omp task default(none) shared(grid, gc_array)                          \
    firstprivate(position_next_pair)
    generate_tile_position(grid.positions_, position_next_pair.first,
                           gc_array.get_local_thread_item());

  GridPosition<CoordT> origin;
  std::fill(origin.begin(), origin.end(), 0);
  grid.bounding_box_ = generate_bounding_box_tree(grid, origin, dimensions);

  return grid;
}
} // namespace sapi

#endif
