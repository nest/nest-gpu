/*
 *  tile2d_geometry.cpp
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

#include <random>

#include "coordinate_geometry.h"
#include "tile.h"
#include "tile2d_geometry.h"
#include "type_erasure_helpers.h"

namespace sapi {
void initialize_vertices_2D(std::vector<Coord2D> &vertices,
                            CircumscribedRadius<Coord2D> &c_radius,
                            const std::vector<space_t> &side_lengths,
                            const angle_t angular_offset,
                            const bool triangular_vertices) {
  assert(vertices.empty());
  for (const auto &side : side_lengths)
    if (almost_zero(side * side) || std::signbit(side))
      throw std::invalid_argument("Invalid 2D side length");

  const auto half_width = side_lengths[0] / 2;
  const auto half_height =
      1 < side_lengths.size() ? side_lengths[1] / 2 : half_width;

  const vertidx_t total_vertices = triangular_vertices ? 3 : 4;
  vertices.reserve(total_vertices);

  if (angular_offset == 0) {
    for (vertidx_t vert = 0; vert < total_vertices; ++vert) {
      const bool sign_width = vert == 1 || vert == 2;
      const bool sign_height = 1 < vert;

      vertices.emplace_back(construct_coord_2D(
          c_radius.origin_.x_ + (sign_width ? -half_width : half_width),
          c_radius.origin_.y_ + (sign_height ? -half_height : half_height)));
    }
  } else {
    const auto rotation = create_angular_offset(angular_offset);
    for (vertidx_t vert = 0; vert < total_vertices; ++vert) {
      const bool sign_width = vert == 1 || vert == 2;
      const bool sign_height = 1 < vert;

      vertices.emplace_back(
          c_radius.origin_ +
          rotate_displacement(
              construct_coord_2D(sign_width ? -half_width : half_width,
                                 sign_height ? -half_height : half_height),
              rotation));
    }
  }

  c_radius.radius2_ = distance2(c_radius.origin_, vertices[0]);
  if (almost_zero(c_radius.radius2_))
    throw std::invalid_argument("Invalid 2D radius");
}

void initialize_hexagon_vertices(std::vector<Coord2D> &vertices,
                                 CircumscribedRadius<Coord2D> &c_radius,
                                 const space_t side_length,
                                 const angle_t angular_offset) {
  assert(vertices.empty());

  c_radius.radius2_ = side_length * side_length;
  if (std::signbit(side_length) || almost_zero(c_radius.radius2_))
    throw std::invalid_argument("Invalid hexagon side length");

  vertices.reserve(6);
  for (angle_t angle_ix = 0; angle_ix < 6; ++angle_ix)
    vertices.emplace_back(
        c_radius.origin_ +
        (create_angular_offset(angle_ix * 60 + angular_offset) * side_length));
}

void initialize_helpers_2D(std::vector<Coord2D> &helper_vectors,
                           std::vector<space_t> &helper_scalars,
                           const std::vector<Coord2D> &vertices) {
  assert((vertices.size() == 3 || vertices.size() == 4) &&
         helper_vectors.empty() && helper_scalars.empty());

  helper_vectors.resize(2);
  helper_vectors[0] = vertices[0] - vertices[1];
  helper_vectors[1] = vertices[2] - vertices[1];

  helper_scalars.resize(
      1, vector_cross(helper_vectors[0], helper_vectors[1]).sum());

  assert(!almost_zero(helper_scalars[0]));
}

void initialize_hexagon_helpers(std::vector<Coord2D> &helper_vectors,
                                std::vector<space_t> &helper_scalars,
                                const std::vector<Coord2D> &vertices) {
  assert(vertices.size() == 6 && helper_vectors.empty() &&
         helper_scalars.empty());

  // Hexagon is divided into rectangles composed of opposing edges
  // used for parallelogram projection comparison
  helper_vectors.reserve(6);
  for (vertidx_t vix = 0; vix < 3; ++vix) {
    helper_vectors.emplace_back(vertices[vix + 1] - vertices[vix]);
    helper_vectors.emplace_back(vertices[(vix + 4) % 6] - vertices[vix]);
  }

  helper_scalars.resize(
      1, vector_cross(helper_vectors[0], helper_vectors[1]).sum());

  assert(!almost_zero(helper_scalars[0]));
}

void split_rectangle(std::vector<Tile<Coord2D>> &sub_tiles,
                     std::vector<const Tile<Coord2D> *> &leaf_tiles,
                     const std::vector<Coord2D> &vertices,
                     const CircumscribedRadius<Coord2D> &c_radius,
                     SplitBranch &split_tree,
                     const std::vector<split_t> &possible_branches,
                     const split_t splits,
                     const bool generate_total_leaves_vector) {
  assert(0 < splits && sub_tiles.empty() && split_tree.children_.empty() &&
         !possible_branches.empty() &&
         leaf_tiles.empty() != generate_total_leaves_vector);

  sub_tiles.reserve(4);
  split_tree.children_.reserve(4);
  const split_t rem_splits = splits - 1;
  const space_t radius2 = c_radius.radius2_ / 4;
  assert(!almost_zero(radius2));

  Coord2D midpoints[4];
  for (vertidx_t vix = 0; vix < 4; ++vix)
    midpoints[vix] = midpoint(vertices[vix], vertices[(vix + 1) % 4]);

  for (split_t partition = 0; partition < 4; ++partition) {
    sub_tiles.emplace_back(Tile<Coord2D>(
        TILE_SHAPE::RECTANGLE,
        construct_circumscribed_radius(
            midpoint(c_radius.origin_, vertices[partition]), radius2),
        {vertices[partition], midpoints[partition], c_radius.origin_,
         midpoints[(partition + 3) % 4]}));

    split_tree.children_.emplace_back(partition, &split_tree);
  }

#pragma omp taskloop num_tasks(strict : 4)                                     \
    mergeable final(rem_splits < 3) default(none)                              \
    shared(sub_tiles, leaf_tiles, split_tree, possible_branches)               \
    firstprivate(rem_splits, generate_total_leaves_vector)
  for (vertidx_t partition = 0; partition < 4; ++partition) {
    const auto st_it = sub_tiles.begin() + partition;
    initialize_helpers_2D(st_it->helper_vectors_, st_it->helper_scalars_,
                          st_it->vertices_);

    if (0 < rem_splits) {
      split_rectangle(st_it->sub_tiles_, leaf_tiles, st_it->vertices_,
                      st_it->c_radius_, split_tree.children_[partition],
                      possible_branches, rem_splits,
                      generate_total_leaves_vector);
    } else {
      st_it->index_ = get_index_from_branches(&split_tree.children_[partition],
                                              possible_branches);

      if (generate_total_leaves_vector) {
        const auto lt_it = leaf_tiles.begin() + st_it->index_;
        assert(lt_it != leaf_tiles.end() && *lt_it == nullptr);
        *lt_it = &(*st_it);
      }
    }
  }
}

void split_triangle(std::vector<Tile<Coord2D>> &sub_tiles,
                    std::vector<const Tile<Coord2D> *> &leaf_tiles,
                    const std::vector<Coord2D> &vertices,
                    SplitBranch &split_tree,
                    const std::vector<split_t> &possible_branches,
                    const split_t splits,
                    const bool generate_total_leaves_vector) {
  assert(0 < splits && sub_tiles.empty() && split_tree.children_.empty() &&
         !possible_branches.empty() &&
         leaf_tiles.empty() != generate_total_leaves_vector);

  sub_tiles.reserve(2);
  split_tree.children_.reserve(2);
  const split_t rem_splits = splits - 1;

  // Get AB, the largest edge of triangle ABC
  const auto sorted_coords = std::get<1>(sort_triangular_vertices(vertices));

  // Get D midpoint of AB,
  // this midpoint is now a vertex of each sub-triangle DCA and DCB,
  // this guarantees each sub-triangle has same area size.
  const Coord2D coordD(midpoint(sorted_coords[0], sorted_coords[1]));

  std::vector<Coord2D> left = {sorted_coords[2], coordD, sorted_coords[0]};
  sub_tiles.emplace_back(Tile<Coord2D>(
      TILE_SHAPE::TRIANGLE, compute_c_radius(left), std::move(left)));
  split_tree.children_.emplace_back(0, &split_tree);

  std::vector<Coord2D> right = {sorted_coords[2], coordD, sorted_coords[1]};
  sub_tiles.emplace_back(Tile<Coord2D>(
      TILE_SHAPE::TRIANGLE, compute_c_radius(right), std::move(right)));
  split_tree.children_.emplace_back(1, &split_tree);

#pragma omp taskloop num_tasks(strict : 2)                                     \
    mergeable final(rem_splits < 5) default(none)                              \
    shared(sub_tiles, leaf_tiles, split_tree, possible_branches)               \
    firstprivate(rem_splits, generate_total_leaves_vector)
  for (vertidx_t partition = 0; partition < 2; ++partition) {
    const auto st_it = sub_tiles.begin() + partition;
    initialize_helpers_2D(st_it->helper_vectors_, st_it->helper_scalars_,
                          st_it->vertices_);

    if (0 < rem_splits) {
      split_triangle(st_it->sub_tiles_, leaf_tiles, st_it->vertices_,
                     split_tree.children_[partition], possible_branches,
                     rem_splits, generate_total_leaves_vector);
    } else {
      st_it->index_ = get_index_from_branches(&split_tree.children_[partition],
                                              possible_branches);

      if (generate_total_leaves_vector) {
        const auto lt_it = leaf_tiles.begin() + st_it->index_;
        assert(lt_it != leaf_tiles.end() && *lt_it == nullptr);
        *lt_it = &(*st_it);
      }
    }
  }
}

void split_hexagon(std::vector<Tile<Coord2D>> &sub_tiles,
                   std::vector<const Tile<Coord2D> *> &leaf_tiles,
                   const std::vector<Coord2D> &vertices,
                   const CircumscribedRadius<Coord2D> &c_radius,
                   SplitBranch &split_tree,
                   const std::vector<split_t> &possible_branches,
                   const split_t splits,
                   const bool generate_total_leaves_vector) {
  assert(0 < splits && sub_tiles.empty() && split_tree.children_.empty() &&
         !possible_branches.empty() &&
         leaf_tiles.empty() != generate_total_leaves_vector);

  sub_tiles.reserve(6);
  split_tree.children_.reserve(6);
  const split_t rem_splits = splits - 1;
  const space_t radius2 = c_radius.radius2_ / 3;
  assert(!almost_zero(radius2));

  for (split_t partition = 0; partition < 6; ++partition) {
    const split_t next_partition = (partition + 1) % 6;

    sub_tiles.emplace_back(Tile<Coord2D>(
        TILE_SHAPE::TRIANGLE,
        construct_circumscribed_radius(
            (c_radius.origin_ / 3) +
                ((vertices[partition] + vertices[next_partition]) / 3),
            radius2),
        {c_radius.origin_, vertices[partition], vertices[next_partition]}));

    split_tree.children_.emplace_back(partition, &split_tree);
  }

#pragma omp taskloop num_tasks(strict : 3)                                     \
    mergeable final(rem_splits < 4) default(none)                              \
    shared(sub_tiles, leaf_tiles, split_tree, possible_branches)               \
    firstprivate(rem_splits, generate_total_leaves_vector)
  for (split_t partition = 0; partition < 6; ++partition) {
    const auto st_it = sub_tiles.begin() + partition;
    initialize_helpers_2D(st_it->helper_vectors_, st_it->helper_scalars_,
                          st_it->vertices_);

    if (0 < rem_splits) {
      split_triangle(st_it->sub_tiles_, leaf_tiles, st_it->vertices_,
                     split_tree.children_[partition], possible_branches,
                     rem_splits, generate_total_leaves_vector);
    } else {
      st_it->index_ = get_index_from_branches(&split_tree.children_[partition],
                                              possible_branches);

      if (generate_total_leaves_vector) {
        const auto lt_it = leaf_tiles.begin() + st_it->index_;
        assert(lt_it != leaf_tiles.end() && *lt_it == nullptr);
        *lt_it = &(*st_it);
      }
    }
  }
}

bool coord_in_rectangle(const Coord2D &coord,
                        const CircumscribedRadius<Coord2D> &c_radius,
                        const std::vector<Coord2D> &vertices,
                        const std::vector<Coord2D> &helper_vectors,
                        const std::vector<space_t> &helper_scalars) {
  if (!c_radius.coord_in_radius(coord).first)
    return false;

  // Only works for squares or triangles
  return algebraic_projection_comparison<Coord2D, false>(
      coord - vertices[1], helper_vectors[0], helper_vectors[1],
      helper_scalars[0]);
}

bool coord_in_triangle(const Coord2D &coord,
                       const CircumscribedRadius<Coord2D> &c_radius,
                       const std::vector<Coord2D> &vertices,
                       const std::vector<Coord2D> &helper_vectors,
                       const std::vector<space_t> &helper_scalars) {
  if (!c_radius.coord_in_radius(coord).first)
    return false;

  // Only works for squares or triangles
  return algebraic_projection_comparison<Coord2D, true>(
      coord - vertices[1], helper_vectors[0], helper_vectors[1],
      helper_scalars[0]);
}

bool coord_in_hexagon(const Coord2D &coord,
                      const CircumscribedRadius<Coord2D> &c_radius,
                      const std::vector<Coord2D> &vertices,
                      const std::vector<Coord2D> &helper_vectors,
                      const std::vector<space_t> &helper_scalars) {
  if (!c_radius.coord_in_radius(coord).first)
    return false;

  // Parallelogram based comparison using rotating rectangles
  bool res = false;
  vertidx_t rel_vix;
  for (vertidx_t vix = 0; vix < 3; ++vix) {
    rel_vix = 2 * vix;
    res = algebraic_projection_comparison<Coord2D, false>(
        coord - vertices[vix], helper_vectors[rel_vix],
        helper_vectors[rel_vix + 1], helper_scalars[0]);
    if (res)
      break;
  }

  return res;
}

std::vector<std::pair<nodeidx_t, Coord2D>>
generate_coords_in_rectangle(const nodeidx_t first_index,
                             const nodeidx_t coord_count, AnyRNG &rng,
                             const std::vector<Coord2D> &vertices,
                             const std::vector<Coord2D> &helper_vectors) {
  assert(0 <= first_index && 0 < coord_count);
  std::vector<std::pair<nodeidx_t, Coord2D>> vec;
  vec.reserve(coord_count);

  space_t m0, m1;
  std::uniform_real_distribution<space_t> dist(0., 1.);
  for (nodeidx_t idx = 0; idx < coord_count; ++idx) {
    m0 = static_cast<space_t>(dist(rng));
    m1 = static_cast<space_t>(dist(rng));
    vec.emplace_back(std::make_pair(
        first_index + idx,
        construct_coord_2D(vertices[1].x_ + helper_vectors[0].x_ * m0 +
                               helper_vectors[1].x_ * m1,
                           vertices[1].y_ + helper_vectors[0].y_ * m0 +
                               helper_vectors[1].y_ * m1)));
  }

  return vec;
}

std::vector<std::pair<nodeidx_t, Coord2D>>
generate_coords_in_triangle(const nodeidx_t first_index,
                            const nodeidx_t coord_count, AnyRNG &rng,
                            const std::vector<Coord2D> &vertices) {
  assert(0 <= first_index && 0 < coord_count);
  std::vector<std::pair<nodeidx_t, Coord2D>> vec;
  vec.reserve(coord_count);

  space_t m0, m1, m2;
  std::uniform_real_distribution<space_t> dist(0., 1.);
  for (nodeidx_t idx = 0; idx < coord_count; ++idx) {
    m0 = static_cast<space_t>(dist(rng));
    m1 = static_cast<space_t>(dist(rng));
    const bool adjust = std::isless(1., m0 + m1);
    m0 = adjust * std::fmax(std::fmin(1., 1. - m0), 0.) + !adjust * m0;
    m1 = adjust * std::fmax(std::fmin(1., 1. - m1), 0.) + !adjust * m1;
    m2 = std::fmax(std::fmin(1., 1. - m0 - m1), 0.);
    vec.emplace_back(std::make_pair(
        first_index + idx,
        construct_coord_2D(
            vertices[0].x_ * m0 + vertices[1].x_ * m1 + vertices[2].x_ * m2,
            vertices[0].y_ * m0 + vertices[1].y_ * m1 + vertices[2].y_ * m2)));
  }

  return vec;
}

std::vector<std::pair<nodeidx_t, Coord2D>>
generate_coords_in_hexagon(const nodeidx_t first_index,
                           const nodeidx_t coord_count, AnyRNG &rng,
                           const std::vector<Coord2D> &vertices,
                           const CircumscribedRadius<Coord2D> &c_radius) {
  assert(0 <= first_index && 0 < coord_count);
  std::vector<std::pair<nodeidx_t, Coord2D>> vec;
  vec.reserve(coord_count);

  vertidx_t hex_idx;
  space_t m0, m1, m2;
  std::uniform_real_distribution<space_t> space_dist(0., 1.);
  std::uniform_int_distribution<vertidx_t> partition_dist(0, 5);
  for (nodeidx_t idx = 0; idx < coord_count; ++idx) {
    m0 = static_cast<space_t>(space_dist(rng));
    m1 = static_cast<space_t>(space_dist(rng));
    const bool adjust = std::isless(1., m0 + m1);
    m0 = adjust * std::fmax(std::fmin(1., 1. - m0), 0.) + !adjust * m0;
    m1 = adjust * std::fmax(std::fmin(1., 1. - m1), 0.) + !adjust * m1;
    m2 = std::fmax(std::fmin(1., 1. - m0 - m1), 0.);
    hex_idx = partition_dist(rng);
    vec.emplace_back(std::make_pair(
        first_index + idx,
        construct_coord_2D(
            c_radius.origin_.x_ * m0 + vertices[hex_idx].x_ * m1 +
                vertices[(hex_idx + 1) % 6].x_ * m2,
            c_radius.origin_.y_ * m0 + vertices[hex_idx].y_ * m1 +
                vertices[(hex_idx + 1) % 6].y_ * m2)));
  }

  return vec;
}

Coord2D project_point_to_2D_perimeter(const Coord2D &coord,
                                      const std::vector<Coord2D> &vertices) {
  vertidx_t v_index = 0;
  vertidx_t first = 0;
  space_t d_first = std::numeric_limits<space_t>::max();
  vertidx_t second = 0;
  space_t d_second = std::numeric_limits<space_t>::max();
  for (const auto &vertex : vertices) {
    const auto d2 = distance2(coord, vertex);
    if (std::isless(d2, d_first)) {
      second = first;
      d_second = d_first;
      first = v_index;
      d_first = d2;
    } else if (std::isless(d2, d_second)) {
      second = v_index;
      d_second = d2;
    }
    ++v_index;
  }

  if (almost_zero(d_first))
    return vertices[first];

  return projection_coord<Coord2D, true>(coord, vertices[first],
                                         vertices[second]);
}

Coord2D project_point_to_triangle_perimeter(
    const Coord2D &coord, const std::vector<Coord2D> &vertices,
    const CircumscribedRadius<Coord2D> &c_radius) {
  if (almost_zero(distance2(coord, c_radius.origin_)))
    return c_radius.origin_;

  vertidx_t v_index = 0;
  vertidx_t first = 0;
  space_t d_first = std::numeric_limits<space_t>::max();
  vertidx_t second = 0;
  space_t d_second = std::numeric_limits<space_t>::max();
  for (const auto &vertex : vertices) {
    const auto d2 = distance2(coord, vertex);
    if (std::isless(d2, d_first)) {
      second = first;
      d_second = d_first;
      first = v_index;
      d_first = d2;
    } else if (std::isless(d2, d_second)) {
      second = v_index;
      d_second = d2;
    }
    ++v_index;
  }

  if (almost_zero(d_first))
    return vertices[first];

  return projection_coord<Coord2D, true>(coord, vertices[first],
                                         vertices[second]);
}
} // namespace sapi
