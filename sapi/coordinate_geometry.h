/*
 *  coordinate_geometry.h
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

#ifndef COORDINATE_GEOMETRY_H
#define COORDINATE_GEOMETRY_H

#include <array>
#include <cassert>
#include <tuple>
#include <type_traits>
#include <vector>

#include "coordinates.h"

namespace sapi {
template <typename T>
using rcvref = typename std::remove_cv_t<typename std::remove_reference_t<T>>;

template <typename CoordT>
inline space_t vector_dot(const CoordT &coordA, const CoordT &coordB) {
  return (coordA * coordB).sum();
}

inline Coord2D vector_cross(const Coord2D &coordA, const Coord2D &coordB) {
  return construct_coord_2D(coordA.x_ * coordB.y_, -(coordA.y_ * coordB.x_));
}

inline Coord3D vector_cross(const Coord3D &coordA, const Coord3D &coordB) {
  return construct_coord_3D(
      vector_cross(construct_coord_2D(coordA.y_, coordB.y_),
                   construct_coord_2D(coordA.z_, coordB.z_))
          .sum(),
      -vector_cross(construct_coord_2D(coordA.x_, coordB.x_),
                    construct_coord_2D(coordA.z_, coordB.z_))
           .sum(),
      vector_cross(construct_coord_2D(coordA.x_, coordB.x_),
                   construct_coord_2D(coordA.y_, coordB.y_))
          .sum());
}

template <typename CoordT> inline space_t vector_norm2(const CoordT &coord) {
  return vector_dot(coord, coord);
}

template <typename CoordT> inline space_t vector_norm(const CoordT &coord) {
  return std::sqrt(vector_norm2(coord));
}

template <typename CoordT> inline CoordT normalize_vector(const CoordT &coord) {
  return coord.is_null() ? CoordT() : coord / vector_norm(coord);
}

template <typename CoordT>
inline space_t distance2(const CoordT &coordA, const CoordT &coordB) {
  return vector_norm2(coordA - coordB);
}

template <typename CoordT>
inline space_t distance(const CoordT &coordA, const CoordT &coordB) {
  return std::sqrt(distance2(coordA, coordB));
}

template <typename CoordT>
inline CoordT midpoint(const CoordT &coordA, const CoordT &coordB) {
  return (coordA + coordB) / 2;
}

template <typename CoordT, bool clamp>
space_t projection_coeff(const CoordT &vectorBA, const CoordT &vectorBC) {
  // We project BA onto BC using orthogonal vector projection.
  const space_t dot_BA_BC = vector_dot(vectorBA, vectorBC);
  const space_t n2BC = vector_norm2(vectorBC);

  const bool n2_BC_az = almost_zero(n2BC);
  const bool dot_BA_BC_az = almost_zero(dot_BA_BC);
  const bool dot_BA_BC_aeq_n2BC = almost_equal(dot_BA_BC, n2BC);
  assert(!n2_BC_az || (dot_BA_BC_az && n2_BC_az));

  switch ((dot_BA_BC_az << 0) + (dot_BA_BC_aeq_n2BC << 1)) {
  case 1:
    return 0;

  case 2:
    return 1;

  case 3:
    return 0;

  default:
    // Safe to divide at this point
    if constexpr (clamp)
      return std::fmin(std::fmax(dot_BA_BC / n2BC, 0.), 1.);
    else
      return dot_BA_BC / n2BC;
  }
}

template <typename CoordT, bool clamp>
inline CoordT projection_vector(const CoordT &vectorBA,
                                const CoordT &vectorBC) {
  // We project BA onto BC using orthogonal vector projection.
  return vectorBC * projection_coeff<CoordT, clamp>(vectorBA, vectorBC);
}

template <typename CoordT, bool clamp>
inline CoordT projection_coord(const CoordT &coordA, const CoordT &coordB,
                               const CoordT &coordC) {
  // We project A onto BC using orthogonal vector projection.
  return coordB +
         projection_vector<CoordT, clamp>(coordA - coordB, coordC - coordB);
}

inline Coord2D create_angular_offset(const space_t angle) {
  const space_t rad_angle = radians(angle);
  return construct_coord_2D(std::cos(rad_angle), std::sin(rad_angle));
}

inline Coord2D rotate_displacement(const Coord2D &displacement,
                                   const Coord2D &angular_coord) {
  // With X = x * cos @ + y * sin @
  // and  Y = -x * sin @ + y * cos @
  return construct_coord_2D((angular_coord * displacement).sum(),
                            vector_cross(angular_coord, displacement).sum());
}

template <typename CoordT, bool triangular_comparison>
bool algebraic_projection_comparison(const CoordT &vectorBA,
                                     const CoordT &vectorBC,
                                     const CoordT &vectorBD,
                                     const space_t det_BCD) {
  // Solving comparison as equation system form
  // with pre-computed determinant
  const space_t cross_ABC = vector_cross(vectorBC, vectorBA).sum();
  const space_t cross_ABD = vector_cross(vectorBA, vectorBD).sum();

  const bool cross_ABC_az = almost_zero(cross_ABC);
  const bool cross_ABC_aeq_det_BCD = almost_equal(cross_ABC, det_BCD);
  assert(!(cross_ABC_az && cross_ABC_aeq_det_BCD));

  const bool cross_ABD_az = almost_zero(cross_ABD);
  const bool cross_ABD_aeq_det_BCD = almost_equal(cross_ABD, det_BCD);
  assert(!(cross_ABD_az && cross_ABD_aeq_det_BCD));

  const space_t coeffABC =
      !cross_ABC_az *
      (!cross_ABC_aeq_det_BCD * cross_ABC / det_BCD + cross_ABC_aeq_det_BCD);
  const space_t coeffABD =
      !cross_ABD_az *
      (!cross_ABD_aeq_det_BCD * cross_ABD / det_BCD + cross_ABD_aeq_det_BCD);

  switch ((cross_ABC_az << 0) + (cross_ABD_az << 1) +
          (cross_ABC_aeq_det_BCD << 2) + (cross_ABD_aeq_det_BCD << 3)) {
  case 1:
    return std::isless(0, coeffABD) && std::isless(coeffABD, 1);

  case 2:
    return std::isless(0, coeffABC) && std::isless(coeffABC, 1);

  case 3:
    return true;

  case 4:
    return !triangular_comparison && std::isless(0, coeffABD) &&
           std::isless(coeffABD, 1);

  case 8:
    return !triangular_comparison && std::isless(0, coeffABC) &&
           std::isless(coeffABC, 1);

  case 12:
    return !triangular_comparison;

  case 6:
    return true;

  case 9:
    return true;

  default:
    return std::isless(0, coeffABC) && std::isless(coeffABC, 1) &&
           std::isless(0, coeffABD) && std::isless(coeffABD, 1) &&
           leq_test(coeffABC + coeffABD, 2 - triangular_comparison);
  }
}

template <typename CoordT>
std::tuple<vertidx_t, std::array<CoordT, 3>, std::array<CoordT, 3>,
           std::array<space_t, 3>>
sort_triangular_vertices(const std::vector<CoordT> &coord_in) {
  assert(coord_in.size() == 3);

  vertidx_t A_vix = 0, B_vix = 0, C_vix = 0, vix_next = 0, equal_lengths = 0;
  space_t largest_length = 0, prev_length = 0;

  CoordT indexed_vectors[3];
  space_t indexed_lengths[3];
  bool almost_equal_lengths[3] = {false};
  for (vertidx_t vix = 0; vix < 3; ++vix) {
    vix_next = (vix + 1) % 3;
    indexed_vectors[vix] = coord_in[vix_next] - coord_in[vix];
    indexed_lengths[vix] = vector_norm2(indexed_vectors[vix]);
    assert(!almost_zero(indexed_lengths[vix]));

    if (0 < vix && almost_equal(prev_length, indexed_lengths[vix])) {
      almost_equal_lengths[vix - 1] = true;
      ++equal_lengths;
    } else if (0 == vix || std::isless(largest_length, indexed_lengths[vix])) {
      A_vix = vix;
      B_vix = vix_next;
      largest_length = indexed_lengths[vix];
    }

    prev_length = indexed_lengths[vix];
  }
  almost_equal_lengths[2] = almost_equal(prev_length, indexed_lengths[0]);
  equal_lengths += almost_equal_lengths[2];
  C_vix = (B_vix + 1) % 3;

  // Due to commutative comparison equal lengths == 2 implies equal lengths == 3
  equal_lengths += equal_lengths == 2;

  // A triangle either starts as a right triangle from a square
  // i.e. isosceles right triangle or an equilateral triangle.
  // When spliting an isosceles right triangle, symmetry is preserved
  // and resulting triangles are isosceles right triangles too.
  // When splitting an equilateral triangle resulting triangles
  // are two right triangles with no equal lengths.
  // For right triangles with no equal lengths we return
  // the hypotenuse vertices and apex vertex in that order.
  // For an equilateral triangle any combination is fine.
  // For an isosceles triangle (right or not) we return the base vertices
  // and apex vertex in that order.
  if (equal_lengths == 1) {
    // Get end vertex of base edge of isosceles triangle
    for (B_vix = 0; B_vix < 3; ++B_vix)
      if (almost_equal_lengths[B_vix])
        break;

    // AB edge is base of isosceles
    A_vix = (B_vix + 2) % 3;
    // CB edge is one of the two equal sides
    C_vix = (B_vix + 1) % 3;
  }

  return std::make_tuple(
      equal_lengths,
      std::array<CoordT, 3>{coord_in[A_vix], coord_in[B_vix], coord_in[C_vix]},
      std::array<CoordT, 3>{indexed_vectors[A_vix], indexed_vectors[B_vix],
                            indexed_vectors[C_vix]},
      std::array<space_t, 3>{indexed_lengths[A_vix], indexed_lengths[B_vix],
                             indexed_lengths[C_vix]});
}

template <typename CoordT>
CircumscribedRadius<CoordT>
compute_c_radius(const std::vector<CoordT> &coords) {
  assert(coords.size() == 3);

  CircumscribedRadius<CoordT> c_radius;

  const auto [equal_lengths, sorted_coords, sorted_vectors, sorted_lengths] =
      sort_triangular_vertices(coords);

  const CoordT base_vertices_add(sorted_coords[0] + sorted_coords[1]);

  if (equal_lengths == 3) {
    // Implies equilateral triangle, circumscribed center is
    // a scaled orthogonal projection of apex vertex onto base edge
    // here apex vertex and base edge are an arbitrary choice

    // With side length a, radius r is equal to a / sqrt( 3 )
    c_radius.radius2_ = sorted_lengths[0] / 3;

    // With side length a, height h is equal to sqrt( 3 ) * a / 2
    // projection coefficient of apex vertex onto base edge is r / h == 2 / 3
    // Using vector combining formula of apex + ( midlepoint of base edge - apex
    // ) * coefficient we reduce to apex * ( 1 - coefficient ) + ( base edge *
    // coefficient / 2 ) with coefficient == 2 / 3 we have apex / 3 + base edge
    // / 3
    c_radius.origin_ = (sorted_coords[2] / 3) + (base_vertices_add / 3);
  } else if (almost_zero(vector_dot(sorted_vectors[1], sorted_vectors[2]))) {
    // Right triangle case, circumscribed center is the midpoint of the
    // hypotenuse hypotenuse == base and radius == | base | / 2
    c_radius.origin_ = base_vertices_add / 2;
    c_radius.radius2_ = sorted_lengths[0] / 4;
  } else if (equal_lengths == 1) {
    // Plain isosceles case, circumscribed center is
    // a scaled orthogonal projection of apex vertex onto base edge

    // With base length b and side length a
    // height h is equal to sqrt( a2 - ( b2 / 4 ) )
    // => 4 * h2 = 4 * a2 - b2
    const space_t height2_4 =
        std::fma(4., sorted_lengths[2], -sorted_lengths[0]);

    // With base height h and side length a
    // radius r is equal to a2 / ( 2 * h )
    //  => radius2 = a4 / ( 4 * h2 )
    c_radius.radius2_ = (sorted_lengths[2] * sorted_lengths[2]) / height2_4;

    // Projection coefficient of apex vertex onto base edge is r / h
    const space_t projection_coefficient = sorted_lengths[2] / (height2_4 / 2);

    c_radius.origin_ = sorted_coords[2] * (1. - projection_coefficient) +
                       base_vertices_add * (projection_coefficient / 2);
  } else {
    // Compute 2D circumscribed center via equation system
    // to find equidistant point to all vertices
    // solved using Cramer's rule
    // Compute 3D circumscribed center by projecting 3D
    // coordinates onto 2D plane, then using 2D equation
    // system, and finally projecting back to 3D space
    c_radius.origin_ = circumscribed_center(coords);
    c_radius.radius2_ = distance2(c_radius.origin_, sorted_coords[0]);
  }

  assert(!almost_zero(c_radius.radius2_));

  return c_radius;
}

template <typename CoordT>
inline void update_min(CoordT &min, const CoordT &value) {
  min.x_ = std::isless(value.x_, min.x_) ? value.x_ : min.x_;
  min.y_ = std::isless(value.y_, min.y_) ? value.y_ : min.y_;

  if constexpr (std::is_same_v<rcvref<CoordT>, Coord3D>)
    min.z_ = std::isless(value.z_, min.z_) ? value.z_ : min.z_;
}

template <typename CoordT>
inline void update_max(CoordT &max, const CoordT &value) {
  max.x_ = std::isless(max.x_, value.x_) ? value.x_ : max.x_;
  max.y_ = std::isless(max.y_, value.y_) ? value.y_ : max.y_;

  if constexpr (std::is_same_v<rcvref<CoordT>, Coord3D>)
    max.z_ = std::isless(max.z_, value.z_) ? value.z_ : max.z_;
}

template <typename CoordT> inline std::pair<CoordT, CoordT> minmax_coords() {

  CoordT min, max;
  min.x_ = std::numeric_limits<space_t>::max();
  min.y_ = std::numeric_limits<space_t>::max();
  max.x_ = std::numeric_limits<space_t>::lowest();
  max.y_ = std::numeric_limits<space_t>::lowest();

  if constexpr (std::is_same_v<rcvref<CoordT>, Coord3D>) {
    min.z_ = std::numeric_limits<space_t>::max();
    max.z_ = std::numeric_limits<space_t>::lowest();
  }

  return std::make_pair(min, max);
}

template <typename CoordT>
std::pair<CoordT, CoordT>
minmax_vertices(const std::vector<CoordT> &coord_vec) {
  assert(1 < coord_vec.size());

  auto min_max = minmax_coords<CoordT>();

  for (const auto &coord : coord_vec) {
    update_min(min_max.first, coord);
    update_max(min_max.second, coord);
  }

  return min_max;
}

Coord2D circumscribed_center(const std::vector<Coord2D> &coords);

Coord3D circumscribed_center(const std::vector<Coord3D> &coords);

// Given ellipse with h, k center, a semi major, b semi minor, @ rotation
// If x, y point is in ellipse then
// X^2 / a^2 + Y^2 / b^2 <= 1
// With X = ( x - h ) * cos @ + ( y - k ) * sin @
// and  Y = - ( x - h ) * sin @ + ( y - k ) * cos @
bool displacement_in_ellipse(const Coord2D &displacement,
                             const Coord2D &semi_axes);

bool displacement_in_ellipsoid(const Coord3D &displacement,
                               const Coord3D &semi_axes);
} // namespace sapi

#endif
