/*
 *  mask2d_geometry.h
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

#ifndef MASK2D_GEOMETRY_H
#define MASK2D_GEOMETRY_H

#include "mask.h"

namespace sapi {
// Forward definition to link with coordinate_geometry.h
Coord2D rotate_displacement(const Coord2D &displacement,
                            const Coord2D &angular_coord);
bool displacement_in_ellipse(const Coord2D &displacement,
                             const Coord2D &semi_axes);
template <typename CoordT, bool triangular_comparison>
bool algebraic_projection_comparison(const CoordT &vectorBA,
                                     const CoordT &vectorBC,
                                     const CoordT &vectorBD,
                                     const space_t det_BCD);

void initialize_circular_mask_helpers(Mask<Coord2D> &mask,
                                      const std::vector<space_t> &mask_params);

void initialize_elliptical_mask_helpers(
    Mask<Coord2D> &mask, const std::vector<space_t> &mask_params);

void initialize_parallelogram_mask_helpers(
    Mask<Coord2D> &mask, const std::vector<space_t> &mask_params);

void initialize_triangular_mask_helpers(
    Mask<Coord2D> &mask, const std::vector<space_t> &mask_params);

inline OptDisp<Coord2D> coord_in_circular_mask(const Coord2D &coord,
                                               const Mask<Coord2D> &mask) {
  return mask.c_radius_.coord_in_radius(coord);
}

inline OptDisp<Coord2D> coord_in_circular_mask(const Coord2D &a,
                                               const Coord2D &b,
                                               const Mask<Coord2D> &mask) {
  OptDisp<Coord2D> od(false, construct_displacement(b - (a + mask.offset_)));
  od.first = leq_test(od.second.distance2_, mask.c_radius_.radius2_);
  return od;
}

inline OptDisp<Coord2D> coord_in_elliptical_mask(const Coord2D &coord,
                                                 const Mask<Coord2D> &mask) {
  auto od = mask.c_radius_.coord_in_radius(coord);
  if (od.first)
    od.first = displacement_in_ellipse(
        1 < mask.helper_vectors_.size()
            ? rotate_displacement(od.second.displacement_,
                                  mask.helper_vectors_[1])
            : od.second.displacement_,
        mask.helper_vectors_[0]);
  return od;
}

inline OptDisp<Coord2D> coord_in_elliptical_mask(const Coord2D &a,
                                                 const Coord2D &b,
                                                 const Mask<Coord2D> &mask) {
  OptDisp<Coord2D> od(false, construct_displacement(b - (a + mask.offset_)));
  if (leq_test(od.second.distance2_, mask.c_radius_.radius2_))
    od.first = displacement_in_ellipse(
        1 < mask.helper_vectors_.size()
            ? rotate_displacement(od.second.displacement_,
                                  mask.helper_vectors_[1])
            : od.second.displacement_,
        mask.helper_vectors_[0]);
  return od;
}

inline OptDisp<Coord2D> coord_in_parallelogram_mask(const Coord2D &coord,
                                                    const Mask<Coord2D> &mask) {
  auto od = mask.c_radius_.coord_in_radius(coord);
  if (od.first)
    od.first = algebraic_projection_comparison<Coord2D, false>(
        od.second.displacement_ + mask.helper_vectors_[0],
        mask.helper_vectors_[1], mask.helper_vectors_[2],
        mask.helper_scalars_[0]);

  return od;
}

inline OptDisp<Coord2D> coord_in_parallelogram_mask(const Coord2D &a,
                                                    const Coord2D &b,
                                                    const Mask<Coord2D> &mask) {
  OptDisp<Coord2D> od(false, construct_displacement(b - (a + mask.offset_)));
  if (leq_test(od.second.distance2_, mask.c_radius_.radius2_))
    od.first = algebraic_projection_comparison<Coord2D, false>(
        od.second.displacement_ + mask.helper_vectors_[0],
        mask.helper_vectors_[1], mask.helper_vectors_[2],
        mask.helper_scalars_[0]);

  return od;
}

inline OptDisp<Coord2D> coord_in_triangular_mask(const Coord2D &coord,
                                                 const Mask<Coord2D> &mask) {
  auto od = mask.c_radius_.coord_in_radius(coord);
  if (od.first)
    od.first = algebraic_projection_comparison<Coord2D, true>(
        od.second.displacement_, mask.helper_vectors_[0],
        mask.helper_vectors_[1], mask.helper_scalars_[0]);

  return od;
}

inline OptDisp<Coord2D> coord_in_triangular_mask(const Coord2D &a,
                                                 const Coord2D &b,
                                                 const Mask<Coord2D> &mask) {
  OptDisp<Coord2D> od(false, construct_displacement(b - (a + mask.offset_)));
  if (leq_test(od.second.distance2_, mask.c_radius_.radius2_))
    od.first = algebraic_projection_comparison<Coord2D, true>(
        od.second.displacement_, mask.helper_vectors_[0],
        mask.helper_vectors_[1], mask.helper_scalars_[0]);

  return od;
}
} // namespace sapi

#endif
