/*
 *  mask2d_geometry.cpp
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

#include "mask2d_geometry.h"
#include "coordinate_geometry.h"

namespace sapi {
void initialize_circular_mask_helpers(Mask<Coord2D> &mask,
                                      const std::vector<space_t> &mask_params) {
  mask.c_radius_.radius2_ = mask_params[0] * mask_params[0];
  if (std::signbit(mask_params[0]) || almost_zero(mask.c_radius_.radius2_))
    throw std::invalid_argument("Invalid circular mask param");
}

void initialize_elliptical_mask_helpers(
    Mask<Coord2D> &mask, const std::vector<space_t> &mask_params) {
  if (std::signbit(mask_params[0]) || std::signbit(mask_params[1]))
    throw std::invalid_argument("Invalid elliptical mask axes");

  Coord2D axes = construct_coord_2D(mask_params[0] * mask_params[0],
                                    mask_params[1] * mask_params[1]);

  if (almost_zero(axes.x_) || almost_zero(axes.y_))
    throw std::invalid_argument("Invalid elliptical mask axes");

  mask.c_radius_.radius2_ = std::fmax(axes.x_, axes.y_);

  const space_t rotation = 2 < mask_params.size() ? mask_params[2] : 0;
  if (almost_zero(rotation)) {
    mask.helper_vectors_.resize(1, axes);
  } else {
    mask.helper_vectors_.resize(2);
    mask.helper_vectors_[0] = axes;
    mask.helper_vectors_[1] = create_angular_offset(rotation);
  }
}

void initialize_parallelogram_mask_helpers(
    Mask<Coord2D> &mask, const std::vector<space_t> &mask_params) {
  auto it = mask_params.begin();
  auto axial_vector0 = Coord2D::copy_from_vec(it);
  auto axial_vector1 = Coord2D::copy_from_vec(it);

  mask.c_radius_.radius2_ =
      std::fmax(vector_norm2(axial_vector0), vector_norm2(axial_vector1));

  if (almost_zero(mask.c_radius_.radius2_))
    throw std::invalid_argument("Invalid parallelogram mask vectors");

  mask.helper_vectors_.resize(3);
  mask.helper_vectors_[0] = axial_vector0;
  mask.helper_vectors_[1] = axial_vector0 - axial_vector1;
  mask.helper_vectors_[2] = axial_vector0 + axial_vector1;

  mask.helper_scalars_.resize(
      1, vector_cross(mask.helper_vectors_[1], mask.helper_vectors_[2]).sum());

  if (almost_zero(mask.helper_scalars_[0]))
    throw std::invalid_argument("Invalid parallelogram mask vectors");
}

void initialize_triangular_mask_helpers(
    Mask<Coord2D> &mask, const std::vector<space_t> &mask_params) {
  auto it = mask_params.begin();
  auto axial_vector0 = Coord2D::copy_from_vec(it);
  auto axial_vector1 = Coord2D::copy_from_vec(it);

  mask.c_radius_.radius2_ =
      std::fmax(vector_norm2(axial_vector0), vector_norm2(axial_vector1));

  if (almost_zero(mask.c_radius_.radius2_))
    throw std::invalid_argument("Invalid triangular mask vectors");

  mask.helper_scalars_.resize(1,
                              vector_cross(axial_vector0, axial_vector1).sum());

  if (almost_zero(mask.helper_scalars_[0]))
    throw std::invalid_argument("Invalid triangular mask vectors");

  mask.helper_vectors_.resize(2);
  mask.helper_vectors_[0] = axial_vector0;
  mask.helper_vectors_[1] = axial_vector1;
}
} // namespace sapi
