/*
 *  mask_geometry.h
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

#ifndef MASK_GEOMETRY_H
#define MASK_GEOMETRY_H

#include <stdexcept>
#include <type_traits>
#include <vector>

#include "coordinates.h"
#include "enum_store.h"

namespace sapi {
// Forward definition to mask.h
template <typename CoordT> struct Mask;

void initialize_circular_mask_helpers(Mask<Coord2D> &mask,
                                      const std::vector<space_t> &mask_params);

void initialize_elliptical_mask_helpers(
    Mask<Coord2D> &mask, const std::vector<space_t> &mask_params);

void initialize_parallelogram_mask_helpers(
    Mask<Coord2D> &mask, const std::vector<space_t> &mask_params);

void initialize_triangular_mask_helpers(
    Mask<Coord2D> &mask, const std::vector<space_t> &mask_params);

template <typename CoordT>
void initialize_mask_helpers(Mask<CoordT> &mask,
                             const std::vector<space_t> &mask_params) {
  const auto params_length = mask_params.size();

  if constexpr (std::is_same_v<CoordT, Coord2D>) {
    switch (mask.shape_) {
    case MASK_SHAPE::CIRCULAR: {
      if (params_length != 1)
        throw std::invalid_argument("Invalid circular mask params vector");

      initialize_circular_mask_helpers(mask, mask_params);

      break;
    }

    case MASK_SHAPE::ELLIPTICAL: {
      if (params_length < 2 || 3 < params_length)
        throw std::invalid_argument("Invalid elliptical mask params vector");

      initialize_elliptical_mask_helpers(mask, mask_params);

      break;
    }

    case MASK_SHAPE::PARALLELOGRAM: {
      if (params_length != 4)
        throw std::invalid_argument("Invalid parallelogram mask params vector");

      initialize_parallelogram_mask_helpers(mask, mask_params);

      break;
    }

    case MASK_SHAPE::TRIANGULAR: {
      if (params_length != 4)
        throw std::invalid_argument("Invalid triangular mask params vector");

      initialize_triangular_mask_helpers(mask, mask_params);

      break;
    }

    default:
      throw std::invalid_argument("Invalid mask shape");
    }
  } else {
    throw std::runtime_error("3D implementation not available yet");
  }
}

OptDisp<Coord2D> coord_in_circular_mask(const Coord2D &coord,
                                        const Mask<Coord2D> &mask);

OptDisp<Coord2D> coord_in_elliptical_mask(const Coord2D &coord,
                                          const Mask<Coord2D> &mask);

OptDisp<Coord2D> coord_in_parallelogram_mask(const Coord2D &coord,
                                             const Mask<Coord2D> &mask);

OptDisp<Coord2D> coord_in_triangular_mask(const Coord2D &coord,
                                          const Mask<Coord2D> &mask);

template <typename CoordT>
OptDisp<CoordT> coord_in_mask(const CoordT &coord, const Mask<CoordT> &mask) {
  if constexpr (std::is_same_v<CoordT, Coord2D>) {
    switch (mask.shape_) {
    case MASK_SHAPE::CIRCULAR:
      return coord_in_circular_mask(coord, mask);

    case MASK_SHAPE::ELLIPTICAL:
      return coord_in_elliptical_mask(coord, mask);

    case MASK_SHAPE::PARALLELOGRAM:
      return coord_in_parallelogram_mask(coord, mask);

    case MASK_SHAPE::TRIANGULAR:
      return coord_in_triangular_mask(coord, mask);

    default:
      throw std::invalid_argument("Invalid mask shape");
    }
  } else {
    throw std::runtime_error("3D implementation not available yet");
  }
}

OptDisp<Coord2D> coord_in_circular_mask(const Coord2D &a, const Coord2D &b,
                                        const Mask<Coord2D> &mask);

OptDisp<Coord2D> coord_in_elliptical_mask(const Coord2D &a, const Coord2D &b,
                                          const Mask<Coord2D> &mask);

OptDisp<Coord2D> coord_in_parallelogram_mask(const Coord2D &a, const Coord2D &b,
                                             const Mask<Coord2D> &mask);

OptDisp<Coord2D> coord_in_triangular_mask(const Coord2D &a, const Coord2D &b,
                                          const Mask<Coord2D> &mask);

template <typename CoordT>
OptDisp<CoordT> coord_in_mask(const CoordT &a, const CoordT &b,
                              const Mask<CoordT> &mask) {
  if constexpr (std::is_same_v<CoordT, Coord2D>) {
    switch (mask.shape_) {
    case MASK_SHAPE::CIRCULAR:
      return coord_in_circular_mask(a, b, mask);

    case MASK_SHAPE::ELLIPTICAL:
      return coord_in_elliptical_mask(a, b, mask);

    case MASK_SHAPE::PARALLELOGRAM:
      return coord_in_parallelogram_mask(a, b, mask);

    case MASK_SHAPE::TRIANGULAR:
      return coord_in_triangular_mask(a, b, mask);

    default:
      throw std::invalid_argument("Invalid mask shape");
    }
  } else {
    throw std::runtime_error("3D implementation not available yet");
  }
}
} // namespace sapi

#endif
