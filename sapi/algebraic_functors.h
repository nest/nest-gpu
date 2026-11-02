/*
 *  algebraic_functors.h
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

#ifndef ALGEBRAIC_FUNCTORS_H
#define ALGEBRAIC_FUNCTORS_H

#include <stdexcept>

#include "enum_store.h"
#include "numerics.h"

namespace sapi {
// Forward definition to link with coordinates.h
struct Coord2D;
struct Coord3D;
template <typename CoordT> struct Displacement;

struct UnaryFunctor {
  UNARY_FUNCTION func_ = UNARY_FUNCTION::NULL_UF;
  space_t var_[2] = {0.};

  inline space_t operator()(const space_t x) const;
};

inline UnaryFunctor construct_unary_functor(const UNARY_FUNCTION function,
                                            const space_t var0 = 0.,
                                            const space_t var1 = 0.) {
  UnaryFunctor uf;
  uf.func_ = function;
  uf.var_[0] = var0;
  uf.var_[1] = var1;
  return uf;
}

inline space_t UnaryFunctor::operator()(const space_t x) const {
  switch (func_) {
  case UNARY_FUNCTION::IDENTITY:
    return x;

  case UNARY_FUNCTION::MIN:
    return std::fmin(x, var_[0]);

  case UNARY_FUNCTION::MAX:
    return std::fmax(x, var_[0]);

  case UNARY_FUNCTION::LOWER_BOUND:
    return leq_test(x, var_[0]) ? var_[1] : x;

  case UNARY_FUNCTION::UPPER_BOUND:
    return leq_test(var_[0], x) ? var_[1] : x;

  case UNARY_FUNCTION::INVERSE:
    return almost_zero(x) ? std::signbit(x)
                                ? -std::numeric_limits<space_t>::infinity()
                                : std::numeric_limits<space_t>::infinity()
                          : 1. / x;

  case UNARY_FUNCTION::FACTOR:
    return x * var_[0];

  case UNARY_FUNCTION::OFFSET:
    return x + var_[0];

  case UNARY_FUNCTION::EXPONENTIAL:
    return std::exp(x * var_[0]);

  case UNARY_FUNCTION::GAUSSIAN: {
    const auto u = x + var_[0];
    return std::exp(u * u * var_[1]);
  }

  default:
    throw std::invalid_argument("Invalid unary function name");
  }
}

struct DisplacementFunctor {
  DISPLACEMENT_FUNCTION func_ = DISPLACEMENT_FUNCTION::NULL_DF;
  space_t var_ = 0;

  template <typename CoordT>
  space_t operator()(const Displacement<CoordT> &d) const;
};

inline DisplacementFunctor
construct_displacement_functor(const DISPLACEMENT_FUNCTION function,
                               const space_t var = 0.) {
  DisplacementFunctor df;
  df.func_ = function;
  df.var_ = var;
  return df;
}

template <typename CoordT>
inline space_t
DisplacementFunctor::operator()(const Displacement<CoordT> &d) const {
  switch (func_) {
  case DISPLACEMENT_FUNCTION::CONSTANT:
    return var_;

  case DISPLACEMENT_FUNCTION::DISTANCE:
    return d.get_distance();

  case DISPLACEMENT_FUNCTION::DISPLACEMENT_X:
    return d.displacement_.x_;

  case DISPLACEMENT_FUNCTION::DISPLACEMENT_Y:
    return d.displacement_.y_;

  case DISPLACEMENT_FUNCTION::DISPLACEMENT_Z: {
    if constexpr (std::is_same_v<CoordT, Coord3D>)
      return d.displacement_.z_;
    else
      throw std::invalid_argument("Invalid displacement function name");
  }

  case DISPLACEMENT_FUNCTION::DISTANCE_X:
    return std::fabs(d.displacement_.x_);

  case DISPLACEMENT_FUNCTION::DISTANCE_Y:
    return std::fabs(d.displacement_.y_);

  case DISPLACEMENT_FUNCTION::DISTANCE_Z: {
    if constexpr (std::is_same_v<CoordT, Coord3D>)
      return std::fabs(d.displacement_.z_);
    else
      throw std::invalid_argument("Invalid displacement function name");
  }

  default:
    throw std::invalid_argument("Invalid displacement function name");
  }
}
} // namespace sapi

#endif
