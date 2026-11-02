/*
 *  numerics.h
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

#ifndef NUMERICS_H
#define NUMERICS_H

#include <cmath>
#include <limits>

#include "sapi_config.h"

namespace sapi {
constexpr static const space_t SP_PI_180 = space_t(M_PI) / space_t(180);
constexpr static const space_t SP_EPSILON =
    std::numeric_limits<space_t>::epsilon() * TOLERANCE;
constexpr static const space_t SP_RELATIVE_EPSILON =
    space_t(1) / (SP_EPSILON * (1UL << RELATIVE_TOLERANCE));

constexpr inline bool positiveTix(const tileidx_t v) { return 0 < v; }

constexpr inline bool negativeTix(const tileidx_t v) { return v < 0; }

constexpr inline bool evenTix(const tileidx_t v) { return !(v % 2); }

constexpr inline tileidx_t minusTix(const tileidx_t x, const tileidx_t y) {
  return x - y;
}

constexpr inline tileidx_t minusOneTix(const tileidx_t x) { return x - 1; }

constexpr inline space_t radians(const space_t degrees) {
  return SP_PI_180 * degrees;
}

constexpr inline bool abs_less(const space_t smaller, const space_t bigger) {
  return std::isless(std::fabs(smaller), std::fabs(bigger));
}

constexpr inline bool almost_zero(const space_t x) {
  return (x == 0) || std::islessequal(std::fabs(x), SP_EPSILON);
}

constexpr inline bool almost_equal(const space_t a, const space_t b) {
  const space_t diff = std::fabs(a - b);
  switch (
      // Trivial case
      ((a == b) << 0)
      // Check if the numbers are really close -- needed
      // when comparing numbers near zero.
      + (std::islessequal(diff, SP_EPSILON) << 1)
      // Sign comparison
      + ((std::signbit(a) != std::signbit(b)) << 2)) {
  case 1:
    return true;

  case 3:
    return true;

  case 5:
    return true;

  case 7:
    return true;

  case 2:
    return true;

  case 6:
    return true;

  case 4:
    return false;

  default:
    // Relative epsilon difference comparison
    return std::islessequal(diff * SP_RELATIVE_EPSILON,
                            std::fabs(a) + std::fabs(b));
  }
}

constexpr inline bool interval_test(const space_t lower_bound,
                                    const space_t value,
                                    const space_t upper_bound) {
  return (std::isless(lower_bound, value) && std::isless(value, upper_bound)) ||
         almost_equal(lower_bound, value) || almost_equal(value, upper_bound);
}

constexpr inline bool leq_test(const space_t smaller, const space_t bigger) {
  return std::isless(smaller, bigger) || almost_equal(smaller, bigger);
}

constexpr inline space_t determinant3x3(const space_t a, const space_t b,
                                        const space_t c, const space_t d,
                                        const space_t e, const space_t f,
                                        const space_t g, const space_t h,
                                        const space_t i) {
  // Using Laplace expansion on first row
  return (a * e * i) - (a * f * h) - (b * d * i) + (b * f * g) + (c * d * h) -
         (c * e * g);
}
} // namespace sapi

#endif
