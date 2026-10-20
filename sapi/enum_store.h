/*
 *  enum_store.h
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

#ifndef ENUM_STORE_H
#define ENUM_STORE_H

#include <cstdint>

namespace sapi {
enum class TILE_SHAPE : uint8_t { RECTANGLE, TRIANGLE, HEXAGON, NULL_TS };

constexpr static const char *const TILE_SHAPE_NAMES[uint8_t(
    TILE_SHAPE::NULL_TS)] = {"rectangle", "triangle", "hexagon"};

enum class MASK_SHAPE : uint8_t {
  CIRCULAR,
  ELLIPTICAL,
  PARALLELOGRAM,
  TRIANGULAR,
  NULL_MS
};

constexpr static const char *const
    MASK_SHAPE_NAMES[uint8_t(MASK_SHAPE::NULL_MS)] = {
        "circular", "elliptical", "parallelogram", "triangular"};

enum class OVERLAP_LEVEL : uint8_t { NONE, PARTIAL, FULL };

enum class DISTRIBUTION_MODE : uint8_t { FREE, SQUEEZED, BALANCED, NULL_DM };

constexpr static const char *const DISTRIBUTION_MODE_NAMES[uint8_t(
    DISTRIBUTION_MODE::NULL_DM)] = {"free", "squeezed", "balanced"};

enum class CONNECTION_RULE : uint8_t {
  PAIRWISE_BERNOULLI,
  PAIRWISE_POISSON,
  FIXED_IN_DEGREE,
  FIXED_OUT_DEGREE,
  NULL_CM
};

constexpr static const char *const CONNECTION_RULE_NAMES[uint8_t(
    CONNECTION_RULE::NULL_CM)] = {"pairwise_bernoulli", "pairwise_poisson",
                                  "fixed_indegree", "fixed_outdegree"};

enum class UNARY_FUNCTION : uint8_t {
  IDENTITY,
  MIN,
  MAX,
  LOWER_BOUND,
  UPPER_BOUND,
  INVERSE,
  FACTOR,
  OFFSET,
  EXPONENTIAL,
  GAUSSIAN,
  NULL_UF
};

constexpr static const char *const
    UNARY_FUNCTION_NAMES[uint8_t(UNARY_FUNCTION::NULL_UF)] = {
        "identity", "min",    "max",    "lower_bound", "upper_bound",
        "inverse",  "factor", "offset", "exponential", "gaussian",
};

enum class DISPLACEMENT_FUNCTION : uint8_t {
  CONSTANT,
  DISTANCE,
  DISPLACEMENT_X,
  DISPLACEMENT_Y,
  DISPLACEMENT_Z,
  DISTANCE_X,
  DISTANCE_Y,
  DISTANCE_Z,
  NULL_DF
};

constexpr static const char *const
    DISPLACEMENT_FUNCTION_NAMES[uint8_t(DISPLACEMENT_FUNCTION::NULL_DF)] = {
        "constant",       "distance",   "displacement_x", "displacement_y",
        "displacement_z", "distance_x", "distance_y",     "distance_z",
};

enum class RNG32 : uint8_t { PCG, FAKE, NULL_RNG };

constexpr static const char *const RNG_NAMES_32BIT[uint8_t(RNG32::NULL_RNG)] = {
    "pcg", "fake"};

enum class RNG64 : uint8_t { PCG, MT, FAKE, NULL_RNG };

constexpr static const char *const RNG_NAMES_64BIT[uint8_t(RNG64::NULL_RNG)] = {
    "pcg", "mersenne_twister"
           "fake"};
} // namespace sapi

#endif
