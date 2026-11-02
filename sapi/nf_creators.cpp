/*
 *  nf_creators.cpp
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

#include "nf_creators.h"
#include "algebraic_functors.h"
#include "creator_registry.h"
#include "type_erasure_helpers.h"

namespace sapi {
struct IdentityUFCreator final : public StateLessCreator<UnaryFunctor> {
  UnaryFunctor create(const std::vector<space_t> &params) const override {
    if (!params.empty())
      throw std::invalid_argument("Incorrect IdentityUF params");
    return construct_unary_functor(UNARY_FUNCTION::IDENTITY);
  }
};

struct MinUFCreator final : public StateLessCreator<UnaryFunctor> {
  UnaryFunctor create(const std::vector<space_t> &params) const override {
    if (params.size() != 1)
      throw std::invalid_argument("Incorrect MinUF params");
    return construct_unary_functor(UNARY_FUNCTION::MIN, params[0]);
  }
};

struct MaxUFCreator final : public StateLessCreator<UnaryFunctor> {
  UnaryFunctor create(const std::vector<space_t> &params) const override {
    if (params.size() != 1)
      throw std::invalid_argument("Incorrect MaxUF params");
    return construct_unary_functor(UNARY_FUNCTION::MAX, params[0]);
  }
};

struct LowerBoundUFCreator final : public StateLessCreator<UnaryFunctor> {
  UnaryFunctor create(const std::vector<space_t> &params) const override {
    if (params.size() != 2)
      throw std::invalid_argument("Incorrect LowerBoundUF params");
    return construct_unary_functor(UNARY_FUNCTION::LOWER_BOUND, params[0],
                                   params[1]);
  }
};

struct UpperBoundUFCreator final : public StateLessCreator<UnaryFunctor> {
  UnaryFunctor create(const std::vector<space_t> &params) const override {
    if (params.size() != 2)
      throw std::invalid_argument("Incorrect UpperBoundUF params");
    return construct_unary_functor(UNARY_FUNCTION::UPPER_BOUND, params[0],
                                   params[1]);
  }
};

struct InverseUFCreator final : public StateLessCreator<UnaryFunctor> {
  UnaryFunctor create(const std::vector<space_t> &params) const override {
    if (!params.empty())
      throw std::invalid_argument("Incorrect InverseUF params");
    return construct_unary_functor(UNARY_FUNCTION::INVERSE);
  }
};

struct FactorUFCreator final : public StateLessCreator<UnaryFunctor> {
  UnaryFunctor create(const std::vector<space_t> &params) const override {
    if (params.size() != 1)
      throw std::invalid_argument("Incorrect FactorUF params");
    return construct_unary_functor(UNARY_FUNCTION::FACTOR, params[0]);
  }
};

struct OffsetUFCreator final : public StateLessCreator<UnaryFunctor> {
  UnaryFunctor create(const std::vector<space_t> &params) const override {
    if (params.size() != 1)
      throw std::invalid_argument("Incorrect OffsetUF params");
    return construct_unary_functor(UNARY_FUNCTION::OFFSET, params[0]);
  }
};

struct ExponentialUFCreator final : public StateLessCreator<UnaryFunctor> {
  UnaryFunctor create(const std::vector<space_t> &params) const override {
    if (params.size() != 1 || almost_zero(params.at(0)))
      throw std::invalid_argument("Incorrect ExponentialUF params");
    return construct_unary_functor(UNARY_FUNCTION::EXPONENTIAL,
                                   -1. / params[0]);
  }
};

struct GaussianUFCreator final : public StateLessCreator<UnaryFunctor> {
  UnaryFunctor create(const std::vector<space_t> &params) const override {
    if (params.size() != 2 || almost_zero(params.at(1) * params.at(1)))
      throw std::invalid_argument("Incorrect GaussianUF params");
    return construct_unary_functor(UNARY_FUNCTION::GAUSSIAN, -params[0],
                                   -1. / (2. * params[1] * params[1]));
  }
};

struct ConstantDFCreator final : public StateLessCreator<DisplacementFunctor> {
  DisplacementFunctor
  create(const std::vector<space_t> &params) const override {
    if (params.size() != 1)
      throw std::invalid_argument("Incorrect ConstantDF params");
    return construct_displacement_functor(DISPLACEMENT_FUNCTION::CONSTANT,
                                          params[0]);
  }
};

struct DistanceDFCreator final : public StateLessCreator<DisplacementFunctor> {
  DisplacementFunctor
  create(const std::vector<space_t> &params) const override {
    if (!params.empty())
      throw std::invalid_argument("Incorrect DistanceDF params");
    return construct_displacement_functor(DISPLACEMENT_FUNCTION::DISTANCE);
  }
};

struct DisplacementXDFCreator final
    : public StateLessCreator<DisplacementFunctor> {
  DisplacementFunctor
  create(const std::vector<space_t> &params) const override {
    if (!params.empty())
      throw std::invalid_argument("Incorrect DisplacementXDF params");
    return construct_displacement_functor(
        DISPLACEMENT_FUNCTION::DISPLACEMENT_X);
  }
};

struct DisplacementYDFCreator final
    : public StateLessCreator<DisplacementFunctor> {
  DisplacementFunctor
  create(const std::vector<space_t> &params) const override {
    if (!params.empty())
      throw std::invalid_argument("Incorrect DisplacementYDF params");
    return construct_displacement_functor(
        DISPLACEMENT_FUNCTION::DISPLACEMENT_Y);
  }
};

struct DisplacementZDFCreator final
    : public StateLessCreator<DisplacementFunctor> {
  DisplacementFunctor
  create(const std::vector<space_t> &params) const override {
    if (!params.empty())
      throw std::invalid_argument("Incorrect DisplacementZDF params");
    return construct_displacement_functor(
        DISPLACEMENT_FUNCTION::DISPLACEMENT_Z);
  }
};

struct DistanceXDFCreator final : public StateLessCreator<DisplacementFunctor> {
  DisplacementFunctor
  create(const std::vector<space_t> &params) const override {
    if (!params.empty())
      throw std::invalid_argument("Incorrect DistanceXDF params");
    return construct_displacement_functor(DISPLACEMENT_FUNCTION::DISTANCE_X);
  }
};

struct DistanceYDFCreator final : public StateLessCreator<DisplacementFunctor> {
  DisplacementFunctor
  create(const std::vector<space_t> &params) const override {
    if (!params.empty())
      throw std::invalid_argument("Incorrect DistanceYDF params");
    return construct_displacement_functor(DISPLACEMENT_FUNCTION::DISTANCE_Y);
  }
};

struct DistanceZDFCreator final : public StateLessCreator<DisplacementFunctor> {
  DisplacementFunctor
  create(const std::vector<space_t> &params) const override {
    if (!params.empty())
      throw std::invalid_argument("Incorrect DistanceZDF params");
    return construct_displacement_functor(DISPLACEMENT_FUNCTION::DISTANCE_Z);
  }
};

void initialize_uf_registry(CreatorRegistry<UnaryFunctor> &ufr) {
  ufr.register_creator<IdentityUFCreator>(
      UNARY_FUNCTION_NAMES[uint8_t(UNARY_FUNCTION::IDENTITY)]);
  ufr.register_creator<MinUFCreator>(
      UNARY_FUNCTION_NAMES[uint8_t(UNARY_FUNCTION::MIN)]);
  ufr.register_creator<MaxUFCreator>(
      UNARY_FUNCTION_NAMES[uint8_t(UNARY_FUNCTION::MAX)]);
  ufr.register_creator<LowerBoundUFCreator>(
      UNARY_FUNCTION_NAMES[uint8_t(UNARY_FUNCTION::LOWER_BOUND)]);
  ufr.register_creator<UpperBoundUFCreator>(
      UNARY_FUNCTION_NAMES[uint8_t(UNARY_FUNCTION::UPPER_BOUND)]);
  ufr.register_creator<InverseUFCreator>(
      UNARY_FUNCTION_NAMES[uint8_t(UNARY_FUNCTION::INVERSE)]);
  ufr.register_creator<FactorUFCreator>(
      UNARY_FUNCTION_NAMES[uint8_t(UNARY_FUNCTION::FACTOR)]);
  ufr.register_creator<OffsetUFCreator>(
      UNARY_FUNCTION_NAMES[uint8_t(UNARY_FUNCTION::OFFSET)]);
  ufr.register_creator<ExponentialUFCreator>(
      UNARY_FUNCTION_NAMES[uint8_t(UNARY_FUNCTION::EXPONENTIAL)]);
  ufr.register_creator<GaussianUFCreator>(
      UNARY_FUNCTION_NAMES[uint8_t(UNARY_FUNCTION::GAUSSIAN)]);
}

void initialize_df_registry(CreatorRegistry<DisplacementFunctor> &dfr) {
  dfr.register_creator<ConstantDFCreator>(
      DISPLACEMENT_FUNCTION_NAMES[uint8_t(DISPLACEMENT_FUNCTION::CONSTANT)]);
  dfr.register_creator<DistanceDFCreator>(
      DISPLACEMENT_FUNCTION_NAMES[uint8_t(DISPLACEMENT_FUNCTION::DISTANCE)]);
  dfr.register_creator<DisplacementXDFCreator>(
      DISPLACEMENT_FUNCTION_NAMES[uint8_t(
          DISPLACEMENT_FUNCTION::DISPLACEMENT_X)]);
  dfr.register_creator<DisplacementYDFCreator>(
      DISPLACEMENT_FUNCTION_NAMES[uint8_t(
          DISPLACEMENT_FUNCTION::DISPLACEMENT_Y)]);
  dfr.register_creator<DisplacementZDFCreator>(
      DISPLACEMENT_FUNCTION_NAMES[uint8_t(
          DISPLACEMENT_FUNCTION::DISPLACEMENT_Z)]);
  dfr.register_creator<DistanceXDFCreator>(
      DISPLACEMENT_FUNCTION_NAMES[uint8_t(DISPLACEMENT_FUNCTION::DISTANCE_X)]);
  dfr.register_creator<DistanceYDFCreator>(
      DISPLACEMENT_FUNCTION_NAMES[uint8_t(DISPLACEMENT_FUNCTION::DISTANCE_Y)]);
  dfr.register_creator<DistanceZDFCreator>(
      DISPLACEMENT_FUNCTION_NAMES[uint8_t(DISPLACEMENT_FUNCTION::DISTANCE_Z)]);
}
} // namespace sapi
