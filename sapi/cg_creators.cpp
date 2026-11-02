/*
 *  cg_creators.cpp
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

#include "cg_creators.h"
#include "connection_generator.h"
#include "creator_registry.h"
#include "type_erasure_helpers.h"

namespace sapi {
template <CONNECTION_RULE method>
struct BaseGCCreator final : public StateLessCreator<ConnectionGenerator> {
  ConnectionGenerator create() const override {
    ConnectionGenerator cg;
    cg.rule_ = method;
    return cg;
  }
};

void initialize_cg_registry(CreatorRegistry<ConnectionGenerator> &cgr) {
  cgr.register_creator<BaseGCCreator<CONNECTION_RULE::PAIRWISE_BERNOULLI>>(
      CONNECTION_RULE_NAMES[uint8_t(CONNECTION_RULE::PAIRWISE_BERNOULLI)]);
  cgr.register_creator<BaseGCCreator<CONNECTION_RULE::PAIRWISE_POISSON>>(
      CONNECTION_RULE_NAMES[uint8_t(CONNECTION_RULE::PAIRWISE_POISSON)]);
  cgr.register_creator<BaseGCCreator<CONNECTION_RULE::FIXED_IN_DEGREE>>(
      CONNECTION_RULE_NAMES[uint8_t(CONNECTION_RULE::FIXED_IN_DEGREE)]);
  cgr.register_creator<BaseGCCreator<CONNECTION_RULE::FIXED_OUT_DEGREE>>(
      CONNECTION_RULE_NAMES[uint8_t(CONNECTION_RULE::FIXED_OUT_DEGREE)]);
}
} // namespace sapi
