/*
 *  rng_creators.cpp
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

#include "rng_creators.h"
#include "creator_registry.h"
#include "enum_store.h"
#include "type_erasure_helpers.h"

// Third party
#include "pcg_random.hpp"
#include "randutils.hpp"

namespace sapi {
template <typename RNG, typename std::enable_if_t<
                            std::is_nothrow_copy_constructible_v<RNG>, bool>
                            b>
void RNGModel<RNG, b>::seed(const std::initializer_list<rng_seed_t> &l) {
  randutils::auto_seed_256 seq(l);
  engine_.seed(seq);
}

template <typename RT> struct FakeRNG {
  using result_type = RT;

  RT state_ = 0;

  template <typename seed_seq> void seed(const seed_seq &) { state_ = 0; }

  RT operator()() { return state_++; }
};

struct PCGGenerator32 final : public StateLessCreator<AnyRNG_T<uint32_t>> {
  AnyRNG_T<uint32_t> create() const override {
    return AnyRNG_T<uint32_t>(pcg32_k1024_fast{});
  }
};

struct PCGGenerator64 final : public StateLessCreator<AnyRNG_T<uint64_t>> {
  AnyRNG_T<uint64_t> create() const override {
    return AnyRNG_T<uint64_t>(pcg64_k1024_fast{});
  }
};

struct MersenneTwister64 final : public StateLessCreator<AnyRNG_T<uint64_t>> {
  AnyRNG_T<uint64_t> create() const override {
    return AnyRNG_T<uint64_t>(std::mt19937_64{});
  }
};

template <typename RT>
struct FakeGenerator final : public StateLessCreator<AnyRNG_T<RT>> {
  AnyRNG_T<RT> create() const override { return AnyRNG_T<RT>(FakeRNG<RT>{}); }
};

void initialize_rng_registry(CreatorRegistry<AnyRNG_T<uint32_t>> &anr) {
  anr.register_creator<PCGGenerator32>(RNG_NAMES_32BIT[uint8_t(RNG32::PCG)]);
  anr.register_creator<FakeGenerator<uint32_t>>(
      RNG_NAMES_32BIT[uint8_t(RNG32::FAKE)]);
}

void initialize_rng_registry(CreatorRegistry<AnyRNG_T<uint64_t>> &anr) {
  anr.register_creator<PCGGenerator64>(RNG_NAMES_64BIT[uint8_t(RNG64::PCG)]);
  anr.register_creator<MersenneTwister64>(RNG_NAMES_64BIT[uint8_t(RNG64::MT)]);
  anr.register_creator<FakeGenerator<uint64_t>>(
      RNG_NAMES_64BIT[uint8_t(RNG64::FAKE)]);
}
} // namespace sapi
