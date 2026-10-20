/*
 *  rng_creators.h
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

#ifndef RNG_CREATORS_H
#define RNG_CREATORS_H

#include <type_traits>

#include "sapi_config.h"

namespace sapi {
// Forward definition to creator_registry.h
template <typename RT> class CreatorRegistry;

// Forward definition to link with type_erasure_helpers.h
template <typename RT, typename std::enable_if_t<
                           std::disjunction_v<std::is_same<RT, uint32_t>,
                                              std::is_same<RT, uint64_t>>,
                           bool>
                           b>
class AnyRNG_T;

void initialize_rng_registry(CreatorRegistry<AnyRNG_T<uint32_t, true>> &anr);
void initialize_rng_registry(CreatorRegistry<AnyRNG_T<uint64_t, true>> &anr);
} // namespace sapi

#endif
