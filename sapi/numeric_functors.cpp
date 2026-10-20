/*
 *  numeric_functors.cpp
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

#include "numeric_functors.h"
#include "creator_registry.h"
#include "type_erasure_helpers.h"

namespace sapi {
NumericFunctor
construct_numeric_functor(const std::string &df_name,
                          const std::vector<space_t> &df_params,
                          const CreatorRegistry<DisplacementFunctor> &df_reg,
                          const std::vector<std::string> &ufs_names,
                          const std::vector<std::vector<space_t>> &ufs_params,
                          const CreatorRegistry<UnaryFunctor> &uf_reg) {
  NumericFunctor nf;

  if (df_name.empty()) {
    if (!(df_params.empty() && ufs_names.empty() && ufs_params.empty()))
      throw std::invalid_argument(
          "Mismatched input vectors for UnaryFunctors given to NumericFunctor");
    return nf;
  }

  if (ufs_names.size() != ufs_params.size())
    throw std::invalid_argument(
        "Mismatched input vectors for UnaryFunctors given to NumericFunctor");

  nf.df_ = df_reg.get_creator(df_name)->create(df_params);

  if (!ufs_names.empty()) {
    nf.ufs_.reserve(ufs_names.size());

    auto params_it = ufs_params.cbegin();
    for (const auto &uf_name : ufs_names)
      nf.ufs_.emplace_back(uf_reg.get_creator(uf_name)->create(*params_it++));
  }

  return nf;
}
} // namespace sapi
