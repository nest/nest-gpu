/*
 *  numeric_functors.h
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

#ifndef NUMERIC_FUNCTORS_H
#define NUMERIC_FUNCTORS_H

#include <cassert>
#include <vector>

#include "algebraic_functors.h"

namespace sapi {
// Forward definition to link with creator_registry.h
template <typename T> class CreatorRegistry;

struct NumericFunctor {
  DisplacementFunctor df_;
  std::vector<UnaryFunctor> ufs_;

  bool is_initialized() const;

  template <typename CoordT>
  conn_param_t operator()(const Displacement<CoordT> &d) const;
};

NumericFunctor
construct_numeric_functor(const std::string &df_name,
                          const std::vector<space_t> &df_params,
                          const CreatorRegistry<DisplacementFunctor> &df_reg,
                          const std::vector<std::string> &ufs_names,
                          const std::vector<std::vector<space_t>> &ufs_params,
                          const CreatorRegistry<UnaryFunctor> &uf_reg);

inline bool NumericFunctor::is_initialized() const {
  return df_.func_ != DISPLACEMENT_FUNCTION::NULL_DF;
}

template <typename CoordT>
conn_param_t NumericFunctor::operator()(const Displacement<CoordT> &dc) const {
  auto val = df_(dc);
  for (const auto &uf : ufs_)
    val = uf(val);

  return std::fmin(std::fmax(val, std::numeric_limits<conn_param_t>::lowest()),
                   std::numeric_limits<conn_param_t>::max());
}

struct NFCollection {
  NumericFunctor weight_functor_;
  NumericFunctor delay_functor_;
  NumericFunctor probability_functor_;
};
} // namespace sapi

#endif
