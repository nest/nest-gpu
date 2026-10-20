/*
 *  split_tree.h
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

#ifndef SPLIT_TREE_H
#define SPLIT_TREE_H

#include <vector>

#include "sapi_config.h"

namespace sapi {
struct SplitBranch {
  const split_t index_ = 0;
  const SplitBranch *const parent_ = nullptr;
  std::vector<SplitBranch> children_;

  SplitBranch() noexcept = default;
  SplitBranch(const SplitBranch &) = delete;
  SplitBranch(SplitBranch &&) noexcept = default;
  ~SplitBranch() noexcept = default;

  SplitBranch(const split_t index, const SplitBranch *const parent) noexcept
      : index_(index), parent_(parent) {}

  SplitBranch &operator=(const SplitBranch &) = delete;
  SplitBranch &operator=(SplitBranch &&) = delete;
};

tileidx_t collapse_dimensions(const std::vector<split_t> &branch_sequence,
                              const std::vector<split_t> &split_sequence);

tileidx_t
get_index_from_branches(const SplitBranch *from_node,
                        const std::vector<split_t> &possible_branches);
} // namespace sapi

#endif
