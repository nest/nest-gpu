/*
 *  split_tree.cpp
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

#include <cassert>

#include "split_tree.h"

namespace sapi {
tileidx_t collapse_dimensions(const std::vector<split_t> &branch_sequence,
                              const std::vector<split_t> &split_sequence) {
  assert(branch_sequence.size() == split_sequence.size());
  auto bs_it = branch_sequence.begin();
  tileidx_t index = *bs_it++;
  for (auto ss_it = split_sequence.begin() + 1; ss_it != split_sequence.end();
       ++ss_it)
    index = *bs_it++ + *ss_it * index;
  return index;
}

tileidx_t
get_index_from_branches(const SplitBranch *from_node,
                        const std::vector<split_t> &possible_branches) {
  std::vector<split_t> branches(possible_branches.size());
  for (auto rev_it = branches.rbegin(); rev_it != branches.rend(); ++rev_it) {
    assert(from_node != nullptr);
    *rev_it = from_node->index_;
    from_node = from_node->parent_;
  }
  return collapse_dimensions(branches, possible_branches);
}
} // namespace sapi
