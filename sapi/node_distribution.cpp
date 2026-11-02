/*
 *  node_distribution.cpp
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

#include "node_distribution.h"
#include "grid_neighborhood.h"

namespace sapi {
DistributedTiledNodeSequenceMap consolidate_node_sequences_per_tile_per_rank(
    const RankNodeSequenceMap &node_sequences_per_rank,
    const RankTileIdxNodeCountPairs &node_counts_per_tile_per_rank) {
  nodeidx_t first_node_idx;
  DistributedTiledNodeSequenceMap dist_tns;
  for (const auto &[rank, node_sequence] : node_sequences_per_rank) {
    if (node_sequence.first < 0 || node_sequence.second < 1)
      throw std::invalid_argument("Invalid node sequence");

    first_node_idx = node_sequence.first;
    TileIdxNodeSequenceMap tile_idx_node_sequence_map;
    for (const auto &[tile_index, node_count] :
         node_counts_per_tile_per_rank.at(rank)) {
      if (node_count < 1)
        continue;

      const bool emplace_res =
          tile_idx_node_sequence_map
              .emplace(tile_index, NodeSequence(first_node_idx, node_count))
              .second;
      assert(emplace_res);

      first_node_idx += node_count;
    }
    assert((first_node_idx - node_sequence.first) == node_sequence.second);

    if (!tile_idx_node_sequence_map.empty())
      // Rank value guaranteed to be unique from RankNodeSequenceMap
      // implementation
      dist_tns[rank].swap(tile_idx_node_sequence_map);
  }

  return dist_tns;
}
} // namespace sapi
