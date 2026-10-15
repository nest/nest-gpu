/*
 *  mask_node_processing.h
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

#ifndef MASK_NODE_PROCESSING_H
#define MASK_NODE_PROCESSING_H

#include "connection_generator.h"

namespace sapi {
// Forward definition to link with vp_interface.h
vp_t get_thread_num();
vp_t get_max_omp_threads();

// Forward definition to connection_containers.h
struct RankConnectionInfo;

// Forward definition to mask_collection.h
template <typename CoordT> struct MaskCollection;

// Forward definition to thread_aligned_array.h
template <typename T,
          typename std::enable_if_t<std::is_copy_constructible_v<T>, bool>>
class TAArray;

template <typename CoordT>
void add_task(TaskMap<CoordT> &aggregation_map, TaskQueue<CoordT> &task_queue,
              const FilteredLeafView<CoordT> pivot_view,
              const FilteredLeafView<CoordT> combination_view,
              const std::vector<CoordT> *const image_displacements,
              const count_t used_displacements, const combined_idx_t pivot_key,
              const combined_idx_t combination_key) {
  static_assert(std::is_trivially_copyable_v<FilteredLeafView<CoordT>>);

  const auto [task_it, success] =
      aggregation_map.try_emplace(pivot_key, nullptr);

  if (success) {
    task_it->second = &task_queue.emplace_back(pivot_key, pivot_view).second;
  } else {
    assert(task_it->second->pivot_view_ == pivot_view);
  }

  const auto combination_emplace =
      task_it->second->possible_combinations_.emplace(
          combination_key,
          construct_possible_connections<CoordT>(
              used_displacements, image_displacements, combination_view));
  assert(combination_emplace.second);

  task_it->second->total_possible_combinations_ +=
      static_cast<nodeidx_t>(combination_view->second.size());
}

template <typename CoordT, bool aggregate_by_local>
void generate_connection_tasks(TaskQueue<CoordT> &task_queue,
                               const RankPairInfo<CoordT> &rpi) {
  TaskMap<CoordT> aggregation_map;
  if constexpr (aggregate_by_local) {
    aggregation_map.reserve(rpi.tile_pairs_set_.valid_leaves_);
  } else {
    aggregation_map.reserve(rpi.receiver_info_.total_received_num_leaves_);
  }

  for (const auto &[local_tile_index, remote_tile_pairings] :
       rpi.tile_pairs_set_.tile_pairings_) {
    // Get aggregated local node coordinates
    const auto local_filtered_tile =
        rpi.sender_info_.filtered_coords_.find(local_tile_index);

    // Local tile index not found -> filtered out during masking
    if (local_filtered_tile == rpi.sender_info_.filtered_coords_.end() ||
        local_filtered_tile->second.empty())
      continue;

    const auto local_key_high = static_cast<combined_idx_t>(local_tile_index)
                                << 32;

    // Loop over remote tile pairings
    for (const auto &[remote_tile_index, tile_pair_info] :
         remote_tile_pairings) {
      if (tile_pair_info.aggregated_leaf_pairs_.empty())
        continue;

      // Get aggregated remote node coordinates
      const auto remote_filtered_tile =
          rpi.receiver_info_.filtered_coords_.find(remote_tile_index);

      // Remote tile not found -> filtered out during masking
      if (remote_filtered_tile == rpi.receiver_info_.filtered_coords_.end() ||
          remote_filtered_tile->second.empty())
        continue;

      assert(tile_pair_info.image_displacements_ != nullptr);

      const auto remote_key_high =
          static_cast<combined_idx_t>(remote_tile_index) << 32;

      // Loop over paired leafs aggregated by local leaf
      for (const auto &[local_leaf_index, remote_leaf_pairings] :
           tile_pair_info.aggregated_leaf_pairs_) {
        // Leaf pairing procedure guarantees that map entries
        // are only generated for matching pairs
        assert(!remote_leaf_pairings.empty());

        // Get node coordinates in local leaf
        const auto local_filtered_leaf =
            local_filtered_tile->second.find(local_leaf_index);

        // Local leaf not found -> filtered out during masking
        if (local_filtered_leaf == local_filtered_tile->second.end())
          continue;

        // If the local leaf is found then its vector cannot be empty
        assert(!local_filtered_leaf->second.empty());

        const auto local_key =
            local_key_high | static_cast<combined_idx_t>(local_leaf_index);

        // Loop over target leaf pairs
        for (const auto &[remote_leaf_index, used_displacements] :
             remote_leaf_pairings) {
          // Leaf pairing procedure guarantees that map entries
          // are only generated for matching pairs
          assert(0 < used_displacements);

          // Get node coordinates in remote leaf
          const auto remote_filtered_leaf =
              remote_filtered_tile->second.find(remote_leaf_index);

          // Remote leaf not found -> filtered out during masking
          if (remote_filtered_leaf == remote_filtered_tile->second.end())
            continue;

          if constexpr (aggregate_by_local) {
            add_task(aggregation_map, task_queue, local_filtered_leaf,
                     remote_filtered_leaf, tile_pair_info.image_displacements_,
                     used_displacements,
                     // Compose 64bit key made of tile index (high bits) and
                     // leaf index (low bits)
                     local_key,
                     remote_key_high |
                         static_cast<combined_idx_t>(remote_leaf_index));
          } else {
            add_task(aggregation_map, task_queue, remote_filtered_leaf,
                     local_filtered_leaf, tile_pair_info.image_displacements_,
                     used_displacements,
                     // Compose 64bit key made of tile index (high bits) and
                     // leaf index (low bits)
                     remote_key_high |
                         static_cast<combined_idx_t>(remote_leaf_index),
                     local_key);
          }
        }
      }
    }
  }
}

template <typename CoordT, bool inverted_source_target>
void compute_displacement_checks_across_tile_pairs(
    RankConnectionInfo &rci, RankPairInfo<CoordT> &rpi,
    const TAArray<MaskCollection<CoordT>> &mc_array,
    const TAArray<ConnectionGenerator> &cg_array,
    const RandomManager &rng_manager, const vp_t target_rank) {
  assert(rci.procedural_connections_.empty() && mc_array.is_initialized() &&
         cg_array.is_initialized() && rng_manager.is_initialized());

  // In the unlikely case that tiles passed mask filter
  // but no nodes in the tile passed the mask filter
  // (possible either in sender and/or receiver side)
  // then no work to be done
  if (rpi.tile_pairs_set_.tile_pairings_.empty() ||
      rpi.sender_info_.filtered_coords_.empty() ||
      rpi.receiver_info_.filtered_coords_.empty()) {
    rpi.tile_pairs_set_.tile_pairings_.clear();
    rpi.sender_info_.filtered_coords_.clear();
    rpi.receiver_info_.filtered_coords_.clear();
    rpi.remote_indexed_coord_cache_.clear();
    return;
  }

  const auto local_cg = cg_array.get_local_thread_item();
  rci.sort_by_pool_indexes_ = local_cg->sort_by_pool_indexes();
  rci.partition_connections_ = local_cg->partition_connections_;

  TaskQueue<CoordT> task_queue;
  if (inverted_source_target == rci.sort_by_pool_indexes_)
    generate_connection_tasks<CoordT, true>(task_queue, rpi);
  else
    generate_connection_tasks<CoordT, false>(task_queue, rpi);

  rpi.tile_pairs_set_.tile_pairings_.clear();

  if (!task_queue.empty()) {
    const auto num_threads = get_max_omp_threads();
    const std::size_t total_tasks = task_queue.size();
    rci.procedural_connections_.resize(total_tasks);

#pragma omp taskloop num_tasks(strict : num_threads) default(none)             \
    shared(rci, cg_array, mc_array, rng_manager, task_queue)                   \
    firstprivate(target_rank, total_tasks)
    for (std::size_t task_index = 0; task_index < total_tasks; ++task_index) {
      const auto tid = get_thread_num();
      auto &id_task_pair = task_queue[task_index];

      const auto total_conns =
          cg_array.get_thread_item(tid)->generate_connections(
              *rng_manager.reseed_rank_paired_rng(
                  tid, target_rank, inverted_source_target, id_task_pair.first),
              rci.procedural_connections_[task_index], id_task_pair.second,
              mc_array.get_thread_item(tid)->blueprint_,
              rng_manager.local_rank_ != static_cast<rng_seed_t>(target_rank));

#pragma omp atomic
      rci.total_generated_connections_ += total_conns;
    }
  }

  rpi.sender_info_.filtered_coords_.clear();
  rpi.receiver_info_.filtered_coords_.clear();
  rpi.remote_indexed_coord_cache_.clear();
}
} // namespace sapi

#endif
