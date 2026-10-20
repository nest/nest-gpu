/*
 *  node_distribution.h
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

#ifndef NODE_DISTRIBUTION
#define NODE_DISTRIBUTION

#include <array>
#include <random>

#include "node_containers.h"
#include "random_manager.h"

namespace sapi {
// Forward definition to link with vp_interface.h
vp_t get_thread_num();
vp_t get_max_omp_threads();

// Forward definition to link with split_tree.h
tileidx_t collapse_dimensions(const std::vector<split_t> &branch_sequence,
                              const std::vector<split_t> &split_sequence);

template <typename T>
using rcvref = typename std::remove_cv_t<typename std::remove_reference_t<T>>;

template <typename I,
          typename std::enable_if_t<std::is_integral_v<rcvref<I>>, bool> = true>
NodeCountVector safe_convert_vector(std::vector<I> &&count_vector) {
  if constexpr (std::is_same_v<rcvref<I>, nodeidx_t>)
    return std::forward<NodeCountVector>(count_vector);
  else {
    NodeCountVector output_copy(count_vector.size());
    auto out_it = output_copy.begin();
    for (const auto &count : count_vector) {
      if (0 <= count && count < std::numeric_limits<nodeidx_t>::max())
        *out_it++ = static_cast<nodeidx_t>(count);
      else
        throw std::runtime_error(
            "Could not distribute node counts over available bins");
    }

    count_vector.clear();

    return output_copy;
  }
}

template <typename I, bool balanced,
          typename std::enable_if_t<std::is_integral_v<rcvref<I>>, bool> = true>
inline I balance_node_counts_by_bins(I &num_nodes, const tileidx_t num_bins) {
  if constexpr (balanced) {
    const auto balancing = std::div(num_nodes, static_cast<I>(num_bins));
    num_nodes = balancing.rem;
    return balancing.quot;
  }

  return 0;
}

template <typename I,
          typename std::enable_if_t<std::is_integral_v<rcvref<I>>, bool> = true>
inline std::pair<I, I>
batch_node_counts_by_thread_count(const I num_nodes, const vp_t num_threads) {
  // Batch tile index drawing with steps equal to the total number of threads
  I full_batches = 0, partial_batches = 0;

  // Only batch if running in multithreaded environment
  if (1 < num_threads) {
    const auto batch_steps = std::div(num_nodes, static_cast<I>(num_threads));
    full_batches = batch_steps.quot;
    partial_batches = batch_steps.rem;
  }

  if (full_batches < static_cast<I>(BATCHING_THRESHOLD)) {
    full_batches = 0;
    partial_batches = num_nodes;
  }

  assert((full_batches * num_threads + partial_batches) == num_nodes);

  return {full_batches, partial_batches};
}

template <typename I,
          typename std::enable_if_t<std::is_integral_v<rcvref<I>>, bool> = true>
void left_sum_vectors(std::vector<I> &left, std::vector<I> &right) {
  auto r_it = right.begin();
  for (auto l_it = left.begin(); l_it != left.end(); ++l_it)
    *l_it += *r_it++;
  right.clear();
}

template <typename I,
          typename std::enable_if_t<std::is_integral_v<rcvref<I>>, bool> = true>
void left_merge_buffers(
    std::vector<std::vector<I>> &buffer_vec,
    const vp_t num_buffers // Buffer size is always in number of threads
) {
  vp_t step = 1, step2 = 2;
  do {
#pragma omp taskloop num_tasks(strict : num_buffers) default(none)             \
    shared(buffer_vec) firstprivate(num_buffers, step, step2)
    for (vp_t curr_idx = 0; curr_idx < num_buffers; curr_idx += step2) {
      const auto next_idx = curr_idx + step;
      if (next_idx < num_buffers)
        left_sum_vectors(buffer_vec[curr_idx], buffer_vec[next_idx]);
    }
    step = step2;
    step2 *= 2;
  } while (step < num_buffers);
}

template <typename I, bool balanced,
          typename std::enable_if_t<std::is_integral_v<rcvref<I>>, bool> = true>
void batched_draw(std::vector<I> &buffer, I &batches, AnyRNG &rng,
                  std::uniform_int_distribution<tileidx_t> &distribution,
                  const tileidx_t num_bins) {
  if constexpr (balanced) {
    assert(batches < num_bins && 1 < num_bins);
    std::vector<tileidx_t> indexes(num_bins);
    std::iota(indexes.begin(), indexes.end(), 0);
    std::shuffle(indexes.begin(), indexes.end(), rng);

    while (0 < batches) {
      ++buffer[indexes[batches]];
      --batches;
    }
  } else {
    while (0 < batches) {
      ++buffer[distribution(rng)];
      --batches;
    }
  }
}

template <std::size_t N>
tileidx_t collapse_dimensions(const std::vector<tileidx_t> &grid_pos,
                              const std::array<tileidx_t, N> &grid_dimensions) {
  assert(grid_pos.size() == N);
  auto gp_it = grid_pos.rbegin();
  tileidx_t index = *gp_it++;
  for (auto gd_it = grid_dimensions.rbegin() + 1;
       gd_it != grid_dimensions.rend(); ++gd_it)
    index = *gp_it++ + *gd_it * index;
  return index;
}

template <typename I, typename ForwardDistIT, typename DimensionBoundsT,
          typename std::enable_if_t<std::is_integral_v<rcvref<I>>, bool> = true>
void batched_draw(std::vector<I> &buffer, I &batches, AnyRNG &rng,
                  const ForwardDistIT &distributions_first,
                  const DimensionBoundsT &dimensions) {
  std::vector<typename DimensionBoundsT::value_type> dimension_pos(
      dimensions.size(), 0);
  while (0 < batches) {
    auto dist_it = distributions_first;
    for (auto pos_it = dimension_pos.begin(); pos_it != dimension_pos.end();
         ++pos_it)
      *pos_it = (*dist_it++)(rng);
    ++buffer[collapse_dimensions(dimension_pos, dimensions)];
    --batches;
  }
}

template <typename I, bool balanced, bool global,
          typename std::enable_if_t<std::is_integral_v<rcvref<I>>, bool> = true>
NodeCountVector
uniform_distribute_node_counts(I num_nodes, const tileidx_t num_bins,
                               const RandomManager &rng_manager) {
  assert(0 < num_bins &&
         0 < num_nodes // RNG manager initialization is assumed already tested
  );

  // Trivial case
  if (num_bins == 1)
    return safe_convert_vector(std::vector<I>{num_nodes});

  // Initialize map and rng distribution
  const auto init_val =
      balance_node_counts_by_bins<I, balanced>(num_nodes, num_bins);
  std::vector<I> count_vec(num_bins, init_val);
  if (num_nodes == 0)
    return safe_convert_vector(std::move(count_vec));

  const auto num_threads = get_max_omp_threads();
  auto [full_batches, partial_batches] =
      batch_node_counts_by_thread_count(num_nodes, num_threads);

  std::uniform_int_distribution<tileidx_t> tile_idx_dist(0, num_bins - 1);

  if (0 < full_batches) {
    std::vector<std::vector<I>> buffers(num_threads);

#pragma omp taskloop num_tasks(strict : num_threads) default(none)             \
    shared(rng_manager, buffers)                                               \
    firstprivate(num_threads, num_bins, full_batches, tile_idx_dist)
    for (vp_t curr_idx = 0; curr_idx < num_threads; ++curr_idx) {
      const auto curr_buff = buffers.begin() + curr_idx;
      curr_buff->resize(num_bins, 0);
      if constexpr (global) {
        batched_draw<I, balanced>(*curr_buff, full_batches,
                                  *rng_manager.get_tid_synced_rng(curr_idx),
                                  tile_idx_dist, num_bins);
      } else {
        batched_draw<I, balanced>(*curr_buff, full_batches,
                                  *rng_manager.get_tid_specific_rng(curr_idx),
                                  tile_idx_dist, num_bins);
      }
    }

    left_merge_buffers<I>(buffers, num_threads);
    count_vec.swap(buffers[0]);
    buffers.clear();

    if (0 < init_val)
      for (auto &nc : count_vec)
        nc += init_val;
  }

  if (0 < partial_batches) {
    if constexpr (global) {
      batched_draw<I, balanced>(count_vec, partial_batches,
                                *rng_manager.get_rank_synced_rng(),
                                tile_idx_dist, num_bins);
    } else {
      batched_draw<I, balanced>(count_vec, partial_batches,
                                *rng_manager.get_rank_specific_rng(),
                                tile_idx_dist, num_bins);
    }
  }

  return safe_convert_vector(std::move(count_vec));
}

template <typename I, typename DimensionBoundsT, bool global,
          typename std::enable_if_t<std::is_integral_v<rcvref<I>>, bool> = true>
NodeCountVector
uniform_distribute_node_counts(I num_nodes, const tileidx_t num_bins,
                               const DimensionBoundsT &dimensions,
                               const RandomManager &rng_manager) {
  // Trivial case
  if (num_bins == 1)
    return safe_convert_vector(std::vector<I>{num_nodes});

  // Initialize map and rng distribution
  std::vector<I> count_vec(num_bins, 0);
  if (num_nodes == 0)
    return safe_convert_vector(std::move(count_vec));

  const auto num_threads = get_max_omp_threads();
  auto [full_batches, partial_batches] =
      batch_node_counts_by_thread_count(num_nodes, num_threads);

  std::vector<std::uniform_int_distribution<tileidx_t>> distributions;
  distributions.reserve(dimensions.size());
  for (const auto &dim : dimensions)
    distributions.emplace_back(0, static_cast<tileidx_t>(dim - 1));

  if (0 < full_batches) {
    std::vector<std::vector<I>> buffers(num_threads);

#pragma omp taskloop num_tasks(strict : num_threads) default(none)             \
    shared(rng_manager, buffers) firstprivate(                                 \
            num_threads, num_bins, full_batches, distributions, dimensions)
    for (vp_t curr_idx = 0; curr_idx < num_threads; ++curr_idx) {
      const auto curr_buff = buffers.begin() + curr_idx;
      curr_buff->resize(num_bins, 0);
      if constexpr (global) {
        batched_draw(*curr_buff, full_batches,
                     *rng_manager.get_tid_synced_rng(curr_idx),
                     distributions.begin(), dimensions);
      } else {
        batched_draw(*curr_buff, full_batches,
                     *rng_manager.get_tid_specific_rng(curr_idx),
                     distributions.begin(), dimensions);
      }
    }

    left_merge_buffers<I>(buffers, num_threads);
    count_vec.swap(buffers[0]);
    buffers.clear();
  }

  if (0 < partial_batches) {
    if constexpr (global) {
      batched_draw(count_vec, partial_batches,
                   *rng_manager.get_rank_synced_rng(), distributions.begin(),
                   dimensions);
    } else {
      batched_draw(count_vec, partial_batches,
                   *rng_manager.get_rank_specific_rng(), distributions.begin(),
                   dimensions);
    }
  }

  return safe_convert_vector(std::move(count_vec));
}

// Here it is assumed that node sequences in each rank are generated externally
// i.e. this library generates a number of nodes per tiles then aggregates by
// rank then another library instantiates the nodes in the rank and returns the
// sequence of nodes generated in the rank
DistributedTiledNodeSequenceMap consolidate_node_sequences_per_tile_per_rank(
    const RankNodeSequenceMap &node_sequences_per_rank,
    const RankTileIdxNodeCountPairs &node_counts_per_tile_per_rank);
} // namespace sapi

#endif
