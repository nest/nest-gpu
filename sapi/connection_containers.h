/*
 *  connection_containers.h
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

#ifndef CONNECTION_CONTAINERS_H
#define CONNECTION_CONTAINERS_H

#include <limits>
#include <map>

#include "node_containers.h"

namespace sapi {
struct ConnectionInfo {
  conn_index_t index_;
  conn_param_t weight_;
  conn_param_t delay_;

  bool operator==(const ConnectionInfo &) const;
};

inline ConnectionInfo construct_connection_info(const conn_index_t index,
                                                const conn_param_t weight,
                                                const conn_param_t delay) {
  ConnectionInfo ci;
  ci.index_ = index;
  ci.weight_ = weight;
  ci.delay_ = delay;
  return ci;
}

inline bool ConnectionInfo::operator==(const ConnectionInfo &ci) const {
  return index_ == ci.index_ && weight_ == ci.weight_ && delay_ == ci.delay_;
}

typedef std::vector<std::pair<conn_param_t, std::deque<ConnectionInfo>>>
    ProceduralConnectivityBlocks;

struct ConnectionBounds {
  conn_index_t first_source_index_ = std::numeric_limits<conn_index_t>::max();
  conn_index_t last_source_index_ = 0;
  conn_index_t first_target_index_ = std::numeric_limits<conn_index_t>::max();
  conn_index_t last_target_index_ = 0;

  void update_bounds(const ConnectionBounds &);
  void update_first_last_source(const conn_index_t index);
  void update_first_last_target(const conn_index_t index);
};

inline void ConnectionBounds::update_bounds(const ConnectionBounds &cb) {
  first_source_index_ = std::min(first_source_index_, cb.first_source_index_);
  last_source_index_ = std::max(last_source_index_, cb.last_source_index_);
  first_target_index_ = std::min(first_target_index_, cb.first_target_index_);
  last_target_index_ = std::max(last_target_index_, cb.last_target_index_);
}

inline void
ConnectionBounds::update_first_last_source(const conn_index_t index) {
  first_source_index_ = std::min(index, first_source_index_);
  last_source_index_ = std::max(index, last_source_index_);
}

inline void
ConnectionBounds::update_first_last_target(const conn_index_t index) {
  first_target_index_ = std::min(index, first_target_index_);
  last_target_index_ = std::max(index, last_target_index_);
}

struct ConnectionVectors {
  std::size_t sizes_ = 0;
  ConnectionBounds bounds_;

  std::vector<conn_index_t> connection_sources_;
  std::vector<conn_index_t> connection_targets_;
  std::vector<conn_param_t> connection_weights_;
  std::vector<conn_param_t> connection_delays_;

  ConnectionVectors() noexcept = default;
  ConnectionVectors(const ConnectionVectors &) = delete;
  ConnectionVectors(ConnectionVectors &&) noexcept = default;
  ~ConnectionVectors() noexcept = default;

  ConnectionVectors &operator=(const ConnectionVectors &) = delete;
  ConnectionVectors &operator=(ConnectionVectors &&) = delete;

  void prepare_vectors(const std::size_t size);

  void copy_from_procedural_connections(
      std::vector<ProceduralConnectivityBlocks> &procedural_connections,
      const std::size_t total_procedural_connections,
      const bool inverted_pivot);
};

struct RankConnectionInfo {
  bool sort_by_pool_indexes_ = false;
  bool partition_connections_ = false;
  std::size_t total_generated_connections_ = 0;

  std::vector<ProceduralConnectivityBlocks> procedural_connections_;

  std::vector<ConnectionVectors> partitioned_connections_;

  RankConnectionInfo() noexcept = default;
  RankConnectionInfo(const RankConnectionInfo &) = delete;
  RankConnectionInfo(RankConnectionInfo &&) noexcept = default;
  ~RankConnectionInfo() noexcept = default;

  RankConnectionInfo &operator=(const RankConnectionInfo &) = delete;
  RankConnectionInfo &operator=(RankConnectionInfo &&) = delete;

  bool operator==(const RankConnectionInfo &) const;

  std::unordered_multimap<combined_idx_t, combined_idx_t>
  build_connection_map() const;

  void consolidate_connection_map();
};

inline bool
RankConnectionInfo::operator==(const RankConnectionInfo &tci) const {
  return build_connection_map() == tci.build_connection_map();
}

struct DistributedConnectionInfo {
  std::unordered_map<vp_t, RankConnectionInfo> outgoing_connections_;
  std::unordered_map<vp_t, RankConnectionInfo> incoming_connections_;

  DistributedConnectionInfo() noexcept = default;
  DistributedConnectionInfo(const DistributedConnectionInfo &) = delete;
  DistributedConnectionInfo(DistributedConnectionInfo &&rci) noexcept = default;
  ~DistributedConnectionInfo() noexcept = default;

  DistributedConnectionInfo &
  operator=(const DistributedConnectionInfo &) = delete;
  DistributedConnectionInfo &operator=(DistributedConnectionInfo &&) = delete;

  bool operator==(const DistributedConnectionInfo &) const;

  void clear();
};

inline bool DistributedConnectionInfo::operator==(
    const DistributedConnectionInfo &rci) const {
  return outgoing_connections_ == rci.outgoing_connections_ &&
         incoming_connections_ == rci.incoming_connections_;
}

inline void DistributedConnectionInfo::clear() {
  outgoing_connections_.clear();
  incoming_connections_.clear();
}

template <typename CoordT> struct PossibleConnections {
  count_t used_displacements_;
  const std::vector<CoordT> *image_displacements_;
  FilteredLeafView<CoordT> possible_pairs_;
};

template <typename CoordT>
inline PossibleConnections<CoordT> construct_possible_connections(
    const count_t used_displacements,
    const std::vector<CoordT> *const image_displacements,
    const FilteredLeafView<CoordT> possible_pairs) {
  static_assert(std::is_trivially_copyable_v<FilteredLeafView<CoordT>>);
  PossibleConnections<CoordT> cp;
  cp.used_displacements_ = used_displacements;
  cp.image_displacements_ = image_displacements;
  cp.possible_pairs_ = possible_pairs;
  return cp;
}

template <typename CoordT> struct ConnectionTask {
  nodeidx_t total_possible_combinations_ = 0;
  const FilteredLeafView<CoordT> pivot_view_;
  // Sorted for reproducibility
  std::map<combined_idx_t, PossibleConnections<CoordT>> possible_combinations_;

  ConnectionTask() = delete;
  ConnectionTask(const ConnectionTask &) = delete;
  ConnectionTask(ConnectionTask &&) noexcept = default;
  ~ConnectionTask() noexcept = default;

  ConnectionTask(const FilteredLeafView<CoordT> pivot_view) noexcept;

  ConnectionTask &operator=(const ConnectionTask &) = delete;
  ConnectionTask &operator=(ConnectionTask &&) = delete;
};

template <typename CoordT>
ConnectionTask<CoordT>::ConnectionTask(
    const FilteredLeafView<CoordT> pivot_view) noexcept
    : pivot_view_(pivot_view) {}

template <typename CoordT>
using TaskMap = std::unordered_map<combined_idx_t, ConnectionTask<CoordT> *>;

template <typename CoordT>
using TaskQueue = std::deque<std::pair<combined_idx_t, ConnectionTask<CoordT>>>;
} // namespace sapi

#endif
