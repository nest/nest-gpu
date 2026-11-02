/*
 *  spatial_manager.cpp
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

#include "spatial_manager.h"

namespace sapi {
BaseSpatialManager::BaseSpatialManager(const vp_t local_rank,
                                       const vp_t num_processes) noexcept
    : random_manager_(local_rank, num_processes),
      grid_neighborhood_(local_rank, num_processes) {}

ConnectionCounts
BaseSpatialManager::get_connection_counts(const std::size_t index) {
  const auto &dci = cached_connection_maps_.at(index);

  ConnectionCounts cc;
  cc.incoming_ranks_ = dci.incoming_connections_.size();
  cc.outgoing_ranks_ = dci.outgoing_connections_.size();

  cc.source_ranks_.resize(cc.incoming_ranks_);
  cc.incoming_counts_.resize(cc.incoming_ranks_);
  cc.target_ranks_.resize(cc.outgoing_ranks_);
  cc.outgoing_counts_.resize(cc.outgoing_ranks_);

  auto sr_it = cc.source_ranks_.begin();
  auto ic_it = cc.incoming_counts_.begin();
  for (const auto &rci : dci.incoming_connections_) {
    (*sr_it++) = rci.first;
    (*ic_it++) = rci.second.total_generated_connections_;
  }
  assert(sr_it == cc.source_ranks_.end() && ic_it == cc.incoming_counts_.end());

  auto tr_it = cc.target_ranks_.begin();
  auto oc_it = cc.outgoing_counts_.begin();
  for (const auto &rci : dci.outgoing_connections_) {
    (*tr_it++) = rci.first;
    (*oc_it++) = rci.second.total_generated_connections_;
  }
  assert(tr_it == cc.target_ranks_.end() && oc_it == cc.outgoing_counts_.end());

  return cc;
}
} // namespace sapi
