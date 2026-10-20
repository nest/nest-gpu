/*
 *  spatial_communication.h
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

#ifndef SPATIAL_COMMUNICATION_H
#define SPATIAL_COMMUNICATION_H

#include "mask_node_processing.h"
#include "mask_tile_processing.h"
#include "timer_manager.h"
#include "vp_interface.h"

namespace sapi {
// Forward definition to link with payload_preparation.h
template <typename CoordT>
void reconstruct_received_info(CommunicationInfo<CoordT> &comm_info,
                               RemoteIndexedCoordCache<CoordT> &remote_cache);

template <typename CoordT, bool inverted_source_target>
void compute_target_connections(RankConnectionInfo &rci,
                                RankPairInfo<CoordT> &rpi,
                                const TAArray<MaskCollection<CoordT>> &mc_array,
                                const TAArray<ConnectionGenerator> &cg_array,
                                const RandomManager &rng_manager,
                                const TimerManager &timer_manager,
                                const vp_t target_rank) {
  const auto thread_timer_registry = timer_manager.get_thread_registry();

  const auto node_disp_timer = thread_timer_registry->get_register_timer(
      "compute_node_displacements_time");
  node_disp_timer->start();

  compute_displacement_checks_across_tile_pairs<CoordT, inverted_source_target>(
      rci, rpi, mc_array, cg_array, rng_manager, target_rank);

  node_disp_timer->stop();

  const auto map_consolidation_timer =
      thread_timer_registry->get_register_timer("consolidate_connections_time");
  map_consolidation_timer->start();

  rci.consolidate_connection_map();

  map_consolidation_timer->stop();
}

template <typename CoordT>
DistributedConnectionInfo compute_distributed_spatial_connections(
    const DistributedTiledNodeSequenceMap &dist_tns_source,
    const DistributedTiledNodeSequenceMap &dist_tns_target,
    const TileGrid<CoordT> &tile_grid,
    const GridNeighborhood &grid_neighborhood,
    const GridNodeCollection<CoordT> &grid_node_col,
    const TAArray<MaskCollection<CoordT>> &mc_array,
    const TAArray<ConnectionGenerator> &cg_array,
    const RandomManager &rng_manager, const TimerManager &timer_manager,
    const bool edge_wrap, const bool only_neighborhood) {
  assert(timer_manager.is_initialized());

  if (!mc_array.get_local_thread_item()->has_blueprint())
    throw std::invalid_argument("Spatial connections require mask blueprints");

  const auto rank_timer_registry = timer_manager.get_rank_registry();

  const auto connect_timer = rank_timer_registry->get_register_timer(
      "compute_distributed_spatial_connections_time");
  connect_timer->start();

  DistributedConnectionInfo dci;
  DistributedPairInfo<CoordT> dpi;
  RankPairInfo<CoordT> local_pairings;

#pragma omp parallel default(none)                                             \
    shared(dci, dpi, local_pairings, dist_tns_source, dist_tns_target,         \
               tile_grid, grid_neighborhood, grid_node_col, mc_array,          \
               cg_array, rng_manager, timer_manager)                           \
    firstprivate(rank_timer_registry, edge_wrap, only_neighborhood)
#pragma omp master
#pragma omp taskgroup
  {
    const auto tile_overlap_timer = rank_timer_registry->get_register_timer(
        "compute_distributed_tile_overlap_time");
    tile_overlap_timer->start();

    std::pair<bool, bool> available_targets;

    switch ((edge_wrap << 0) + (only_neighborhood << 1)) {
    case 1: {
      available_targets = compute_distributed_tile_overlap<CoordT, true, false>(
          dpi, dist_tns_source, dist_tns_target, tile_grid, grid_neighborhood,
          grid_node_col, mc_array);

      break;
    }

    case 2: {
      available_targets = compute_distributed_tile_overlap<CoordT, false, true>(
          dpi, dist_tns_source, dist_tns_target, tile_grid, grid_neighborhood,
          grid_node_col, mc_array);

      break;
    }

    case 3: {
      available_targets = compute_distributed_tile_overlap<CoordT, true, true>(
          dpi, dist_tns_source, dist_tns_target, tile_grid, grid_neighborhood,
          grid_node_col, mc_array);

      break;
    }

    default: {
      available_targets =
          compute_distributed_tile_overlap<CoordT, false, false>(
              dpi, dist_tns_source, dist_tns_target, tile_grid,
              grid_neighborhood, grid_node_col, mc_array);

      break;
    }
    }

    tile_overlap_timer->stop();

    const auto node_overlap_timer = rank_timer_registry->get_register_timer(
        "dispatch_distributed_node_overlap_time");
    node_overlap_timer->start();

    if (available_targets.first) {
      const auto ssi_it =
          dpi.source_side_info_.find(grid_neighborhood.local_rank_);
      const auto tsi_it =
          dpi.target_side_info_.find(grid_neighborhood.local_rank_);
      const bool source_side = ssi_it != dpi.source_side_info_.end();
      const bool target_side = tsi_it != dpi.target_side_info_.end();
      assert(source_side == target_side &&
             source_side == available_targets.second);

      if (available_targets.second) {
        assert(ssi_it->second.sender_info_.data_payload_.empty() &&
               tsi_it->second.sender_info_.data_payload_.empty());

        local_pairings = std::move(ssi_it->second);
        local_pairings.receiver_info_ = std::move(tsi_it->second.sender_info_);

        dpi.source_side_info_.erase(ssi_it);
        dpi.target_side_info_.erase(tsi_it);

        const auto [emplace_it, success] = dci.incoming_connections_.emplace(
            grid_neighborhood.local_rank_, RankConnectionInfo());
        assert(success);

#pragma omp task default(none)                                                 \
    shared(local_pairings, mc_array, cg_array, rng_manager, timer_manager)     \
    firstprivate(emplace_it)
        compute_target_connections<CoordT, false>(
            emplace_it->second, local_pairings, mc_array, cg_array, rng_manager,
            timer_manager, emplace_it->first);
      }

      const bool empty_dpi =
          dpi.source_side_info_.empty() && dpi.target_side_info_.empty();
      assert(empty_dpi <= available_targets.first);

      if (!empty_dpi && grid_neighborhood.num_processes_ > 1) {
        const auto communicate_timer = rank_timer_registry->get_register_timer(
            "dispatch_communicate_node_positions_time");
        communicate_timer->start();

        communicate_connect_distributed_pair_data(dci, dpi, mc_array, cg_array,
                                                  rng_manager, timer_manager);

        communicate_timer->stop();
      }

      node_overlap_timer->stop();
    }
  }

  connect_timer->stop();

  const auto reseed_timer =
      rank_timer_registry->get_register_timer("seed_update_cleanup_time");
  reseed_timer->start();

  for (auto &rank_map :
       {&dci.incoming_connections_, &dci.outgoing_connections_}) {
    auto rci_it = rank_map->begin();
    const auto rci_end = rank_map->end();
    while (rci_it != rci_end) {
      rng_manager.update_rank_paired_seed(rci_it->first);

      if (rci_it->second.partitioned_connections_.empty())
        rci_it = rank_map->erase(rci_it);
      else
        ++rci_it;
    }
  }

  reseed_timer->stop();

  return dci;
}

#ifdef HAVE_MPI
struct QueuedRequests {
  std::unordered_set<vp_t> pending_ranks_;
  std::unordered_set<vp_t> done_ranks_;
};

template <typename CoordT>
void communicate_payload_async(
    QueuedRequests &request_queue,
    std::vector<MPI_Request>::iterator &request_it,
    const std::unordered_map<vp_t, RankPairInfo<CoordT>> &rank_info_map,
    const vp_t tag) {
  for (const auto &[rank, rpi] : rank_info_map) {
    assert(request_queue.pending_ranks_.find(rank) ==
               request_queue.pending_ranks_.end() &&
           rpi.sender_info_.data_payload_.size() <
               std::numeric_limits<vp_t>::max());
    const auto payload_size =
        static_cast<vp_t>(rpi.sender_info_.data_payload_.size());
    assert(0 < payload_size);

    MPI_Isend(rpi.sender_info_.data_payload_.data(), payload_size, MPI_UINT64_T,
              rank, tag, MPI_COMM_WORLD, &(*request_it++));

    request_queue.pending_ranks_.insert(rank);
  }
}

template <typename CoordT>
void probe_payload_recv(
    QueuedRequests &request_queue, MPI_Status &status,
    std::unordered_map<vp_t, RankPairInfo<CoordT>> &rank_info_map,
    const vp_t tag) {
  if (request_queue.pending_ranks_.empty())
    return;

  vp_t flag;
  auto ranks_it = request_queue.pending_ranks_.cbegin();
  const auto ranks_end = request_queue.pending_ranks_.cend();
  while (ranks_it != ranks_end) {
    flag = 0;
    MPI_Iprobe(*ranks_it, tag, MPI_COMM_WORLD, &flag, &status);
    if (flag) {
      MPI_Get_count(&status, MPI_UINT64_T, &flag);
      assert(0 < flag);

      const auto rank_map_it = rank_info_map.find(*ranks_it);
      assert(rank_map_it != rank_info_map.end() &&
             rank_map_it->second.receiver_info_.data_payload_.empty());
      rank_map_it->second.receiver_info_.data_payload_.resize(flag);

      MPI_Recv(rank_map_it->second.receiver_info_.data_payload_.data(), flag,
               MPI_UINT64_T, *ranks_it, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      assert(request_queue.done_ranks_.find(*ranks_it) ==
             request_queue.done_ranks_.end());
      request_queue.done_ranks_.insert(*ranks_it);
      ranks_it = request_queue.pending_ranks_.erase(ranks_it);
    } else
      ++ranks_it;
  }
}

template <typename CoordT, bool inverted_source_target>
void dispatch_connection_procedures(
    QueuedRequests &request_queue,
    std::unordered_map<vp_t, RankConnectionInfo> &connection_info_map,
    std::unordered_map<vp_t, RankPairInfo<CoordT>> &rank_info_map,
    const TAArray<MaskCollection<CoordT>> &mc_array,
    const TAArray<ConnectionGenerator> &cg_array,
    const RandomManager &rng_manager, const TimerManager &timer_manager) {
  if (request_queue.done_ranks_.empty())
    return;

  auto ranks_it = request_queue.done_ranks_.cbegin();
  const auto ranks_end = request_queue.done_ranks_.cend();
  while (ranks_it != ranks_end) {
    const auto rank_info_it = rank_info_map.find(*ranks_it);
    assert(rank_info_it != rank_info_map.end());
    ranks_it = request_queue.done_ranks_.erase(ranks_it);

    const auto [emplace_it, success] =
        connection_info_map.emplace(rank_info_it->first, RankConnectionInfo());
    assert(success);

#pragma omp task default(none)                                                 \
    shared(mc_array, cg_array, rng_manager, timer_manager)                     \
    firstprivate(emplace_it, rank_info_it)
    {
      reconstruct_received_info(
          rank_info_it->second.receiver_info_,
          rank_info_it->second.remote_indexed_coord_cache_);
      compute_target_connections<CoordT, inverted_source_target>(
          emplace_it->second, rank_info_it->second, mc_array, cg_array,
          rng_manager, timer_manager, rank_info_it->first);
    }
  }
}
#endif

template <typename CoordT>
void communicate_connect_distributed_pair_data(
    DistributedConnectionInfo &distributed_connection_map,
    DistributedPairInfo<CoordT> &distributed_pair_data,
    const TAArray<MaskCollection<CoordT>> &mc_array,
    const TAArray<ConnectionGenerator> &cg_array,
    const RandomManager &rng_manager, const TimerManager &timer_manager) {
#ifdef HAVE_MPI
  const auto num_requests = distributed_pair_data.source_side_info_.size() +
                            distributed_pair_data.target_side_info_.size();
  assert(0 < num_requests);

  std::vector<MPI_Request> send_requests(num_requests);
  auto req_it = send_requests.begin();

  QueuedRequests source_side_queue;
  QueuedRequests target_side_queue;

  const auto rank_timer_registry = timer_manager.get_rank_registry();

  const auto send_async_timer =
      rank_timer_registry->get_register_timer("send_async_positions_time");
  send_async_timer->start();

  if (!distributed_pair_data.source_side_info_.empty())
    communicate_payload_async(source_side_queue, req_it,
                              distributed_pair_data.source_side_info_,
                              0 // tag 0 for sending source side positions
    );

  if (!distributed_pair_data.target_side_info_.empty())
    communicate_payload_async(target_side_queue, req_it,
                              distributed_pair_data.target_side_info_,
                              1 // tag 1 for sending target side positions
    );

  assert(req_it == send_requests.end());

  send_async_timer->stop();

  const auto trigger_send_timer =
      rank_timer_registry->get_register_timer("trigger_send_timer");
  trigger_send_timer->start();

  vp_t flag;
  MPI_Testall(send_requests.size(), send_requests.data(), &flag,
              MPI_STATUSES_IGNORE);

  trigger_send_timer->stop();

  const auto last_dispatch_timer = rank_timer_registry->get_register_timer(
      "last_compute_connection_dispatch_timer");
  const auto first_dispatch_timer = rank_timer_registry->get_register_timer(
      "first_compute_connection_dispatch_timer");

  last_dispatch_timer->start();
  first_dispatch_timer->start();

  MPI_Status status;
  bool first_dispatch = true;
  while (!(source_side_queue.done_ranks_.empty() &&
           source_side_queue.pending_ranks_.empty()) ||
         !(target_side_queue.done_ranks_.empty() &&
           target_side_queue.pending_ranks_.empty())) {
    probe_payload_recv(source_side_queue, status,
                       distributed_pair_data.source_side_info_,
                       1 // mirrored tag for receiving
    );

    if (!source_side_queue.done_ranks_.empty()) {
      dispatch_connection_procedures<CoordT, false>(
          source_side_queue, distributed_connection_map.outgoing_connections_,
          distributed_pair_data.source_side_info_, mc_array, cg_array,
          rng_manager, timer_manager);

      if (first_dispatch) {
        first_dispatch_timer->stop();
        first_dispatch = false;
      }
    }

    probe_payload_recv(target_side_queue, status,
                       distributed_pair_data.target_side_info_,
                       0 // mirrored tag for receiving
    );

    if (!target_side_queue.done_ranks_.empty()) {
      dispatch_connection_procedures<CoordT, true>(
          target_side_queue, distributed_connection_map.incoming_connections_,
          distributed_pair_data.target_side_info_, mc_array, cg_array,
          rng_manager, timer_manager);

      if (first_dispatch) {
        first_dispatch_timer->stop();
        first_dispatch = false;
      }
    }
  }

  last_dispatch_timer->stop();

  const auto wait_send_timer =
      rank_timer_registry->get_register_timer("wait_send_timer");
  wait_send_timer->start();

  MPI_Waitall(send_requests.size(), send_requests.data(), MPI_STATUSES_IGNORE);

  // At this point it is safe to clear sender payloads
  for (const auto &rpi_map_ptr : {&distributed_pair_data.source_side_info_,
                                  &distributed_pair_data.target_side_info_})
    for (auto &rpi_pair : *rpi_map_ptr)
      rpi_pair.second.sender_info_.data_payload_.clear();

  wait_send_timer->stop();
#endif
}
} // namespace sapi

#endif
