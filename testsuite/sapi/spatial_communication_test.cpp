#include <chrono>
#include <cstdio>

#include "coordinate_geometry.h"

#include "bounding_box.h"
#include "connection_containers.h"
#include "connection_rules.h"
#include "gf2d_geometry.h"
#include "gf_creators.h"
#include "grid_generation.h"
#include "mask2d_geometry.h"
#include "mask_collection.h"
#include "mask_containers.h"
#include "node_containers_init.h"
#include "node_creation.h"
#include "spatial_communication.h"
#include "test_helpers.h"

using namespace sapi;

void test_all_to_all_connections(
    const std::vector<DistributedConnectionInfo> &all_to_all_connections,
    const int local_rank) {
#ifdef HAVE_MPI
  for (const auto &rci : all_to_all_connections) {
    std::forward_list<MPI_Request> io_send_requests;
    std::forward_list<std::vector<uint64_t>> send_buffers;
    std::array<std::unordered_map<vp_t, bool>, 2> io_done_map;
    std::array<std::unordered_map<vp_t, std::unordered_multimap<
                                            combined_idx_t, combined_idx_t>>,
               2>
        local_io_conns;
    std::array<std::unordered_map<vp_t, std::unordered_multimap<
                                            combined_idx_t, combined_idx_t>>,
               2>
        remote_io_conns;
    const std::array<const std::unordered_map<vp_t, RankConnectionInfo> *, 2>
        rank_conn_map = {&rci.outgoing_connections_,
                         &rci.incoming_connections_};
    for (const auto &io_idx : {0, 1}) {
      for (const auto &conn_map : *rank_conn_map[io_idx]) {
        if (conn_map.first == local_rank)
          continue;

        const auto &compressed_map =
            assert_emplace(
                local_io_conns[io_idx].emplace(
                    conn_map.first, conn_map.second.build_connection_map()))
                .first->second;

        assert(!compressed_map.empty() &&
               2 * compressed_map.size() < std::numeric_limits<count_t>::max());

        count_t index = 0;
        auto &buffer = send_buffers.emplace_front(2 * compressed_map.size());
        for (const auto &entry : compressed_map) {
          buffer[index] = entry.first;
          buffer[index + 1] = entry.second;
          index += 2;
        }
        assert(static_cast<std::size_t>(index) == buffer.size());

        assert_emplace(remote_io_conns[io_idx].emplace(
            conn_map.first,
            std::unordered_multimap<combined_idx_t, combined_idx_t>()));

        assert_emplace(io_done_map[io_idx].emplace(conn_map.first, false));

        io_send_requests.emplace_front(MPI_Request());

        MPI_Isend(buffer.data(), index, MPI_UINT64_T, conn_map.first, io_idx,
                  MPI_COMM_WORLD, &io_send_requests.front());
      }
    }

    // Ensure send triggered with testall
    for (auto &req : io_send_requests) {
      int flag;
      MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
    }

    MPI_Status status;
    bool done_recv = false;
    while (!done_recv) {
      done_recv = true;
      for (const auto &io_idx : {0, 1}) {
        const auto next_idx = (io_idx + 1) % 2;
        for (auto io_done_it = io_done_map[io_idx].begin();
             io_done_it != io_done_map[io_idx].end(); ++io_done_it) {
          if (io_done_it->second)
            continue;

          int flag;
          MPI_Iprobe(io_done_it->first, next_idx, MPI_COMM_WORLD, &flag,
                     &status);
          if (flag) {
            count_t recv_count;
            MPI_Get_count(&status, MPI_UINT64_T, &recv_count);
            assert(static_cast<std::size_t>(recv_count) ==
                   2 * local_io_conns[io_idx].at(io_done_it->first).size());
            std::vector<uint64_t> buffer(recv_count);
            MPI_Recv(buffer.data(), recv_count, MPI_UINT64_T, io_done_it->first,
                     next_idx, // mirrored recv
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            auto &recv_map = remote_io_conns[io_idx].at(io_done_it->first);
            recv_map.reserve(recv_count / 2);
            for (count_t index = 0; index < recv_count; index += 2) {
              recv_map.emplace(buffer[index], buffer[index + 1]);
            }

            assert(recv_map.size() == std::size_t(recv_count / 2));
            io_done_it->second = true;
          } else {
            done_recv = false;
          }
        }
      }
    }

    assert(local_io_conns == remote_io_conns);

    // Wait for send to complete
    for (auto &req : io_send_requests)
      MPI_Wait(&req, MPI_STATUS_IGNORE);
  }
#endif
}

void test_remote_node_connection() {
  const vp_t local_rank = get_mpi_rank();
  const tileidx_t num_processes =
      static_cast<tileidx_t>(get_num_mpi_processes());
  const nodeidx_t scale = 1000;

  TimerManager tman;
  tman.initialize();
  const auto treg = tman.get_rank_registry();
  const auto man_init_timer = treg->get_register_timer("manager_init_time");
  man_init_timer->start();

  NodeCountVector total_nodes_per_rank(num_processes, 0);
  RandomManager rman;
  rman.set_rng_seed(12345u);

  MaskCollection<Coord2D> mc;
  mc.blueprint_ = construct_mask<Coord2D>(MASK_SHAPE::CIRCULAR, {}, {0.5}, {});
  TAArray<MaskCollection<Coord2D>> mc_array;
  mc_array.clone(mc);

  ConnectionGenerator cg;
  cg.rule_ = CONNECTION_RULE::PAIRWISE_BERNOULLI;
  cg.numeric_functors_.delay_functor_.df_.func_ =
      DISPLACEMENT_FUNCTION::DISPLACEMENT_X;
  cg.numeric_functors_.weight_functor_.df_.func_ =
      DISPLACEMENT_FUNCTION::DISTANCE_Y;
  cg.numeric_functors_.probability_functor_.df_.func_ =
      DISPLACEMENT_FUNCTION::DISTANCE;

  TAArray<ConnectionGenerator> cgarr;
  cgarr.clone(cg);

  TileGrid<Coord2D> tg;
  GridNeighborhood gnb;
  GridNodeCollection<Coord2D> gnc;
  TAArray<GFCollection<Coord2D>> gc_array;
  CreatorRegistry<GridTargetPositionShifts<Coord2D>> gscr;
  CreatorRegistry<ShiftedOriginCreator<Coord2D>> socr;
  CreatorRegistry<CachedTileCreator<Coord2D>> ctcr;

  initialize_gsc_registry(gscr);
  initialize_soc_registry(socr);
  initialize_ctc_registry(ctcr);

  const auto gc = construct_gf_collection<Coord2D>(
      {0., 0.}, {2, num_processes}, "triangle", {1.}, {0}, gscr, socr, ctcr);
  gc_array.clone(gc);
  tg = generate_tile_grid(gc_array);

  std::vector<std::set<tileidx_t>> rank_tile_ownership_map(num_processes);
  for (tileidx_t rank = 0; rank < num_processes; ++rank)
    rank_tile_ownership_map[rank] = {rank, num_processes + rank};
  gnb.set_tile_ownership(std::move(rank_tile_ownership_map), tg);

  man_init_timer->stop();

  const auto split_timer = treg->get_register_timer("tile_split_time");
  split_timer->start();

  split_tiles_in_grid(tg, *gnb.locally_owned_tiles_, 10);
  initialize_local_grid_node_collection(gnc, *gnb.locally_owned_tiles_, tg);

  split_timer->stop();

  const auto node_create_timer =
      treg->get_register_timer("position_create_time");
  node_create_timer->start();

  std::vector<DistributedTiledNodeSequenceMap> dist_map(num_processes);
  for (tileidx_t idx = 0; idx < num_processes; ++idx) {
    const auto aggregated_node_counts = distribute_node_counts_in_grid(
        scale, {}, tg, gnb, rman, DISTRIBUTION_MODE::BALANCED);

    RankNodeSequenceMap rns_map;
    for (tileidx_t rank = 0; rank < num_processes; ++rank) {
      const auto num_nodes = aggregated_node_counts.first[rank];
      if (num_nodes < 1)
        continue;

      const auto next_index = total_nodes_per_rank[rank];
      assert_emplace(rns_map.emplace(
          std::make_pair(vp_t(rank), NodeSequence(next_index, num_nodes))));
      total_nodes_per_rank[rank] += num_nodes;
    }

    dist_map[idx] = consolidate_node_sequences_per_tile_per_rank(
        rns_map, aggregated_node_counts.second);

    if (const auto local_tns = dist_map[idx].find(local_rank);
        local_tns != dist_map[idx].end())
      distribute_node_counts_in_tiles(gnc, local_tns->second, tg, rman,
                                      DISTRIBUTION_MODE::BALANCED);
    assert(std::accumulate(total_nodes_per_rank.cbegin(),
                           total_nodes_per_rank.cend(),
                           0) == scale * (idx + 1));
    assert_equal_across_ranks(total_nodes_per_rank);
  }

  node_create_timer->stop();

  const auto node_connect_timer =
      treg->get_register_timer("external_connect_time");
  node_connect_timer->start();

  std::vector<DistributedConnectionInfo> all_to_all_connections;
  all_to_all_connections.reserve(num_processes * num_processes);
  for (const auto &dist0 : dist_map)
    for (const auto &dist1 : dist_map)
      all_to_all_connections.emplace_back(
          compute_distributed_spatial_connections(dist0, dist1, tg, gnb, gnc,
                                                  mc_array, cgarr, rman, tman,
                                                  true, true));

  node_connect_timer->stop();

  const auto test_conns_timer =
      treg->get_register_timer("test_distributed_connections_time");
  test_conns_timer->start();

  if (num_processes > 1)
    test_all_to_all_connections(all_to_all_connections, local_rank);

  test_conns_timer->stop();

  printf("RANK %i:\n%s", local_rank, tman.get_times().to_string(1).c_str());
}

int main(int argc, char **argv) {
  init_mpi(&argc, &argv);
  init_omp(4);

  test_remote_node_connection();

  finalize_mpi();
  return 0;
}
