#include <cstdio>

#include <unordered_set>

#include "api_converters.h"
#include "c_api.h"
#include "numerics.h"
#include "test_helpers.h"
#include "vp_interface.h"

using namespace sapi;

static GC gc;
static CAPI capi;
static std::vector<nodeidx_t> total_nodes_per_rank;

inline void reset_api() {
  capi.reset();
  gc.free_gc();
  total_nodes_per_rank.clear();
  total_nodes_per_rank.resize(capi.get_num_processes(), 0);
}

std::pair<tileidx_t, tileidx_t> get_middle_factors(const vp_t num) {
  const vp_t step = (num % 2 == 0) ? 1 : 2;
  const vp_t bound = vp_t(std::floor(std::sqrt(num))) + 1;
  vp_t lower_factor = 1;
  vp_t higher_factor = num;
  for (vp_t i = 1; i < bound; i += step) {
    if (num % i == 0) {
      lower_factor = i;
      higher_factor = vp_t(num / i);
    }
  }

  return {static_cast<tileidx_t>(lower_factor),
          static_cast<tileidx_t>(higher_factor)};
}

inline combined_idx_t get_global_index(const nodeidx_t local_index) {
  assert(0 <= local_index);
  return (static_cast<combined_idx_t>(capi.get_rank()) << 32) |
         static_cast<combined_idx_t>(local_index);
}

inline combined_idx_t get_global_index(const nodeidx_t local_index,
                                       const vp_t rank) {
  assert(0 <= local_index && 0 <= rank);
  return (static_cast<combined_idx_t>(rank) << 32) |
         static_cast<combined_idx_t>(local_index);
}

std::size_t insert_nodes(const std::vector<space_t> &anycoord_vec) {
  PositionViewStruct pos;
  pos.dimensions_ = 2;
  pos.coord_count_ = anycoord_vec.size();
  pos.coordinates_ = anycoord_vec.data();
  const auto [node_counts_per_rank, leftovers] =
      capi.insert_positions_in_grid(pos);
  assert(leftovers->coord_count_ == 0);

  RankNodeSequenceMap rns_map;
  for (tileidx_t rank = 0; rank < capi.get_num_processes(); ++rank) {
    const auto num_nodes = node_counts_per_rank[rank];
    if (num_nodes < 1)
      continue;

    const auto next_index = total_nodes_per_rank[rank];
    assert_emplace(rns_map.emplace(
        std::make_pair(vp_t(rank), NodeSequence(next_index, num_nodes))));
    total_nodes_per_rank[rank] += num_nodes;
  }
  assert_equal_across_ranks(total_nodes_per_rank);

  return capi.insert_positions_in_tiles(rns_map);
}

std::size_t generate_nodes(const nodeidx_t num_nodes) {
  assert(0 < num_nodes);
  const auto node_counts_per_rank =
      capi.generate_nodes_in_grid(num_nodes, {}, 2);

  RankNodeSequenceMap rns_map;
  for (tileidx_t rank = 0; rank < capi.get_num_processes(); ++rank) {
    const auto num_nodes = node_counts_per_rank[rank];
    if (num_nodes < 1)
      continue;

    const auto next_index = total_nodes_per_rank[rank];
    assert_emplace(rns_map.emplace(
        std::make_pair(vp_t(rank), NodeSequence(next_index, num_nodes))));
    total_nodes_per_rank[rank] += num_nodes;
  }
  assert_equal_across_ranks(total_nodes_per_rank);

  return capi.generate_nodes_in_tiles(rns_map, 2);
}

void communicate_all_nodes(const NodesViewStruct &nodes,
                           const std::size_t total_size) {
  assert(0 < nodes.node_count_);

  std::unordered_set<uint64_t> indexes;
  for (std::size_t index = 0; index < nodes.node_count_; ++index)
    indexes.insert(get_global_index(nodes.indexes_[index]));

  if (1 < capi.get_num_processes()) {
    assert(nodes.node_count_ < total_size);
#ifdef HAVE_MPI

    std::vector<MPI_Request> requests(capi.get_num_processes() - 1);
    auto req_it = requests.begin();
    for (vp_t rank = 0; rank < capi.get_num_processes(); ++rank) {
      if (rank == capi.get_rank())
        continue;
      MPI_Isend(nodes.indexes_, nodes.node_count_, MPI_INT32_T, rank, 0,
                MPI_COMM_WORLD, &(*req_it++));
    }

    vp_t flag;
    MPI_Testall(requests.size(), requests.data(), &flag, MPI_STATUSES_IGNORE);

    MPI_Status status;
    std::set<vp_t> recv_set{capi.get_rank()};
    while (recv_set.size() <
           static_cast<std::size_t>(capi.get_num_processes())) {
      for (vp_t rank = 0; rank < capi.get_num_processes(); ++rank) {
        if (recv_set.find(rank) != recv_set.end())
          continue;

        flag = 0;
        MPI_Iprobe(rank, 0, MPI_COMM_WORLD, &flag, &status);
        if (!flag)
          continue;

        MPI_Get_count(&status, MPI_INT32_T, &flag);
        assert(0 < flag);
        std::vector<int32_t> recv_payload(flag);
        MPI_Recv(recv_payload.data(), flag, MPI_UINT64_T, rank, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        recv_set.insert(rank);

        for (auto rp_it = recv_payload.cbegin(); rp_it != recv_payload.cend();
             ++rp_it)
          indexes.insert(get_global_index(*rp_it, rank));
      }
    }

    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
#endif
  }

  assert(indexes.size() == total_size);
}

void setup_square_grid(const split_t splits) {
  reset_api();
  capi.set_rng_seed(12345);
  std::vector<std::set<tileidx_t>> ownership_map(capi.get_num_processes());
  tileidx_t tix = 0;
  for (auto &owned_tiles : ownership_map)
    owned_tiles.insert(tix++);

  const auto factors = get_middle_factors(capi.get_num_processes());
  const space_t tile_width = 2. / factors.first;
  const space_t tile_height = 2. / factors.second;
  const space_t origin_x = -std::fma(-0.5, tile_width, 1.);
  const space_t origin_y = -std::fma(-0.5, tile_height, 1.);

  GPStruct gps;
  copy_to_array_from_collection(gps.grid_origin_,
                                std::vector<space_t>{origin_x, origin_y}, gc);
  copy_to_array_from_collection(
      gps.grid_dimensions_,
      std::vector<tileidx_t>{factors.first, factors.second}, gc);
  copy_to_charray_from_string(gps.tile_type_, "rectangle", gc);
  copy_to_array_from_collection(gps.tile_side_lengths_,
                                std::vector<space_t>{tile_width, tile_height},
                                gc);
  gps.compute_splits_ = false;
  gps.num_splits_ = splits;

  capi.generate_tile_grid(*nested_collection_to_nested_array(ownership_map, gc),
                          gps);
}

std::pair<std::size_t, std::size_t>
generate_fixed_positions(const nodeidx_t num_positions) {
  assert(0 < num_positions);
  std::vector<space_t> stride(2 * num_positions + 1);
  auto s_it = stride.begin();
  for (space_t s = -num_positions; s <= num_positions; ++s)
    *s_it++ = s / num_positions;

  const std::size_t total_size = stride.size() * stride.size() * 2;
  std::vector<space_t> anycoord_vec(total_size);

  std::size_t index = 0;
  for (const auto &x : stride) {
    for (const auto &y : stride) {
      anycoord_vec[index++] = x;
      anycoord_vec[index++] = y;
    }
  }
  assert(index == total_size);

  return std::make_pair(insert_nodes(anycoord_vec), anycoord_vec.size() / 2);
}

std::pair<std::size_t, std::size_t> inline generate_random_positions(
    const nodeidx_t num_nodes) {
  assert(0 < num_nodes);
  const auto nn = num_nodes * num_nodes;
  return std::make_pair(generate_nodes(nn), nn);
}

void stable_positions_test() {
  for (const auto &rand : {true, false}) {
    for (const auto &positions : {5, 10, 15}) {
      for (const auto &splits : {0, 1, 2, 3}) {
        setup_square_grid(splits);
        const auto idx_pos_pair = rand ? generate_random_positions(positions)
                                       : generate_fixed_positions(positions);
        communicate_all_nodes(*capi.view_node_positions({}, {}),
                              idx_pos_pair.second);
      }
    }
  }
}

std::pair<std::size_t, RemoteConnectionViewPair *>
generate_connections(const std::size_t dtns_idx,
                     const bool allow_self_connections,
                     const bool partition_by_source) {
  MPStruct mps;
  copy_to_charray_from_string(mps.mask_blueprint_name_, "circular", gc);
  copy_to_array_from_collection(mps.mask_blueprint_params_,
                                std::vector<space_t>{M_SQRT2}, gc);

  CPStruct cps;
  cps.edge_wrap_ = true;
  cps.only_neighborhood_ = false;
  cps.allow_self_connections_ = allow_self_connections;
  cps.partition_connections_ = partition_by_source;
  copy_to_charray_from_string(cps.rule_, "pairwise_bernoulli", gc);
  copy_to_charray_from_string(cps.prob_df_name_, "constant", gc);
  copy_to_charray_from_string(cps.weight_df_name_, "constant", gc);
  copy_to_charray_from_string(cps.delay_df_name_, "constant", gc);
  copy_to_array_from_collection(cps.prob_df_params_, std::vector<space_t>{1.0},
                                gc);
  copy_to_array_from_collection(cps.delay_df_params_, std::vector<space_t>{1.0},
                                gc);
  copy_to_array_from_collection(cps.weight_df_params_,
                                std::vector<space_t>{1.0}, gc);

  const auto conn_idx =
      capi.compute_spatial_connections(dtns_idx, dtns_idx, mps, cps).first;

  return std::make_pair(conn_idx, capi.view_spatial_connections(conn_idx));
}

void stable_connections_test() {
  bool success = true;
  const bool check_partitions = capi.get_num_processes() == 1;
  for (const auto &rand : {true, false}) {
    for (const auto &allow_sc : {true, false}) {
      for (const auto &partition : {false}) {
        for (const auto &positions : {5, 10}) {
          for (const auto &splits : {0, 2}) {
            setup_square_grid(splits);
            const auto idx_pos_pair = rand
                                          ? generate_random_positions(positions)
                                          : generate_fixed_positions(positions);
            const auto &incoming_connections =
                generate_connections(idx_pos_pair.first, allow_sc, partition)
                    .second->first_;

            std::unordered_map<combined_idx_t,
                               std::unordered_set<combined_idx_t>>
                conns;
            for (std::size_t rix = 0; rix < incoming_connections.size_; ++rix) {
              const auto &rcip = incoming_connections.array_[rix];
              if (check_partitions)
                assert(partition == (1 < rcip.second_.num_partitions_));
              for (std::size_t pix = 0; pix < rcip.second_.num_partitions_;
                   ++pix) {
                const auto partition_size = rcip.second_.partition_sizes_[pix];
                const auto &partition_sources = rcip.second_.sources_[pix];
                const auto &partition_targets = rcip.second_.targets_[pix];
                for (std::size_t cix = 0; cix < partition_size; ++cix) {
                  const auto source_gid =
                      get_global_index(partition_sources[cix], rcip.first_);
                  const auto target_gid =
                      get_global_index(partition_targets[cix]);
                  auto &set = conns[target_gid];

                  assert(set.find(source_gid) ==
                         set.end()); // Guaranteed by pairwise bernoulli
                  set.insert(source_gid);
                }
              }
            }

            const std::size_t expected_length =
                (idx_pos_pair.second - !allow_sc);
            for (const auto &source_targets : conns) {
              if (source_targets.second.size() != expected_length) {
                printf("RANK %i fail with target lengths: %li vs expected "
                       "lengths: %li conf: rand %i allow_sc %i positions %i "
                       "splits %i partition %i\n",
                       capi.get_rank(), source_targets.second.size(),
                       expected_length, rand, allow_sc, positions, splits,
                       partition);
                success = false;
              }
            }
          }
        }
      }
    }
  }
  assert(success);
}

int main(int argc, char **argv) {
  init_mpi(&argc, &argv);

  capi.set_rank(get_mpi_rank());
  capi.set_num_processes(get_num_mpi_processes());
  capi.set_num_threads(2);

  stable_positions_test();
  stable_connections_test();

  finalize_mpi();
  return 0;
}
