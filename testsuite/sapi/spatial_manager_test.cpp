#include <cstdio>

#include "spatial_manager.h"
#include "test_helpers.h"

using namespace sapi;

std::size_t generate_nodes(SpatialManager<Coord2D> &spam,
                           std::vector<nodeidx_t> &total_nodes_per_rank,
                           const tileidx_t num_procs,
                           const std::optional<std::set<tileidx_t>> &tiles) {
  const auto node_counts_per_rank =
      spam.distribute_nodes_in_grid(100, tiles, DISTRIBUTION_MODE::SQUEEZED);

  RankNodeSequenceMap rns_map;
  for (tileidx_t rank = 0; rank < num_procs; ++rank) {
    const auto num_nodes = node_counts_per_rank[rank];
    if (num_nodes < 1)
      continue;

    const auto next_index = total_nodes_per_rank[rank];
    assert_emplace(rns_map.emplace(
        std::make_pair(vp_t(rank), NodeSequence(next_index, num_nodes))));
    total_nodes_per_rank[rank] += num_nodes;
  }
  assert_equal_across_ranks(total_nodes_per_rank);

  return spam.generate_nodes_in_tiles(rns_map, DISTRIBUTION_MODE::FREE);
}

std::size_t insert_nodes(SpatialManager<Coord2D> &spam,
                         std::vector<nodeidx_t> &total_nodes_per_rank,
                         const tileidx_t num_procs,
                         const std::vector<space_t> &anycoord_vec) {
  const auto [node_counts_per_rank, leftovers] = spam.insert_positions_in_grid(
      2, anycoord_vec.size(), anycoord_vec.data());

  assert(!leftovers.empty());

  RankNodeSequenceMap rns_map;
  for (tileidx_t rank = 0; rank < num_procs; ++rank) {
    const auto num_nodes = node_counts_per_rank[rank];
    if (num_nodes < 1)
      continue;

    const auto next_index = total_nodes_per_rank[rank];
    assert_emplace(rns_map.emplace(
        std::make_pair(vp_t(rank), NodeSequence(next_index, num_nodes))));
    total_nodes_per_rank[rank] += num_nodes;
  }
  assert_equal_across_ranks(total_nodes_per_rank);

  return spam.insert_positions_in_tiles(rns_map);
}

int main(int argc, char **argv) {
  init_mpi(&argc, &argv);
  init_omp(4);

  const auto local_rank = get_mpi_rank();
  const auto num_procs = static_cast<tileidx_t>(get_num_mpi_processes());
  std::vector<nodeidx_t> total_nodes_per_rank(num_procs, 0);
  std::vector<std::set<tileidx_t>> rank_ownership(num_procs);
  for (tileidx_t rank = 0; rank < num_procs; ++rank)
    rank_ownership[rank] = {rank, num_procs + rank};

  SpatialManager<Coord2D> spam;
  spam.update_num_threads();
  spam.set_rng_seed(12345u);

  GridParameters gps;
  gps.grid_origin_ = {0, 0};
  gps.grid_dimensions_ = {1, 2 * num_procs};
  gps.tile_type_ = "hexagon";
  gps.tile_side_lengths_ = {1};

  spam.initialize_tile_grid(std::move(rank_ownership), gps);

  const auto dist_tns_idx0 =
      generate_nodes(spam, total_nodes_per_rank, num_procs,
                     std::optional<std::set<tileidx_t>>());

  std::set<tileidx_t> pair_tiles;
  for (tileidx_t rank = 0; rank < 2 * num_procs; rank += 2)
    pair_tiles.insert(rank);
  assert(!pair_tiles.empty());

  const auto dist_tns_idx1 =
      generate_nodes(spam, total_nodes_per_rank, num_procs,
                     std::optional<std::set<tileidx_t>>(std::move(pair_tiles)));

  MaskParameters mp;

  mp.mask_blueprint_name_ = "circular";
  mp.mask_blueprint_params_ = {0.25};

  ConnectionParameters cp;

  cp.rule_ = "pairwise_bernoulli";
  cp.edge_wrap_ = true;
  cp.allow_multiplicity_ = false;
  cp.connection_counts_ = 10;
  cp.delay_df_name_ = "distance";
  cp.weight_df_name_ = "distance";
  cp.weight_ufs_names_ = {"inverse", "max", "min"};
  cp.weight_ufs_params_ = {{}, {0}, {10}};
  cp.prob_df_name_ = "constant";
  cp.prob_df_params_ = {0.5};

  std::vector<std::size_t> rci_vec(4);
  auto rci_it = rci_vec.begin();

  for (const auto &source : {dist_tns_idx0, dist_tns_idx1})
    for (const auto &target : {dist_tns_idx0, dist_tns_idx1})
      *rci_it++ =
          spam.compute_spatial_connections(source, target, mp, cp).first;

  mp.mask_blueprint_name_.clear();
  mp.mask_blueprint_params_.clear();
  mp.source_mask_name_ = "circular";
  mp.source_mask_origin_ = {0, 0};
  mp.source_mask_params_ = {0.25};

  spam.get_distributed_node_sequence(dist_tns_idx0);
  spam.get_distributed_node_sequence(dist_tns_idx1);

  spam.get_nodes({dist_tns_idx0}, mp);

  mp.source_mask_name_ = "parallelogram";
  mp.source_mask_origin_ = {0, 0};
  mp.source_mask_params_ = {0, 1, 0.25, 0.75};

  spam.get_nodes({dist_tns_idx1}, mp);

  spam.get_grid_vertices();
  spam.get_nodes({}, MaskParameters());

  std::vector<space_t> positions{-0.7, 0.7,  0.7,  -0.7, 0.7,
                                 0.7,  -0.7, -0.7, 1,    1};
  insert_nodes(spam, total_nodes_per_rank, num_procs, positions);

  for (const auto &rci_index : rci_vec) {
    const auto cc = spam.get_connection_counts(rci_index);
    const auto &rci = spam.get_connection_map(rci_index);

    assert(cc.incoming_ranks_ == rci.incoming_connections_.size());
    assert(cc.outgoing_ranks_ == rci.outgoing_connections_.size());

    for (std::size_t index = 0; index < cc.source_ranks_.size(); ++index) {
      assert(rci.incoming_connections_.at(cc.source_ranks_[index])
                 .total_generated_connections_ ==
             cc.incoming_counts_.at(index));
    }

    for (std::size_t index = 0; index < cc.target_ranks_.size(); ++index) {
      assert(rci.outgoing_connections_.at(cc.target_ranks_[index])
                 .total_generated_connections_ ==
             cc.outgoing_counts_.at(index));
    }
  }

  printf("RANK %i:\n%s", local_rank,
         spam.get_timer_data().to_string(1).c_str());

  finalize_mpi();
  return 0;
}
