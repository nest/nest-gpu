#include <chrono>
#include <cstdio>

#include "coordinate_geometry.h"
#include "vp_interface.h"

#include "bounding_box.h"
#include "connection_rules.h"
#include "gf2d_geometry.h"
#include "gf_creators.h"
#include "grid_generation.h"
#include "mask2d_geometry.h"
#include "mask_collection.h"
#include "mask_containers.h"
#include "mask_node_processing.h"
#include "mask_tile_processing.h"
#include "node_containers_init.h"
#include "node_creation.h"
#include "test_helpers.h"

using namespace sapi;

void test_local_node_processing() {
  auto base = get_mem_usage();
  printf("Peak mem before start: %li\n", base);

  const vp_t local_rank = get_mpi_rank();
  const tileidx_t num_processes =
      static_cast<tileidx_t>(get_num_mpi_processes());
  const nodeidx_t scale = 10;
  const split_t splits = 4;

  NodeCountVector total_nodes_per_rank(num_processes, 0);
  RandomManager rman;
  rman.set_rng_seed(12345u);

  const auto init_start = std::chrono::high_resolution_clock::now();

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
      {0., 0.}, {1, 1}, "rectangle", {2.}, {0}, gscr, socr, ctcr);
  gc_array.clone(gc);
  tg = generate_tile_grid(gc_array);

  std::vector<std::set<tileidx_t>> rank_tile_ownership_map(num_processes);
  for (tileidx_t rank = 0; rank < num_processes; ++rank)
    rank_tile_ownership_map[rank] = {0};
  gnb.set_tile_ownership(std::move(rank_tile_ownership_map), tg);
  split_tiles_in_grid(tg, *gnb.locally_owned_tiles_, splits);
  initialize_local_grid_node_collection(gnc, *gnb.locally_owned_tiles_, tg);
  const tileidx_t num_leaves = tg.positions_.at(0).tile_.leaf_tiles_.size();

  const auto init_end = std::chrono::high_resolution_clock::now();
  const auto init_time = init_end - init_start;
  printf("Time grid manager init: %f\n", init_time.count() / 1e9);

  auto new_base = get_mem_usage();
  printf("Peak mem after grid init: %li - gain: %li\n", new_base,
         new_base - base);

  DistributedTiledNodeSequenceMap dist_map[2];

  const auto node_gen_start = std::chrono::high_resolution_clock::now();

  for (const auto &dist_idx : {0, 1}) {
    const auto aggregated_node_counts = distribute_node_counts_in_grid(
        scale * num_leaves, {}, tg, gnb, rman, DISTRIBUTION_MODE::BALANCED);

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

    dist_map[dist_idx] = consolidate_node_sequences_per_tile_per_rank(
        rns_map, aggregated_node_counts.second);

    if (const auto local_tns = dist_map[dist_idx].find(local_rank);
        local_tns != dist_map[dist_idx].end())
      distribute_node_counts_in_tiles(gnc, local_tns->second, tg, rman,
                                      DISTRIBUTION_MODE::BALANCED);
    assert(std::accumulate(total_nodes_per_rank.cbegin(),
                           total_nodes_per_rank.cend(),
                           0) == scale * num_leaves * (dist_idx + 1));
    assert_equal_across_ranks(total_nodes_per_rank);
  }

  const auto node_gen_end = std::chrono::high_resolution_clock::now();
  const auto node_gen_time = node_gen_end - node_gen_start;
  printf("Time node gen: %f\n", node_gen_time.count() / 1e9);

  new_base = get_mem_usage();
  printf("Peak mem after node gen: %li - gain: %li\n", new_base,
         new_base - base);

  MaskCollection<Coord2D> mc;
  mc.blueprint_ = construct_mask<Coord2D>(MASK_SHAPE::CIRCULAR, {}, {1.}, {});
  TAArray<MaskCollection<Coord2D>> mc_array;
  mc_array.clone(mc);

  ConnectionGenerator cg_pb;
  cg_pb.rule_ = CONNECTION_RULE::PAIRWISE_BERNOULLI;
  cg_pb.numeric_functors_.delay_functor_.df_.func_ =
      DISPLACEMENT_FUNCTION::DISPLACEMENT_X;
  cg_pb.numeric_functors_.weight_functor_.df_.func_ =
      DISPLACEMENT_FUNCTION::DISTANCE_Y;
  cg_pb.numeric_functors_.probability_functor_.df_.func_ =
      DISPLACEMENT_FUNCTION::DISTANCE;
  TAArray<ConnectionGenerator> cg_array_pb;
  cg_array_pb.clone(cg_pb);

  ConnectionGenerator cg_fid;
  cg_fid.rule_ = CONNECTION_RULE::FIXED_IN_DEGREE;
  cg_fid.connection_counts_ = scale * num_leaves * 2 / 10;
  cg_fid.numeric_functors_ = cg_pb.numeric_functors_;
  TAArray<ConnectionGenerator> cg_array_fid;
  cg_array_fid.clone(cg_fid);

  new_base = get_mem_usage();
  printf("Peak mem after TAArray prep: %li - gain: %li\n", new_base,
         new_base - base);

  DistributedPairInfo<Coord2D> dpi[2];

  const auto dto_start = std::chrono::high_resolution_clock::now();

#pragma omp parallel default(shared)
#pragma omp master
  for (const auto &dist_idx : {0, 1}) {
    const auto res = compute_distributed_tile_overlap<Coord2D, true, true>(
                         dpi[dist_idx], dist_map[dist_idx],
                         dist_map[(dist_idx + 1) % 2], tg, gnb, gnc, mc_array)
                         .second;
    assert(res);
  }

  const auto dto_end = std::chrono::high_resolution_clock::now();
  const auto dto_time = dto_end - dto_start;
  printf("Time spatial overlap: %f\n", dto_time.count() / 1e9);

  new_base = get_mem_usage();
  printf("Peak mem after tile processing: %li - gain: %li\n", new_base,
         new_base - base);

  for (const auto &dist_idx : {0, 1}) {

    const auto ssi_it = dpi[dist_idx].source_side_info_.find(local_rank);
    const auto tsi_it = dpi[dist_idx].target_side_info_.find(local_rank);

    assert(ssi_it != dpi[dist_idx].source_side_info_.end() &&
           tsi_it != dpi[dist_idx].target_side_info_.end());

    assert(!ssi_it->second.tile_pairs_set_.tile_pairings_.empty() &&
           !ssi_it->second.sender_info_.filtered_coords_.empty() &&
           ssi_it->second.receiver_info_.filtered_coords_.empty() &&
           !tsi_it->second.tile_pairs_set_.tile_pairings_.empty() &&
           !tsi_it->second.sender_info_.filtered_coords_.empty() &&
           tsi_it->second.receiver_info_.filtered_coords_.empty());

    assert(ssi_it->second.sender_info_.filtered_coords_ !=
           tsi_it->second.sender_info_.filtered_coords_);

    // Get mirrored pairings
    ssi_it->second.receiver_info_.filtered_coords_ =
        tsi_it->second.sender_info_.filtered_coords_;
    tsi_it->second.receiver_info_.filtered_coords_ =
        ssi_it->second.sender_info_.filtered_coords_;
  }

  const auto ssi_0 = dpi[0].source_side_info_.find(local_rank);
  const auto ssi_1 = dpi[1].source_side_info_.find(local_rank);
  const auto tsi_0 = dpi[0].target_side_info_.find(local_rank);
  const auto tsi_1 = dpi[1].target_side_info_.find(local_rank);

  assert(ssi_0->second.sender_info_.filtered_coords_ ==
             tsi_1->second.sender_info_.filtered_coords_ &&
         ssi_1->second.sender_info_.filtered_coords_ ==
             tsi_0->second.sender_info_.filtered_coords_);

  RankConnectionInfo sorted_by_driver_source_first;
  RankConnectionInfo sorted_by_driver_target_first;
  RankConnectionInfo sorted_by_pool_source_first;
  RankConnectionInfo sorted_by_pool_target_first;

  const auto dc_start = std::chrono::high_resolution_clock::now();

#pragma omp parallel default(shared)
#pragma omp master
#pragma omp taskgroup
  {
#pragma omp task default(shared)
    compute_displacement_checks_across_tile_pairs<Coord2D, false>(
        sorted_by_driver_source_first, ssi_0->second, mc_array, cg_array_pb,
        rman, 0);
#pragma omp task default(shared)
    compute_displacement_checks_across_tile_pairs<Coord2D, true>(
        sorted_by_driver_target_first, tsi_0->second, mc_array, cg_array_pb,
        rman, 0);
#pragma omp task default(shared)
    compute_displacement_checks_across_tile_pairs<Coord2D, false>(
        sorted_by_pool_source_first, ssi_1->second, mc_array, cg_array_fid,
        rman, 0);
#pragma omp task default(shared)
    compute_displacement_checks_across_tile_pairs<Coord2D, true>(
        sorted_by_pool_target_first, tsi_1->second, mc_array, cg_array_fid,
        rman, 0);
  }

  const auto dc_end = std::chrono::high_resolution_clock::now();
  const auto dc_time = dc_end - dc_start;
  printf("Time displacement check: %f\n", dc_time.count() / 1e9);

  new_base = get_mem_usage();
  printf("Peak mem after node processing: %li - gain: %li\nStarting procedural "
         "check\n",
         new_base, new_base - base);

  const auto total_procedural0 =
      sorted_by_driver_source_first.procedural_connections_.size();
  const auto total_procedural1 =
      sorted_by_pool_source_first.procedural_connections_.size();

  assert(total_procedural0 ==
             sorted_by_driver_target_first.procedural_connections_.size() &&
         total_procedural1 ==
             sorted_by_pool_target_first.procedural_connections_.size());

  std::size_t matching0 = 0;
  for (std::size_t i = 0; i < total_procedural0; ++i) {
    std::size_t matches = 0;
    const auto &base = sorted_by_driver_source_first.procedural_connections_[i];
    for (std::size_t j = 0; j < total_procedural0; ++j)
      matches +=
          base == sorted_by_driver_target_first.procedural_connections_[j];
    assert(matches == 1);
    ++matching0;
  }
  assert(matching0 == total_procedural0);

  std::size_t matching1 = 0;
  for (std::size_t i = 0; i < total_procedural1; ++i) {
    std::size_t matches = 0;
    const auto &base = sorted_by_pool_source_first.procedural_connections_[i];
    for (std::size_t j = 0; j < total_procedural1; ++j)
      matches += base == sorted_by_pool_target_first.procedural_connections_[j];
    assert(matches == 1);
    ++matching1;
  }
  assert(matching1 == total_procedural1);

  printf("Done procedural check\n");

  const auto cons_start = std::chrono::high_resolution_clock::now();

#pragma omp parallel default(shared)
#pragma omp master
#pragma omp taskgroup
  {
#pragma omp task default(shared)
    sorted_by_driver_source_first.consolidate_connection_map();
#pragma omp task default(shared)
    sorted_by_driver_target_first.consolidate_connection_map();
#pragma omp task default(shared)
    sorted_by_pool_source_first.consolidate_connection_map();
#pragma omp task default(shared)
    sorted_by_pool_target_first.consolidate_connection_map();
  }

  const auto cons_end = std::chrono::high_resolution_clock::now();
  const auto cons_time = cons_end - cons_start;
  printf("Time consolidate connections: %f\n", cons_time.count() / 1e9);

  new_base = get_mem_usage();
  printf("Peak mem after node processing: %li - gain: %li\nStarting "
         "consolidated check\n",
         new_base, new_base - base);

  assert(sorted_by_pool_source_first == sorted_by_pool_target_first);
  assert(sorted_by_driver_source_first == sorted_by_driver_target_first);
}

int main(int argc, char **argv) {
  init_mpi(&argc, &argv);
  init_omp();

  test_local_node_processing();

  finalize_mpi();
  return 0;
}
