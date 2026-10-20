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

void repeatable_construction(
    const rng_seed_t seed, const tileidx_t grid_width,
    const tileidx_t grid_height, const nodeidx_t num_nodes,
    const split_t splits, const space_t mask_radius, const space_t conn_prob,
    const tileidx_t num_processes,
    std::forward_list<TileGrid<Coord2D>> &tile_grid_history,
    std::forward_list<DistributedTiledNodeSequenceMap> &rank_allocation_history,
    std::forward_list<std::unordered_map<
        tileidx_t, std::unordered_map<tileidx_t, NodeSequence>>>
        &tile_ns_allocation_history,
    std::forward_list<std::unordered_map<nodeidx_t, Coord2D>>
        &coord_allocation_history,
    std::forward_list<DistributedConnectionInfo> &rci_allocation_history) {
  TimerManager tman;
  tman.initialize();
  RandomManager rman;
  rman.set_rng_seed(seed);

  MaskCollection<Coord2D> mc;
  mc.blueprint_ =
      construct_mask<Coord2D>(MASK_SHAPE::CIRCULAR, {}, {mask_radius}, {});
  TAArray<MaskCollection<Coord2D>> mc_array;
  mc_array.clone(mc);

  ConnectionGenerator cg;
  cg.rule_ = CONNECTION_RULE::PAIRWISE_BERNOULLI;
  cg.numeric_functors_.delay_functor_.df_.func_ =
      DISPLACEMENT_FUNCTION::DISPLACEMENT_X;
  cg.numeric_functors_.weight_functor_.df_.func_ =
      DISPLACEMENT_FUNCTION::DISTANCE_Y;
  cg.numeric_functors_.probability_functor_.df_.func_ =
      DISPLACEMENT_FUNCTION::CONSTANT;
  cg.numeric_functors_.probability_functor_.df_.var_ = conn_prob;

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

  const auto gc =
      construct_gf_collection<Coord2D>({0., 0.}, {grid_width, grid_height},
                                       "triangle", {1.}, {0}, gscr, socr, ctcr);
  gc_array.clone(gc);
  tg = generate_tile_grid(gc_array);

  std::vector<std::set<tileidx_t>> rank_tile_ownership_map(num_processes);
  for (tileidx_t rank = 0; rank < num_processes; ++rank) {
    std::set<tileidx_t> tix_set;
    for (tileidx_t tix = 0; tix < grid_width * grid_height; ++tix)
      tix_set.insert(tix);
    rank_tile_ownership_map[rank] = std::move(tix_set);
  }
  gnb.set_tile_ownership(std::move(rank_tile_ownership_map), tg);
  split_tiles_in_grid(tg, *gnb.locally_owned_tiles_, splits);
  initialize_local_grid_node_collection(gnc, *gnb.locally_owned_tiles_, tg);

  const auto aggregated_node_counts = distribute_node_counts_in_grid(
      num_nodes, {}, tg, gnb, rman, DISTRIBUTION_MODE::FREE);

  RankNodeSequenceMap rns_map;
  NodeCountVector total_nodes_per_rank(num_processes, 0);
  for (tileidx_t rank = 0; rank < num_processes; ++rank) {
    const auto num_nodes = aggregated_node_counts.first[rank];
    if (num_nodes < 1)
      continue;

    const auto next_index = total_nodes_per_rank[rank];
    assert_emplace(rns_map.emplace(
        std::make_pair(vp_t(rank), NodeSequence(next_index, num_nodes))));
    total_nodes_per_rank[rank] += num_nodes;
  }

  auto dist_tns = consolidate_node_sequences_per_tile_per_rank(
      rns_map, aggregated_node_counts.second);

  if (const auto local_tns = dist_tns.find(gnb.local_rank_);
      local_tns != dist_tns.end())
    distribute_node_counts_in_tiles(gnc, local_tns->second, tg, rman,
                                    DISTRIBUTION_MODE::FREE);

  assert(std::accumulate(total_nodes_per_rank.cbegin(),
                         total_nodes_per_rank.cend(), 0) == num_nodes);
  assert_equal_across_ranks(total_nodes_per_rank);

  std::unordered_map<tileidx_t, std::unordered_map<tileidx_t, NodeSequence>>
      tile_ns_allocations;
  std::unordered_map<nodeidx_t, Coord2D> coord_allocations;

  for (const auto &tix : *gnb.locally_owned_tiles_) {
    std::unordered_map<tileidx_t, NodeSequence> leaf_ns_allocations;
    tileidx_t st_index = 0;
    for (const auto &leaf_coord_vec : gnc.at(tix)) {
      if (!leaf_coord_vec.empty()) {
        assert(leaf_coord_vec.size() == 1);
        auto &node_coord_vec = leaf_coord_vec[0];
        assert_emplace(leaf_ns_allocations.emplace(
            std::make_pair(st_index, NodeSequence(node_coord_vec[0].first,
                                                  node_coord_vec.size()))));
        for (const auto &[index, coord] : node_coord_vec) {
          assert_emplace(
              coord_allocations.emplace(std::make_pair(index, coord)));
        }
      }
      ++st_index;
    }
    if (!leaf_ns_allocations.empty())
      assert_emplace(tile_ns_allocations.emplace(
          tileidx_t(tix), std::move(leaf_ns_allocations)));
  }

  tile_ns_allocation_history.emplace_front(std::move(tile_ns_allocations));
  coord_allocation_history.emplace_front(std::move(coord_allocations));

  rci_allocation_history.emplace_front(compute_distributed_spatial_connections(
      dist_tns, dist_tns, tg, gnb, gnc, mc_array, cgarr, rman, tman, true,
      true));

  rank_allocation_history.emplace_front(std::move(dist_tns));

  tile_grid_history.emplace_front(std::move(tg));
}

void repeatability_test(
    const rng_seed_t seed, const tileidx_t grid_width,
    const tileidx_t grid_height, const nodeidx_t num_nodes,
    const split_t splits, const space_t mask_radius, const space_t conn_prob,
    const tileidx_t num_processes,
    std::forward_list<TileGrid<Coord2D>> &tile_grid_history,
    std::forward_list<DistributedTiledNodeSequenceMap> &rank_allocation_history,
    std::forward_list<std::unordered_map<
        tileidx_t, std::unordered_map<tileidx_t, NodeSequence>>>
        &tile_ns_allocation_history,
    std::forward_list<std::unordered_map<nodeidx_t, Coord2D>>
        &coord_allocation_history,
    std::forward_list<DistributedConnectionInfo> &rci_allocation_history) {
  printf("Running iterations with seed %i\n", seed);

  for (int i = 0; i < 10; ++i) {
    const auto now = std::chrono::high_resolution_clock::now();
    repeatable_construction(seed, grid_width, grid_height, num_nodes, splits,
                            mask_radius, conn_prob, num_processes,
                            tile_grid_history, rank_allocation_history,
                            tile_ns_allocation_history,
                            coord_allocation_history, rci_allocation_history);
    printf("Iteration %i time %f\n", i,
           (std::chrono::high_resolution_clock::now() - now).count() / 1e9);
  }

  const auto first_tga = tile_grid_history.begin();
  for (auto tgh_it = tile_grid_history.begin();
       tgh_it != tile_grid_history.end(); ++tgh_it)
    assert(*first_tga == *tgh_it);

  const auto first_ra = rank_allocation_history.begin();
  for (auto rah_it = rank_allocation_history.begin();
       rah_it != rank_allocation_history.end(); ++rah_it)
    assert(*first_ra == *rah_it);

  const auto first_ta = tile_ns_allocation_history.begin();
  for (auto tah_it = tile_ns_allocation_history.begin();
       tah_it != tile_ns_allocation_history.end(); ++tah_it)
    assert(*first_ta == *tah_it);

  const auto first_ca = coord_allocation_history.begin();
  for (auto cah_it = coord_allocation_history.begin();
       cah_it != coord_allocation_history.end(); ++cah_it)
    assert(*first_ca == *cah_it);

  const auto first_rca = rci_allocation_history.begin();
  for (auto rch_it = rci_allocation_history.begin();
       rch_it != rci_allocation_history.end(); ++rch_it)
    assert(*first_rca == *rch_it);
}

int main(int argc, char **argv) {
  init_mpi(&argc, &argv);
  init_omp(4);

  const uint32_t seeds[2] = {12345, 12346};
  const tileidx_t grid_width = 2;
  const tileidx_t grid_height = 1;
  const nodeidx_t num_nodes = 100;
  const split_t splits = 2;
  const space_t mask_radius = 0.5;
  const space_t conn_prob = 0.5;

  const tileidx_t num_processes =
      static_cast<tileidx_t>(get_num_mpi_processes());
  std::forward_list<TileGrid<Coord2D>> tile_grid_history[2];
  std::forward_list<DistributedTiledNodeSequenceMap> rank_allocation_history[2];
  std::forward_list<std::unordered_map<
      tileidx_t, std::unordered_map<tileidx_t, NodeSequence>>>
      tile_ns_allocation_history[2];
  std::forward_list<std::unordered_map<nodeidx_t, Coord2D>>
      coord_allocation_history[2];
  std::forward_list<DistributedConnectionInfo> rci_allocation_history[2];

  repeatability_test(seeds[0], grid_width, grid_height, num_nodes, splits,
                     conn_prob, mask_radius, num_processes,
                     tile_grid_history[0], rank_allocation_history[0],
                     tile_ns_allocation_history[0], coord_allocation_history[0],
                     rci_allocation_history[0]);

  repeatability_test(seeds[1], grid_width, grid_height, num_nodes, splits,
                     conn_prob, mask_radius, num_processes,
                     tile_grid_history[1], rank_allocation_history[1],
                     tile_ns_allocation_history[1], coord_allocation_history[1],
                     rci_allocation_history[1]);

  assert(*tile_grid_history[0].begin() == *tile_grid_history[1].begin());

  if (num_processes > 1)
    assert(!(*rank_allocation_history[0].begin() ==
             *rank_allocation_history[1].begin()));

  assert(!(*tile_ns_allocation_history[0].begin() ==
           *tile_ns_allocation_history[1].begin()));
  assert(!(*coord_allocation_history[0].begin() ==
           *coord_allocation_history[1].begin()));
  assert(!(*rci_allocation_history[0].begin() ==
           *rci_allocation_history[1].begin()));

  finalize_mpi();
  return 0;
}
