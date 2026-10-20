#include <chrono>
#include <cstdio>

#include "coordinate_geometry.h"
#include "creator_registry.h"
#include "thread_aligned_array.h"
#include "type_erasure_helpers.h"
#include "vp_interface.h"

#include "bounding_box.h"
#include "gf2d_geometry.h"
#include "gf_creators.h"
#include "grid_generation.h"
#include "mask2d_geometry.h"
#include "mask_collection.h"
#include "node_containers_init.h"
#include "node_creation.h"
#include "node_slicing.h"
#include "test_helpers.h"
#include "tile.h"

using namespace sapi;

void test_slicing() {
  const vp_t local_rank = get_mpi_rank();
  const tileidx_t num_processes =
      static_cast<tileidx_t>(get_num_mpi_processes());
  const nodeidx_t scale = 100000;

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

  const auto gc = construct_gf_collection<Coord2D>({0., 0.}, {1, 1}, "hexagon",
                                                   {1.}, {0}, gscr, socr, ctcr);
  gc_array.clone(gc);
  tg = generate_tile_grid(gc_array);

  std::vector<std::set<tileidx_t>> rank_tile_ownership_map(num_processes);
  for (tileidx_t rank = 0; rank < num_processes; ++rank)
    rank_tile_ownership_map[rank] = {0};
  gnb.set_tile_ownership(std::move(rank_tile_ownership_map), tg);
  split_tiles_in_grid(tg, *gnb.locally_owned_tiles_, 10);
  initialize_local_grid_node_collection(gnc, *gnb.locally_owned_tiles_, tg);

  const auto init_end = std::chrono::high_resolution_clock::now();
  const auto init_time = init_end - init_start;
  printf("Time grid manager init: %f\n", init_time.count() / 1e9);

  DistributedTiledNodeSequenceMap dist_map[2];

  const auto node_gen_start = std::chrono::high_resolution_clock::now();
  for (auto idx : {0, 1}) {
    const auto aggregated_node_counts = distribute_node_counts_in_grid(
        scale * num_processes, {}, tg, gnb, rman, DISTRIBUTION_MODE::BALANCED);

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
                           0) == scale * num_processes * (idx + 1));
    assert_equal_across_ranks(total_nodes_per_rank);
  }
  const auto node_gen_end = std::chrono::high_resolution_clock::now();
  const auto node_gen_time = node_gen_end - node_gen_start;
  printf("Time node gen: %f\n", node_gen_time.count() / 1e9);

  MaskCollection<Coord2D> mc;
  TAArray<MaskCollection<Coord2D>> mc_array;
  mc_array.clone(mc);

  const auto all_nodes_start = std::chrono::high_resolution_clock::now();

  auto tiled_nodes = slice_tiled_node_maps(nullptr, tg, gnb, gnc, mc_array);

  const auto all_nodes_end = std::chrono::high_resolution_clock::now();
  const auto all_nodes_time = all_nodes_end - all_nodes_start;
  printf("Time slice all nodes: %f\n", all_nodes_time.count() / 1e9);

  nodeidx_t node_count = 0;
  for (const auto &leaf_nodes : tiled_nodes)
    node_count += leaf_nodes.size();

  assert(0 <= node_count && total_nodes_per_rank[local_rank] == node_count);

  mc.source_mask_ =
      construct_mask<Coord2D>(MASK_SHAPE::CIRCULAR, {0, 0}, {1}, {});
  mc_array.clone(mc);

  const auto big_mask_start = std::chrono::high_resolution_clock::now();

  auto new_nodes = slice_tiled_node_maps(nullptr, tg, gnb, gnc, mc_array);

  const auto big_mask_end = std::chrono::high_resolution_clock::now();
  const auto big_mask_time = big_mask_end - big_mask_start;
  printf("Time slice big mask: %f\n", big_mask_time.count() / 1e9);

  assert(new_nodes == tiled_nodes);
  new_nodes.clear();
  tiled_nodes.clear();

  auto mask = construct_mask<Coord2D>(MASK_SHAPE::CIRCULAR, {0, 0}, {0.5}, {});
  mc.source_mask_ = mask;
  mc_array.clone(mc);

  const auto small_mask_start = std::chrono::high_resolution_clock::now();

  tiled_nodes = slice_tiled_node_maps(nullptr, tg, gnb, gnc, mc_array);

  const auto small_mask_end = std::chrono::high_resolution_clock::now();
  const auto small_mask_time = small_mask_end - small_mask_start;
  printf("Time slice small mask: %f\n", small_mask_time.count() / 1e9);

  nodeidx_t new_node_count = tiled_nodes.size();
  for (const auto &leaf_nodes : tiled_nodes)
    for (const auto &nc_view : leaf_nodes)
      assert(mask.coord_in_mask(nc_view->second).first);
  assert(new_node_count < node_count);

  mc.source_mask_ = Mask<Coord2D>();
  mc_array.clone(mc);

  const auto dtns_slice_start = std::chrono::high_resolution_clock::now();

  tiled_nodes = slice_tiled_node_maps(&dist_map[0], tg, gnb, gnc, mc_array);

  const auto dtns_slice_end = std::chrono::high_resolution_clock::now();
  const auto dtns_slice_time = dtns_slice_end - dtns_slice_start;
  printf("Time slice dtns: %f\n", dtns_slice_time.count() / 1e9);

  if (const auto rank_search = dist_map[0].find(local_rank);
      rank_search != dist_map[0].end()) {
    for (const auto &leaf_nodes : tiled_nodes) {
      for (const auto &nc_view : leaf_nodes) {
        std::size_t found = 0;
        for (const auto &[tile_index, ns] : rank_search->second)
          found += (ns.first <= nc_view->first &&
                    nc_view->first - ns.first <= ns.second);
        assert(found == 1);
      }
    }
  }

  mask = construct_mask<Coord2D>(MASK_SHAPE::CIRCULAR, {0.5, 0.5}, {1}, {});
  mc.source_mask_ = mask;
  mc_array.clone(mc);

  const auto dtns_and_mask_slice_start =
      std::chrono::high_resolution_clock::now();

  tiled_nodes = slice_tiled_node_maps(&dist_map[1], tg, gnb, gnc, mc_array);

  const auto dtns_and_mask_slice_end =
      std::chrono::high_resolution_clock::now();
  const auto dtns_and_mask_slice_time =
      dtns_and_mask_slice_end - dtns_and_mask_slice_start;
  printf("Time slice dtns and mask: %f\n",
         dtns_and_mask_slice_time.count() / 1e9);

  if (const auto rank_search = dist_map[1].find(local_rank);
      rank_search != dist_map[1].end()) {
    for (const auto &leaf_nodes : tiled_nodes) {
      for (const auto &nc_view : leaf_nodes) {
        std::size_t found = 0;
        for (const auto &[tile_index, ns] : rank_search->second) {
          found += (ns.first <= nc_view->first &&
                    nc_view->first - ns.first <= ns.second);
          assert(mask.coord_in_mask(nc_view->second).first);
        }
        assert(found == 1);
      }
    }
  }
}

int main(int argc, char **argv) {
  init_mpi(&argc, &argv);
  init_omp(5);

  test_slicing();

  finalize_mpi();
  return 0;
}
