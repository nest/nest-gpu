#include <cstdio>

#include "coordinate_geometry.h"
#include "creator_registry.h"
#include "type_erasure_helpers.h"
#include "vp_interface.h"

#include "gf2d_geometry.h"
#include "gf_creators.h"
#include "grid_generation.h"
#include "node_containers_init.h"
#include "node_insertion.h"
#include "test_helpers.h"
#include "tile2d_geometry.h"

using namespace sapi;

void print_dtns(const DistributedTiledNodeSequenceMap &dtns) {
  std::string msg = "";
  for (const auto &[rank, tns] : dtns) {
    std::string rank_msg = "R" + std::to_string(rank) + ": ( ";
    for (const auto &[tile, ns] : tns)
      rank_msg += "T" + std::to_string(tile) + ": ( " +
                  std::to_string(ns.first) + " -> " +
                  std::to_string(ns.second) + " ), ";
    msg += rank_msg + " )\n";
  }
  printf("%s", msg.c_str());
}

void test_insertion() {
  RandomManager rman;
  rman.set_rng_seed(12345u);
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
      {0., 0.}, {1, 1}, "rectangle", {M_SQRT2}, {0}, gscr, socr, ctcr);
  gc_array.clone(gc);
  tg = generate_tile_grid(gc_array);

  std::vector<std::set<tileidx_t>> rank_tile_ownership_map(gnb.num_processes_);
  for (tileidx_t rank = 0; rank < gnb.num_processes_; ++rank)
    rank_tile_ownership_map[rank] = {0};
  gnb.set_tile_ownership(std::move(rank_tile_ownership_map), tg);
  split_tiles_in_grid(tg, *gnb.locally_owned_tiles_, 5);
  initialize_local_grid_node_collection(gnc, *gnb.locally_owned_tiles_, tg);

  NodeCountVector total_nodes_per_rank(gnb.num_processes_, 0);

  std::deque<Coord2D> insertion_coords{
      construct_coord_2D(-0.7, 0.7), construct_coord_2D(0.7, -0.7),
      construct_coord_2D(0.7, 0.7), construct_coord_2D(-0.7, -0.7),
      construct_coord_2D(1, 1)};

  auto [node_counts_per_rank, tiled_node_counts_per_rank,
        local_tiled_coord_map] =
      insert_node_positions_in_grid(insertion_coords, tg, gnb, rman);

  assert(insertion_coords.size() == 1 &&
         insertion_coords.front() == construct_coord_2D(1, 1));

  RankNodeSequenceMap rns_map;
  for (tileidx_t rank = 0; rank < gnb.num_processes_; ++rank) {
    const auto num_nodes = node_counts_per_rank[rank];
    if (num_nodes < 1)
      continue;

    const auto next_index = total_nodes_per_rank[rank];
    assert_emplace(rns_map.emplace(
        std::make_pair(vp_t(rank), NodeSequence(next_index, num_nodes))));
    total_nodes_per_rank[rank] += num_nodes;
  }

  const auto dist_tns = consolidate_node_sequences_per_tile_per_rank(
      rns_map, tiled_node_counts_per_rank);

  print_dtns(dist_tns);

  if (const auto tns = dist_tns.find(gnb.local_rank_); tns != dist_tns.end())
    insert_node_positions_in_tiles(local_tiled_coord_map, gnc, tns->second, tg);
}

int main(int argc, char **argv) {
  init_mpi(&argc, &argv);
  init_omp(4);

  test_insertion();

  finalize_mpi();
  return 0;
}
