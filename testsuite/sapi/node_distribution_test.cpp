#include <cstdio>

#include "coordinate_geometry.h"
#include "vp_interface.h"

#include "bounding_box.h"
#include "gf2d_geometry.h"
#include "gf_creators.h"
#include "grid_generation.h"
#include "node_containers_init.h"
#include "node_creation.h"
#include "test_helpers.h"

using namespace sapi;

int main(int argc, char **argv) {
  init_mpi(&argc, &argv);
  init_omp(5);

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

  std::string tile_type = "rectangle";
  split_t splits = 3;
  nodeidx_t nodes = 100;
  if (argc > 1) {
    std::string args = "";
    for (int i = 1; i < argc; ++i) {
      args += argv[i];
      args += " | ";
    }
    printf("Args %i: %s\n", argc - 1, args.c_str());
    tile_type = std::string(argv[1]);
    if (argc > 2)
      splits = static_cast<split_t>(std::atoi(argv[2]));
    if (argc > 3)
      nodes = static_cast<split_t>(std::atoi(argv[3]));
  }
  printf("Testing %s tile with %i splits and %i nodes\n", tile_type.c_str(),
         splits, nodes);

  const auto gc = construct_gf_collection<Coord2D>({0., 0.}, {1, 1}, tile_type,
                                                   {1.}, {0}, gscr, socr, ctcr);
  gc_array.clone(gc);
  tg = generate_tile_grid(gc_array);

  std::vector<std::set<tileidx_t>> rank_tile_ownership_map(gnb.num_processes_);
  for (tileidx_t rank = 0; rank < gnb.num_processes_; ++rank)
    rank_tile_ownership_map[rank] = {0};
  gnb.set_tile_ownership(std::move(rank_tile_ownership_map), tg);
  split_tiles_in_grid(tg, *gnb.locally_owned_tiles_, splits);
  initialize_local_grid_node_collection(gnc, *gnb.locally_owned_tiles_, tg);

  NodeCountVector total_nodes_per_rank(gnb.num_processes_, 0);
  DistributedTiledNodeSequenceMap dist_map;

  const auto aggregated_node_counts = distribute_node_counts_in_grid(
      nodes, {}, tg, gnb, rman, DISTRIBUTION_MODE::FREE);

  RankNodeSequenceMap rns_map;
  for (tileidx_t rank = 0; rank < gnb.num_processes_; ++rank) {
    const auto num_nodes = aggregated_node_counts.first[rank];
    if (num_nodes < 1)
      continue;

    const auto next_index = total_nodes_per_rank[rank];
    assert_emplace(rns_map.emplace(
        std::make_pair(vp_t(rank), NodeSequence(next_index, num_nodes))));
    total_nodes_per_rank[rank] += num_nodes;
  }

  dist_map = consolidate_node_sequences_per_tile_per_rank(
      rns_map, aggregated_node_counts.second);

  if (const auto local_tns = dist_map.find(gnb.local_rank_);
      local_tns != dist_map.end())
    distribute_node_counts_in_tiles(gnc, local_tns->second, tg, rman,
                                    DISTRIBUTION_MODE::FREE);
  assert(std::accumulate(total_nodes_per_rank.cbegin(),
                         total_nodes_per_rank.cend(), 0) == nodes);
  assert_equal_across_ranks(total_nodes_per_rank);

  const auto tile_pos = gnc.begin();
  const double num_leaf_tiles = tile_pos->size();
  assert(0 < num_leaf_tiles);
  const double expected_mean =
      double(nodes) / num_leaf_tiles / double(gnb.num_processes_);
  std::vector<double> actual_distrib(tile_pos->size(), 0.);
  auto ad_it = actual_distrib.begin();
  double empty = 0.0, min = std::numeric_limits<double>::max(),
         max = std::numeric_limits<double>::lowest(), mean = 0.0;
  for (const auto &ncm : *tile_pos) {
    double count = 0.0;
    for (const auto &node_coords : ncm)
      count += node_coords.size();
    if (count == 0.0)
      ++empty;
    if (std::isless(count, min))
      min = count;
    if (std::isless(max, count))
      max = count;
    mean += count;
    (*ad_it++) = count;
  }
  assert(mean == total_nodes_per_rank[gnb.local_rank_]);
  mean /= num_leaf_tiles;
  empty /= num_leaf_tiles;

  double var = 0.0;
  for (const auto &d : actual_distrib) {
    const auto u = d - mean;
    var += u * u;
  }
  var = std::sqrt(var / num_leaf_tiles);

  printf("Stats:\nmin=%f\nmax=%f\nexpected mean=%f\nactual "
         "mean=%f\nvar=%f\nemptiness=%f\n",
         min, max, expected_mean, mean, var, empty);

  finalize_mpi();
  return 0;
}
