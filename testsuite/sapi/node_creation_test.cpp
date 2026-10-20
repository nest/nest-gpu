#include <cstdio>

#include "coordinate_geometry.h"
#include "creator_registry.h"
#include "type_erasure_helpers.h"
#include "vp_interface.h"

#include "bounding_box.h"
#include "gf2d_geometry.h"
#include "gf_creators.h"
#include "grid_generation.h"
#include "node_containers_init.h"
#include "node_creation.h"
#include "test_helpers.h"

using namespace sapi;

constexpr static int NODES = 5000;
constexpr static int ITERATIONS = 10;

template <bool balanced>
void node_count_distribution_test(const RandomManager &rman) {
  for (int i = 0; i < ITERATIONS; ++i) {
    const auto nodes_per_tile =
        uniform_distribute_node_counts<nodeidx_t, balanced, true>(NODES, 100,
                                                                  rman);

    assert(nodes_per_tile.size() == 100);
    assert(std::accumulate(nodes_per_tile.cbegin(), nodes_per_tile.cend(), 0) ==
           NODES);
    assert_equal_across_ranks(nodes_per_tile);
  }
}

template <typename CoordT>
void node_creation_test(NodeCountVector &total_nodes_per_rank,
                        const std::optional<std::set<tileidx_t>> &tile_set,
                        const TileGrid<CoordT> &tg, const GridNeighborhood &gnb,
                        GridNodeCollection<CoordT> &gnc,
                        const RandomManager &rman,
                        const DISTRIBUTION_MODE mode) {
  const auto safe_mode =
      mode == DISTRIBUTION_MODE::FREE
          ? tile_set.has_value() ? DISTRIBUTION_MODE::SQUEEZED : mode
          : mode;
  for (int i = 0; i < ITERATIONS; ++i) {
    const auto aggregated_node_counts = distribute_node_counts_in_grid(
        NODES, tile_set, tg, gnb, rman, safe_mode);

    assert(std::accumulate(aggregated_node_counts.first.cbegin(),
                           aggregated_node_counts.first.cend(), 0) == NODES);

    NodeCountVector reconstructed_tile_node_counts(tg.num_tiles_, 0);
    for (const auto &rank_tile_node_counts : aggregated_node_counts.second)
      for (const auto &tile_node_count_pair : rank_tile_node_counts)
        reconstructed_tile_node_counts[tile_node_count_pair.first] +=
            tile_node_count_pair.second;

    assert(std::accumulate(reconstructed_tile_node_counts.cbegin(),
                           reconstructed_tile_node_counts.cend(), 0) == NODES);
    assert_equal_across_ranks(reconstructed_tile_node_counts);

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

    const auto rns_map_size = static_cast<int>(rns_map.size());
    assert(0 <= rns_map_size);
    assert_equal_across_ranks(std::vector<int>{rns_map_size});
    assert_equal_across_ranks(total_nodes_per_rank);

    std::vector<int> rank_order, index_order, nodes_order;
    rank_order.resize(rns_map_size, 0);
    index_order.resize(rns_map_size, 0);
    nodes_order.resize(rns_map_size, 0);
    vp_t rank_idx = 0;
    for (const auto &rank_node_seq_pair : rns_map) {
      rank_order[rank_idx] = static_cast<int>(rank_node_seq_pair.first);
      index_order[rank_idx] = static_cast<int>(rank_node_seq_pair.second.first);
      nodes_order[rank_idx] =
          static_cast<int>(rank_node_seq_pair.second.second);
      ++rank_idx;
    }
    assert_equal_across_ranks(rank_order);
    assert_equal_across_ranks(index_order);
    assert_equal_across_ranks(nodes_order);

    const auto dist_tns = consolidate_node_sequences_per_tile_per_rank(
        rns_map, aggregated_node_counts.second);
    assert(static_cast<int>(dist_tns.size()) == rns_map_size);

    rank_idx = 0;
    std::vector<int> tile_order;
    rank_order.clear();
    rank_order.resize(rns_map_size, 0);
    for (const auto &rank_tns_map_pair : dist_tns) {
      rank_order[rank_idx] = static_cast<int>(rank_tns_map_pair.first);
      const auto tns_map_size = rank_tns_map_pair.second.size();
      tile_order.clear();
      index_order.clear();
      nodes_order.clear();
      tile_order.resize(tns_map_size, 0);
      index_order.resize(tns_map_size, 0);
      nodes_order.resize(tns_map_size, 0);
      tileidx_t tile_idx = 0;
      for (const auto &tns_pair : rank_tns_map_pair.second) {
        tile_order[tile_idx] = static_cast<int>(tns_pair.first);
        index_order[tile_idx] = static_cast<int>(tns_pair.second.first);
        nodes_order[tile_idx] = static_cast<int>(tns_pair.second.second);
        ++tile_idx;
      }
      assert_equal_across_ranks(tile_order);
      assert_equal_across_ranks(index_order);
      assert_equal_across_ranks(nodes_order);
      ++rank_idx;
    }
    assert_equal_across_ranks(rank_order);

    if (const auto local_tns = dist_tns.find(gnb.local_rank_);
        local_tns != dist_tns.end())
      distribute_node_counts_in_tiles(gnc, local_tns->second, tg, rman, mode);

    nodeidx_t count = 0;
    for (const auto &tix : *gnb.locally_owned_tiles_)
      for (const auto &leaf_ncm : gnc.at(tix))
        for (const auto &nix_coord_pair_vec : leaf_ncm)
          count += nix_coord_pair_vec.size();

    assert(count == total_nodes_per_rank[gnb.local_rank_]);
  }
}

void tile_injected_rank_subjected_test(const RandomManager &rman,
                                       const vp_t num_processes,
                                       const DISTRIBUTION_MODE mode) {
  assert(rman.is_initialized());
  NodeCountVector total_nodes_per_rank(num_processes, 0);
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
  assert(!gnb.rank_tile_bijection_);

  std::optional<std::set<tileidx_t>> tile_set;

  node_creation_test(total_nodes_per_rank, tile_set, tg, gnb, gnc, rman, mode);

  tile_set.emplace({0});

  node_creation_test(total_nodes_per_rank, tile_set, tg, gnb, gnc, rman, mode);
}

void tile_subjected_rank_injected_test(const RandomManager &rman,
                                       const vp_t num_processes,
                                       const DISTRIBUTION_MODE mode) {
  assert(rman.is_initialized());
  NodeCountVector total_nodes_per_rank(num_processes, 0);
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
  split_tiles_in_grid(tg, *gnb.locally_owned_tiles_, 10);
  initialize_local_grid_node_collection(gnc, *gnb.locally_owned_tiles_, tg);
  assert(!gnb.rank_tile_bijection_);

  std::optional<std::set<tileidx_t>> tile_set;

  node_creation_test(total_nodes_per_rank, tile_set, tg, gnb, gnc, rman, mode);

  tile_set.emplace({0});

  node_creation_test(total_nodes_per_rank, tile_set, tg, gnb, gnc, rman, mode);
}

void tile_subjected_rank_subjected_test(const RandomManager &rman,
                                        const vp_t num_processes,
                                        const DISTRIBUTION_MODE mode) {
  assert(rman.is_initialized());
  NodeCountVector total_nodes_per_rank(num_processes, 0);
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

  const auto gc = construct_gf_collection<Coord2D>({0., 0.}, {2, 2}, "hexagon",
                                                   {1.}, {0}, gscr, socr, ctcr);
  gc_array.clone(gc);
  tg = generate_tile_grid(gc_array);

  std::vector<std::set<tileidx_t>> rank_tile_ownership_map(num_processes);
  for (tileidx_t rank = 0; rank < num_processes; ++rank)
    rank_tile_ownership_map[rank] = {0, 1, 2, 3};
  gnb.set_tile_ownership(std::move(rank_tile_ownership_map), tg);
  split_tiles_in_grid(tg, *gnb.locally_owned_tiles_, 10);
  initialize_local_grid_node_collection(gnc, *gnb.locally_owned_tiles_, tg);
  assert(!gnb.rank_tile_bijection_);

  std::optional<std::set<tileidx_t>> tile_set;

  node_creation_test(total_nodes_per_rank, tile_set, tg, gnb, gnc, rman, mode);

  tile_set.emplace({0});

  node_creation_test(total_nodes_per_rank, tile_set, tg, gnb, gnc, rman, mode);
}

void tile_rank_bijected_test(const RandomManager &rman,
                             const vp_t num_processes,
                             const DISTRIBUTION_MODE mode) {
  assert(rman.is_initialized());
  NodeCountVector total_nodes_per_rank(num_processes, 0);
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
      {0., 0.}, {1, num_processes}, "rectangle", {1.}, {0}, gscr, socr, ctcr);
  gc_array.clone(gc);
  tg = generate_tile_grid(gc_array);

  std::vector<std::set<tileidx_t>> rank_tile_ownership_map(num_processes);
  for (tileidx_t rank = 0; rank < num_processes; ++rank)
    rank_tile_ownership_map[rank] = {rank};
  gnb.set_tile_ownership(std::move(rank_tile_ownership_map), tg);
  split_tiles_in_grid(tg, *gnb.locally_owned_tiles_, 10);
  initialize_local_grid_node_collection(gnc, *gnb.locally_owned_tiles_, tg);
  assert(gnb.rank_tile_bijection_);

  std::optional<std::set<tileidx_t>> tile_set;

  node_creation_test(total_nodes_per_rank, tile_set, tg, gnb, gnc, rman, mode);

  tile_set.emplace({0});

  node_creation_test(total_nodes_per_rank, tile_set, tg, gnb, gnc, rman, mode);
}

void tile_rank_sparse_domain_test(const RandomManager &rman,
                                  const vp_t num_processes,
                                  const DISTRIBUTION_MODE mode) {
  assert(rman.is_initialized());
  NodeCountVector total_nodes_per_rank(num_processes, 0);
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
      {0., 0.}, {1, num_processes}, "rectangle", {1.}, {0}, gscr, socr, ctcr);
  gc_array.clone(gc);
  tg = generate_tile_grid(gc_array);

  std::set<tileidx_t> tile_set;
  std::vector<std::set<tileidx_t>> rank_tile_ownership_map(num_processes);
  for (tileidx_t rank = 0; rank < num_processes; ++rank) {
    if (rank % 2 == 0) {
      rank_tile_ownership_map[rank] = {rank};
      tile_set.insert(rank);
    }
  }
  gnb.set_tile_ownership(std::move(rank_tile_ownership_map), tg);
  split_tiles_in_grid(tg, *gnb.locally_owned_tiles_, 10);
  initialize_local_grid_node_collection(gnc, *gnb.locally_owned_tiles_, tg);
  assert(!gnb.rank_tile_bijection_);

  node_creation_test(total_nodes_per_rank, std::make_optional(tile_set), tg,
                     gnb, gnc, rman, mode);
}

int main(int argc, char **argv) {
  init_mpi(&argc, &argv);
  init_omp(5);

  const tileidx_t num_processes =
      static_cast<tileidx_t>(get_num_mpi_processes());

  if (num_processes > 2) {
    fprintf(stderr, "Skipping test with MPI WORLD SIZE larger than 2");
    finalize_mpi();
    return 0;
  }

  RandomManager rman;
  rman.set_rng_seed(12345u);

  node_count_distribution_test<true>(rman);
  node_count_distribution_test<false>(rman);

  for (const auto &mode : {DISTRIBUTION_MODE::FREE, DISTRIBUTION_MODE::SQUEEZED,
                           DISTRIBUTION_MODE::BALANCED}) {

    if (num_processes > 1)
      tile_injected_rank_subjected_test(rman, num_processes, mode);

    tile_subjected_rank_injected_test(rman, num_processes, mode);

    if (num_processes > 1)
      tile_subjected_rank_subjected_test(rman, num_processes, mode);

    tile_rank_bijected_test(rman, num_processes, mode);

    if (num_processes > 1)
      tile_rank_sparse_domain_test(rman, num_processes, mode);
  }

  finalize_mpi();
  return 0;
}
