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
#include "grid_neighborhood.h"
#include "node_containers_init.h"
#include "tile.h"

using namespace sapi;

void bijection_test(
    TAArray<GFCollection<Coord2D>> &gc_array,
    const CreatorRegistry<GridTargetPositionShifts<Coord2D>> &gscr,
    const CreatorRegistry<ShiftedOriginCreator<Coord2D>> &socr,
    const CreatorRegistry<CachedTileCreator<Coord2D>> &ctcr,
    const tileidx_t num_processes) {
  if (num_processes > 1) {
    const tileidx_t num_tiles = num_processes - 1;
    TileGrid<Coord2D> tg;
    GridNeighborhood gnb;
    const auto gc = construct_gf_collection<Coord2D>(
        {0., 0.}, {1, num_processes}, "rectangle", {1.}, {0}, gscr, socr, ctcr);
    gc_array.clone(gc);
    tg = generate_tile_grid(gc_array);
    std::vector<std::set<tileidx_t>> rank_tile_ownership_map(num_processes);
    for (tileidx_t rank = 0; rank < num_processes; ++rank)
      rank_tile_ownership_map[rank] = {rank % num_tiles};
    gnb.set_tile_ownership(std::move(rank_tile_ownership_map), tg);
    assert(!gnb.rank_tile_bijection_);
  }

  {
    TileGrid<Coord2D> tg;
    GridNeighborhood gnb;
    const auto gc = construct_gf_collection<Coord2D>(
        {0., 0.}, {2, num_processes}, "rectangle", {1.}, {0}, gscr, socr, ctcr);
    gc_array.clone(gc);
    tg = generate_tile_grid(gc_array);
    std::vector<std::set<tileidx_t>> rank_tile_ownership_map(num_processes);
    for (tileidx_t rank = 0; rank < num_processes; ++rank)
      rank_tile_ownership_map[rank] = {rank, num_processes + rank};
    gnb.set_tile_ownership(std::move(rank_tile_ownership_map), tg);
    assert(!gnb.rank_tile_bijection_);
  }

  if (num_processes > 1) {
    TileGrid<Coord2D> tg;
    GridNeighborhood gnb;
    const auto gc = construct_gf_collection<Coord2D>(
        {0., 0.}, {1, num_processes}, "rectangle", {1.}, {0}, gscr, socr, ctcr);
    gc_array.clone(gc);
    tg = generate_tile_grid(gc_array);
    std::vector<std::set<tileidx_t>> rank_tile_ownership_map(num_processes);
    for (tileidx_t rank = 0; rank < num_processes; ++rank)
      if (rank % 2 == 0)
        rank_tile_ownership_map[rank] = {rank};
    gnb.set_tile_ownership(std::move(rank_tile_ownership_map), tg);
    assert(!gnb.rank_tile_bijection_);
  }
}

void splitting_test(
    TAArray<GFCollection<Coord2D>> &gc_array,
    const CreatorRegistry<GridTargetPositionShifts<Coord2D>> &gscr,
    const CreatorRegistry<ShiftedOriginCreator<Coord2D>> &socr,
    const CreatorRegistry<CachedTileCreator<Coord2D>> &ctcr,
    const tileidx_t num_processes, const vp_t local_rank) {
  const tileidx_t num_tiles = num_processes;
  TileGrid<Coord2D> tg;
  GridNeighborhood gnb;
  const auto gc = construct_gf_collection<Coord2D>(
      {0., 0.}, {1, num_processes}, "rectangle", {1.}, {0}, gscr, socr, ctcr);
  gc_array.clone(gc);
  tg = generate_tile_grid(gc_array);

  assert(!gnb.has_owners_ && num_processes == gnb.num_processes_ &&
         local_rank == gnb.local_rank_);

  std::vector<std::set<tileidx_t>> rank_tile_ownership_map(num_processes);
  for (tileidx_t rank = 0; rank < num_processes; ++rank)
    rank_tile_ownership_map[rank] = {rank};
  gnb.set_tile_ownership(std::move(rank_tile_ownership_map), tg);

  assert(gnb.has_owners_ && gnb.rank_tile_bijection_);

  if (1 < num_processes) {
    for (tileidx_t rank = 0; rank < num_processes; ++rank) {
      const auto tile_map = gnb.rank_tiles_ownership_map_.cbegin() + rank;
      assert(tile_map->size() == 1);
      const auto neighbors_map = gnb.rank_neighbors_map_.cbegin() + rank;
      assert(!neighbors_map->first.empty() && !neighbors_map->second.empty());
      for (const auto &tile_idx : *tile_map) {
        const auto tile_pos = tg.positions_.cbegin() + tile_idx;
        for (const auto &neighbor_index :
             tile_pos->wrapped_tile_neighborhood_) {
          const auto tile_owners =
              gnb.tile_ranks_ownership_map_.cbegin() + neighbor_index;
          assert(tile_owners->size() == 1);
          for (const auto &target_rank : *tile_owners)
            assert(neighbors_map->second.find(target_rank) !=
                   neighbors_map->second.end());
        }
      }
    }
  }

  split_tiles_in_grid(tg, *gnb.locally_owned_tiles_, 1);

  assert(tg.has_split_);

  for (tileidx_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
    const auto tile_pos = tg.positions_.cbegin() + tile_idx;
    const auto tile_owners = gnb.tile_ranks_ownership_map_.cbegin() + tile_idx;
    if (tile_owners->find(local_rank) != tile_owners->end()) {
      assert(!tile_pos->tile_.sub_tiles_.empty());
      assert(!tile_pos->tile_.leaf_tiles_.empty());
    } else {
      bool in_neighborhood = false;
      for (const auto &owned_tiles : *gnb.locally_owned_tiles_)
        in_neighborhood |=
            tg.positions_.at(owned_tiles).in_neighborhood(tile_idx, true);

      if (in_neighborhood) {
        assert(!tile_pos->tile_.sub_tiles_.empty());
        assert(tile_pos->tile_.leaf_tiles_.empty());
      } else {
        assert(tile_pos->tile_.sub_tiles_.empty());
        assert(tile_pos->tile_.leaf_tiles_.empty());
      }
    }
  }

  mpi_barrier();
  for (tileidx_t rank = 0; rank < num_processes; ++rank) {
    if (rank == static_cast<tileidx_t>(local_rank))
      printf("RANK %i\n%s\n", local_rank, gnb.to_string().c_str());
    mpi_barrier();
  }
}

int main(int argc, char **argv) {
  init_mpi(&argc, &argv);
  init_omp();

  const vp_t local_rank = get_mpi_rank();
  const tileidx_t num_processes =
      static_cast<tileidx_t>(get_num_mpi_processes());

  TAArray<GFCollection<Coord2D>> gc_array;
  CreatorRegistry<GridTargetPositionShifts<Coord2D>> gscr;
  CreatorRegistry<ShiftedOriginCreator<Coord2D>> socr;
  CreatorRegistry<CachedTileCreator<Coord2D>> ctcr;

  initialize_gsc_registry(gscr);
  initialize_soc_registry(socr);
  initialize_ctc_registry(ctcr);

  bijection_test(gc_array, gscr, socr, ctcr, num_processes);

  splitting_test(gc_array, gscr, socr, ctcr, num_processes, local_rank);

  finalize_mpi();
  return 0;
}
