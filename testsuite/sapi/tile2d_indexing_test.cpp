#include "coordinate_geometry.h"
#include "vp_interface.h"

#include "bounding_box.h"
#include "gf2d_geometry.h"
#include "gf_creators.h"
#include "grid_generation.h"
#include "random_manager.h"
#include "tile.h"

using namespace sapi;

constexpr const auto *const rectangle_str = "rectangle";
constexpr const auto *const triangle_str = "triangle";
constexpr const auto *const hexagon_str = "hexagon";

int main(int argc, char **argv) {
  init_mpi(&argc, &argv);
  init_omp();

  RandomManager rman;
  rman.set_rng_seed(12345u);
  TAArray<GFCollection<Coord2D>> gc_array;
  CreatorRegistry<GridTargetPositionShifts<Coord2D>> gscr;
  CreatorRegistry<ShiftedOriginCreator<Coord2D>> socr;
  CreatorRegistry<CachedTileCreator<Coord2D>> ctcr;

  initialize_gsc_registry(gscr);
  initialize_soc_registry(socr);
  initialize_ctc_registry(ctcr);

  for (const auto &tile_type : {rectangle_str, triangle_str, hexagon_str}) {
    for (const auto &splits : {0, 1, 2, 3, 4, 5}) {
      const auto gc = construct_gf_collection<Coord2D>(
          {0., 0.}, {1, 1}, tile_type, {1.}, {0}, gscr, socr, ctcr);
      gc_array.clone(gc);
      auto tg = generate_tile_grid(gc_array);
      const auto tile_pos = tg.positions_.begin();
      tile_pos->tile_.split(splits, true);
      assert(!tile_pos->tile_.leaf_tiles_.empty());
      for (std::size_t idx = 0; idx < tile_pos->tile_.leaf_tiles_.size(); ++idx)
        assert(tile_pos->tile_.leaf_tiles_[idx] != nullptr &&
               tile_pos->tile_.leaf_tiles_[idx]->index_ ==
                   static_cast<tileidx_t>(idx));
    }
  }

  finalize_mpi();
  return 0;
}
