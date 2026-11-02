#include "coordinate_geometry.h"
#include "creator_registry.h"
#include "thread_aligned_array.h"
#include "type_erasure_helpers.h"
#include "vp_interface.h"

#include "bounding_box.h"
#include "gf2d_geometry.h"
#include "gf_creators.h"
#include "grid_generation.h"
#include "tile.h"

using namespace sapi;

constexpr const auto *const rectangle_str = "rectangle";
constexpr const auto *const triangle_str = "triangle";
constexpr const auto *const hexagon_str = "hexagon";

void tile_test() {
  for (angle_t tile_rotation = 0; tile_rotation <= 360; tile_rotation += 15) {
    std::vector<Tile<Coord2D>> tiles2d(7);
    tiles2d[0] =
        Tile<Coord2D>(TILE_SHAPE::RECTANGLE, Coord2D(), {1}, {tile_rotation});
    tiles2d[1] =
        Tile<Coord2D>(TILE_SHAPE::TRIANGLE, Coord2D(), {1}, {tile_rotation});
    tiles2d[2] = Tile<Coord2D>(TILE_SHAPE::RECTANGLE, Coord2D(), {1, 2},
                               {tile_rotation});
    tiles2d[3] =
        Tile<Coord2D>(TILE_SHAPE::TRIANGLE, Coord2D(), {1, 2}, {tile_rotation});
    tiles2d[4] = Tile<Coord2D>(TILE_SHAPE::RECTANGLE, Coord2D(), {2, 1},
                               {tile_rotation});
    tiles2d[5] =
        Tile<Coord2D>(TILE_SHAPE::TRIANGLE, Coord2D(), {2, 1}, {tile_rotation});
    tiles2d[6] =
        Tile<Coord2D>(TILE_SHAPE::HEXAGON, Coord2D(), {1}, {tile_rotation});

    for (const auto &t2d : tiles2d) {
      BoundingBox<Coord2D> bb(minmax_vertices(t2d.vertices_), 0);
      for (const auto &vertex : t2d.vertices_)
        assert(bb.coord_in_box(vertex));
    }
  }
}

void test_bb(std::unordered_set<tileidx_t> &pset, const TileGrid<Coord2D> &tg,
             const BoundingBox<Coord2D> &bb) {
  assert((bb.tile_index_ < 0) != bb.inner_boxes_.empty());

  if (0 <= bb.tile_index_) {
    assert(pset.find(bb.tile_index_) == pset.end());
    for (const auto &coord : tg.positions_.at(bb.tile_index_).tile_.vertices_)
      assert(bb.coord_in_box(coord));
    pset.insert(bb.tile_index_);
  } else {
    for (const auto &ibb : bb.inner_boxes_) {
      assert(bb.coord_in_box(ibb.minmax_bounds_.first) &&
             bb.coord_in_box(ibb.minmax_bounds_.second));
      test_bb(pset, tg, ibb);
    }
  }
}

int main() {
  init_omp();

  tile_test();

  TAArray<GFCollection<Coord2D>> gc_array;
  CreatorRegistry<GridTargetPositionShifts<Coord2D>> gscr;
  CreatorRegistry<ShiftedOriginCreator<Coord2D>> socr;
  CreatorRegistry<CachedTileCreator<Coord2D>> ctcr;

  initialize_gsc_registry(gscr);
  initialize_soc_registry(socr);
  initialize_ctc_registry(ctcr);

  for (angle_t tile_rotation = 0; tile_rotation <= 360; tile_rotation += 15) {
    for (tileidx_t width = 1; width <= 10; ++width) {
      for (tileidx_t height = 1; height <= 10; ++height) {
        for (const auto &tile_shape :
             {rectangle_str, triangle_str, hexagon_str}) {
          for (const auto &double_width : {false, true}) {
            if (tile_shape == hexagon_str && double_width)
              continue;

            const auto many_tiles = 2 < (width + height);
            if (many_tiles && tile_shape == rectangle_str &&
                tile_rotation % 45 != 0)
              continue;
            if (many_tiles && tile_shape == triangle_str &&
                (width % 2 != 0 || tile_rotation % 180 != 0))
              continue;
            if (many_tiles && tile_shape == hexagon_str &&
                (height % 2 != 0 || tile_rotation % 30 != 0))
              continue;

            const auto gc = construct_gf_collection<Coord2D>(
                {0., 0.}, {width, height}, tile_shape,
                double_width ? std::vector<space_t>{2, 1}
                             : std::vector<space_t>{1},
                {tile_rotation}, gscr, socr, ctcr);
            gc_array.clone(gc);
            const auto tg = generate_tile_grid(gc_array);
            std::unordered_set<tileidx_t> position_set;
            test_bb(position_set, tg, tg.bounding_box_);
            assert(position_set.size() ==
                   static_cast<std::size_t>(tg.num_tiles_));
          }
        }
      }
    }
  }

  return 0;
}
