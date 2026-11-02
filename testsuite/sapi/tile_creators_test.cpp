#include <cstdio>

#include "coordinate_geometry.h"

#include "gf2d_geometry.h"
#include "tile.h"

using namespace sapi;

int main() {
  std::vector<CachedTileCreator<Coord2D>> ctc_vec(7);
  ctc_vec[0] =
      construct_cached_tile_creator<Coord2D>(TILE_SHAPE::RECTANGLE, {1}, {0});
  ctc_vec[1] =
      construct_cached_tile_creator<Coord2D>(TILE_SHAPE::TRIANGLE, {1}, {0});
  ctc_vec[2] = construct_cached_tile_creator<Coord2D>(TILE_SHAPE::RECTANGLE,
                                                      {1, 2}, {0});
  ctc_vec[3] =
      construct_cached_tile_creator<Coord2D>(TILE_SHAPE::TRIANGLE, {1, 2}, {0});
  ctc_vec[4] = construct_cached_tile_creator<Coord2D>(TILE_SHAPE::RECTANGLE,
                                                      {2, 1}, {0});
  ctc_vec[5] =
      construct_cached_tile_creator<Coord2D>(TILE_SHAPE::TRIANGLE, {2, 1}, {0});
  ctc_vec[6] =
      construct_cached_tile_creator<Coord2D>(TILE_SHAPE::HEXAGON, {1}, {0});
  GridPositionParity<Coord2D> pos_status = {false, false};
  for (const auto &ctc : ctc_vec)
    for (const auto &not_mirrored : {true, false}) {
      pos_status[0] = not_mirrored;
      auto tile = ctc.create_tile(Coord2D(), pos_status);
      printf("\nMirrored %i : tile: %s\n", not_mirrored,
             tile.to_string().c_str());
    }
  return 0;
}
