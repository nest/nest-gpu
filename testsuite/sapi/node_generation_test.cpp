#include "coordinate_geometry.h"
#include "vp_interface.h"

#include "random_manager.h"
#include "tile.h"

using namespace sapi;

int main(int argc, char **argv) {
  init_mpi(&argc, &argv);
  init_omp(1);
  RandomManager rng_manager;
  rng_manager.set_rng_seed(12345u);
  std::vector<Tile<Coord2D>> tiles2d(7);
  for (const auto &tile_rotation : std::vector<angle_t>{0, 10, 20}) {
    for (const auto &x_step : {-8, 0, 8}) {
      for (const auto &y_step : {-8, 0, 8}) {
        for (const auto &r_step : {-8, 0, 8}) {
          for (const auto &seed :
               std::initializer_list<std::uint32_t>{1, 2, 3}) {
            rng_manager.set_rng_seed(seed);
            const Coord2D pos = construct_coord_2D(
                x_step != 0 ? 1. / x_step : 0, y_step != 0 ? 1. / y_step : 0);
            const space_t tile_radius = 1. / std::pow(2, r_step);
            tiles2d[0] = Tile<Coord2D>(TILE_SHAPE::RECTANGLE, pos,
                                       {tile_radius}, {tile_rotation});
            tiles2d[1] = Tile<Coord2D>(TILE_SHAPE::TRIANGLE, pos, {tile_radius},
                                       {tile_rotation});
            tiles2d[2] =
                Tile<Coord2D>(TILE_SHAPE::RECTANGLE, pos,
                              {tile_radius, 2 * tile_radius}, {tile_rotation});
            tiles2d[3] =
                Tile<Coord2D>(TILE_SHAPE::TRIANGLE, pos,
                              {tile_radius, 2 * tile_radius}, {tile_rotation});
            tiles2d[4] =
                Tile<Coord2D>(TILE_SHAPE::RECTANGLE, pos,
                              {2 * tile_radius, tile_radius}, {tile_rotation});
            tiles2d[5] =
                Tile<Coord2D>(TILE_SHAPE::TRIANGLE, pos,
                              {2 * tile_radius, tile_radius}, {tile_rotation});
            tiles2d[6] = Tile<Coord2D>(TILE_SHAPE::HEXAGON, pos, {tile_radius},
                                       {tile_rotation});
            for (const auto &tile : tiles2d) {
              auto coords = tile.generate_coords_in_tile(
                  0, 100, *rng_manager.get_rank_synced_rng());
              for (const auto &[index, coord] : coords)
                assert(tile.coord_in_tile(coord));
            }
          }
        }
      }
    }
  }
  finalize_mpi();
  return 0;
}
