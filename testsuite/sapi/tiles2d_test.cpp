#include "coordinate_geometry.h"
#include "vp_interface.h"

#include "tile.h"

using namespace sapi;

void test_num_splits() {
  for (const auto &tile_type :
       {TILE_SHAPE::RECTANGLE, TILE_SHAPE::TRIANGLE, TILE_SHAPE::HEXAGON}) {
    for (nodeidx_t shift_nodes = 10; shift_nodes < 20; ++shift_nodes) {
      for (tileidx_t shift_tiles = 10; shift_tiles < 20; ++shift_tiles) {
        for (nodeidx_t leaf_expectation = 10; leaf_expectation < 101;
             leaf_expectation += 10) {
          const double num_nodes = (1 << shift_nodes) + 1;
          const double num_tiles = (1 << shift_tiles) + 1;
          const auto num_splits = compute_minimal_splits(
              num_nodes, num_tiles, leaf_expectation, tile_type);

          switch (tile_type) {
          case TILE_SHAPE::RECTANGLE:
            assert(
                leaf_expectation <=
                    num_nodes / (num_tiles * std::pow(4., num_splits)) ||
                (num_splits == 0 && num_nodes / num_tiles <= leaf_expectation));
            break;

          case TILE_SHAPE::TRIANGLE:
            assert(
                leaf_expectation <=
                    num_nodes / (num_tiles * std::pow(2., num_splits)) ||
                (num_splits == 0 && num_nodes / num_tiles <= leaf_expectation));
            break;

          case TILE_SHAPE::HEXAGON:
            assert(
                leaf_expectation <=
                    num_nodes /
                        (num_tiles * (0 < num_splits
                                          ? 6 * std::pow(2., num_splits - 1)
                                          : 1)) ||
                (num_splits == 0 && num_nodes / num_tiles <= leaf_expectation));
            break;

          default:
            break;
          }
        }
      }
    }
  }
}

int main() {
  init_omp();
  test_num_splits();
  Tile<Coord2D> tiles[7];
  for (const angle_t rotation : {0, 90, 180, 270}) {
    for (const auto &x_step : {-1, 0, 1}) {
      for (const auto &y_step : {-1, 0, 1}) {
        for (const auto &radius : {0.00001, 1.}) {
          for (const auto &splits : {0, 5}) {
            const Coord2D pos =
                construct_coord_2D(radius * x_step, radius * y_step);
            tiles[0] =
                Tile<Coord2D>(TILE_SHAPE::RECTANGLE, pos, {radius}, {rotation});
            tiles[1] =
                Tile<Coord2D>(TILE_SHAPE::TRIANGLE, pos, {radius}, {rotation});
            tiles[2] =
                Tile<Coord2D>(TILE_SHAPE::HEXAGON, pos, {radius}, {rotation});
            tiles[3] = Tile<Coord2D>(TILE_SHAPE::RECTANGLE, pos,
                                     {radius, 2 * radius}, {rotation});
            tiles[4] = Tile<Coord2D>(TILE_SHAPE::TRIANGLE, pos,
                                     {radius, 2 * radius}, {rotation});
            tiles[5] = Tile<Coord2D>(TILE_SHAPE::RECTANGLE, pos,
                                     {2 * radius, radius}, {rotation});
            tiles[6] = Tile<Coord2D>(TILE_SHAPE::TRIANGLE, pos,
                                     {2 * radius, radius}, {rotation});

#pragma omp parallel default(shared)
#pragma omp master
            {
#pragma omp taskgroup
              {
#pragma omp task default(shared)
                tiles[0].split(splits, false);
#pragma omp task default(shared)
                tiles[1].split(splits * 2, false);
#pragma omp task default(shared)
                tiles[2].split(0 < splits ? splits * 2 - 1 : 0, false);
#pragma omp task default(shared)
                tiles[3].split(splits, false);
#pragma omp task default(shared)
                tiles[4].split(splits * 2, false);
#pragma omp task default(shared)
                tiles[5].split(splits, false);
#pragma omp task default(shared)
                tiles[6].split(splits * 2, false);
              }

              std::deque<const Tile<Coord2D> *> work{
                  &tiles[0], &tiles[1], &tiles[2], &tiles[3],
                  &tiles[4], &tiles[5], &tiles[6]};
              std::deque<const Tile<Coord2D> *> queue;

              do {

#pragma omp taskgroup
                do {
                  const auto tile = work.front();
                  work.pop_front();

                  std::deque<const Tile<Coord2D> *> subwork;

                  if (!tile->sub_tiles_.empty()) {
                    assert(tile->index_ == -1);
                    for (const auto &st : tile->sub_tiles_) {
                      queue.emplace_front(&st);
                      subwork.emplace_front(&st);
                    }
                  } else {
                    assert(0 <= tile->index_);
                  }

                  subwork.emplace_front(tile);

#pragma omp task default(none) firstprivate(subwork)
                  {
                    const auto parent_tile = subwork.front();
                    do {
                      const auto subtile = subwork.front();
                      subwork.pop_front();

                      for (const auto &vertex : subtile->vertices_) {
                        const auto par_disp = construct_displacement(
                            vertex - parent_tile->c_radius_.origin_);
                        const auto sub_disp = construct_displacement(
                            vertex - subtile->c_radius_.origin_);
                        assert(almost_equal(subtile->c_radius_.radius2_,
                                            sub_disp.distance2_));
                        assert(parent_tile->c_radius_.disp_in_radius(par_disp));
                        assert(parent_tile->coord_in_tile(vertex));
                        assert(subtile->coord_in_tile(vertex));
                      }

                    } while (!subwork.empty());
                  }

                } while (!work.empty());

                while (!queue.empty()) {
                  work.emplace_front(queue.front());
                  queue.pop_front();
                }

              } while (!work.empty());
            }
          }
        }
      }
    }
  }

  return 0;
}
