#include "coordinate_geometry.h"
#include "mask2d_geometry.h"
#include "test_helpers.h"
#include "tile.h"

using namespace sapi;

void test_square() {
  Tile<Coord2D> square(TILE_SHAPE::RECTANGLE, Coord2D(), {M_SQRT2}, {});
  square.split(1, false);

  Mask<Coord2D> cmask =
      construct_mask<Coord2D>(MASK_SHAPE::CIRCULAR, {1.1, 0.0}, {0.5}, {});
  auto leaf_tiles = cmask.get_overlapping_leaf_sub_tiles(square);
  assert(!leaf_tiles.empty());
  auto sq = leaf_tiles.front();
  leaf_tiles.pop_front();
  assert(sq->index_ == 0);
  assert(!leaf_tiles.empty());
  sq = leaf_tiles.front();
  leaf_tiles.pop_front();
  assert(sq->index_ == 3);
  assert(leaf_tiles.empty());

  Mask<Coord2D> rmask = construct_mask<Coord2D>(
      MASK_SHAPE::PARALLELOGRAM, {0.0, 0.0}, {0.5, 0.5, -0.5, 0.5}, {});
  leaf_tiles = rmask.get_overlapping_leaf_sub_tiles(square);
  assert(leaf_tiles.size() == 4);
  leaf_tiles.clear();

  Mask<Coord2D> emask = construct_mask<Coord2D>(
      MASK_SHAPE::ELLIPTICAL, {-0.5, 0.0}, {0.2, 0.1, 90}, {});
  leaf_tiles = emask.get_overlapping_leaf_sub_tiles(square);
  assert(!leaf_tiles.empty());
  sq = leaf_tiles.front();
  leaf_tiles.pop_front();
  assert(sq->index_ == 1);
  assert(!leaf_tiles.empty());
  sq = leaf_tiles.front();
  leaf_tiles.pop_front();
  assert(sq->index_ == 2);
}

void test_triangle() {
  Tile<Coord2D> triangle(TILE_SHAPE::TRIANGLE, construct_coord_2D(0.0, 0.0),
                         {M_SQRT2}, {});
  triangle.split(2, false);

  Mask<Coord2D> cmask =
      construct_mask<Coord2D>(MASK_SHAPE::CIRCULAR, {0.0, 0.0}, {0.1}, {});
  assert(cmask.get_overlapping_leaf_sub_tiles(triangle).size() == 4);
  assert(Mask<Coord2D>(cmask).get_overlapping_leaf_sub_tiles(triangle).size() ==
         4);

  Mask<Coord2D> pmask = construct_mask<Coord2D>(
      MASK_SHAPE::PARALLELOGRAM, {1.0, -1.0}, {0.5, 0.5, -0.5, 0.5}, {});
  assert(pmask.get_overlapping_leaf_sub_tiles(triangle).empty());
  assert(Mask<Coord2D>(pmask).get_overlapping_leaf_sub_tiles(triangle).empty());

  Mask<Coord2D> emask = construct_mask<Coord2D>(
      MASK_SHAPE::ELLIPTICAL, {-0.7, -0.8}, {0.2, 0.1, 0}, {});
  assert(emask.get_overlapping_leaf_sub_tiles(triangle).size() == 1);
  assert(Mask<Coord2D>(emask).get_overlapping_leaf_sub_tiles(triangle).size() ==
         1);
}

void test_hexagon() {
  Tile<Coord2D> hexagon(TILE_SHAPE::HEXAGON, construct_coord_2D(0.0, 0.0),
                        {0.1}, {});
  hexagon.split(1, false);

  Mask<Coord2D> cmask =
      construct_mask<Coord2D>(MASK_SHAPE::CIRCULAR, {0.0, 0.0}, {1}, {});
  assert(cmask.get_overlapping_leaf_sub_tiles(hexagon).size() == 6);
  assert(Mask<Coord2D>(cmask).get_overlapping_leaf_sub_tiles(hexagon).size() ==
         6);

  Mask<Coord2D> pmask = construct_mask<Coord2D>(MASK_SHAPE::PARALLELOGRAM,
                                                {-1.0, 1.0}, {-2, 2, 2, 2}, {});
  assert(pmask.get_overlapping_leaf_sub_tiles(hexagon).size() == 6);
  assert(Mask<Coord2D>(pmask).get_overlapping_leaf_sub_tiles(hexagon).size() ==
         6);

  Mask<Coord2D> emask = construct_mask<Coord2D>(MASK_SHAPE::ELLIPTICAL,
                                                {0.1, 0.1}, {0.3, 0.2, 0}, {});
  assert(emask.get_overlapping_leaf_sub_tiles(hexagon).size() == 6);
  assert(Mask<Coord2D>(emask).get_overlapping_leaf_sub_tiles(hexagon).size() ==
         6);
}

void test_updated_dist_check() {
  Mask<Coord2D> pmask = construct_mask<Coord2D>(MASK_SHAPE::PARALLELOGRAM,
                                                {0, 0}, {1, 0, 0, 0.5}, {});
  Mask<Coord2D> emask =
      construct_mask<Coord2D>(MASK_SHAPE::ELLIPTICAL, {0, 0}, {1, 0.5}, {});

  Coord2D c;
  c.x_ = 0;
  c.y_ = 1;

  assert(construct_circumscribed_radius<Coord2D>(Coord2D(),
                                                 pmask.c_radius_.radius2_)
             .coord_in_radius(c)
             .first &&
         construct_circumscribed_radius<Coord2D>(Coord2D(),
                                                 emask.c_radius_.radius2_)
             .coord_in_radius(c)
             .first);

  assert(!pmask.coord_in_mask(c).first && !emask.coord_in_mask(c).first);
}

void test_offset() {
  Mask<Coord2D> cmask =
      construct_mask<Coord2D>(MASK_SHAPE::CIRCULAR, {0, 0}, {1.0}, {});
  Mask<Coord2D> pmask =
      construct_mask<Coord2D>(MASK_SHAPE::PARALLELOGRAM, {0, 0},
                              {M_SQRT2, M_SQRT2, M_SQRT2, -M_SQRT2}, {});
  Mask<Coord2D> emask =
      construct_mask<Coord2D>(MASK_SHAPE::ELLIPTICAL, {0, 0}, {2, 1}, {});

  Coord2D c;
  c.x_ = 2;
  c.y_ = 2;

  assert(!cmask.coord_in_mask(c).first && !pmask.coord_in_mask(c).first &&
         !emask.coord_in_mask(c).first);

  cmask = construct_mask<Coord2D>(MASK_SHAPE::CIRCULAR, {0, 0}, {1.0}, {2, 1});
  pmask =
      construct_mask<Coord2D>(MASK_SHAPE::PARALLELOGRAM, {0, 0},
                              {M_SQRT2, M_SQRT2, M_SQRT2, -M_SQRT2}, {1, 1});
  emask =
      construct_mask<Coord2D>(MASK_SHAPE::ELLIPTICAL, {0, 0}, {2, 1}, {0, 2});

  assert(cmask.coord_in_mask(c).first && pmask.coord_in_mask(c).first &&
         emask.coord_in_mask(c).first);
  assert(cmask.coord_in_mask(Coord2D(), c).first &&
         pmask.coord_in_mask(Coord2D(), c).first &&
         emask.coord_in_mask(Coord2D(), c).first);
}

void test_triangular_mask() {
  Mask<Coord2D> tmask =
      construct_mask<Coord2D>(MASK_SHAPE::TRIANGULAR, {0, 0},
                              {M_SQRT2, M_SQRT2, M_SQRT2, -M_SQRT2}, {});
  Mask<Coord2D> pmask =
      construct_mask<Coord2D>(MASK_SHAPE::PARALLELOGRAM, {0, 0},
                              {M_SQRT2, M_SQRT2, M_SQRT2, -M_SQRT2}, {});

  vertidx_t triangular_comparisons = 0;
  for (const auto &x : {M_SQRT2, -M_SQRT2, 1., 0., -1., 0.5, -0.5}) {
    for (const auto &y : {M_SQRT2, -M_SQRT2, 1., 0., -1., 0.5, -0.5}) {
      const Coord2D test_v = construct_coord_2D(x, y);
      if (leq_test(std::fabs(y), std::fabs(x)) && 0 <= x) {
        assert(tmask.coord_in_mask(test_v).first);
        ++triangular_comparisons;
      } else
        assert(!tmask.coord_in_mask(test_v).first);

      assert(pmask.coord_in_mask(test_v).first);
    }
  }

  assert(triangular_comparisons == 16);
}

int main() {
  test_square();
  test_triangle();
  test_hexagon();
  test_updated_dist_check();
  test_offset();
  test_triangular_mask();
  return 0;
}
