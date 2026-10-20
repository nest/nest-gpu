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

using namespace sapi;

constexpr const auto *const rectangle_str = "rectangle";
constexpr const auto *const triangle_str = "triangle";
constexpr const auto *const hexagon_str = "hexagon";

vertidx_t vertex_matches(const Tile<Coord2D> &source,
                         const Tile<Coord2D> &target) {
  vertidx_t vertex_overlaps = 0;
  assert(source.vertices_.size() < std::numeric_limits<vertidx_t>::max() &&
         target.vertices_.size() < std::numeric_limits<vertidx_t>::max());

  // Test vertex overlaps
  for (const auto &svx : source.vertices_) {
    uint matches = 0;
    for (const auto &tvx : target.vertices_) {
      if (almost_zero(distance2(tvx, svx)))
        ++matches;
    }
    assert(matches <= 1);
    vertex_overlaps += matches;
  }

  return vertex_overlaps;
}

void test_images(const space_t side_length, const space_t tile_rotation,
                 const Tile<Coord2D> &tile,
                 const GridPosition<Coord2D> &tile_position,
                 const std::string &tile_shape, const Coord2D &grid_origin,
                 const bool double_height) {
  const space_t hex_v_displacement = 3 * side_length / 2;
  const space_t hex_h_displacement = std::sqrt(space_t(3)) * side_length;
  const space_t hex_h_offset = hex_h_displacement / 2;
  const bool is_rotated = tile_rotation > 0;
  const Coord2D angular_offset = create_angular_offset(tile_rotation);

  Coord2D displacement;
  if (tile_shape == rectangle_str) {
    displacement.x_ = tile_position[0] * side_length;
    displacement.y_ = tile_position[1] * side_length * (double_height ? 2 : 1);
  } else if (tile_shape == triangle_str) {
    displacement.x_ = (tile_position[0] % 2 == 0 ? tile_position[0] / 2
                                                 : (tile_position[0] - 1) / 2) *
                      side_length;
    displacement.y_ = tile_position[1] * side_length * (double_height ? 2 : 1);
  } else {
    displacement.x_ =
        tile_position[1] % 2 == 0
            ? hex_h_displacement * tile_position[0]
            : std::fma(hex_h_displacement, tile_position[0], hex_h_offset);
    displacement.y_ = hex_v_displacement * tile_position[1];
  }

  if (is_rotated)
    displacement = rotate_displacement(displacement, angular_offset);

  assert(almost_zero(
      distance2(grid_origin + displacement, tile.c_radius_.origin_)));
}

void test_grid(TAArray<GFCollection<Coord2D>> &gc_array,
               const CreatorRegistry<GridTargetPositionShifts<Coord2D>> &gscr,
               const CreatorRegistry<ShiftedOriginCreator<Coord2D>> &socr,
               const CreatorRegistry<CachedTileCreator<Coord2D>> &ctcr,
               const tileidx_t width, const tileidx_t height,
               const std::string tile_shape, const space_t side_length,
               const angle_t tile_rotation) {
  for (const bool double_height : {true, false}) {
    if (double_height && tile_shape == hexagon_str)
      continue;

    const auto gc = construct_gf_collection<Coord2D>(
        {0., 0.}, {width, height}, tile_shape,
        double_height ? std::vector<space_t>{side_length, side_length * 2}
                      : std::vector<space_t>{side_length},
        {tile_rotation}, gscr, socr, ctcr);
    gc_array.clone(gc);
    const auto tile_grid = generate_tile_grid(gc_array);

    for (tileidx_t s_pos = 0; s_pos < tile_grid.num_tiles_; ++s_pos) {
      const tileidx_t s_col = s_pos % width;
      const tileidx_t s_row = s_pos / width;
      const bool even_s_col = evenTix(s_col);
      const bool even_s_row = evenTix(s_row);
      assert(s_pos == gc.compute_index({s_col, s_row}));
      const auto s_tile_pos = tile_grid.positions_.cbegin() + s_pos;
      assert(s_tile_pos->position_[0] == s_col &&
             s_tile_pos->position_[1] == s_row &&
             s_tile_pos->tile_.c_radius_.origin_ ==
                 gc.shift_origin({s_col, s_row}, {even_s_col, even_s_row}));

      test_images(side_length, tile_rotation, s_tile_pos->tile_,
                  s_tile_pos->position_, tile_shape, gc.grid_origin_,
                  double_height);

      assert(s_tile_pos->tile_images_.size() == 9 &&
             s_tile_pos->image_displacements_.size() == 9 &&
             s_tile_pos->tile_images_.cbegin()->shape_ == TILE_SHAPE::NULL_TS &&
             s_tile_pos->image_displacements_.cbegin()->is_null());

      auto image_it = s_tile_pos->tile_images_.cbegin() + 1;
      auto displacement_it = s_tile_pos->image_displacements_.cbegin() + 1;

      for (shift_t idx = 1; idx < 9; ++idx) {
        assert(image_it->shape_ != TILE_SHAPE::NULL_TS &&
               *displacement_it == (image_it->c_radius_.origin_ -
                                    s_tile_pos->tile_.c_radius_.origin_));

        shift_t equivalent_images = 0;
        bool found_correct_pos = false;
        GridPosition<Coord2D> image_pos;
        for (const auto &width_shift : {-1, 0, 1}) {
          for (const auto &height_shift : {-1, 0, 1}) {
            if (width_shift == 0 && height_shift == 0)
              continue;

            if (!found_correct_pos) {
              image_pos = {s_tile_pos->position_[0] + width_shift * width,
                           s_tile_pos->position_[1] + height_shift * height};
              const auto test_origin = gc.shift_origin(
                  image_pos, {evenTix(image_pos[0]), evenTix(image_pos[1])});
              found_correct_pos = image_it->c_radius_.origin_ == test_origin;
              equivalent_images += found_correct_pos;
            } else {
              const auto test_origin = gc.shift_origin(
                  {s_tile_pos->position_[0] + width_shift * width,
                   s_tile_pos->position_[1] + height_shift * height},
                  {evenTix(image_pos[0]), evenTix(image_pos[1])});
              equivalent_images += image_it->c_radius_.origin_ == test_origin;
            }
          }
        }
        assert(equivalent_images == 1);

        test_images(side_length, tile_rotation, *image_it, image_pos,
                    tile_shape, gc.grid_origin_, double_height);

        ++image_it;
        ++displacement_it;
      }

      for (const auto test_edge_wrappings : {true, false}) {
        for (const auto &neighbor_index :
             s_tile_pos->get_tile_neighborhood(test_edge_wrappings)) {
          // Test indexing
          const tileidx_t n_col = neighbor_index % width;
          const tileidx_t n_row = neighbor_index / width;
          assert(gc.compute_index({n_col, n_row}) == neighbor_index);
          const auto n_tile_pos =
              tile_grid.positions_.cbegin() + neighbor_index;
          assert(n_tile_pos->position_[0] == n_col &&
                 n_tile_pos->position_[1] == n_row);

          if (neighbor_index == s_pos) {
            assert(*s_tile_pos == *n_tile_pos);
            continue;
          } else
            assert(s_tile_pos->position_ != n_tile_pos->position_);

          // Test  neighborhood reciprocity
          assert(n_tile_pos->in_neighborhood(s_pos, test_edge_wrappings));

          // Test vertex overlap
          vertidx_t max_overlaps = 0;
          if (test_edge_wrappings)
            for (const auto &shifted_image : n_tile_pos->tile_images_)
              max_overlaps = std::max(
                  max_overlaps,
                  vertex_matches(s_tile_pos->tile_,
                                 shifted_image.shape_ != TILE_SHAPE::NULL_TS
                                     ? shifted_image
                                     : n_tile_pos->tile_));
          else
            max_overlaps = vertex_matches(s_tile_pos->tile_, n_tile_pos->tile_);

          if (tile_shape == rectangle_str) {
            if (s_col == n_col || s_row == n_row)
              assert(max_overlaps == 2);
            else
              assert(max_overlaps == 1);
          } else if (tile_shape == triangle_str) {
            if (test_edge_wrappings)
              assert(0 < max_overlaps && max_overlaps <= 2);
            else if ((std::abs(s_row - n_row) == 1 &&
                      (even_s_col ? n_col - s_col : s_col - n_col) == 1) ||
                     (s_row == n_row && std::abs(s_col - n_col) == 1))
              assert(max_overlaps == 2);
            else
              assert(max_overlaps == 1);
          } else
            assert(max_overlaps == 2);
        }
      }
    }
  }
}

int main() {
  init_omp();
  TAArray<GFCollection<Coord2D>> gc_array;
  CreatorRegistry<GridTargetPositionShifts<Coord2D>> gscr;
  CreatorRegistry<ShiftedOriginCreator<Coord2D>> socr;
  CreatorRegistry<CachedTileCreator<Coord2D>> ctcr;

  initialize_gsc_registry(gscr);
  initialize_soc_registry(socr);
  initialize_ctc_registry(ctcr);

  for (const auto &tile_shape : {rectangle_str, triangle_str, hexagon_str}) {
    for (angle_t tile_rotation = 0; tile_rotation <= 360; tile_rotation += 15) {
      if ((tile_shape == rectangle_str && tile_rotation % 45 != 0) ||
          (tile_shape == triangle_str && tile_rotation % 180 != 0) ||
          (tile_shape == hexagon_str && tile_rotation % 30 != 0))
        continue;

      for (tileidx_t width = 1; width <= 10; ++width) {
        for (tileidx_t height = 1; height <= 10; ++height) {
          if (((((height + width) > 2) && width % 2 != 0) &&
               tile_shape == triangle_str) ||
              ((((height + width) > 2) && height % 2 != 0) &&
               tile_shape == hexagon_str))
            continue;

          // Above 1e-8 tile projection vector magnitudes are almost zero
          for (uint magnitude = 1; magnitude < 1e8; magnitude *= 10) {
            test_grid(gc_array, gscr, socr, ctcr, width, height, tile_shape,
                      1. / magnitude, tile_rotation);
          }
        }
      }
    }
  }

  return 0;
}
