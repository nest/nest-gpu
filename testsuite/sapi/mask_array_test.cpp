#include "coordinate_geometry.h"
#include "vp_interface.h"

#include "mask2d_geometry.h"
#include "mask_collection.h"
#include "thread_aligned_array.h"
#include "tile.h"
#include "type_erasure_helpers.h"

using namespace sapi;

int main() {
  init_omp();

  Mask<Coord2D> cb =
      construct_mask<Coord2D>(MASK_SHAPE::CIRCULAR, {}, {1.0}, {});
  MaskCollection<Coord2D> mc;
  mc.blueprint_ = cb;
  TAArray<MaskCollection<Coord2D>> mc_array;
  mc_array.clone(mc);

  assert(mc_array.is_initialized() && mc.has_blueprint() &&
         !mc.has_source_mask() && !mc.has_target_mask());

#pragma omp parallel default(shared)
  {
    const auto mc = mc_array.get_local_thread_item();
    assert(mc->has_blueprint() && !mc->has_source_mask() &&
           !mc->has_target_mask());
  }

  mc = MaskCollection<Coord2D>();
  mc.source_mask_ = cb;
  mc_array.clone(mc);

  assert(mc_array.is_initialized() && !mc.has_blueprint() &&
         mc.has_source_mask() && !mc.has_target_mask());

#pragma omp parallel default(shared)
  {
    const auto mc = mc_array.get_local_thread_item();
    assert(!mc->has_blueprint() && mc->has_source_mask() &&
           !mc->has_target_mask());
  }

  mc = MaskCollection<Coord2D>();
  mc.target_mask_ = cb;
  mc_array.clone(mc);

  assert(mc_array.is_initialized() && !mc.has_blueprint() &&
         !mc.has_source_mask() && mc.has_target_mask());

#pragma omp parallel default(shared)
  {
    const auto mc = mc_array.get_local_thread_item();
    assert(!mc->has_blueprint() && !mc->has_source_mask() &&
           mc->has_target_mask());
  }

  return 0;
}
