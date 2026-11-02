#include <chrono>
#include <iomanip>
#include <iostream>

#include "vp_interface.h"

#include "node_containers.h"
#include "random_manager.h"

using namespace sapi;

#define NODES 100

NodeCountVector while_parallel(const nodeidx_t num_nodes,
                               const tileidx_t num_tiles,
                               const RandomManager *const rng_manager) {
  assert(rng_manager != nullptr && rng_manager->is_initialized() &&
         num_nodes > 0 && num_tiles > 0);

  // Trivial case
  if (num_tiles == 1)
    return NodeCountVector{num_nodes};

  // Initialize map and rng distribution
  NodeCountVector nc_vec(num_tiles, 0);
  std::uniform_int_distribution<tileidx_t> tile_idx_dist(0, num_tiles - 1);

  // Batch tile index drawing with steps equal to the total number of threads
  const int num_threads = static_cast<int>(get_max_omp_threads());
  int full_batches, partial_batches;

  // Only batch if running in multithreaded environment
  if (num_threads > 1) {
    const auto batch_steps = std::div(static_cast<int>(num_nodes), num_threads);
    full_batches = batch_steps.quot;
    partial_batches = batch_steps.rem;
  } else {
    full_batches = 0;
    partial_batches = num_nodes;
  }
  assert((full_batches > 0) || (partial_batches > 0));

  if (full_batches > 0) {
    std::vector<uint> batch_vector(num_threads);
    uint *batch_buffer = batch_vector.data();

    // Batch execution
    while (0 < full_batches--) {

      // Parallel drawing
#pragma omp parallel default(none)                                             \
    firstprivate(batch_buffer, tile_idx_dist, rng_manager)
      {
        const auto tid = get_thread_num();
        const auto rng = rng_manager->get_tid_specific_rng(tid);
        batch_buffer[tid] = tile_idx_dist(*rng);
      }

      // Sequential writing to avoid collisions due to rng
      for (vp_t tid = 0; tid < num_threads; ++tid)
        ++nc_vec[batch_buffer[tid]];
    }
  }

  if (partial_batches > 0) {
    const auto generator = rng_manager->get_rank_synced_rng();
    while (0 < partial_batches--)
      ++nc_vec[tile_idx_dist(*generator)];
  }

  return nc_vec;
}

NodeCountVector parallel_while(const nodeidx_t num_nodes,
                               const tileidx_t num_tiles,
                               const RandomManager *const rng_manager) {
  assert(rng_manager != nullptr && rng_manager->is_initialized() &&
         num_nodes > 0 && num_tiles > 0);

  // Trivial case
  if (num_tiles == 1)
    return NodeCountVector{num_nodes};

  // Initialize map and rng distribution
  NodeCountVector nc_vec(num_tiles, 0);
  std::uniform_int_distribution<tileidx_t> tile_idx_dist(0, num_tiles - 1);

  // Batch tile index drawing with steps equal to the total number of threads
  const int num_threads = static_cast<int>(get_max_omp_threads());
  int full_batches, partial_batches;

  // Only batch if running in multithreaded environment
  if (num_threads > 1) {
    const auto batch_steps = std::div(static_cast<int>(num_nodes), num_threads);
    full_batches = batch_steps.quot;
    partial_batches = batch_steps.rem;
  } else {
    full_batches = 0;
    partial_batches = num_nodes;
  }
  assert((full_batches > 0) || (partial_batches > 0));

  if (full_batches > 0) {

    // Batch execution
#pragma omp parallel default(none) shared(nc_vec)                              \
    firstprivate(tile_idx_dist, rng_manager, full_batches)
    {
      const auto rng = rng_manager->get_tid_specific_rng(get_thread_num());
      while (0 < full_batches--) {
        // Sequential writing to avoid collisions due to rng
#pragma omp critical
        ++nc_vec[tile_idx_dist(*rng)];
      }
    }
  }

  if (partial_batches > 0) {
    const auto generator = rng_manager->get_rank_synced_rng();
    while (0 < partial_batches--)
      ++nc_vec[tile_idx_dist(*generator)];
  }

  return nc_vec;
}

// Resulting length is equal to number of tiles in argument
NodeCountVector sequential_control(const nodeidx_t num_nodes,
                                   const tileidx_t num_tiles,
                                   const RandomManager *const rng_manager,
                                   const bool batched = false) {
  assert(rng_manager != nullptr && rng_manager->is_initialized() &&
         num_nodes > 0 && num_tiles > 0);

  // Trivial case
  if (num_tiles == 1)
    return NodeCountVector{num_nodes};

  // Initialize map and rng distribution
  NodeCountVector nc_vec(num_tiles, 0);
  std::uniform_int_distribution<tileidx_t> tile_idx_dist(0, num_tiles - 1);

  // Batch tile index drawing with steps equal to the total number of threads
  const int num_threads = static_cast<int>(get_max_omp_threads());
  int full_batches, partial_batches;

  // Only batch if running in multithreaded environment
  if (num_threads > 1 && batched) {
    const auto batch_steps = std::div(static_cast<int>(num_nodes), num_threads);
    full_batches = batch_steps.quot;
    partial_batches = batch_steps.rem;
  } else {
    full_batches = 0;
    partial_batches = num_nodes;
  }
  assert((full_batches > 0) || (partial_batches > 0));

  if (full_batches > 0) {
    std::vector<uint> batch_vector(num_threads);
    uint *batch_buffer = batch_vector.data();

    // Batch execution
    while (0 < full_batches--) {

      // Parallel drawing
#pragma omp parallel default(none)                                             \
    firstprivate(batch_buffer, tile_idx_dist, rng_manager)
      {
        const auto tid = get_thread_num();
        const auto rng = rng_manager->get_tid_specific_rng(tid);
        batch_buffer[tid] = tile_idx_dist(*rng);
      }

      // Sequential writing to avoid collisions due to rng
      for (vp_t tid = 0; tid < num_threads; ++tid)
        ++nc_vec[batch_buffer[tid]];
    }
  }

  if (partial_batches > 0) {
    const auto generator = rng_manager->get_rank_synced_rng();
    while (0 <= --partial_batches)
      ++nc_vec[tile_idx_dist(*generator)];
  }

  return nc_vec;
}

// Resulting length is equal to number of tiles in argument
NodeCountVector parallel_batches(const nodeidx_t num_nodes,
                                 const tileidx_t num_tiles,
                                 const RandomManager *const rng_manager,
                                 const bool batched = false) {
  assert(rng_manager != nullptr && rng_manager->is_initialized() &&
         num_nodes > 0 && num_tiles > 0);

  // Trivial case
  if (num_tiles == 1)
    return NodeCountVector{num_nodes};

  // Initialize map and rng distribution
  NodeCountVector nc_vec(num_tiles, 0);
  std::uniform_int_distribution<tileidx_t> tile_idx_dist(0, num_tiles - 1);

  // Batch tile index drawing with steps equal to the total number of threads
  const int num_threads = static_cast<int>(get_max_omp_threads());
  int full_batches, partial_batches;

  // Only batch if running in multithreaded environment
  if (num_threads > 1 && batched) {
    const auto batch_steps = std::div(static_cast<int>(num_nodes), num_threads);
    full_batches = batch_steps.quot;
    partial_batches = batch_steps.rem;
  } else {
    full_batches = 0;
    partial_batches = num_nodes;
  }
  assert((full_batches > 0) || (partial_batches > 0));

  if (full_batches > 0) {
    std::vector<std::vector<uint>> batch_vector(num_threads);
    auto batch_buffer = batch_vector.data();
    // Parallel drawing
#pragma omp parallel default(none)                                             \
    firstprivate(batch_buffer, tile_idx_dist, rng_manager, full_batches)
    {
      const auto tid = get_thread_num();
      const auto rng = rng_manager->get_tid_specific_rng(tid);
      batch_buffer[tid].resize(full_batches);
      auto tid_buffer = batch_buffer[tid].data();

      // Batch execution
      while (0 <= --full_batches)
        tid_buffer[full_batches] = tile_idx_dist(*rng);
    }

    for (int tid = 0; tid < num_threads; ++tid) {
      const auto tid_buffer = batch_buffer[tid];
      for (int batch = 0; batch < full_batches; ++batch) {
        ++nc_vec[tid_buffer[batch]];
      }
    }
  }

  if (partial_batches > 0) {
    const auto generator = rng_manager->get_rank_synced_rng();
    while (0 <= --partial_batches)
      ++nc_vec[tile_idx_dist(*generator)];
  }

  return nc_vec;
}

void batched_draw(nodeidx_t *buffer, int batches,
                  std::uniform_int_distribution<tileidx_t> &dist, AnyRNG &rng) {
  while (0 <= --batches)
    ++buffer[dist(rng)];
}

// Resulting length is equal to number of tiles in argument
NodeCountVector parallel_batches_v2(const nodeidx_t num_nodes,
                                    const tileidx_t num_tiles,
                                    const RandomManager *const rng_manager,
                                    const bool batched = false) {
  assert(rng_manager != nullptr && rng_manager->is_initialized() &&
         num_nodes > 0 && num_tiles > 0);

  // Trivial case
  if (num_tiles == 1)
    return NodeCountVector{num_nodes};

  // Initialize map and rng distribution
  NodeCountVector nc_vec(num_tiles, 0);
  std::uniform_int_distribution<tileidx_t> tile_idx_dist(0, num_tiles - 1);

  // Batch tile index drawing with steps equal to the total number of threads
  const int num_threads = static_cast<int>(get_max_omp_threads());
  int full_batches, partial_batches;

  // Only batch if running in multithreaded environment
  if (num_threads > 1 && batched) {
    const auto batch_steps = std::div(static_cast<int>(num_nodes), num_threads);
    full_batches = batch_steps.quot;
    partial_batches = batch_steps.rem;
  } else {
    full_batches = 0;
    partial_batches = num_nodes;
  }
  assert((full_batches > 0) || (partial_batches > 0));

  if (full_batches > 0) {
    std::vector<NodeCountVector> batch_buffer(num_threads);
    NodeCountVector *bb_ptr = batch_buffer.data();
    // Parallel drawing
#pragma omp parallel default(none)                                             \
    firstprivate(bb_ptr, tile_idx_dist, rng_manager, full_batches, num_tiles,  \
                     num_threads)
    {
      const auto tid = get_thread_num();
      const auto rng = rng_manager->get_tid_specific_rng(tid);
      bb_ptr[tid].resize(num_tiles);
      nodeidx_t *tid_buffer = bb_ptr[tid].data();

      batched_draw(tid_buffer, full_batches, tile_idx_dist, *rng);
#pragma omp barrier

      vp_t next_tid;
      vp_t step = 1;
      tileidx_t tix;
      nodeidx_t *next_buffer;
      bool merge = true;
      do {
        if (merge) {
          next_tid = tid + step;
          if ((tid / step) % 2 == 0 && next_tid < num_threads) {
            next_buffer = bb_ptr[next_tid].data();
            for (tix = 0; tix < num_tiles; ++tix)
              tid_buffer[tix] += next_buffer[tix];
          } else
            merge = false;
        }
        step *= 2;
#pragma omp barrier
      } while (step < num_threads);
    }
    nc_vec.swap(bb_ptr[0]);
  }

  if (partial_batches > 0) {
    const auto generator = rng_manager->get_rank_synced_rng();
    batched_draw(nc_vec.data(), partial_batches, tile_idx_dist, *generator);
  }

  return nc_vec;
}

// Resulting length is equal to number of tiles in argument
NodeCountVector task_parallel_batches(const nodeidx_t num_nodes,
                                      const tileidx_t num_tiles,
                                      const RandomManager *const rng_manager,
                                      const bool global) {
  assert(rng_manager != nullptr && rng_manager->is_initialized() &&
         num_nodes > 0 && num_tiles > 0);

  // Trivial case
  if (num_tiles == 1)
    return NodeCountVector{num_nodes};

  // Initialize map and rng distribution
  NodeCountVector nc_vec(num_tiles, 0);
  std::uniform_int_distribution<tileidx_t> tile_idx_dist(0, num_tiles - 1);

  // Batch tile index drawing with steps equal to the total number of threads
  const int num_threads = static_cast<int>(get_max_omp_threads());
  int full_batches, partial_batches;

  // Only batch if running in multithreaded environment
  if (num_threads > 1) {
    const auto batch_steps = std::div(static_cast<int>(num_nodes), num_threads);
    full_batches = batch_steps.quot;
    partial_batches = batch_steps.rem;
    if (full_batches < BATCHING_THRESHOLD) {
      full_batches = 0;
      partial_batches = num_nodes;
    }
  } else {
    full_batches = 0;
    partial_batches = num_nodes;
  }
  assert((full_batches > 0) || (partial_batches > 0));

  if (full_batches > 0) {
    std::vector<NodeCountVector> batch_vector(num_threads);
    NodeCountVector *batch_buffer = batch_vector.data();
    int curr_idx, next_idx, step = 1, step2 = 2;
    nodeidx_t *curr_buffer, *next_buffer;
    tileidx_t tix;

#pragma omp taskloop num_tasks(strict : num_threads) default(none) private(    \
        curr_idx, curr_buffer)                                                 \
    firstprivate(num_threads, num_tiles, batch_buffer, global, rng_manager,    \
                     full_batches, tile_idx_dist)
    for (curr_idx = 0; curr_idx < num_threads; ++curr_idx) {
      batch_buffer[curr_idx].resize(num_tiles);
      curr_buffer = batch_buffer[curr_idx].data();
      const auto rng = global ? rng_manager->get_tid_synced_rng(curr_idx)
                              : rng_manager->get_tid_specific_rng(curr_idx);

      batched_draw(curr_buffer, full_batches, tile_idx_dist, *rng);
    }

    do {
#pragma omp taskloop num_tasks(strict : num_threads) default(none) private(    \
        curr_idx, next_idx, curr_buffer, next_buffer, tix)                     \
    firstprivate(num_tiles, num_threads, step, step2, batch_buffer)
      for (curr_idx = 0; curr_idx < num_threads; curr_idx += step2) {
        next_idx = curr_idx + step;
        if (next_idx < num_threads) {
          curr_buffer = batch_buffer[curr_idx].data();
          next_buffer = batch_buffer[next_idx].data();
          for (tix = 0; tix < num_tiles; ++tix)
            curr_buffer[tix] += next_buffer[tix];
        }
      }
      step = step2;
      step2 *= 2;
    } while (step < num_threads);

    nc_vec.swap(batch_buffer[0]);
  }

  if (partial_batches > 0) {
    const auto generator = global ? rng_manager->get_rank_synced_rng()
                                  : rng_manager->get_rank_specific_rng();
    batched_draw(nc_vec.data(), partial_batches, tile_idx_dist, *generator);
  }

  return nc_vec;
}

#define ITERATIONS 10
#define NUM_NODES 1'000'000'000
#define NUM_TILES 24'000

int main(int argc, char **argv) {
  init_mpi(&argc, &argv);
  init_omp(10);
  RandomManager rman;
  rman.set_rng_seed(12345u);

  std::cout << std::fixed << std::setprecision(10) << std::left;

  const bool configurations[6] = {false, false, false, false, true, true};
  for (int i = 0; i < ITERATIONS; ++i) {
    std::cout << "Iteration " << i << ":\n";

    if (configurations[0]) {
      const auto start = std::chrono::high_resolution_clock::now();
      const auto res = while_parallel(NUM_NODES, NUM_TILES, &rman);
      const auto end = std::chrono::high_resolution_clock::now();
      const auto diff = end - start;
      std::cout << "While parallel: " << static_cast<double>(diff.count()) / 1e9
                << "s\n";
      assert(res.size() == NUM_TILES);
      nodeidx_t count = 0;
      for (nodeidx_t idx = 0; idx < NUM_TILES; ++idx)
        count += res[idx];
      assert(count == NUM_NODES);
    }

    if (configurations[1]) {
      const auto start = std::chrono::high_resolution_clock::now();
      const auto res = parallel_while(NUM_NODES, NUM_TILES, &rman);
      const auto end = std::chrono::high_resolution_clock::now();
      const auto diff = end - start;
      std::cout << "Parallel while: " << static_cast<double>(diff.count()) / 1e9
                << "s\n";
      assert(res.size() == NUM_TILES);
      nodeidx_t count = 0;
      for (nodeidx_t idx = 0; idx < NUM_TILES; ++idx)
        count += res[idx];
      assert(count == NUM_NODES);
    }

    if (configurations[2]) {
      const auto start = std::chrono::high_resolution_clock::now();
      const auto res = sequential_control(NUM_NODES, NUM_TILES, &rman, false);
      const auto end = std::chrono::high_resolution_clock::now();
      const auto diff = end - start;
      std::cout << "Sequential: " << static_cast<double>(diff.count()) / 1e9
                << "s\n";
      assert(res.size() == NUM_TILES);
      nodeidx_t count = 0;
      for (nodeidx_t idx = 0; idx < NUM_TILES; ++idx)
        count += res[idx];
      assert(count == NUM_NODES);
    }

    if (configurations[3]) {
      const auto start = std::chrono::high_resolution_clock::now();
      const auto res = parallel_batches(NUM_NODES, NUM_TILES, &rman, true);
      const auto end = std::chrono::high_resolution_clock::now();
      const auto diff = end - start;
      std::cout << "Parallel batches: "
                << static_cast<double>(diff.count()) / 1e9 << "s\n";
      assert(res.size() == NUM_TILES);
      nodeidx_t count = 0;
      for (nodeidx_t idx = 0; idx < NUM_TILES; ++idx)
        count += res[idx];
      assert(count == NUM_NODES);
    }

    if (configurations[4]) {
      const auto start = std::chrono::high_resolution_clock::now();
      const auto res = parallel_batches_v2(NUM_NODES, NUM_TILES, &rman, true);
      const auto end = std::chrono::high_resolution_clock::now();
      const auto diff = end - start;
      std::cout << "Parallel batches V2: "
                << static_cast<double>(diff.count()) / 1e9 << "s\n";
      assert(res.size() == NUM_TILES);
      nodeidx_t count = 0;
      for (nodeidx_t idx = 0; idx < NUM_TILES; ++idx)
        count += res[idx];
      assert(count == NUM_NODES);
    }

    if (configurations[5]) {
      const auto start = std::chrono::high_resolution_clock::now();
      NodeCountVector res;
#pragma omp parallel default(shared)
#pragma omp master
      res = task_parallel_batches(NUM_NODES, NUM_TILES, &rman, true);
      const auto end = std::chrono::high_resolution_clock::now();
      const auto diff = end - start;
      std::cout << "Task parallel batches: "
                << static_cast<double>(diff.count()) / 1e9 << "s\n";
      assert(res.size() == NUM_TILES);
      nodeidx_t count = 0;
      for (nodeidx_t idx = 0; idx < NUM_TILES; ++idx)
        count += res[idx];
      assert(count == NUM_NODES);
    }
  }

  finalize_mpi();
  return 0;
}
