#include <cstdio>
#include <random>

#include "random_manager.h"
#include "vp_interface.h"

using namespace sapi;

int main(int argc, char **argv) {
  init_mpi(&argc, &argv);
  init_omp();
  std::string args = "";
  for (int i = 1; i < argc; ++i) {
    args += argv[i];
    args += " ";
  }
  printf("Testing with %i args: %s\n", argc - 1, args.c_str());
  uint32_t seed = 1234;
  if (argc > 1) {
    seed = (uint32_t)std::atoi(argv[1]);
  }

  RandomManager manager;
  assert(!manager.is_initialized());
  manager.set_rng_seed(seed);
  assert(manager.is_initialized());

  std::uniform_real_distribution<space_t> dist;

  printf("Printing 3 random numbers in [0,1) per rank synchronized:\n");
  auto rank_rng = manager.get_rank_synced_rng();
  printf("tid %i: %f %f %f\n", manager.local_rank_, dist(*rank_rng),
         dist(*rank_rng), dist(*rank_rng));

  printf("\nPrinting 3 random numbers in [0,1) per rank individual:\n");
  rank_rng = manager.get_rank_specific_rng();
  printf("tid %i: %f %f %f\n", manager.local_rank_, dist(*rank_rng),
         dist(*rank_rng), dist(*rank_rng));

  printf("\nPrinting 3 random numbers in [0,1) per thread synchronized:\n");
#pragma omp parallel default(shared) firstprivate(dist)
  {
    const auto tid = get_thread_num();
    const auto tid_rng = manager.get_tid_synced_rng(tid);
    printf("tid %i: %f %f %f\n", tid, dist(*tid_rng), dist(*tid_rng),
           dist(*tid_rng));
  }

  printf("\nPrinting 3 random numbers in [0,1) per thread individual:\n");
#pragma omp parallel default(shared) firstprivate(dist)
  {
    const auto tid = get_thread_num();
    const auto tid_rng = manager.get_tid_specific_rng(tid);
    printf("tid %i: %f %f %f\n", tid, dist(*tid_rng), dist(*tid_rng),
           dist(*tid_rng));
  }

  printf(
      "\nPrinting 3 random numbers in [0,1) per thread paired with rank 0:\n");
#pragma omp parallel default(shared) firstprivate(dist)
  {
    const auto tid = get_thread_num();
    const auto d0 = dist(*manager.reseed_rank_paired_rng(tid, 0, false, 0));
    const auto d1 = dist(*manager.reseed_rank_paired_rng(tid, 0, false, 1));
    const auto d2 = dist(*manager.reseed_rank_paired_rng(tid, 0, false, 2));
    printf("tid %i: %f %f %f\n", tid, d0, d1, d2);
  }

  manager.update_rank_paired_seed(0);

  printf("\nPrinting 3 random numbers in [0,1) per thread paired with rank 0 "
         "after update:\n");
#pragma omp parallel default(shared) firstprivate(dist)
  {
    const auto tid = get_thread_num();
    const auto d0 = dist(*manager.reseed_rank_paired_rng(tid, 0, false, 0));
    const auto d1 = dist(*manager.reseed_rank_paired_rng(tid, 0, false, 1));
    const auto d2 = dist(*manager.reseed_rank_paired_rng(tid, 0, false, 2));
    printf("tid %i: %f %f %f\n", tid, d0, d1, d2);
  }

  manager.set_rng_type("fake");
  printf("Changed rng type to fake\n");

  printf("Printing 3 random numbers in [0,1) per rank synchronized:\n");
  rank_rng = manager.get_rank_synced_rng();
  printf("tid %i: %f %f %f\n", manager.local_rank_, dist(*rank_rng),
         dist(*rank_rng), dist(*rank_rng));

  printf("\nPrinting 3 random numbers in [0,1) per rank individual:\n");
  rank_rng = manager.get_rank_specific_rng();
  printf("tid %i: %f %f %f\n", manager.local_rank_, dist(*rank_rng),
         dist(*rank_rng), dist(*rank_rng));

  printf("\nPrinting 3 random numbers in [0,1) per thread synchronized:\n");
#pragma omp parallel default(shared) firstprivate(dist)
  {
    const auto tid = get_thread_num();
    const auto tid_rng = manager.get_tid_synced_rng(tid);
    printf("tid %i: %f %f %f\n", tid, dist(*tid_rng), dist(*tid_rng),
           dist(*tid_rng));
  }

  printf("\nPrinting 3 random numbers in [0,1) per thread individual:\n");
#pragma omp parallel default(shared) firstprivate(dist)
  {
    const auto tid = get_thread_num();
    const auto tid_rng = manager.get_tid_specific_rng(tid);
    printf("tid %i: %f %f %f\n", tid, dist(*tid_rng), dist(*tid_rng),
           dist(*tid_rng));
  }

  printf(
      "\nPrinting 3 random numbers in [0,1) per thread paired with rank 0:\n");
#pragma omp parallel default(shared) firstprivate(dist)
  {
    const auto tid = get_thread_num();
    const auto d0 = dist(*manager.reseed_rank_paired_rng(tid, 0, false, 0));
    const auto d1 = dist(*manager.reseed_rank_paired_rng(tid, 0, false, 1));
    const auto d2 = dist(*manager.reseed_rank_paired_rng(tid, 0, false, 2));
    printf("tid %i: %f %f %f\n", tid, d0, d1, d2);
  }

  manager.update_rank_paired_seed(0);

  printf("\nPrinting 3 random numbers in [0,1) per thread paired with rank 0 "
         "after update:\n");
#pragma omp parallel default(shared) firstprivate(dist)
  {
    const auto tid = get_thread_num();
    const auto d0 = dist(*manager.reseed_rank_paired_rng(tid, 0, false, 0));
    const auto d1 = dist(*manager.reseed_rank_paired_rng(tid, 0, false, 1));
    const auto d2 = dist(*manager.reseed_rank_paired_rng(tid, 0, false, 2));
    printf("tid %i: %f %f %f\n", tid, d0, d1, d2);
  }

  finalize_mpi();
  return 0;
}
