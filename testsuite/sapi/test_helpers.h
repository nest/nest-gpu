#ifndef TEST_HELPERS_H
#define TEST_HELPERS_H

#include <cassert>
#include <cstring>
#include <limits>
#include <sys/resource.h>
#include <type_traits>
#include <vector>

#include "sapi_config.h"

#ifdef HAVE_MPI
#include <mpi.h>
#endif

namespace sapi {
long get_mem_usage() {
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  return usage.ru_maxrss; // in KB
}

template <typename Iterator>
inline std::pair<Iterator, bool> &&
assert_emplace(std::pair<Iterator, bool> &&res) {
  assert(res.second);
  return std::forward<std::pair<Iterator, bool>>(res);
}

template <typename I,
          typename std::enable_if_t<std::is_arithmetic_v<I>, bool> = true>
int coordinated_all_gatherv_as_ul(const std::vector<I> &send_buff,
                                  std::vector<I> &recv_buff,
                                  std::vector<int> &recv_counts,
                                  std::vector<int> &displacements) {
#ifdef HAVE_MPI
  int initialized;
  MPI_Initialized(&initialized);
  assert(static_cast<bool>(initialized) && recv_buff.empty() &&
         recv_counts.empty() && displacements.empty());

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  recv_counts.resize(world_size);
  assert(send_buff.size() < std::numeric_limits<int>::max());
  const int data_length = static_cast<int>(send_buff.size());
  MPI_Allgather(&data_length, 1, MPI_INT, recv_counts.data(), 1, MPI_INT,
                MPI_COMM_WORLD);

  displacements.resize(world_size, 0);
  for (int i = 1; i < world_size; ++i)
    displacements[i] = displacements[i - 1] + recv_counts[i - 1];

  std::vector<uint64_t> send_ul_data(data_length);
  auto send_it = send_ul_data.begin();
  for (const auto &data : send_buff)
    std::memcpy(&(*send_it++), &data, sizeof(I));

  std::vector<uint64_t> recv_ul_data(displacements[world_size - 1] +
                                     recv_counts[world_size - 1]);

  MPI_Allgatherv(send_ul_data.data(), data_length, MPI_UINT64_T,
                 recv_ul_data.data(), recv_counts.data(), displacements.data(),
                 MPI_UINT64_T, MPI_COMM_WORLD);

  recv_buff.resize(recv_ul_data.size());
  auto recv_it = recv_buff.begin();
  for (const auto &data : recv_ul_data)
    std::memcpy(&(*recv_it++), &data, sizeof(I));

  return world_size;
#endif
  return 0;
}

template <typename I,
          typename std::enable_if_t<std::is_arithmetic_v<I>, bool> = true>
void assert_equal_across_ranks(const std::vector<I> &send_buff) {
#ifdef HAVE_MPI
  std::vector<I> recv_buff;
  std::vector<int> recv_counts;
  std::vector<int> displacements;
  const auto world_size = coordinated_all_gatherv_as_ul(
      send_buff, recv_buff, recv_counts, displacements);

  const auto recv0 = recv_counts[0];
  const auto disp0 = displacements[0];
  for (int rank = 1; rank < world_size; ++rank) {
    assert(recv0 == recv_counts[rank]);
    const auto disp1 = displacements[rank];
    for (int count = 0; count < recv0; ++count)
      assert(recv_buff[disp0 + count] == recv_buff[disp1 + count]);
  }
#endif
}
} // namespace sapi

#endif
