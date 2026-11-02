/*
 *  random_manager.h
 *
 *  This file is part of NEST GPU.
 *
 *  Copyright (C) 2021 The NEST Initiative
 *
 *  NEST GPU is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST GPU is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST GPU.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef RANDOM_MANAGER_H
#define RANDOM_MANAGER_H

#include "creator_registry.h"
#include "thread_aligned_array.h"
#include "type_erasure_helpers.h"

namespace sapi {
class RandomManager {
public:
  const rng_seed_t local_rank_;
  const rng_seed_t num_processes_;

  RandomManager();
  RandomManager(const RandomManager &) = delete;
  RandomManager(RandomManager &&) noexcept = default;
  ~RandomManager() noexcept = default;

  RandomManager(const vp_t local_rank, const vp_t num_processes);

  RandomManager &operator=(const RandomManager &) = delete;
  RandomManager &operator=(RandomManager &&) = delete;

  void initialize();

  bool is_initialized() const;

  AnyRNG *get_rank_synced_rng() const;
  AnyRNG *get_rank_specific_rng() const;
  AnyRNG *get_tid_synced_rng(const vp_t tid) const;
  AnyRNG *get_tid_specific_rng(const vp_t tid) const;

  AnyRNG *reseed_rank_paired_rng(const rng_seed_t tile_index,
                                 const rng_seed_t leaf_index) const;

  AnyRNG *reseed_rank_paired_rng(const vp_t tid, const rng_seed_t rank,
                                 const bool inverted_source_target,
                                 const combined_idx_t combined_seed) const;

  void update_rank_paired_seed(const vp_t rank) const;

  void set_rng_seed(const rng_seed_t seed);

  void set_rng_type(const std::string &name);

protected:
  CreatorRegistry<AnyRNG> rng_registry_;
  StateLessCreator<AnyRNG> *current_creator_;
  std::string current_rng_type_ = DEFAULT_RNG_TYPE_;

  std::unique_ptr<AnyRNG> rank_synced_rng_;
  std::unique_ptr<AnyRNG> rank_specific_rng_;
  TAArray<AnyRNG> tid_synced_rng_vec_;
  TAArray<AnyRNG> tid_specific_rng_vec_;
  TAArray<AnyRNG> tid_rank_paired_rng_vec_;

  rng_seed_t base_seed_ = DEFAULT_BASE_SEED_;
  mutable std::vector<rng_seed_t> rank_paired_seeds_;
};

inline bool RandomManager::is_initialized() const {
  return rank_synced_rng_ && rank_specific_rng_ &&
         tid_synced_rng_vec_.is_initialized() &&
         tid_specific_rng_vec_.is_initialized() &&
         tid_rank_paired_rng_vec_.is_initialized();
}

inline AnyRNG *RandomManager::get_rank_synced_rng() const {
  assert(rank_synced_rng_);
  return rank_synced_rng_.get();
}

inline AnyRNG *RandomManager::get_rank_specific_rng() const {
  assert(rank_specific_rng_);
  return rank_specific_rng_.get();
}

inline AnyRNG *RandomManager::get_tid_synced_rng(const vp_t tid) const {
  return tid_synced_rng_vec_.get_thread_item(tid);
}

inline AnyRNG *RandomManager::get_tid_specific_rng(const vp_t tid) const {
  return tid_specific_rng_vec_.get_thread_item(tid);
}

inline AnyRNG *
RandomManager::reseed_rank_paired_rng(const rng_seed_t tile_index,
                                      const rng_seed_t leaf_index) const {
  auto rng = tid_rank_paired_rng_vec_.get_thread_item(get_thread_num());
  rng->seed({base_seed_, THREAD_SEEDER_, PARITY_SEEDER_,
             rank_paired_seeds_.at(local_rank_), tile_index, leaf_index});
  return rng;
}

inline AnyRNG *RandomManager::reseed_rank_paired_rng(
    const vp_t tid, const rng_seed_t rank, const bool inverted_source_target,
    const combined_idx_t combined_seed) const {
  auto rng = tid_rank_paired_rng_vec_.get_thread_item(tid);
  if (inverted_source_target)
    rng->seed({
        base_seed_, THREAD_SEEDER_, PARITY_SEEDER_, rank, local_rank_,
        rank_paired_seeds_.at(rank),
        static_cast<rng_seed_t>(combined_seed >> 32),
        static_cast<rng_seed_t>(combined_seed &
                                4294967295ul) // ( 1ul << 32 ) - 1ul
    });
  else
    rng->seed({
        base_seed_, THREAD_SEEDER_, PARITY_SEEDER_, local_rank_, rank,
        rank_paired_seeds_.at(rank),
        static_cast<rng_seed_t>(combined_seed >> 32),
        static_cast<rng_seed_t>(combined_seed &
                                4294967295ul) // ( 1ul << 32 ) - 1ul
    });

  return rng;
}

inline void RandomManager::update_rank_paired_seed(const vp_t rank) const {
  ++rank_paired_seeds_.at(rank);
}

inline void RandomManager::set_rng_seed(const rng_seed_t seed) {
  base_seed_ = seed;
  initialize();
}

inline void RandomManager::set_rng_type(const std::string &name) {
  current_creator_ = rng_registry_.get_creator(name).get();
  current_rng_type_ = name;
  initialize();
}
} // namespace sapi

#endif
