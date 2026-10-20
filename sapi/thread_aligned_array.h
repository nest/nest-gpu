/*
 *  thread_aligned_array.h
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

#ifndef THREAD_ALIGNED_ARRAY_H
#define THREAD_ALIGNED_ARRAY_H

#include <cassert>
#include <memory>
#include <type_traits>
#include <vector>

#include "sapi_config.h"

namespace sapi {
// Forward definition to link with vp_interface
vp_t get_thread_num();
vp_t get_max_omp_threads();

template <typename T, typename std::enable_if_t<std::is_copy_constructible_v<T>,
                                                bool> = true>
class TAArray {
public:
  bool is_initialized() const;

  void clear();
  void prepare();
  void clone(const T &item);

  T *get_thread_item(const vp_t tid) const;
  const T *get_local_thread_item() const;

protected:
  bool cloned_ = false;
  bool prepared_ = false;
  std::vector<std::unique_ptr<T>> item_vec_;
};

template <typename T,
          typename std::enable_if_t<std::is_copy_constructible_v<T>, bool> b>
inline void TAArray<T, b>::clear() {
  item_vec_.clear();
}

template <typename T,
          typename std::enable_if_t<std::is_copy_constructible_v<T>, bool> b>
inline bool TAArray<T, b>::is_initialized() const {
  assert(cloned_ <= prepared_);
  ;
  return cloned_;
}

template <typename T,
          typename std::enable_if_t<std::is_copy_constructible_v<T>, bool> b>
inline void TAArray<T, b>::prepare() {
  if (!prepared_ || cloned_ ||
      item_vec_.size() != static_cast<std::size_t>(get_max_omp_threads())) {
    item_vec_.clear();
    item_vec_.resize(get_max_omp_threads());
    prepared_ = true;
    cloned_ = false;
  }
}

template <typename T,
          typename std::enable_if_t<std::is_copy_constructible_v<T>, bool> b>
inline void TAArray<T, b>::clone(const T &item) {
  prepare();
  assert(!cloned_ && prepared_);

#pragma omp parallel default(none) shared(item_vec_, item)
  item_vec_.at(get_thread_num()) = std::make_unique<T>(item);

  cloned_ = true;
}

template <typename T,
          typename std::enable_if_t<std::is_copy_constructible_v<T>, bool> b>
inline T *TAArray<T, b>::get_thread_item(const vp_t tid) const {
  return item_vec_.at(tid).get();
}

template <typename T,
          typename std::enable_if_t<std::is_copy_constructible_v<T>, bool> b>
inline const T *TAArray<T, b>::get_local_thread_item() const {
  return item_vec_.at(get_thread_num()).get();
}
} // namespace sapi

#endif
