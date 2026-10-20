/*
 *  timer_manager.h
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

#ifndef TIMER_MANAGER_H
#define TIMER_MANAGER_H

#include <chrono>
#include <string>
#include <unordered_map>

#include "thread_aligned_array.h"

namespace sapi {
struct StopWatch {
  std::chrono::high_resolution_clock::time_point start_;
  std::chrono::nanoseconds duration_;

  void start();

  void stop();

  double time() const;
};

inline void StopWatch::start() {
  start_ = std::chrono::high_resolution_clock::now();
}

inline void StopWatch::stop() {
  duration_ += std::chrono::high_resolution_clock::now() - start_;
}

inline double StopWatch::time() const { return duration_.count() / 1e9; }

struct TimerRegister {
  std::unordered_map<std::string, StopWatch> registered_timers_;

  StopWatch *get_register_timer(std::string &&name);

  std::unordered_map<std::string, double> to_map() const;
};

inline StopWatch *TimerRegister::get_register_timer(std::string &&name) {
  return &registered_timers_[std::move(name)];
}

struct RecordedTimes {
  std::unordered_map<std::string, double> rank_times_;
  std::unordered_map<std::string, std::vector<double>> thread_times_;

  std::string to_string(uint8_t tabs = 0) const;
};

class TimerManager {
public:
  void initialize();

  bool is_initialized() const;

  TimerRegister *get_rank_registry() const;

  TimerRegister *get_thread_registry() const;

  RecordedTimes get_times() const;

protected:
  std::unique_ptr<TimerRegister> rank_registry_;
  TAArray<TimerRegister> thread_registers_;
};

inline void TimerManager::initialize() {
  rank_registry_ = std::make_unique<TimerRegister>();
  thread_registers_.clone(TimerRegister());
}

inline bool TimerManager::is_initialized() const {
  return rank_registry_ && thread_registers_.is_initialized();
}

inline TimerRegister *TimerManager::get_rank_registry() const {
  assert(rank_registry_);
  return rank_registry_.get();
}

inline TimerRegister *TimerManager::get_thread_registry() const {
  return thread_registers_.get_thread_item(get_thread_num());
}
} // namespace sapi

#endif
