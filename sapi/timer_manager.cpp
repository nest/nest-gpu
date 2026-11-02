/*
 *  timer_manager.cpp
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

#include "timer_manager.h"
#include "vp_interface.h"

namespace sapi {
std::unordered_map<std::string, double> TimerRegister::to_map() const {
  std::unordered_map<std::string, double> map;
  for (const auto &[name, timer] : registered_timers_)
    map[name] = timer.time();
  return map;
}

std::string RecordedTimes::to_string(uint8_t tabs) const {
  std::string res = "";
  std::string t_str = "";
  for (; tabs > 0; --tabs)
    t_str += "\t";
  for (const auto &[name, time] : rank_times_)
    res += t_str + name + ": " + std::to_string(time) + "\n";
  for (const auto &[name, times] : thread_times_) {
    res += t_str + name + ": [ ";
    for (const auto &time : times) {
      res += std::to_string(time) + ", ";
    }
    res += " ]\n";
  }
  return res;
}

RecordedTimes TimerManager::get_times() const {
  RecordedTimes times;

  if (!is_initialized())
    return times;

  times.rank_times_ = rank_registry_->to_map();

  std::vector<std::unordered_map<std::string, double>> thread_times(
      get_max_omp_threads());

#pragma omp parallel default(none) shared(thread_times, thread_registers_)
  {
    const auto tid = get_thread_num();
    thread_times[tid] = thread_registers_.get_thread_item(tid)->to_map();
  }

  vp_t tid = 0;
  for (const auto &tts : thread_times) {
    for (const auto &[name, time] : tts) {
      auto &vec = times.thread_times_[name];
      if (vec.empty())
        vec.resize(thread_times.size(), 0.);
      vec[tid] = time;
    }
    ++tid;
  }

  return times;
}
} // namespace sapi
