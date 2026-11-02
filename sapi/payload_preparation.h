/*
 *  payload_preparation.h
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

#ifndef PAYLOAD_PREPARATION_H
#define PAYLOAD_PREPARATION_H

#include <stdexcept>

#include "mask_containers.h"
#include "numerics.h"

namespace sapi {
inline void header_write(std::vector<uint64_t>::iterator &payload_it,
                         const uint64_t item, const int32_t index) {
  if (evenTix(index)) {
    *payload_it = item << 32;
  } else {
    (*payload_it++) |= item;
  }
}

inline int32_t header_read(std::vector<uint64_t>::const_iterator &payload_it,
                           const int32_t index) {
  if (evenTix(index)) {
    return *payload_it >> 32;
  } else {
    return (*payload_it++) & 4294967295ul; // ( 1ul << 32 ) - 1ul
  }
}

template <typename CoordT>
void prepare_data_payload(CommunicationInfo<CoordT> &comm_info,
                          const TileSetInfo<CoordT> &tile_set_info) {
  static_assert(sizeof(tileidx_t) == 4ul && sizeof(nodeidx_t) == 4ul);

  // Compute total data length
  const int32_t _32bit_header_size =
      3 + 2 * (tile_set_info.valid_tiles_ + tile_set_info.valid_leaves_) +
      tile_set_info.total_node_count_;

  // compact for 64bit
  const int32_t total_header_size =
      (_32bit_header_size + _32bit_header_size % 2) / 2;
  const int32_t total_content_size =
      tile_set_info.total_node_count_ * CoordT::D;
  const int32_t total_payload_size = total_header_size + total_content_size;

  // Check overflow
  if (!(2 <= total_header_size && 0 <= total_content_size &&
        2 <= total_payload_size))
    throw std::runtime_error("Could not compute total payload size");

  assert((2 < total_payload_size) != comm_info.filtered_coords_.empty());

  // Prepare payload buffer
  comm_info.data_payload_.resize(total_payload_size);

  // Prepare writing iterators
  auto info_writing_pos = comm_info.data_payload_.begin();
  const auto info_writing_end =
      comm_info.data_payload_.begin() + total_header_size;
  auto content_writing_pos = info_writing_end;
  const auto content_writing_end = comm_info.data_payload_.end();

  // Write fixed header metadata
  int32_t header_pos = 0;
  header_write(info_writing_pos, total_header_size, header_pos++);
  header_write(info_writing_pos, total_content_size, header_pos++);
  header_write(info_writing_pos, tile_set_info.valid_tiles_, header_pos++);

  for (const auto &[tile_index, leaf_ncm] : comm_info.filtered_coords_) {
    assert(!leaf_ncm.empty());

    // Write tile index and total number of sub tiles in tile
    header_write(info_writing_pos, tile_index, header_pos++);
    header_write(info_writing_pos, leaf_ncm.size(), header_pos++);

    for (const auto &[leaf_index, node_coord_pairs] : leaf_ncm) {
      assert(!node_coord_pairs.empty());

      // Write sub tile index and total number of nodes in sub tile
      const auto num_nodes = node_coord_pairs.size();
      header_write(info_writing_pos, leaf_index, header_pos++);
      header_write(info_writing_pos, num_nodes, header_pos++);

      assert(num_nodes <=
                 static_cast<std::size_t>(_32bit_header_size - header_pos) &&
             CoordT::D * num_nodes <=
                 static_cast<std::size_t>(
                     std::distance(content_writing_pos, content_writing_end)));

      for (const auto &nc_ptr : node_coord_pairs) {
        // Write node index and coord
        header_write(info_writing_pos, nc_ptr->first, header_pos++);
        nc_ptr->second.bit_copy_to_vec(content_writing_pos);
      }
    }
  }

  assert(header_pos == _32bit_header_size &&
         (std::distance(info_writing_pos, info_writing_end) == 1 ||
          info_writing_pos == info_writing_end) &&
         content_writing_pos == content_writing_end);
}

template <typename CoordT>
void reconstruct_received_info(CommunicationInfo<CoordT> &comm_info,
                               RemoteIndexedCoordCache<CoordT> &remote_cache) {
  static_assert(sizeof(tileidx_t) == 4ul && sizeof(nodeidx_t) == 4ul);
  assert(comm_info.filtered_coords_.empty() && remote_cache.empty());

  const auto payload_size = comm_info.data_payload_.size();

  if (payload_size < 2)
    throw std::runtime_error("Cannot reconstruct invalid data payload");

  // If only two elements are received it means
  // the payload content is empty and only
  // minimum header was received
  if (2 == payload_size) {
    comm_info.data_payload_.clear();
    return;
  }

  // Prepare reading iterators
  int32_t header_pos = 0;
  auto info_reading_pos = comm_info.data_payload_.cbegin();

  const auto recv_header_length = header_read(info_reading_pos, header_pos++);
  const auto recv_content_length = header_read(info_reading_pos, header_pos++);
  const auto recv_payload_length = recv_header_length + recv_content_length;
  const auto max_32bit_header = recv_header_length * 2;

  assert(4 <= recv_header_length && // Minimum 1 tile 1 subtile 1 node -> 8
                                    // header items -> 4 compacted items
         0 < recv_content_length && 4 < recv_payload_length &&
         recv_content_length % CoordT::D == 0 &&
         payload_size == static_cast<std::size_t>(recv_payload_length));

  const auto info_reading_end =
      comm_info.data_payload_.cbegin() + recv_header_length;
  auto content_reading_pos = info_reading_end;
  const auto content_reading_end = comm_info.data_payload_.cend();

  // Read total number of tiles
  auto recv_num_tiles = header_read(info_reading_pos, header_pos++);
  assert(0 < recv_num_tiles);

  comm_info.filtered_coords_.reserve(recv_num_tiles);

  for (; recv_num_tiles > 0; --recv_num_tiles) {
    // Read tile index
    auto &tile_map =
        comm_info.filtered_coords_[header_read(info_reading_pos, header_pos++)];
    assert(tile_map.empty());

    // Read total number of sub tiles
    auto recv_num_leaves = header_read(info_reading_pos, header_pos++);
    assert(0 < recv_num_leaves);

    tile_map.reserve(recv_num_leaves);
    comm_info.total_received_num_leaves_ += recv_num_leaves;

    for (; recv_num_leaves > 0; --recv_num_leaves) {
      // Read sub tile index
      auto &leaf_vec = tile_map[header_read(info_reading_pos, header_pos++)];
      assert(leaf_vec.empty());

      // Read total number of nodes
      auto recv_num_nodes = header_read(info_reading_pos, header_pos++);
      assert(0 < recv_num_nodes &&
             recv_num_nodes <= (max_32bit_header - header_pos) &&
             CoordT::D * recv_num_nodes <=
                 std::distance(content_reading_pos, content_reading_end));

      leaf_vec.reserve(recv_num_nodes);
      auto ncp_it = remote_cache.emplace_front(recv_num_nodes).begin();

      for (; recv_num_nodes > 0; --recv_num_nodes) {
        // Read node idx and coord data
        ncp_it->first = header_read(info_reading_pos, header_pos++);
        ncp_it->second = CoordT::bit_copy_from_vec(content_reading_pos);
        leaf_vec.emplace_back(ncp_it++);
      }
    }
  }

  assert(
      (header_pos == max_32bit_header - 1 || header_pos == max_32bit_header) &&
      (std::distance(info_reading_pos, info_reading_end) == 1 ||
       info_reading_pos == info_reading_end) &&
      content_reading_pos == content_reading_end);

  // Automatically clear after successful reconstruction
  comm_info.data_payload_.clear();
}
} // namespace sapi

#endif
