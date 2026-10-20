/*
 *  mask_containers.h
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

#ifndef MASK_CONTAINERS_H
#define MASK_CONTAINERS_H

#include <forward_list>

#include "node_containers.h"

namespace sapi {
// Forward definition to link with coordinates.h
template <typename CoordT> struct Displacement;

// Forward definition to link with tile.h
template <typename CoordT> struct Tile;

// Forward definition to comparisons.h
template <typename CoordT>
bool compare_indexed_node_ptr_maps(
    const FilteredTileNodeCollection<CoordT> &left,
    const FilteredTileNodeCollection<CoordT> &right);

struct LeafPairInfo {
  tileidx_t source_index_;
  tileidx_t target_index_;
  shift_t image_index_;
};

inline LeafPairInfo construct_leaf_pair_info(const tileidx_t source,
                                             const tileidx_t target,
                                             const shift_t image) {
  LeafPairInfo lpi;
  lpi.source_index_ = source;
  lpi.target_index_ = target;
  lpi.image_index_ = image;
  return lpi;
}

template <typename CoordT> struct TilePairInfo {
  const std::vector<CoordT> *image_displacements_ = nullptr;

  // Local leaf index
  //  -> Remote leaf index
  //      -> compressed used pairings
  std::unordered_map<tileidx_t, std::unordered_map<tileidx_t, count_t>>
      aggregated_leaf_pairs_;

  TilePairInfo() noexcept = default;
  TilePairInfo(const TilePairInfo &) = delete;
  TilePairInfo(TilePairInfo &&) noexcept = default;
  ~TilePairInfo() noexcept = default;

  TilePairInfo &operator=(const TilePairInfo &) = delete;
  TilePairInfo &operator=(TilePairInfo &&) noexcept;

  bool operator==(const TilePairInfo &) const;
};

template <typename CoordT>
inline TilePairInfo<CoordT> &
TilePairInfo<CoordT>::operator=(TilePairInfo &&tpi) noexcept {
  image_displacements_ = tpi.image_displacements_;
  tpi.image_displacements_ = nullptr;

  aggregated_leaf_pairs_.swap(tpi.aggregated_leaf_pairs_);
  tpi.aggregated_leaf_pairs_.clear();

  return *this;
}

template <typename CoordT>
inline bool TilePairInfo<CoordT>::operator==(const TilePairInfo &tpi) const {
  return aggregated_leaf_pairs_ == tpi.aggregated_leaf_pairs_;
}

template <typename CoordT> struct TileSetInfo {
  tileidx_t valid_tiles_ = 0;
  tileidx_t valid_leaves_ = 0;
  nodeidx_t total_node_count_ = 0;

  // Stores the tile to tile pair comparison combinations between
  // locally owned tiles and tiles from another rank
  // Local tile index
  //      remote tile index
  //          tile pair info
  std::vector<std::pair<
      tileidx_t, std::vector<std::pair<tileidx_t, TilePairInfo<CoordT>>>>>
      tile_pairings_;

  TileSetInfo() noexcept = default;
  TileSetInfo(const TileSetInfo &) = delete;
  TileSetInfo(TileSetInfo &&) noexcept = default;
  ~TileSetInfo() noexcept = default;

  TileSetInfo &operator=(const TileSetInfo &) = delete;
  TileSetInfo &operator=(TileSetInfo &&) noexcept;

  bool operator==(const TileSetInfo &) const;
};

template <typename CoordT>
inline TileSetInfo<CoordT> &
TileSetInfo<CoordT>::operator=(TileSetInfo &&tsi) noexcept {
  valid_tiles_ = tsi.valid_tiles_;
  tsi.valid_tiles_ = 0;

  valid_leaves_ = tsi.valid_leaves_;
  tsi.valid_leaves_ = 0;

  total_node_count_ = tsi.total_node_count_;
  tsi.total_node_count_ = 0;

  tile_pairings_.swap(tsi.tile_pairings_);
  tsi.tile_pairings_.clear();

  return *this;
}

template <typename CoordT>
inline bool TileSetInfo<CoordT>::operator==(const TileSetInfo &tsi) const {
  return tile_pairings_ == tsi.tile_pairings_;
}

template <typename CoordT> struct CommunicationInfo {
  count_t total_received_num_leaves_ = 0;

  // From source rank
  // This map is then copied onto data_payload_
  // to be sent via MPI to receiving rank
  FilteredTileNodeCollection<CoordT> filtered_coords_;

  // Payload is composed by:
  //
  // Header:
  //  header length: 1 element
  //  content length: 1 element
  //  number of tiles: 1 element
  //  for each tile:
  //      index of tile: valid tiles count
  //      number of sub tiles: valid tiles count
  //      for each sub tiles
  //          index of sub tile: valid sub tiles count
  //          number of nodes: valid sub tiles count
  //          for each node in sub tile
  //              node index: total nodes
  //
  // Content:
  //  for the coordinate of each node:
  //      each dimension bit-casted to an unsigned 64bit integer: total
  //      sequences count * number of dimensions
  //
  // Payload on sender side must be manually cleared once the target has
  // received the data on receiver side payload is cleared as soon as
  // reconstruction is done
  std::vector<uint64_t> data_payload_;

  CommunicationInfo() noexcept = default;
  CommunicationInfo(const CommunicationInfo &) = delete;
  CommunicationInfo(CommunicationInfo &&) noexcept = default;
  ~CommunicationInfo() noexcept = default;

  CommunicationInfo &operator=(const CommunicationInfo &) = delete;
  CommunicationInfo &operator=(CommunicationInfo &&) noexcept;

  bool operator==(const CommunicationInfo &) const;
};

template <typename CoordT>
inline CommunicationInfo<CoordT> &
CommunicationInfo<CoordT>::operator=(CommunicationInfo &&ci) noexcept {
  total_received_num_leaves_ = ci.total_received_num_leaves_;
  ci.total_received_num_leaves_ = 0;

  filtered_coords_.swap(ci.filtered_coords_);
  ci.filtered_coords_.clear();

  data_payload_.swap(ci.data_payload_);
  ci.data_payload_.clear();

  return *this;
}

template <typename CoordT>
inline bool
CommunicationInfo<CoordT>::operator==(const CommunicationInfo &ci) const {
  return compare_indexed_node_ptr_maps<CoordT>(filtered_coords_,
                                               ci.filtered_coords_);
}

// Storage for received indexed coord pairs
// one storage is instantiated per rank
template <typename CoordT>
using RemoteIndexedCoordCache =
    std::forward_list<std::vector<std::pair<nodeidx_t, CoordT>>>;

template <typename CoordT> struct RankPairInfo {
  TileSetInfo<CoordT> tile_pairs_set_;
  CommunicationInfo<CoordT> sender_info_;
  CommunicationInfo<CoordT> receiver_info_;

  RemoteIndexedCoordCache<CoordT> remote_indexed_coord_cache_;

  RankPairInfo() noexcept = default;
  RankPairInfo(const RankPairInfo &) = delete;
  RankPairInfo(RankPairInfo &&) noexcept = default;
  ~RankPairInfo() noexcept = default;

  RankPairInfo &operator=(const RankPairInfo &) = delete;
  RankPairInfo &operator=(RankPairInfo &&) noexcept;

  bool operator==(const RankPairInfo &) const;
};

template <typename CoordT>
inline RankPairInfo<CoordT> &
RankPairInfo<CoordT>::operator=(RankPairInfo &&rpi) noexcept {
  tile_pairs_set_ = std::move(rpi.tile_pairs_set_);
  sender_info_ = std::move(rpi.sender_info_);
  receiver_info_ = std::move(rpi.receiver_info_);

  remote_indexed_coord_cache_.swap(rpi.remote_indexed_coord_cache_);
  rpi.remote_indexed_coord_cache_.clear();

  return *this;
}

template <typename CoordT>
inline bool RankPairInfo<CoordT>::operator==(const RankPairInfo &rpi) const {
  return tile_pairs_set_ == rpi.tile_pairs_set_ &&
         sender_info_ == rpi.sender_info_ &&
         receiver_info_ == rpi.receiver_info_;
}

template <typename CoordT> struct DistributedPairInfo {
  std::unordered_map<vp_t, RankPairInfo<CoordT>> source_side_info_;
  std::unordered_map<vp_t, RankPairInfo<CoordT>> target_side_info_;

  DistributedPairInfo() noexcept = default;
  DistributedPairInfo(const DistributedPairInfo &) = delete;
  DistributedPairInfo(DistributedPairInfo &&) noexcept = default;
  ~DistributedPairInfo() noexcept = default;

  DistributedPairInfo &operator=(const DistributedPairInfo &) = delete;
  DistributedPairInfo &operator=(DistributedPairInfo &&) = delete;

  bool operator==(const DistributedPairInfo &) const;
};

template <typename CoordT>
inline bool
DistributedPairInfo<CoordT>::operator==(const DistributedPairInfo &dci) const {
  return source_side_info_ == dci.source_side_info_ &&
         target_side_info_ == dci.target_side_info_;
}
} // namespace sapi

#endif
