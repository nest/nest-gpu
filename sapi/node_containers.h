/*
 *  node_containers.h
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

#ifndef NODE_CONTAINERS_H
#define NODE_CONTAINERS_H

#include <algorithm>
#include <deque>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sapi_config.h"

namespace sapi {
// First index in sequence and sequence length
typedef std::pair<nodeidx_t, nodeidx_t> NodeSequence;

// Map of tile index to NodeSequence contained in tile
typedef std::unordered_map<tileidx_t, NodeSequence> TileIdxNodeSequenceMap;

// Map of MPI ranks to NodeSequence instantiated in rank
typedef std::unordered_map<vp_t, NodeSequence> RankNodeSequenceMap;

// Map of MPI ranks to tile index node sequence maps.
// Given a global node creation procedure, a global node sequence is generated,
// this sequence is randomly distributed across the tiles in a grid,
// each tile then holds a sub sequence of the global node sequence .
// As each tile is owned by a single rank,
// tile index node sequence can then be aggregated by MPI rank.
typedef std::unordered_map<vp_t, TileIdxNodeSequenceMap>
    DistributedTiledNodeSequenceMap;

// Vector containing node counts.
// Given a number of processes or number of tiles,
// the vector holds the number of nodes in each rank or tile index,
// the index of the node count is equal to either the rank or tile index.
typedef std::vector<nodeidx_t> NodeCountVector;

// Queue of individual mappings of tile index to corresponding node count.
// As tiles owned by an individual rank may not have contiguous indexes,
// node counts need to be paired with the corresponding tile index,
// during node creation the queue is iterated to instantiate nodes given their
// count.
typedef std::deque<std::pair<tileidx_t, nodeidx_t>> TileIdxNodeCountPairs;

// After instantiating node counts for each tile during global node sequence
// assignment, each rank will hold a queue of node counts per owned tile, the
// index of each queue corresponds to the rank of the owner.
typedef std::vector<TileIdxNodeCountPairs> RankTileIdxNodeCountPairs;

// Map of tile index to sized forward queue of coordinates.
// This map is procedurally generated when manually inserting coordinates
// into the grid using bounding box tree filtering approach.
template <typename CoordT>
using TiledCoordMap = std::unordered_map<tileidx_t, std::deque<CoordT>>;

// Vector of node coordinates
template <typename CoordT>
using IndexedCoordinates = std::vector<std::pair<nodeidx_t, CoordT>>;

// This container maps node indexes to coordinates.
// A vector is instantiated for each tile position in a tile grid.
// Each entry contains the first node id of a node sequence and
// a vector containing all the coords of the node sequence.
// Length of the coordinate vector is equal to the size of the node sequence.
template <typename CoordT>
using IndexedCoordCollection =
    std::deque<std::vector<std::pair<nodeidx_t, CoordT>>>;

// This container maps leaf sub tiles to node index coord vectors.
// In a grid, for each owned tile, each leaf sub tile owns a node index coord
// vector. The length of this vector is equal to the number of leaf sub tiles in
// a tile. When generating node coordinates a leaf sub tile is chosen and
// sequences of node coordinates are inserted into the corresponding vector.
template <typename CoordT>
using LeafNodeCollection = std::vector<IndexedCoordCollection<CoordT>>;

// This container maps tiles to leaf node coord vectors.
// The length of this vector is equal to the number of tiles in the grid.
template <typename CoordT>
using GridNodeCollection = std::vector<LeafNodeCollection<CoordT>>;

// View of a node coordinate
template <typename CoordT>
using IndexedCoordView =
    typename std::vector<std::pair<nodeidx_t, CoordT>>::const_iterator;

// Filtered view of vector of node coordinates
template <typename CoordT>
using FilteredIndexedCoordinates = std::vector<IndexedCoordView<CoordT>>;

// Filtered view of a leaf node collection
template <typename CoordT>
using FilteredLeafNodeCollection =
    std::unordered_map<tileidx_t, FilteredIndexedCoordinates<CoordT>>;

// View of filtered leaf node collection entry
template <typename CoordT>
using FilteredLeafView =
    typename FilteredLeafNodeCollection<CoordT>::const_iterator;

// Filtered view a tile node collection
template <typename CoordT>
using FilteredTileNodeCollection =
    std::unordered_map<tileidx_t, FilteredLeafNodeCollection<CoordT>>;

constexpr inline NodeSequence shift_sequence(const NodeSequence &origin,
                                             const NodeSequence &target,
                                             const nodeidx_t shift) {
  return {origin.first,
          std::min<nodeidx_t>(std::max<nodeidx_t>(target.second - shift, 0),
                              origin.second)};
}

constexpr inline NodeSequence join_sequences(const NodeSequence &left,
                                             const NodeSequence &right) {
  return (right.first < left.first)
             ? shift_sequence(left, right, left.first - right.first)
             : shift_sequence(right, left, right.first - left.first);
}
} // namespace sapi

#endif
