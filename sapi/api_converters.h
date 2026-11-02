/*
 *  api_converters.h
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

#ifndef SAPI_CONVERTERS_H
#define SAPI_CONVERTERS_H

#include <set>
#include <stdexcept>
#include <type_traits>

#include "api_containers.h"
#include "connection_containers.h"
#include "node_containers.h"
#include "spatial_containers.h"

namespace sapi {
// Forward definition to link with timer_manager.h
struct RecordedTimes;

inline std::string charray_to_string(const CharArray &cstr) {
  if (cstr.size_ < 1)
    return "";
  if (cstr.array_ == nullptr || cstr.array_[cstr.size_ - 1] != '\0')
    throw std::invalid_argument("Corrupted CharArray");
  return std::string(cstr.array_);
}

inline void copy_to_charray_from_string(CharArray &cstr, const std::string &str,
                                        GC &gc) {
  const auto str_size = str.size();
  cstr.resize(str_size + 1, gc);
  if (str_size > 0)
    str.copy(cstr.array_, str_size);
  cstr.array_[str_size] = '\0';
}

inline CharArray *string_to_charray(const std::string &str, GC &gc) {
  auto ca_ptr = gc.make_collected<CharArray>();
  copy_to_charray_from_string(*ca_ptr, str, gc);
  return ca_ptr;
}

std::vector<std::string>
nested_charray_to_string_vector(const NestedCharArray &cstrings);

void copy_to_nested_charray_from_string_vector(
    NestedCharArray &cstrings, const std::vector<std::string> &strings, GC &gc);

inline NestedCharArray *
string_vector_to_nested_charray(const std::vector<std::string> &strings,
                                GC &gc) {
  auto cstrings = gc.make_collected<NestedCharArray>();
  copy_to_nested_charray_from_string_vector(*cstrings, strings, gc);
  return cstrings;
}

template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
std::vector<T> array_to_vector(const ArrayT<T> &carray) {
  if (carray.size_ < 1)
    return {};
  if (carray.array_ == nullptr)
    throw std::invalid_argument("Corrupted ArrayT");
  std::vector<T> vec(carray.size_);
  for (std::size_t idx = 0; idx < carray.size_; ++idx)
    vec[idx] = carray.array_[idx];
  return vec;
}

template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
std::vector<std::vector<T>>
nested_array_to_nested_vector(const ArrayT<ArrayT<T>> &nested_carray) {
  if (nested_carray.size_ < 1)
    return {};
  if (nested_carray.array_ == nullptr)
    throw std::invalid_argument("Corrupted NestedTarray");
  std::vector<std::vector<T>> nvec(nested_carray.size_);
  for (std::size_t idx = 0; idx < nested_carray.size_; ++idx)
    nvec[idx] = array_to_vector(nested_carray.array_[idx]);
  return nvec;
}

template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
std::set<T> array_to_set(const ArrayT<T> &carray) {
  if (carray.size_ < 1)
    return {};
  if (carray.array_ == nullptr)
    throw std::invalid_argument("Corrupted ArrayT");
  std::set<T> set;
  for (std::size_t idx = 0; idx < carray.size_; ++idx)
    set.insert(carray.array_[idx]);
  if (set.size() != carray.size_)
    throw std::invalid_argument("Corrupted set from ArrayT");
  return set;
}

template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
std::vector<std::set<T>>
nested_array_to_set_vector(const ArrayT<ArrayT<T>> &nested_carray) {
  if (nested_carray.size_ < 1)
    return {};
  if (nested_carray.array_ == nullptr)
    throw std::invalid_argument("Corrupted NestedTarray");
  std::vector<std::set<T>> setvec(nested_carray.size_);
  for (std::size_t idx = 0; idx < nested_carray.size_; ++idx)
    setvec[idx] = array_to_set(nested_carray.array_[idx]);
  return setvec;
}

template <
    typename CollectionT,
    std::enable_if_t<std::is_arithmetic_v<typename CollectionT::value_type>,
                     bool> = true>
void copy_to_array_from_collection(
    ArrayT<typename CollectionT::value_type> &carray,
    const CollectionT &collection, GC &gc) {
  carray.resize(collection.size(), gc);
  std::size_t t_idx = 0;
  for (const auto &t : collection)
    carray.array_[t_idx++] = t;
}

template <
    typename CollectionT,
    std::enable_if_t<std::is_arithmetic_v<typename CollectionT::value_type>,
                     bool> = true>
inline ArrayT<typename CollectionT::value_type> *
collection_to_array(const CollectionT &collection, GC &gc) {
  auto ta_ptr = gc.make_collected<ArrayT<typename CollectionT::value_type>>();
  copy_to_array_from_collection(*ta_ptr, collection, gc);
  return ta_ptr;
}

template <
    typename NestedCollectionT,
    std::enable_if_t<std::is_arithmetic_v<
                         typename NestedCollectionT::value_type::value_type>,
                     bool> = true>
void copy_to_nested_array_from_nested_collection(
    ArrayT<ArrayT<typename NestedCollectionT::value_type::value_type>>
        &nested_carray,
    const NestedCollectionT &nested_collection, GC &gc) {
  nested_carray.resize(nested_collection.size(), gc);
  std::size_t ta_idx = 0;
  for (const auto &collection : nested_collection)
    copy_to_array_from_collection(nested_carray.array_[ta_idx++], collection,
                                  gc);
}

template <
    typename NestedCollectionT,
    std::enable_if_t<std::is_arithmetic_v<
                         typename NestedCollectionT::value_type::value_type>,
                     bool> = true>
inline ArrayT<ArrayT<typename NestedCollectionT::value_type::value_type>> *
nested_collection_to_nested_array(const NestedCollectionT &nested_collection,
                                  GC &gc) {
  auto nta_ptr = gc.make_collected<
      ArrayT<ArrayT<typename NestedCollectionT::value_type::value_type>>>();
  copy_to_nested_array_from_nested_collection(*nta_ptr, nested_collection, gc);
  return nta_ptr;
}

void make_view_from_indexed_node_coords(NodesViewStruct &nvs,
                                        const IndexedNodeCoordinates &inc);

inline NodesViewStruct *
make_view_from_indexed_node_coords(const IndexedNodeCoordinates &inc, GC &gc) {
  const auto nvs = gc.make_collected<NodesViewStruct>();
  make_view_from_indexed_node_coords(*nvs, inc);
  return nvs;
}

void make_view_from_positions(PositionViewStruct &pvs, const dim_t dimensions,
                              const std::vector<space_t> &coordinates);

inline PositionViewStruct *
make_view_from_positions(const dim_t dimensions,
                         const std::vector<space_t> &coordinates, GC &gc) {
  const auto pvs = gc.make_collected<PositionViewStruct>();
  make_view_from_positions(*pvs, dimensions, coordinates);
  return pvs;
}

void make_view_from_conn_info_map(
    ConnectionViewPairArray &cvpa,
    const std::unordered_map<vp_t, RankConnectionInfo> &ci_map, GC &gc);

inline RemoteConnectionViewPair *
make_view_from_distributed_connection_info(const DistributedConnectionInfo &dci,
                                           GC &gc) {
  const auto rcvp = gc.make_collected<RemoteConnectionViewPair>();
  make_view_from_conn_info_map(rcvp->first_, dci.incoming_connections_, gc);
  make_view_from_conn_info_map(rcvp->second_, dci.outgoing_connections_, gc);
  return rcvp;
}

void make_view_from_connection_counts(ConnectionCountsViewStruct &cc_view,
                                      const ConnectionCounts &cc);

inline ConnectionCountsViewStruct *
make_view_from_connection_counts(const ConnectionCounts &cc, GC &gc) {
  const auto cc_view = gc.make_collected<ConnectionCountsViewStruct>();
  make_view_from_connection_counts(*cc_view, cc);
  return cc_view;
}

void make_view_from_grid_vertex_map(GridViewStruct &cgrid_vertices,
                                    const GridVertexMap &grid_vertices);

inline GridViewStruct *
make_view_from_grid_vertex_map(const GridVertexMap &grid_vertices, GC &gc) {
  auto cgrid_vertices = gc.make_collected<GridViewStruct>();
  make_view_from_grid_vertex_map(*cgrid_vertices, grid_vertices);
  return cgrid_vertices;
}

void copy_to_tns_pair_array_from_dist_tns_map(
    TiledNodeSequencePairArray &cdtns,
    const DistributedTiledNodeSequenceMap &dtns, GC &gc);

inline TiledNodeSequencePairArray *
dist_tns_map_to_tns_pair_array(const DistributedTiledNodeSequenceMap &dtns,
                               GC &gc) {
  auto cdtns = gc.make_collected<TiledNodeSequencePairArray>();
  copy_to_tns_pair_array_from_dist_tns_map(*cdtns, dtns, gc);
  return cdtns;
}

void copy_to_timer_data_pair_array_from_timer_data_map(
    RecordedTimesArrayPair &ctimes, const RecordedTimes &times, GC &gc);

inline RecordedTimesArrayPair *
recorded_times_to_array_pair(const RecordedTimes &times, GC &gc) {
  auto ctimes = gc.make_collected<RecordedTimesArrayPair>();
  copy_to_timer_data_pair_array_from_timer_data_map(*ctimes, times, gc);
  return ctimes;
}

void generate_param_name_pair_array(ParameterNamesPairArray &pnpa, GC &gc);

GridParameters gpstruct_to_grid_params(const GPStruct &gps);

MaskParameters mpstruct_to_mask_params(const MPStruct &mps);

ConnectionParameters cpstruct_to_conn_params(const CPStruct &cps);
} // namespace sapi

#endif
