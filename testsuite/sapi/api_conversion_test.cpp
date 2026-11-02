#include "api_converters.h"
#include "enum_store.h"
#include "test_helpers.h"
#include "timer_manager.h"

using namespace sapi;

void array_test(GC &gc) {
  std::string str_test = "Null";
  CharArray carr_test;
  copy_to_charray_from_string(carr_test, str_test, gc);
  assert(charray_to_string(carr_test) == str_test);

  std::vector<std::string> str_vec = {"This", "is", "a", "test"};
  NestedCharArray nested_carr;
  copy_to_nested_charray_from_string_vector(nested_carr, str_vec, gc);
  assert(nested_charray_to_string_vector(nested_carr) == str_vec);

  std::vector<space_t> spt_vec = {0.1, 1. / 10., 10. / 100.};
  SpaceTArray spt_arr;
  copy_to_array_from_collection(spt_arr, spt_vec, gc);
  assert(array_to_vector(spt_arr) == spt_vec);

  std::vector<std::vector<space_t>> spt_vecvec = {{2.5, 1.5, 0.5},
                                                  {0.5, 1.5, 20, 10}};
  NestedSpaceTArray nested_spt_arr;
  copy_to_nested_array_from_nested_collection(nested_spt_arr, spt_vecvec, gc);
  assert(nested_array_to_nested_vector(nested_spt_arr) == spt_vecvec);

  std::set<tileidx_t> tix_set = {10, 20, 30, 40, 50};
  TileIdxArray tix_arr;
  copy_to_array_from_collection(tix_arr, tix_set, gc);
  assert(array_to_set(tix_arr) == tix_set);

  std::vector<std::set<tileidx_t>> tix_vecset = {{1, 2, 3}, {4, 5, 6}};
  NestedTileIdxArray nested_tix_arr;
  copy_to_nested_array_from_nested_collection(nested_tix_arr, tix_vecset, gc);
  assert(nested_array_to_set_vector(nested_tix_arr) == tix_vecset);
}

void gpstruct_test(GC &gc) {
  GridParameters gp;
  gp.grid_origin_ = {0.1, 0.01, 0.001};
  gp.grid_dimensions_ = {1, 2, 3};
  gp.tile_type_ = "Trapezoid";
  gp.tile_side_lengths_ = {1, 2, 1, 4};
  gp.tile_angular_offsets_ = {30, 60, 100};

  GPStruct gps;
  copy_to_array_from_collection(gps.grid_origin_, gp.grid_origin_, gc);
  copy_to_array_from_collection(gps.grid_dimensions_, gp.grid_dimensions_, gc);
  copy_to_charray_from_string(gps.tile_type_, gp.tile_type_, gc);
  copy_to_array_from_collection(gps.tile_side_lengths_, gp.tile_side_lengths_,
                                gc);
  copy_to_array_from_collection(gps.tile_angular_offsets_,
                                gp.tile_angular_offsets_, gc);

  const auto test = gpstruct_to_grid_params(gps);
  assert(test.grid_origin_ == gp.grid_origin_ &&
         test.grid_dimensions_ == gp.grid_dimensions_ &&
         test.tile_type_ == gp.tile_type_ &&
         test.tile_side_lengths_ == gp.tile_side_lengths_ &&
         test.tile_angular_offsets_ == gp.tile_angular_offsets_);
}

void mpstruct_test(GC &gc) {
  MaskParameters mp;
  mp.mask_blueprint_name_ = "Circular";
  mp.mask_blueprint_params_ = {1.};
  mp.mask_blueprint_offset_ = {1., 2};
  mp.source_mask_name_ = "Parallelogram";
  mp.source_mask_params_ = {0, 0, 1, 0, 0, 1};
  mp.source_mask_offset_ = {0, 1};
  mp.target_mask_name_ = "Elliptical";
  mp.target_mask_params_ = {2, 2, 0.5, 0.4, 90};
  mp.target_mask_offset_ = {};

  MPStruct mps;
  copy_to_charray_from_string(mps.mask_blueprint_name_, mp.mask_blueprint_name_,
                              gc);
  copy_to_array_from_collection(mps.mask_blueprint_params_,
                                mp.mask_blueprint_params_, gc);
  copy_to_array_from_collection(mps.mask_blueprint_offset_,
                                mp.mask_blueprint_offset_, gc);
  copy_to_charray_from_string(mps.source_mask_name_, mp.source_mask_name_, gc);
  copy_to_array_from_collection(mps.source_mask_params_, mp.source_mask_params_,
                                gc);
  copy_to_array_from_collection(mps.source_mask_offset_, mp.source_mask_offset_,
                                gc);
  copy_to_charray_from_string(mps.target_mask_name_, mp.target_mask_name_, gc);
  copy_to_array_from_collection(mps.target_mask_params_, mp.target_mask_params_,
                                gc);
  copy_to_array_from_collection(mps.target_mask_offset_, mp.target_mask_offset_,
                                gc);

  const auto mp_test = mpstruct_to_mask_params(mps);
  assert(mp_test.mask_blueprint_name_ == mp.mask_blueprint_name_ &&
         mp_test.mask_blueprint_params_ == mp.mask_blueprint_params_ &&
         mp_test.mask_blueprint_offset_ == mp.mask_blueprint_offset_ &&
         mp_test.source_mask_name_ == mp.source_mask_name_ &&
         mp_test.source_mask_params_ == mp.source_mask_params_ &&
         mp_test.source_mask_offset_ == mp.source_mask_offset_ &&
         mp_test.target_mask_name_ == mp.target_mask_name_ &&
         mp_test.target_mask_params_ == mp.target_mask_params_ &&
         mp_test.target_mask_offset_ == mp.target_mask_offset_);
}

void cpstruct_test(GC &gc) {
  ConnectionParameters cp;
  cp.edge_wrap_ = true;
  cp.only_neighborhood_ = true;
  cp.allow_multiplicity_ = true;
  cp.allow_self_connections_ = true;
  cp.connection_counts_ = 100;
  cp.rule_ = "FixedNumber";
  cp.weight_df_name_ = "Distance";
  cp.weight_ufs_names_ = {"Inverse", "Max", "Min"};
  cp.weight_ufs_params_ = {{}, {0}, {10}};
  cp.delay_df_name_ = "Constant";
  cp.delay_df_params_ = {1};

  CPStruct cps;
  cps.edge_wrap_ = cp.edge_wrap_;
  cps.only_neighborhood_ = cp.only_neighborhood_;
  cps.allow_multiplicity_ = cp.allow_multiplicity_;
  cps.allow_self_connections_ = cp.allow_self_connections_;
  cps.connection_counts_ = cp.connection_counts_;
  copy_to_charray_from_string(cps.rule_, cp.rule_, gc);
  copy_to_charray_from_string(cps.weight_df_name_, cp.weight_df_name_, gc);
  copy_to_array_from_collection(cps.weight_df_params_, cp.weight_df_params_,
                                gc);
  copy_to_nested_charray_from_string_vector(cps.weight_ufs_names_,
                                            cp.weight_ufs_names_, gc);
  copy_to_nested_array_from_nested_collection(cps.weight_ufs_params_,
                                              cp.weight_ufs_params_, gc);
  copy_to_charray_from_string(cps.delay_df_name_, cp.delay_df_name_, gc);
  copy_to_array_from_collection(cps.delay_df_params_, cp.delay_df_params_, gc);
  copy_to_nested_charray_from_string_vector(cps.delay_ufs_names_,
                                            cp.delay_ufs_names_, gc);
  copy_to_nested_array_from_nested_collection(cps.delay_ufs_params_,
                                              cp.delay_ufs_params_, gc);
  copy_to_charray_from_string(cps.prob_df_name_, cp.prob_df_name_, gc);
  copy_to_array_from_collection(cps.prob_df_params_, cp.prob_df_params_, gc);
  copy_to_nested_charray_from_string_vector(cps.prob_ufs_names_,
                                            cp.prob_ufs_names_, gc);
  copy_to_nested_array_from_nested_collection(cps.prob_ufs_params_,
                                              cp.prob_ufs_params_, gc);

  const auto cp_test = cpstruct_to_conn_params(cps);
  assert(cp_test.edge_wrap_ == cp.edge_wrap_ &&
         cp_test.only_neighborhood_ == cp.only_neighborhood_ &&
         cp_test.allow_multiplicity_ == cp.allow_multiplicity_ &&
         cp_test.allow_self_connections_ == cp.allow_self_connections_ &&
         cp_test.connection_counts_ == cp.connection_counts_ &&
         cp_test.rule_ == cp.rule_ &&
         cp_test.weight_df_name_ == cp.weight_df_name_ &&
         cp_test.weight_df_params_ == cp.weight_df_params_ &&
         cp_test.weight_ufs_names_ == cp.weight_ufs_names_ &&
         cp_test.weight_ufs_params_ == cp.weight_ufs_params_ &&
         cp_test.delay_df_name_ == cp.delay_df_name_ &&
         cp_test.delay_df_params_ == cp.delay_df_params_ &&
         cp_test.delay_ufs_names_ == cp.delay_ufs_names_ &&
         cp_test.delay_ufs_params_ == cp.delay_ufs_params_ &&
         cp_test.prob_df_name_ == cp.prob_df_name_ &&
         cp_test.prob_df_params_ == cp.prob_df_params_ &&
         cp_test.prob_ufs_names_ == cp.prob_ufs_names_ &&
         cp_test.prob_ufs_params_ == cp.prob_ufs_params_);
}

void timer_data_test(GC &gc) {
  RecordedTimes rt;
  rt.rank_times_ = {{"", 350}, {"t1", 0.1}, {"t2", 0.2}};

  rt.thread_times_ = {{"1", std::vector<double>{1, 2, 3}},
                      {"2", std::vector<double>{4, 5, 6}}};

  const auto rtpa = recorded_times_to_array_pair(rt, gc);
  assert(rtpa != nullptr && rtpa->first_.size_ == 3 &&
         rtpa->second_.size_ == 2);

  for (std::size_t i = 0; i < rtpa->first_.size_; ++i) {
    const auto tdpp = &rtpa->first_.array_[i];
    const auto search = rt.rank_times_.find(charray_to_string(tdpp->first_));
    assert(search != rt.rank_times_.end() && search->second == tdpp->second_);
  }

  for (std::size_t i = 0; i < rtpa->second_.size_; ++i) {
    const auto tdpp = &rtpa->second_.array_[i];
    const auto search = rt.thread_times_.find(charray_to_string(tdpp->first_));
    assert(search != rt.thread_times_.end() &&
           search->second == array_to_vector(tdpp->second_));
  }
}

void indexed_node_coord_test(GC &gc) {
  IndexedNodeCoordinates inc;
  inc.dimensions_ = 2;
  inc.indexes_ = {10, 2};
  inc.coordinates_ = {1, 2, 3, 4};

  const auto ncs = make_view_from_indexed_node_coords(inc, gc);

  assert(ncs != nullptr);
  assert(ncs->node_count_ == 2 && ncs->dimensions_ == 2);
  assert(ncs->indexes_ != nullptr && ncs->coordinates_ != nullptr);
  assert(ncs->indexes_[0] == 10 && ncs->indexes_[1] == 2);
  assert(ncs->coordinates_[0] == 1 && ncs->coordinates_[1] == 2 &&
         ncs->coordinates_[2] == 3 && ncs->coordinates_[3] == 4);
}

void positions_view_test(GC &gc) {
  std::vector<space_t> coords = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  const auto pvs = make_view_from_positions(3, coords, gc);

  assert(pvs->dimensions_ == 3 && pvs->coord_count_ == 9 &&
         pvs->coordinates_ == coords.data());
}

void grid_vertices_test(GC &gc) {
  GridVertexMap gvm;
  gvm.dimensions_ = 2;
  gvm.num_tiles_ = 1;
  gvm.leaves_per_tile_ = 2;
  gvm.vertices_per_tile_ = 2;
  gvm.vertices_per_leaf_ = 1;

  gvm.tile_indexes_ = {1};
  gvm.tile_vertices_ = {1, 2, 3, 4};
  gvm.leaf_vertices_ = {11, 22, 33, 44};

  const auto gvs = make_view_from_grid_vertex_map(gvm, gc);
  assert(gvs != nullptr && gvs->dimensions_ == gvm.dimensions_ &&
         gvs->num_tiles_ == gvm.num_tiles_ &&
         gvs->leaves_per_tile_ == gvm.leaves_per_tile_ &&
         gvs->vertices_per_tile_ == gvm.vertices_per_tile_ &&
         gvs->vertices_per_leaf_ == gvm.vertices_per_leaf_ &&
         gvs->tile_indexes_ == gvm.tile_indexes_.data() &&
         gvs->tile_vertices_ == gvm.tile_vertices_.data() &&
         gvs->leaf_vertices_ == gvm.leaf_vertices_.data());
}

void dtns_test(GC &gc) {
  DistributedTiledNodeSequenceMap dtns{{0, {{0, {0, 1}}}},
                                       {1, {{1, {1, 1}}, {2, {3, 2}}}}};

  const auto tnss = dist_tns_map_to_tns_pair_array(dtns, gc);
  assert(tnss != nullptr && tnss->size_ == 2);

  for (std::size_t i = 0; i < tnss->size_; ++i) {
    const auto rank_pp = &tnss->array_[i];
    const auto rank_search = dtns.find(rank_pp->first_);
    assert(rank_search != dtns.end() && rank_pp->second_.size_ > 0);

    for (std::size_t j = 0; j < rank_pp->second_.size_; ++j) {
      const auto tile_pp = &rank_pp->second_.array_[j];
      const auto tile_search = rank_search->second.find(tile_pp->first_);
      assert(tile_search != rank_search->second.end() &&
             tile_pp->second_.first_ == tile_search->second.first &&
             tile_pp->second_.second_ == tile_search->second.second);
    }
  }
}

void rci_test(GC &gc) {
  RankConnectionInfo t0;
  t0.total_generated_connections_ = 2;
  ConnectionVectors cvec0;
  cvec0.sizes_ = 2;
  cvec0.connection_sources_ = {0, 3};
  cvec0.connection_targets_ = {0, 1};
  cvec0.connection_weights_ = {2, 3};
  cvec0.connection_delays_ = {1, 1};
  t0.partitioned_connections_.reserve(1);
  t0.partitioned_connections_.emplace_back(std::move(cvec0));

  RankConnectionInfo t1;
  ConnectionVectors cvec10;
  ConnectionVectors cvec11;
  cvec10.sizes_ = 2;
  cvec11.sizes_ = 1;
  t1.total_generated_connections_ = 3;
  cvec10.connection_sources_ = {1, 2};
  cvec10.connection_targets_ = {2, 3};
  cvec10.connection_weights_ = {5, 6};
  cvec10.connection_delays_ = {7, 8};
  cvec11.connection_sources_ = {1};
  cvec11.connection_targets_ = {1};
  cvec11.connection_weights_ = {1};
  cvec11.connection_delays_ = {7};
  t1.partitioned_connections_.reserve(2);
  t1.partitioned_connections_.emplace_back(std::move(cvec10));
  t1.partitioned_connections_.emplace_back(std::move(cvec11));

  DistributedConnectionInfo rci;
  assert_emplace(
      rci.incoming_connections_.emplace(std::make_pair(1, std::move(t0))));
  assert_emplace(
      rci.outgoing_connections_.emplace(std::make_pair(3, std::move(t1))));

  const auto rci_pair = make_view_from_distributed_connection_info(rci, gc);

  assert(rci_pair != nullptr);

  assert(rci_pair->first_.size_ == 1 && rci_pair->second_.size_ == 1);
  assert(rci_pair->first_.array_[0].first_ == 1 &&
         rci_pair->first_.array_[0].second_.num_partitions_ == 1 &&
         rci_pair->first_.array_[0].second_.partition_sizes_[0] == 2);
  assert(rci_pair->second_.array_[0].first_ == 3 &&
         rci_pair->second_.array_[0].second_.num_partitions_ == 2 &&
         rci_pair->second_.array_[0].second_.partition_sizes_[0] == 2 &&
         rci_pair->second_.array_[0].second_.partition_sizes_[1] == 1);
  assert(rci_pair->first_.array_[0].second_.sources_[0][0] == 0 &&
         rci_pair->first_.array_[0].second_.targets_[0][0] == 0 &&
         rci_pair->first_.array_[0].second_.weights_[0][0] == 2. &&
         rci_pair->first_.array_[0].second_.delays_[0][0] == 1.);
  assert(rci_pair->first_.array_[0].second_.sources_[0][1] == 3 &&
         rci_pair->first_.array_[0].second_.targets_[0][1] == 1 &&
         rci_pair->first_.array_[0].second_.weights_[0][1] == 3. &&
         rci_pair->first_.array_[0].second_.delays_[0][1] == 1.);
  assert(rci_pair->second_.array_[0].second_.sources_[0][0] == 1 &&
         rci_pair->second_.array_[0].second_.targets_[0][0] == 2 &&
         rci_pair->second_.array_[0].second_.weights_[0][0] == 5. &&
         rci_pair->second_.array_[0].second_.delays_[0][0] == 7.);
  assert(rci_pair->second_.array_[0].second_.sources_[0][1] == 2 &&
         rci_pair->second_.array_[0].second_.targets_[0][1] == 3 &&
         rci_pair->second_.array_[0].second_.weights_[0][1] == 6. &&
         rci_pair->second_.array_[0].second_.delays_[0][1] == 8.);
  assert(rci_pair->second_.array_[0].second_.sources_[1][0] == 1 &&
         rci_pair->second_.array_[0].second_.targets_[1][0] == 1 &&
         rci_pair->second_.array_[0].second_.weights_[1][0] == 1. &&
         rci_pair->second_.array_[0].second_.delays_[1][0] == 7.);
}

void rci_view_test(GC &gc) {
  RankConnectionInfo t0;
  ConnectionVectors cvec0;
  cvec0.sizes_ = 2;
  cvec0.connection_sources_ = {0, 3};
  cvec0.connection_targets_ = {0, 1};
  cvec0.connection_weights_ = {2, 3};
  cvec0.connection_delays_ = {1, 1};
  t0.partitioned_connections_.reserve(1);
  t0.partitioned_connections_.emplace_back(std::move(cvec0));

  RankConnectionInfo t1;
  ConnectionVectors cvec10;
  ConnectionVectors cvec11;
  cvec10.sizes_ = 2;
  cvec11.sizes_ = 1;
  cvec10.connection_sources_ = {1, 2};
  cvec10.connection_targets_ = {2, 3};
  cvec10.connection_weights_ = {5, 6};
  cvec10.connection_delays_ = {7, 8};
  cvec11.connection_sources_ = {1};
  cvec11.connection_targets_ = {1};
  cvec11.connection_weights_ = {1};
  cvec11.connection_delays_ = {7};
  t1.partitioned_connections_.reserve(2);
  t1.partitioned_connections_.emplace_back(std::move(cvec10));
  t1.partitioned_connections_.emplace_back(std::move(cvec11));

  DistributedConnectionInfo rci;
  assert_emplace(
      rci.incoming_connections_.emplace(std::make_pair(1, std::move(t0))));
  assert_emplace(
      rci.incoming_connections_.emplace(std::make_pair(3, std::move(t1))));

  const auto view = make_view_from_distributed_connection_info(rci, gc);

  assert(view != nullptr);
  assert(view->first_.size_ == 2 && view->second_.size_ == 0);
  assert(view->first_.array_ != nullptr && view->second_.array_ == nullptr);

  const auto &real_first =
      rci.incoming_connections_.at(view->first_.array_[0].first_);
  const auto &view_first = view->first_.array_[0].second_;
  assert(view_first.num_partitions_ ==
         real_first.partitioned_connections_.size());

  for (std::size_t i = 0; i < view_first.num_partitions_; ++i) {
    const auto &cv = real_first.partitioned_connections_.at(i);
    assert(view_first.partition_sizes_[i] == cv.sizes_);
    assert(view_first.sources_[i] == cv.connection_sources_.data());
    assert(view_first.targets_[i] == cv.connection_targets_.data());
    assert(view_first.weights_[i] == cv.connection_weights_.data());
    assert(view_first.delays_[i] == cv.connection_delays_.data());
  }

  const auto &real_second =
      rci.incoming_connections_.at(view->first_.array_[1].first_);
  const auto &view_second = view->first_.array_[1].second_;
  assert(view_second.num_partitions_ ==
         real_second.partitioned_connections_.size());

  for (std::size_t i = 0; i < view_second.num_partitions_; ++i) {
    const auto &cv = real_second.partitioned_connections_.at(i);
    assert(view_second.partition_sizes_[i] == cv.sizes_);
    assert(view_second.sources_[i] == cv.connection_sources_.data());
    assert(view_second.targets_[i] == cv.connection_targets_.data());
    assert(view_second.weights_[i] == cv.connection_weights_.data());
    assert(view_second.delays_[i] == cv.connection_delays_.data());
  }
}

void test_parameter_names(GC &gc) {
  std::unordered_map<std::string, std::vector<std::string>> names;

  names["tile_shapes"] = {TILE_SHAPE_NAMES[0], TILE_SHAPE_NAMES[1],
                          TILE_SHAPE_NAMES[2]};
  names["mask_shapes"] = {MASK_SHAPE_NAMES[0], MASK_SHAPE_NAMES[1],
                          MASK_SHAPE_NAMES[2], MASK_SHAPE_NAMES[3]};
  names["distribution_modes"] = {DISTRIBUTION_MODE_NAMES[0],
                                 DISTRIBUTION_MODE_NAMES[1],
                                 DISTRIBUTION_MODE_NAMES[2]};
  names["connection_rules"] = {
      CONNECTION_RULE_NAMES[0], CONNECTION_RULE_NAMES[1],
      CONNECTION_RULE_NAMES[2], CONNECTION_RULE_NAMES[3]};
  names["unary_functions"] = {UNARY_FUNCTION_NAMES[0], UNARY_FUNCTION_NAMES[1],
                              UNARY_FUNCTION_NAMES[2], UNARY_FUNCTION_NAMES[3],
                              UNARY_FUNCTION_NAMES[4], UNARY_FUNCTION_NAMES[5],
                              UNARY_FUNCTION_NAMES[6], UNARY_FUNCTION_NAMES[7],
                              UNARY_FUNCTION_NAMES[8], UNARY_FUNCTION_NAMES[9]};
  names["displacement_functions"] = {
      DISPLACEMENT_FUNCTION_NAMES[0], DISPLACEMENT_FUNCTION_NAMES[1],
      DISPLACEMENT_FUNCTION_NAMES[2], DISPLACEMENT_FUNCTION_NAMES[3],
      DISPLACEMENT_FUNCTION_NAMES[4], DISPLACEMENT_FUNCTION_NAMES[5],
      DISPLACEMENT_FUNCTION_NAMES[6], DISPLACEMENT_FUNCTION_NAMES[7],
  };
  names["random_generators"] = {RNG_NAMES_32BIT[0], RNG_NAMES_32BIT[1]};

  ParameterNamesPairArray pnpa;
  generate_param_name_pair_array(pnpa, gc);

  std::unordered_map<std::string, std::vector<std::string>> test;

  for (std::size_t i = 0; i < pnpa.size_; ++i) {
    const auto &entry = pnpa.array_[i];
    test[charray_to_string(entry.first_)] =
        nested_charray_to_string_vector(entry.second_);
  }

  assert(names == test);
}

void test_conn_counts(GC &gc) {
  ConnectionCounts cc;
  cc.incoming_ranks_ = 10;
  cc.source_ranks_.resize(10);
  for (vp_t rank = 0; rank < 10 * 2; rank += 2)
    cc.source_ranks_.at(rank / 2) = rank;

  cc.incoming_counts_.resize(10, 2);

  cc.outgoing_ranks_ = 3;
  cc.target_ranks_.resize(3);
  auto tr_it = cc.target_ranks_.begin();
  for (vp_t rank = 5; rank < 8; ++rank)
    (*tr_it++) = rank;

  const auto cc_view = make_view_from_connection_counts(cc, gc);
  assert(cc_view != nullptr);
  assert(cc_view->incoming_ranks_ == cc.incoming_ranks_);
  assert(cc_view->outgoing_ranks_ == cc.outgoing_ranks_);
  assert(cc_view->source_ranks_ == cc.source_ranks_.data());
  assert(cc_view->incoming_counts_ == cc.incoming_counts_.data());
  assert(cc_view->target_ranks_ == cc.target_ranks_.data());
  assert(cc_view->outgoing_counts_ == cc.outgoing_counts_.data());
}

int main() {
  GC gc;
  array_test(gc);
  gpstruct_test(gc);
  mpstruct_test(gc);
  cpstruct_test(gc);
  timer_data_test(gc);
  dtns_test(gc);
  rci_test(gc);
  positions_view_test(gc);

  return 0;
}
