#include <chrono>
#include <cstdio>

#include "coordinate_geometry.h"
#include "creator_registry.h"
#include "thread_aligned_array.h"
#include "type_erasure_helpers.h"
#include "vp_interface.h"

#include "bounding_box.h"
#include "comparisons.h"
#include "gf2d_geometry.h"
#include "gf_creators.h"
#include "grid_generation.h"
#include "mask2d_geometry.h"
#include "mask_collection.h"
#include "mask_tile_processing.h"
#include "node_containers_init.h"
#include "node_creation.h"
#include "payload_preparation.h"
#include "test_helpers.h"

using namespace sapi;

void test_leaf_tiles_within_range() {
  Tile<Coord2D> t0(TILE_SHAPE::TRIANGLE, Coord2D(), {1}, {0});
  Tile<Coord2D> t1(TILE_SHAPE::TRIANGLE, Coord2D(), {1}, {180});

  MaskCollection<Coord2D> mc;
  mc.blueprint_ = construct_mask<Coord2D>(MASK_SHAPE::CIRCULAR, {}, {0.1}, {});
  TAArray<MaskCollection<Coord2D>> mc_array;
  mc_array.clone(mc);

  assert(mc.blueprint_overlap(t0, t1));

  TilePairInfo<Coord2D> par_pairs;
  TilePairInfo<Coord2D> seq_pairs;

  std::unordered_map<tileidx_t, const Tile<Coord2D> *> st_idx_ptr_map0;
  std::unordered_map<tileidx_t, const Tile<Coord2D> *> st_idx_ptr_map1;

#pragma omp parallel default(shared)
#pragma omp master
  {
    const auto split_start = std::chrono::high_resolution_clock::now();
#pragma omp taskgroup
    {
      t0.split(13, true);
      t1.split(13, true);
    }
    const auto split_end = std::chrono::high_resolution_clock::now();
    const auto split_time = split_end - split_start;
    printf("Time to split: %f\n", split_time.count() / 1e9);

    const auto indexing_start = std::chrono::high_resolution_clock::now();
#pragma omp taskgroup
    {
#pragma omp task default(shared)
      for (const auto &st_ptr : t0.leaf_tiles_)
        assert_emplace(
            st_idx_ptr_map0.emplace(std::make_pair(st_ptr->index_, st_ptr)));

#pragma omp task default(shared)
      for (const auto &st_ptr : t1.leaf_tiles_)
        assert_emplace(
            st_idx_ptr_map1.emplace(std::make_pair(st_ptr->index_, st_ptr)));
    }
    const auto indexing_end = std::chrono::high_resolution_clock::now();
    const auto indexing_time = indexing_end - indexing_start;
    printf("Time to index: %f\n", indexing_time.count() / 1e9);

    const auto range_start = std::chrono::high_resolution_clock::now();
#pragma omp taskgroup
    insert_leaf_tiles_within_range<Coord2D, false, false, false>(
        par_pairs.aggregated_leaf_pairs_, t0, t1, mc_array, 0, false, 13);
    const auto range_end = std::chrono::high_resolution_clock::now();
    const auto range_time = range_end - range_start;
    printf("Time range check parallel: %f\n", range_time.count() / 1e9);
  }

  const auto range_start = std::chrono::high_resolution_clock::now();
  insert_leaf_tiles_within_range<Coord2D, false, false, false>(
      seq_pairs.aggregated_leaf_pairs_, t0, t1, mc_array, 0, false, 13);
  const auto range_end = std::chrono::high_resolution_clock::now();
  const auto range_time = range_end - range_start;
  printf("Time range check sequential: %f\n", range_time.count() / 1e9);

  for (auto &source_target_wrappings_map_pair :
       seq_pairs.aggregated_leaf_pairs_) {
    const auto source_tile_it =
        st_idx_ptr_map0.find(source_target_wrappings_map_pair.first);
    assert(source_tile_it != st_idx_ptr_map0.end());
    const auto source_map_it = par_pairs.aggregated_leaf_pairs_.find(
        source_target_wrappings_map_pair.first);
    assert(source_map_it != par_pairs.aggregated_leaf_pairs_.end());
    for (auto &target_wrappings_fl_pair :
         source_target_wrappings_map_pair.second) {
      const auto target_tile_it =
          st_idx_ptr_map1.find(target_wrappings_fl_pair.first);
      assert(target_tile_it != st_idx_ptr_map1.end());
      const auto target_map_it =
          source_map_it->second.find(target_wrappings_fl_pair.first);
      assert(target_map_it != source_map_it->second.end() &&
             target_map_it->second == target_wrappings_fl_pair.second);
      assert(mc.blueprint_overlap(*source_tile_it->second,
                                  *target_tile_it->second));
      assert(mc.blueprint_overlap(*target_tile_it->second,
                                  *source_tile_it->second));
    }
  }

  assert(seq_pairs == par_pairs);
}

void test_multi_image_aggregation() {
  TileGrid<Coord2D> tg;
  TAArray<GFCollection<Coord2D>> gc_array;
  CreatorRegistry<GridTargetPositionShifts<Coord2D>> gscr;
  CreatorRegistry<ShiftedOriginCreator<Coord2D>> socr;
  CreatorRegistry<CachedTileCreator<Coord2D>> ctcr;

  initialize_gsc_registry(gscr);
  initialize_soc_registry(socr);
  initialize_ctc_registry(ctcr);

  const auto gc = construct_gf_collection<Coord2D>({0., 0.}, {1, 1}, "hexagon",
                                                   {1.}, {0}, gscr, socr, ctcr);
  gc_array.clone(gc);
  tg = generate_tile_grid(gc_array);

  const auto split_start = std::chrono::high_resolution_clock::now();

  for (auto &pos : tg.positions_) {
    pos.tile_.split(5, true);
    for (auto &img : pos.tile_images_)
      if (img.shape_ != TILE_SHAPE::NULL_TS)
        img.split(5, false);
  }

  const auto split_end = std::chrono::high_resolution_clock::now();
  const auto split_time = split_end - split_start;
  printf("Time to split: %f\n", split_time.count() / 1e9);

  auto tp = tg.positions_.begin();

  MaskCollection<Coord2D> mc;
  mc.blueprint_ = construct_mask<Coord2D>(MASK_SHAPE::CIRCULAR, {}, {0.1}, {});
  TAArray<MaskCollection<Coord2D>> mc_array;
  mc_array.clone(mc);

  TilePairInfo<Coord2D> tpi;

#pragma omp parallel default(shared)
#pragma omp master
  {
    const auto overlap_start = std::chrono::high_resolution_clock::now();
#pragma omp taskgroup
    tile_pair_overlap<Coord2D, true, false, false, false>(tpi, *tp, *tp,
                                                          mc_array, 5);
    const auto overlap_end = std::chrono::high_resolution_clock::now();
    const auto overlap_time = overlap_end - overlap_start;
    printf("Time tile target overlap: %f\tsize: %li\n",
           overlap_time.count() / 1e9, tpi.aggregated_leaf_pairs_.size());
  }

  for (const auto &source_target_map : tpi.aggregated_leaf_pairs_)
    for (const auto &target_set_pair : source_target_map.second)
      assert(0 < target_set_pair.second);
}

void test_single_rank_aggregation() {
  TileGrid<Coord2D> tg;
  GridNeighborhood gnb;
  GridNodeCollection<Coord2D> gnc;
  TAArray<GFCollection<Coord2D>> gc_array;
  CreatorRegistry<GridTargetPositionShifts<Coord2D>> gscr;
  CreatorRegistry<ShiftedOriginCreator<Coord2D>> socr;
  CreatorRegistry<CachedTileCreator<Coord2D>> ctcr;

  initialize_gsc_registry(gscr);
  initialize_soc_registry(socr);
  initialize_ctc_registry(ctcr);

  const auto gc = construct_gf_collection<Coord2D>({0., 0.}, {1, 2}, "hexagon",
                                                   {1.}, {0}, gscr, socr, ctcr);
  gc_array.clone(gc);
  tg = generate_tile_grid(gc_array);

  std::vector<std::set<tileidx_t>> rank_tile_ownership_map(gnb.num_processes_);
  for (tileidx_t rank = 0; rank < gnb.num_processes_; ++rank)
    if (rank == gnb.local_rank_)
      rank_tile_ownership_map[rank] = {0, 1};
  gnb.set_tile_ownership(std::move(rank_tile_ownership_map), tg);
  split_tiles_in_grid(tg, *gnb.locally_owned_tiles_, 0);
  initialize_local_grid_node_collection(gnc, *gnb.locally_owned_tiles_, tg);

  const DistributedTiledNodeSequenceMap map0{std::make_pair(
      gnb.local_rank_,
      TileIdxNodeSequenceMap{std::make_pair(0, NodeSequence(0, 25))})};

  std::vector<std::pair<nodeidx_t, Coord2D>> vec;
  vec.resize(25);
  nodeidx_t idx = 0;
  for (auto &p : vec)
    p.first = idx++;
  gnc.at(0).at(0).push_back(vec);

  // Not accounted for in dtns
  vec.resize(15);
  idx = 25;
  for (auto &p : vec)
    p.first = idx++;
  gnc.at(0).at(0).push_back(vec);

  const DistributedTiledNodeSequenceMap map1{std::make_pair(
      gnb.local_rank_,
      TileIdxNodeSequenceMap{std::make_pair(0, NodeSequence(40, 5)),
                             std::make_pair(1, NodeSequence(50, 75))})};

  vec.resize(5);
  idx = 40;
  for (auto &p : vec)
    p.first = idx++;
  gnc.at(0).at(0).push_back(vec);

  // Nodeseq smaller and shifted than what is accounted for in dtns
  vec.resize(50);
  idx = 45;
  for (auto &p : vec)
    p.first = idx++;
  gnc.at(1).at(0).push_back(vec);

  RankPairInfo<Coord2D> rpi0;
  RankPairInfo<Coord2D> rpi1;

  MaskCollection<Coord2D> mc;
  mc.blueprint_ = construct_mask<Coord2D>(MASK_SHAPE::CIRCULAR, {}, {0.1}, {});
  TAArray<MaskCollection<Coord2D>> mc_array;
  mc_array.clone(mc);

  const auto aggregate_start = std::chrono::high_resolution_clock::now();
#pragma omp parallel default(shared)
#pragma omp master
#pragma omp taskgroup
  {
    rank_pair_overlap<Coord2D, true, true, false, false, false>(
        rpi0, map0.at(gnb.local_rank_), map1.at(gnb.local_rank_), tg, gnc,
        mc_array, false);
    rank_pair_overlap<Coord2D, true, true, true, false, false>(
        rpi1, map1.at(gnb.local_rank_), map0.at(gnb.local_rank_), tg, gnc,
        mc_array, false);
  }
  const auto aggregate_end = std::chrono::high_resolution_clock::now();
  const auto aggregate_time = aggregate_end - aggregate_start;
  printf("Time rank pair aggregate: %f\n", aggregate_time.count() / 1e9);

  assert(rpi0.tile_pairs_set_.tile_pairings_.size() == 1 &&
         rpi1.tile_pairs_set_.tile_pairings_.size() == 2);

  assert(rpi0.sender_info_.filtered_coords_.size() == 1 &&
         rpi0.sender_info_.filtered_coords_.find(0) !=
             rpi0.sender_info_.filtered_coords_.end() &&
         rpi1.sender_info_.filtered_coords_.size() == 2 &&
         rpi1.sender_info_.filtered_coords_.find(0) !=
             rpi1.sender_info_.filtered_coords_.end() &&
         rpi1.sender_info_.filtered_coords_.find(1) !=
             rpi1.sender_info_.filtered_coords_.end());

  // As splits == 0, only root tile exists with index 0
  const auto &send0 = rpi0.sender_info_.filtered_coords_.at(0).at(0);
  const auto &send10 = rpi1.sender_info_.filtered_coords_.at(0).at(0);
  const auto &send11 = rpi1.sender_info_.filtered_coords_.at(1).at(0);

  assert(send0.size() == 25 && send10.size() == 5 && send11.size() == 45);

  for (const auto &node_idx_coord_pair : send0)
    assert(0 <= node_idx_coord_pair->first && node_idx_coord_pair->first < 25);

  for (const auto &node_idx_coord_pair : send10)
    assert(40 <= node_idx_coord_pair->first && node_idx_coord_pair->first < 45);

  for (const auto &node_idx_coord_pair : send11)
    assert(50 <= node_idx_coord_pair->first && node_idx_coord_pair->first < 95);

  const auto prepare_start = std::chrono::high_resolution_clock::now();
  prepare_data_payload(rpi0.sender_info_, rpi0.tile_pairs_set_);
  prepare_data_payload(rpi1.sender_info_, rpi1.tile_pairs_set_);
  const auto prepare_end = std::chrono::high_resolution_clock::now();
  const auto prepare_time = prepare_end - prepare_start;
  printf("Time prepare data payload: %f\n", prepare_time.count() / 1e9);

  assert(rpi0.sender_info_.data_payload_.size() ==
             (3 + 2 + 2 + 25) / 2 + 25 * 2 &&
         rpi0.sender_info_.data_payload_[0] ==
             (((3ul + 2ul + 2ul + 25ul) / 2ul) << 32 | (25ul * 2ul)) &&
         rpi1.sender_info_.data_payload_.size() ==
             (3 + 4 + 4 + 5 + 45 + 1) / 2 + (5 + 45) * 2 &&
         rpi1.sender_info_.data_payload_[0] ==
             (((3ul + 4ul + 4ul + 5ul + 45ul + 1ul) / 2ul) << 32 |
              ((5ul + 45ul) * 2ul)));

  rpi0.receiver_info_.data_payload_.swap(rpi1.sender_info_.data_payload_);
  rpi1.sender_info_.data_payload_.clear();
  rpi1.receiver_info_.data_payload_.swap(rpi0.sender_info_.data_payload_);
  rpi0.sender_info_.data_payload_.clear();

  const auto reconstruct_start = std::chrono::high_resolution_clock::now();
  reconstruct_received_info(rpi0.receiver_info_,
                            rpi0.remote_indexed_coord_cache_);
  reconstruct_received_info(rpi1.receiver_info_,
                            rpi1.remote_indexed_coord_cache_);
  const auto reconstruct_end = std::chrono::high_resolution_clock::now();
  const auto reconstruct_time = reconstruct_end - reconstruct_start;
  printf("Time received info reconstruct: %f\n",
         reconstruct_time.count() / 1e9);

  assert(rpi0.sender_info_ == rpi1.receiver_info_ &&
         rpi0.receiver_info_ == rpi1.sender_info_);

  DistributedPairInfo<Coord2D> dpi;

  const auto dto_start = std::chrono::high_resolution_clock::now();
#pragma omp parallel default(shared)
#pragma omp master
#pragma omp taskgroup
  {
    distributed_overlap<Coord2D, true, true, false, false, false>(
        dpi.source_side_info_, map0.at(gnb.local_rank_), map1, tg, gnb, gnc,
        mc_array, true);
    distributed_overlap<Coord2D, true, true, true, false, false>(
        dpi.target_side_info_, map1.at(gnb.local_rank_), map0, tg, gnb, gnc,
        mc_array, true);
  }
  const auto dto_end = std::chrono::high_resolution_clock::now();
  const auto dto_time = dto_end - dto_start;
  printf("Time distributed overlap: %f\n", dto_time.count() / 1e9);

  assert(dpi.source_side_info_.size() == 1 &&
         dpi.source_side_info_.find(gnb.local_rank_) !=
             dpi.source_side_info_.end() &&
         dpi.target_side_info_.size() == 1 &&
         dpi.target_side_info_.find(gnb.local_rank_) !=
             dpi.target_side_info_.end());

  CommunicationInfo<Coord2D> dpi_sender_source(
      std::move(dpi.source_side_info_.at(gnb.local_rank_).sender_info_));
  CommunicationInfo<Coord2D> dpi_sender_target(
      std::move(dpi.target_side_info_.at(gnb.local_rank_).sender_info_));

  // Pointer address comparison
  assert(dpi_sender_source.filtered_coords_ ==
             rpi0.sender_info_.filtered_coords_ &&
         dpi_sender_target.filtered_coords_ ==
             rpi1.sender_info_.filtered_coords_);

  CommunicationInfo<Coord2D> dpi_receiver_source(
      std::move(dpi.source_side_info_.at(gnb.local_rank_).receiver_info_));
  CommunicationInfo<Coord2D> dpi_receiver_target(
      std::move(dpi.target_side_info_.at(gnb.local_rank_).receiver_info_));

  dpi_receiver_source.data_payload_.swap(dpi_sender_target.data_payload_);
  dpi_sender_target.data_payload_.clear();

  dpi_receiver_target.data_payload_.swap(dpi_sender_source.data_payload_);
  dpi_sender_source.data_payload_.clear();

  RemoteIndexedCoordCache<Coord2D> recv_cache_source;
  RemoteIndexedCoordCache<Coord2D> recv_cache_target;

  reconstruct_received_info(dpi_receiver_source, recv_cache_source);
  reconstruct_received_info(dpi_receiver_target, recv_cache_target);

  assert(dpi_receiver_source == rpi0.receiver_info_ &&
         dpi_receiver_target == rpi1.receiver_info_);
}

void test_rank_aggregation() {
  const vp_t local_rank = get_mpi_rank();
  const tileidx_t num_processes =
      static_cast<tileidx_t>(get_num_mpi_processes());

  NodeCountVector total_nodes_per_rank(num_processes, 0);
  RandomManager rman;
  rman.set_rng_seed(12345u);

  const auto init_start = std::chrono::high_resolution_clock::now();

  TileGrid<Coord2D> tg;
  GridNeighborhood gnb;
  GridNodeCollection<Coord2D> gnc;
  TAArray<GFCollection<Coord2D>> gc_array;
  CreatorRegistry<GridTargetPositionShifts<Coord2D>> gscr;
  CreatorRegistry<ShiftedOriginCreator<Coord2D>> socr;
  CreatorRegistry<CachedTileCreator<Coord2D>> ctcr;

  initialize_gsc_registry(gscr);
  initialize_soc_registry(socr);
  initialize_ctc_registry(ctcr);

  const auto gc = construct_gf_collection<Coord2D>(
      {0., 0.}, {2, 2 * num_processes}, "hexagon", {1.}, {0}, gscr, socr, ctcr);
  gc_array.clone(gc);
  tg = generate_tile_grid(gc_array);

  std::vector<std::set<tileidx_t>> rank_tile_ownership_map(num_processes);
  for (tileidx_t rank = 0; rank < num_processes; ++rank)
    rank_tile_ownership_map[rank] = {rank, num_processes + rank,
                                     2 * num_processes + rank,
                                     3 * num_processes + rank};
  gnb.set_tile_ownership(std::move(rank_tile_ownership_map), tg);
  split_tiles_in_grid(tg, *gnb.locally_owned_tiles_, 10);
  initialize_local_grid_node_collection(gnc, *gnb.locally_owned_tiles_, tg);

  const auto init_end = std::chrono::high_resolution_clock::now();
  const auto init_time = init_end - init_start;
  printf("Time grid manager init: %f\n", init_time.count() / 1e9);

  std::vector<DistributedTiledNodeSequenceMap> dist_maps(4);

  const auto node_gen_start = std::chrono::high_resolution_clock::now();
  for (auto &dist_map : dist_maps) {
    const auto aggregated_node_counts = distribute_node_counts_in_grid(
        250000, {}, tg, gnb, rman, DISTRIBUTION_MODE::FREE);

    RankNodeSequenceMap rns_map;
    for (tileidx_t rank = 0; rank < num_processes; ++rank) {
      const auto num_nodes = aggregated_node_counts.first[rank];
      if (num_nodes < 1)
        continue;

      const auto next_index = total_nodes_per_rank[rank];
      assert_emplace(rns_map.emplace(
          std::make_pair(vp_t(rank), NodeSequence(next_index, num_nodes))));
      total_nodes_per_rank[rank] += num_nodes;
    }

    dist_map = consolidate_node_sequences_per_tile_per_rank(
        rns_map, aggregated_node_counts.second);

    if (const auto local_tns = dist_map.find(local_rank);
        local_tns != dist_map.end())
      distribute_node_counts_in_tiles(gnc, local_tns->second, tg, rman,
                                      DISTRIBUTION_MODE::FREE);
  }
  assert(std::accumulate(total_nodes_per_rank.cbegin(),
                         total_nodes_per_rank.cend(), 0) == 1e6);
  assert_equal_across_ranks(total_nodes_per_rank);

  const auto node_gen_end = std::chrono::high_resolution_clock::now();
  const auto node_gen_time = node_gen_end - node_gen_start;
  printf("Time node gen: %f\n", node_gen_time.count() / 1e9);

  std::vector<DistributedPairInfo<Coord2D>> dpi_vec(16);
  auto dpi_it = dpi_vec.begin();

  MaskCollection<Coord2D> mc;
  mc.blueprint_ = construct_mask<Coord2D>(MASK_SHAPE::CIRCULAR, {}, {0.1}, {});
  TAArray<MaskCollection<Coord2D>> mc_array;
  mc_array.clone(mc);

  const auto dto_start = std::chrono::high_resolution_clock::now();
#pragma omp parallel default(shared)
#pragma omp master
  for (const auto &dist_map0 : dist_maps)
    for (const auto &dist_map1 : dist_maps)
      compute_distributed_tile_overlap<Coord2D, true, true>(
          *dpi_it++, dist_map0, dist_map1, tg, gnb, gnc, mc_array, true);
  const auto dto_end = std::chrono::high_resolution_clock::now();
  const auto dto_time = dto_end - dto_start;
  printf("Time all to all distributed overlap: %f\n", dto_time.count() / 1e9);
}

int main(int argc, char **argv) {
  init_mpi(&argc, &argv);
  init_omp(5);

  test_leaf_tiles_within_range();
  test_multi_image_aggregation();
  test_single_rank_aggregation();
  test_rank_aggregation();

  finalize_mpi();
  return 0;
}
