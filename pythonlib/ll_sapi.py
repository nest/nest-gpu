#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

CTypes bindings for low level Spatial API.

Authors: JoseJVS.

"""

import ctypes
import pathlib
import sys
import typing
import functools

NP = None

try:
    import numpy as np

    NP = np
except ImportError:
    pass


vp_t: typing.TypeAlias = ctypes.c_int32
nix_t: typing.TypeAlias = ctypes.c_int32
lnix_t: typing.TypeAlias = ctypes.c_int64
tix_t: typing.TypeAlias = ctypes.c_int32
dim_t: typing.TypeAlias = ctypes.c_uint8
split_t: typing.TypeAlias = ctypes.c_uint8
count_t: typing.TypeAlias = ctypes.c_int32
angle_t: typing.TypeAlias = ctypes.c_int16
space_t: typing.TypeAlias = ctypes.c_double
conn_index_t: typing.TypeAlias = ctypes.c_uint32
conn_param_t: typing.TypeAlias = ctypes.c_float
rng_seed_t: typing.TypeAlias = ctypes.c_uint32
CData: typing.TypeAlias = ctypes._SimpleCData | ctypes.Structure | ctypes._Pointer


class SpatialNodeSeq:
    def __init__(
        self,
        spatial_index: int,
        total_length: int,
        local_index: int | None = None,
        local_length: int | None = None,
    ) -> None:
        self._spatial_index = spatial_index
        self._total_length = total_length
        self._local_index = local_index
        self._local_length = local_length

    @property
    def spatial_index(self) -> int:
        return self._spatial_index

    @spatial_index.setter
    def spatial_index(self, **_) -> typing.NoReturn:
        raise AttributeError("Cannot set spatial index")

    @property
    def total_length(self) -> int:
        return self._total_length

    @total_length.setter
    def total_length(self, **_) -> typing.NoReturn:
        raise AttributeError("Cannot set total length")

    @property
    def local_index(self) -> int | None:
        return self._local_index

    @local_index.setter
    def local_index(self, **_) -> typing.NoReturn:
        raise AttributeError("Cannot set local index")

    @property
    def local_length(self) -> int | None:
        return self._local_length

    @local_length.setter
    def local_length(self, **_) -> typing.NoReturn:
        raise AttributeError("Cannot set local length")


class NodesViewStruct(ctypes.Structure):
    _fields_ = [
        ("dimensions_", dim_t),
        ("node_count_", ctypes.c_size_t),
        ("indexes_", ctypes.POINTER(nix_t)),
        ("coordinates_", ctypes.POINTER(space_t)),
    ]

    def from_tuple(
        self,
        t: typing.Tuple[
            typing.Sequence[int],
            typing.Sequence[typing.Sequence[float]],
        ],
    ) -> None:
        if len(t) < 1:
            self.dimensions_ = 0
            self.node_count_ = 0
            self.indexes_ = None
            self.coordinates_ = None
            return

        elif len(t) != 2:
            raise ValueError("Invalid nodes coords tuple")

        indexes_length = len(t[0])
        dimensions_length = len(t[1])
        if (
            indexes_length < 1
            or dimensions_length < 2
            or 3 < dimensions_length
            or any(len(c) != indexes_length for c in t[1])
        ):
            raise ValueError("Invalid nodes coords tuple")

        try:
            self.dimensions_ = dimensions_length
            self.node_count_ = indexes_length
            self.indexes_ = (nix_t * indexes_length)()
            self.coordinates_ = (space_t * (indexes_length * dimensions_length))()

            c_count = 0
            fill_once = True
            for d in range(dimensions_length):
                if fill_once:
                    fill_once = False
                    for i in range(indexes_length):
                        self.indexes_[i] = t[0][i]
                        self.coordinates_[c_count] = t[1][d][i]
                        c_count += 1
                else:
                    for i in range(indexes_length):
                        self.coordinates_[c_count] = t[1][d][i]
                        c_count += 1

            if c_count != dimensions_length * indexes_length:
                raise ValueError("Error converting nodes coords from tuple")

        except Exception as e:
            self.dimensions_ = 0
            self.node_count_ = 0
            self.indexes_ = None
            self.coordinates_ = None
            raise e

    def to_tuple(
        self,
    ) -> typing.Tuple[
        typing.List[int],
        typing.List[typing.List[float]],
    ]:
        if (0 < self.node_count_) != (0 < self.dimensions_):
            raise ValueError("Corrupted NodesViewStruct")

        elif self.node_count_ < 1:
            if bool(self.indexes_) or bool(self.coordinates_):
                raise ValueError("Corrupted NodesViewStruct")
            return [], []

        elif not (bool(self.indexes_) and bool(self.coordinates_)):
            raise ValueError("Corrupted NodesViewStruct")

        indexes = [0] * self.node_count_
        coordinates = [[]] * self.dimensions_

        c_count = 0
        fill_once = True
        for d in range(self.dimensions_):
            coordinates[d] = [0] * self.node_count_
            if fill_once:
                fill_once = False
                for n in range(self.node_count_):
                    indexes[n] = self.indexes_[n]
                    coordinates[d][n] = self.coordinates_[c_count]
                    c_count += 1
            else:
                for n in range(self.node_count_):
                    coordinates[d][n] = self.coordinates_[c_count]
                    c_count += 1

        if c_count != self.dimensions_ * self.node_count_:
            raise ValueError("Error converting nodes coords to tuple")

        return indexes, coordinates

    def to_np_data(self) -> tuple:
        if NP is None:
            raise RuntimeError("Cannot create node views without Numpy")

        elif (0 < self.node_count_) != (0 < self.dimensions_):
            raise ValueError("Corrupted NodesViewStruct")

        elif self.node_count_ < 1:
            if bool(self.indexes_) or bool(self.coordinates_):
                raise ValueError("Corrupted NodesViewStruct")
            return NP.empty((0,)), NP.empty((0,))

        elif not (bool(self.indexes_) and bool(self.coordinates_)):
            raise ValueError("Corrupted NodesViewStruct")

        return NP.ctypeslib.as_array(
            self.indexes_, (self.node_count_,)
        ), NP.ctypeslib.as_array(
            self.coordinates_, (self.dimensions_ * self.node_count_,)
        ).reshape((self.dimensions_, self.node_count_), copy=False)


class PositionViewStruct(ctypes.Structure):
    _fields_ = [
        ("dimensions_", dim_t),
        ("coord_count_", ctypes.c_size_t),
        ("coordinates_", ctypes.POINTER(space_t)),
    ]

    def from_tuple(
        self,
        t: typing.Sequence[typing.Sequence[float]],
    ) -> None:
        if len(t) < 1:
            self.dimensions_ = 0
            self.node_count_ = 0
            self.coordinates_ = None
            return

        pos_length = len(t)
        dimensions_length = len(t[0])
        if (
            dimensions_length < 2
            or 3 < dimensions_length
            or any(len(c) != dimensions_length for c in t)
        ):
            raise ValueError("Mismatched dimensions in position sequence")

        try:
            self.dimensions_ = dimensions_length
            self.coord_count_ = pos_length * dimensions_length
            self.coordinates_ = (space_t * self.coord_count_)()

            c_count = 0
            for coord_tuple in t:
                for coord in coord_tuple:
                    self.coordinates_[c_count] = coord
                    c_count += 1

            if c_count != self.coord_count_:
                raise ValueError("Error converting nodes coords from tuple")

        except Exception as e:
            self.dimensions_ = 0
            self.node_count_ = 0
            self.indexes_ = None
            self.coordinates_ = None
            raise e

    def to_tuple(self) -> typing.List[typing.List[float]]:
        if self.dimensions_ < 2 or 3 < self.dimensions_:
            raise ValueError("Corrupted PositionViewStruct")

        elif self.coord_count_ < 1:
            if bool(self.coordinates_):
                raise ValueError("Corrupted PositionViewStruct")
            return []

        elif not bool(self.coordinates_):
            raise ValueError("Corrupted PositionViewStruct")

        total_positions = int(self.coord_count_ / self.dimensions_)
        coordinates = [[]] * total_positions

        c_count = 0
        for pos in range(total_positions):
            coordinates[pos] = [0] * self.dimensions_
            for d in range(self.dimensions_):
                coordinates[pos][d] = self.coordinates_[c_count]
                c_count += 1

        if c_count != self.coord_count_:
            raise ValueError("Error converting positions to tuples")

        return coordinates


class ConnectionViewStruct(ctypes.Structure):
    _fields_ = [
        ("num_partitions_", ctypes.c_size_t),
        ("partition_sizes_", ctypes.POINTER(ctypes.c_size_t)),
        ("sources_", ctypes.POINTER(ctypes.POINTER(conn_index_t))),
        ("targets_", ctypes.POINTER(ctypes.POINTER(conn_index_t))),
        ("weights_", ctypes.POINTER(ctypes.POINTER(conn_param_t))),
        ("delays_", ctypes.POINTER(ctypes.POINTER(conn_param_t))),
    ]

    def from_tuple(
        self,
        t: typing.Sequence[
            typing.Tuple[
                typing.Sequence[int],
                typing.Sequence[int],
                typing.Sequence[float],
                typing.Sequence[float],
            ]
        ],
    ) -> None:
        num_partitions = len(t)
        if num_partitions < 1:
            self.num_partitions_ = 0
            self.partition_sizes_ = None
            self.sources_ = None
            self.targets_ = None
            self.weights_ = None
            self.delays_ = None
            return

        try:
            self.num_partitions_ = num_partitions
            self.partition_sizes_ = (ctypes.c_size_t * num_partitions)()
            self.sources_ = (ctypes.POINTER(conn_index_t) * num_partitions)()
            self.targets_ = (ctypes.POINTER(conn_index_t) * num_partitions)()
            self.weights_ = (ctypes.POINTER(conn_param_t) * num_partitions)()
            self.delays_ = (ctypes.POINTER(conn_param_t) * num_partitions)()

            for i, partition in enumerate(t):
                if len(partition) != 4:
                    raise ValueError("Invalid connection partition sequence")

                partition_size = len(partition[0])
                if partition_size < 1 or any(
                    len(p) != partition_size for p in partition
                ):
                    raise ValueError("Invalid connection partition sequence")

                self.partition_sizes_[i] = partition_size
                self.sources_[i] = (conn_index_t * partition_size)()
                self.targets_[i] = (conn_index_t * partition_size)()
                self.weights_[i] = (conn_param_t * partition_size)()
                self.delays_[i] = (conn_param_t * partition_size)()

                c_sources, c_targets, c_weights, c_delays = (
                    self.sources_[i],
                    self.targets_[i],
                    self.weights_[i],
                    self.delays_[i],
                )
                p_sources, p_targets, p_weights, p_delays = partition

                for p in range(partition_size):
                    c_sources[p] = p_sources[p]
                    c_targets[p] = p_targets[p]
                    c_weights[p] = p_weights[p]
                    c_delays[p] = p_delays[p]

        except Exception as e:
            self.num_partitions_ = 0
            self.partition_sizes_ = None
            self.sources_ = None
            self.targets_ = None
            self.weights_ = None
            self.delays_ = None
            raise e

    def to_tuple(
        self,
    ) -> typing.List[
        typing.Tuple[
            typing.List[int],
            typing.List[int],
            typing.List[float],
            typing.List[float],
        ]
    ]:
        if self.num_partitions_ < 1:
            if (
                bool(self.partition_sizes_)
                or bool(self.sources_)
                or bool(self.targets_)
                or bool(self.weights_)
                or bool(self.delays_)
            ):
                raise ValueError("Corrupted ConnectionViewStruct")
            return []

        elif not (
            bool(self.partition_sizes_)
            and bool(self.sources_)
            and bool(self.targets_)
            and bool(self.weights_)
            and bool(self.delays_)
        ):
            raise ValueError("Corrupted ConnectionViewStruct")

        res = [tuple([])] * self.num_partitions_
        for i in range(self.num_partitions_):
            check_ptr(self.sources_[i])
            check_ptr(self.targets_[i])
            check_ptr(self.weights_[i])
            check_ptr(self.delays_[i])

            partition_size = self.partition_sizes_[i]
            if partition_size < 1:
                raise ValueError("Corrupted partition size")

            c_sources, c_targets, c_weights, c_delays = (
                self.sources_[i],
                self.targets_[i],
                self.weights_[i],
                self.delays_[i],
            )

            p_sources, p_targets, p_weights, p_delays = (
                [0] * partition_size,
                [0] * partition_size,
                [0] * partition_size,
                [0] * partition_size,
            )

            for p in range(partition_size):
                p_sources[p] = c_sources[p]
                p_targets[p] = c_targets[p]
                p_weights[p] = c_weights[p]
                p_delays[p] = c_delays[p]

            res[i] = (p_sources, p_targets, p_weights, p_delays)

        return res

    def to_np_data(self) -> typing.List[tuple]:
        if NP is None:
            raise RuntimeError("Cannot create connection views without Numpy")

        elif self.num_partitions_ < 1:
            if (
                bool(self.partition_sizes_)
                or bool(self.sources_)
                or bool(self.targets_)
                or bool(self.weights_)
                or bool(self.delays_)
            ):
                raise ValueError("Corrupted ConnectionViewStruct")
            return []

        elif not (
            bool(self.partition_sizes_)
            and bool(self.sources_)
            and bool(self.targets_)
            and bool(self.weights_)
            and bool(self.delays_)
        ):
            raise ValueError("Corrupted ConnectionViewStruct")

        res = [tuple([])] * self.num_partitions_
        for i in range(self.num_partitions_):
            check_ptr(self.sources_[i])
            check_ptr(self.targets_[i])
            check_ptr(self.weights_[i])
            check_ptr(self.delays_[i])

            partition_size = self.partition_sizes_[i]
            if partition_size < 1:
                raise ValueError("Corrupted partition size")

            c_sources, c_targets, c_weights, c_delays = (
                self.sources_[i],
                self.targets_[i],
                self.weights_[i],
                self.delays_[i],
            )

            res[i] = (
                NP.ctypeslib.as_array(c_sources, (partition_size,)),
                NP.ctypeslib.as_array(c_targets, (partition_size,)),
                NP.ctypeslib.as_array(c_weights, (partition_size,)),
                NP.ctypeslib.as_array(c_delays, (partition_size,)),
            )

        return res


class ConnectionCountsViewStruct(ctypes.Structure):
    _fields_ = [
        ("incoming_ranks_", ctypes.c_size_t),
        ("outgoing_ranks_", ctypes.c_size_t),
        ("source_ranks_", ctypes.POINTER(vp_t)),
        ("incoming_counts_", ctypes.POINTER(ctypes.c_size_t)),
        ("target_ranks_", ctypes.POINTER(vp_t)),
        ("outgoing_counts_", ctypes.POINTER(ctypes.c_size_t)),
    ]

    def from_tuple(
        self,
        t: typing.Tuple[
            typing.Tuple[typing.Sequence[int], typing.Sequence[int]],
            typing.Tuple[typing.Sequence[int], typing.Sequence[int]],
        ],
    ) -> None:
        if len(t) != 2 or len(t[0]) != 2 or len(t[1]) != 2:
            raise ValueError("Invalid connection count tuple")

        incoming_ranks = len(t[0][0])
        outgoing_ranks = len(t[1][0])
        if len(t[0][1]) != incoming_ranks or len(t[1][1]) != outgoing_ranks:
            raise ValueError("Invalid connection count tuple")

        if 0 < incoming_ranks:
            try:
                self.incoming_ranks_ = incoming_ranks
                self.source_ranks_ = (vp_t * incoming_ranks)()
                self.incoming_counts_ = (ctypes.c_size_t * incoming_ranks)()
                for i, rank in enumerate(t[0][0]):
                    self.source_ranks_[i] = rank
                    self.incoming_counts_[i] = t[0][1][i]

            except Exception as e:
                self.incoming_ranks_ = 0
                self.source_ranks_ = None
                self.incoming_counts_ = None
                raise e
        else:
            self.incoming_ranks_ = 0
            self.source_ranks_ = None
            self.incoming_counts_ = None

        if 0 < outgoing_ranks:
            try:
                self.outgoing_ranks_ = outgoing_ranks
                self.target_ranks_ = (vp_t * outgoing_ranks)()
                self.outgoing_counts_ = (ctypes.c_size_t * outgoing_ranks)()
                for i, rank in enumerate(t[1][0]):
                    self.target_ranks_[i] = rank
                    self.outgoing_counts_[i] = t[1][1][i]

            except Exception as e:
                self.outgoing_ranks_ = 0
                self.target_ranks_ = None
                self.outgoing_counts_ = None
                raise e
        else:
            self.outgoing_ranks_ = 0
            self.target_ranks_ = None
            self.outgoing_counts_ = None

    def to_tuple(
        self,
    ) -> typing.Tuple[
        typing.Tuple[typing.List[int], typing.List[int]],
        typing.Tuple[typing.List[int], typing.List[int]],
    ]:
        res = (([], []), ([], []))

        if 0 < self.incoming_ranks_:
            if not (bool(self.source_ranks_) and bool(self.incoming_counts_)):
                raise ValueError("Corrupted connection count view")

            for i in range(self.incoming_ranks_):
                res[0][0].append(self.source_ranks_[i])
                res[0][1].append(self.incoming_counts_[i])

        elif bool(self.source_ranks_) or bool(self.incoming_counts_):
            raise ValueError("Corrupted connection count view")

        if 0 < self.outgoing_ranks_:
            if not (bool(self.target_ranks_) and bool(self.outgoing_counts_)):
                raise ValueError("Corrupted connection count view")

            for i in range(self.outgoing_ranks_):
                res[1][0].append(self.target_ranks_[i])
                res[1][1].append(self.outgoing_counts_[i])

        elif bool(self.target_ranks_) or bool(self.outgoing_counts_):
            raise ValueError("Corrupted connection count view")

        return res

    def to_np_data(self) -> typing.Tuple[tuple, tuple]:
        if NP is None:
            raise RuntimeError("Cannot create connection views without Numpy")

        source_ranks = NP.empty(0)
        incoming_counts = NP.empty(0)
        target_ranks = NP.empty(0)
        outgoing_counts = NP.empty(0)

        if 0 < self.incoming_ranks_:
            if not (bool(self.source_ranks_) and bool(self.incoming_counts_)):
                raise ValueError("Corrupted connection count view")

            source_ranks = NP.ctypeslib.as_array(
                self.source_ranks_, (self.incoming_ranks_,)
            )
            incoming_counts = NP.ctypeslib.as_array(
                self.incoming_counts_, (self.incoming_ranks_,)
            )

        elif bool(self.source_ranks_) or bool(self.incoming_counts_):
            raise ValueError("Corrupted connection count view")

        if 0 < self.outgoing_ranks_:
            if not (bool(self.target_ranks_) and bool(self.outgoing_counts_)):
                raise ValueError("Corrupted connection count view")

            target_ranks = NP.ctypeslib.as_array(
                self.target_ranks_, (self.outgoing_ranks_,)
            )
            outgoing_counts = NP.ctypeslib.as_array(
                self.outgoing_counts_, (self.outgoing_ranks_,)
            )

        elif bool(self.target_ranks_) or bool(self.outgoing_counts_):
            raise ValueError("Corrupted connection count view")

        return ((source_ranks, incoming_counts), (target_ranks, outgoing_counts))


class GridViewStruct(ctypes.Structure):
    _fields_ = [
        ("dimensions_", dim_t),
        ("num_tiles_", ctypes.c_size_t),
        ("leaves_per_tile_", ctypes.c_size_t),
        ("vertices_per_tile_", ctypes.c_size_t),
        ("vertices_per_leaf_", ctypes.c_size_t),
        ("tile_indexes_", ctypes.POINTER(tix_t)),
        ("tile_vertices_", ctypes.POINTER(space_t)),
        ("leaf_vertices_", ctypes.POINTER(space_t)),
    ]

    def from_tuple(
        self,
        dimensions: int,
        num_tiles: int,
        leaves_per_tile: int,
        vertices_per_tile: int,
        vertices_per_leaf: int,
        t: typing.Tuple[
            typing.Sequence[int],
            typing.Sequence[typing.Sequence[typing.Sequence[float]]],
            typing.Sequence[typing.Sequence[typing.Sequence[typing.Sequence[float]]]],
        ],
    ) -> None:
        sizes = (
            dimensions,
            num_tiles,
            leaves_per_tile,
            vertices_per_tile,
            vertices_per_leaf,
        )
        if all(s < 1 for s in sizes):
            if len(t) < 1:
                raise ValueError("Invalid grid vertices tuple")
            self.dimensions_ = 0
            self.num_tiles_ = 0
            self.leaves_per_tile_ = 0
            self.vertices_per_tile_ = 0
            self.vertices_per_leaf_ = 0
            self.tile_indexes_ = None
            self.tile_vertices_ = None
            self.leaf_vertices_ = None
            return

        elif any(s < 1 for s in sizes):
            raise ValueError("Invalid sizes for grid vertex conversion")

        elif len(t) != 3:
            raise ValueError("Invalid grid vertices tuple")

        total_tile_size = num_tiles * vertices_per_tile * dimensions
        total_leaf_size = num_tiles * leaves_per_tile * vertices_per_leaf * dimensions

        if (0 < total_leaf_size) != (0 < total_tile_size):
            raise ValueError("Invalid sizes for grid vertex conversion")

        if any(len(t[i]) != num_tiles for i in range(3)):
            raise ValueError("Invalid grid vertices tuple")

        try:
            self.dimensions_ = dimensions
            self.num_tiles_ = num_tiles
            self.leaves_per_tile_ = leaves_per_tile
            self.vertices_per_tile_ = vertices_per_tile
            self.vertices_per_leaf_ = vertices_per_leaf
            self.tile_indexes_ = (tix_t * num_tiles)()
            self.tile_vertices_ = (space_t * total_tile_size)()
            self.leaf_vertices_ = (space_t * total_leaf_size)()

            tv_count = 0
            lv_count = 0
            for i, tile_index in enumerate(t[0]):
                self.tile_indexes_[i] = tile_index
                for t_vertices in t[1][i]:
                    for coord in t_vertices:
                        self.tile_vertices_[tv_count] = coord
                        tv_count += 1
                for leaves in t[2][i]:
                    for l_vertices in leaves:
                        for coord in l_vertices:
                            self.leaf_vertices_[lv_count] = coord
                            lv_count += 1

            if tv_count != total_tile_size or lv_count != total_leaf_size:
                raise ValueError("Error during grid vertex conversion")

        except Exception as e:
            self.dimensions_ = 0
            self.num_tiles_ = 0
            self.leaves_per_tile_ = 0
            self.vertices_per_tile_ = 0
            self.vertices_per_leaf_ = 0
            self.tile_indexes_ = None
            self.tile_vertices_ = None
            self.leaf_vertices_ = None
            raise e

    def to_tuple(
        self,
    ) -> typing.Tuple[
        typing.List[int],
        typing.List[typing.List[typing.List[float]]],
        typing.List[typing.List[typing.List[typing.List[float]]]],
    ]:
        total_tile_size = self.num_tiles_ * self.vertices_per_tile_ * self.dimensions_
        total_leaf_size = (
            self.num_tiles_
            * self.leaves_per_tile_
            * self.vertices_per_leaf_
            * self.dimensions_
        )

        if (0 < total_leaf_size) != (0 < total_tile_size):
            raise ValueError("Corrupted GridViewStruct")

        elif total_tile_size < 1:
            if (
                bool(self.tile_indexes_)
                or bool(self.tile_vertices_)
                or bool(self.leaf_vertices_)
            ):
                raise ValueError("Corrupted GridViewStruct")
            return [], [], []

        elif not (
            bool(self.tile_indexes_)
            and bool(self.tile_vertices_)
            and bool(self.leaf_vertices_)
        ):
            raise ValueError("Corrupted GridViewStruct")

        t_indexes = [0] * self.num_tiles_
        t_vertices = [[]] * self.num_tiles_
        l_vertices = [[]] * self.num_tiles_

        tv_count = 0
        lv_count = 0
        for t in range(self.num_tiles_):
            t_indexes[t] = self.tile_indexes_[t]
            t_vertices[t] = [[]] * self.vertices_per_tile_
            for v in range(self.vertices_per_tile_):
                t_vertices[t][v] = [0] * self.dimensions_
                for c in range(self.dimensions_):
                    t_vertices[t][v][c] = self.tile_vertices_[tv_count]
                    tv_count += 1
            l_vertices[t] = [[]] * self.leaves_per_tile_
            for l in range(self.leaves_per_tile_):
                l_vertices[t][l] = [[]] * self.vertices_per_leaf_
                for v in range(self.vertices_per_leaf_):
                    l_vertices[t][l][v] = [0] * self.dimensions_
                    for c in range(self.dimensions_):
                        l_vertices[t][l][v][c] = self.leaf_vertices_[lv_count]
                        lv_count += 1

        if tv_count != total_tile_size or lv_count != total_leaf_size:
            raise ValueError("Error during grid vertex conversion")

        return t_indexes, t_vertices, l_vertices

    def to_np_data(self) -> tuple:
        if NP is None:
            raise RuntimeError("Cannot create node views without Numpy")

        total_tile_size = self.num_tiles_ * self.vertices_per_tile_ * self.dimensions_
        total_leaf_size = (
            self.num_tiles_
            * self.leaves_per_tile_
            * self.vertices_per_leaf_
            * self.dimensions_
        )

        if (0 < total_leaf_size) != (0 < total_tile_size):
            raise ValueError("Corrupted GridViewStruct")

        elif total_tile_size < 1:
            if (
                bool(self.tile_indexes_)
                or bool(self.tile_vertices_)
                or bool(self.leaf_vertices_)
            ):
                raise ValueError("Corrupted GridViewStruct")
            return NP.empty((0,)), NP.empty((0,)), NP.empty((0,))

        elif not (
            bool(self.tile_indexes_)
            and bool(self.tile_vertices_)
            and bool(self.leaf_vertices_)
        ):
            raise ValueError("Corrupted GridViewStruct")

        return (
            NP.ctypeslib.as_array(self.tile_indexes_, (self.num_tiles_,)),
            NP.ctypeslib.as_array(self.tile_vertices_, (total_tile_size,)).reshape(
                (self.num_tiles_, self.vertices_per_tile_, self.dimensions_), copy=False
            ),
            NP.ctypeslib.as_array(self.leaf_vertices_, (total_leaf_size,)).reshape(
                (
                    self.num_tiles_,
                    self.leaves_per_tile_,
                    self.vertices_per_leaf_,
                    self.dimensions_,
                ),
                copy=False,
            ),
        )


def pair_template(t0: type[CData], t1: type[CData]) -> type[ctypes.Structure]:
    class PairT(ctypes.Structure):
        _fields_ = [("first_", t0), ("second_", t1)]

    return PairT


def triplet_template(
    t0: type[CData],
    t1: type[CData],
    t2: type[CData],
) -> type[ctypes.Structure]:
    class TripletT(ctypes.Structure):
        _fields_ = [("first_", t0), ("second_", t1), ("third_", t2)]

    return TripletT


def array_template(t: type[CData]) -> type[ctypes.Structure]:
    class ArrayT(ctypes.Structure):
        _fields_ = [("size_", ctypes.c_size_t), ("array_", ctypes.POINTER(t))]
        array_type: type[CData] = t

        def resize(self, size: int) -> None:
            if size < 0:
                raise ValueError("Incorrect resize size")
            self.size_ = ctypes.c_size_t(size)
            self.array_ = None
            if size > 0:
                self.array_ = (self.array_type * size)()

    return ArrayT


def pair_array_template(k: type[CData], v: type[CData]) -> type[ctypes.Structure]:
    return array_template(pair_template(k, v))


OptionalIndex = pair_template(ctypes.c_bool, ctypes.c_size_t)
SpatialNodeSequence = triplet_template(ctypes.c_size_t, nix_t, nix_t)
CharArray = array_template(ctypes.c_char)
SpaceTArray = array_template(space_t)
TileIdxArray = array_template(tix_t)
AngleTArray = array_template(angle_t)
NodeIdxArray = array_template(nix_t)

NestedCharArray = array_template(CharArray)
NestedSpaceTArray = array_template(SpaceTArray)
NestedTileIdxArray = array_template(TileIdxArray)

TiledNodeSequencePairArray = pair_array_template(
    vp_t, pair_array_template(tix_t, pair_template(nix_t, nix_t))
)

DoubleArray = array_template(ctypes.c_double)
RankTimerDataPairArray = pair_array_template(CharArray, ctypes.c_double)
ThreadTimerDataPairArray = pair_array_template(CharArray, DoubleArray)
RecordedTimesArrayPair = pair_template(RankTimerDataPairArray, ThreadTimerDataPairArray)

ConnectionViewPairArray = pair_array_template(
    vp_t,
    ConnectionViewStruct,
)
RemoteConnectionViewPair = pair_template(
    ConnectionViewPairArray, ConnectionViewPairArray
)

ParameterNamesPairArray = pair_array_template(CharArray, NestedCharArray)


def check_bool(b_val: ctypes.c_bool) -> None:
    if not b_val:
        raise ValueError("is false")


def check_ptr(ptr: ctypes._Pointer) -> None:
    if not bool(ptr):
        raise ValueError("is nullptr")


def str_to_carr(string: str) -> ctypes.Structure:
    ca = CharArray()
    ca.resize(len(string) + 1)
    if ca.size_ > 1:
        ca.array_ = ctypes.create_string_buffer(
            string.encode("utf-8", "strict") + b"\0"
        )
    else:
        ca.array_[0] = b"\0"
    return ca


def carr_to_str(carr: ctypes.Structure) -> str:
    string = ""
    if 1 < carr.size_:
        check_ptr(carr.array_)
        string = (
            bytes()
            .join(carr.array_[i] for i in range(carr.size_))
            .decode("utf-8", "strict")
            .rstrip("\0")
        )
    elif (carr.size_ == 1 and (not bool(carr.array_) or carr.array_[0] != b"\0")) or (
        carr.size_ < 1 and bool(carr.array_)
    ):
        raise ValueError("Corrupted CharArray")
    return string


def str_seq_to_nested_carr(str_seq: typing.Sequence[str]) -> ctypes.Structure:
    nca = NestedCharArray()
    nca.resize(len(str_seq))
    for i, string in enumerate(str_seq):
        nca.array_[i] = str_to_carr(string)
    return nca


def nested_carr_to_str_seq(nested_carr: ctypes.Structure) -> typing.List[str]:
    l = []
    if 0 < nested_carr.size_:
        check_ptr(nested_carr.array_)
        l = [carr_to_str(nested_carr.array_[i]) for i in range(nested_carr.size_)]
    elif bool(nested_carr.array_):
        raise ValueError("Corrupted NestedCharray")
    return l


def num_seq_to_num_arr(
    num_arr_type: type[ctypes.Structure],
    num_seq: typing.Sequence[int] | typing.Sequence[float] | typing.Set[int],
) -> ctypes.Structure:
    arr = num_arr_type()
    arr.resize(len(num_seq))
    for i, v in enumerate(num_seq):
        arr.array_[i] = v
    return arr


def num_arr_to_num_seq(
    num_arr: ctypes.Structure,
) -> typing.List[int] | typing.List[float]:
    l = []
    if 0 < num_arr.size_:
        check_ptr(num_arr.array_)
        l = [num_arr.array_[i] for i in range(num_arr.size_)]
    elif bool(num_arr.array_):
        raise ValueError("Corrupted numerical array")
    return l


def nested_num_seq_to_nested_arr(
    nested_num_arr_type: type[ctypes.Structure],
    nested_num_seq: (
        typing.Sequence[typing.Sequence[int]]
        | typing.Sequence[typing.Sequence[float]]
        | typing.Sequence[typing.Set[int]]
    ),
) -> ctypes.Structure:
    na = nested_num_arr_type()
    na.resize(len(nested_num_seq))
    for i, num_seq in enumerate(nested_num_seq):
        na.array_[i] = num_seq_to_num_arr(na.array_type, num_seq)
    return na


def nested_num_arr_to_nested_num_seq(
    nested_num_arr: ctypes.Structure,
) -> typing.List[typing.List[int] | typing.List[float]]:
    l = []
    if 0 < nested_num_arr.size_:
        check_ptr(nested_num_arr.array_)
        l = [
            num_arr_to_num_seq(nested_num_arr.array_[i])
            for i in range(nested_num_arr.size_)
        ]
    elif bool(nested_num_arr.array_):
        raise ValueError("Corrupted nested numerical array")
    return l


_CONVERTERS = {
    ctypes.c_bool: (
        bool,
        ctypes.c_bool,
    ),
    split_t: (
        int,
        split_t,
    ),
    count_t: (
        int,
        count_t,
    ),
    nix_t: (
        int,
        nix_t,
    ),
    CharArray: (carr_to_str, str_to_carr),
    NestedCharArray: (nested_carr_to_str_seq, str_seq_to_nested_carr),
    SpaceTArray: (num_arr_to_num_seq, lambda seq: num_seq_to_num_arr(SpaceTArray, seq)),
    NestedSpaceTArray: (
        nested_num_arr_to_nested_num_seq,
        lambda nested_seq: nested_num_seq_to_nested_arr(NestedSpaceTArray, nested_seq),
    ),
    TileIdxArray: (
        num_arr_to_num_seq,
        lambda seq: num_seq_to_num_arr(TileIdxArray, seq),
    ),
    NestedTileIdxArray: (
        nested_num_arr_to_nested_num_seq,
        lambda nested_seq: nested_num_seq_to_nested_arr(NestedTileIdxArray, nested_seq),
    ),
    AngleTArray: (num_arr_to_num_seq, lambda seq: num_seq_to_num_arr(AngleTArray, seq)),
}

_GPS_FIELDS = (
    ("grid_origin", SpaceTArray, lambda: tuple(), lambda go: 2 <= len(go) <= 3),
    (
        "grid_dimensions",
        TileIdxArray,
        lambda: tuple(),
        lambda gd: 2 <= len(gd) <= 3 and all(0 < d < (1 << 32) for d in gd),
    ),
    ("tile_type", CharArray, lambda: "", "tile_shapes"),
    (
        "tile_side_lengths",
        SpaceTArray,
        lambda: tuple(),
        lambda tsl: 1 <= len(tsl) <= 3 and all(0 < sl for sl in tsl),
    ),
    (
        "tile_angular_offsets",
        AngleTArray,
        lambda: tuple(),
        lambda tao: all(-(1 << 16) < a < (1 << 16) for a in tao),
    ),
    (
        "compute_splits",
        ctypes.c_bool,
        lambda: False,
        lambda cs: 0 <= cs <= 1,
    ),
    ("num_splits", split_t, lambda: 0, lambda ns: 0 <= ns < (1 << 8)),
    ("expected_total_nodes", nix_t, lambda: 0, lambda etn: 0 <= etn < (1 << 32)),
    ("expected_nodes_per_leaf", nix_t, lambda: 0, lambda enpl: 0 <= enpl < (1 << 32)),
)

_MPS_FIELDS = (
    ("mask_blueprint_name", CharArray, lambda: "", "mask_shapes"),
    ("mask_blueprint_params", SpaceTArray, lambda: tuple(), None),
    (
        "mask_blueprint_offset",
        SpaceTArray,
        lambda: tuple(),
        lambda o: 2 <= len(o) <= 3,
    ),
    ("source_mask_name", CharArray, lambda: "", "mask_shapes"),
    (
        "source_mask_origin",
        SpaceTArray,
        lambda: tuple(),
        lambda o: 2 <= len(o) <= 3,
    ),
    ("source_mask_params", SpaceTArray, lambda: tuple(), None),
    (
        "source_mask_offset",
        SpaceTArray,
        lambda: tuple(),
        lambda o: 2 <= len(o) <= 3,
    ),
    ("target_mask_name", CharArray, lambda: "", "mask_shapes"),
    (
        "target_mask_origin",
        SpaceTArray,
        lambda: tuple(),
        lambda o: 2 <= len(o) <= 3,
    ),
    ("target_mask_params", SpaceTArray, lambda: tuple(), None),
    (
        "target_mask_offset",
        SpaceTArray,
        lambda: tuple(),
        lambda o: 2 <= len(o) <= 3,
    ),
)

_CPS_FIELDS = (
    ("edge_wrap", ctypes.c_bool, lambda: False, lambda cs: cs == True or cs == False),
    (
        "only_neighborhood",
        ctypes.c_bool,
        lambda: False,
        lambda cs: 0 <= cs <= 1,
    ),
    (
        "allow_multiplicity",
        ctypes.c_bool,
        lambda: False,
        lambda cs: 0 <= cs <= 1,
    ),
    (
        "allow_self_connections",
        ctypes.c_bool,
        lambda: False,
        lambda cs: 0 <= cs <= 1,
    ),
    (
        "partition_connections",
        ctypes.c_bool,
        lambda: False,
        lambda cs: 0 <= cs <= 1,
    ),
    ("connection_counts", count_t, lambda: 0, lambda cc: 0 <= cc < (1 << 32)),
    ("rule", CharArray, lambda: "", "connection_rules"),
    ("weight_df_name", CharArray, lambda: "", "displacement_functions"),
    ("weight_df_params", SpaceTArray, lambda: tuple(), None),
    ("weight_ufs_names", NestedCharArray, lambda: tuple(), "unary_functions"),
    ("weight_ufs_params", NestedSpaceTArray, lambda: tuple(), None),
    ("delay_df_name", CharArray, lambda: "", "displacement_functions"),
    ("delay_df_params", SpaceTArray, lambda: tuple(), None),
    ("delay_ufs_names", NestedCharArray, lambda: tuple(), "unary_functions"),
    ("delay_ufs_params", NestedSpaceTArray, lambda: tuple(), None),
    ("prob_df_name", CharArray, lambda: "", "displacement_functions"),
    ("prob_df_params", SpaceTArray, lambda: tuple(), None),
    ("prob_ufs_names", NestedCharArray, lambda: tuple(), "unary_functions"),
    ("prob_ufs_params", NestedSpaceTArray, lambda: tuple(), None),
)


def test_param(
    name: str,
    value: typing.Any,
    test: str | typing.Callable | None,
    param_names: typing.Mapping[str, typing.Sequence[str]],
):
    if test is not None:
        if isinstance(test, str):
            names = param_names[test]
            if not isinstance(value, str) and isinstance(value, typing.Iterable):
                for v in value:
                    if v not in names:
                        raise KeyError(f"Invalid name {v} for parameter {name}")
            elif value not in names:
                raise KeyError(f"Invalid name {value} for parameter {name}")
        elif not test(value):
            raise ValueError(f"Invalid value for parameter {name}")


def io_struct_template(
    field_map,
) -> type[ctypes.Structure]:
    class IOStruct(ctypes.Structure):
        _fields_ = [(field[0], field[1]) for field in field_map]
        _fmp = field_map

        def to_dict(self) -> dict:
            res = dict()
            for name, type, _, _ in self._fmp:
                try:
                    res[name] = _CONVERTERS[type][0](self.__getattribute__(name))
                except KeyError:
                    raise KeyError("Unknown field in structure")
            return res

        def from_dict(
            self, d: dict, param_names: typing.Mapping[str, typing.Sequence[str]]
        ) -> None:
            used_keys = []
            for name, type, dfg, test in self._fmp:
                if name in d:
                    val = d[name]
                    test_param(name, val, test, param_names)
                    self.__setattr__(name, _CONVERTERS[type][1](val))
                    used_keys.append(name)
                else:
                    self.__setattr__(
                        name,
                        _CONVERTERS[type][1](dfg()),
                    )

            for name in d:
                if name not in used_keys:
                    raise KeyError(f"Unknown key {name}")

    return IOStruct


GPStruct = io_struct_template(_GPS_FIELDS)
MPStruct = io_struct_template(_MPS_FIELDS)
CPStruct = io_struct_template(_CPS_FIELDS)


def dict_to_parameter_name_pair_array(
    d: typing.Mapping[str, typing.Sequence[str]],
) -> ctypes.Structure:
    pnpa = ParameterNamesPairArray()
    pnpa.resize(len(d))
    for p, (category, names) in enumerate(d.items()):
        entry = pnpa.array_[p]
        entry.first_ = str_to_carr(category)
        entry.second_ = str_seq_to_nested_carr(names)

    return pnpa


def parameter_name_pair_array_to_dict(
    pnpa: ctypes.Structure,  # ParameterNamesPairArray
) -> typing.Dict[str, typing.List[str]]:
    res = dict()
    if 0 < pnpa.size_:
        check_ptr(pnpa.array_)
        for p in range(pnpa.size_):
            entry = pnpa.array_[p]
            res[carr_to_str(entry.first_)] = nested_carr_to_str_seq(entry.second_)

        if len(res) != pnpa.size_:
            raise ValueError("Corrupted ParameterNamesPairArray")

    elif bool(pnpa.array_):
        raise ValueError("Corrupted ParameterNamesPairArray")

    return res


def dict_to_connection_view_pair_array(
    d: typing.Mapping[
        int,
        typing.Sequence[
            typing.Tuple[
                typing.Sequence[int],
                typing.Sequence[int],
                typing.Sequence[float],
                typing.Sequence[float],
            ]
        ],
    ],
) -> ctypes.Structure:  # ConnectionInfoPairArray
    cvpa = ConnectionViewPairArray()
    cvpa.resize(len(d))
    for r, (rank, conn_collection) in enumerate(d.items()):
        rank_conn_parr = cvpa.array_[r]
        rank_conn_parr.first_ = rank
        rank_conn_parr.second_.from_tuple(conn_collection)

    return cvpa


def connection_view_pair_array_to_dict(
    cvpa: ctypes.Structure,  # ConnectionViewPairArray
) -> typing.Dict[
    int,
    typing.List[
        typing.Tuple[
            typing.List[int],
            typing.List[int],
            typing.List[float],
            typing.List[float],
        ]
    ],
]:
    res = dict()
    if 0 < cvpa.size_:
        check_ptr(cvpa.array_)
        for r in range(cvpa.size_):
            rank_cvs = cvpa.array_[r]
            res[rank_cvs.first_] = rank_cvs.second_.to_tuple()

        if len(res) != cvpa.size_:
            raise ValueError("Corrupted connection info pair array")

    elif bool(cvpa.array_):
        raise ValueError("Corrupted connection info pair array")

    return res


def connection_view_pair_array_to_np_dict(
    cvpa: ctypes.Structure,  # ConnectionInfoPairArray
) -> typing.Dict[int, typing.List[tuple]]:
    res = dict()
    if 0 < cvpa.size_:
        check_ptr(cvpa.array_)
        for r in range(cvpa.size_):
            rank_cvs = cvpa.array_[r]
            res[rank_cvs.first_] = rank_cvs.second_.to_np_data()

        if len(res) != cvpa.size_:
            raise ValueError("Corrupted connection info pair array")

    elif bool(cvpa.array_):
        raise ValueError("Corrupted connection info pair array")

    return res


def dict_to_tiled_node_sequence_pair_array(
    d: typing.Mapping[
        int,  # MPI rank
        typing.Mapping[
            int,  # Tile index
            typing.Tuple[int, int],  # First node in sequence, length of sequence
        ],
    ],
) -> ctypes.Structure:  # TiledNodeSequencePairArray
    tnspa = TiledNodeSequencePairArray()
    tnspa.resize(len(d))
    for r, (rank, tix_ns_map) in enumerate(d.items()):
        rank_tns_parr = tnspa.array_[r]
        rank_tns_parr.first_ = rank
        rank_tns_parr.second_.resize(len(tix_ns_map))
        for t, (tix, ns) in enumerate(tix_ns_map.items()):
            tix_ns_pair = rank_tns_parr.second_.array_[t]
            tix_ns_pair.first_ = tix
            tix_ns_pair.second_.first_ = ns[0]
            tix_ns_pair.second_.second_ = ns[1]

    return tnspa


def tiled_node_sequence_pair_array_to_dict(
    tnspa: ctypes.Structure,  # TiledNodeSequencePairArray
) -> typing.Dict[
    int,  # MPI rank
    typing.Dict[
        int,  # Tile index
        typing.Tuple[int, int],  # First node in sequence, length of sequence
    ],
]:
    res = dict()
    if 0 < tnspa.size_:
        check_ptr(tnspa.array_)
        for r in range(tnspa.size_):
            rank_tns_parr = tnspa.array_[r]
            if 0 < rank_tns_parr.second_.size_:
                check_ptr(rank_tns_parr.second_.array_)
                rank_dict = res[rank_tns_parr.first_] = dict()
                for t in range(rank_tns_parr.second_.size_):
                    tix_ns_pair = rank_tns_parr.second_.array_[t]
                    rank_dict[tix_ns_pair.first_] = (
                        tix_ns_pair.second_.first_,
                        tix_ns_pair.second_.second_,
                    )

                if len(rank_dict) != rank_tns_parr.second_.size_:
                    raise ValueError("Corrupted tiled node sequence pair array")

            elif bool(rank_tns_parr.second_.array_):
                raise ValueError("Corrupted tiled node sequence pair array")

        if len(res) != tnspa.size_:
            raise ValueError("Corrupted rank tiled node sequence pair array")

    elif bool(tnspa.array_):
        raise ValueError("Corrupted rank tiled node sequence pair array")

    return res


def dict_to_rank_timer_data_pair_array(
    d: typing.Mapping[str, float],  # Timer name : Time
) -> ctypes.Structure:  # RankTimerDataPairArray
    rtdpa = RankTimerDataPairArray()
    rtdpa.resize(len(d))
    for i, (timer_name, time) in enumerate(d.items()):
        td = rtdpa.array_[i]
        td.first_ = str_to_carr(timer_name)
        td.second_ = time

    return rtdpa


def rank_timer_data_pair_array_to_dict(
    rtdpa: ctypes.Structure,  # RankTimerDataPairArray
) -> typing.Dict[str, float]:  # Timer name : Time
    res = dict()
    if 0 < rtdpa.size_:
        check_ptr(rtdpa.array_)
        for i in range(rtdpa.size_):
            td = rtdpa.array_[i]
            res[carr_to_str(td.first_)] = td.second_

        if len(res) != rtdpa.size_:
            raise ValueError("Corrupted rank timer data pair array")

    elif bool(rtdpa.array_):
        raise ValueError("Corrupted rank timer data pair array")

    return res


def dict_to_thread_timer_data_pair_array(
    d: typing.Mapping[str, typing.Sequence[float]],  # Timer name : Time
) -> ctypes.Structure:  # TimerDataPairArray
    ttdpa = ThreadTimerDataPairArray()
    ttdpa.resize(len(d))
    for i, (timer_name, times) in enumerate(d.items()):
        td = ttdpa.array_[i]
        td.first_ = str_to_carr(timer_name)
        td.second_ = num_seq_to_num_arr(DoubleArray, times)

    return ttdpa


def thread_timer_data_pair_array_to_dict(
    rtdpa: ctypes.Structure,  # TimerDataPairArray
) -> typing.Dict[str, typing.List[float]]:  # Timer name : Time
    res = dict()
    if 0 < rtdpa.size_:
        check_ptr(rtdpa.array_)
        for i in range(rtdpa.size_):
            td = rtdpa.array_[i]
            res[carr_to_str(td.first_)] = num_arr_to_num_seq(td.second_)

        if len(res) != rtdpa.size_:
            raise ValueError("Corrupted thread timer data pair array")

    elif bool(rtdpa.array_):
        raise ValueError("Corrupted thread timer data pair array")

    return res


def check_optional(opt: ctypes.Structure):
    check_bool(opt.first_)
    return int(opt.second_)


def safe_ptr_deref(ptr: ctypes._Pointer):
    check_ptr(ptr)
    return ptr.contents


def parse_distribution_mode(
    mode: str | int,
    param_names: typing.Mapping[str, typing.Sequence[str]],
) -> ctypes.c_uint8:
    if isinstance(mode, str):
        for i, name in enumerate(param_names["distribution_modes"]):
            if mode == name:
                return ctypes.c_uint8(i)
    elif isinstance(mode, int):
        if not (0 <= mode < len(param_names["distribution_modes"])):
            raise ValueError("Invalid distribution mode")
        return ctypes.c_uint8(mode)
    raise ValueError("Invalid distribution  mode")


def simple_prime_factorization(n: int) -> typing.Generator[int, None, None]:
    if n < 1:
        raise ValueError("Cannot factorize negative or null values")

    while n % 2 == 0:
        yield 2
        n //= 2

    d = 3
    while d * d <= n:
        while n % d == 0:
            yield d
            n //= d
        d += 2

    if n > 1:
        yield n


def largest_m_factors(n: int, m: int) -> typing.Sequence[int]:
    if m < 1:
        raise ValueError("Invalid factor count")
    factors = [1] * m
    for i, f in enumerate(simple_prime_factorization(n)):
        factors[i % m] *= f
    return factors


def check_rank_tile_ownership(
    rank_tile_ownership: str | typing.Sequence[typing.Set[int]],
    grid_dimensions: typing.Sequence[int],
    num_processes: int,
) -> typing.Sequence[typing.Set[int]]:
    total_tiles = functools.reduce(lambda x, y: x * y, grid_dimensions, 1)
    if isinstance(rank_tile_ownership, str):
        match rank_tile_ownership.lower():
            case "unique":
                if total_tiles != num_processes:
                    raise ValueError(
                        "Cannot generate an ownership map with a unique tile per rank with given dimensions"
                    )
                return [{i} for i in range(total_tiles)]
            case "round_robin":
                map = [set()] * num_processes
                for p in range(num_processes):
                    map[p] = set()
                for t in range(total_tiles):
                    map[t % num_processes].add(t)
                return map
            case "shared":
                return [{i for i in range(total_tiles)} for _ in range(num_processes)]
            case _:
                raise ValueError(
                    "Unknown ownership directive, possible are: unique, round_robin, shared"
                )
    else:
        if len(rank_tile_ownership) != num_processes:
            raise ValueError(
                "Ownership map must be equal in length to the number of processes"
            )
        if not all(
            all(0 <= t < total_tiles for t in tiles) for tiles in rank_tile_ownership
        ):
            raise ValueError("Invalid tile indexes in ownership map")
        return rank_tile_ownership
