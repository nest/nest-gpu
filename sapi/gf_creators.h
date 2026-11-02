/*
 *  gf_creators.h
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

#ifndef GF_CREATORS_H
#define GF_CREATORS_H

namespace sapi {
// Forward definition to creator_registry.h
template <typename RT> class CreatorRegistry;

// Forward definition to coordinates.h
struct Coord2D;
struct Coord3D;

// Forward definition to grid_functors.h
template <typename CoordT> struct GridTargetPositionShifts;
template <typename CoordT> struct ShiftedOriginCreator;
template <typename CoordT> struct CachedTileCreator;

void initialize_gsc_registry(
    CreatorRegistry<GridTargetPositionShifts<Coord2D>> &gscr);
void initialize_gsc_registry(
    CreatorRegistry<GridTargetPositionShifts<Coord3D>> &gscr);
void initialize_soc_registry(
    CreatorRegistry<ShiftedOriginCreator<Coord2D>> &socr);
void initialize_soc_registry(
    CreatorRegistry<ShiftedOriginCreator<Coord3D>> &socr);
void initialize_ctc_registry(CreatorRegistry<CachedTileCreator<Coord2D>> &ctcr);
void initialize_ctc_registry(CreatorRegistry<CachedTileCreator<Coord3D>> &ctcr);
} // namespace sapi

#endif
