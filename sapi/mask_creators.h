/*
 *  mask_creators.h
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

#ifndef MASK_CREATORS_H
#define MASK_CREATORS_H

namespace sapi {
// Forward definition to creator_registry.h
template <typename RT> class CreatorRegistry;

// Forward definition to coordinates.h
struct Coord2D;
struct Coord3D;

// Forward definition to mask.h
template <typename CoordT> struct Mask;

void initialize_mk_registry(CreatorRegistry<Mask<Coord2D>> &mkr);
void initialize_mk_registry(CreatorRegistry<Mask<Coord3D>> &mkr);
} // namespace sapi

#endif
