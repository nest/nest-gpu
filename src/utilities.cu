/*
 *  utilities.cu
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

int64_t
IntPow( int64_t x, unsigned int p )
{
  if ( p == 0 )
  {
    return 1;
  }
  if ( p == 1 )
  {
    return x;
  }

  int64_t tmp = IntPow( x, p / 2 );
  if ( p % 2 == 0 )
  {
    return tmp * tmp;
  }
  else
  {
    return x * tmp * tmp;
  }
}
