# nest/CMakeLists.txt
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

set( nestgpu_lib_sources
    nestgpu_C.cpp nestgpu_C.h
    propagate_error.h
)

add_library( nestgpu SHARED ${nestgpu_lib_sources} )
set_target_properties( nestgpu PROPERTIES POSITION_INDEPENDENT_CODE ON )

target_link_libraries( nestgpu PRIVATE
    nestgpu_kernel
    sapi_kernel
)

install( TARGETS nestgpu
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/${PYEXECDIR}/nestgpu
)

FILTER_HEADERS( "${nestgpu_lib_sources}" install_headers )
install( FILES ${install_headers}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/nestgpu )
