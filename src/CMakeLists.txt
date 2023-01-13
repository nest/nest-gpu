# src/CMakeLists.txt
#
# This file is part of NEST GPU.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST GPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST GPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST GPU.  If not, see <http://www.gnu.org/licenses/>.

set ( nestgpu_sources
	utilities.h
	distribution.h
	copass_kernels.h
	copass_sort.h
	connect.h
	iaf_psc_exp_g.h
	iaf_psc_exp_hc_params.h
	iaf_psc_exp_hc.h
	iaf_psc_exp.h
	ext_neuron.h
	aeif_cond_alpha.h
	aeif_cond_alpha_kernel.h
	aeif_cond_alpha_rk5.h
	aeif_cond_beta.h
	aeif_cond_beta_kernel.h
	aeif_cond_beta_rk5.h
	aeif_psc_alpha.h
	aeif_psc_alpha_kernel.h
	aeif_psc_alpha_rk5.h
	aeif_psc_delta.h
	aeif_psc_delta_kernel.h
	aeif_psc_delta_rk5.h
	aeif_psc_exp.h
	aeif_psc_exp_kernel.h
	aeif_psc_exp_rk5.h
	base_neuron.h
	connect_rules.h
	connect_spec.h
	cuda_error.h
	getRealTime.h
	get_spike.h
	multimeter.h
	nested_loop.h
	neuron_models.h
	nestgpu.h
	ngpu_exception.h
	node_group.h
	parrot_neuron.h
	poiss_gen.h
	poiss_gen_variables.h
	prefix_scan.h
	propagate_error.h
	random.h
	rev_spike.h
	rk5.h
	rk5_const.h
	rk5_interface.h
	send_spike.h
	spike_buffer.h
	spike_detector.h
	spike_generator.h
	spike_mpi.h
	stdp.h
	syn_model.h
	test_syn_model.h
	izhikevich_cond_beta.h
	izhikevich_cond_beta_kernel.h
	izhikevich_cond_beta_rk5.h
	izhikevich.h
	izhikevich_psc_exp.h
	izhikevich_psc_exp_5s.h
	izhikevich_psc_exp_2s.h
	user_m1.h
	user_m1_kernel.h
	user_m1_rk5.h
	user_m2.h
	user_m2_kernel.h
	user_m2_rk5.h
	utilities.cu
	distribution.cu
	copass_kernels.cu
	copass_sort.cu
	connect.cu
	iaf_psc_exp_g.cu
	iaf_psc_exp_hc.cu
	iaf_psc_exp.cu
	ext_neuron.cu
	aeif_cond_alpha.cu
	aeif_cond_beta.cu
	aeif_psc_alpha.cu
	aeif_psc_delta.cu
	aeif_psc_exp.cu
	base_neuron.cu
	connect_rules.cu
	getRealTime.cu
	get_spike.cu
	multimeter.cu
	nested_loop.cu
	neuron_models.cu
	nestgpu.cu
	node_group.cu
	parrot_neuron.cu
	poiss_gen.cu
	prefix_scan.cu
	random.cu
	rev_spike.cu
	rk5.cu
	scan.cu
	send_spike.cu
	spike_buffer.cu
	spike_detector.cu
	spike_generator.cu
	spike_mpi.cu
	stdp.cu
	syn_model.cu
	test_syn_model.cu
	izhikevich_cond_beta.cu
	izhikevich.cu
	izhikevich_psc_exp_5s.cu
	izhikevich_psc_exp_2s.cu
	izhikevich_psc_exp.cu
	user_m1.cu
	user_m2.cu
	nestgpu_C.h
	nestgpu_C.cpp
    )

if ( HAVE_MPI )
  set( nestgpu_sources
    ${nestgpu_sources}
    connect_mpi.h connect_mpi.cu
  )
endif ()

add_library( nestgpukernel ${nestgpu_sources} )
set_target_properties( nestgpukernel
    PROPERTIES
    VERSION ${NEST_GPU_VERSION}
    SOVERSION 1
    )

target_link_libraries( nestgpukernel
    ${LTDL_LIBRARIES}
    ${MPI_CXX_LIBRARIES}
    )

target_include_directories( nestgpukernel PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_BINARY_DIR}/libnestutil
    )

set_target_properties(nestgpukernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(nestgpukernel PROPERTIES POSITION_INDEPENDENT_CODE ON)


install( TARGETS nestgpukernel
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/nestgpu
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/nestgpu
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

FILTER_HEADERS("${nestgpu_sources}" install_headers )
install( FILES ${install_headers}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/nestgpu )
