AUTOMAKE_OPTIONS = 1.4 no-dependencies foreign

## ------------------- ##
#
## ------------------- ##
ACLOCAL_AMFLAGS = -I m4

#SUBDIRS=pythonlib
python_PYTHON = $(top_srcdir)/pythonlib/nestgpu.py
#dist_data_DATA=autogen.sh
CCLD = $(CC)
LINK = $(CCLD) $(AM_CFLAGS) $(CFLAGS) $(AM_LDFLAGS) $(LDFLAGS) -o $@

ngpu_h_files=\
$(top_srcdir)/src/iaf_psc_exp_g.h \
$(top_srcdir)/src/iaf_psc_exp_hc_params.h \
$(top_srcdir)/src/iaf_psc_exp_hc.h \
$(top_srcdir)/src/iaf_psc_exp.h \
$(top_srcdir)/src/ext_neuron.h \
$(top_srcdir)/src/aeif_cond_alpha.h \
$(top_srcdir)/src/aeif_cond_alpha_kernel.h \
$(top_srcdir)/src/aeif_cond_alpha_rk5.h \
$(top_srcdir)/src/aeif_cond_beta.h \
$(top_srcdir)/src/aeif_cond_beta_kernel.h \
$(top_srcdir)/src/aeif_cond_beta_rk5.h \
$(top_srcdir)/src/aeif_psc_alpha.h \
$(top_srcdir)/src/aeif_psc_alpha_kernel.h \
$(top_srcdir)/src/aeif_psc_alpha_rk5.h \
$(top_srcdir)/src/aeif_psc_delta.h \
$(top_srcdir)/src/aeif_psc_delta_kernel.h \
$(top_srcdir)/src/aeif_psc_delta_rk5.h \
$(top_srcdir)/src/aeif_psc_exp.h \
$(top_srcdir)/src/aeif_psc_exp_kernel.h \
$(top_srcdir)/src/aeif_psc_exp_rk5.h \
$(top_srcdir)/src/base_neuron.h \
$(top_srcdir)/src/connect.h \
$(top_srcdir)/src/connect_mpi.h \
$(top_srcdir)/src/connect_rules.h \
$(top_srcdir)/src/connect_spec.h \
$(top_srcdir)/src/cuda_error.h \
$(top_srcdir)/src/dir_connect.h \
$(top_srcdir)/src/getRealTime.h \
$(top_srcdir)/src/get_spike.h \
$(top_srcdir)/src/multimeter.h \
$(top_srcdir)/src/nested_loop.h \
$(top_srcdir)/src/neuron_models.h \
$(top_srcdir)/src/nestgpu.h \
$(top_srcdir)/src/nestgpu_C.h \
$(top_srcdir)/src/ngpu_exception.h \
$(top_srcdir)/src/node_group.h \
$(top_srcdir)/src/parrot_neuron.h \
$(top_srcdir)/src/poiss_gen.h \
$(top_srcdir)/src/poiss_gen_variables.h \
$(top_srcdir)/src/poisson.h \
$(top_srcdir)/src/prefix_scan.h \
$(top_srcdir)/src/propagate_error.h \
$(top_srcdir)/src/random.h \
$(top_srcdir)/src/rev_spike.h \
$(top_srcdir)/src/rk5.h \
$(top_srcdir)/src/rk5_const.h \
$(top_srcdir)/src/rk5_interface.h \
$(top_srcdir)/src/send_spike.h \
$(top_srcdir)/src/spike_buffer.h \
$(top_srcdir)/src/spike_detector.h \
$(top_srcdir)/src/spike_generator.h \
$(top_srcdir)/src/spike_mpi.h \
$(top_srcdir)/src/stdp.h \
$(top_srcdir)/src/syn_model.h \
$(top_srcdir)/src/test_syn_model.h \
$(top_srcdir)/src/izhikevich_cond_beta.h \
$(top_srcdir)/src/izhikevich_cond_beta_kernel.h \
$(top_srcdir)/src/izhikevich_cond_beta_rk5.h \
$(top_srcdir)/src/izhikevich.h \
$(top_srcdir)/src/izhikevich_psc_exp.h \
$(top_srcdir)/src/izhikevich_psc_exp_5s.h \
$(top_srcdir)/src/izhikevich_psc_exp_2s.h \
$(top_srcdir)/src/user_m1.h \
$(top_srcdir)/src/user_m1_kernel.h \
$(top_srcdir)/src/user_m1_rk5.h \
$(top_srcdir)/src/user_m2.h \
$(top_srcdir)/src/user_m2_kernel.h \
$(top_srcdir)/src/user_m2_rk5.h

ngpu_src_files=\
$(top_srcdir)/src/iaf_psc_exp_g.cu \
$(top_srcdir)/src/iaf_psc_exp_hc.cu \
$(top_srcdir)/src/iaf_psc_exp.cu \
$(top_srcdir)/src/ext_neuron.cu \
$(top_srcdir)/src/aeif_cond_alpha.cu \
$(top_srcdir)/src/aeif_cond_beta.cu \
$(top_srcdir)/src/aeif_psc_alpha.cu \
$(top_srcdir)/src/aeif_psc_delta.cu \
$(top_srcdir)/src/aeif_psc_exp.cu \
$(top_srcdir)/src/base_neuron.cu \
$(top_srcdir)/src/connect.cu \
$(top_srcdir)/src/connect_mpi.cu \
$(top_srcdir)/src/connect_rules.cpp \
$(top_srcdir)/src/dummyfile.cpp \
$(top_srcdir)/src/getRealTime.cu \
$(top_srcdir)/src/get_spike.cu \
$(top_srcdir)/src/multimeter.cu \
$(top_srcdir)/src/nested_loop.cu \
$(top_srcdir)/src/neuron_models.cu \
$(top_srcdir)/src/nestgpu.cu \
$(top_srcdir)/src/nestgpu_C.cpp \
$(top_srcdir)/src/node_group.cu \
$(top_srcdir)/src/parrot_neuron.cu \
$(top_srcdir)/src/poiss_gen.cu \
$(top_srcdir)/src/poisson.cu \
$(top_srcdir)/src/prefix_scan.cu \
$(top_srcdir)/src/random.cu \
$(top_srcdir)/src/rev_spike.cu \
$(top_srcdir)/src/rk5.cu \
$(top_srcdir)/src/scan.cu \
$(top_srcdir)/src/send_spike.cu \
$(top_srcdir)/src/spike_buffer.cu \
$(top_srcdir)/src/spike_detector.cu \
$(top_srcdir)/src/spike_generator.cu \
$(top_srcdir)/src/spike_mpi.cu \
$(top_srcdir)/src/stdp.cu \
$(top_srcdir)/src/syn_model.cu \
$(top_srcdir)/src/test_syn_model.cu \
$(top_srcdir)/src/izhikevich_cond_beta.cu \
$(top_srcdir)/src/izhikevich.cu \
$(top_srcdir)/src/izhikevich_psc_exp_5s.cu \
$(top_srcdir)/src/izhikevich_psc_exp_2s.cu \
$(top_srcdir)/src/izhikevich_psc_exp.cu \
$(top_srcdir)/src/user_m1.cu \
$(top_srcdir)/src/user_m2.cu

COMPILER_FLAGS= -O3 -Wall -fPIC
if OSX
	COMPILER_FLAGS += -Xpreprocessor $(OPENMP_CFLAGS)
# -fopenmp
else
	COMPILER_FLAGS += $(OPENMP_CFLAGS)
endif

CUDA_FLAGS= -arch $(GPU_ARCH) --ptxas-options=-v --maxrregcount=55 --relocatable-device-code true
CUDA_LIBS= -lm -lcurand
#-lomp

libnestgpu.la: $(ngpu_src_files) $(ngpu_h_files)
	$(NVCC) -ccbin=$(MPICXX) --compiler-options  "$(COMPILER_FLAGS)" -I . $(CUDA_FLAGS) --shared -o $@ $(ngpu_src_files) $(CUDA_LIBS)

lib_LTLIBRARIES = libnestgpu.la

clean-local:
	rm -f *.so

install-exec-hook:
if OSX
	$(top_srcdir)/postinstall.sh
endif
