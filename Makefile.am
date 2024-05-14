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

HCUSRC=\
connect.h \
conn12b.h \
conn16b.h \
remote_connect.h \
connect_rules.h \
connect_spec.h \
nestgpu.h \
get_spike.h \
send_spike.h \
rev_spike.h \
spike_buffer.h \
remote_spike.h \
mpi_comm.h \
base_neuron.h \
node_group.h \
neuron_models.h \
poiss_gen.h \
poiss_gen_variables.h \
cuda_error.h \
ngpu_exception.h \
getRealTime.h \
random.h \
utilities.h \
distribution.h \
scan.h \
prefix_scan.h \
nested_loop.h \
copass_kernels.h \
copass_sort.h \
rk5_const.h \
rk5.h \
rk5_interface.h \
propagate_error.h \
propagator_stability.h \
stdp.h \
syn_model.h \
test_syn_model.h \
spike_detector.h \
spike_generator.h \
parrot_neuron.h \
multimeter.h \
ext_neuron.h \
user_m1.h \
user_m1_kernel.h \
user_m1_rk5.h \
user_m2.h \
user_m2_kernel.h \
user_m2_rk5.h \
aeif_cond_alpha.h \
aeif_cond_alpha_kernel.h \
aeif_cond_alpha_multisynapse.h \
aeif_cond_alpha_multisynapse_kernel.h \
aeif_cond_alpha_multisynapse_rk5.h \
aeif_cond_beta.h \
aeif_cond_beta_kernel.h \
aeif_cond_beta_multisynapse.h \
aeif_cond_beta_multisynapse_kernel.h \
aeif_cond_beta_multisynapse_rk5.h \
aeif_psc_alpha.h \
aeif_psc_alpha_kernel.h \
aeif_psc_alpha_multisynapse.h \
aeif_psc_alpha_multisynapse_kernel.h \
aeif_psc_alpha_multisynapse_rk5.h \
aeif_psc_delta.h \
aeif_psc_delta_kernel.h \
aeif_psc_delta_rk5.h \
aeif_psc_exp.h \
aeif_psc_exp_kernel.h \
aeif_psc_exp_multisynapse.h \
aeif_psc_exp_multisynapse_kernel.h \
aeif_psc_exp_multisynapse_rk5.h \
iaf_psc_exp_g.h \
iaf_psc_exp.h \
iaf_psc_exp_hc.h \
iaf_psc_exp_hc_params.h \
iaf_psc_alpha.h \
izhikevich_cond_beta.h \
izhikevich_cond_beta_kernel.h \
izhikevich_cond_beta_rk5.h \
izhikevich.h \
izhikevich_psc_exp_2s.h \
izhikevich_psc_exp_5s.h \
izhikevich_psc_exp.h

CUSRC=\
connect.cu \
conn12b.cu \
conn16b.cu \
remote_connect.cu \
connect_rules.cu \
nestgpu.cu \
get_spike.cu \
send_spike.cu \
rev_spike.cu \
spike_buffer.cu \
remote_spike.cu \
mpi_comm.cu \
base_neuron.cu \
node_group.cu \
neuron_models.cu \
poiss_gen.cu \
getRealTime.cu \
random.cu \
utilities.cu \
distribution.cu \
scan.cu \
prefix_scan.cu \
nested_loop.cu \
copass_kernels.cu \
copass_sort.cu \
rk5.cu \
propagator_stability.cu \
stdp.cu \
syn_model.cu \
test_syn_model.cu \
spike_detector.cu \
spike_generator.cu \
parrot_neuron.cu \
multimeter.cu \
ext_neuron.cu \
user_m1.cu \
user_m2.cu \
aeif_cond_alpha.cu \
aeif_cond_alpha_multisynapse.cu \
aeif_cond_beta.cu \
aeif_cond_beta_multisynapse.cu \
aeif_psc_alpha.cu \
aeif_psc_alpha_multisynapse.cu \
aeif_psc_delta.cu \
aeif_psc_exp.cu \
aeif_psc_exp_multisynapse.cu \
iaf_psc_exp.cu \
iaf_psc_exp_g.cu \
iaf_psc_exp_hc.cu \
iaf_psc_alpha.cu \
izhikevich_cond_beta.cu \
izhikevich.cu \
izhikevich_psc_exp_2s.cu \
izhikevich_psc_exp_5s.cu \
izhikevich_psc_exp.cu


HCPPSRC=\
nestgpu_C.h

CPPSRC=\
nestgpu_C.cpp

COMPILER_FLAGS= -O3 -Wall -fPIC
if OSX
	COMPILER_FLAGS += -Xpreprocessor $(OPENMP_CFLAGS)
# -fopenmp
else
	COMPILER_FLAGS += $(OPENMP_CFLAGS)
endif

CUDA_FLAGS= -arch $(GPU_ARCH) --ptxas-options=-v --maxrregcount=55 --relocatable-device-code true
CUDA_LIBS= -lm -lcurand -lcuda -lcudart
#-lomp

CUOBJ=$(CUSRC:.cu=.cu.o)
CPPOBJ=$(CPPSRC:.cpp=.cpp.o)
OBJDIR=obj
OBJS=$(addprefix $(OBJDIR)/, $(CUOBJ)) $(addprefix $(OBJDIR)/, $(CPPOBJ))
$(OBJDIR)/%.cu.o: $(top_srcdir)/src/%.cu $(top_srcdir)/src/%.h
	@mkdir -p obj
	$(NVCC) -ccbin=$(MPICXX) --compiler-options  "$(COMPILER_FLAGS)" -I . $(CUDA_FLAGS) $< -c -o $@
$(OBJDIR)/%.cpp.o: $(top_srcdir)/src/%.cpp $(top_srcdir)/src/%.h
	@mkdir -p obj
	$(MPICXX) $(COMPILER_FLAGS) -I . $< -c -o $@

libnestgpu.la: $(OBJS)
	$(NVCC) -ccbin=$(MPICXX) --compiler-options  "$(COMPILER_FLAGS)" -I . $(CUDA_FLAGS) --shared -o $@ $^ $(CUDA_LIBS)

lib_LTLIBRARIES = libnestgpu.la

%.cu: 

clean-local:
	rm -f *.so obj/*.o

install-exec-hook:
if OSX
	$(top_srcdir)/postinstall.sh
endif
