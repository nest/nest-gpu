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
base_neuron.h \
connect.h \
connect_mpi.h \
connect_spec.h \
cuda_error.h \
dir_connect.h \
ext_neuron.h \
getRealTime.h \
get_spike.h \
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
izhikevich_psc_exp.h \
locate.h \
multimeter.h \
nestgpu.h \
neuron_models.h \
ngpu_exception.h \
node_group.h \
parrot_neuron.h \
poiss_gen.h \
poiss_gen_variables.h \
poisson.h \
prefix_scan.h \
propagate_error.h \
propagator_stability.h \
random.h \
rev_spike.h \
rk5_const.h \
rk5.h \
rk5_interface.h \
scan.h \
send_spike.h \
spike_buffer.h \
spike_detector.h \
spike_generator.h \
spike_mpi.h \
stdp.h \
syn_model.h \
test_syn_model.h \
user_m1.h \
user_m1_kernel.h \
user_m1_rk5.h \
user_m2.h \
user_m2_kernel.h \
user_m2_rk5.h

CUSRC=\
aeif_cond_alpha.cu \
aeif_cond_alpha_multisynapse.cu \
aeif_cond_beta.cu \
aeif_cond_beta_multisynapse.cu \
aeif_psc_alpha.cu \
aeif_psc_alpha_multisynapse.cu \
aeif_psc_delta.cu \
aeif_psc_exp.cu \
aeif_psc_exp_multisynapse.cu \
base_neuron.cu \
connect.cu \
connect_mpi.cu \
ext_neuron.cu \
getRealTime.cu \
get_spike.cu \
iaf_psc_exp.cu \
iaf_psc_exp_g.cu \
iaf_psc_exp_hc.cu \
iaf_psc_alpha.cu \
izhikevich_cond_beta.cu \
izhikevich.cu \
izhikevich_psc_exp_2s.cu \
izhikevich_psc_exp_5s.cu \
izhikevich_psc_exp.cu \
locate.cu \
multimeter.cu \
nestgpu.cu \
neuron_models.cu \
node_group.cu \
parrot_neuron.cu \
poiss_gen.cu \
poisson.cu \
prefix_scan.cu \
propagator_stability.cu \
random.cu \
rev_spike.cu \
rk5.cu \
scan.cu \
send_spike.cu \
spike_buffer.cu \
spike_detector.cu \
spike_generator.cu \
spike_mpi.cu \
stdp.cu \
syn_model.cu \
test_syn_model.cu \
user_m1.cu \
user_m2.cu

HCPPSRC=\
connect_rules.h \
nestgpu_C.h

CPPSRC=\
connect_rules.cpp \
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
