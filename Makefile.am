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

CUSRC=\
iaf_psc_exp_g.cu \
iaf_psc_exp_hc.cu \
iaf_psc_exp.cu \
ext_neuron.cu \
aeif_cond_alpha.cu \
aeif_cond_beta.cu \
aeif_psc_alpha.cu \
aeif_psc_delta.cu \
aeif_psc_exp.cu \
base_neuron.cu \
connect.cu \
connect_mpi.cu \
getRealTime.cu \
get_spike.cu \
multimeter.cu \
nested_loop.cu \
neuron_models.cu \
nestgpu.cu \
node_group.cu \
parrot_neuron.cu \
poiss_gen.cu \
poisson.cu \
prefix_scan.cu \
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
izhikevich_cond_beta.cu \
izhikevich.cu \
izhikevich_psc_exp_5s.cu \
izhikevich_psc_exp_2s.cu \
izhikevich_psc_exp.cu \
user_m1.cu \
user_m2.cu

CPPSRC=\
connect_rules.cpp \
nestgpu_C.cpp

# $(top_srcdir)/src/

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
$(OBJDIR)/%.cu.o: $(top_srcdir)/src/%.cu
	@mkdir -p obj
	$(NVCC) -ccbin=$(MPICXX) --compiler-options  "$(COMPILER_FLAGS)" -I . $(CUDA_FLAGS) $< -c -o $@
$(OBJDIR)/%.cpp.o: $(top_srcdir)/src/%.cpp
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
