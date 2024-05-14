import sys
import math
import nestgpu as ngpu

def STDPUpdate(w, Dt, tau_plus, tau_minus, Wplus, alpha, mu_plus, mu_minus, \
               Wmax):
    if (Dt>=0):
        fact = Wplus*math.exp(-Dt/tau_plus)
        w1 = w + fact*math.pow(1.0 - w/Wmax, mu_plus)
        if w1>Wmax:
            w1 = Wmax
        
    else:
        fact = -alpha*Wplus*math.exp(Dt/tau_minus)
        w1 = w + fact*math.pow(w/Wmax, mu_minus)
        if w1<0.0:
            w1 = 0.0
    return w1

ngpu.SetKernelStatus({'spike_buffer_algo': 0})

tolerance = 1.0e-5
dt_step = 5.0
N = 50

tau_plus = 20.0
tau_minus = 20.0
lambd = 1.0
alpha = 1.0
mu_plus = 1.0
mu_minus = 1.0
Wmax = 0.001
den_delay = 0.0

syn_group = ngpu.CreateSynGroup \
            ("stdp", {"tau_plus":tau_plus, "tau_minus":tau_minus, \
                      "lambda":lambd, "alpha":alpha, "mu_plus":mu_plus, \
                      "mu_minus":mu_minus,  "Wmax":Wmax})

sg = ngpu.Create("spike_generator", N)
neuron = ngpu.Create("aeif_cond_beta_multisynapse", 2*N)
ngpu.SetStatus(neuron, {"t_ref": 1000.0, "den_delay":den_delay}) 
neuron0 = neuron[0:N]
neuron1 = neuron[N:2*N]
dt_list = []
for i in range(N):
    dt_list.append(dt_step*(-0.5*(N-1) + i))

spike_time = [50.0]
spike_mul = [1.0]
n_spikes = 1
time_diff = 200.0

# set spike times and height
ngpu.SetStatus(sg, {"spike_times": spike_time, "spike_gen_mul":spike_mul})
delay0 = 1.0
delay1 = delay0 + time_diff
weight_sg = 17.9 # to make it spike immediately and only once
weight_stdp = Wmax/2

conn_dict={"rule": "one_to_one"}
syn_dict0={"weight":weight_sg, "delay":delay0}
syn_dict1={"weight":weight_sg, "delay":delay1}

ngpu.Connect(sg, neuron0, conn_dict, syn_dict0)
ngpu.Connect(sg, neuron1, conn_dict, syn_dict1)

for i in range(N):
    delay_stdp = time_diff - dt_list[i]
    syn_dict_stdp={"weight":weight_stdp, "delay":delay_stdp, \
                   "synapse_group":syn_group}
    ngpu.Connect([neuron0[i]], [neuron1[i]], conn_dict, syn_dict_stdp)

ngpu.Simulate(1000.0)

conn_id = ngpu.GetConnections(neuron0, neuron1)
dt = dt_list
w = ngpu.GetStatus(conn_id, "weight")


expect_w = []
dw = []
for i in range(N):
    w1 = STDPUpdate(weight_stdp, dt[i], tau_plus, tau_minus, lambd*Wmax, alpha, \
                    mu_plus, mu_minus, Wmax)
    expect_w.append(w1)
    dw.append(w1-w[i])
    if abs(dw[i])>tolerance:
        print("Expected weight: ", w1, " simulated: ", w[i])
        sys.exit(1)

sys.exit(0)
