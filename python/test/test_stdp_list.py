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
                      

sg = ngpu.Create("spike_generator")
neuron0 = ngpu.Create("aeif_cond_beta_multisynapse")
neuron1 = ngpu.Create("aeif_cond_beta_multisynapse", N)
ngpu.SetStatus(neuron1, {"t_ref": 1000.0, "den_delay":den_delay}) 

time_diff = 200.0
dt_list = []
delay_stdp_list = []
for i in range(N):
    dt_list.append(dt_step*(-0.5*(N-1) + i))
    delay_stdp_list.append(time_diff - dt_list[i])

spike_times = [50.0]
spike_mul = [1.0]
n_spikes = 1


# set spike times and height
ngpu.SetStatus(sg, {"spike_times": spike_times, "spike_gen_mul":spike_mul})
delay0 = 1.0
delay1 = delay0 + time_diff
weight_sg = 17.9 # to make it spike immediately and only once
weight_stdp = Wmax/2

conn_dict={"rule": "one_to_one"}
conn_dict_full={"rule": "all_to_all"}
syn_dict0={"weight":weight_sg, "delay":delay0}
syn_dict1={"weight":weight_sg, "delay":delay1}

ngpu.Connect(sg, neuron0, conn_dict, syn_dict0)
ngpu.Connect(sg, neuron1, conn_dict_full, syn_dict1)


syn_dict_stdp={"weight":weight_stdp, "delay_array":delay_stdp_list, \
               "synapse_group":syn_group}

ngpu.Connect(neuron0, neuron1, conn_dict_full, syn_dict_stdp)

ngpu.Simulate(1000.0)

#conn_id = ngpu.GetConnections(neuron0, neuron1)
dt = dt_list
#w = ngpu.GetStatus(conn_id, "weight")


expect_w = []
dw = []
sim_w = []
for i in range(N):
    conn_id = ngpu.GetConnections(neuron0, neuron1[i])
    w = ngpu.GetStatus(conn_id, "weight")
    w1 = STDPUpdate(weight_stdp, dt[i], tau_plus, tau_minus, Wmax*lambd, alpha, \
                    mu_plus, mu_minus, Wmax)
    expect_w.append(w1)
    sim_w.append(w[0])
    dw.append(w1-w[0])
    if abs(dw[i])>tolerance:
        print("Expected weight: ", w1, " simulated: ", w)
        sys.exit(1)

sys.exit(0)

#import matplotlib.pyplot as plt

#plt.figure(1)
#plt.plot(dt, sim_w)

#plt.figure(2)
#plt.plot(dt, expect_w)

#plt.draw()
#plt.pause(1)
#raw_input("<Hit Enter To Close>")
#plt.close()
