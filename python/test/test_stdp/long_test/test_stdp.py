import nestgpu as ngpu
import math
import matplotlib.pyplot as plt

# STDP weight update theoretical formula for comparison
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
Dt_offset = 2.0 # time difference between presynaptic and postsynaptic spike
N=400 # number of presynaptic and postsynaptic neurons
Dt_max = 5.0 # maximum axonal/dendritic delay

sg_delay_m = 20.0
Dt_spike = 100.0
n_spikes = 10;

sim_time = Dt_spike*(n_spikes + 2)

# STDP connection parameters
tau_plus = 20.0
tau_minus = 20.0
lambd = 0.01
alpha = 1.0
mu_plus = 1.0
mu_minus = 1.0
Wmax = 10.0
weight_stdp = 1.0


# create presynaptic and postsynaptic parrot neurons
neuron_pre = ngpu.Create("parrot_neuron", N)
neuron_post = ngpu.Create("parrot_neuron", N)

#spike generator
sg = ngpu.Create("spike_generator")

# spike generator produces n_spikes spikes with time interval Dt_spike
spike_times = []
for i in range(n_spikes):
    spike_times.append(Dt_spike*(i+1))
    
ngpu.SetStatus(sg, {"spike_times": spike_times})


#connect spike generator to parrot neurons
sg_conn_dict={"rule": "all_to_all"}
syn_dict_sg_pre={"weight":1.0, "delay":sg_delay_m-Dt_offset/2.0}
ngpu.Connect(sg, neuron_pre, sg_conn_dict, syn_dict_sg_pre)
syn_dict_sg_post={"weight":1.0, "delay":sg_delay_m+Dt_offset/2.0}
ngpu.Connect(sg, neuron_post, sg_conn_dict, syn_dict_sg_post)

syn_group = ngpu.CreateSynGroup \
            ("stdp", {"tau_plus":tau_plus, "tau_minus":tau_minus, \
                      "lambda":lambd, "alpha":alpha, "mu_plus":mu_plus, \
                      "mu_minus":mu_minus,  "Wmax":Wmax})
conn_dict={"rule": "all_to_all"}
for j in range(N):
    delay_post = 0.1 + round(Dt_max*j/N,1)
    ngpu.SetStatus([neuron_post[j]], {"den_delay": delay_post})

delay_pre = []    
for i in range(N):
    for j in range(N):
        delay_pre.append(0.1 + round(Dt_max*i/N,1))
        
syn_dict_stdp={"weight":weight_stdp, "delay_array":delay_pre, \
               "synapse_group":syn_group, "receptor":1}
        
ngpu.Connect(neuron_pre, neuron_post, conn_dict, syn_dict_stdp)

ngpu.Simulate(sim_time)

Wplus = Wmax*lambd
max_dw_rel = 0
mse = 0
count = 0
for j in range(N):
    for i in range(N):
        conn_id = ngpu.GetConnections(neuron_pre[i], neuron_post[j])
        w = ngpu.GetStatus(conn_id, "weight")
        # print("Initial weight: ", weight_stdp)
        # print("Simulated weight: ", w[0])
        delay_pre = 0.1 + round(Dt_max*i/N,1)
        delay_post = 0.1 + round(Dt_max*j/N,1)
        Dt = Dt_offset + delay_post - delay_pre
        if Dt>=0:
            Dt1 = -(Dt_spike - Dt) 
        else:
            Dt1 = Dt_spike + Dt 
        if (Dt > 1.0e-6) | (Dt<-1.0e-6):
            w1 = weight_stdp
            for ispike in range(n_spikes):
                w1 = STDPUpdate(w1, Dt, tau_plus, tau_minus, Wplus, \
                                alpha, mu_plus, mu_minus, Wmax)
                if ispike<n_spikes-1:
                    w1 = STDPUpdate(w1, Dt1, tau_plus, tau_minus, Wplus, \
                                    alpha, mu_plus, mu_minus, Wmax)
                    
            # print("Expected theoretical weight: ", w1)
            dw_rel = (w1 - w[0])/w1
            mse = mse + (w1 - w[0])**2
            count = count + 1
            # print("dw/w: ", dw_rel) 
            if abs(dw_rel)>max_dw_rel:
                max_dw_rel = abs(dw_rel)
mse = mse/count
print("max abs(dw/w): ", max_dw_rel) 
print("rmse: ", math.sqrt(mse)) 
