import nestgpu as ngpu
import math
import matplotlib.pyplot as plt

sim_time = 20.0

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
# postsynaptic and presynaptic neurons
neuron_post = ngpu.Create("parrot_neuron")
ngpu.ActivateRecSpikeTimes(neuron_post, 20)
neuron_pre = ngpu.Create("parrot_neuron")
ngpu.ActivateRecSpikeTimes(neuron_pre, 20)

#spike generators
sg_post = ngpu.Create("spike_generator")
sg_pre = ngpu.Create("spike_generator")


# spike times
spike_times_post = [2.0, 6.5]
spike_times_pre = [1.0, 5.0, 6.0]
ngpu.SetStatus(sg_post, {"spike_times": spike_times_post})
ngpu.SetStatus(sg_pre, {"spike_times": spike_times_pre})

# connect spike generators to neurons
syn_dict={"weight":1.0, "delay":1.0}
conn_dict={"rule": "one_to_one"}
ngpu.Connect(sg_post, neuron_post, conn_dict, syn_dict)
ngpu.Connect(sg_pre, neuron_pre, conn_dict, syn_dict)

# STDP connection parameters
tau_plus = 20.0
tau_minus = 20.0
lambd = 0.01
alpha = 1.0
mu_plus = 1.0
mu_minus = 1.0
Wmax = 10.0
den_delay = 1.0 
weight_stdp = 1.0

delay = 3.0

syn_group = ngpu.CreateSynGroup \
            ("stdp", {"tau_plus":tau_plus, "tau_minus":tau_minus, \
                      "lambda":lambd, "alpha":alpha, "mu_plus":mu_plus, \
                      "mu_minus":mu_minus,  "Wmax":Wmax})

syn_dict_stdp={"weight":weight_stdp, "delay":delay, \
               "synapse_group":syn_group, "receptor":1}

ngpu.SetStatus(neuron_post, {"den_delay": den_delay})

ngpu.Connect(neuron_pre, neuron_post, conn_dict, syn_dict_stdp)

ngpu.Simulate(sim_time)

conn_id = ngpu.GetConnections(neuron_pre, neuron_post)
w = ngpu.GetStatus(conn_id, "weight")

print("Initial weight: ", weight_stdp)
print("Simulated weight: ", w[0])

Wplus = Wmax*lambd
Dt1 = -1.0
w1 = STDPUpdate(weight_stdp, Dt1, tau_plus, tau_minus, Wplus, \
                alpha, mu_plus, mu_minus, Wmax)

Dt2 = 3.5
w2 = STDPUpdate(w1, Dt2, tau_plus, tau_minus, Wplus, \
                alpha, mu_plus, mu_minus, Wmax)

Dt3 = -0.5
w3 = STDPUpdate(w2, Dt3, tau_plus, tau_minus, Wplus, \
                alpha, mu_plus, mu_minus, Wmax)

Dt4 = -1.5
w4 = STDPUpdate(w3, Dt4, tau_plus, tau_minus, Wplus, \
                alpha, mu_plus, mu_minus, Wmax)

print("Expected theoretical weight: ", w4)

print("dw/w: ", (w4 - w[0])/w4)

#spike_times0=ngpu.GetRecSpikeTimes(neuron_post[0])
#spike_times1=ngpu.GetRecSpikeTimes(neuron_pre[0])
#print(spike_times0)
#print(spike_times1)
