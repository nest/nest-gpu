import sys
import ctypes
import nestgpu as ngpu
from random import randrange

if len(sys.argv) != 2:
    print ("Usage: python %s n_neurons" % sys.argv[0])
    quit()
    
order = int(sys.argv[1])//5

print("Building ...")

ngpu.SetKernelStatus("rnd_seed", 1234) # seed for GPU random numbers

n_receptors = 2

NE = 4 * order       # number of excitatory neurons
NI = 1 * order       # number of inhibitory neurons
n_neurons = NE + NI  # number of neurons in total

CPN = 1000 # number of output connections per neuron

Wex = 0.05
Win = 0.35

# poisson generator parameters
poiss_rate = 20000.0 # poisson signal rate in Hz
poiss_weight = 0.37
poiss_delay = 0.2 # poisson signal delay in ms

# create poisson generator
pg = ngpu.Create("poisson_generator")
ngpu.SetStatus(pg, "rate", poiss_rate)

# Create n_neurons neurons with n_receptor receptor ports
neuron = ngpu.Create("aeif_cond_beta_multisynapse", n_neurons, n_receptors)
exc_neuron = neuron[0:NE]      # excitatory neurons
inh_neuron = neuron[NE:n_neurons]   # inhibitory neurons
  
# receptor parameters
E_rev = [0.0, -85.0]
tau_decay = [1.0, 1.0]
tau_rise = [1.0, 1.0]
ngpu.SetStatus(neuron, {"E_rev":E_rev, "tau_decay":tau_decay,
                        "tau_rise":tau_rise})

mean_delay = 0.5
std_delay = 0.25
min_delay = 0.1
# Excitatory connections
# connect excitatory neurons to port 0 of all neurons
# normally distributed delays, weight Wex and CPN connections per neuron
exc_conn_dict={"rule": "fixed_outdegree", "outdegree": CPN}
exc_syn_dict={"weight": Wex, "delay": {"distribution":"normal_clipped",
                                       "mu":mean_delay, "low":min_delay,
                                       "high":mean_delay+3*std_delay,
                                       "sigma":std_delay}, "receptor":0}
ngpu.Connect(exc_neuron, neuron, exc_conn_dict, exc_syn_dict)


# Inhibitory connections
# connect inhibitory neurons to port 1 of all neurons
# normally distributed delays, weight Win and CPN connections per neuron
inh_conn_dict={"rule": "fixed_outdegree", "outdegree": CPN}
inh_syn_dict={"weight": Win, "delay":{"distribution":"normal_clipped",
                                       "mu":mean_delay, "low":min_delay,
                                       "high":mean_delay+3*std_delay,
                                       "sigma":std_delay}, "receptor":1}
ngpu.Connect(inh_neuron, neuron, inh_conn_dict, inh_syn_dict)


#connect poisson generator to port 0 of all neurons
pg_conn_dict={"rule": "all_to_all"}
pg_syn_dict={"weight": poiss_weight, "delay": poiss_delay,
              "receptor":0}

ngpu.Connect(pg, neuron, pg_conn_dict, pg_syn_dict)


filename = "test_brunel_outdegree.dat"
# any set of neuron indexes
i_neuron_arr = [neuron[0], neuron[randrange(n_neurons)],
                neuron[randrange(n_neurons)], neuron[randrange(n_neurons)],
                neuron[randrange(n_neurons)], neuron[randrange(n_neurons)],
                neuron[randrange(n_neurons)], neuron[randrange(n_neurons)],
                neuron[randrange(n_neurons)], neuron[n_neurons-1]]
i_receptor_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# create multimeter record of V_m
var_name_arr = ["V_m", "V_m", "V_m", "V_m", "V_m", "V_m", "V_m", "V_m", "V_m",
                "V_m"]
record = ngpu.CreateRecord(filename, var_name_arr, i_neuron_arr,
                                i_receptor_arr)

ngpu.Simulate()

nrows=ngpu.GetRecordDataRows(record)
ncol=ngpu.GetRecordDataColumns(record)
#print nrows, ncol

data_list = ngpu.GetRecordData(record)
t=[row[0] for row in data_list]
V1=[row[1] for row in data_list]
V2=[row[2] for row in data_list]
V3=[row[3] for row in data_list]
V4=[row[4] for row in data_list]
V5=[row[5] for row in data_list]
V6=[row[6] for row in data_list]
V7=[row[7] for row in data_list]
V8=[row[8] for row in data_list]
V9=[row[9] for row in data_list]
V10=[row[10] for row in data_list]

import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(t, V1)

plt.figure(2)
plt.plot(t, V2)

plt.figure(3)
plt.plot(t, V3)

plt.figure(4)
plt.plot(t, V4)

plt.figure(5)
plt.plot(t, V5)

plt.figure(6)
plt.plot(t, V6)

plt.figure(7)
plt.plot(t, V7)

plt.figure(8)
plt.plot(t, V8)

plt.figure(9)
plt.plot(t, V9)

plt.figure(10)
plt.plot(t, V10)

plt.draw()
plt.pause(0.5)
ngpu.waitenter("<Hit Enter To Close>")
plt.close()
