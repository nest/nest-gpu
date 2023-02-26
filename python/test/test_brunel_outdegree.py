import sys
import math
import ctypes
import nestgpu as ngpu
from random import randrange
import numpy as np

order = 200
n_test = 100

expected_rate = 30.78
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
pg_list = pg.ToList()

# Create n_neurons neurons with n_receptor receptor ports
neuron = ngpu.Create("aeif_cond_beta_multisynapse", n_neurons, n_receptors)
exc_neuron = neuron[0:NE]      # excitatory neurons
inh_neuron = neuron[NE:n_neurons]   # inhibitory neurons
neuron_list = neuron.ToList()
exc_neuron_list = exc_neuron.ToList()
inh_neuron_list = inh_neuron.ToList()

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
ngpu.Connect(exc_neuron, neuron_list, exc_conn_dict, exc_syn_dict)

# Inhibitory connections
# connect inhibitory neurons to port 1 of all neurons
# normally distributed delays, weight Win and CPN connections per neuron
inh_conn_dict={"rule": "fixed_outdegree", "outdegree": CPN}
inh_syn_dict={"weight": Win, "delay":{"distribution":"normal_clipped",
                                       "mu":mean_delay, "low":min_delay,
                                       "high":mean_delay+3*std_delay,
                                       "sigma":std_delay}, "receptor":1}
ngpu.Connect(inh_neuron_list, neuron, inh_conn_dict, inh_syn_dict)

#connect poisson generator to port 0 of all neurons
pg_conn_dict={"rule": "all_to_all"}
pg_syn_dict={"weight": poiss_weight, "delay": poiss_delay, "receptor":0}

ngpu.Connect(pg_list, neuron_list, pg_conn_dict, pg_syn_dict)

i_neuron_list = [neuron[0], neuron[n_neurons-1]]
i_receptor_list = [0, 0]
var_name_list = ["spike", "spike"]
                 
for i in range(n_test-2):
    i_neuron_list.append(neuron[randrange(n_neurons)])
    i_receptor_list.append(0)
    var_name_list.append("spike")

# create multimeter record of spikes
record = ngpu.CreateRecord("", var_name_list, i_neuron_list, i_receptor_list)

ngpu.Simulate()

data_list = ngpu.GetRecordData(record)

for i in range(1000):
    conn_id = ngpu.GetConnections(i+1)
    n_out_conn = len(conn_id)
    if (n_out_conn!=NE+NI):
        print("Expected number of out connections per neuron: ", NE+NI)
        print("Number of out connections of neuron ", i + 1, ": ", \
              n_out_conn)
        #sys.exit(1)
        

for i in range(10):
    i_target = randrange(n_neurons)
    conn_id = ngpu.GetConnections(target=i_target+1)
    n_in_conn = len(conn_id)
    if (n_in_conn!=NE+NI+1):
        print("Expected number of in connections per neuron: ", NE+NI+1)
        print("Number of in connections of neuron ", i_target, ": ", \
              n_in_conn)
        #sys.exit(1)


row_sum = list(data_list[0])
for row in data_list[1:len(data_list)]:
    for i in range(len(row_sum)):
        row_sum[i] = row_sum[i] + row[i]

spike = row_sum[1:len(row_sum)]
spike_arr = np.array(spike)

min_spike_num = np.min(spike_arr)
max_spike_num = np.max(spike_arr)
if (min_spike_num < expected_rate - 3.0*math.sqrt(expected_rate)):
    print ("Expected rate: ", expected_rate)
    print("Min rate :", min_spike_num)
    sys.exit(1)
    
if (max_spike_num > expected_rate + 3.0*math.sqrt(expected_rate)):
    print ("Expected rate: ", expected_rate)
    print("Max rate :", max_spike_num)
    sys.exit(1)

mean_spike_num = np.mean(spike_arr)
diff = abs(mean_spike_num - expected_rate)
max_diff = 3.0*np.sqrt(expected_rate)/np.sqrt(n_test)
print ("Expected rate: ", expected_rate)
print("Mean rate: ", mean_spike_num)
if diff > max_diff:
    sys.exit(1)
else:
    sys.exit(0)

