import sys
import ctypes
import nestgpu as ngpu
from random import randrange


ngpu.ConnectMpiInit();
mpi_np = ngpu.MpiNp()

if (mpi_np < 2) | (len(sys.argv) != 2):
    print ("Usage: mpirun -np NP python %s n_neurons" % sys.argv[0])
    quit()
    
order = int(sys.argv[1])//5

mpi_id = ngpu.MpiId()
print("Building on host ", mpi_id, " ...")

ngpu.SetKernelStatus("rnd_seed", 1234) # seed for GPU random numbers

n_receptors = 2

delay = 1.0       # synaptic delay in ms

NE = 4 * order       # number of excitatory neurons
NI = 1 * order       # number of inhibitory neurons
n_neurons = NE + NI  # number of neurons in total

CE = 800   # number of excitatory synapses per neuron
CI = CE//4  # number of inhibitory synapses per neuron

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
neuron = []
exc_neuron = []
inh_neuron = []
for i in range(mpi_np):
    neuron.append(ngpu.RemoteCreate(i, "aeif_cond_beta_multisynapse", n_neurons, n_receptors).node_seq)
    exc_neuron.append(neuron[i][0:NE])      # excitatory neurons
    inh_neuron.append(neuron[i][NE:n_neurons])   # inhibitory neurons

# receptor parameters
E_rev = [0.0, -85.0]
tau_decay = [1.0, 1.0]
tau_rise = [1.0, 1.0]
ngpu.SetStatus(neuron[mpi_id], {"E_rev":E_rev, "tau_decay":tau_decay,
                        "tau_rise":tau_rise})

# Excitatory local connections, defined on all hosts
# connect excitatory neurons to port 0 of all neurons
# weight Wex and fixed indegree CE*3/4

exc_conn_dict={"rule": "fixed_indegree", "indegree": CE*3//4}
exc_syn_dict={"weight": Wex, "delay": delay,
              "receptor":0}
ngpu.Connect(exc_neuron[mpi_id], neuron[mpi_id], exc_conn_dict, exc_syn_dict)


# Inhibitory local connections, defined on all hosts
# connect inhibitory neurons to port 1 of all neurons
# weight Win and fixed indegree CI*3/4

inh_conn_dict={"rule": "fixed_indegree", "indegree": CI*3//4}
inh_syn_dict={"weight": Win, "delay": delay,
              "receptor":1}
ngpu.Connect(inh_neuron[mpi_id], neuron[mpi_id], inh_conn_dict, inh_syn_dict)


#connect poisson generator to port 0 of all neurons
pg_conn_dict={"rule": "all_to_all"}
pg_syn_dict={"weight": poiss_weight, "delay": poiss_delay,
              "receptor":0}

ngpu.Connect(pg, neuron[mpi_id], pg_conn_dict, pg_syn_dict)


filename = "test_brunel_mpi" + str(mpi_id) + ".dat"
i_neuron_arr = [neuron[mpi_id][0], neuron[mpi_id][randrange(n_neurons)], neuron[mpi_id][n_neurons-1]]
i_receptor_arr = [0, 0, 0]
# any set of neuron indexes
# create multimeter record of V_m
var_name_arr = ["V_m", "V_m", "V_m"]
record = ngpu.CreateRecord(filename, var_name_arr, i_neuron_arr,
                                i_receptor_arr)

######################################################################
## WRITE HERE REMOTE CONNECTIONS
######################################################################

# Excitatory remote connections
# connect excitatory neurons to port 0 of all neurons
# weight Wex and fixed indegree CE//4
# host 0 to host 1
re_conn_dict={"rule": "fixed_indegree", "indegree": (CE//4)//(mpi_np-1)}
re_syn_dict={"weight": Wex, "delay": delay,
              "receptor":0}
# host 0 to host 1
#ngpu.RemoteConnect(0, exc_neuron[0], 1, neuron[1], re_conn_dict, re_syn_dict)
# host 1 to host 0
#ngpu.RemoteConnect(1, exc_neuron[1], 0, neuron[0], re_conn_dict, re_syn_dict)

# Inhibitory remote connections
# connect inhibitory neurons to port 1 of all neurons
# weight Win and fixed indegree CI//4
# host 0 to host 1
ri_conn_dict={"rule": "fixed_indegree", "indegree": (CI//4)//(mpi_np-1)}
ri_syn_dict={"weight": Win, "delay": delay,
              "receptor":1}
# host 0 to host 1
#ngpu.RemoteConnect(0, inh_neuron[0], 1, neuron[1], ri_conn_dict, ri_syn_dict)
# host 1 to host 0
#ngpu.RemoteConnect(1, inh_neuron[1], 0, neuron[0], ri_conn_dict, ri_syn_dict)

for i in range(mpi_np):
    for j in range(mpi_np):
        if(i!=j):
            ngpu.RemoteConnect(i, exc_neuron[i], j, neuron[j], re_conn_dict, re_syn_dict)
            ngpu.RemoteConnect(i, inh_neuron[i], j, neuron[j], ri_conn_dict, ri_syn_dict)


ngpu.Simulate()

nrows=ngpu.GetRecordDataRows(record)
ncol=ngpu.GetRecordDataColumns(record)
#print nrows, ncol

data_list = ngpu.GetRecordData(record)
t=[row[0] for row in data_list]
V1=[row[1] for row in data_list]
V2=[row[2] for row in data_list]
V3=[row[3] for row in data_list]

import matplotlib.pyplot as plt

fig1 = plt.figure(1)
fig1.suptitle("host " + str(mpi_id), fontsize=20)
plt.plot(t, V1)

fig2 = plt.figure(2)
fig2.suptitle("host " + str(mpi_id), fontsize=20)
plt.plot(t, V2)

fig3 = plt.figure(3)
fig3.suptitle("host " + str(mpi_id), fontsize=20)
plt.plot(t, V3)

plt.show()
