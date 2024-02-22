import sys
import math
import ctypes
import nestgpu as ngpu
from random import randrange
import numpy as np

ngpu.SetKernelStatus("rnd_seed", 1234) # seed for GPU random numbers

n_neurons = 30
eps = 1.0e-6

# poisson generator parameters
poiss_rate = 500.0 # poisson signal rate in Hz
poiss_weight = 4.0
poiss_delay = 0.2 # poisson signal delay in ms

# create poisson generator
pg = ngpu.Create("poisson_generator")
ngpu.SetStatus(pg, "rate", poiss_rate)

# Create n_neurons neurons
neuron = ngpu.Create("aeif_cond_beta_multisynapse", n_neurons)
ngpu.ActivateSpikeCount(neuron)
ngpu.ActivateRecSpikeTimes(neuron, 500)
ngpu.SetRecSpikeTimesStep(neuron, 5)

# Create n_neurons spike detectors
sd = ngpu.Create("spike_detector", n_neurons)

#connect poisson generator to all neurons
pg_conn_dict={"rule": "all_to_all"}
pg_syn_dict={"weight": poiss_weight, "delay": poiss_delay}

ngpu.Connect(pg, neuron, pg_conn_dict, pg_syn_dict)

#connect neurons to spike detectors
sd_conn_dict={"rule": "one_to_one"}
sd_syn_dict={"weight": 1.0, "delay": 0.1}

ngpu.Connect(neuron, sd, sd_conn_dict, sd_syn_dict)

# create multimeter record of spikes
i_node_list = sd.ToList()
i_receptor_list = [0]*n_neurons
var_name_list = ["spike_mul"]*n_neurons

record = ngpu.CreateRecord("", var_name_list, i_node_list, \
                           i_receptor_list)

ngpu.Simulate(490)
ngpu.SetStatus(pg, "rate", 0.0)
ngpu.Simulate(10)


data_list = ngpu.GetRecordData(record)
row_sum = list(data_list[0])
for row in data_list[1:len(data_list)]:
    for i in range(len(row_sum)):
        row_sum[i] = row_sum[i] + row[i]

        
spike_times = []
for i in range(len(neuron)):
    spike_times.append([])
    
for row in data_list[0:len(data_list)]:
    for i in range(1,len(row)):
        y = row[i]
        if y>0.5:
            #print(i, row[0])
            #print (spike_times)
            spike_times[i-1].append(round(row[0]-0.2,4))
            

spike = row_sum[1:len(row_sum)]
#print (spike)

spike_count = ngpu.GetStatus(neuron, "spike_count")
#print (spike_count)

if (len(spike) != len(spike_count)):
    print("Error: len(spike) != len(spike_count)")
    print("len(spike) ", len(spike))
    print("len(spike_count) ", len(spike_count)) 
    sys.exit(1)
    
for i in range(len(spike)):
    #print spike_count[i][0]
    #print (spike_count[i], spike[i])
    diff = spike[i] - spike_count[i][0]
    if abs(diff) > eps:
        print("Error: inconsistent number of spikes of node n. ", i)
        print("spike detector count ", spike[i])
        print("node count ", spike_count[i][0])
        sys.exit(1)
        
if (len(spike_times) != len(neuron)):
    print("Error: len(spike_times) != len(neuron)")
    print("len(spike_times) ", len(spike_times))
    print("len(neuron) ", len(neuron)) 
    sys.exit(1)


spike_times_list = ngpu.GetRecSpikeTimes(neuron)
for j in range(len(neuron)):
    spike_times1 = spike_times_list[j]
    #print (spike_times1)
    #print (spike_times[j])
    if (len(spike_times1) != spike_count[j][0]):
        print("Error: inconsistent number of spikes of node n. ", j)
        print("n. of recorded spike times ", len(spike_times1))
        print("node count ", spike_count[j][0])
        sys.exit(1)
    
    for i in range(len(spike_times1)):
        spike_times1[i]=round(spike_times1[i],4)
        diff = spike_times1[i] - spike_times[j][i]
        if abs(diff) > eps:
            print("Error: inconsistent recorded spikes times of node n. ", j, \
                  " spike n. ", i)
            print("multimeter spike time ", spike_times[j][i])
            print("node recorded spike time ", spike_times1[i])
            sys.exit(1)

sys.exit(0)
