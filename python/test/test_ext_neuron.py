import sys

import nestgpu as ngpu
import numpy as np

tolerance = 0.0005
neuron = ngpu.Create("ext_neuron", 1, 3)
spike = ngpu.Create("spike_generator")
spike_times = [50.0, 100.0, 400.0, 600.0]
n_spikes = 4

# set spike times and heights
ngpu.SetStatus(spike, {"spike_times": spike_times})
delay = [1.0, 50.0, 100.0]
weight = [0.1, 0.2, 0.5]

conn_spec = {"rule": "all_to_all"}
for syn in range(3):
    syn_spec = {"receptor": syn, "weight": weight[syn], "delay": delay[syn]}
    ngpu.Connect(spike, neuron, conn_spec, syn_spec)

i_neuron_arr = [neuron[0], neuron[0], neuron[0]]
i_receptor_arr = [0, 1, 2]
var_name_arr = ["port_value", "port_value", "port_value"]
record = ngpu.CreateRecord("", var_name_arr, i_neuron_arr, i_receptor_arr)

ngpu.Simulate(800.0)

data_list = ngpu.GetRecordData(record)
t = [row[0] for row in data_list]
val1 = [row[1] for row in data_list]
val2 = [row[2] for row in data_list]
val3 = [row[3] for row in data_list]

import matplotlib.pyplot as plt

fig1 = plt.figure(1)
plt.plot(t, val1)
fig2 = plt.figure(2)
plt.plot(t, val2)
fig3 = plt.figure(3)
plt.plot(t, val3)

plt.draw()
plt.pause(1)
ngpu.waitenter("<Hit Enter To Close>")
plt.close()
