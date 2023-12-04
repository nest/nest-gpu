import sys

import nestgpu as ngpu

neuron = ngpu.Create("user_m1", 1, 3)
ngpu.SetStatus(neuron, {"V_peak": 0.0, "a": 4.0, "b": 80.5, "E_L": -70.6, "g_L": 300.0})
ngpu.SetStatus(
    neuron,
    {
        "E_rev": [20.0, 0.0, -85.0],
        "tau_decay": [40.0, 20.0, 30.0],
        "tau_rise": [20.0, 10.0, 5.0],
    },
)
spike = ngpu.Create("spike_generator")
spike_times = [10.0, 400.0]
spike_heights = [1.0, 0.5]
n_spikes = 2

# set spike times and height
ngpu.SetStatus(spike, {"spike_times": spike_times, "spike_heights": spike_heights})
delay = [1.0, 100.0, 130.0]
weight = [0.1, 0.2, 0.15]

conn_spec = {"rule": "all_to_all"}
for syn in range(3):
    syn_spec = {  #'model': 'static_synapse', 'receptor_type': syn,
        "receptor": syn,
        "weight": weight[syn],
        "delay": delay[syn],
    }
    ngpu.Connect(spike, neuron, conn_spec, syn_spec)

record = ngpu.CreateRecord("", ["V_m"], [neuron[0]], [0])

ngpu.Simulate(800.0)

data_list = ngpu.GetRecordData(record)
t = [row[0] for row in data_list]
V_m = [row[1] for row in data_list]

import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(t, V_m)

plt.draw()
plt.pause(1)
raw_input("<Hit Enter To Close>")
plt.close()
