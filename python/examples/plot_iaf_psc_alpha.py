import sys
import nestgpu as ngpu
import numpy as np
tolerance = 0.00005

neuron = ngpu.Create('iaf_psc_alpha')
spike = ngpu.Create("spike_generator")
spike_times = [10.0, 400.0]
n_spikes = 2

print(ngpu.GetStatus(neuron))
ngpu.SetStatus(neuron, {"tau_syn_ex": 10.0, "tau_syn_in": 5.0})

# set spike times and height
ngpu.SetStatus(spike, {"spike_times": spike_times})
delay = [1.0, 100.0]
weight = [1.0, -2.0]

conn_spec={"rule": "all_to_all"}

syn_spec_ex={'receptor':0, 'weight': weight[0], 'delay': delay[0]}
syn_spec_in={'receptor':1, 'weight': weight[1], 'delay': delay[1]}
ngpu.Connect(spike, neuron, conn_spec, syn_spec_ex)
ngpu.Connect(spike, neuron, conn_spec, syn_spec_in)

record = ngpu.CreateRecord("", ["V_m_rel"], [neuron[0]], [0])
#voltmeter = nest.Create('voltmeter')
#nest.Connect(voltmeter, neuron)

ngpu.Simulate(800.0)

data_list = ngpu.GetRecordData(record)
t=[row[0] for row in data_list]
V_m=[-70.0+row[1] for row in data_list]

data = np.loadtxt('../test/test_iaf_psc_alpha_nest.txt', delimiter="\t")
t1=[x[0]+0.1 for x in data ]
V_m1=[x[1] for x in data ]
print (len(t))
print (len(t1))

import matplotlib.pyplot as plt

fig1 = plt.figure(1)
plt.plot(t, V_m, "r-", label="NEST GPU")
plt.plot(t1, V_m1, "b--", label="NEST")
plt.legend()
plt.draw()
plt.pause(1)
ngpu.waitenter("<Hit Enter To Close>")
plt.close()
