import sys
import nest
import numpy as np

tolerance = 0.00005
neuron = nest.Create('iaf_psc_alpha')
spike = nest.Create("spike_generator")
spike_times = [10.0, 400.0]
n_spikes = 2

print(nest.GetStatus(neuron))
nest.SetStatus(neuron, {"tau_syn_ex": 10.0, "tau_syn_in": 5.0})

# set spike times and height
nest.SetStatus(spike, {"spike_times": spike_times})
delay = [1.0, 100.0]
weight = [1.0, -2.0]

conn_spec={"rule": "all_to_all"}

syn_spec_ex={'weight': weight[0], 'delay': delay[0]}
syn_spec_in={'weight': weight[1], 'delay': delay[1]}
nest.Connect(spike, neuron, conn_spec, syn_spec_ex)
nest.Connect(spike, neuron, conn_spec, syn_spec_in)

#record = nest.CreateRecord("", ["V_m_rel"], [neuron[0]], [0])
voltmeter = nest.Create('voltmeter')
nest.Connect(voltmeter, neuron)

nest.Simulate(800.0)

#data_list = nest.GetRecordData(record)
#t=[row[0] for row in data_list]
#V_m=[-70.0+row[1] for row in data_list]

#import matplotlib.pyplot as plt
#plt.figure()
#plt.plot(t, V_m, "r-")
#plt.show()

#sys.exit()
dmm = nest.GetStatus(voltmeter)[0]
V_m = dmm["events"]["V_m"]
t = dmm["events"]["times"]
with open('test_iaf_psc_alpha_nest.txt', 'w') as f:
    for i in range(len(t)):
        f.write("%s\t%s\n" % (t[i], V_m[i]))

import matplotlib.pyplot as plt
plt.figure()
plt.plot(t, V_m, "r-")
plt.show()

sys.exit()
data = np.loadtxt('test_iaf_psc_alpha_nest.txt', delimiter="\t")
t1=[x[0] for x in data ]
V_m1=[x[1] for x in data ]
print (len(t))
print (len(t1))

dV=[V_m[i*10+20]-V_m1[i] for i in range(len(t1))]
rmse =np.std(dV)/abs(np.mean(V_m))
print("rmse : ", rmse, " tolerance: ", tolerance)
if rmse>tolerance:
    sys.exit(1)

sys.exit(0)
