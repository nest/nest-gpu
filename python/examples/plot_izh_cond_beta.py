import sys

import nestgpu as ngpu
import numpy as np

tolerance = 0.0005
neuron = ngpu.Create("izhikevich_cond_beta", 1, 3)
# ngpu.SetStatus(neuron, {'u':-0.2*70.0, 'V_m':-70.0})
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
n_spikes = 2

# set spike times and heights
ngpu.SetStatus(spike, {"spike_times": spike_times})
delay = [1.0, 100.0, 130.0]
fact = 0.00225
weight = [0.1 * fact, 0.2 * fact, 0.5 * fact]

conn_spec = {"rule": "all_to_all"}
for syn in range(3):
    syn_spec = {"receptor": syn, "weight": weight[syn], "delay": delay[syn]}
    ngpu.Connect(spike, neuron, conn_spec, syn_spec)

record = ngpu.CreateRecord("", ["V_m"], [neuron[0]], [0])
# voltmeter = nest.Create('voltmeter')
# nest.Connect(voltmeter, neuron)

ngpu.Simulate(800.0)

data_list = ngpu.GetRecordData(record)
t = [row[0] for row in data_list]
V_m = [row[1] for row in data_list]
# dmm = nest.GetStatus(voltmeter)[0]
# V_m = dmm["events"]["V_m"]
# t = dmm["events"]["times"]
# with open('test_user_m1_nest.txt', 'w') as f:
#    for i in range(len(t)):
#        f.write("%s\t%s\n" % (t[i], V_m[i]))

data = np.loadtxt("../test/test_aeif_cond_beta_nest.txt", delimiter="\t")
t1 = [x[0] for x in data]
V_m1 = [x[1] for x in data]
print(len(t))
print(len(t1))

dV = [V_m[i * 10 + 20] - V_m1[i] for i in range(len(t1))]
rmse = np.std(dV) / abs(np.mean(V_m))
print("rmse : ", rmse, " tolerance: ", tolerance)
# if rmse>tolerance:
#    sys.exit(1)

# sys.exit(0)
import matplotlib.pyplot as plt

fig1 = plt.figure(1)
plt.plot(t, V_m, "r-", label="NEST GPU")
plt.plot(t1, V_m1, "b--", label="NEST")
plt.legend()
plt.draw()
plt.pause(1)
ngpu.waitenter("<Hit Enter To Close>")
plt.close()
