import sys
import nestgpu as ngpu
import numpy as np
tolerance = 0.0005
neuron = ngpu.Create('iaf_psc_exp_g', 1)
spike = ngpu.Create("spike_generator")
spike_times = [10.0, 400.0]
n_spikes = 2

# set spike times and height
ngpu.SetStatus(spike, {"spike_times": spike_times})
delay = [1.0, 100.0]
weight = [1.0, -2.0]

conn_spec={"rule": "all_to_all"}


syn_spec_ex={'weight': weight[0], 'delay': delay[0]}
syn_spec_in={'weight': weight[1], 'delay': delay[1]}
ngpu.Connect(spike, neuron, conn_spec, syn_spec_ex)
ngpu.Connect(spike, neuron, conn_spec, syn_spec_in)

record = ngpu.CreateRecord("", ["V_m_rel"], [neuron[0]], [0])
#voltmeter = nest.Create('voltmeter')
#nest.Connect(voltmeter, neuron)

ngpu.Simulate(800.0)

data_list = ngpu.GetRecordData(record)
t=[row[0] for row in data_list]
V_m=[-65.0+row[1] for row in data_list]
#dmm = nest.GetStatus(voltmeter)[0]
#V_m = dmm["events"]["V_m"]
#t = dmm["events"]["times"]
#with open('test_iaf_psc_exp_nest.txt', 'w') as f:
#    for i in range(len(t)):
#        f.write("%s\t%s\n" % (t[i], V_m[i]))

data = np.loadtxt('../test/test_fast_iaf_psc_exp_nest.txt', delimiter="\t")
t1=[x[0] for x in data ]
V_m1=[x[1] for x in data ]
#print (len(t))
#print (len(t1))

#dV=[V_m[i*10+20]-V_m1[i] for i in range(len(t1))]
#rmse =np.std(dV)/abs(np.mean(V_m))
#print("rmse : ", rmse, " tolerance: ", tolerance)
#if rmse>tolerance:
#    sys.exit(1)

#sys.exit(0)
import matplotlib.pyplot as plt

fig1 = plt.figure(1)
plt.plot(t, V_m, "r-", label="NEST GPU")
plt.plot(t1, V_m1, "b--", label="NEST")
plt.legend()
plt.draw()
plt.pause(1)
ngpu.waitenter("<Hit Enter To Close>")
plt.close()
