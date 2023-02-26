import sys
import nestgpu as ngpu
import numpy as np
tolerance = 5e-6
neuron = ngpu.Create('aeif_psc_exp', 1)
ngpu.SetStatus(neuron, {"V_peak": 0.0, "a": 4.0, "b":80.5,
                        "E_L":-70.6,
                        "g_L":300.0, 
                        "tau_syn_ex": 40.0,
                        "tau_syn_in": 20.0})
spike = ngpu.Create("spike_generator")
spike_times = [10.0, 400.0]
n_spikes = 2

# set spike times and height
ngpu.SetStatus(spike, {"spike_times": spike_times})
delay = [1.0, 100.0]
weight = [1.0, 2.0]

conn_spec={"rule": "all_to_all"}


syn_spec_ex={'receptor':0, 'weight': weight[0], 'delay': delay[0]}
syn_spec_in={'receptor':1, 'weight': weight[1], 'delay': delay[1]}
ngpu.Connect(spike, neuron, conn_spec, syn_spec_ex)
ngpu.Connect(spike, neuron, conn_spec, syn_spec_in)

record = ngpu.CreateRecord("", ["V_m"], [neuron[0]], [0])
#voltmeter = nest.Create('voltmeter')
#nest.Connect(voltmeter, neuron)

ngpu.Simulate(800.0)

data_list = ngpu.GetRecordData(record)
t=[row[0] for row in data_list]
V_m=[row[1] for row in data_list]
#dmm = nest.GetStatus(voltmeter)[0]
#V_m = dmm["events"]["V_m"]
#t = dmm["events"]["times"]
#with open('test_aeif_psc_exp_nest.txt', 'w') as f:
#    for i in range(len(t)):
#        f.write("%s\t%s\n" % (t[i], V_m[i]))

data = np.loadtxt('test_aeif_psc_exp_nest.txt', delimiter="\t")
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
