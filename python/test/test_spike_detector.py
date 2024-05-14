import sys
import nestgpu as ngpu

tolerance = 1.0e-6

neuron = ngpu.Create("aeif_cond_beta_multisynapse", 3)

ngpu.SetStatus(neuron, {"I_e":1000.0})

spike_det = ngpu.Create("spike_detector")


conn_dict={"rule": "one_to_one"}
syn_dict1={"weight": 1.0, "delay": 10.0, "receptor":0}
syn_dict2={"weight": 2.0, "delay": 20.0, "receptor":0}
syn_dict3={"weight": 3.0, "delay": 30.0, "receptor":0}

ngpu.Connect([neuron[0]], spike_det, conn_dict, syn_dict1)

ngpu.Connect([neuron[1]], spike_det, conn_dict, syn_dict2)

ngpu.Connect([neuron[2]], spike_det, conn_dict, syn_dict3)

record_n = ngpu.CreateRecord("", ["spike"], [neuron[0]], [0])

record_sd = ngpu.CreateRecord("", ["spike_mul"], [spike_det[0]], [0])

ngpu.Simulate()

data_n = ngpu.GetRecordData(record_n)
t_n=[row[0] for row in data_n]
spike_n=[row[1] for row in data_n]

data_sd = ngpu.GetRecordData(record_sd)
t_sd=[row[0] for row in data_sd]
spike_sd=[row[1] for row in data_sd]

for i in range(len(t_n)-400):
    if spike_n[i]>0.5:
        j1 = i + 101
        j2 = i + 201
        j3 = i + 301
        if abs(spike_sd[j1] - 1.0)>tolerance:
            print("Expected spike height: 1.0, simulated: ", spike_sd[j1])
            sys.exit(1)
        if abs(spike_sd[j2] - 2.0)>tolerance:
            print("Expected spike height: 2.0, simulated: ", spike_sd[j2])
            sys.exit(1)
        if abs(spike_sd[j3] - 3.0)>tolerance:
            print("Expected spike height: 3.0, simulated: ", spike_sd[j3])
            sys.exit(1)
            
#import matplotlib.pyplot as plt

#plt.figure(1)
#plt.plot(t_n, spike_n)

#plt.figure(2)
#plt.plot(t_sd, spike_sd)

#plt.draw()
#plt.pause(1)
#raw_input("<Hit Enter To Close>")
#plt.close()
sys.exit(0)
