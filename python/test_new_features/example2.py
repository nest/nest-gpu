import nestgpu as ngpu

neuron = ngpu.Create("aeif_cond_beta")
poiss_gen = ngpu.Create("poisson_generator");

ngpu.SetStatus(poiss_gen, "rate", 12000.0)

conn_dict={"rule": "all_to_all"}
syn_dict={"weight": 0.05, "delay": 2.0, "receptor":0}

ngpu.Connect(poiss_gen, neuron, conn_dict, syn_dict)
record = ngpu.CreateRecord("", ["V_m"], [neuron[0]], [0])

ngpu.Simulate()

data_list = ngpu.GetRecordData(record)
t=[row[0] for row in data_list]
V_m=[row[1] for row in data_list]

import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(t, V_m)

plt.draw()
plt.pause(1)
ngpu.waitenter("<Hit Enter To Close>")
plt.close()
