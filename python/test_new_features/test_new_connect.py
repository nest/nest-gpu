import nestgpu as ngpu

neuron = ngpu.Create('iaf_psc_exp', 2)
sg = ngpu.Create('spike_generator', 3)
spike_times = [10.0, 400.0, 600.0]
# set spike times
for i in range(3):
    ngpu.SetStatus(sg[i:i+1], {"spike_times": [spike_times[i]]})

conn_dict={"rule": "all_to_all"}
syn_dict={"delay": {"distribution":"normal_clipped",
                    "mu":0.4, "low":0.1, "high":1.0, "sigma":0.2},
          "weight": {"distribution":"normal_clipped",
                     "mu":1.5, "low":0.5, "high":2.0, "sigma":0.25}}

ngpu.Connect(sg, neuron, conn_dict, syn_dict)

filename = "test_new_connect.dat"
i_neuron_arr = [neuron[0], neuron[1]]
i_receptor_arr = [0, 0]
# create multimeter record of V_m
var_name_arr = ["V_m_rel", "V_m_rel"]
record = ngpu.CreateRecord(filename, var_name_arr, i_neuron_arr,
                           i_receptor_arr)

ngpu.Simulate(1000.0)

data_list = ngpu.GetRecordData(record)
t=[row[0] for row in data_list]
V1=[row[1] for row in data_list]
V2=[row[2] for row in data_list]

import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(t, V1)

plt.figure(2)
plt.plot(t, V2)

plt.draw()
plt.pause(0.5)
ngpu.waitenter("<Hit Enter To Close>")
plt.close()
