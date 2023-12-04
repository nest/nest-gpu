import nestgpu as ngpu

neuron = ngpu.Create("aeif_cond_beta", 2)
neuron0 = neuron[0:0]
neuron1 = neuron[1:1]

ngpu.SetStatus(neuron0, {"I_e": 1000.0})

parrot = ngpu.Create("parrot_neuron", 2)
parrot0 = parrot[0:0]
parrot1 = parrot[1:1]

conn_dict = {"rule": "one_to_one"}
syn_dict0 = {"weight": 0.5, "delay": 1.0, "receptor": 0}
ngpu.Connect(neuron0, parrot0, conn_dict, syn_dict0)
syn_dict1 = {"weight": 0.1, "delay": 1.0, "receptor": 0}
ngpu.Connect(parrot0, neuron1, conn_dict, syn_dict1)
ngpu.Connect(parrot0, parrot1, conn_dict, syn_dict1)

neuron0_record = ngpu.CreateRecord("", ["V_m"], [neuron0[0]], [0])
parrot0_record = ngpu.CreateRecord("", ["V", "spike"], [parrot0[0], parrot0[0]], [0, 0])
neuron1_record = ngpu.CreateRecord("", ["g1"], [neuron1[0]], [0])
parrot1_record = ngpu.CreateRecord("", ["V", "spike"], [parrot1[0], parrot1[0]], [0, 0])

ngpu.Simulate()

neuron0_data_list = ngpu.GetRecordData(neuron0_record)
t_neuron0 = [row[0] for row in neuron0_data_list]
V_m = [row[1] for row in neuron0_data_list]

parrot0_data_list = ngpu.GetRecordData(parrot0_record)
t_parrot0 = [row[0] for row in parrot0_data_list]
V_parrot0 = [row[1] for row in parrot0_data_list]
spike_parrot0 = [row[2] for row in parrot0_data_list]

neuron1_data_list = ngpu.GetRecordData(neuron1_record)
t_neuron1 = [row[0] for row in neuron1_data_list]
g1 = [row[1] for row in neuron1_data_list]

parrot1_data_list = ngpu.GetRecordData(parrot1_record)
t_parrot1 = [row[0] for row in parrot1_data_list]
V_parrot1 = [row[1] for row in parrot1_data_list]
spike_parrot1 = [row[2] for row in parrot1_data_list]

import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(t_neuron0, V_m)

plt.figure(2)
plt.plot(t_parrot0, V_parrot0)

plt.figure(3)
plt.plot(t_parrot0, spike_parrot0)

plt.figure(4)
plt.plot(t_neuron1, g1)

plt.figure(5)
plt.plot(t_parrot1, V_parrot1)

plt.figure(6)
plt.plot(t_parrot1, spike_parrot1)

plt.draw()
plt.pause(1)
raw_input("<Hit Enter To Close>")
plt.close()
