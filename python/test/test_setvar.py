import sys
import nestgpu as ngpu

n_neurons = 3

# create n_neurons neurons with 2 receptor ports
neuron = ngpu.Create('aeif_cond_beta_multisynapse', n_neurons, 2)
ngpu.SetStatus(neuron, {'tau_decay':[60.0, 10.0],
                        'tau_rise':[40.0, 5.0]})

neuron0 = neuron[0:0]
neuron1 = neuron[1:1]
neuron2 = neuron[2:2]
  
ngpu.SetStatus(neuron0, {'V_m':-80.0})
ngpu.SetStatus(neuron1, {'g1':[0.0, 0.1]})
ngpu.SetStatus(neuron2, {'g1':[0.1, 0.0]})


# reading parameters and variables test
read_td = ngpu.GetNeuronStatus(neuron, "tau_decay")
read_tr = ngpu.GetNeuronStatus(neuron, "tau_rise")
read_Vm = ngpu.GetNeuronStatus(neuron, "V_m")
read_Vth = ngpu.GetNeuronStatus(neuron, "V_th")
read_g1 = ngpu.GetNeuronStatus(neuron, "g1")

print("read_td", read_td)
print("read_tr", read_tr)
print("read_Vm", read_Vm)
print("read_Vth", read_Vth)
print("read_g1", read_g1)

# reading parameters and variables from neuron list test
neuron_list = [neuron[1], neuron[2], neuron[0]]
read1_td = ngpu.GetNeuronStatus(neuron_list, "tau_decay")
read1_tr = ngpu.GetNeuronStatus(neuron_list, "tau_rise")
read1_Vm = ngpu.GetNeuronStatus(neuron_list, "V_m")
read1_Vth = ngpu.GetNeuronStatus(neuron_list, "V_th")
read1_g1 = ngpu.GetNeuronStatus(neuron_list, "g1")

print("read1_td", read1_td)
print("read1_tr", read1_tr)
print("read1_Vm", read1_Vm)
print("read1_Vth", read1_Vth)
print("read1_g1", read1_g1)

i_neuron_arr = [neuron[0], neuron[1], neuron[2]]
i_receptor_arr = [0, 0, 0]
# create multimeter record of V_m
var_name_arr = ["V_m", "V_m", "V_m"]
record = ngpu.CreateRecord("", var_name_arr, i_neuron_arr,
                           i_receptor_arr)

ngpu.Simulate(800.0)

data_list = ngpu.GetRecordData(record)
t=[row[0] for row in data_list]
V1=[row[1] for row in data_list]
V2=[row[2] for row in data_list]
V3=[row[3] for row in data_list]

import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(t, V1)

plt.figure(2)
plt.plot(t, V2)

plt.figure(3)
plt.plot(t, V3)

plt.draw()
plt.pause(0.5)
raw_input("<Hit Enter To Close>")
