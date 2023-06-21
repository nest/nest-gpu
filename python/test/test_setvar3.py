import sys
import nestgpu as ngpu

n_neurons = 6

# create n_neurons neurons with 2 receptor ports
neuron = ngpu.Create('aeif_cond_beta_multisynapse', n_neurons, 2)
neuron_even = [neuron[0], neuron[2], neuron[4]]
neuron_odd = [neuron[3], neuron[5], neuron[1]]
ngpu.SetStatus(neuron_even, {'tau_decay':[80.0, 40.0],
                             'tau_rise':[60.0, 20.0]})
ngpu.SetStatus(neuron_odd, {'tau_decay':[70.0, 30.0],
                            'tau_rise':[50.0, 10.0]})

ngpu.SetStatus(neuron_even, {'V_m':-80.0})
ngpu.SetStatus(neuron_odd, {'V_m':-90.0})

ngpu.SetStatus(neuron_even, {'g1':[0.4, 0.2]})
ngpu.SetStatus(neuron_odd, {'g1':[0.3, 0.1]})

ngpu.SetStatus(neuron_even, {'V_th':-40.0})
ngpu.SetStatus(neuron_odd, {'V_th':-30.0})

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
neuron_list = [neuron[0], neuron[2], neuron[4], neuron[1], neuron[3],
               neuron[5]]
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

print(ngpu.GetNScalVar(neuron[0]))
print(ngpu.GetScalVarNames(neuron[0]))
print(ngpu.GetNPortVar(neuron[0]))
print(ngpu.GetPortVarNames(neuron[0]))

print(ngpu.GetNScalParam(neuron[0]))
print(ngpu.GetScalParamNames(neuron[0]))
print(ngpu.GetNPortParam(neuron[0]))
print(ngpu.GetPortParamNames(neuron[0]))

print()

print(ngpu.GetStatus(neuron[0]))

print()

print(ngpu.GetStatus(neuron_odd))

print()

print(ngpu.GetStatus(neuron_even, "tau_rise"))

print()

print(ngpu.GetStatus(neuron_odd, "tau_rise"))

print()

print(ngpu.GetStatus(neuron_even, "V_m"))

print()

print(ngpu.GetStatus(neuron_odd, "V_m"))


