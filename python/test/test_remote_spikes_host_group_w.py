import sys
import math
import ctypes
import nestgpu as ngpu
from random import randrange
import numpy as np


ngpu.ConnectMpiInit();
mpi_np = ngpu.HostNum()

if mpi_np != 3:
    print ("Usage: mpirun -np 3 python %s" % sys.argv[0])
    quit()

mpi_id = ngpu.HostId()
print("Building on host ", mpi_id, " ...")

ngpu.SetKernelStatus("rnd_seed", 1234) # seed for GPU random numbers

whg = ngpu.CreateHostGroup([0, 1, 2])

neuron = ngpu.Create('iaf_psc_exp_g', 3)

spike = ngpu.Create("spike_generator", 3)

for i in range(3):
    spike_times = [1.0*(mpi_id*20 + i*5 + 10), 50.0 + 1.0*(mpi_id*20 + i*5 + 10)]
    n_spikes = 2
    # set spike times and height
    ngpu.SetStatus([spike[i]], {"spike_times": spike_times})
    

conn_spec = {"rule": "one_to_one"}

for ish in range(3):
    for ith in range(3):
        if ish != ith:
            for isn in range(3):
                for itn in range(3):
                    if itn != isn:
                        delay = 100 + 100.0*ith + 50.0*itn
                        weight = 5.0 + 10.0*isn
                        syn_spec = {'weight': weight, 'delay': delay}
                        #print(f"PyRC this_host_ {mpi_id}, ish {ish}, isn {spike[isn]}, ith {ith}, itn {neuron[itn]}")
                        ngpu.RemoteConnect(ish, [spike[isn]], \
                                           ith, [neuron[itn]], \
                                           conn_spec, syn_spec, whg)

i_neuron_arr = [neuron[0], neuron[1], neuron[2]]
i_receptor_arr = [0, 0, 0]
# create multimeter record of V_m
var_name_arr = ["V_m_rel", "V_m_rel", "V_m_rel"]
record = ngpu.CreateRecord("", var_name_arr, i_neuron_arr,
                           i_receptor_arr)

ngpu.Simulate(1000.0)

nrows=ngpu.GetRecordDataRows(record)
ncol=ngpu.GetRecordDataColumns(record)
#print nrows, ncol

data_list = ngpu.GetRecordData(record)
t=[row[0] for row in data_list]
V1=[row[1] for row in data_list]
V2=[row[2] for row in data_list]
V3=[row[3] for row in data_list]

import matplotlib.pyplot as plt

fig1 = plt.figure(mpi_id*3 + 1)
fig1.suptitle("host " + str(mpi_id), fontsize=20)
plt.plot(t, V1)
delay = 100 + 100.0*mpi_id
plt.xlim(delay, delay+150)

fig2 = plt.figure(mpi_id*3 + 2)
fig2.suptitle("host " + str(mpi_id), fontsize=20)
plt.plot(t, V2)
delay = 100 + 100.0*mpi_id + 50.0
plt.xlim(delay, delay+150)

fig3 = plt.figure(mpi_id*3 + 3)
fig3.suptitle("host " + str(mpi_id), fontsize=20)
plt.plot(t, V3)
delay = 100 + 100.0*mpi_id + 100.0
plt.xlim(delay, delay+150)

#print("Check")
i_fig = 0
spike_list = []
for ith in range(3):
    for itn in range(3):
        spike_list.append([])    
        for ish in range(3):
            if ish != ith:
                for isn in range(3):
                    if itn != isn:
                        spike
                        delay = 100 + 100.0*ith + 50.0*itn
                        weight = 5.0 + 10.0*isn
                        spike_times1 = 1.0*(ish*20 + isn*5 + 10)
                        spike_times2 = 50.0 + spike_times1
                        x1 = delay + spike_times1
                        x2 = delay + spike_times2
                        spike_list[i_fig].append(x1)
                        spike_list[i_fig].append(x2)
                        #print (ish, [spike[isn]], ith, [neuron[itn]])
                        if ith==mpi_id:
                            plt.figure(mpi_id*3 + itn + 1)
                            plt.axvline(x=x1, color='r')
                            plt.axvline(x=x2, color='r')
        i_fig = i_fig + 1
        
if (mpi_id==0):
    print(spike_list)
    
ngpu.MpiFinalize()

plt.show()

