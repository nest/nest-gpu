import sys
import math
import ctypes
import nestgpu as ngpu
from random import randrange
import numpy as np


ngpu.ConnectMpiInit()
mpi_np = ngpu.HostNum()

if mpi_np != 2:
    print("Usage: mpirun -np 2 python %s" % sys.argv[0])
    quit()

mpi_id = ngpu.HostId()
print("Building on host ", mpi_id, " ...")

ngpu.SetKernelStatus(
    {"check_node_maps": True, "rnd_seed": 1234}
)  # seed for GPU random numbers

neuron = ngpu.Create("iaf_psc_exp", 200)

delay = 2.0
weight = 1.0
syn_spec = {"weight": weight, "delay": delay}

conn_spec1 = {"rule": "fixed_total_number", "total_num": 50}
conn_spec2 = {"rule": "fixed_total_number", "total_num": 200}

ngpu.RemoteConnect(0, neuron, 1, neuron, conn_spec1, syn_spec)
ngpu.RemoteConnect(0, neuron[80:130], 1, neuron, conn_spec2, syn_spec)
ngpu.RemoteConnect(0, neuron, 1, neuron, conn_spec1, syn_spec)
ngpu.RemoteConnect(0, neuron[120:170], 1, neuron, conn_spec2, syn_spec)

ngpu.RemoteConnect(1, neuron, 0, neuron, conn_spec1, syn_spec)
ngpu.RemoteConnect(1, neuron[80:130], 0, neuron, conn_spec2, syn_spec)
ngpu.RemoteConnect(1, neuron, 0, neuron, conn_spec1, syn_spec)
ngpu.RemoteConnect(1, neuron[120:170], 0, neuron, conn_spec2, syn_spec)

ngpu.Calibrate

ngpu.MpiFinalize()
