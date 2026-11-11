import sys
import math
import ctypes
import nestgpu as ngpu
from random import randrange
import numpy as np
from mpi4py import MPI


ngpu.ConnectMpiInit()
mpi_np = ngpu.HostNum()

if mpi_np != 3:
    print("Usage: mpirun -np 3 python %s" % sys.argv[0])
    quit()

mpi_id = ngpu.HostId()
print("Building on host ", mpi_id, " ...")

ngpu.SetKernelStatus("rnd_seed", 1234)  # seed for GPU random numbers

neuron = ngpu.Create("iaf_psc_exp_g", 3)

spike = ngpu.Create("spike_generator", 3)

for i in range(3):
    spike_times = [
        1.0 * (mpi_id * 20 + i * 5 + 10),
        50.0 + 1.0 * (mpi_id * 20 + i * 5 + 10),
    ]
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
                        delay = 100 + 100.0 * ith + 50.0 * itn
                        weight = 5.0 + 10.0 * isn
                        syn_spec = {"weight": weight, "delay": delay}
                        # print (ish, [spike[isn]], ith, [neuron[itn]])
                        # ngpu.RemoteConnect(ish, spike[isn:isn+1], \
                        #                   ith, neuron[itn:itn+1], \
                        #                   conn_spec, syn_spec)
                        ngpu.RemoteConnect(
                            ish, [spike[isn]], ith, [neuron[itn]], conn_spec, syn_spec
                        )


ngpu.Simulate(1)
conn_id = ngpu.GetConnections()
conn_status_dict = ngpu.GetStatus(conn_id)
# Gather the connections of all ranks on rank 0, which prints them ordered
# by rank, source and target ids, so that the output is deterministic
all_conn = MPI.COMM_WORLD.gather(conn_status_dict, root=0)
if mpi_id == 0:
    for rank, rank_conn in enumerate(all_conn):
        for c in sorted(rank_conn, key=lambda c: (c["source"], c["target"])):
            print("CHECK", rank, c)
    sys.stdout.flush()
ngpu.MpiFinalize()
