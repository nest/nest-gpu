#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test for checking memory consumption during network construction.
Run with:

nsys profile -t cuda,mpi,nvtx --cuda-memory-usage=true --force-overwrite true -o report mpirun -np $NP python3 test.py |& tee log.txt && nsys-ui report.nsys-rep

Where $NP is the number (2 or more) of MPI processes to be used.

Arguments:
  N: Number of neurons to be created PER MPI process. Integer.
  C: Number of connections to be created between neuron populations. Integer.
  	The generated network connects all populations to each other. No self connections.
  R: Connection rule to be used. Integer.
  	0 -> No connection rule i.e. script will only create neurons.
  	1 -> Fixed indegree rule.
  	2 -> Fixed outdegree rule.
  	3 -> Fixed total number rule.
  	4 -> All to all connections. Argument C will be ignored.

"""


import nestgpu as ngpu

from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--N", type=int, default=1000)
parser.add_argument("--C", type=int, default=10000)
parser.add_argument("--R", type=int, default=1)
args = parser.parse_args()

rules_dict = {
    0: None,
    1: [{"rule": "fixed_indegree"}, "indegree"],
    2: [{"rule": "fixed_outdegree"}, "outdegree"],
    3: [{"rule": "fixed_total_number"}, "total_num"],
    4: [{"rule": "all_to_all"}],
}

assert args.N > 0 and args.C > 0 and args.R in rules_dict


ngpu.ConnectMpiInit()


mpi_id = ngpu.HostId()
mpi_np = ngpu.HostNum()
rank_list = list(range(mpi_np))


rule = rules_dict[args.R]
if rule is not None:
    if len(rule) > 1:
        rule[0].update({rule[1]: args.C})


if mpi_id == 0:
    print(f"Executing test with arguments: {args}")
    print(f"Running with {mpi_np} MPI ranks")
    print(f"Creating {args.N} neurons per MPI rank")
    print(f"Connection rule: {rule}")


neurons = []
for i in rank_list:
    neurons.append(ngpu.RemoteCreate(i, 'iaf_psc_exp', args.N, 1, {}).node_seq)


if rule is not None:
    for i in rank_list:
        for j in rank_list:
            if i != j:
                ngpu.RemoteConnect(i, neurons[i], j, neurons[j], rule[0], {})


ngpu.MpiFinalize()

