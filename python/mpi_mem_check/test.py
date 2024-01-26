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
  T: Connection struct type. Intege
        0 -> 12 byte connection structure
        1 -> 16 byte connection structure

"""

from mpi4py import MPI
import nestgpu as ngpu

from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--N", type=int, default=1000)
parser.add_argument("--C", type=int, default=10000)
parser.add_argument("--R", type=int, default=1)
parser.add_argument("--T", type=int, default=0)
args = parser.parse_args()

rules_dict = {
    0: None,
    1: [{"rule": "fixed_indegree"}, "indegree"],
    2: [{"rule": "fixed_outdegree"}, "outdegree"],
    3: [{"rule": "fixed_total_number"}, "total_num"],
    4: [{"rule": "all_to_all"}],
}

conn_struct_type = args.T
assert args.N > 0 and args.C > 0 and args.R in rules_dict and args.T >= 0 and args.T <= 1  

ngpu.SetKernelStatus({"verbosity_level": 5, "conn_struct_type": conn_struct_type})

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

block_size = 10000000
bytes_per_storage = 4
bytes_per_node = 4
if conn_struct_type==0:
    bytes_per_conn = 12
else:
    bytes_per_conn = 16

margin = 10 # margin in MB

if args.R==0:
    cuda_mem_exp = 0
    cuda_mem_exp_woh = 0
else:
    if args.R==1 or args.R==2:
        n_conn = int(args.C*args.N)
    elif args.R==3:
        n_conn = int(args.C)
    elif args.R==4:
        n_conn = int(args.N*args.N)
    else:
        n_conn = int(0)

    n_blocks = (n_conn*(mpi_np - 1) - 1) // block_size + 1

    cuda_mem_exp = (n_blocks*block_size*bytes_per_conn \
                    + block_size*bytes_per_storage)/1024/1024

    cuda_mem_exp_oh = n_conn*bytes_per_node/1024/1024
    
    cuda_mem_exp_woh = cuda_mem_exp + cuda_mem_exp_oh

# Total CUDA memory (for all hosts)
cuda_mem_tot = ngpu.getCUDAMemTotal()/1024/1024

# Free CUDA memory (for all hosts)
cuda_mem_free = ngpu.getCUDAMemFree()/1024/1024


req_mem_str = f"{mpi_np}\t{mpi_id}\t{args.N}\t{args.C}\t{args.R}\t" \
    f"{cuda_mem_tot:>9.3f}\t{cuda_mem_free:>9.3f}\t" \
    f"{cuda_mem_exp:>9.3f}\t{cuda_mem_exp_woh:>9.3f}\n"

print(f"CUDA available and requested memory summary\n"
      f"mpi_np\tmpi_id\tN\tC\tR\ttotal (MB)\tfree (MB)\t"
      f"exp/hst(no OH)\texp/hst(+OH)\n" + req_mem_str)

req_mem_file_name = f"req_mem_{mpi_id}.dat"
with open(req_mem_file_name, "w") as req_mem_file:
    req_mem_file.write(req_mem_str)


comm = MPI.COMM_WORLD
comm.Barrier()

neurons = []
for i in rank_list:
    neurons.append(ngpu.RemoteCreate(i, 'iaf_psc_exp', args.N, 1, {}).node_seq)


if rule is not None:
    for i in rank_list:
        for j in rank_list:
            if i != j:
                ngpu.RemoteConnect(i, neurons[i], j, neurons[j], rule[0], {})

cuda_mem_used = ngpu.getCUDAMemHostUsed()/1024/1024

cuda_mem_max = ngpu.getCUDAMemHostPeak()/1024/1024

if cuda_mem_max>=cuda_mem_exp and cuda_mem_max<(cuda_mem_exp_woh+margin):
    test_passed = 1
else:
    test_passed = 0
    
out_str = f"{mpi_np}\t{mpi_id}\t{args.N}\t{args.C}\t{args.R}\t" \
    f"{cuda_mem_used:>9.3f}\t{cuda_mem_max:>9.3f}\t" \
    f"{cuda_mem_exp:>9.3f}\t{cuda_mem_exp_woh:>9.3f}\t" \
    f"{test_passed}\n"

print(f"CUDA memory usage summary\n"
      f"mpi_np\tmpi_id\tN\tC\tR\tused (MB)\tmax (MB)\t"
      f"exp/hst(no OH)\texp/hst(+OH)\t"
      f"passed\n" + out_str)

test_file_name = f"test_{mpi_id}.dat"
with open(test_file_name, "w") as test_file:
    test_file.write(out_str)

ngpu.MpiFinalize()

