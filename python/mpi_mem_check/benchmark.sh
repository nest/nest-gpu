#!/bin/bash -x

for P in 2 3 4 5 6; do
    for N in 1 10 100 1000 10000; do
        R=0
        id=P$P-N$N-R$R
        nsys profile -t cuda,mpi,nvtx --cuda-memory-usage=true --force-overwrite true -o report_$id mpirun -np $P python3 test.py --N=$N --R=$R |& tee log_$id.txt
    
        for C in 1 10 100 1000 10000; do
    	    for R in 1 2 3; do
    	        id=P$P-N$N-C$C-R$R
    	        nsys profile -t cuda,mpi,nvtx --cuda-memory-usage=true --force-overwrite true -o report_$id mpirun -np $P python3 test.py --N=$N --C=$C --R=$R |& tee log_$id.txt
    	    done;
        done;
    
        R=4
        id=P$P-N$N-R$R
        nsys profile -t cuda,mpi,nvtx --cuda-memory-usage=true --force-overwrite true -o report_$id mpirun -np $P python3 test.py --N=$N --R=$R |& tee log_$id.txt
    done;
done;
