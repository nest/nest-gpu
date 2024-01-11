#!/bin/bash -x

NP=2
if [ ! -z $1 ]; then
    NP=$1
fi

nsys profile -t cuda,mpi,nvtx --cuda-memory-usage=true --force-overwrite true -o report mpirun -np $NP python3 test.py |& tee log.txt && nsys-ui report.nsys-rep