#!/bin/bash -x

NP=2
if [ ! -z $1 ]; then
    NP=$1
fi

mpirun -np $NP python3 test.py |& tee log.txt
