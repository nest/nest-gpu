#!/bin/bash -x

:>full_test.dat
:>full_req_mem.dat
:>full_out_of_mem.dat

for P in 2 3 4 5 6; do
    for N in 1 10 100 1000 10000; do
        R=0
        id=P$P-N$N-R$R
	for iP in $(seq 0 $(($P-1))); do
	    :> test_$iP.dat
	    :> req_mem_$iP.dat
	done
        mpirun -np $P python3 test_16b.py --N=$N --R=$R |& tee log_$id.txt
	for iP in $(seq 0 $(($P-1))); do
	    cat test_$iP.dat >> full_test.dat
	    cat req_mem_$iP.dat >> full_req_mem.dat
	done
	l=$(cat test_0.dat | wc -l)
	if [ $l -eq 0 ]; then
	    cat req_mem_0.dat >> full_out_of_mem.dat
	fi
        for C in 1 10 100 1000 10000; do
    	    for R in 1 2 3; do
    	        id=P$P-N$N-C$C-R$R
		for iP in $(seq 0 $(($P-1))); do
		    :> test_$iP.dat
		    :> req_mem_$iP.dat
		done
    	        mpirun -np $P python3 test_16b.py --N=$N --C=$C --R=$R |& tee log_$id.txt
		for iP in $(seq 0 $(($P-1))); do
		    cat test_$iP.dat >> full_test.dat
		    cat req_mem_$iP.dat >> full_req_mem.dat
		done
		l=$(cat test_0.dat | wc -l)
		if [ $l -eq 0 ]; then
		    cat req_mem_0.dat >> full_out_of_mem.dat
		fi
    	    done
        done
    
        R=4
        id=P$P-N$N-R$R
	for iP in $(seq 0 $(($P-1))); do
	    :> test_$iP.dat
	    :> req_mem_$iP.dat
	done
        mpirun -np $P python3 test_16b.py --N=$N --R=$R |& tee log_$id.txt
	for iP in $(seq 0 $(($P-1))); do
	    cat test_$iP.dat >> full_test.dat
	    cat req_mem_$iP.dat >> full_req_mem.dat
	done
	l=$(cat test_0.dat | wc -l)
	if [ $l -eq 0 ]; then
	    cat req_mem_0.dat >> full_out_of_mem.dat
	fi
    done
done
