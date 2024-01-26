#!/bin/bash -x

:>full_test.dat
:>full_req_mem.dat
:>full_out_of_mem.dat

for T in 0 1; do
    for P in $(cat n_mpi_list.txt); do
	for N in $(cat n_neuron_list.txt); do
	    R=0
	    id=P$P-N$N-R$R
	    for iP in $(seq 0 $(($P-1))); do
		:> test_$iP.dat
		:> req_mem_$iP.dat
	    done
	    mpirun -np $P python3 test.py --N=$N --R=$R --T=$T |& tee log_$id.txt
	    for iP in $(seq 0 $(($P-1))); do
		cat test_$iP.dat >> full_test.dat
		cat req_mem_$iP.dat >> full_req_mem.dat
	    done
	    l=$(cat test_0.dat | wc -l)
	    if [ $l -eq 0 ]; then
		cat req_mem_0.dat >> full_out_of_mem.dat
	    fi
	    for C in $(cat n_conn_list.txt); do
		for R in 1 2 3; do
		    id=P$P-N$N-C$C-R$R
		    for iP in $(seq 0 $(($P-1))); do
			:> test_$iP.dat
			:> req_mem_$iP.dat
		    done
		    mpirun -np $P python3 test.py --N=$N --C=$C --R=$R --T=$T |& tee log_$id.txt
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
	    mpirun -np $P python3 test.py --N=$N --R=$R --T=$T |& tee log_$id.txt
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
done

./summary.sh | tee summary.txt
