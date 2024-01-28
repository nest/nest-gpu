#evaluates total number of MPI processes launched by benchmark_terminal script
n_mpi=$(cat n_mpi_list.txt | head -1 | sed 's/^ *//;s/ *$//;s/  */+/g' | bc -l)
n_loop_neur=$(cat n_neuron_list.txt | head -1 | awk '{print NF}')
n_loop_conn=$(cat n_conn_list.txt | head -1 | awk '{print NF}')
Ntot_th=$(( 2 * ( $n_mpi * $n_loop_neur * ( 2 + 3 * $n_loop_conn ) ) ))        # 3400

cat full_req_mem.dat | awk '{print $1, $2, $3, $4, $5}' | sort -n > list_all.dat
Ntot_proc=$(cat list_all.dat | wc -l)
echo "$Ntot_proc MPI processes out of $Ntot_th expected have been processed"
if [ $Ntot_proc -lt $Ntot_th ]; then
    echo "Error: not all expected MPI processes have been processed"
elif [ $Ntot_proc -gt $Ntot_th ]; then
    echo "Error: number of processed MPI processes is larger than expected"
fi

N_complete=$(cat full_test.dat | wc -l)
echo "$N_complete MPI processes have been completed"
N_passed=$(cat full_test.dat | awk '{print $10}' | grep '^1' | wc -l)
echo "$N_passed MPI processes out of $N_complete completed have GPU memory usage in the predicted range"
N_notpassed=$(($N_complete - $N_passed))
if [ $N_notpassed -ne 0 ]; then
    cat full_test.dat | awk '{print $10}' | grep '^0'
    echo "$N_notpassed MPI processes out of $N_complete completed do not have GPU memory usage in the predicted range"
    echo "TEST NOT PASSED"
    exit 1
fi

cat full_test.dat | awk '{print $1, $2, $3, $4, $5}' | sort -n > list_complete.dat
diff list_complete.dat list_all.dat | grep '>' | awk '{print $2, $4, $5, $6}' | sort -n | uniq > list_not_complete.dat
diff list_complete.dat list_all.dat | grep '>' | awk '{print $2, $3, $4, $5, $6}' | sort -n > list_not_complete_proc.dat

N_not_complete_mpirun=$(cat list_not_complete.dat | wc -l)
echo "$N_not_complete_mpirun mpirun launches have not been completed"
N_not_complete=$(($Ntot_proc - $N_complete))
echo "$N_not_complete MPI processes have not been completed"
N_not_complete_check=$(cat list_not_complete_proc.dat | wc -l)
if [ $N_not_complete_check -ne $N_not_complete ]; then
    echo "Error: inconsistent number of MPI processes that have not been completed. Check this script"
fi

cat full_req_mem.dat | while read a b c d e f g h i; do
    out_of_mem=$(echo "($i * $a) > $g" | bc -l)
    echo "$a $b $c $d $e $f $g $h $i $out_of_mem"
done | grep '1$' | awk '{print $1, $2, $3, $4, $5, $6, $7, $8, $9}' | sort -n > list_out_of_mem_proc.dat

N_not_complete_expected=0
N_not_complete_unexpected=0
while read l; do
    if grep -q "^$l" list_out_of_mem_proc.dat; then
	N_not_complete_expected=$(( $N_not_complete_expected + 1 ))
    else
	N_not_complete_unexpected=$(( $N_not_complete_unexpected + 1 ))
    fi
done <<< "$(cat list_not_complete_proc.dat)"

echo -n "$N_not_complete_expected MPI processes out of $N_not_complete MPI processes that have not been completed"
echo " are in the list of the procesess that were predicted to go out of memory"
echo -n "$N_not_complete_unexpected MPI processes out of $N_not_complete MPI processes that have not been completed"
echo " are NOT in the list of the procesess that were predicted to go out of memory"
if [ $N_not_complete_unexpected -eq 0 ]; then
    echo "TEST PASSED"
    exit 0
else
    echo "TEST NOT PASSED"
    exit 1
fi
