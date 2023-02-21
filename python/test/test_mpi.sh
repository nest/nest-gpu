mpi_pass_str[0]="MPI TEST PASSED"
mpi_pass_str[1]="MPI TEST NOT PASSED"
:>log.txt
for fn in test_brunel_mpi.py test_brunel_outdegree_mpi.py; do
    mpirun -np 2 python3 $fn >> log.txt 2>err.txt
    res=$?
    cat err.txt >> log.txt
    rm -f err.txt
    if [ "$res" -ne "0" ]; then
        res=1
    fi
    echo $fn : ${mpi_pass_str[$res]}
done
