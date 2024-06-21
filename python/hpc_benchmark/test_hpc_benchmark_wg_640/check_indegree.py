mpi_np = 640
mpi_id = 113

CE = 9000
CI = 2250

for j in range(mpi_np):
    CE_total = 0
    CI_total = 0
    for i in range(mpi_np):
        if(i!=j):
            CE_distrib = CE // mpi_np
            if ( (i + j) % mpi_np ) < ( CE % mpi_np ):
                CE_distrib = CE_distrib + 1
                    
            CI_distrib = CI // mpi_np
            if ( (i + j) % mpi_np ) < ( CI % mpi_np ):
                CI_distrib = CI_distrib + 1

            CE_total = CE_total + CE_distrib
            CI_total = CI_total + CI_distrib
            #print('{} {} CE_distrib {} '.format(i, j, CE_distrib))
            #print('{} {} CI_distrib {} '.format(i, j, CI_distrib))

    mpi_id = j
    # number of indegrees from current MPI process
    CE_local = CE // mpi_np
    if ( (2*mpi_id) % mpi_np ) < ( CE % mpi_np ):
        CE_local = CE_local + 1
        
    CI_local = CI // mpi_np
    if ( (2*mpi_id) % mpi_np ) < ( CI % mpi_np ):
        CI_local = CI_local + 1
    print('{} CE_local {}'.format(mpi_id, CE_local))
    print('{} CI_local {}'.format(mpi_id, CI_local))
    CE_total = CE_total + CE_local
    CI_total = CI_total + CI_local

    print('{} CE_total {} '.format(j, CE_total))
    print('{} CI_total {} '.format(j, CI_total))
