mpi_np = 640
mpi_id = 113

CE = 9000
CI = 2250

for j in range(mpi_np):
    CE_total = 0
    CI_total = 0
    for i in range(mpi_np):
        if(i!=j):
            CE_distrib = CE // ( mpi_np - 1 )
            if ( (i + j) % ( mpi_np - 1 ) ) < ( CE % ( mpi_np - 1 ) ):
                CE_distrib = CE_distrib + 1

            CI_distrib = CI // ( mpi_np - 1 )
            if ( (i + j) % ( mpi_np - 1 ) ) < ( CI % ( mpi_np - 1 ) ):
                CI_distrib = CI_distrib + 1
                
            CE_total = CE_total + CE_distrib
            CI_total = CI_total + CI_distrib
            #print('{} {} CE_distrib {} '.format(i, j, CE_distrib))
            #print('{} {} CI_distrib {} '.format(i, j, CI_distrib))

    print('{} CE_total {} '.format(j, CE_total))
    print('{} CI_total {} '.format(j, CI_total))
