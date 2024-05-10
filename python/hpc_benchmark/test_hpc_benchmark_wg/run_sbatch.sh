#!/bin/bash -x
#SBATCH --account=jinb33
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --partition=dc-gpu
#SBATCH --gres=gpu:4
#SBATCH --output=/p/scratch/cjinb33/golosio/test/test_ngpu_hpcb_wg_out.%j
#SBATCH --error=/p/scratch/cjinb33/golosio/test/test_ngpu_hpcb_wg_err.%j
# *** start of job script ***
# Note: The current working directory at this point is
# the directory where sbatch was executed.

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
srun python3 hpc_benchmark_wg_jureca.py
