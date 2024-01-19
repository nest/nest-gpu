# MPI GPU memory check example

Simple test for checking GPU memory consumption during network construction.

Run with:
```shell
nsys profile -t cuda,mpi,nvtx --cuda-memory-usage=true --force-overwrite true -o report mpirun -np $NP python3 test.py |& tee log.txt && nsys-ui report.nsys-rep
```
Where $NP is the number (2 or more) of MPI processes to be used.

To check memory consumption you need to open a report with Nsight-Systems, for an MPI rank you open the CUDA API category and hover over the memory used bar.

## Test script arguments:
* --N: Number of neurons to be created PER MPI process. Integer.
* --C: Number of connections to be created between neuron populations. Integer.
  * The generated network connects all populations to each other. No self connections.
* --R: Connection rule to be used. Integer.
  * 0 -> No connection rule i.e. script will only create neurons.
  * 1 -> Fixed indegree rule.
  * 2 -> Fixed outdegree rule.
  * 3 -> Fixed total number rule.
  * 4 -> All to all connections. Argument C will be ignored.

## Example run files:
* bash script to run a single test run.sh you can optionally give a number of MPI processes to run it with.
  * execute with: ```bash run.sh [NP]```
* bash script to run the parameter space exploration benchmark.sh it will run a grid scan using 2 to 6 MPI processes, 1 to 10K neurons (by multiples of 10), 1 to 10K synapses per neuron (by multiples of 10), with no connections, fixed indegree rule, fixed outdegree rule, fixed total number rule, and all to all connections.
  * execute with: ```bash benchmark.sh```
 