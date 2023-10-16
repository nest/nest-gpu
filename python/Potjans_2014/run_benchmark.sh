#!/bin/bash

mkdir -p sim_output

for seed in {0..9}; do
	# Best nested loop algorithm measured was BlockStep -> algo == 0
	for algo in 0; do
		echo "Benchmark: seed $seed, algo $algo"
		data_path=sim_output/benchmark-seed-$seed-algo-$algo
		mkdir $data_path
		sim_date=$(date +%F_%H-%M-%S)
		{ time python3 run_benchmark.py benchmark_times_$sim_date.json --path=$data_path --seed=$seed --algo=$algo 2> sim_output/run_benchmark_$sim_date.err; } &> sim_output/run_benchmark_$sim_date.out
	done
done
