#!/bin/bash

mkdir -p sim_output

for seed in {0..9}; do
	# Best nested loop algorithm measured was BlockStep -> algo == 0
	for algo in 0; do
		echo "Benchmark: seed $seed, algo $algo"
		data_path=sim_output/benchmark-seed-$seed-algo-$algo
		mkdir $data_path
		{ time python3 run_benchmark.py --file=benchmark_times_$seed_$algo --path=$data_path --seed=$seed --algo=$algo 2> sim_output/run_benchmark_$seed_$algo.err; } &> sim_output/run_benchmark_$seed_$algo.out
	done
done
