#
#  eval_microcircuit_time.py
#
#  This file is part of NEST GPU.
#
#  Copyright (C) 2021 The NEST Initiative
#
#  NEST GPU is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 2 of the License, or
#  (at your option) any later version.
#
#  NEST GPU is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with NEST GPU. If not, see <http://www.gnu.org/licenses/>.
#
#
#
#
"""PyNEST Microcircuit: Run Simulation
-----------------------------------------

This is an example script for running the microcircuit model and generating
basic plots of the network activity.

"""

###############################################################################
# Import the necessary modules and start the time measurements.

from stimulus_params import stim_dict
from network_params import net_dict
from sim_params import sim_dict
import network

from time import perf_counter_ns
from argparse import ArgumentParser
from pathlib import Path
from json import dump, dumps

# Get and check file path
parser = ArgumentParser()
parser.add_argument("--file", type=str, default="benchmark_log", help='Output file without extention (default: benchmark_log).')
parser.add_argument("--path", type=str, default=None, help='Path for the simulaiton output. Default indicated in sim_params.py.')
parser.add_argument("--seed", type=int, default=12345, help='Seed for random number generation (default: 12345).')
parser.add_argument("--algo", type=int, default=0, help='Algorithm id for nested loop operation (default: 0). See the script for more detail.')
args = parser.parse_args()

if args.path is None:
    data_path = Path(sim_dict["data_path"])
else:
    data_path = Path(args.path)
    sim_dict["data_path"] = str(data_path) + "/" # Path to str never ends with /

file_name = args.file + ".json"
file_path = data_path / args.file

assert 0 <= args.algo and args.algo < 9
sim_dict["master_seed"] = args.seed

print(f"Arguments: {args}")

nl_dict = {
        0: "BlockStep",
        1: "CumulSum",
        2: "Simple",
        3: "ParallelInner",
        4: "ParallelOuter",
        5: "Frame1D",
        6: "Frame2D",
        7: "Smart1D",
        8: "Smart2D",
    }

time_start = perf_counter_ns()

###############################################################################
# Initialize the network with simulation, network and stimulation parameters,
# then create and connect all nodes, and finally simulate.
# The times for a presimulation and the main simulation are taken
# independently. A presimulation is useful because the spike activity typically
# exhibits a startup transient. In benchmark simulations, this transient should
# be excluded from a time measurement of the state propagation phase. Besides,
# statistical measures of the spike activity should only be computed after the
# transient has passed.


sim_dict.update({
    # presimulation of a single timestep to trigger calibration
    't_presim': 0.1,
    't_sim': 10000.,
    'rec_dev': [],
    'master_seed': args.seed,
    'nl_algo': args.algo})

net_dict.update({
    'N_scaling': 1.,
    'K_scaling': 1.,
    'poisson_input': True,
    'V0_type': 'optimized'})

net = network.Network(sim_dict, net_dict, stim_dict)
time_network = perf_counter_ns()

net.create()
time_create = perf_counter_ns()

net.connect()
time_connect = perf_counter_ns()

net.simulate(sim_dict['t_presim'])
time_presimulate = perf_counter_ns()

net.simulate(sim_dict['t_sim'])
time_simulate = perf_counter_ns()

time_dict = {
        "time_network": time_network - time_start,
        "time_create": time_create - time_network,
        "time_connect": time_connect - time_create,
        "time_presimulate": time_presimulate - time_connect,
        "time_simulate": time_simulate - time_presimulate,
        "time_total": time_simulate - time_start,
        }

info_dict = {
        "nested_loop_algo": nl_dict[args.algo],
        "timers": time_dict
    }

with file_path.open("w") as f:
    dump(info_dict, f, indent=4)

print(dumps(info_dict, indent=4))

