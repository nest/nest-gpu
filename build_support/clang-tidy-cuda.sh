#!/bin/bash

#
#   This file is part of NEST GPU.
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
#  along with NEST GPU.  If not, see <http://www.gnu.org/licenses/>.
#

cuda_default_path="/usr/local/cuda/include"

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 [--include-path=INCLUDE_PATHS] [--cuda-path=CUDA_PATHS] [--mpi-path=MPI_PATHS] input-file"
    echo "where INCLUDE_PATHS are optional header paths separated by colons,"
    echo "CUDA_PATHS are the paths of CUDA headers separated by colons"
    echo "(default: $cuda_default_path)"
    echo "and MPI_PATHS are the paths of MPI headers separated by colons"
    exit 0
fi

cuda_path=""
mpi_path=""
include_path=""

for i in "$@"; do
    case $i in
	--include-path=*)
	    include_path="${i#*=}"
	    shift # past argument=value
	    ;;
	--cuda-path=*)
	    cuda_path="${i#*=}"
	    shift # past argument=value
	    ;;
	--mpi-path=*)
	    mpi_path="${i#*=}"
	    shift # past argument=value
	    ;;
	-*|--*)
	    echo "Error: unknown option $i"
	    exit 1
	    ;;
	*)
	    ;;
    esac
done

if [[ -n $1 ]]; then
    echo "Input file: $1"
else
    echo "Error: input file not specified."
    exit 1
fi

if [ ! -f $1 ]; then
    echo "Error: input file $1 not found."
    exit 1
fi

if [ "$include_path" != "" ]; then    
    include_path=$(echo ":$include_path" | sed 's/::*/:/g;s/:$//;s/:/ -I /g')
fi

# Searches the paths of CUDA headers
if [ "$cuda_path" == "" ]; then    
    cuda_path=":/usr/local/cuda/include"
else
    cuda_path=$(echo ":$cuda_path" | sed 's/::*/:/g;s/:$//')
fi

cuda_path_spaced=$(echo $cuda_path | tr ':' ' ')
cuda_err=1
for dn in $cuda_path_spaced; do
    if test -f "$dn/cuda.h" ; then
	echo "cuda.h found in path $dn"
	cuda_err=0
	break
    fi
done

if [ $cuda_err -eq 1 ]; then
    echo "cuda.h not found in path(s) $cuda_path_spaced"
    echo "You can specify path for CUDA headers with the option --cuda-path=CUDA_PATHS"
    echo "where CUDA_PATHS are the paths of CUDA headers separated by colons"
    echo "(default: $cuda_default_path)"
    exit 1
fi

cuda_include=$(echo $cuda_path | sed 's/:/ -isystem /g')

#cat $1 | sed 's://<BEGIN-CLANG-TIDY-SKIP>//:#if 0:;s://<END-CLANG-TIDY-SKIP>//:#endif:' > tmp~
    
#cat ../build_cmake/compile_commands.json | sed "s:-Xcompiler=-fPIC::;s:-forward-unknown-to-host-compiler::;s:--compiler-options='.*'::;s:--generate-code=arch=compute_80,code=\[compute_80,sm_80\]::;s:--maxrregcount=55::" > compile_commands.json

# Searches the paths of MPI headers
if [ "$mpi_path" == "" ]; then    
    mpi_include=$( \
		   for l in  $(mpicc -showme); do \
		       echo $l; \
		   done | grep '^-I')
    if [ "$mpi_include" == "" ]; then
	echo "Error: cannot find MPI include paths"
	echo "You can specify path for MPI headers with the option --mpi-path=MPI_PATHS"
	echo "where MPI_PATHS are the paths of MPI headers separated by colons"
	exit 1
    fi
    mpi_include=$(echo $mpi_include | sed 's/-I/ -isystem /g')
    mpi_path_spaced=$(echo $mpi_include | sed 's/-I/ /g')
else
    mpi_path=$(echo ":$mpi_path" | sed 's/::*/:/g;s/:$//')
    mpi_path_spaced=$(echo $mpi_path | tr ':' ' ')
    mpi_include=$(echo $mpi_path | sed 's/:/ -isystem /g')
fi

mpi_err=1
for dn in $mpi_path_spaced; do
    if test -f "$dn/mpi.h" ; then
	echo "mpi.h found in path $dn"
	mpi_err=0
	break
    fi
done

if [ $mpi_err -eq 1 ]; then
    echo "mpi.h not found in path(s) $mpi_path_spaced"
    echo "You can specify path for MPI headers with the option --mpi-path=MPI_PATHS"
    echo "where MPI_PATHS are the paths of MPI headers separated by colons"
    exit 1
fi

echo "clang-tidy $1 -p . -- $include_path $mpi_include $cuda_include --no-cuda-version-check"

clang-tidy $1 -p .  -- $include_path $mpi_include $cuda_include --no-cuda-version-check
