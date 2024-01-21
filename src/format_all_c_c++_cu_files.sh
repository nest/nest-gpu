#!/bin/bash

for fn in $(ls *.cu *.cuh *.cpp *.cc *.c *.h *.hpp); do
    echo " - Format C/C++/CUDA file: $fn"
    ./clang-format-cuda $fn
done
