for fn in $(ls *.cu *.h); do
    cat $fn | tr '\n' '\r' | sed 's/gpuErrchk(cudaMalloc(\([^,]*\),\([^;]*\));/CUDAMALLOCCTRL("\1",\1,\2;/g' | tr '\r' '\n' > tmp.txt
    /bin/mv tmp.txt $fn
done
