for fn in $(ls *.cu *.h); do
    cat $fn | tr '\n' '\r' | sed 's/gpuErrchk(cudaFree(\([^)]*\)));/CUDAFREECTRL("\1",\1);/g' | tr '\r' '\n' > tmp.txt
    /bin/mv tmp.txt $fn
done
