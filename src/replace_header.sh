for fn in $(ls *.cu *.h); do
    cat $fn | sed 's/part of NESTGPU/part of NEST GPU/;s/*  NESTGPU/*  NEST GPU/;s/with NESTGPU/with NEST GPU/' > tmp.txt
    /bin/mv tmp.txt $fn
done
