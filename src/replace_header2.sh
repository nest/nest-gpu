for fn in $(ls *.cu *.h); do
    cat $fn | sed "s/ \*  This file is/ \*  $fn\n \*\n \*  This file is/" > tmp.txt
    /bin/mv tmp.txt $fn
done
